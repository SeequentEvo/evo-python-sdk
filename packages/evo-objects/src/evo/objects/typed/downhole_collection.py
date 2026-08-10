#  Copyright © 2026 Bentley Systems, Incorporated
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Annotated, Any, ClassVar, TypeAlias

import numpy as np
import pandas as pd
from numpy._typing import NDArray

from evo.common import IFeedback
from evo.common.interfaces import IContext
from evo.common.utils import NoFeedback
from evo.objects import SchemaVersion
from evo.objects.typed._data import DataTable, DataTableAndAttributes
from evo.objects.typed._downhole import HoleIdCategory
from evo.objects.typed._model import DataLocation, SchemaList, SchemaLocation, SchemaModel
from evo.objects.typed.attributes import (
    AttributeDescription,
    Attributes,
)
from evo.objects.typed.exceptions import ObjectValidationError
from evo.objects.typed.spatial import BaseSpatialObject, BaseSpatialObjectData
from evo.objects.typed.types import BoundingBox
from evo.objects.utils.table_formats import (
    DOWNHOLE_COLLECTION_LOCATION_HOLES,
    FLOAT_ARRAY_1,
    FLOAT_ARRAY_2,
    FLOAT_ARRAY_3,
    KnownTableFormat,
)

__all__ = [
    "DistanceCollection",
    "DownholeCollection",
    "DownholeCollectionData",
    "IntervalCollection",
]

_X = "x"
_Y = "y"
_Z = "z"
_COORDINATE_COLUMNS = [_X, _Y, _Z]


HolePath: TypeAlias = pd.DataFrame  # [ distance | dip | azimuth | <attributes> ]
HoleChunks: TypeAlias = pd.DataFrame  # [ hole_index | offset | count ]
HoleProperties: TypeAlias = pd.DataFrame  # [ hole_id | final | target | current | x | y | z ]
HoleAttributes: TypeAlias = pd.DataFrame

# If `Depths` has unit descriptions in its `DataFrame.attrs` dictionary, then those units will be used when building
# the schema object.
# This is the expected structure:
#   >>> depths_df.attrs
#   {'attribute_descriptions': {<column names>: <AttributeDescription>}, ...}
Depths: TypeAlias = pd.DataFrame  # [ distance | <attributes> ]
Intervals: TypeAlias = pd.DataFrame  # [ from | to | <attributes> ]


@dataclass
class DistanceCollection:
    name: str
    holes: HoleChunks
    table: Depths
    collection_type: str = "distance"
    unit: str | None = None


@dataclass
class IntervalCollection:
    name: str
    holes: HoleChunks
    table: Intervals
    collection_type: str = "interval"
    unit: str | None = None


DownholeCollectionEntry: TypeAlias = DistanceCollection | IntervalCollection


@dataclass(kw_only=True, frozen=True)
class DownholeCollectionData(BaseSpatialObjectData):
    """Data class for creating a new DownholeCollection

    :param name: The name of the object.
    :param holes: A DataFrame describing which parts of `path` belong to which holes.
            Columns: hole_index, offset, count. ``hole_index`` is a code in ``properties.hole_id``.
    :param properties: DataFrame for the properties of the holes, joined to ``holes`` by the categorical hole-id code.
            Mandatory columns: hole_id, final, target, current, x, y, z
    :param attributes: DataFrame for the attributes of the holes, in the same order as ``properties``.
    :param path: Dataframe of [ distance | dip | azimuth | <attributes> ]. Distance/dip/azimuth describe the geometry as
            the step since the previous row.
    :param collections: Distance and interval collection tables.
    :param distance_unit: The distance unit for the `path` table and the `properties` x/y/y.
    :param desurvey: The desurvey method appropriate for this collection.
            Must be one of: "minimum_curvature", "balanced_tangent", "trench".
    :param coordinate_reference_system: Optional EPSG code or WKT string for the coordinate reference system.
    :param description: Optional description of the object.
    :param tags: Optional dictionary of tags for the object.
    :param extensions: Optional dictionary of extensions for the object.
    """

    path: HolePath
    holes: HoleChunks
    properties: HoleProperties
    attributes: HoleAttributes | None
    collections: list[DownholeCollectionEntry]
    distance_unit: str | None = None
    desurvey: str | None = None

    @property
    def hole_id_dtype(self) -> pd.CategoricalDtype:
        """The categorical dtype whose codes are used by every ``hole_index``."""
        hole_ids = self.properties["hole_id"]
        if isinstance(hole_ids.dtype, pd.CategoricalDtype):
            return hole_ids.dtype
        return pd.CategoricalDtype(categories=sorted(hole_ids.dropna().unique()))

    def __post_init__(self):
        if self.attributes is not None and len(self.holes) != len(self.attributes):
            raise ObjectValidationError("The number of attributes rows must match the number or holes rows")

        assert self.attributes is None or len(self.holes) == len(self.attributes)

        self._validate_hole_chunks(self.holes, len(self.path), require_coverage=True)
        for collection in self.collections:
            table = collection.table
            self._validate_hole_chunks(collection.holes, len(table), require_coverage=False)

    def _validate_hole_chunks(self, holes: HoleChunks, table_length: int, *, require_coverage: bool) -> None:
        required = {"hole_index", "offset", "count"}
        if missing := required - set(holes.columns):
            raise ObjectValidationError(f"Hole chunks are missing columns: {sorted(missing)}")
        indices = holes["hole_index"].astype(int)
        valid = set(range(len(self.hole_id_dtype.categories)))
        if not set(indices).issubset(valid):
            raise ObjectValidationError("hole_index must be a code in properties['hole_id'] categorical dtype")
        if indices.duplicated().any():
            raise ObjectValidationError("Each hole_index may occur only once in a holes table")
        if require_coverage and set(indices) != valid:
            raise ObjectValidationError("Location holes must cover every hole_id categorical code exactly once")
        offsets = holes["offset"].astype(int)
        counts = holes["count"].astype(int)
        if (offsets < 0).any() or (counts < 0).any() or ((offsets + counts) > table_length).any():
            raise ObjectValidationError("Hole chunk offsets and counts must be within the associated table")
        non_empty = sorted(zip(offsets[counts > 0], counts[counts > 0], strict=True))
        expected_offset = 0
        for offset, count in non_empty:
            if offset != expected_offset:
                raise ObjectValidationError("Hole chunk ranges must cover the associated table exactly once")
            expected_offset = offset + count
        if expected_offset != table_length:
            raise ObjectValidationError("Hole chunk ranges must cover the associated table exactly once")

    def compute_bounding_box(self) -> BoundingBox:
        bboxes = []

        collars = self.properties.copy()
        collars["_hole_index"] = collars["hole_id"].astype(self.hole_id_dtype).cat.codes
        collars_by_index = collars.set_index("_hole_index")
        for chunk in self.holes.itertuples(index=False):
            offset = int(chunk.offset)
            count = int(chunk.count)
            collar = tuple(collars_by_index.loc[int(chunk.hole_index), _COORDINATE_COLUMNS])
            path_table = self.path[offset : offset + count]
            bboxes.append(self._compute_hole_bounding_box(path_table, collar))

        return BoundingBox.combine(bboxes)

    @staticmethod
    def _compute_bounding_box_np(
        depths: NDArray[np.float64],
        dips: NDArray[np.float64],
        azimuths: NDArray[np.float64],
        offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> BoundingBox:
        if not np.all(depths[:-1] <= depths[1:]):
            raise ObjectValidationError("depths must be sorted")

        if len(depths) != len(dips) or len(depths) != len(azimuths):
            raise ObjectValidationError("depths, dips, and azimuths must have same length")

        # Process NaNs
        # `depths`, `dips`, and `azimuths` could be read-only views, so take copies instead of mutating
        depths = depths[~np.isnan(depths)]
        dips = np.where(np.isnan(dips), 90.0, dips)
        azimuths = np.where(np.isnan(azimuths), 0.0, azimuths)

        dips_rad = np.deg2rad(dips)
        azimuths_rad = np.deg2rad(azimuths)

        # Prepend 0 so `step` has the same shape as `dips` and `azimuths`, and so the first depth gets treated as the
        # first step. The depth column might already start with 0, in which case the first step will be length 0, which
        # is a no-op as far as the following calculation is concerned.
        step = np.diff(depths, prepend=0.0)

        dz_down = step * np.sin(dips_rad)
        horiz = step * np.cos(dips_rad)

        # Horizontal into N/E (0° = North, 90° = East)
        dN = horiz * np.cos(azimuths_rad)
        dE = horiz * np.sin(azimuths_rad)

        # Convert to XYZ increments (Z up)
        dX = dE
        dY = dN
        dZ = -dz_down

        x = np.cumsum(dX)
        y = np.cumsum(dY)
        z = np.cumsum(dZ)

        def ensure_zero(a, b):
            return min(a, 0), max(b, 0)

        x0, x1 = ensure_zero(x.min(), x.max())
        y0, y1 = ensure_zero(y.min(), y.max())
        z0, z1 = ensure_zero(z.min(), z.max())

        return BoundingBox(
            min_x=x0 + offset[0],
            max_x=x1 + offset[0],
            min_y=y0 + offset[1],
            max_y=y1 + offset[1],
            min_z=z0 + offset[2],
            max_z=z1 + offset[2],
        )

    @staticmethod
    def _compute_hole_bounding_box(
        depths_dips_azimuths_table: pd.DataFrame,
        collar: tuple[float, float, float],
    ) -> BoundingBox:
        """
        Compute 3D bounding box for a deviated hole given collar XYZ and
        depth / dip / azimuth data.

        Conventions
        -----------
        - depths: measured depth along the hole (m), positive downward.
        - dips: inclination FROM VERTICAL (degrees).
            90° = vertical down, 0° = horizontal.
        - azimuths: degrees clockwise from North.
        - Coordinates: X = Easting, Y = Northing, Z = elevation (up).
        """
        df = depths_dips_azimuths_table.dropna(subset=["distance"])
        box = DownholeCollectionData._compute_bounding_box_np(
            df["distance"].astype(float).to_numpy(),
            df["dip"].astype(float).to_numpy(),
            df["azimuth"].astype(float).to_numpy(),
            offset=collar,
        )

        return box


class HoleChunksTable(DataTable):
    table_format: ClassVar[KnownTableFormat] = DOWNHOLE_COLLECTION_LOCATION_HOLES
    data_columns: ClassVar[list[str]] = ["hole_index", "offset", "count"]


class PathTable(DataTable):
    table_format: ClassVar[KnownTableFormat] = FLOAT_ARRAY_3
    data_columns: ClassVar[list[str]] = ["distance", "azimuth", "dip"]


class DownholePath(DataTableAndAttributes):
    _table: Annotated[PathTable, SchemaLocation(""), DataLocation("")]


class DistancesTable(DataTable):
    table_format: ClassVar[KnownTableFormat] = FLOAT_ARRAY_3
    data_columns: ClassVar[list[str]] = ["final", "target", "current"]

    @classmethod
    def _extract_distances(cls, data: HoleAttributes) -> pd.DataFrame:
        return data[["final", "target", "current"]].astype(np.float64)

    @classmethod
    async def _data_to_schema(cls, data: HoleAttributes, context: IContext) -> Any:
        distances_df = cls._extract_distances(data)
        return await super()._data_to_schema(distances_df, context)


class CollarCoordinates(DataTable):
    table_format: ClassVar[KnownTableFormat] = FLOAT_ARRAY_3
    data_columns: ClassVar[list[str]] = _COORDINATE_COLUMNS

    @classmethod
    def _extract_coordinates(cls, data: HoleAttributes):
        return data[["x", "y", "z"]].astype(np.float64)

    @classmethod
    async def _data_to_schema(cls, data: HoleAttributes, context: IContext) -> Any:
        distances_df = cls._extract_coordinates(data)
        return await super()._data_to_schema(distances_df, context)


class DownholeLocation(SchemaModel):
    hole_id: Annotated[HoleIdCategory, SchemaLocation("hole_id"), DataLocation("properties")]
    path: Annotated[DownholePath, SchemaLocation("path"), DataLocation("path")]
    holes: Annotated[HoleChunksTable, SchemaLocation("holes"), DataLocation("holes")]
    distances: Annotated[DistancesTable, SchemaLocation("distances"), DataLocation("properties")]
    coordinates: Annotated[CollarCoordinates, SchemaLocation("coordinates"), DataLocation("properties")]
    attributes: Annotated[Attributes, SchemaLocation("attributes"), DataLocation("attributes")]

    async def to_dataframe(self, fb: IFeedback = NoFeedback) -> pd.DataFrame:
        """Return collars with a categorical ``hole_id`` aligned to hole-index codes."""
        parts = [
            await self.hole_id.to_dataframe(fb=fb),
            await self.coordinates.to_dataframe(fb=fb),
            await self.distances.to_dataframe(fb=fb),
        ]
        if len(self.attributes):
            parts.append(await self.attributes.to_dataframe(fb=fb))
        return pd.concat(parts, axis=1)

    async def path_to_dataframe(self, fb: IFeedback = NoFeedback) -> pd.DataFrame:
        """Return the desurvey path and its attributes."""
        return await self.path.to_dataframe(fb=fb)


class _Distances(DataTable):
    table_format: ClassVar[KnownTableFormat] = FLOAT_ARRAY_1
    data_columns: ClassVar[list[str]] = ["distance"]


class DistanceTableDistances(DataTableAndAttributes):
    _table: Annotated[_Distances, SchemaLocation("values"), DataLocation("")]
    unit: Annotated[str | None, SchemaLocation("unit")]

    @classmethod
    async def _data_to_schema(cls, data: pd.DataFrame, context: IContext) -> Any:
        result = await super()._data_to_schema(data, context)
        unit = data.attrs.get("unit")
        attr_desc: AttributeDescription = data.attrs.get("attribute_descriptions", {}).get("distance")
        if unit is None and attr_desc is not None:
            unit = attr_desc.unit
        if unit is not None:
            # "unit" can be missing, but it must not be `None`
            result["unit"] = unit
        return result


class DistanceTable(SchemaModel):
    name: Annotated[str, SchemaLocation("name"), DataLocation("name")]
    collection_type: Annotated[str, SchemaLocation("collection_type"), DataLocation("collection_type")]
    distance: Annotated[DistanceTableDistances, SchemaLocation("distance"), DataLocation("table")]


class DownholeDistanceTable(DistanceTable):
    holes: Annotated[HoleChunksTable, SchemaLocation("holes"), DataLocation("holes")]

    async def to_dataframe(self, fb: IFeedback = NoFeedback) -> pd.DataFrame:
        return await self.distance.to_dataframe(fb=fb)

    async def to_dataframe_by_hole(self, fb: IFeedback = NoFeedback) -> dict[str, pd.DataFrame]:
        return await _table_by_hole(self, await self.to_dataframe(fb=fb), fb=fb)


class _Intervals(DataTable):
    table_format: ClassVar[KnownTableFormat] = FLOAT_ARRAY_2
    data_columns: ClassVar[list[str]] = ["from", "to"]


class IntervalTableFromTo(DataTableAndAttributes):
    _table: Annotated[_Intervals, SchemaLocation("intervals.start_and_end"), DataLocation("")]
    unit: Annotated[str | None, SchemaLocation("unit")]

    @classmethod
    async def _data_to_schema(cls, data: pd.DataFrame, context: IContext) -> Any:
        result = await super()._data_to_schema(data, context)
        unit = data.attrs.get("unit")
        description = data.attrs.get("attribute_descriptions", {}).get("from")
        if unit is None and description is not None:
            unit = description.unit
        if unit is not None:
            result["unit"] = unit
        return result


class DownholeIntervalTable(SchemaModel):
    name: Annotated[str, SchemaLocation("name"), DataLocation("name")]
    collection_type: Annotated[str, SchemaLocation("collection_type"), DataLocation("collection_type")]
    from_to: Annotated[IntervalTableFromTo, SchemaLocation("from_to"), DataLocation("table")]
    holes: Annotated[HoleChunksTable, SchemaLocation("holes"), DataLocation("holes")]

    async def to_dataframe(self, fb: IFeedback = NoFeedback) -> pd.DataFrame:
        return await self.from_to.to_dataframe(fb=fb)

    async def to_dataframe_by_hole(self, fb: IFeedback = NoFeedback) -> dict[str, pd.DataFrame]:
        return await _table_by_hole(self, await self.to_dataframe(fb=fb), fb=fb)


class DownholeCollectionTables(SchemaList[DownholeDistanceTable | DownholeIntervalTable]):
    @classmethod
    def _resolve_item_type(cls, document: dict[str, Any]) -> type[DownholeDistanceTable | DownholeIntervalTable]:
        return DownholeIntervalTable if "from_to" in document else DownholeDistanceTable

    @classmethod
    async def _data_to_schema(cls, data: Any, context: IContext) -> list[Any]:
        if data is None:
            return []
        result = []
        for collection in data:
            model = DownholeIntervalTable if isinstance(collection, IntervalCollection) else DownholeDistanceTable
            table = collection.table
            table = table.copy()
            table.attrs = dict(table.attrs)
            if collection.unit is not None:
                table.attrs["unit"] = collection.unit
            result.append(await model._data_to_schema(replace(collection, table=table), context))
        return result

    def get(self, name: str) -> DownholeDistanceTable | DownholeIntervalTable | None:
        return next((collection for collection in self if collection.name == name), None)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and self.get(name) is not None

    def names(self) -> list[str]:
        return [collection.name for collection in self]

    async def add(self, collection: DownholeCollectionEntry, *, replace: bool = False) -> None:
        existing = self.names()
        if collection.name in existing and not replace:
            raise ValueError(f"Collection '{collection.name}' already exists")
        location_holes = await self._context.root_model.location.holes.to_dataframe()
        valid_indices = set(location_holes["hole_index"].astype(int))
        if not set(collection.holes["hole_index"].astype(int)).issubset(valid_indices):
            raise ObjectValidationError("Collection hole_index is not present in the location holes table")
        table = collection.table
        _validate_chunk_ranges(collection.holes, len(table))
        schema = await self._data_to_schema([collection], self._obj)
        if collection.name in existing:
            self._document[existing.index(collection.name)] = schema[0]
        else:
            self._document.append(schema[0])

    def remove(self, *names: str) -> int:
        requested = set(names)
        previous = len(self._document)
        self._document[:] = [item for item in self._document if item.get("name") not in requested]
        return previous - len(self._document)


class DownholeCollection(BaseSpatialObject):
    """A GeoscienceObject representing a collection of downholes."""

    _data_class = DownholeCollectionData
    sub_classification = "downhole-collection"
    creation_schema_version = SchemaVersion(major=1, minor=3, patch=1)

    location: Annotated[DownholeLocation, SchemaLocation("location"), DataLocation("")]
    collections: Annotated[DownholeCollectionTables, SchemaLocation("collections"), DataLocation("collections")]
    distance_unit: Annotated[str | None, SchemaLocation("distance_unit")]
    desurvey: Annotated[str | None, SchemaLocation("desurvey")]

    type: ClassVar[Annotated[str, SchemaLocation("type")]] = "downhole"

    async def prefetch_collections(self, *names: str, include_location: bool = True, **kwargs: Any) -> None:
        """Prefetch data referenced by named collections and optionally location data."""
        from evo.objects.typed._prefetch import collect_data_ids

        documents = []
        if include_location:
            documents.append(self.location.as_dict())
        for name in names:
            collection = self.collections.get(name)
            if collection is None:
                raise KeyError(f"Unknown collection '{name}'")
            documents.append(collection.as_dict())
        await self.prefetch(data_ids=collect_data_ids(documents), **kwargs)


def _validate_chunk_ranges(holes: HoleChunks, table_length: int) -> None:
    required = {"hole_index", "offset", "count"}
    if missing := required - set(holes.columns):
        raise ObjectValidationError(f"Hole chunks are missing columns: {sorted(missing)}")
    if holes["hole_index"].astype(int).duplicated().any():
        raise ObjectValidationError("Each hole_index may occur only once in a holes table")
    offsets = holes["offset"].astype(int)
    counts = holes["count"].astype(int)
    if (offsets < 0).any() or (counts < 0).any() or ((offsets + counts) > table_length).any():
        raise ObjectValidationError("Hole chunk offsets and counts must be within the associated table")
    expected_offset = 0
    for offset, count in sorted(zip(offsets[counts > 0], counts[counts > 0], strict=True)):
        if offset != expected_offset:
            raise ObjectValidationError("Hole chunk ranges must cover the associated table exactly once")
        expected_offset = offset + count
    if expected_offset != table_length:
        raise ObjectValidationError("Hole chunk ranges must cover the associated table exactly once")


async def _table_by_hole(
    table: DownholeDistanceTable | DownholeIntervalTable, data: pd.DataFrame, *, fb: IFeedback
) -> dict[str, pd.DataFrame]:
    root = table._context.root_model
    collar_ids = await root.location.hole_id.to_dataframe(fb=fb)
    categories = collar_ids.iloc[:, 0]
    result: dict[str, pd.DataFrame] = {}
    for chunk in (await table.holes.to_dataframe(fb=fb)).itertuples(index=False):
        hole_id = str(categories.cat.categories[int(chunk.hole_index)])
        result[hole_id] = data.iloc[int(chunk.offset) : int(chunk.offset) + int(chunk.count)].reset_index(drop=True)
    return result
