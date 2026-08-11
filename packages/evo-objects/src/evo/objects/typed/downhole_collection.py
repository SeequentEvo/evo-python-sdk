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

import warnings
from dataclasses import dataclass
from typing import Annotated, Any, ClassVar, TypeAlias

import numpy as np
import pandas as pd
from numpy._typing import NDArray

from evo.common import IFeedback
from evo.common.interfaces import IContext
from evo.common.utils import NoFeedback
from evo.objects import SchemaVersion
from evo.objects.typed import attributes as _attributes
from evo.objects.typed._data import DataTable, DataTableAndAttributes
from evo.objects.typed._downhole import DepthIntervalsTable, HoleIdCategory
from evo.objects.typed._model import DataLocation, SchemaList, SchemaLocation, SchemaModel
from evo.objects.typed._prefetch import collect_data_ids
from evo.objects.typed.attributes import Attributes
from evo.objects.typed.exceptions import ObjectValidationError
from evo.objects.typed.spatial import BaseSpatialObject, BaseSpatialObjectData
from evo.objects.typed.types import BoundingBox
from evo.objects.utils.table_formats import (
    DOWNHOLE_COLLECTION_LOCATION_HOLES,
    FLOAT_ARRAY_1,
    FLOAT_ARRAY_3,
    INTEGER_ARRAY_1_INT32,
    LOOKUP_TABLE_INT32,
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

Depths: TypeAlias = pd.DataFrame  # [ distance | <attributes> ]
Intervals: TypeAlias = pd.DataFrame  # [ from | to | <attributes> ]


@dataclass
class DistanceCollection:
    name: str
    holes: HoleChunks
    table: Depths
    unit: str | None
    collection_type: str = "distance"


@dataclass
class IntervalCollection:
    name: str
    holes: HoleChunks
    table: Intervals
    unit: str | None
    collection_type: str = "interval"


DownholeCollectionEntry: TypeAlias = DistanceCollection | IntervalCollection


@dataclass(kw_only=True, frozen=True)
class DownholeCollectionData(BaseSpatialObjectData):
    """Data class representing a DownholeCollection.

    :param name: The name of the object.
    :param holes: A DataFrame describing which parts of `path` belong to which holes.
            Columns: hole_index, offset, count. ``hole_index`` is an integer lookup key, not a row position.
    :param properties: DataFrame for the properties of the holes.
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
    distance_unit: str | None
    desurvey: str | None

    def __post_init__(self):
        if self.attributes is not None and len(self.holes) != len(self.attributes):
            raise ObjectValidationError("The number of attributes rows must match the number or holes rows")

        assert self.attributes is None or len(self.holes) == len(self.attributes)

        names = [collection.name for collection in self.collections]
        if len(names) != len(set(names)):
            raise ObjectValidationError("Collection names must be unique")

        self._validate_hole_chunks(self.holes, len(self.path), require_coverage=True)
        for collection in self.collections:
            table = collection.table
            self._validate_hole_chunks(collection.holes, len(table), require_coverage=False)

    @staticmethod
    def _validate_hole_chunks(holes: HoleChunks, table_length: int, *, require_coverage: bool) -> None:
        required = {"hole_index", "offset", "count"}
        if missing := required - set(holes.columns):
            raise ObjectValidationError(f"Hole chunks are missing columns: {sorted(missing)}")
        offsets = holes["offset"].astype(int)
        counts = holes["count"].astype(int)
        if (offsets < 0).any() or (counts < 0).any() or ((offsets + counts) > table_length).any():
            raise ObjectValidationError("Hole chunk offsets and counts must be within the associated table")
        if not require_coverage:
            return

        non_empty = sorted(zip(offsets[counts > 0], counts[counts > 0], strict=True))
        expected_offset = 0
        for offset, count in non_empty:
            if offset != expected_offset:
                raise ObjectValidationError("Hole chunk ranges must cover the associated table exactly once")
            expected_offset = offset + count
        if expected_offset != table_length:
            raise ObjectValidationError("Hole chunk ranges must cover the associated table exactly once")

    def _collars_by_hole_index(self) -> dict[int, tuple[float, float, float]] | None:
        """Map hole lookup keys to collar coordinates, or ``None`` when the mapping is not known.

        Keys are taken from an explicit ``properties["hole_index"]`` column when present. Otherwise the SDK creation
        convention (dense zero-based keys in sorted hole-ID order) is used, and only when it accounts for every key
        referenced by ``holes``.
        """
        hole_ids = self.properties["hole_id"].astype(str)
        if "hole_index" in self.properties:
            keys = self.properties["hole_index"].astype(int)
        else:
            dense = {hole_id: index for index, hole_id in enumerate(sorted(hole_ids.unique()))}
            keys = hole_ids.map(dense).astype(int)
        coordinates = self.properties[_COORDINATE_COLUMNS].astype(float)
        collars = {
            int(key): coordinate
            for key, coordinate in zip(keys, coordinates.itertuples(index=False, name=None), strict=True)
        }
        referenced = {int(chunk.hole_index) for chunk in self.holes.itertuples(index=False)}
        return collars if referenced <= set(collars) else None

    def compute_bounding_box(self) -> BoundingBox:
        collars_by_hole_index = self._collars_by_hole_index()
        if collars_by_hole_index is not None:
            bboxes = [
                self._compute_hole_bounding_box(
                    self.path[int(chunk.offset) : int(chunk.offset) + int(chunk.count)],
                    collars_by_hole_index[int(chunk.hole_index)],
                )
                for chunk in self.holes.itertuples(index=False)
                if int(chunk.count)
            ]
            if bboxes:
                return BoundingBox.combine(bboxes)

        # The collar for each chunk cannot be identified, so fall back to an envelope that contains every hole.
        collars = BoundingBox.from_points(self.properties[_COORDINATE_COLUMNS].astype(float).to_numpy())
        relative_bboxes = []
        for chunk in self.holes.itertuples(index=False):
            offset = int(chunk.offset)
            count = int(chunk.count)
            if count:
                relative_bboxes.append(
                    self._compute_hole_bounding_box(self.path[offset : offset + count], (0.0, 0.0, 0.0))
                )
        if not relative_bboxes:
            return collars
        relative = BoundingBox.combine(relative_bboxes)
        return BoundingBox(
            min_x=collars.min_x + relative.min_x,
            max_x=collars.max_x + relative.max_x,
            min_y=collars.min_y + relative.min_y,
            max_y=collars.max_y + relative.max_y,
            min_z=collars.min_z + relative.min_z,
            max_z=collars.max_z + relative.max_z,
        )

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


class LocationHoleIdCategory(HoleIdCategory):
    """Collar hole IDs persisted with explicit integer lookup keys.

    ``properties`` may provide a ``hole_index`` column to persist specific lookup keys, for example when rewriting a
    downloaded object. When it is absent, dense zero-based keys are generated in sorted hole-ID order as an SDK
    creation convenience.
    """

    @classmethod
    async def _data_to_schema(cls, data: pd.DataFrame, context: IContext) -> Any:
        hole_ids = data["hole_id"].astype(str)
        if hole_ids.duplicated().any():
            raise ObjectValidationError("properties hole_id values must be unique")
        if "hole_index" in data:
            keys = data["hole_index"].astype("int32")
            if keys.duplicated().any():
                raise ObjectValidationError("properties hole_index values must be unique")
        else:
            dense = {hole_id: index for index, hole_id in enumerate(sorted(hole_ids.unique()))}
            keys = hole_ids.map(dense).astype("int32")
        data_client = _attributes.get_data_client(context)
        return {
            "values": await data_client.upload_dataframe(
                pd.DataFrame({"hole_id": keys}), table_format=INTEGER_ARRAY_1_INT32
            ),
            "table": await data_client.upload_dataframe(
                pd.DataFrame({"key": keys, "value": hole_ids}), table_format=LOOKUP_TABLE_INT32
            ),
        }


class DownholeLocation(SchemaModel):
    hole_id: Annotated[LocationHoleIdCategory, SchemaLocation("hole_id"), DataLocation("properties")]
    path: Annotated[DownholePath, SchemaLocation("path"), DataLocation("path")]
    holes: Annotated[HoleChunksTable, SchemaLocation("holes"), DataLocation("holes")]
    distances: Annotated[DistancesTable, SchemaLocation("distances"), DataLocation("properties")]
    coordinates: Annotated[CollarCoordinates, SchemaLocation("coordinates"), DataLocation("properties")]
    attributes: Annotated[Attributes, SchemaLocation("attributes"), DataLocation("attributes")]

    async def to_dataframe(self, *keys: str, fb: IFeedback = NoFeedback) -> pd.DataFrame:
        """Return collars with resolved ``hole_id`` values and selected attributes."""
        parts = [
            await self.hole_id.to_dataframe(fb=fb),
            await self.coordinates.to_dataframe(fb=fb),
            await self.distances.to_dataframe(fb=fb),
        ]
        if len(self.attributes):
            parts.append(await self.attributes.to_dataframe(*keys, fb=fb))
        return pd.concat(parts, axis=1)

    async def path_to_dataframe(self, *keys: str, fb: IFeedback = NoFeedback) -> pd.DataFrame:
        """Return the desurvey path and its attributes."""
        return await self.path.to_dataframe(*keys, fb=fb)


class _Distances(DataTable):
    table_format: ClassVar[KnownTableFormat] = FLOAT_ARRAY_1
    data_columns: ClassVar[list[str]] = ["distance"]


class DistanceTableDistances(DataTableAndAttributes):
    _table: Annotated[_Distances, SchemaLocation("values"), DataLocation("")]
    unit: Annotated[str | None, SchemaLocation("unit")]


class DistanceTable(SchemaModel):
    name: Annotated[str, SchemaLocation("name"), DataLocation("name")]
    collection_type: Annotated[str, SchemaLocation("collection_type"), DataLocation("collection_type")]
    distance: Annotated[DistanceTableDistances, SchemaLocation("distance"), DataLocation("table")]

    async def to_dataframe(self, *keys: str, fb: IFeedback = NoFeedback) -> pd.DataFrame:
        """Return distance values and selected attributes."""
        return await self.distance.to_dataframe(*keys, fb=fb)


class _DownholeCollectionChild:
    """Shared behavior for models that must be nested in a DownholeCollection."""

    @property
    def _downhole_collection(self) -> DownholeCollection:
        """Return the containing DownholeCollection or raise if this model is used out of context."""
        root = self._context.root_model
        if not isinstance(root, DownholeCollection):
            raise ObjectValidationError(
                f"{type(self).__name__} must be attached to a DownholeCollection, not {type(root).__name__}"
            )
        return root

    async def _table_by_hole(self, data: pd.DataFrame, *, fb: IFeedback) -> dict[str, pd.DataFrame]:
        """Group table rows by hole using the containing DownholeCollection's hole-id lookup."""
        lookup = await self._downhole_collection.location.hole_id.to_indexed_dataframe(fb=fb)
        hole_ids = dict(zip(lookup["key"].astype(int), lookup["value"].astype(str), strict=True))
        result: dict[str, list[tuple[int, pd.DataFrame]]] = {}
        for chunk in (await self.holes.to_dataframe(fb=fb)).itertuples(index=False):
            try:
                hole_id = hole_ids[int(chunk.hole_index)]
            except KeyError as exc:
                raise ObjectValidationError(f"Unknown hole_index in collection chunks: {chunk.hole_index}") from exc
            offset = int(chunk.offset)
            result.setdefault(hole_id, []).append(
                (offset, data.iloc[offset : offset + int(chunk.count)].reset_index(drop=True))
            )
        return {
            hole_id: pd.concat([chunk for _, chunk in sorted(chunks, key=lambda item: item[0])], ignore_index=True)
            for hole_id, chunks in result.items()
        }


class DownholeDistanceTable(_DownholeCollectionChild, DistanceTable):
    holes: Annotated[HoleChunksTable, SchemaLocation("holes"), DataLocation("holes")]

    async def to_dataframe_by_hole(self, *keys: str, fb: IFeedback = NoFeedback) -> dict[str, pd.DataFrame]:
        """Return per-hole collection values and selected attributes."""
        return await self._table_by_hole(await self.to_dataframe(*keys, fb=fb), fb=fb)


class IntervalTableFromTo(DataTableAndAttributes):
    _table: Annotated[DepthIntervalsTable, SchemaLocation("intervals.start_and_end"), DataLocation("")]
    unit: Annotated[str | None, SchemaLocation("unit")]


class DownholeIntervalTable(_DownholeCollectionChild, SchemaModel):
    name: Annotated[str, SchemaLocation("name"), DataLocation("name")]
    collection_type: Annotated[str, SchemaLocation("collection_type"), DataLocation("collection_type")]
    from_to: Annotated[IntervalTableFromTo, SchemaLocation("from_to"), DataLocation("table")]
    holes: Annotated[HoleChunksTable, SchemaLocation("holes"), DataLocation("holes")]

    async def to_dataframe(self, *keys: str, fb: IFeedback = NoFeedback) -> pd.DataFrame:
        """Return collection intervals and selected attributes."""
        return await self.from_to.to_dataframe(*keys, fb=fb)

    async def to_dataframe_by_hole(self, *keys: str, fb: IFeedback = NoFeedback) -> dict[str, pd.DataFrame]:
        """Return per-hole collection intervals and selected attributes."""
        return await self._table_by_hole(await self.to_dataframe(*keys, fb=fb), fb=fb)


class DownholeCollectionTables(_DownholeCollectionChild, SchemaList[DownholeDistanceTable | DownholeIntervalTable]):
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
            schema = await model._data_to_schema(collection, context)
            if collection.unit is not None:
                if isinstance(collection, IntervalCollection):
                    schema["from_to"]["unit"] = collection.unit
                else:
                    schema["distance"]["unit"] = collection.unit
            result.append(schema)
        return result

    def _indices_for_name(self, name: str) -> list[int]:
        return [index for index, document in enumerate(self._document) if document.get("name") == name]

    def get(self, name: str) -> DownholeDistanceTable | DownholeIntervalTable | None:
        """Return the first collection named ``name``.

        Collection names are the only schema-provided identifier. Legacy documents may contain duplicate names; in
        that case the first collection in document order is returned and a warning is emitted.
        """
        indices = self._indices_for_name(name)
        if len(indices) > 1:
            warnings.warn(
                f"Multiple collections named '{name}' were found; returning the first one",
                UserWarning,
            )
        return self[indices[0]] if indices else None

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and bool(self._indices_for_name(name))

    def names(self) -> list[str]:
        return [collection.name for collection in self]

    async def add(self, collection: DownholeCollectionEntry, *, replace: bool = False) -> None:
        existing_indices = self._indices_for_name(collection.name)
        if len(existing_indices) > 1:
            raise ObjectValidationError(
                f"Multiple collections named '{collection.name}' already exist; remove duplicates before adding a replacement"
            )
        if existing_indices and not replace:
            raise ObjectValidationError(f"Collection '{collection.name}' already exists")
        location_holes = await self._downhole_collection.location.holes.to_dataframe()
        table = collection.table
        DownholeCollectionData._validate_hole_chunks(collection.holes, len(table), require_coverage=False)
        valid_indices = set(location_holes["hole_index"].astype(int))
        if not set(collection.holes["hole_index"].astype(int)).issubset(valid_indices):
            raise ObjectValidationError("hole_index must reference a key in the location holes table")
        schema = await self._data_to_schema([collection], self._obj)
        if existing_indices:
            self._document[existing_indices[0]] = schema[0]
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

    async def prefetch_collections(
        self,
        *names: str,
        include_location: bool = True,
        max_concurrent: int = 8,
        fb: IFeedback = NoFeedback,
    ) -> None:
        """Prefetch data referenced by named collections and optionally location data."""

        documents = []
        if include_location:
            documents.append(self.location.as_dict())
        for name in names:
            collection = self.collections.get(name)
            if collection is None:
                raise KeyError(f"Unknown collection '{name}'")
            documents.append(collection.as_dict())
        await self.prefetch(data_ids=collect_data_ids(documents), max_concurrent=max_concurrent, fb=fb)
