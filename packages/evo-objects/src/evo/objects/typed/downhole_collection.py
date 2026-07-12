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

from dataclasses import dataclass
from typing import Annotated, Any, ClassVar, TypeAlias

import numpy as np
import pandas as pd
from numpy._typing import NDArray

from evo.common.interfaces import IContext
from evo.objects import SchemaVersion
from evo.objects.typed._data import DataTable, DataTableAndAttributes
from evo.objects.typed._model import DataLocation, SchemaList, SchemaLocation, SchemaModel
from evo.objects.typed.attributes import (
    AttributeDescription,
    Attributes,
    Category,
)
from evo.objects.typed.exceptions import ObjectValidationError
from evo.objects.typed.spatial import BaseSpatialObject, BaseSpatialObjectData
from evo.objects.typed.types import BoundingBox
from evo.objects.utils.table_formats import (
    DOWNHOLE_COLLECTION_LOCATION_HOLES,
    FLOAT_ARRAY_1,
    FLOAT_ARRAY_3,
    KnownTableFormat,
)

__all__ = [
    "DownholeCollection",
    "DownholeCollectionData",
]

_X = "x"
_Y = "y"
_Z = "z"
_COORDINATE_COLUMNS = [_X, _Y, _Z]


HolePath: TypeAlias = pd.DataFrame  # [ distance | dip | azimuth | <attributes> ]
HoleChunks: TypeAlias = pd.DataFrame  # [ hole_id | offset | count ]
HoleProperties: TypeAlias = pd.DataFrame  # [ hole_id | final | target | current | x | y | z ]
HoleAttributes: TypeAlias = pd.DataFrame

# If `Depths` has unit descriptions in its `DataFrame.attrs` dictionary, then those units will be used when building
# the schema object.
# This is the expected structure:
#   >>> depths_df.attrs
#   {'attribute_description': {<column names>:  <AttributeDescription>}, ...}
Depths: TypeAlias = pd.DataFrame  # [ distance | <attributes> ]


@dataclass
class DistanceCollection:
    name: str
    holes: HoleChunks
    distance_table: Depths
    collection_type: str = "distance"


@dataclass(kw_only=True, frozen=True)
class DownholeCollectionData(BaseSpatialObjectData):
    """Data class for creating a new DownholeCollection

    :param name: The name of the object.
    :param holes: A DataFrame describing which parts of `path` belong to which holes.
            Columns: hole_id, offset, count
    :param properties: DataFrame for the properties of the holes. The ith row corresponds to the ith element of `holes`.
            Mandatory columns: hole_id, final, target, current, x, y, z
    :param attributes: DataFrame for the attributes of the holes. The ith row corresponds to the ith element of `holes`.
    :param path: Dataframe of [ distance | dip | azimuth | <attributes> ]. Distance/dip/azimuth describe the geometry as
            the step since the previous row.
    :param collections: A list of `DistanceCollection` describing a table of distances with attributes.
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
    collections: list[DistanceCollection]
    distance_unit: str | None
    desurvey: str | None

    def __post_init__(self):
        if self.attributes is not None and len(self.holes) != len(self.attributes):
            raise ObjectValidationError("The number of attributes rows must match the number or holes rows")

        assert self.attributes is None or len(self.holes) == len(self.attributes)

    def compute_bounding_box(self) -> BoundingBox:
        bboxes = []

        for i in range(len(self.holes)):
            offset = self.holes.iat[i, 1]
            count = self.holes.iat[i, 2]
            collar = tuple(self.properties.loc[i, _COORDINATE_COLUMNS])
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


class DownholeCategory(Category):
    @classmethod
    def _extract_category_table(cls, data: HoleAttributes):
        return data[["hole_id"]].astype("category")

    @classmethod
    async def _data_to_schema(cls, data: HoleAttributes, context: IContext) -> Any:
        category_table = cls._extract_category_table(data)
        return await super()._data_to_schema(category_table, context=context)


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
    hole_id: Annotated[DownholeCategory, SchemaLocation("hole_id"), DataLocation("properties")]
    path: Annotated[DownholePath, SchemaLocation("path"), DataLocation("path")]
    holes: Annotated[HoleChunksTable, SchemaLocation("holes"), DataLocation("holes")]
    distances: Annotated[DistancesTable, SchemaLocation("distances"), DataLocation("properties")]
    coordinates: Annotated[CollarCoordinates, SchemaLocation("coordinates"), DataLocation("properties")]
    attributes: Annotated[Attributes, SchemaLocation("attributes"), DataLocation("attributes")]


class _Distances(DataTable):
    table_format: ClassVar[KnownTableFormat] = FLOAT_ARRAY_1
    data_columns: ClassVar[list[str]] = ["distance"]


class DistanceTableDistances(DataTableAndAttributes):
    _table: Annotated[_Distances, SchemaLocation("values"), DataLocation("")]
    unit: Annotated[str | None, SchemaLocation("unit")]

    @classmethod
    async def _data_to_schema(cls, data: pd.DataFrame, context: IContext) -> Any:
        result = await super()._data_to_schema(data, context)
        attr_desc: AttributeDescription = data.attrs.get("attribute_descriptions", {}).get("distance")
        if attr_desc is not None and attr_desc.unit is not None:
            # "unit" can be missing, but it must not be `None`
            result["unit"] = attr_desc.unit
        return result


class DistanceTable(SchemaModel):
    name: Annotated[str, SchemaLocation("name"), DataLocation("name")]
    collection_type: Annotated[str, SchemaLocation("collection_type"), DataLocation("collection_type")]
    distance: Annotated[DistanceTableDistances, SchemaLocation("distance"), DataLocation("distance_table")]


class DownholeDistanceTable(DistanceTable):
    holes: Annotated[HoleChunksTable, SchemaLocation("holes"), DataLocation("holes")]


class DownholeCollectionTables(SchemaList[DownholeDistanceTable]):
    pass


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
