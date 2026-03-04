#  Copyright © 2025 Bentley Systems, Incorporated
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
from typing import Annotated, ClassVar

import pandas as pd

from evo.common.interfaces import IFeedback
from evo.common.utils import NoFeedback
from evo.objects import SchemaVersion
from evo.objects.utils.table_formats import FLOAT_ARRAY_3, KnownTableFormat

from ._data import DataTable, DataTableAndAttributes
from ._model import DataLocation, SchemaLocation
from .attributes import Attributes
from .exceptions import ObjectValidationError
from .spatial import BaseSpatialObject, BaseSpatialObjectData
from .types import BoundingBox

__all__ = [
    "Locations",
    "PointSet",
    "PointSetData",
]

_X = "x"
_Y = "y"
_Z = "z"
_COORDINATE_COLUMNS = [_X, _Y, _Z]


def _bounding_box_from_dataframe(df: pd.DataFrame) -> BoundingBox:
    return BoundingBox.from_points(
        df[_X].values,
        df[_Y].values,
        df[_Z].values,
    )


@dataclass(kw_only=True, frozen=True)
class PointSetData(BaseSpatialObjectData):
    """Data class for creating a new PointSet object.

    :param name: The name of the object.
    :param locations: A DataFrame containing the point data. Must have 'x', 'y', 'z' columns for coordinates.
        Any additional columns will be treated as point attributes.
    :param coordinate_reference_system: Optional EPSG code or WKT string for the coordinate reference system.
    :param description: Optional description of the object.
    :param tags: Optional dictionary of tags for the object.
    :param extensions: Optional dictionary of extensions for the object.
    """

    locations: pd.DataFrame

    def __post_init__(self):
        missing = set(_COORDINATE_COLUMNS) - set(self.locations.columns)
        if missing:
            raise ObjectValidationError(f"locations DataFrame must have 'x', 'y', 'z' columns. Missing: {missing}")

    def compute_bounding_box(self) -> BoundingBox:
        return _bounding_box_from_dataframe(self.locations)


class CoordinateTable(DataTable):
    """DataTable subclass for point coordinates with x, y, z columns."""

    table_format: ClassVar[KnownTableFormat] = FLOAT_ARRAY_3
    data_columns: ClassVar[list[str]] = _COORDINATE_COLUMNS

    async def from_dataframe(self, df: pd.DataFrame, fb: IFeedback = NoFeedback):
        """Update the coordinate values and recalculate the bounding box.

        :param df: DataFrame containing x, y, z coordinate columns.
        :param fb: Optional feedback object to report upload progress.
        """
        await super().from_dataframe(df, fb)

        # Update the bounding box in the parent object context
        self._context.root_model.bounding_box = _bounding_box_from_dataframe(df)


class Locations(DataTableAndAttributes):
    """A dataset representing the locations (points) of a PointSet.

    Contains the coordinates of each point and optional attributes.
    """

    _table: Annotated[CoordinateTable, SchemaLocation("coordinates")]


class PointSet(BaseSpatialObject):
    """A GeoscienceObject representing a set of points in 3D space.

    The object contains a locations dataset with coordinates and optional attributes
    for each point.
    """

    _data_class = PointSetData

    sub_classification = "pointset"
    creation_schema_version = SchemaVersion(major=1, minor=2, patch=0)

    locations: Annotated[Locations, SchemaLocation("locations"), DataLocation("locations")]

    @property
    def num_points(self) -> int:
        """The number of points in this pointset."""
        return self.locations.length

    @property
    def attributes(self) -> "Attributes":
        """The attributes associated with the points in this pointset."""
        return self.locations.attributes

    async def coordinates(self, fb: IFeedback = NoFeedback) -> pd.DataFrame:
        """Get the coordinates dataframe for the pointset.

        :param fb: Optional feedback object to report download progress.
        :return: A DataFrame with 'x', 'y', 'z' columns representing point coordinates.
        """
        return await self.locations._table.get_dataframe(fb=fb)

    async def to_dataframe(self, *keys: str, fb: IFeedback = NoFeedback) -> pd.DataFrame:
        """Get the full dataframe for the pointset, including coordinates and attributes.

        :param keys: Optional list of attribute keys to include. If not provided, all attributes are included.
        :param fb: Optional feedback object to report download progress.
        :return: A DataFrame with 'x', 'y', 'z' columns and any additional attribute columns.
        """
        return await self.locations.to_dataframe(*keys, fb=fb)
