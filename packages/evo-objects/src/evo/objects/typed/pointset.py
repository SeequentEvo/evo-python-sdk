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
from .exceptions import ObjectValidationError
from .spatial import BaseSpatialObject, BaseSpatialObjectData
from .types import BoundingBox

__all__ = [
    "Locations",
    "PointSet",
    "PointSetData",
]

_COORDINATE_COLUMNS = ["x", "y", "z"]


def _bounding_box_from_dataframe(df: pd.DataFrame) -> BoundingBox:
    return BoundingBox.from_points(
        df["x"].values,
        df["y"].values,
        df["z"].values,
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
    """DataTable subclass for point coordinates with X, Y, Z columns."""

    table_format: ClassVar[KnownTableFormat] = FLOAT_ARRAY_3
    data_columns: ClassVar[list[str]] = _COORDINATE_COLUMNS

    async def set_dataframe(self, df: pd.DataFrame, fb: IFeedback = NoFeedback):
        await super().set_dataframe(df, fb)

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
