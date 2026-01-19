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

import pandas as pd
from pydantic import TypeAdapter

from evo.common import IFeedback
from evo.common.utils import NoFeedback
from evo.objects import SchemaVersion
from evo.objects.utils.table_formats import FLOAT_ARRAY_3

from ._adapters import AttributesAdapter, TableAdapter
from ._property import SchemaProperty
from .base import BaseSpatialObject, BaseSpatialObjectData, ConstructableObject, DatasetProperty
from .dataset import Attributes, Dataset
from .types import BoundingBox

__all__ = [
    "Locations",
    "PointSet",
    "PointSetData",
]


@dataclass(kw_only=True, frozen=True)
class PointSetData(BaseSpatialObjectData):
    """Data for creating a PointSet.

    A pointset is a collection of points in 3D space with associated attributes.
    """

    locations: pd.DataFrame | None = None

    def compute_bounding_box(self) -> BoundingBox:
        """Compute the bounding box from the point locations.

        The locations dataframe must have 'x', 'y', 'z' columns containing coordinates.
        """
        if self.locations is None or self.locations.empty:
            return BoundingBox(min_x=0, max_x=0, min_y=0, max_y=0, min_z=0, max_z=0)

        return BoundingBox.from_points(
            self.locations["x"].values,
            self.locations["y"].values,
            self.locations["z"].values,
        )


class Locations(Dataset):
    """A dataset representing the locations and attributes of points in a pointset.

    The dataset contains x, y, z coordinates as well as any additional attributes.
    """

    bounding_box: BoundingBox = SchemaProperty("bounding_box", TypeAdapter(BoundingBox))

    async def set_dataframe(self, df: pd.DataFrame, *, fb: IFeedback = NoFeedback):
        """Set the locations dataframe, ensuring it has the required columns."""
        await super().set_dataframe(df, fb=fb)

        # Compute and store the bounding box from the coordinates
        if len(df) > 0:
            self.bounding_box = BoundingBox.from_points(
                df["x"].values,
                df["y"].values,
                df["z"].values,
            )
        else:
            self.bounding_box = BoundingBox(min_x=0, max_x=0, min_y=0, max_y=0, min_z=0, max_z=0)


class PointSet(BaseSpatialObject, ConstructableObject[PointSetData]):
    """A GeoscienceObject representing a set of points in 3D space.

    The object contains a dataset for the locations and attributes of the points.
    The coordinates (x, y, z) are stored as a required float-array-3 in the locations dataset,
    along with any additional optional attributes.
    """

    _data_class = PointSetData

    sub_classification = "pointset"
    creation_schema_version = SchemaVersion(major=1, minor=3, patch=0)

    locations: Locations = DatasetProperty(
        Locations,
        value_adapters=[
            TableAdapter(
                min_major_version=1,
                max_major_version=1,
                column_names=("x", "y", "z"),
                values_path="locations.coordinates",
                table_formats=[FLOAT_ARRAY_3],
            ),
        ],
        attributes_adapters=[
            AttributesAdapter(min_major_version=1, max_major_version=1, attribute_list_path="locations.attributes")
        ],
        extract_data=lambda data: data.locations,
    )

    @property
    def attributes(self) -> Attributes:
        return self.locations.attributes

    async def coordinates(self, fb: IFeedback = NoFeedback) -> pd.DataFrame:
        """Get the coordinates dataframe for the pointset.

        Returns:
            A DataFrame with 'x', 'y', 'z' columns representing point coordinates.
        """
        return await self.locations._values.to_dataframe(fb=fb)

    async def to_dataframe(self, *keys: str, fb: IFeedback = NoFeedback) -> pd.DataFrame:
        """Get the full dataframe for the pointset, including coordinates and attributes.

        Returns:
            A DataFrame with 'x', 'y', 'z' columns and any additional attribute columns.
        """
        return await self.locations.to_dataframe(*keys, fb=fb)