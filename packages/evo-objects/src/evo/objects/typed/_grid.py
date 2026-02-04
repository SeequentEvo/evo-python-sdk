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

"""Base classes for 3D grid objects."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Annotated

import pandas as pd

from evo.common import IFeedback
from evo.common.utils import NoFeedback

from ._model import DataLocation, SchemaLocation, SchemaModel
from .attributes import Attributes
from .exceptions import ObjectValidationError
from .spatial import BaseSpatialObject, BaseSpatialObjectData
from .types import BoundingBox, Point3, Rotation, Size3d, Size3i

__all__ = [
    "Base3DGrid",
    "Base3DGridData",
    "BaseRegular3DGrid",
    "BaseRegular3DGridData",
]


@dataclass(kw_only=True, frozen=True)
class Base3DGridData(BaseSpatialObjectData):
    """Base class for all 3D grid data.

    Contains the common properties shared by all grid types: origin, size, rotation, and cell_data.
    """

    origin: Point3
    size: Size3i
    cell_data: pd.DataFrame | None = None
    rotation: Rotation | None = None


@dataclass(kw_only=True, frozen=True)
class BaseRegular3DGridData(Base3DGridData):
    """Base class for regular 3D grid data (both masked and non-masked).

    Contains the common properties shared by Regular3DGridData and RegularMasked3DGridData.
    Adds cell_size to the base grid properties.
    """

    cell_size: Size3d

    def compute_bounding_box(self) -> BoundingBox:
        return BoundingBox.from_regular_grid(self.origin, self.size, self.cell_size, self.rotation)


class Base3DGrid(BaseSpatialObject, ABC):
    """Base class for all 3D grid objects.

    Contains the common properties shared by all grid types: origin, size, and rotation.
    The bounding box is dynamically computed from the grid properties.
    """

    origin: Annotated[Point3, SchemaLocation("origin")]
    size: Annotated[Size3i, SchemaLocation("size")]
    rotation: Annotated[Rotation | None, SchemaLocation("rotation")]

    @abstractmethod
    def compute_bounding_box(self) -> BoundingBox:
        """Compute the bounding box for the grid."""
        # This class does not have enough information about the grid cell sizes to compute the bounding box.
        raise NotImplementedError(
            "Subclasses must implement compute_bounding_box to derive bounding box from grid properties."
        )

    @property
    def bounding_box(self) -> BoundingBox:
        return self.compute_bounding_box()

    @bounding_box.setter
    def bounding_box(self, value: BoundingBox) -> None:
        raise AttributeError("Cannot set bounding_box on this object, as it is dynamically derived from the data.")

    async def update(self):
        """Update the object on the geoscience object service, including recomputing the bounding box."""
        self._bounding_box = self.compute_bounding_box()
        await super().update()


class BaseRegular3DGrid(Base3DGrid):
    """Base class for regular 3D grid objects (both masked and non-masked).

    Contains the common properties shared by Regular3DGrid and RegularMasked3DGrid.
    Adds cell_size to the base grid properties.
    """

    cell_size: Annotated[Size3d, SchemaLocation("cell_size")]

    def compute_bounding_box(self) -> BoundingBox:
        return BoundingBox.from_regular_grid(self.origin, self.size, self.cell_size, self.rotation)


class Cells3D(SchemaModel):
    """A dataset representing the cells of a non-masked 3D grid.

    The order of the cells are in column-major order, i.e. for a unrotated grid: x changes fastest, then y, then z.
    """

    _grid_size: Annotated[Size3i, SchemaLocation("size")]
    attributes: Annotated[Attributes, SchemaLocation("cell_attributes"), DataLocation("cell_data")]

    @property
    def size(self) -> Size3i:
        return self._grid_size

    @property
    def expected_length(self) -> int:
        return self.size.total_size

    async def get_dataframe(self, fb: IFeedback = NoFeedback) -> pd.DataFrame:
        """Load a DataFrame containing the cell attribute values."""
        return await self.attributes.get_dataframe(fb=fb)

    async def to_dataframe(self, *keys: str, fb: IFeedback = NoFeedback) -> pd.DataFrame:
        """Load a DataFrame containing the cell attribute values.

        :param keys: Optional list of attribute keys to include. If not provided, all attributes are included.
        :param fb: Optional feedback object to report download progress.
        :return: A DataFrame with cell attribute columns.
        """
        if keys:
            return await self.attributes.get_dataframe(*keys, fb=fb)
        return await self.attributes.get_dataframe(fb=fb)

    async def set_dataframe(self, df: pd.DataFrame, fb: IFeedback = NoFeedback) -> None:
        """Set the cell attributes from a DataFrame."""
        if df.shape[0] != self.expected_length:
            raise ObjectValidationError(
                f"The number of rows in the dataframe ({df.shape[0]}) does not match the number of cells in the grid ({self.expected_length})."
            )
        await self.attributes.set_attributes(df, fb=fb)

    def validate(self) -> None:
        """Validate that all attributes have the correct length."""
        self.attributes.validate_lengths(self.expected_length)


class Vertices3D(SchemaModel):
    """A dataset representing the vertices of a non-masked 3D grid.

    The order of the vertices are in column-major order, i.e. for a unrotated grid: x changes fastest, then y, then z.
    """

    _grid_size: Annotated[Size3i, SchemaLocation("size")]
    attributes: Annotated[Attributes, SchemaLocation("vertex_attributes"), DataLocation("vertex_data")]

    @property
    def size(self) -> Size3i:
        grid_size = self._grid_size
        return Size3i(nx=grid_size.nx + 1, ny=grid_size.ny + 1, nz=grid_size.nz + 1)

    @property
    def expected_length(self) -> int:
        return self.size.total_size

    async def get_dataframe(self, fb: IFeedback = NoFeedback) -> pd.DataFrame:
        """Load a DataFrame containing the vertex attribute values."""
        return await self.attributes.get_dataframe(fb=fb)

    async def to_dataframe(self, *keys: str, fb: IFeedback = NoFeedback) -> pd.DataFrame:
        """Load a DataFrame containing the vertex attribute values.

        :param keys: Optional list of attribute keys to include. If not provided, all attributes are included.
        :param fb: Optional feedback object to report download progress.
        :return: A DataFrame with vertex attribute columns.
        """
        if keys:
            return await self.attributes.get_dataframe(*keys, fb=fb)
        return await self.attributes.get_dataframe(fb=fb)

    async def set_dataframe(self, df: pd.DataFrame, fb: IFeedback = NoFeedback) -> None:
        """Set the vertex attributes from a DataFrame."""
        if df.shape[0] != self.expected_length:
            raise ObjectValidationError(
                f"The number of rows in the dataframe ({df.shape[0]}) does not match the number of vertices in the grid ({self.expected_length})."
            )
        await self.attributes.set_attributes(df, fb=fb)

    def validate(self) -> None:
        """Validate that all attributes have the correct length."""
        self.attributes.validate_lengths(self.expected_length)
