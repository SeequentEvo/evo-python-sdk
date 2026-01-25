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
from typing import Annotated

import pandas as pd

from evo.common import IFeedback
from evo.common.utils import NoFeedback
from evo.objects import SchemaVersion

from ._model import DataLocation, SchemaLocation, SchemaModel
from .attributes import Attributes
from .exceptions import ObjectValidationError
from .spatial import BaseSpatialObject, BaseSpatialObjectData
from .types import BoundingBox, Point3, Rotation, Size3d, Size3i

__all__ = [
    "Cells",
    "Regular3DGrid",
    "Regular3DGridData",
    "Vertices",
]


@dataclass(kw_only=True, frozen=True)
class Regular3DGridData(BaseSpatialObjectData):
    origin: Point3
    size: Size3i
    cell_size: Size3d
    cell_data: pd.DataFrame | None = None
    vertex_data: pd.DataFrame | None = None
    rotation: Rotation | None = None

    def __post_init__(self):
        if self.cell_data is not None and self.cell_data.shape[0] != self.size.total_size:
            raise ObjectValidationError(
                f"The number of rows in the cell_data dataframe ({self.cell_data.shape[0]}) does not match the number of cells in the grid ({self.size.nx * self.size.ny * self.size.nz})."
            )
        vertices_expected_length = (self.size.nx + 1) * (self.size.ny + 1) * (self.size.nz + 1)
        if self.vertex_data is not None and self.vertex_data.shape[0] != vertices_expected_length:
            raise ObjectValidationError(
                f"The number of rows in the vertex_data dataframe ({self.vertex_data.shape[0]}) does not match the number of vertices in the grid ({self.size.nx * self.size.ny * self.size.nz})."
            )

    def compute_bounding_box(self) -> BoundingBox:
        return BoundingBox.from_regular_grid(self.origin, self.size, self.cell_size, self.rotation)


class Cells(SchemaModel):
    """A dataset representing the cells of a regular 3D grid.

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

    async def as_dataframe(self, fb: IFeedback = NoFeedback) -> pd.DataFrame:
        """Load a DataFrame containing the cell attribute values."""
        return await self.attributes.as_dataframe(fb=fb)

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


class Vertices(SchemaModel):
    """A dataset representing the vertices of a regular 3D grid.

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

    async def as_dataframe(self, fb: IFeedback = NoFeedback) -> pd.DataFrame:
        """Load a DataFrame containing the vertex attribute values."""
        return await self.attributes.as_dataframe(fb=fb)

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


class Regular3DGrid(BaseSpatialObject):
    """A GeoscienceObject representing a regular 3D grid.

    The object contains a dataset for both the cells and the vertices of the grid.

    Each of these datasets only contain attribute columns. The actual geometry of the grid is defined by
    the properties: origin, size, cell_size, and rotation.
    """

    _data_class = Regular3DGridData

    sub_classification = "regular-3d-grid"
    creation_schema_version = SchemaVersion(major=1, minor=3, patch=0)

    cells: Annotated[Cells, SchemaLocation("")]
    vertices: Annotated[Vertices, SchemaLocation("")]
    origin: Annotated[Point3, SchemaLocation("origin")]
    size: Annotated[Size3i, SchemaLocation("size")]
    cell_size: Annotated[Size3d, SchemaLocation("cell_size")]
    rotation: Annotated[Rotation | None, SchemaLocation("rotation")]

    def compute_bounding_box(self) -> BoundingBox:
        return BoundingBox.from_regular_grid(self.origin, self.size, self.cell_size, self.rotation)
