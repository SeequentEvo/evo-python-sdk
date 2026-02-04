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
from typing import TYPE_CHECKING, Annotated

import numpy as np
import pandas as pd
from pydantic import AfterValidator, PlainSerializer

from evo.common.interfaces import IFeedback
from evo.common.utils import NoFeedback
from evo.objects import SchemaVersion

from ._grid import Base3DGrid, Base3DGridData, Cells3D, Vertices3D
from ._model import SchemaLocation
from .exceptions import ObjectValidationError
from .types import BoundingBox, Size3d

if TYPE_CHECKING:
    from .attributes import Attributes

__all__ = [
    "Tensor3DGrid",
    "Tensor3DGridData",
]


# Custom type for numpy float arrays that validate/serialize from/to lists
def _validate_numpy_array(v: list[float] | np.ndarray) -> np.ndarray:
    if isinstance(v, np.ndarray):
        return v
    return np.array(v, dtype=np.float64)


def _serialize_numpy_array(v: np.ndarray | list[float]) -> list[float]:
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


NumpyFloat1D = Annotated[
    list[float],
    AfterValidator(_validate_numpy_array),
    PlainSerializer(_serialize_numpy_array, return_type=list[float]),
]


@dataclass(kw_only=True, frozen=True)
class Tensor3DGridData(Base3DGridData):
    """Data for creating a Tensor3DGrid.

    A tensor grid is a 3D grid where cells may have different sizes.
    The grid is defined by an origin, the number of cells in each direction,
    and arrays of cell sizes along each axis.
    """

    cell_sizes_x: np.ndarray  # Array of cell sizes along x-axis (length = size.nx)
    cell_sizes_y: np.ndarray  # Array of cell sizes along y-axis (length = size.ny)
    cell_sizes_z: np.ndarray  # Array of cell sizes along z-axis (length = size.nz)
    vertex_data: pd.DataFrame | None = None

    def __post_init__(self):
        # Validate cell size array lengths
        if self.cell_sizes_x.shape[0] != self.size.nx:
            raise ObjectValidationError(
                f"The number of x cell sizes ({self.cell_sizes_x.shape[0]}) does not match the grid size ({self.size.nx})."
            )
        if self.cell_sizes_y.shape[0] != self.size.ny:
            raise ObjectValidationError(
                f"The number of y cell sizes ({self.cell_sizes_y.shape[0]}) does not match the grid size ({self.size.ny})."
            )
        if self.cell_sizes_z.shape[0] != self.size.nz:
            raise ObjectValidationError(
                f"The number of z cell sizes ({self.cell_sizes_z.shape[0]}) does not match the grid size ({self.size.nz})."
            )

        # Validate cell sizes are positive
        if np.any(self.cell_sizes_x <= 0):
            raise ObjectValidationError("All x cell sizes must be positive.")
        if np.any(self.cell_sizes_y <= 0):
            raise ObjectValidationError("All y cell sizes must be positive.")
        if np.any(self.cell_sizes_z <= 0):
            raise ObjectValidationError("All z cell sizes must be positive.")

        # Validate cell data size
        if self.cell_data is not None and self.cell_data.shape[0] != self.size.total_size:
            raise ObjectValidationError(
                f"The number of rows in the cell_data dataframe ({self.cell_data.shape[0]}) does not match the number of cells in the grid ({self.size.total_size})."
            )

        # Validate vertex data size
        vertices_expected_length = (self.size.nx + 1) * (self.size.ny + 1) * (self.size.nz + 1)
        if self.vertex_data is not None and self.vertex_data.shape[0] != vertices_expected_length:
            raise ObjectValidationError(
                f"The number of rows in the vertex_data dataframe ({self.vertex_data.shape[0]}) does not match the number of vertices in the grid ({vertices_expected_length})."
            )

    def compute_bounding_box(self) -> BoundingBox:
        """Compute the bounding box from the origin, cell sizes, and rotation."""
        extent = Size3d(
            dx=float(np.sum(self.cell_sizes_x)),
            dy=float(np.sum(self.cell_sizes_y)),
            dz=float(np.sum(self.cell_sizes_z)),
        )
        return BoundingBox.from_extent(self.origin, extent, self.rotation)


class Tensor3DGrid(Base3DGrid):
    """A GeoscienceObject representing a tensor 3D grid.

    A tensor grid is a 3D grid where cells may have different sizes. The grid is defined
    by an origin, the number of cells in each direction, and arrays of cell sizes along
    each axis. The grid contains datasets for both cells and vertices.
    """

    _data_class = Tensor3DGridData

    sub_classification = "tensor-3d-grid"
    creation_schema_version = SchemaVersion(major=1, minor=3, patch=0)

    cells: Annotated[Cells3D, SchemaLocation("")]
    vertices: Annotated[Vertices3D, SchemaLocation("")]
    cell_sizes_x: Annotated[NumpyFloat1D, SchemaLocation("grid_cells_3d.cell_sizes_x")]
    cell_sizes_y: Annotated[NumpyFloat1D, SchemaLocation("grid_cells_3d.cell_sizes_y")]
    cell_sizes_z: Annotated[NumpyFloat1D, SchemaLocation("grid_cells_3d.cell_sizes_z")]

    def compute_bounding_box(self) -> BoundingBox:
        """Compute the bounding box from the grid properties."""
        extent = Size3d(
            dx=float(np.sum(self.cell_sizes_x)),
            dy=float(np.sum(self.cell_sizes_y)),
            dz=float(np.sum(self.cell_sizes_z)),
        )
        return BoundingBox.from_extent(self.origin, extent, self.rotation)

    @property
    def attributes(self) -> "Attributes":
        """The cell attributes of this grid (alias for cells.attributes)."""
        return self.cells.attributes

    @property
    def vertex_attributes(self) -> "Attributes":
        """The vertex attributes of this grid."""
        return self.vertices.attributes

    async def to_dataframe(self, *keys: str, fb: IFeedback = NoFeedback) -> pd.DataFrame:
        """Get a dataframe containing the cell attributes.

        :param keys: Optional list of attribute keys to include. If not provided, all attributes are included.
        :param fb: Optional feedback object to report download progress.
        :return: A DataFrame with cell attribute columns.
        """
        if keys:
            return await self.cells.to_dataframe(*keys, fb=fb)
        return await self.cells.to_dataframe(fb=fb)

