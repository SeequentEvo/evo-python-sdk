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

import sys
from dataclasses import dataclass
from typing import Annotated

import numpy as np
import pandas as pd
from pydantic import AfterValidator, PlainSerializer, TypeAdapter

from evo.objects import SchemaVersion

from ._adapters import AttributesAdapter
from ._property import SchemaProperty
from .base import BaseSpatialObjectData, ConstructableObject, DatasetProperty, DynamicBoundingBoxSpatialObject
from .exceptions import ObjectValidationError
from .regular_grid import Cells, Vertices
from .types import BoundingBox, Point3, Rotation, Size3i

if sys.version_info >= (3, 11):
    pass
else:
    pass

__all__ = [
    "Tensor3DGrid",
    "Tensor3DGridData",
]


def _calculate_tensor_bounding_box(
    origin: Point3,
    cell_sizes_x: np.ndarray,
    cell_sizes_y: np.ndarray,
    cell_sizes_z: np.ndarray,
    rotation: Rotation | None = None,
) -> BoundingBox:
    """Calculate bounding box for a tensor grid."""
    # Calculate the extent of the grid
    extent_x = np.sum(cell_sizes_x)
    extent_y = np.sum(cell_sizes_y)
    extent_z = np.sum(cell_sizes_z)

    if rotation is not None:
        rotation_matrix = rotation.as_rotation_matrix()
    else:
        rotation_matrix = np.eye(3)

    # Define the 8 corners of the grid
    corners = np.array(
        [
            [0, 0, 0],
            [extent_x, 0, 0],
            [0, extent_y, 0],
            [0, 0, extent_z],
            [extent_x, extent_y, 0],
            [extent_x, 0, extent_z],
            [0, extent_y, extent_z],
            [extent_x, extent_y, extent_z],
        ]
    )

    # Apply rotation
    rotated_corners = rotation_matrix @ corners.T

    # Calculate bounding box from rotated corners plus origin
    return BoundingBox.from_points(
        rotated_corners[0, :] + origin.x,
        rotated_corners[1, :] + origin.y,
        rotated_corners[2, :] + origin.z,
    )


@dataclass(kw_only=True, frozen=True)
class Tensor3DGridData(BaseSpatialObjectData):
    """Data for creating a Tensor3DGrid.

    A tensor grid is a 3D grid where cells may have different sizes.
    The grid is defined by an origin, the number of cells in each direction,
    and arrays of cell sizes along each axis.
    """

    origin: Point3
    size: Size3i
    cell_sizes_x: np.ndarray  # Array of cell sizes along x-axis (length = size.nx)
    cell_sizes_y: np.ndarray  # Array of cell sizes along y-axis (length = size.ny)
    cell_sizes_z: np.ndarray  # Array of cell sizes along z-axis (length = size.nz)
    cell_data: pd.DataFrame | None = None
    vertex_data: pd.DataFrame | None = None
    rotation: Rotation | None = None

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
        return _calculate_tensor_bounding_box(
            self.origin,
            self.cell_sizes_x,
            self.cell_sizes_y,
            self.cell_sizes_z,
            self.rotation,
        )


# Pydantic adapter for numpy float arrays that serialize to/from lists
NumpyFloat1D = Annotated[
    list[float],  # Use list[float] as the base type for Pydantic schema
    AfterValidator(lambda v: np.array(v, dtype=np.float64) if not isinstance(v, np.ndarray) else v),
    PlainSerializer(lambda v: v.tolist() if isinstance(v, np.ndarray) else v, return_type=list[float]),
]


class Tensor3DGrid(DynamicBoundingBoxSpatialObject, ConstructableObject[Tensor3DGridData]):
    """A GeoscienceObject representing a tensor 3D grid.

    A tensor grid is a 3D grid where cells may have different sizes. The grid is defined
    by an origin, the number of cells in each direction, and arrays of cell sizes along
    each axis. The grid contains datasets for both cells and vertices.
    """

    _data_class = Tensor3DGridData

    sub_classification = "tensor-3d-grid"
    creation_schema_version = SchemaVersion(major=1, minor=3, patch=0)

    cells: Cells = DatasetProperty(
        Cells,
        value_adapters=[],
        attributes_adapters=[
            AttributesAdapter(min_major_version=1, max_major_version=1, attribute_list_path="cell_attributes")
        ],
        extract_data=lambda data: data.cell_data,
    )
    vertices: Vertices = DatasetProperty(
        Vertices,
        value_adapters=[],
        attributes_adapters=[
            AttributesAdapter(min_major_version=1, max_major_version=1, attribute_list_path="vertex_attributes")
        ],
        extract_data=lambda data: data.vertex_data,
    )
    origin: Point3 = SchemaProperty(
        "origin",
        TypeAdapter(Point3),
    )
    size: Size3i = SchemaProperty(
        "size",
        TypeAdapter(Size3i),
    )
    rotation: Rotation | None = SchemaProperty(
        "rotation",
        TypeAdapter(Rotation | None),
    )
    cell_sizes_x: np.ndarray = SchemaProperty("grid_cells_3d.cell_sizes_x", TypeAdapter(NumpyFloat1D))
    cell_sizes_y: np.ndarray = SchemaProperty("grid_cells_3d.cell_sizes_y", TypeAdapter(NumpyFloat1D))
    cell_sizes_z: np.ndarray = SchemaProperty("grid_cells_3d.cell_sizes_z", TypeAdapter(NumpyFloat1D))

    def compute_bounding_box(self) -> BoundingBox:
        """Compute the bounding box from the grid properties."""
        return _calculate_tensor_bounding_box(
            self.origin,
            self.cell_sizes_x,
            self.cell_sizes_y,
            self.cell_sizes_z,
            self.rotation,
        )
