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

import numpy as np
import pandas as pd
from pydantic import TypeAdapter

from evo.common import IFeedback
from evo.common.utils import NoFeedback
from evo.objects import SchemaVersion

from ._adapters import AttributesAdapter
from ._property import SchemaProperty
from .base import BaseSpatialObject, BaseSpatialObjectData, ConstructableObject, DatasetProperty
from .dataset import Dataset
from .exceptions import ObjectValidationError
from .types import BoundingBox, Point3, Rotation, Size3d, Size3i

__all__ = [
    "Cells",
    "Regular3DGrid",
    "Regular3DGridData",
    "Vertices",
]


def _calculate_bounding_box(
    origin: Point3,
    size: Size3i,
    cell_size: Size3d,
    rotation: Rotation | None = None,
) -> BoundingBox:
    if rotation is not None:
        rotation_matrix = rotation.as_rotation_matrix()
    else:
        rotation_matrix = np.eye(3)
    corners = np.array(
        [
            [0, 0, 0],
            [size.nx * cell_size.dx, 0, 0],
            [0, size.ny * cell_size.dy, 0],
            [0, 0, size.nz * cell_size.dz],
            [size.nx * cell_size.dx, size.ny * cell_size.dy, 0],
            [size.nx * cell_size.dx, 0, size.nz * cell_size.dz],
            [0, size.ny * cell_size.dy, size.nz * cell_size.dz],
            [size.nx * cell_size.dx, size.ny * cell_size.dy, size.nz * cell_size.dz],
        ]
    )
    rotated_corners = rotation_matrix @ corners.T
    return BoundingBox.from_points(
        rotated_corners[0, :] + origin.x, rotated_corners[1, :] + origin.y, rotated_corners[2, :] + origin.z
    )


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
        return _calculate_bounding_box(self.origin, self.size, self.cell_size, self.rotation)


class Cells(Dataset):
    """A dataset representing the cells of a regular 3D grid.

    The order of the cells are in column-major order, i.e. for a unrotated grid: x changes fastest, then y, then z.
    """

    size: Size3i = SchemaProperty(
        "size",
        TypeAdapter(Size3i),
    )

    async def set_dataframe(self, df: pd.DataFrame, fb: IFeedback = NoFeedback) -> None:
        expected_length = self.size.total_size
        if df.shape[0] != expected_length:
            raise ObjectValidationError(
                f"The number of rows in the dataframe ({df.shape[0]}) does not match the number of cells in the grid ({expected_length})."
            )
        await super().set_dataframe(df, fb=fb)

    def validate(self) -> None:
        self._check_length(self.size.total_size)


class Vertices(Dataset):
    """A dataset representing the vertices of a regular 3D grid.

    The order of the cells are in column-major order, i.e. for a unrotated grid: x changes fastest, then y, then z.
    """

    _grid_size: Size3i = SchemaProperty(
        "size",
        TypeAdapter(Size3i),
    )

    @property
    def size(self) -> Size3i:
        grid_size = self._grid_size
        return Size3i(nx=grid_size.nx + 1, ny=grid_size.ny + 1, nz=grid_size.nz + 1)

    async def set_dataframe(self, df: pd.DataFrame, fb: IFeedback = NoFeedback) -> None:
        expected_length = self.size.total_size
        if df.shape[0] != expected_length:
            raise ObjectValidationError(
                f"The number of rows in the dataframe ({df.shape[0]}) does not match the number of vertices in the grid ({expected_length})."
            )
        await super().set_dataframe(df, fb=fb)

    def validate(self) -> None:
        self._check_length(self.size.total_size)


class Regular3DGrid(BaseSpatialObject, ConstructableObject[Regular3DGridData]):
    """A GeoscienceObject representing a regular 3D grid.

    The object contains a dataset for both the cells and the vertices of the grid.

    Each of these datasets only contain attribute columns. The actual geometry of the grid is defined by
    the properties: origin, size, cell_size, and rotation.
    """

    _data_class = Regular3DGridData

    sub_classification = "regular-3d-grid"
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

    cell_size: Size3d = SchemaProperty(
        "cell_size",
        TypeAdapter(Size3d),
    )
    rotation: Rotation | None = SchemaProperty(
        "rotation",
        TypeAdapter(Rotation | None),
    )

    def compute_bounding_box(self) -> BoundingBox:
        return _calculate_bounding_box(self.origin, self.size, self.cell_size, self.rotation)
