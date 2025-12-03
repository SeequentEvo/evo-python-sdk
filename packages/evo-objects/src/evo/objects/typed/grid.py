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

import numpy as np
import pandas as pd
from pydantic import TypeAdapter

from evo.common import EvoContext, IFeedback
from evo.common.utils import NoFeedback
from evo.objects import SchemaVersion

from ._adapters import AttributesAdapter
from ._store import Dataset
from .base import BaseSpatialObject, BaseSpatialObjectData, DatasetProperty, SchemaProperty
from .types import BoundingBox, Point3, Rotation, Size3d, Size3i

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

__all__ = [
    "Cells",
    "Regular3DGrid",
    "Regular3DGridData",
    "SizeChangeError",
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
    print(rotated_corners)
    print(corners)
    print(rotation_matrix)
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
        if self.cell_data is not None and self.cell_data.shape[0] != self.size.nx * self.size.ny * self.size.nz:
            raise ValueError(
                f"The number of rows in the cell_data dataframe ({self.cell_data.shape[0]}) does not match the number of cells in the grid ({self.size.nx * self.size.ny * self.size.nz})."
            )
        if self.vertex_data is not None and self.vertex_data.shape[0] != (self.size.nx + 1) * (self.size.ny + 1) * (
            self.size.nz + 1
        ):
            raise ValueError(
                f"The number of rows in the vertex_data dataframe ({self.vertex_data.shape[0]}) does not match the number of vertices in the grid ({self.size.nx * self.size.ny * self.size.nz})."
            )

    def compute_bounding_box(self) -> BoundingBox:
        return _calculate_bounding_box(self.origin, self.size, self.cell_size, self.rotation)


class SizeChangeError(Exception):
    """Exception raised when the size of the grid cannot be changed due to existing attributes in the datasets."""

    pass


class Cells(Dataset):
    """A dataset representing the cells of a regular 3D grid.

    The order of the cells is assumed to be in z-fastest order.
    """

    size: Size3i

    async def set_dataframe(self, df: pd.DataFrame, fb: IFeedback = NoFeedback) -> None:
        if df.shape[0] != self.size.nx * self.size.ny * self.size.nz:
            raise SizeChangeError(
                f"The number of rows in the dataframe ({df.shape[0]}) does not match the number of cells in the grid ({self.size.nx * self.size.ny * self.size.nz})."
            )
        await super().set_dataframe(df, fb)


class Vertices(Dataset):
    """A dataset representing the vertices of a regular 3D grid.

    The order of the cells is assumed to be in z-fastest order.
    """

    size: Size3i

    def set_dataframe(self, df: pd.DataFrame, fb: IFeedback = NoFeedback) -> None:
        if df.shape[0] != self.size.nx * self.size.ny * self.size.nz:
            raise SizeChangeError(
                f"The number of rows in the dataframe ({df.shape[0]}) does not match the number of vertices in the grid ({self.size.nx * self.size.ny * self.size.nz})."
            )
        super().set_dataframe(df, fb)


class Regular3DGrid(BaseSpatialObject):
    """A GeoscienceObject representing a regular 3D grid.

    The object contains a dataset for both the cells and the vertices of the grid.

    Each of these datasets only contain attribute columns. The actual geometry of the grid is defined by
    the properties: origin, size, cell_size, and rotation.
    """

    sub_classification = "regular-3d-grid"
    creation_schema_version = SchemaVersion(major=1, minor=3, patch=0)

    cells: Cells = DatasetProperty(
        Cells,
        value_adapters=[],
        attributes_adapters=[
            AttributesAdapter(min_major_version=1, max_major_version=1, attribute_list_path="cell_attributes")
        ],
        data_attribute="cell_data",
    )
    vertices: Vertices = DatasetProperty(
        Vertices,
        value_adapters=[],
        attributes_adapters=[
            AttributesAdapter(min_major_version=1, max_major_version=1, attribute_list_path="vertex_attributes")
        ],
        data_attribute="vertex_data",
    )
    origin: Point3 = SchemaProperty(
        "origin",
        TypeAdapter(Point3),
    )
    size: Size3i = SchemaProperty(
        "size",
        TypeAdapter(Size3i),
    )

    @size.pre_set
    def _check_size_on_datasets(self, value: Size3i) -> None:
        if value != self.cells.size and self.cells.has_attributes():
            raise SizeChangeError("The size property can only be changed if the cells dataset has no attributes.")
        if (
            value != Size3i(nx=self.vertices.size.nx - 1, ny=self.vertices.size.ny - 1, nz=self.vertices.size.nz - 1)
            and self.vertices.has_attributes()
        ):
            raise SizeChangeError("The size property can only be changed if the vertices dataset has no attributes.")

    @size.post_set
    def _set_size_on_datasets(self, value: Size3i) -> None:
        self.cells.size = value
        self.vertices.size = Size3i(nx=value.nx + 1, ny=value.ny + 1, nz=value.nz + 1)

    cell_size: Size3d = SchemaProperty(
        "cell_size",
        TypeAdapter(Size3d),
    )
    rotation: Rotation | None = SchemaProperty(
        "rotation",
        TypeAdapter(Rotation | None),
    )

    @classmethod
    async def create(
        cls,
        evo_context: EvoContext,
        data: Regular3DGridData,
        parent: str | None = None,
    ) -> Self:
        """Create a new Regular3DGrid object.

        :param evo_context: The context to use to call Evo APIs.
        :param data: The data for the Regular3DGrid object.
        :param parent: The parent path for the object.

        :return: The created Regular3DGrid object.
        """
        return await cls._create(
            evo_context=evo_context,
            parent=parent,
            data=data,
        )

    @classmethod
    async def replace(
        cls,
        evo_context: EvoContext,
        reference: str,
        data: Regular3DGridData,
    ) -> Self:
        """Replace an existing Regular3DGrid object.

        :param evo_context: The context to use to call Evo APIs.
        :param reference: The reference of the object to replace.
        :param data: The data for the Regular3DGrid object.

        :return: The new version of the Regular3DGrid object.
        """
        return await cls._replace(
            evo_context=evo_context,
            reference=reference,
            data=data,
        )

    def compute_bounding_box(self) -> BoundingBox:
        return _calculate_bounding_box(self.origin, self.size, self.cell_size, self.rotation)

    def _reset_from_object(self) -> None:
        super()._reset_from_object()
        self.cells.size = self.size
        self.vertices.size = Size3i(nx=self.size.nx + 1, ny=self.size.ny + 1, nz=self.size.nz + 1)
