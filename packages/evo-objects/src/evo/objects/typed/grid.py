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
import uuid
from dataclasses import dataclass
from typing import Annotated, Any

import numpy as np
import pandas as pd
import pyarrow as pa
from pydantic import AfterValidator, PlainSerializer, TypeAdapter

from evo import jmespath
from evo.common import EvoContext, IFeedback
from evo.common.utils import NoFeedback
from evo.objects import DownloadedObject, SchemaVersion
from evo.objects.utils.table_formats import BOOL_ARRAY_1

from ._adapters import AttributesAdapter, DatasetAdapter
from ._property import SchemaProperty
from ._utils import assign_jmespath_value, get_data_client
from .base import BaseSpatialObject, BaseSpatialObjectData, DatasetProperty
from .dataset import DataLoaderError, Dataset
from .exceptions import ObjectValidationError
from .types import BoundingBox, Point3, Rotation, Size3d, Size3i

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

__all__ = [
    "Cells",
    "MaskedCells",
    "Regular3DGrid",
    "Regular3DGridData",
    "RegularMasked3DGrid",
    "RegularMasked3DGridData",
    "Tensor3DGrid",
    "Tensor3DGridData",
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

    The order of the cells is assumed to be in z-fastest order.
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

    The order of the cells is assumed to be in z-fastest order.
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


@dataclass(kw_only=True, frozen=True)
class RegularMasked3DGridData(BaseSpatialObjectData):
    origin: Point3
    size: Size3i
    cell_size: Size3d
    cell_data: pd.DataFrame | None = None
    mask: np.ndarray
    rotation: Rotation | None = None

    def __post_init__(self):
        if self.mask.shape[0] != self.size.total_size:
            raise ObjectValidationError(
                f"The number of rows in the mask ({self.mask.shape[0]}) does not match the number of cells in the grid ({self.size.nx * self.size.ny * self.size.nz})."
            )
        number_active = np.sum(self.mask)
        if self.cell_data is not None and self.cell_data.shape[0] != number_active:
            raise ObjectValidationError(
                f"The number of rows in the cell_data dataframe ({self.cell_data.shape[0]}) does not match the number of active cells in the grid ({number_active})."
            )

    def compute_bounding_box(self) -> BoundingBox:
        return _calculate_bounding_box(self.origin, self.size, self.cell_size, self.rotation)


class MaskedCells(Dataset):
    """A dataset representing the cells of a masked regular 3D grid.

    The order of the cells is assumed to be in z-fastest order.
    """

    size: Size3i = SchemaProperty(
        "size",
        TypeAdapter(Size3i),
    )
    number_active: int = SchemaProperty(
        "number_of_active_cells",
        TypeAdapter(int),
    )

    def __init__(
        self,
        document: dict,
        dataset_adapter: DatasetAdapter,
        evo_context: EvoContext,
        obj: DownloadedObject | None = None,
    ):
        super().__init__(document, dataset_adapter, evo_context, obj)
        self._evo_context = evo_context
        self._obj = obj

    async def get_mask(self, *, fb: IFeedback = NoFeedback) -> np.ndarray:
        """Get the mask for the grid cells.

        :return: A boolean numpy array representing the mask for the grid cells.
        """
        if self._obj is None:
            raise DataLoaderError("Cannot get mask data without an associated DownloadedObject.")
        array = await self._obj.download_array("mask.values", fb=fb)
        if array.dtype != np.bool_:
            raise DataLoaderError(f"Expected mask array to have dtype 'bool', but got '{array.dtype}'")
        return array

    async def set_dataframe(
        self, df: pd.DataFrame, mask: np.ndarray | None = None, *, fb: IFeedback = NoFeedback
    ) -> None:
        if mask is not None:
            expected_length = self.size.total_size
            if mask.shape[0] != expected_length:
                raise ObjectValidationError(
                    f"The length of the mask ({mask.shape[0]}) does not match the number of cells in the grid ({expected_length})."
                )
            number_active = int(np.sum(mask))
        else:
            number_active = self.number_active

        if df.shape[0] != number_active:
            raise ObjectValidationError(
                f"The number of rows in the dataframe ({df.shape[0]}) does not match the number of valid cells in the grid ({number_active})."
            )
        self.number_active = number_active

        if mask is not None:
            # Upload the mask
            data_client = get_data_client(self._evo_context)
            table_info = await data_client.upload_table(
                table=pa.table({"mask": pa.array(mask)}),
                table_format=BOOL_ARRAY_1,
                fb=fb,
            )
            assign_jmespath_value(self._document, "mask.values", table_info)
            self._obj = None  # Invalidate the DownloadedObject since data has changed

        await super().set_dataframe(df, fb=fb)

    def validate(self) -> None:
        self._check_length(self.number_active)
        mask_length = jmespath.search("mask.values.length", self._document)
        if self.size.total_size != mask_length:
            raise DataLoaderError(
                f"The length of the mask ({mask_length}) does not match the number of cells in the grid ({self.size.total_size})."
            )

    @classmethod
    async def create_from_data(
        cls, document: dict, data: Any, dataset_adapter: DatasetAdapter, evo_context: EvoContext
    ) -> Self:
        dataset = await super().create_from_data(document, None, dataset_adapter, evo_context)

        values, mask = data
        if values is None:
            values = pd.DataFrame(index=range(np.sum(mask)))
        await dataset.set_dataframe(values, mask)

        # Add additional mask metadata to the document
        document["mask"] |= {
            "name": "mask",
            "key": str(uuid.uuid4()),
            "attribute_type": "bool",
        }
        return dataset


class RegularMasked3DGrid(BaseSpatialObject):
    """A GeoscienceObject representing a regular masked 3D grid.

    The object contains a dataset for the cells of the grid.
    """

    sub_classification = "regular-masked-3d-grid"
    creation_schema_version = SchemaVersion(major=1, minor=3, patch=0)

    cells: MaskedCells = DatasetProperty(
        MaskedCells,
        value_adapters=[],
        attributes_adapters=[
            AttributesAdapter(min_major_version=1, max_major_version=1, attribute_list_path="cell_attributes")
        ],
        extract_data=lambda data: (data.cell_data, data.mask),
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

    @classmethod
    async def create(
        cls,
        evo_context: EvoContext,
        data: RegularMasked3DGridData,
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
        data: RegularMasked3DGridData,
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


class Tensor3DGrid(BaseSpatialObject):
    """A GeoscienceObject representing a tensor 3D grid.

    A tensor grid is a 3D grid where cells may have different sizes. The grid is defined
    by an origin, the number of cells in each direction, and arrays of cell sizes along
    each axis. The grid contains datasets for both cells and vertices.
    """

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

    @classmethod
    async def create(
        cls,
        evo_context: EvoContext,
        data: Tensor3DGridData,
        parent: str | None = None,
    ) -> Self:
        """Create a new Tensor3DGrid object.

        :param evo_context: The context to use to call Evo APIs.
        :param data: The data for the Tensor3DGrid object.
        :param parent: The parent path for the object.

        :return: The created Tensor3DGrid object.
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
        data: Tensor3DGridData,
    ) -> Self:
        """Replace an existing Tensor3DGrid object.

        :param evo_context: The context to use to call Evo APIs.
        :param reference: The reference of the object to replace.
        :param data: The data for the Tensor3DGrid object.

        :return: The new version of the Tensor3DGrid object.
        """
        return await cls._replace(
            evo_context=evo_context,
            reference=reference,
            data=data,
        )

    def compute_bounding_box(self) -> BoundingBox:
        """Compute the bounding box from the grid properties."""
        return _calculate_tensor_bounding_box(
            self.origin,
            self.cell_sizes_x,
            self.cell_sizes_y,
            self.cell_sizes_z,
            self.rotation,
        )
