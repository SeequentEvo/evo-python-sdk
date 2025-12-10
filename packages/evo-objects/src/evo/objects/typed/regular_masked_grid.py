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
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
from pydantic import TypeAdapter

from evo import jmespath
from evo.common import IContext, IFeedback
from evo.common.utils import NoFeedback
from evo.objects import DownloadedObject, SchemaVersion
from evo.objects.utils.table_formats import BOOL_ARRAY_1

from ._adapters import AttributesAdapter, DatasetAdapter
from ._property import SchemaProperty
from ._utils import assign_jmespath_value, get_data_client
from .base import BaseSpatialObjectData, ConstructableObject, DatasetProperty, DynamicBoundingBoxSpatialObject
from .dataset import DataLoaderError, Dataset
from .exceptions import ObjectValidationError
from .types import BoundingBox, Point3, Rotation, Size3d, Size3i

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

__all__ = [
    "MaskedCells",
    "RegularMasked3DGrid",
    "RegularMasked3DGridData",
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
        context: IContext,
        obj: DownloadedObject | None = None,
    ):
        super().__init__(document, dataset_adapter, context, obj)
        self._context = context
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
            data_client = get_data_client(self._context)
            table_info = await data_client.upload_table(
                table=pa.table({"mask": pa.array(mask)}),
                table_format=BOOL_ARRAY_1,
                fb=fb,
            )
            assign_jmespath_value(self._document, "mask.values", table_info)
            self._obj = None  # Invalidate the DownloadedObject since data has changed

        await super().set_dataframe(df, fb=fb)

    def _expected_length(self) -> int:
        return self.number_active

    def validate(self) -> None:
        super().validate()
        mask_length = jmespath.search("mask.values.length", self._document)
        if self.size.total_size != mask_length:
            raise DataLoaderError(
                f"The length of the mask ({mask_length}) does not match the number of cells in the grid ({self.size.total_size})."
            )

    @classmethod
    async def create_from_data(
        cls, document: dict, data: Any, dataset_adapter: DatasetAdapter, context: IContext
    ) -> Self:
        dataset = await super().create_from_data(document, None, dataset_adapter, context)

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


class RegularMasked3DGrid(DynamicBoundingBoxSpatialObject, ConstructableObject[RegularMasked3DGridData]):
    """A GeoscienceObject representing a regular masked 3D grid.

    The object contains a dataset for the cells of the grid.
    """

    _data_class = RegularMasked3DGridData

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

    def compute_bounding_box(self) -> BoundingBox:
        return _calculate_bounding_box(self.origin, self.size, self.cell_size, self.rotation)
