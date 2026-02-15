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

import uuid
from dataclasses import dataclass
from typing import Annotated, Any

import numpy as np
import pandas as pd
import pyarrow as pa

from evo.common import IContext, IFeedback
from evo.common.utils import NoFeedback
from evo.objects import SchemaVersion
from evo.objects.utils.table_formats import BOOL_ARRAY_1

from ._grid import BaseRegular3DGrid, BaseRegular3DGridData
from ._model import DataLocation, SchemaLocation, SchemaModel
from ._utils import assign_jmespath_value, get_data_client
from .attributes import Attributes
from .exceptions import DataLoaderError, ObjectValidationError
from .types import Size3i


__all__ = [
    "MaskedCells",
    "RegularMasked3DGrid",
    "RegularMasked3DGridData",
]


@dataclass(kw_only=True, frozen=True)
class RegularMasked3DGridData(BaseRegular3DGridData):
    mask: np.ndarray

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


class MaskedCells(SchemaModel):
    """A dataset representing the cells of a masked regular 3D grid.

    The order of the cells is in column-major order, i.e. for an unrotated grid: x changes fastest, then y, then z.
    Only active cells (where mask is True) have attribute values.
    """

    _grid_size: Annotated[Size3i, SchemaLocation("size")]
    _mask_length: Annotated[int, SchemaLocation("mask.values.length")]
    number_active: Annotated[int, SchemaLocation("number_of_active_cells")]
    attributes: Annotated[Attributes, SchemaLocation("cell_attributes"), DataLocation("cell_data")]

    @property
    def size(self) -> Size3i:
        return self._grid_size

    @property
    def expected_length(self) -> int:
        return self.number_active

    async def get_mask(self, *, fb: IFeedback = NoFeedback) -> np.ndarray:
        """Get the mask for the grid cells.

        :return: A boolean numpy array representing the mask for the grid cells.
        """
        array = await self._obj.download_array("mask.values", fb=fb)
        if array.dtype != np.bool_:
            raise DataLoaderError(f"Expected mask array to have dtype 'bool', but got '{array.dtype}'")
        return array

    async def to_dataframe(self, *keys: str, fb: IFeedback = NoFeedback) -> pd.DataFrame:
        """Load a DataFrame containing the cell attribute values.

        :param keys: Optional list of attribute keys to include. If not provided, all attributes are included.
        :param fb: Optional feedback object to report download progress.
        :return: A DataFrame with cell attribute columns.
        """
        return await self.attributes.to_dataframe(*keys, fb=fb)

    async def from_dataframe(
        self, df: pd.DataFrame, mask: np.ndarray | None = None, *, fb: IFeedback = NoFeedback
    ) -> None:
        """Set the cell attributes from a DataFrame.

        :param df: DataFrame containing the values for the active cells.
        :param mask: Optional new mask array. If provided, the mask will be updated.
        :param fb: Optional feedback object to report progress.
        """
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
            data_client = get_data_client(self._obj)
            table_info = await data_client.upload_table(
                table=pa.table({"mask": pa.array(mask)}),
                table_format=BOOL_ARRAY_1,
                fb=fb,
            )
            assign_jmespath_value(self._document, "mask.values", table_info)
            # Mark mask data as modified
            self._context.mark_modified("mask.values.data")

        await self.attributes.set_attributes(df, fb=fb)

    def validate(self) -> None:
        """Validate that all attributes have the correct length and mask is valid."""
        self.attributes.validate_lengths(self.expected_length)
        if self.size.total_size != self._mask_length:
            raise ObjectValidationError(
                f"The length of the mask ({self._mask_length}) does not match the number of cells in the grid ({self.size.total_size})."
            )

    @classmethod
    async def _data_to_schema(cls, data: Any, context: IContext) -> dict[str, Any]:
        """Convert data to a dictionary for schema creation."""
        result = await super()._data_to_schema(data, context)

        mask = data.mask
        data_client = get_data_client(context)
        table_info = await data_client.upload_table(
            table=pa.table({"mask": pa.array(mask)}),
            table_format=BOOL_ARRAY_1,
        )
        result["mask"] = {
            "name": "mask",
            "key": str(uuid.uuid4()),
            "attribute_type": "bool",
            "values": table_info,
        }
        result["number_of_active_cells"] = int(np.sum(mask))
        return result


class RegularMasked3DGrid(BaseRegular3DGrid):
    """A GeoscienceObject representing a regular masked 3D grid.

    The object contains a dataset for the cells of the grid, where only active cells
    (as defined by the mask) have attribute values.
    """

    _data_class = RegularMasked3DGridData

    sub_classification = "regular-masked-3d-grid"
    creation_schema_version = SchemaVersion(major=1, minor=3, patch=0)

    cells: Annotated[MaskedCells, SchemaLocation("")]

    @property
    def attributes(self) -> Attributes:
        """The cell attributes of this grid (alias for cells.attributes)."""
        return self.cells.attributes

    async def to_dataframe(self, *keys: str, fb: IFeedback = NoFeedback) -> pd.DataFrame:
        """Get a dataframe containing the cell attributes.

        :param keys: Optional list of attribute keys to include. If not provided, all attributes are included.
        :param fb: Optional feedback object to report download progress.
        :return: A DataFrame with cell attribute columns.
        """
        return await self.cells.to_dataframe(*keys, fb=fb)

