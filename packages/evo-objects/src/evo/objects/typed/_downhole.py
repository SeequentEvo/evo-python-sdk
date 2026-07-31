#  Copyright © 2026 Bentley Systems, Incorporated
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Shared sub-models for downhole typed objects."""

from __future__ import annotations

from typing import Annotated, Any, ClassVar

import pandas as pd

from evo.common.interfaces import IContext
from evo.objects.utils.table_formats import FLOAT_ARRAY_2, KnownTableFormat

from ._data import DataTable
from ._model import DataLocation, SchemaLocation, SchemaModel
from .attributes import Category

_HOLE_ID_COL = "hole_id"
_FROM_COL = "from"
_TO_COL = "to"
_DEPTH_COLS: list[str] = [_FROM_COL, _TO_COL]


class HoleIdCategory(Category):
    """Categorical hole identifier column, shared by downhole typed objects."""

    @classmethod
    async def _data_to_schema(cls, data: pd.DataFrame, context: IContext) -> Any:
        category_table = data[[_HOLE_ID_COL]].astype("category")
        return await super()._data_to_schema(category_table, context=context)


class DepthIntervalsTable(DataTable):
    table_format: ClassVar[KnownTableFormat] = FLOAT_ARRAY_2
    data_columns: ClassVar[list[str]] = _DEPTH_COLS

    @classmethod
    async def _data_to_schema(cls, data: pd.DataFrame, context: IContext) -> Any:
        return await super()._data_to_schema(data[_DEPTH_COLS], context)


class FromToModel(SchemaModel):
    """Schema model for the `from_to` component of a downhole intervals object."""

    intervals: Annotated[DepthIntervalsTable, SchemaLocation("intervals.start_and_end"), DataLocation("intervals")]
    unit: Annotated[str | None, SchemaLocation("unit"), DataLocation("depth_unit")]

    @classmethod
    async def _data_to_schema(cls, data: Any, context: IContext) -> Any:
        result = await super()._data_to_schema(data, context)
        if data.depth_unit is not None:
            result["unit"] = data.depth_unit
        return result
