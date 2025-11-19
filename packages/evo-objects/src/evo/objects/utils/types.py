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

import sys
from typing import Generic, TypeAlias, TypeVar

if sys.version_info >= (3, 12):
    from typing import NotRequired, TypedDict
else:
    from typing_extensions import NotRequired, TypedDict

__all__ = ["ArrayTableInfo", "AttributeInfo", "CategoryInfo", "LookupTableInfo", "TableInfo"]


class _BaseTableInfo(TypedDict):
    data: str
    length: int


class ArrayTableInfo(_BaseTableInfo):
    """Metadata for a non-lookup table.

    The 'data' field contains the reference to the blob where the table data is stored.

    The 'length', 'width', and 'data_type' fields describe the structure of the table.
    """

    data_type: str
    width: NotRequired[int]


class LookupTableInfo(_BaseTableInfo):
    """Metadata for lookup table, which is used to define categories.

    The 'data' field contains the reference to the blob where the table data is stored.

    The 'length', 'width', and 'data_type' fields describe the structure of the table.
    """

    keys_data_type: str
    values_data_type: str


TableInfo: TypeAlias = ArrayTableInfo | LookupTableInfo


class CategoryInfo(TypedDict):
    """Metadata for category tables.

    In Geoscience Object Schemas, categories are defined by an indices array(values) and a lookup table (table).
    """

    table: LookupTableInfo
    values: TableInfo


T = TypeVar("T")


class _Nan(TypedDict, Generic[T]):
    values: list[T]


class NanCategorical(_Nan[int]):
    """Metadata for representing 'not a number' (NaN) values in categorical/integer attributes.

    In addition to supporting null values within certain tables, additional 'not a number' (NaN) values can be defined,
    which should be interpreted as 'not a number' (NaN).
    """


class NanContinuous(_Nan[float]):
    """Metadata for representing 'not a number' (NaN) values in continuous attributes.

    In addition to supporting null values within certain tables, additional 'not a number' (NaN) values can be defined,
    which should be interpreted as 'not a number' (NaN).
    """


class AttributeInfo(TypedDict):
    """Metadata for attributes."""

    name: str
    key: NotRequired[str]
    nan_description: NotRequired[NanCategorical | NanContinuous]
    values: ArrayTableInfo
    table: NotRequired[LookupTableInfo]
