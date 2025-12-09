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
from typing import Sequence, TypeVar

from ..utils.tables import KnownTableFormat

__all__ = [
    "AttributesAdapter",
    "BaseAdapter",
    "CategoryTableAdapter",
    "DatasetAdapter",
    "TableAdapter",
]


@dataclass
class BaseAdapter:
    min_major_version: int
    max_major_version: int


@dataclass
class TableAdapter(BaseAdapter):
    column_names: tuple[str, ...]
    values_path: str
    table_formats: list[KnownTableFormat]
    nan_values_path: str | None = None


@dataclass
class CategoryTableAdapter(BaseAdapter):
    column_names: tuple[str, ...]
    category_data_path: str
    nan_values_path: str | None = None


@dataclass
class AttributesAdapter(BaseAdapter):
    attribute_list_path: str


class DatasetAdapter:
    """Adapter to extract dataset-related information from a Geoscience Object JSON document using JMESPath expressions."""

    def __init__(
        self,
        value_adapters: Sequence[TableAdapter | CategoryTableAdapter],
        attributes_adapter: AttributesAdapter | None,
    ) -> None:
        """
        :param value_adapters: A sequence of ValuesAdapter instances to extract values information.
        :param attributes_adapter: An AttributesAdapter instance to extract attributes information.
        """
        self.value_adapters = tuple(value_adapters)
        self.attributes_adapter = attributes_adapter

    @classmethod
    def from_adapter_lists(
        cls,
        major_version: int,
        value_adapters: list[TableAdapter | CategoryTableAdapter],
        attributes_adapters: list[AttributesAdapter],
    ) -> DatasetAdapter:
        attributes_adapters = _get_selected_adapters(major_version, attributes_adapters)
        if len(attributes_adapters) > 1:
            raise ValueError(f"Multiple AttributesAdapters found for schema version {major_version}")
        value_adapters = _get_selected_adapters(major_version, value_adapters)
        return DatasetAdapter(
            value_adapters,
            attributes_adapters[0] if attributes_adapters else None,
        )


_T = TypeVar("_T", bound=BaseAdapter)


def _get_selected_adapters(major_version: int, candidate_adapters: list[_T]) -> list[_T]:
    selected_adapters = []
    for attribute_adapter in candidate_adapters:
        if attribute_adapter.min_major_version <= major_version <= attribute_adapter.max_major_version:
            selected_adapters.append(attribute_adapter)
    return selected_adapters
