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

import random
import sys
from collections.abc import Iterator
from datetime import datetime, timezone
from io import BytesIO

import numpy
import pyarrow as pa
import pyarrow.parquet as pq

from evo.objects.utils.tables import BaseTableFormat, _ColumnFormat


class NoImport:
    """Simple context manager to prevent one or more named modules from being imported."""

    def __init__(self, *names: str) -> None:
        """
        :param names: The names of the modules to prevent from being imported.
        """
        self._names = names
        self._unloaded_modules = {}

    def __enter__(self) -> None:
        for name in self._names:
            # If the module is already imported, save it and set to None.
            self._unloaded_modules[name] = sys.modules[name]
            # Set the module to None to prevent it from being re-imported.
            sys.modules[name] = None

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        # Restore the unloaded modules.
        for name, module in self._unloaded_modules.items():
            sys.modules[name] = module


class UnloadModule:
    """Simple context manager to unload one or more named modules on entry and restore on exit."""

    def __init__(self, *names: str) -> None:
        """
        :param names: The names of the modules to unload on entry and restore on exit.
        """
        self._names = names
        self._unloaded_modules = {}

    def _unload_module(self, name: str) -> None:
        if name in sys.modules:
            self._unloaded_modules[name] = sys.modules[name]
            del sys.modules[name]

        parent, *_ = name.rpartition(".")
        if parent:
            self._unload_module(parent)

    def __enter__(self) -> None:
        for name in self._names:
            self._unload_module(name)

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        # Restore the unloaded modules.
        for name, module in self._unloaded_modules.items():
            sys.modules[name] = module


def _generate_float64_data(n_samples: int) -> Iterator[float]:
    max_ = numpy.finfo("float64").max
    for _ in range(n_samples):
        yield max_ * random.uniform(-1.0, 1.0)


def _generate_int_data(int_type: str, n_samples: int) -> Iterator[int]:
    min_, max_ = numpy.iinfo(int_type).min, numpy.iinfo(int_type).max
    for _ in range(n_samples):
        yield random.randint(min_, max_)


def _generate_bool_data(n_samples: int) -> Iterator[bool]:
    for _ in range(n_samples):
        yield random.choice((True, False))


def _generate_string_data(n_samples: int) -> Iterator[str]:
    str_sample = "0123456789ABCDEF "
    for _ in range(n_samples):
        length = random.randint(10, 10000)
        yield "".join(random.choices(str_sample, k=length))


def _generate_timestamp_data(n_samples: int) -> Iterator[datetime]:
    min_ = datetime(1970, 1, 1, tzinfo=timezone.utc).timestamp()
    max_ = datetime(2038, 12, 31, 23, 59, 59, 999999, tzinfo=timezone.utc).timestamp()
    for _ in range(n_samples):
        yield datetime.utcfromtimestamp(random.uniform(min_, max_))


def _generate_data(format_id: str, n_samples: int) -> Iterator:
    match format_id:
        case "float64":
            yield from _generate_float64_data(n_samples)
        case "uint8" | "uint32" | "uint64" | "int32" | "int64" as int_type:
            yield from _generate_int_data(int_type, n_samples)
        case "bool":
            yield from _generate_bool_data(n_samples)
        case "string":
            yield from _generate_string_data(n_samples)
        case "timestamp":
            yield from _generate_timestamp_data(n_samples)
        case unknown_format:
            raise TypeError(f"Unsupported format '{unknown_format}'")


def _change_format(current_format: _ColumnFormat) -> _ColumnFormat:
    match current_format.id:
        case "float64":
            return _ColumnFormat("int64")
        case "uint8" | "uint32" | "uint64" | "int32" | "int64":
            return _ColumnFormat("float64")
        case "bool" | "timestamp":
            return _ColumnFormat("string")
        case "string":
            return _ColumnFormat("bool")
        case unknown_format:
            raise TypeError(f"Unsupported format '{unknown_format}'")


def get_sample_table(
    table_format: BaseTableFormat, n_rows: int, add_column: bool = False, change_types: bool = False
) -> pa.Table:
    column_formats = [column for column in table_format._columns]

    if add_column:
        column_formats.append(_ColumnFormat(column_formats[-1].type))

    if change_types:
        column_formats = [_change_format(column) for column in column_formats]

    if table_format._multi_dimensional:
        # Test multidimensional tables with an arbitrary number of columns. If the number of columns matches a more
        # specific GO type (one with a fixed number of columns), the more specific type would be instantiated.
        column_formats *= 20

    sample_schema = pa.schema(
        [pa.field(f"{column.id}[{i}]", column.type, nullable=False) for i, column in enumerate(column_formats)]
    )
    sample_data = [
        pa.array(_generate_data(column_format.id, n_rows), type=column_format.type, size=n_rows)
        for column_format in column_formats
    ]
    return pa.table(sample_data, names=sample_schema.names).cast(sample_schema)


def get_sample_table_and_bytes(table_format: BaseTableFormat, n_rows: int) -> tuple[pa.Table, bytes]:
    memory = BytesIO()
    table = get_sample_table(table_format, n_rows)
    pq.write_table(table, where=memory, version="2.4", compression="gzip")
    return table, memory.getvalue()
