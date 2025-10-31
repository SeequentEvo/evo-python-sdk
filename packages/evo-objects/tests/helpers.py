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
from typing import Any

import jmespath
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
    table = get_sample_table(table_format, n_rows)
    return table, write_table_to_bytes(table)


def write_table_to_bytes(table: pa.Table) -> bytes:
    memory = BytesIO()
    pq.write_table(table, where=memory, version="2.4", compression="gzip")
    return memory.getvalue()


# Support for assignment operations using JMESPath expressions.
# Could be moved to evo.jmespath in the future, if we want to expose this functionality outside of tests.
class _AssignmentTargetDictEntry:
    """Represents a dictionary entry that potentially can be assigned to."""

    def __init__(self, key: str, obj: dict):
        self.key = key
        self.obj = obj

    @property
    def value(self) -> Any:
        """Get the value at this dictionary entry, creating an empty dict if it doesn't exist."""
        return self.obj.setdefault(self.key, {})


class _AssignmentTargetListEntry:
    """Represents a list entry that potentially can be assigned to."""

    def __init__(self, index: int, obj: list):
        self.index = index
        self.obj = obj

    @property
    def value(self) -> Any:
        """Get the value at this list entry, or None if the index is out of range."""
        try:
            return self.obj[self.index]
        except IndexError:
            return None


class _AssignInterpreter(jmespath.visitor.Visitor):
    """A JMESPath visitor used for processing assignment operations.

    This only supports a subset of JMESPath expressions that can be used for assignment.

    This works by lazily evaluating field and index accesses, so that the last operation can be turned into an
    assignment. If another operation is encountered after a field or index access, the value is evaluated at that
    point.
    """

    def default_visit(self, node, *args, **kwargs):
        raise NotImplementedError(node["type"])

    @staticmethod
    def _evaluate_value(value):
        """Lazily evaluate the value if it's an assignment target."""
        if isinstance(value, (_AssignmentTargetDictEntry, _AssignmentTargetListEntry)):
            return value.value
        else:
            return value

    def visit_field(self, node, value):
        """Visit a field access node, i.e. foo.bar."""
        evaluated_value = self._evaluate_value(value)
        if not isinstance(evaluated_value, dict):
            return None
        return _AssignmentTargetDictEntry(node["value"], evaluated_value)

    def visit_index(self, node, value):
        """Visit an index access node, i.e. foo[0]."""
        evaluated_value = self._evaluate_value(value)
        if not isinstance(evaluated_value, list):
            return None
        return _AssignmentTargetListEntry(node["value"], evaluated_value)

    def _visit_sub_or_index_expression(self, node, value):
        """Visit a subexpression or index expression node, i.e. foo.bar.baz or a[0][1]."""
        result = value
        for node in node["children"]:
            result = self.visit(node, result)
        return result

    visit_subexpression = _visit_sub_or_index_expression
    visit_index_expression = _visit_sub_or_index_expression


def assign_property(obj: dict, expression: str, value: Any) -> None:
    """Assign a value to a property in a dictionary using a JMESPath expression.

     This only supports a subset of JMESPath expressions that can be used for assignment. In particular, only the following
    expression types are supported:
    - Field accesses (e.g. foo.bar)
    - Index accesses (e.g. foo[0])
    - Subexpressions combining the above (e.g. foo.bar[0].baz)
    If the expression is not in that form, a JMESPathError will be raised.

    Also, if the expression attempts to perform an invalid operation like:
    - Accessing a field on a non-object
    - Accessing an index on a non-array
    - Accessing an out-of-bounds index on an array
    then a JMESPathError will be raised.

    Accessing a non-existent field on an object will create an empty object at that field to allow for nested assignments.

    :param obj: The dictionary to assign the property to.
    :param expression: The JMESPath expression representing the property to assign to.
    :param value: The value to assign to the property.
    """
    parsed_expression = jmespath.compile(expression)
    interpreter = _AssignInterpreter()
    target = interpreter.visit(parsed_expression.parsed, obj)

    if isinstance(target, _AssignmentTargetDictEntry):
        target.obj[target.key] = value
    elif isinstance(target, _AssignmentTargetListEntry):
        target.obj[target.index] = value
    else:
        raise TypeError(f"Cannot assign to expression '{expression}'")
