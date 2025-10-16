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

import pyarrow as pa

from evo import logging

from ..exceptions import TableFormatError
from .tables import ArrowTableFormat, KnownTableFormat

__all__ = [
    "all_known_formats",
    "get_known_format",
]

logger = logging.getLogger("object.table_formats")

_float_array_formats: list[KnownTableFormat] = [
    KnownTableFormat(
        name="float-array-1", columns=[pa.float64()], field_names=["data", "length", "width", "data_type"]
    ),
    KnownTableFormat(
        name="float-array-2", columns=[pa.float64()] * 2, field_names=["data", "length", "width", "data_type"]
    ),
    KnownTableFormat(
        name="float-array-3", columns=[pa.float64()] * 3, field_names=["data", "length", "width", "data_type"]
    ),
    KnownTableFormat(
        name="float-array-6", columns=[pa.float64()] * 6, field_names=["data", "length", "width", "data_type"]
    ),
    # float-array-md MUST be last so that formats with fixed sizes are matched first.
    KnownTableFormat(
        name="float-array-md", columns=[pa.float64(), ...], field_names=["data", "length", "width", "data_type"]
    ),
]


_index_array_formats: list[KnownTableFormat] = [
    KnownTableFormat(name="index-array-1", columns=[pa.uint64()], field_names=["data", "length", "width", "data_type"]),
    KnownTableFormat(
        name="index-array-2", columns=[pa.uint64()] * 2, field_names=["data", "length", "width", "data_type"]
    ),
    KnownTableFormat(
        name="index-array-3", columns=[pa.uint64()] * 3, field_names=["data", "length", "width", "data_type"]
    ),
    KnownTableFormat(
        name="index-array-4", columns=[pa.uint64()] * 4, field_names=["data", "length", "width", "data_type"]
    ),
    KnownTableFormat(
        name="index-array-8", columns=[pa.uint64()] * 8, field_names=["data", "length", "width", "data_type"]
    ),
]


_integer_array_formats: list[KnownTableFormat] = [
    KnownTableFormat(
        name="integer-array-1-int32", columns=[pa.int32()], field_names=["data", "length", "width", "data_type"]
    ),
    KnownTableFormat(
        name="integer-array-1-int64", columns=[pa.int64()], field_names=["data", "length", "width", "data_type"]
    ),
    # integer-array-md-* MUST be last so that formats with fixed sizes are matched first.
    KnownTableFormat(
        name="integer-array-md-int32", columns=[pa.int32(), ...], field_names=["data", "length", "width", "data_type"]
    ),
    KnownTableFormat(
        name="integer-array-md-int64", columns=[pa.int64(), ...], field_names=["data", "length", "width", "data_type"]
    ),
]


_boolean_array_formats: list[KnownTableFormat] = [
    KnownTableFormat(name="bool-array-1", columns=[pa.bool_()], field_names=["data", "length", "data_type"]),
    # bool-array-md MUST be last so that formats with fixed sizes are matched first.
    KnownTableFormat(
        name="bool-array-md", columns=[pa.bool_(), ...], field_names=["data", "length", "width", "data_type"]
    ),
]


_other_array_formats: list[KnownTableFormat] = [
    # Simple arrays.
    KnownTableFormat(
        name="brep-container-brep", columns=[pa.uint8()], field_names=["data", "length", "width", "data_type"]
    ),
    KnownTableFormat(name="color-array", columns=[pa.uint32()], field_names=["data", "length", "data_type"]),
    KnownTableFormat(
        name="date-time-array", columns=[pa.timestamp("us", tz="UTC")], field_names=["data", "length", "data_type"]
    ),
    KnownTableFormat(name="string-array", columns=[pa.string()], field_names=["data", "length", "data_type"]),
    # Complex arrays.
    KnownTableFormat(
        name="downhole-collection-location-holes",
        columns=[pa.int32(), pa.uint64(), pa.uint64()],
        field_names=["data", "length", "width", "data_type"],
    ),
    KnownTableFormat(
        name="unstructured-grid-geometry-cells",
        columns=[pa.int32(), pa.uint64(), pa.int32()],
        field_names=["data", "length", "width", "data_type"],
    ),
    KnownTableFormat(
        name="lines-2d-indices",
        columns=[pa.uint64(), pa.uint64(), pa.float64()],
        field_names=["data", "length", "width", "data_type"],
    ),
    KnownTableFormat(
        name="lookup-table-int32",
        columns=[pa.int32(), pa.string()],
        field_names=["data", "length", "keys_data_type", "values_data_type"],
    ),
    KnownTableFormat(
        name="lookup-table-int64",
        columns=[pa.int64(), pa.string()],
        field_names=["data", "length", "keys_data_type", "values_data_type"],
    ),
]


all_known_formats: list[KnownTableFormat] = [
    *_float_array_formats,
    *_index_array_formats,
    *_integer_array_formats,
    *_boolean_array_formats,
    *_other_array_formats,
]


def get_known_format(table: pa.Table) -> KnownTableFormat:
    """Get the known table format that best matches the provided table.

    If both a multidimensional format and a format with fixed dimensions would match, the format with fixed dimensions
    will be returned.

    :param table: The actual table to match.

    :return: The known format that best matches the provided table.

    :raises TableFormatError: If the provided table does not match a known format.
    """
    actual_format = ArrowTableFormat.from_schema(table.schema)
    logger.debug(f"Matching format for {actual_format.name}")
    for known_format in all_known_formats:
        if known_format.is_provided_by(actual_format):
            logger.debug(f"{known_format.name} is provided by {actual_format.name}")
            return known_format

    msg = f"Could not resolve Geoscience Object Schema for {actual_format.name}"
    logger.error(msg)
    raise TableFormatError(msg)
