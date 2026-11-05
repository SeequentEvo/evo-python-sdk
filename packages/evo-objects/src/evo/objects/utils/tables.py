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

import hashlib
import json
import os
from collections.abc import Callable
from pathlib import Path
from tempfile import NamedTemporaryFile
from types import EllipsisType
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from typing_extensions import deprecated

import evo.logging
from evo.common.exceptions import StorageFileNotFoundError

from ..exceptions import SchemaValidationError, TableFormatError

try:
    import pandas as pd
except ImportError:
    _PD_AVAILABLE = False
else:
    _PD_AVAILABLE = True

logger = evo.logging.getLogger("object.tables")

__all__ = [
    "ArrowTableFormat",
    "BaseTableFormat",
    "ColumnFormat",
    "KnownTableFormat",
]

# Block size to limit memory use while calculating the sha256 digest of a file.
_DIGEST_BLOCK_SIZE = 4 * 1024 * 1024

# Arrow schema metadata key that pyarrow uses to reconstruct pandas dtypes and indexes.
_PANDAS_METADATA_KEY = b"pandas"

# Pandas dtypes used for schema column types that numpy cannot represent with nulls.
_NULLABLE_PANDAS_DTYPES = {
    "int32": "Int32",
    "int64": "Int64",
    "uint8": "UInt8",
    "uint32": "UInt32",
    "uint64": "UInt64",
    "bool": "boolean",
}


def _get_schema_from_dataframe(dataframe: pd.DataFrame) -> pa.Schema:
    """Infer an Arrow schema without the dataframe index or pandas metadata, normalizing large strings.

    Dictionary encoding is preserved for category uploads, including the ordering of categorical values.
    """
    schema = pa.Schema.from_pandas(dataframe, preserve_index=False)
    fields = []
    for field in schema:
        if pa.types.is_large_string(field.type):
            fields.append(field.with_type(pa.string()))
        elif pa.types.is_dictionary(field.type) and pa.types.is_large_string(field.type.value_type):
            fields.append(
                field.with_type(pa.dictionary(field.type.index_type, pa.string(), ordered=field.type.ordered))
            )
        else:
            fields.append(field)
    return pa.schema(fields)


def _strip_pandas_metadata(table: pa.Table, *, validate: bool = False) -> pa.Table:
    """Remove pandas reconstruction metadata, so that readers rely only on the Arrow and Geoscience Object schemas.

    The pandas metadata is only a hint for rebuilding pandas dtypes and indexes; the Arrow data is not changed.
    Other schema metadata is preserved.

    :param table: The table to remove pandas metadata from.
    :param validate: Reject tables whose pandas metadata declares serialized index columns or cannot be validated.
        Once the metadata is removed, index columns would be indistinguishable from data columns.

    :return: The table without pandas metadata.

    :raises TableFormatError: If `validate` is true and the table contains serialized pandas index columns or malformed
        pandas index metadata.
    """
    metadata = table.schema.metadata
    if not metadata or _PANDAS_METADATA_KEY not in metadata:
        return table

    if validate:
        try:
            pandas_metadata = json.loads(metadata[_PANDAS_METADATA_KEY])
            if not isinstance(pandas_metadata, dict):
                raise ValueError("Expected a metadata object")
            index_columns = pandas_metadata.get("index_columns", [])
            if not isinstance(index_columns, list) or any(
                not (isinstance(column, str) or isinstance(column, dict) and column.get("kind") == "range")
                for column in index_columns
            ):
                raise ValueError("Expected index column names or RangeIndex descriptors")
        except ValueError as error:
            raise TableFormatError(
                "Cannot validate pandas index metadata. Recreate the table from the dataframe with "
                "`preserve_index=False` before saving."
            ) from error
        stored_index_columns = [column for column in index_columns if isinstance(column, str)]
        if stored_index_columns:
            raise TableFormatError(
                f"Table contains pandas index column(s) {stored_index_columns}, which are not part of any "
                "Geoscience Object table format. Convert the dataframe with `preserve_index=False`."
            )

    remaining = {key: value for key, value in metadata.items() if key != _PANDAS_METADATA_KEY}
    return table.replace_schema_metadata(remaining or None)


def _pandas_types_mapper(table_format: BaseTableFormat | None = None) -> Callable[[pa.DataType], Any]:
    """Get a pyarrow `types_mapper` that maps Geoscience Object column types to pandas dtypes.

    Integer and boolean columns use pandas nullable dtypes, so they keep their type when they contain nulls,
    rather than being converted to float or object columns depending on the data. Other types use pyarrow defaults.

    :param table_format: The expected table format. If omitted, all supported column types are mapped.
    """
    column_formats = (
        table_format.columns
        if table_format is not None
        else [ColumnFormat(format_id) for format_id in _NULLABLE_PANDAS_DTYPES]
    )
    mapping = {
        column_format.type: dtype
        for column_format in column_formats
        if (dtype := column_format.pandas_dtype) is not None
    }
    return mapping.get


def _table_to_dataframe(table: pa.Table, table_format: BaseTableFormat | None = None) -> pd.DataFrame:
    """Convert a table to a pandas dataframe using the Geoscience Object column types, ignoring pandas metadata.

    :param table: The table to convert. It is expected to have been validated against `table_format`.
    :param table_format: The expected table format. If omitted, all supported column types are mapped.
    """
    return _strip_pandas_metadata(table).to_pandas(types_mapper=_pandas_types_mapper(table_format))


class ColumnFormat:
    """A Geoscience Object column type and its corresponding Arrow data type."""

    def __init__(self, format_spec: pa.DataType | str) -> None:
        """
        :param format_spec: A Geoscience Object column type identifier or a supported Arrow data type.
        :raises TypeError: If the column type is not supported.
        """
        if isinstance(format_spec, str):
            self._type = self._get_data_type(format_spec)
            self._format_id = format_spec
        else:
            self._type = format_spec
            self._format_id = self._get_format_id(format_spec)

    @staticmethod
    def _get_data_type(format_id: str) -> pa.DataType:
        match format_id:
            case "float64":
                return pa.float64()
            case "uint8":
                return pa.uint8()
            case "uint32":
                return pa.uint32()
            case "uint64":
                return pa.uint64()
            case "int32":
                return pa.int32()
            case "int64":
                return pa.int64()
            case "bool":
                return pa.bool_()
            case "string":
                return pa.string()
            case "timestamp":
                return pa.timestamp("us", tz="UTC")
            case unknown_format:
                raise TypeError(f"Unsupported column type '{unknown_format}'")

    @staticmethod
    def _get_format_id(data_type: pa.DataType) -> str:
        match str(data_type):
            case "double":
                return "float64"
            case "uint8":
                return "uint8"
            case "uint32":
                return "uint32"
            case "uint64":
                return "uint64"
            case "int32":
                return "int32"
            case "int64":
                return "int64"
            case "bool":
                return "bool"
            case "string":
                return "string"
            case "timestamp[us, tz=UTC]":
                return "timestamp"
            case unknown_format:
                raise TypeError(f"Unsupported column type '{unknown_format}' from {data_type!r}")

    @property
    def id(self) -> str:
        """The Geoscience Object column type identifier."""
        return self._format_id

    @property
    def type(self) -> pa.DataType:
        """The corresponding Arrow data type."""
        return self._type

    @property
    def pandas_dtype(self) -> pd.api.extensions.ExtensionDtype | None:
        """The pandas dtype used when loading this column into a dataframe.

        Integer and boolean columns use pandas nullable dtypes, so nulls do not change the dtype. Returns `None` for
        column types that use the pyarrow default conversion.

        :raises ImportError: If pandas is not installed.
        """
        if (dtype_name := _NULLABLE_PANDAS_DTYPES.get(self._format_id)) is None:
            return None
        if not _PD_AVAILABLE:
            raise ModuleNotFoundError("No module named 'pandas'", name="pandas")
        return pd.api.types.pandas_dtype(dtype_name)


class BaseTableFormat:
    """Base type for comparing table formats"""

    def __init__(self, name: str, columns: list[pa.DataType | str | EllipsisType]) -> None:
        """
        :param name: The display name for this format.
        :param columns: A list of column data types in this format. A single column data type followed by Ellipsis
            ("...") indicates a multidimensional format.
        """
        self._name = name
        self._columns: list[ColumnFormat] = []
        self._multi_dimensional = False
        for column_type in columns:
            if column_type is Ellipsis:
                self._multi_dimensional = True
            elif self._multi_dimensional:
                raise AssertionError("Found new type specification after Ellipsis")
            else:
                self._columns.append(ColumnFormat(format_spec=column_type))

        if self._multi_dimensional and len(self._columns) != 1:
            raise AssertionError("Multidimensional formats must only have one column type")
        elif len(self._columns) < 1:
            raise AssertionError("Table formats must have at least one column")

        # Calculate data type string in constructor to mitigate cost of `_check_format()`.
        column_types = [column_format.id for column_format in self._columns]
        unique_types = set(column_types)
        if len(unique_types) == 1:
            self._data_type = unique_types.pop()
        else:
            self._data_type = "/".join(column_types)

    @property
    def name(self) -> str:
        """Format display name"""
        return self._name

    @property
    def width(self) -> int:
        """The number of columns in this table format"""
        return len(self._columns)

    @property
    def columns(self) -> tuple[ColumnFormat, ...]:
        """Column formats in order. A multidimensional format has one entry for its repeated column type."""
        return tuple(self._columns)

    @property
    def data_type(self) -> str:
        """The column data type(s) in this table format"""
        return self._data_type


class ArrowTableFormat(BaseTableFormat):
    """Specialised table format type that can be generated from a pyarrow table"""

    _pa_schema: pa.Schema | None = None

    @property
    def pa_schema(self) -> pa.Schema:
        """The Arrow schema supplied to `from_schema` or inferred by `from_pandas`.

        :raises ValueError: If the format was constructed directly without a schema.
        """
        if self._pa_schema is None:
            raise ValueError("Create the format with from_schema() or from_pandas() to access its Arrow schema")
        return self._pa_schema

    @classmethod
    def from_schema(cls, pa_schema: pa.Schema) -> ArrowTableFormat:
        """Generate an ArrowTableFormat instance that represents the structure of the provided table schema.

        :param pa_schema: Table schema to generate a format representation for.

        :return: A table format that can be used to compare against known formats.
        """
        table_format = cls(name=f"{cls.__name__}<{'/'.join(pa_schema.names)}>", columns=pa_schema.types)
        table_format._pa_schema = pa_schema
        return table_format

    @classmethod
    def from_pandas(cls, dataframe: pd.DataFrame) -> ArrowTableFormat:
        """Infer a table format from a pandas dataframe using the supported Geoscience Object column types.

        Large strings are normalized to strings. The inferred schema excludes the dataframe index and pandas metadata.
        PyArrow still adds pandas metadata when converting the dataframe; it is stripped when saving the Parquet file.

        :param dataframe: The dataframe to infer the format from.
        :return: A format whose `pa_schema` can be passed to `pyarrow.Table.from_pandas`.
        :raises TypeError: If a column type is not supported. Use `ObjectDataClient.upload_category_dataframe`
            for categorical columns, which are split into value and lookup tables separately.
        """
        return cls.from_schema(_get_schema_from_dataframe(dataframe))


class KnownTableFormat(BaseTableFormat):
    """A definition of a known table format that matches a Geoscience Object Schema model type"""

    def __init__(self, name: str, columns: list[pa.DataType | EllipsisType], field_names: list[str] | None) -> None:
        """
        :param name: The display name for this format.
        :param columns: A list of column data types in this format. A single column data type followed by Ellipsis
            ("...") indicates a multidimensional format.
        :param field_names: The field names that are included when saving a table in this format.
        """
        super().__init__(name, columns)
        if field_names is None:
            self._field_names = None
        else:
            self._field_names = frozenset(field_names)

    def _check_format(self, other: BaseTableFormat) -> None:
        """Check whether another table format satisfies this format.

        :param other: The other table format to compare against.

        :raises TableFormatError: If the other format does not satisfy this format.
        """
        if not self._multi_dimensional and self.width != other.width:
            raise TableFormatError(
                f"Column count ({other.width}) does not match expectation ({self.width}) for {self.name}"
            )

        expected_data_type = self.data_type
        actual_data_type = other.data_type
        if expected_data_type != actual_data_type:
            raise TableFormatError(
                f"{self.name} requires data_type='{expected_data_type}' but the provided table has '{actual_data_type}'"
            )

    def is_provided_by(self, other: BaseTableFormat) -> bool:
        """Test whether the other format meets the requirements of this format

        :param other: The other format to test. Usually generated from actual data.

        :return: True if the other format meets the requirements of this format.
        """
        try:
            self._check_format(other)
        except TableFormatError:
            return False
        else:
            return True

    @staticmethod
    def _get_file_digest(file_path: Path) -> str:
        """Get the sha256 digest of a file

        :param file_path: The path of the target file.

        :return: The sha256 digest of the target file.
        """
        with file_path.open("rb") as file:
            sha256_digest = hashlib.sha256()
            while block := file.read(_DIGEST_BLOCK_SIZE):
                sha256_digest.update(block)
        return sha256_digest.hexdigest()

    @classmethod
    def _save_table_as_parquet(cls, table: pa.Table, destination: Path) -> str:
        """Save a table in parquet format.

        Pandas reconstruction metadata is not saved, so the parquet file can be read by any consumer. Readers must use
        the Geoscience Object schema to interpret column types.

        :param table: The table to save to parquet file.
        :param destination: A local directory to save the parquet file in.

        :return: The sha256 digest of the saved parquet file, which is also the filename.

        :raises StorageFileNotFoundError: If the destination does not exist or is not a directory.
        :raises TableFormatError: If the table contains serialized pandas index columns or malformed pandas index metadata.
        """
        if not destination.is_dir():
            raise StorageFileNotFoundError(f"'{destination}' is not a directory")

        table = _strip_pandas_metadata(table, validate=True)

        try:  # Write temporary file.
            with NamedTemporaryFile(delete=False, dir=destination) as tmp_file:
                logger.debug(f"Writing temporary file {tmp_file.name}")
                pq.write_table(table=table, where=tmp_file, version="2.4", compression="gzip")
            tmp_path = Path(tmp_file.name).resolve()
            logger.debug(f"Calculating file digest for {tmp_file.name}")
            data_ref = cls._get_file_digest(tmp_path)
        except BaseException:
            logger.error(f"Removing temporary file '{tmp_file.name}' due to an unhandled exception", exc_info=True)
            os.unlink(tmp_file.name)
            raise

        new_path = destination / data_ref
        try:  # Rename temporary file using sha256 digest.
            logger.debug(f"Renaming {tmp_path.name} to {new_path.name}")
            tmp_path.replace(new_path)
        except PermissionError:  # When the same file is already opened somewhere else.
            os.unlink(tmp_file.name)
            if new_path.exists():
                logger.info(f"File '{new_path.name}' already exists")
            else:
                logger.error(f"Failed to rename file '{tmp_file.name}' due to an unhandled exception", exc_info=True)
                raise

        return data_ref

    def save_table(self, table: pa.Table, destination: Path) -> dict:
        """Save a pyarrow table in parquet format and return a GO model of the table metadata.

        :param table: The table to save in parquet format.
        :param destination: The directory to save the parquet file in.

        :return: A dictionary representing the Geoscience Object Schema that corresponds to this format.

        :raises TableFormatError: If the provided table does not match this format.
        :raises StorageFileNotFoundError: If the destination does not exist or is not a directory.
        """
        logger.debug(f"Saving table in folder {destination}")
        other_format = ArrowTableFormat.from_schema(table.schema)
        try:
            self._check_format(other_format)
            data_id = self._save_table_as_parquet(table, destination)
        except (TableFormatError, StorageFileNotFoundError) as error:
            logger.error(f"Could not save table because {error}")
            raise

        try:
            # The following fields are all those expected to be needed by table info types.
            data_dict = {
                "data": data_id,
                "length": table.num_rows,
                "width": table.num_columns,
                "data_type": self.data_type,
            }
            if len(self._columns) == 2:  # For lookup table special case.
                data_dict["keys_data_type"] = self._columns[0].id
                data_dict["values_data_type"] = self._columns[1].id

            if self._field_names is not None:
                data_dict = {field: value for field, value in data_dict.items() if field in self._field_names}

            return data_dict

        except Exception as error:
            logger.error(error, exc_info=True)
            parquet_file = destination / data_id
            try:
                parquet_file.unlink(missing_ok=True)
            except Exception:
                pass
            raise

    @classmethod
    def from_table_info(cls, table_info: dict) -> KnownTableFormat:
        """Generate a KnownTableFormat instance that represents the provided table info.

        :param table_info: Table info to generate a format representation for.

        :return: A table format that can be used to compare against other formats.
        """
        # "data" isn't that useful as a name, but it should always exist.
        type_name = table_info.get("schema") or table_info.get("data") or "Unknown"
        n_columns = table_info.get("width", 1)

        if "data_type" in table_info:
            data_type = table_info["data_type"]
        elif "keys_data_type" in table_info and "values_data_type" in table_info:
            data_type = "{keys_data_type}/{values_data_type}".format(**table_info)
            n_columns = 2
        else:
            raise TableFormatError(f"Unknown data type for '{type_name}' table")

        columns = data_type.split("/")
        if n_columns > 1 and len(columns) == 1:
            columns *= n_columns

        if n_columns != len(columns):
            raise TableFormatError(f"Unable to determine column types for '{type_name}' with data type '{data_type}'")

        return KnownTableFormat(name=type_name, columns=columns, field_names=table_info.get("field_names"))

    @classmethod
    @deprecated("KnownTableFormat.load_table is deprecated, use evo.objects.parquet.ParquetLoader instead")
    def load_table(cls, table_info: dict, source: Path) -> pa.Table:
        """Load parquet data as a pyarrow.Table and verify the format against the provided table info.

        The parquet metadata will be used to make sure the file contents matches the expected format before the table
        is read into memory.

        :param table_info: The table info that defines the expected format. The model's `data` field must be the
            filename of the parquet file, which must exist in `source`.
        :param source: The source directory to read the parquet file from.

        :return: A pyarrow table loaded directly from the parquet file.

        :raises StorageFileNotFoundError: If the parquet file does not exist or is not a file.
        :raises TableFormatError: If the parquet data does not match the expected format.
        :raises SchemaValidationError: If the parquet file has a different number of rows than expected.
        """
        parquet_file = source / str(table_info["data"])
        logger.debug(f"Reading parquet data from {parquet_file}")
        try:
            with pa.OSFile(str(parquet_file), mode="r") as parquet_file_reader:
                parquet_data = pq.ParquetFile(parquet_file_reader)

                logger.debug("Checking parquet data format")
                expected_format = KnownTableFormat.from_table_info(table_info)
                actual_format = ArrowTableFormat.from_schema(parquet_data.schema_arrow)
                cls._check_format(expected_format, actual_format)

                logger.debug("Checking parquet data length")
                expected_length = table_info["length"]
                actual_length = parquet_data.metadata.num_rows
                if expected_length != actual_length:
                    raise SchemaValidationError(
                        f"Row count ({actual_length}) does not match expectation ({expected_length})"
                    )

                logger.debug("Parquet metadata checks succeeded")
                return _strip_pandas_metadata(parquet_data.read())

        except FileNotFoundError:
            raise StorageFileNotFoundError(f"Could not find data for table with id '{parquet_file.name}'") from None

        except (TableFormatError, SchemaValidationError) as error:
            logger.error(f"Could not load table because {error}")
            raise
