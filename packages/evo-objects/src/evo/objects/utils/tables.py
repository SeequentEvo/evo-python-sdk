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
import os
import time
from collections.abc import Callable
from pathlib import Path
from tempfile import NamedTemporaryFile
from types import EllipsisType

import pyarrow as pa
import pyarrow.parquet as pq
from typing_extensions import deprecated

import evo.logging
from evo.common.exceptions import StorageFileNotFoundError

from ..exceptions import SchemaValidationError, TableFormatError

logger = evo.logging.getLogger("object.tables")

__all__ = [
    "ArrowTableFormat",
    "BaseTableFormat",
    "KnownTableFormat",
]

# Block size to limit memory use while calculating the sha256 digest of a file.
_DIGEST_BLOCK_SIZE = 4 * 1024 * 1024

# Bounded backoff in seconds for file operations that raise PermissionError.
_FILE_ACCESS_RETRY_DELAYS = (0.01, 0.05, 0.1, 0.25, 0.5)


def _retry_file_access(operation: Callable[[], object], path: Path) -> None:
    """Retry denied file access with bounded backoff; propagate persistent failures and other errors."""
    for delay in (*_FILE_ACCESS_RETRY_DELAYS, None):
        try:
            operation()
            return
        except PermissionError:
            if delay is None:
                raise
            logger.debug(f"Access to file '{path.name}' denied, retrying in {delay} seconds")
            time.sleep(delay)


def _rename_without_replacing(source: Path, target: Path) -> None:
    """Rename a file on Windows, where `os.rename` raises FileExistsError instead of replacing an existing target."""
    _retry_file_access(lambda: os.rename(source, target), source)


def _link_without_replacing(source: Path, target: Path) -> None:
    """Publish with a POSIX hard link; propagate failures rather than risk replacing an existing target."""
    _retry_file_access(lambda: os.link(source, target), source)


def _discard_temporary_file(path: Path) -> None:
    """Remove a temporary file, logging instead of raising if it cannot be removed, because it is only left over."""
    try:
        _retry_file_access(lambda: path.unlink(missing_ok=True), path)
    except OSError:
        logger.warning(f"Failed to remove temporary file '{path.name}'", exc_info=True)


def _publish_without_replacing(source: Path, target: Path) -> bool:
    """Atomically move a completed file to a content-addressed path, unless that path already exists.

    Content-addressed files are immutable, so an existing target is never replaced. This is safe across threads and
    processes, and never disturbs a reader that has the existing target open.

    :param source: The completed temporary file, which is removed on a best-effort basis after publication is attempted.
    :param target: The content-addressed path to publish the file to.

    :return: True if the source was published to the target, or False if the target already existed.

    :raises OSError: If atomic no-overwrite publication fails, including when POSIX hard links are unsupported.
    """
    publish = _rename_without_replacing if os.name == "nt" else _link_without_replacing
    try:
        publish(source, target)
    except FileExistsError:
        return False
    finally:
        _discard_temporary_file(source)
    return True


class _ColumnFormat:
    def __init__(self, format_spec: pa.DataType | str):
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
        return self._format_id

    @property
    def type(self) -> pa.DataType:
        return self._type


class BaseTableFormat:
    """Base type for comparing table formats"""

    def __init__(self, name: str, columns: list[pa.DataType | str | EllipsisType]) -> None:
        """
        :param name: The display name for this format.
        :param columns: A list of column data types in this format. A single column data type followed by Ellipsis
            ("...") indicates a multidimensional format.
        """
        self._name = name
        self._columns: list[_ColumnFormat] = []
        self._multi_dimensional = False
        for column_type in columns:
            if column_type is Ellipsis:
                self._multi_dimensional = True
            elif self._multi_dimensional:
                raise AssertionError("Found new type specification after Ellipsis")
            else:
                self._columns.append(_ColumnFormat(format_spec=column_type))

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
    def data_type(self) -> str:
        """The column data type(s) in this table format"""
        return self._data_type


class ArrowTableFormat(BaseTableFormat):
    """Specialised table format type that can be generated from a pyarrow table"""

    @classmethod
    def from_schema(cls, pa_schema: pa.Schema) -> ArrowTableFormat:
        """Generate an ArrowTableFormat instance that represents the structure of the provided table schema.

        :param pa_schema: Table schema to generate a format representation for.

        :return: A table format that can be used to compare against known formats.
        """
        return ArrowTableFormat(name=f"{cls.__name__}<{'/'.join(pa_schema.names)}>", columns=pa_schema.types)


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

        The file is named after its sha256 digest, so identical tables share one immutable file. Concurrent saves of
        identical tables, from any thread or process, never replace a file that another save has already published.

        :param table: The table to save to parquet file.
        :param destination: A local directory to save the parquet file in.

        :return: The sha256 digest of the saved parquet file, which is also the filename.

        :raises StorageFileNotFoundError: If the destination does not exist or is not a directory.
        :raises OSError: If publication fails. On POSIX, the destination file system must support hard links.
        """
        if not destination.is_dir():
            raise StorageFileNotFoundError(f"'{destination}' is not a directory")

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
        logger.debug(f"Publishing {tmp_path.name} as {new_path.name}")
        try:
            published = _publish_without_replacing(tmp_path, new_path)
        except BaseException:
            logger.error(f"Failed to publish file '{tmp_path.name}' due to an unhandled exception", exc_info=True)
            raise
        if not published:
            logger.debug(f"File '{new_path.name}' already exists")

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
            # The parquet file is not removed, because identical tables saved concurrently share it.
            logger.error(error, exc_info=True)
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
                return parquet_data.read()

        except FileNotFoundError:
            raise StorageFileNotFoundError(f"Could not find data for table with id '{parquet_file.name}'") from None

        except (TableFormatError, SchemaValidationError) as error:
            logger.error(f"Could not load table because {error}")
            raise
