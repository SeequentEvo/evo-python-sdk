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

import hashlib
import subprocess
import sys
import unittest
import uuid
from collections.abc import Iterator
from io import BytesIO
from pathlib import Path
from textwrap import dedent
from typing import BinaryIO
from unittest import mock
from uuid import UUID

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from parameterized import parameterized, parameterized_class

from evo.common import Environment
from evo.common.test_tools import BASE_URL
from evo.common.utils import Cache
from evo.objects.exceptions import SchemaValidationError, TableFormatError
from evo.objects.parquet import ParquetLoader
from evo.objects.utils import (
    ArrowTableFormat,
    BaseTableFormat,
    ColumnFormat,
    KnownTableFormat,
    all_known_formats,
    get_known_format,
)
from helpers import get_sample_table

SAMPLE_DATA_LENGTH = 10
ENVIRONMENT = Environment(hub_url=BASE_URL, org_id=UUID(int=0), workspace_id=UUID(int=0))
CACHE: Cache


def tearDownModule() -> None:
    """Fail-safe cleanup of the cache directory."""
    try:
        CACHE.clear_cache()
        CACHE.root.rmdir()
    except Exception:
        pass  # Already cleaned up


def setUpModule() -> None:
    cache_dir = Path(__file__).parent.resolve() / f".{__name__.lower()}_cache"
    global CACHE
    CACHE = Cache(cache_dir, mkdir=True)
    unittest.addModuleCleanup(tearDownModule)

    # Write gitignore file in case cleanup fails.
    gitignore_file = cache_dir / ".gitignore"
    gitignore_file.write_text("*\n", encoding="utf-8")


def _all_known_formats_for_testing() -> Iterator[dict]:
    for known_format in all_known_formats:
        yield {"data_format": known_format, "expected_field_names": known_format._field_names}


def _get_buffer_digest(buffer: BinaryIO) -> str:
    """Return a sha256 digest of a binary buffer"""
    buffer.seek(0)
    file_hash = hashlib.sha256()
    while next_block := buffer.read(4 * 1024 * 1024):
        file_hash.update(next_block)
    return file_hash.hexdigest()


def _get_table_digest(table: pa.Table) -> str:
    """return a sha256 digest of a pyarrow.Table saved in parquet format"""
    buffer = BytesIO()
    pq.write_table(table=table, where=buffer, version="2.4", compression="gzip")
    return _get_buffer_digest(buffer=buffer)


def _load_table(table_info: dict, source: Path) -> pa.Table:
    with pa.OSFile(str(source / str(table_info["data"])), mode="r") as parquet_file:
        with ParquetLoader(parquet_file) as loader:
            loader.validate_with_table_info(table_info)
            return loader.load_as_table()


def _test_name_from_known_format(cls: type, num: int, params_dict: dict) -> str:
    """Create parameterized test name from DataFormat"""
    data_format = params_dict["data_format"]
    return parameterized.to_safe_name(f"{cls.__name__}_{num}_{data_format.name}")


class TestOptionalPandas(unittest.TestCase):
    @parameterized.expand([("with_pandas", False), ("without_pandas", True)])
    def test_cold_start(self, _name: str, missing_pandas: bool) -> None:
        script = dedent(
            """
            import importlib.abc
            import sys
            from pathlib import Path
            from tempfile import TemporaryDirectory

            missing_pandas = sys.argv[1] == "True"
            if missing_pandas:
                assert not any(name == "pandas" or name.startswith("pandas.") for name in sys.modules)

                class NoPandasImports(importlib.abc.MetaPathFinder):
                    def find_spec(self, fullname, path=None, target=None):
                        if fullname == "pandas" or fullname.startswith("pandas."):
                            raise ModuleNotFoundError("No module named 'pandas'", name="pandas")

                sys.meta_path.insert(0, NoPandasImports())

            import numpy as np
            import pyarrow as pa

            from evo.objects import DownloadedObject, ObjectAPIClient, utils
            from evo.objects.parquet import ParquetLoader
            from evo.objects.utils import ArrowTableFormat, ObjectDataClient, get_known_format

            assert hasattr(ObjectAPIClient, "get_data_client")
            assert all(hasattr(utils, name) for name in utils.__all__)
            for name in ("expand_hole_index", "hole_chunks_from_ids"):
                assert hasattr(utils, name) == (not missing_pandas), name
                assert (name in utils.__all__) == (not missing_pandas), name

            arrow_methods = {
                DownloadedObject: ("download_table", "download_category_table", "download_array"),
                ObjectDataClient: ("save_table", "upload_table", "upload_category_table", "download_table"),
                ParquetLoader: ("load_as_table", "load_as_array"),
            }
            for owner, methods in arrow_methods.items():
                for name in methods:
                    assert hasattr(owner, name), f"{owner.__name__}.{name}"

            dataframe_methods = {
                DownloadedObject: (
                    "download_dataframe", "download_category_dataframe", "download_attribute_dataframe"
                ),
                ObjectDataClient: (
                    "save_dataframe", "upload_dataframe", "upload_category_dataframe", "download_dataframe"
                ),
                ParquetLoader: ("load_as_dataframe",),
            }
            for owner, methods in dataframe_methods.items():
                for name in methods:
                    assert hasattr(owner, name) == (not missing_pandas), f"{owner.__name__}.{name}"

            table = pa.table({"value": pa.array([1, 2, 3], type=pa.int64())})
            table_format = ArrowTableFormat.from_schema(table.schema)
            assert table_format.pa_schema.equals(table.schema)
            with TemporaryDirectory() as directory:
                source = Path(directory)
                table_info = get_known_format(table).save_table(table, source)
                with ParquetLoader(pa.OSFile(str(source / table_info["data"]), "r")) as loader:
                    loader.validate_with_table_info(table_info)
                    assert loader.load_as_table().equals(table)
                    np.testing.assert_array_equal(
                        loader.load_as_array(), np.array([1, 2, 3], dtype=np.int64), strict=True
                    )

            if missing_pandas:
                try:
                    table_format.pandas_types_mapper()
                except ModuleNotFoundError as error:
                    assert error.name == "pandas"
                else:
                    raise AssertionError("Pandas dtype mapping must report the missing dependency")
                assert not any(name == "pandas" or name.startswith("pandas.") for name in sys.modules)
            """
        )
        # Cached pandas-dependent modules can hide import failures when only selected modules are reloaded.
        result = subprocess.run(
            [sys.executable, "-c", script, str(missing_pandas)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(0, result.returncode, result.stdout + result.stderr)


class TestColumnFormat(unittest.TestCase):
    @parameterized.expand(
        [
            ("float64", pa.float64()),
            ("uint8", pa.uint8()),
            ("uint32", pa.uint32()),
            ("uint64", pa.uint64()),
            ("int32", pa.int32()),
            ("int64", pa.int64()),
            ("bool", pa.bool_()),
            ("string", pa.string()),
            ("timestamp", pa.timestamp("us", tz="UTC")),
        ]
    )
    def test_column_types(self, format_id: str, data_type: pa.DataType) -> None:
        for format_spec in (format_id, data_type):
            with self.subTest(format_spec=format_spec):
                column = ColumnFormat(format_spec)
                self.assertEqual(format_id, column.id)
                self.assertEqual(data_type, column.type)

    def test_unsupported_column_type(self) -> None:
        for format_spec in ("float32", pa.float32()):
            with self.subTest(format_spec=format_spec), self.assertRaises(TypeError):
                ColumnFormat(format_spec)

    @parameterized.expand(
        [
            ("float64", None),
            ("uint8", "UInt8"),
            ("uint32", "UInt32"),
            ("uint64", "UInt64"),
            ("int32", "Int32"),
            ("int64", "Int64"),
            ("bool", "boolean"),
            ("string", None),
            ("timestamp", "datetime64[us, UTC]"),
        ]
    )
    def test_pandas_dtype(self, format_id: str, expected_dtype: str | None) -> None:
        column = ColumnFormat(format_id)
        if expected_dtype is None:
            self.assertIsNone(column.pandas_dtype)
            return

        self.assertEqual(pd.api.types.pandas_dtype(expected_dtype), column.pandas_dtype)
        # The pandas dtype must convert back to the same Arrow type, so import and export agree.
        dataframe = pd.DataFrame({"value": pd.array([None], dtype=column.pandas_dtype)})
        self.assertEqual(column.type, ArrowTableFormat.from_pandas(dataframe).columns[0].type)


class TestBaseTableFormat(unittest.TestCase):
    def test_columns(self) -> None:
        table_format = BaseTableFormat("mixed", [pa.int32(), "string", pa.bool_()])

        self.assertIsInstance(table_format.columns, tuple)
        self.assertTrue(all(isinstance(column, ColumnFormat) for column in table_format.columns))
        self.assertEqual(["int32", "string", "bool"], [column.id for column in table_format.columns])
        self.assertEqual([pa.int32(), pa.string(), pa.bool_()], [column.type for column in table_format.columns])

    def test_multidimensional_columns(self) -> None:
        table_format = BaseTableFormat("multidimensional", [pa.float64(), ...])

        self.assertEqual(1, len(table_format.columns))
        self.assertEqual(pa.float64(), table_format.columns[0].type)


class TestArrowTableFormat(unittest.TestCase):
    def test_from_schema(self) -> None:
        schema = pa.schema(
            [
                pa.field("key", pa.int64(), nullable=False, metadata={b"field": b"kept"}),
                pa.field("label", pa.string()),
            ],
            metadata={b"other": b"kept"},
        )

        table_format = ArrowTableFormat.from_schema(schema)

        self.assertEqual(
            pa.schema([pa.field("key", pa.int64(), nullable=False), pa.field("label", pa.string())]),
            table_format.pa_schema,
        )
        self.assertIsNone(table_format.pa_schema.metadata)
        self.assertIsNone(table_format.pa_schema.field("key").metadata)
        self.assertEqual("ArrowTableFormat<key/label>", table_format.name)
        self.assertEqual("int64/string", table_format.data_type)
        self.assertEqual(2, table_format.width)
        self.assertEqual(schema.types, [column.type for column in table_format.columns])

    def test_pandas_types_mapper(self) -> None:
        schema = pa.schema([("count", pa.int64()), ("flag", pa.bool_()), ("value", pa.float64())])

        types_mapper = ArrowTableFormat.from_schema(schema).pandas_types_mapper()

        self.assertEqual(pd.Int64Dtype(), types_mapper(pa.int64()))
        self.assertEqual(pd.BooleanDtype(), types_mapper(pa.bool_()))
        self.assertIsNone(types_mapper(pa.float64()))
        self.assertIsNone(types_mapper(pa.int32()))

    @parameterized.expand(
        [
            ("Int32", [1, None], pa.int32()),
            ("Int64", [2**63 - 1, None], pa.int64()),
            ("UInt8", [255, None], pa.uint8()),
            ("UInt32", [2**32 - 1, None], pa.uint32()),
            ("UInt64", [2**64 - 1, None], pa.uint64()),
            ("boolean", [True, None], pa.bool_()),
            ("float64", [1.5, None], pa.float64()),
            ("string[python]", ["A", None], pa.string()),
            ("string[pyarrow]", ["A", None], pa.string()),
            ("datetime64[us, UTC]", [pd.Timestamp("2025-01-01", tz="UTC"), None], pa.timestamp("us", tz="UTC")),
        ]
    )
    def test_from_pandas(self, dtype: str, values: list, expected_type: pa.DataType) -> None:
        dataframe = pd.DataFrame({"value": pd.array(values, dtype=dtype)})

        table_format = ArrowTableFormat.from_pandas(dataframe)
        table = pa.Table.from_pandas(dataframe, schema=table_format.pa_schema, preserve_index=False)

        self.assertEqual(pa.schema([("value", expected_type)]), table_format.pa_schema)
        self.assertIsNone(table_format.pa_schema.metadata)
        self.assertEqual(expected_type, table_format.columns[0].type)
        self.assertEqual(expected_type, table.column("value").type)
        self.assertEqual(1, table.column("value").null_count)
        self.assertEqual(values[0], table.column("value")[0].as_py())

    def test_from_pandas_excludes_index(self) -> None:
        indexes = [
            pd.RangeIndex(5, 7),
            pd.Index([5, 3], name="row"),
            pd.MultiIndex.from_tuples([("A", 5), ("B", 3)], names=["hole", "row"]),
        ]
        for index in indexes:
            with self.subTest(index=index):
                dataframe = pd.DataFrame({"value": [1.0, 2.0]}, index=index)

                table_format = ArrowTableFormat.from_pandas(dataframe)
                table = pa.Table.from_pandas(dataframe, schema=table_format.pa_schema, preserve_index=False)

                self.assertEqual(["value"], table_format.pa_schema.names)
                self.assertIsNone(table_format.pa_schema.metadata)
                self.assertEqual({"value": [1.0, 2.0]}, table.to_pydict())

    def test_from_pandas_rejects_unsupported_column_type(self) -> None:
        dataframe = pd.DataFrame({"value": pd.array([1.0, 2.0], dtype="float32")})

        with self.assertRaisesRegex(TypeError, "Unsupported column type"):
            ArrowTableFormat.from_pandas(dataframe)

    def test_factory_preserves_subclass(self) -> None:
        class CustomTableFormat(ArrowTableFormat):
            pass

        table_format = CustomTableFormat.from_pandas(pd.DataFrame({"value": [1.0]}))

        self.assertIsInstance(table_format, CustomTableFormat)
        self.assertEqual("CustomTableFormat<value>", table_format.name)

    def test_direct_construction_remains_supported(self) -> None:
        table_format = ArrowTableFormat("custom", [pa.float64()])

        self.assertEqual("custom", table_format.name)
        self.assertEqual("float64", table_format.data_type)
        self.assertEqual(1, table_format.width)
        with self.assertRaisesRegex(ValueError, "from_schema.*from_pandas"):
            _ = table_format.pa_schema


@parameterized_class(_all_known_formats_for_testing(), class_name_func=_test_name_from_known_format)
class TestKnownFormat(unittest.TestCase):
    data_format: KnownTableFormat
    expected_field_names: str

    def setUp(self) -> None:
        self.sample_table = get_sample_table(table_format=self.data_format, n_rows=SAMPLE_DATA_LENGTH)
        self.expected_parquet_digest = _get_table_digest(self.sample_table)
        self.data_dir = CACHE.get_location(ENVIRONMENT, self.__class__.__name__)
        self.parquet_file = self.data_dir / self.expected_parquet_digest

    def tearDown(self) -> None:
        self.parquet_file.unlink(missing_ok=True)

    def test_unknown_format(self) -> None:
        all_other_formats = [other for other in all_known_formats if not other.is_provided_by(self.data_format)]
        with (
            mock.patch("evo.objects.utils.table_formats.all_known_formats", new=all_other_formats),
            self.assertRaises(TableFormatError),
        ):
            known_format = get_known_format(self.sample_table)
            raise AssertionError(f"Got unexpected format: {known_format.name}")

    def test_get_known_format(self) -> None:
        actual_format = get_known_format(self.sample_table)
        actual_field_names = actual_format._field_names

        # Check that the actual format is compatible with the expected format.
        self.data_format._check_format(actual_format)

        self.assertEqual(self.data_format.name, actual_format.name)
        self.assertEqual(self.data_format.width, actual_format.width)
        self.assertEqual(self.data_format.data_type, actual_format.data_type)
        self.assertEqual(self.expected_field_names, actual_field_names)

    def test_save_table(self) -> None:
        self.assertFalse(self.parquet_file.is_file())

        known_format = get_known_format(self.sample_table)
        table_info = known_format.save_table(self.sample_table, self.data_dir)

        self.assertEqual(frozenset(table_info), self.expected_field_names)
        self.assertEqual(self.expected_parquet_digest, table_info["data"])
        self.assertEqual(SAMPLE_DATA_LENGTH, table_info["length"])

        self.assertTrue(self.parquet_file.is_file())

    def test_save_table_when_file_exists_and_is_open(self) -> None:
        self.assertFalse(self.parquet_file.is_file())

        known_format = get_known_format(self.sample_table)
        known_format.save_table(self.sample_table, self.data_dir)
        self.assertTrue(self.parquet_file.is_file())

        with self.parquet_file.open():
            table_info = known_format.save_table(self.sample_table, self.data_dir)

        self.assertEqual(frozenset(table_info), self.expected_field_names)
        self.assertEqual(self.expected_parquet_digest, table_info["data"])
        self.assertEqual(SAMPLE_DATA_LENGTH, table_info["length"])

        self.assertTrue(self.parquet_file.is_file())

    def _save_parquet_file(self, add_column: bool = False, add_row: bool = False, change_type: bool = False) -> dict:
        if add_column or add_row or change_type:
            sample_length = SAMPLE_DATA_LENGTH
            if add_row:
                sample_length += 1
            self.sample_table = get_sample_table(
                table_format=self.data_format, n_rows=sample_length, add_column=add_column, change_types=change_type
            )

        pq.write_table(self.sample_table, where=self.parquet_file, version="2.4", compression="gzip")
        info_dict = {
            "data": self.expected_parquet_digest,
            "length": SAMPLE_DATA_LENGTH,
            "width": self.sample_table.num_columns,
            "data_type": self.data_format.data_type,
            "keys_data_type": self.data_format._columns[0]._format_id,
            "field_names": self.expected_field_names,
        }
        if add_column:
            info_dict["width"] -= 1
        return info_dict

    def test_known_format_from_table_info(self) -> None:
        table_info = self._save_parquet_file()
        self.assertTrue(self.parquet_file.is_file())

        inferred_format = KnownTableFormat.from_table_info(table_info)
        self.assertTrue(self.data_format.is_provided_by(inferred_format))

    def test_load_table(self) -> None:
        table_info = self._save_parquet_file()
        self.assertTrue(self.parquet_file.is_file())
        inferred_format = KnownTableFormat.from_table_info(table_info)

        actual_table = _load_table(table_info, self.data_dir)
        self.assertEqual(inferred_format.width, actual_table.num_columns)
        self.assertEqual(table_info["length"], actual_table.num_rows)

        actual_format = ArrowTableFormat.from_schema(actual_table.schema)
        self.assertEqual(inferred_format.data_type, actual_format.data_type)

    def test_load_table_with_too_many_columns(self) -> None:
        table_info = self._save_parquet_file(add_column=True)
        self.assertTrue(self.parquet_file.is_file())

        with self.assertRaises(SchemaValidationError):
            _load_table(table_info, self.data_dir)

    def test_load_table_with_too_many_rows(self) -> None:
        table_info = self._save_parquet_file(add_row=True)
        self.assertTrue(self.parquet_file.is_file())

        with self.assertRaises(SchemaValidationError):
            _load_table(table_info, self.data_dir)

    def test_load_table_with_wrong_data_types(self) -> None:
        table_info = self._save_parquet_file(change_type=True)
        self.assertTrue(self.parquet_file.is_file())

        with self.assertRaises(SchemaValidationError):
            _load_table(table_info, self.data_dir)

    def test_load_table_with_uuid_identifier(self) -> None:
        table_info = self._save_parquet_file()
        self.assertTrue(self.parquet_file.is_file())

        table_info["data"] = str(uuid.uuid4())
        self.parquet_file = self.parquet_file.rename(self.parquet_file.parent / table_info["data"])
        self.assertTrue(self.parquet_file.is_file())

        inferred_format = KnownTableFormat.from_table_info(table_info)

        actual_table = _load_table(table_info, self.data_dir)
        self.assertEqual(inferred_format.width, actual_table.num_columns)
        self.assertEqual(table_info["length"], actual_table.num_rows)

        actual_format = ArrowTableFormat.from_schema(actual_table.schema)
        self.assertEqual(inferred_format.data_type, actual_format.data_type)


def _complex_formats_for_testing() -> Iterator[dict]:
    """Generator of table formats with their generated GO models and keyword arguments for testing.

    :yields: (<table-format>, <expected-field-names>)
    """
    yield {
        "data_format": BaseTableFormat("downhole-collection-location-path", [pa.float64()] * 3),
        "expected_field_names": ["data", "length", "width", "data_type"],
        "expect_extra_column_fails": False,
    }
    yield {
        "data_format": BaseTableFormat("hexahedrons-indices", [pa.uint64()] * 8),
        "expected_field_names": ["data", "length", "width", "data_type"],
        "expect_extra_column_fails": True,
    }
    yield {
        "data_format": BaseTableFormat("hexahedrons-vertices", [pa.float64()] * 3),
        "expected_field_names": ["data", "length", "width", "data_type"],
        "expect_extra_column_fails": False,
    }
    yield {
        "data_format": BaseTableFormat("quadrilaterals-indices", [pa.uint64()] * 4),
        "expected_field_names": ["data", "length", "width", "data_type"],
        "expect_extra_column_fails": True,
    }
    yield {
        "data_format": BaseTableFormat("quadrilaterals-vertices", [pa.float64()] * 3),
        "expected_field_names": ["data", "length", "width", "data_type"],
        "expect_extra_column_fails": False,
    }
    yield {
        "data_format": BaseTableFormat("segments-indices", [pa.uint64()] * 2),
        "expected_field_names": ["data", "length", "width", "data_type"],
        "expect_extra_column_fails": False,
    }
    yield {
        "data_format": BaseTableFormat("segments-vertices", [pa.float64()] * 3),
        "expected_field_names": ["data", "length", "width", "data_type"],
        "expect_extra_column_fails": False,
    }
    yield {
        "data_format": BaseTableFormat("tetrahedra-indices", [pa.uint64()] * 4),
        "expected_field_names": ["data", "length", "width", "data_type"],
        "expect_extra_column_fails": True,
    }
    yield {
        "data_format": BaseTableFormat("tetrahedra-vertices", [pa.float64()] * 3),
        "expected_field_names": ["data", "length", "width", "data_type"],
        "expect_extra_column_fails": False,
    }
    yield {
        "data_format": BaseTableFormat("triangles-indices", [pa.uint64()] * 3),
        "expected_field_names": ["data", "length", "width", "data_type"],
        "expect_extra_column_fails": False,
    }
    yield {
        "data_format": BaseTableFormat("triangles-vertices", [pa.float64()] * 3),
        "expected_field_names": ["data", "length", "width", "data_type"],
        "expect_extra_column_fails": False,
    }
    yield {
        "data_format": BaseTableFormat("unstructured-grid-geometry-vertices", [pa.float64()] * 3),
        "expected_field_names": ["data", "length", "width", "data_type"],
        "expect_extra_column_fails": False,
    }


@parameterized_class(_complex_formats_for_testing(), class_name_func=_test_name_from_known_format)
class TestComplexFormats(unittest.TestCase):
    data_format: BaseTableFormat
    expected_field_names: str
    expect_extra_column_fails: bool

    def setUp(self) -> None:
        self.sample_table = get_sample_table(table_format=self.data_format, n_rows=SAMPLE_DATA_LENGTH)
        self.expected_parquet_digest = _get_table_digest(self.sample_table)
        self.data_dir = CACHE.get_location(ENVIRONMENT, self.__class__.__name__)
        self.parquet_file = self.data_dir / self.expected_parquet_digest

    def tearDown(self) -> None:
        self.parquet_file.unlink(missing_ok=True)

    def test_save_table(self) -> None:
        self.assertFalse(self.parquet_file.is_file())

        known_format = get_known_format(self.sample_table)
        table_info = known_format.save_table(self.sample_table, self.data_dir)

        self.assertEqual(list(table_info), self.expected_field_names)
        self.assertEqual(self.expected_parquet_digest, table_info["data"])
        self.assertEqual(SAMPLE_DATA_LENGTH, table_info["length"])

        self.assertTrue(self.parquet_file.is_file())

    def test_save_table_extra_column_fails(self) -> None:
        self.assertFalse(self.parquet_file.is_file())
        sample_table = get_sample_table(self.data_format, n_rows=SAMPLE_DATA_LENGTH, add_column=True)

        if self.expect_extra_column_fails:
            with self.assertRaises(TableFormatError):
                _known_format = get_known_format(sample_table)
        else:
            known_format = get_known_format(sample_table)
            known_format.save_table(sample_table, self.data_dir)

        self.assertFalse(self.parquet_file.is_file())

    def test_save_table_different_column_types_fails(self) -> None:
        self.assertFalse(self.parquet_file.is_file())
        sample_table = get_sample_table(self.data_format, n_rows=SAMPLE_DATA_LENGTH, change_types=True)

        known_format = get_known_format(sample_table)
        with self.assertRaises(TableFormatError):
            known_format.save_table(self.sample_table, self.data_dir)

        self.assertFalse(self.parquet_file.is_file())
