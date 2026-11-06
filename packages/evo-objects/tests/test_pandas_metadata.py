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

"""DataFrame exports omit pandas metadata; dataframe readers derive dtypes from the Geoscience Object schema."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pandas.testing import assert_frame_equal

from evo.objects.exceptions import SchemaValidationError, TableFormatError
from evo.objects.parquet import ParquetLoader
from evo.objects.utils import ArrowTableFormat, KnownTableFormat, get_known_format
from evo.objects.utils.data import _table_from_dataframe


def _load_dataframe(table_info: dict, source: Path) -> pd.DataFrame:
    with pa.OSFile(str(source / str(table_info["data"])), mode="r") as parquet_file:
        with ParquetLoader(parquet_file) as loader:
            loader.validate_with_table_info(table_info)
            return loader.load_as_dataframe()


class TestPandasMetadata(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.data_dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _save(self, table: pa.Table) -> dict:
        return get_known_format(table).save_table(table, self.data_dir)

    def _schema_metadata(self, table_info: dict) -> dict:
        return pq.read_schema(self.data_dir / table_info["data"]).metadata or {}

    def test_dataframe_table_has_pandas_metadata(self) -> None:
        """Even an explicit metadata-free schema does not prevent PyArrow from adding pandas metadata."""
        dataframe = pd.DataFrame({"value": pd.array([1, None], dtype="Int64")})
        table_format = ArrowTableFormat.from_pandas(dataframe)
        self.assertIsNone(table_format.pa_schema.metadata)

        table = pa.Table.from_pandas(dataframe, schema=table_format.pa_schema, preserve_index=False)

        self.assertIn(b"pandas", table.schema.metadata)

    def test_saved_dataframe_has_no_pandas_metadata(self) -> None:
        table = _table_from_dataframe(pd.DataFrame({"value": pd.array([1, None, 3], dtype="Int64")}))

        table_info = self._save(table)

        self.assertNotIn(b"pandas", self._schema_metadata(table_info))

    def test_dataframe_index_is_not_stored(self) -> None:
        dataframe = pd.DataFrame({"value": [1.0, 2.0, 3.0]}).iloc[[2, 0]]

        table = _table_from_dataframe(dataframe)

        self.assertEqual(["value"], table.column_names)
        self.assertIsNone(table.schema.metadata)

    def test_load_as_dataframe_uses_nullable_dtypes_from_schema(self) -> None:
        cases = [
            (pa.array([1, None, 3], type=pa.int64()), "Int64"),
            (pa.array([1, 2, 3], type=pa.int64()), "Int64"),
            (pa.array([1, None, 3], type=pa.int32()), "Int32"),
            (pa.array([1, 2, 3], type=pa.int32()), "Int32"),
            (pa.array([2**63 - 1, None, -(2**63)], type=pa.int64()), "Int64"),
            (pa.array([255, None, 0], type=pa.uint8()), "UInt8"),
            (pa.array([2**32 - 1, None, 0], type=pa.uint32()), "UInt32"),
            (pa.array([2**64 - 1, None, 0], type=pa.uint64()), "UInt64"),
            (pa.array([None, None], type=pa.int64()), "Int64"),
            (pa.array([], type=pa.int64()), "Int64"),
            (pa.array([True, None, False], type=pa.bool_()), "boolean"),
            (pa.array([1.5, None, 3.5], type=pa.float64()), "float64"),
            (
                pa.array([pd.Timestamp("2025-01-01", tz="UTC"), None], type=pa.timestamp("us", tz="UTC")),
                "datetime64[us, UTC]",
            ),
        ]
        for array, expected_dtype in cases:
            with self.subTest(type=str(array.type), nulls=array.null_count):
                table_info = self._save(pa.table({"value": array}))

                loaded = _load_dataframe(table_info, self.data_dir)

                self.assertEqual(pd.api.types.pandas_dtype(expected_dtype), loaded["value"].dtype)
                assert_frame_equal(pd.DataFrame({"value": pd.array(array.to_pylist(), dtype=expected_dtype)}), loaded)

    def test_load_ignores_legacy_pandas_metadata(self) -> None:
        """Files written before metadata stripping must load with the schema types, not the stored pandas dtypes."""
        dataframe = pd.DataFrame({"value": pd.array([1, 2, 3], dtype="Int64")}, index=pd.RangeIndex(10, 13))
        table = pa.Table.from_pandas(dataframe)
        table_info = {"data": "legacy.parquet", "length": 3, "width": 1, "data_type": "int64"}
        pq.write_table(table, self.data_dir / table_info["data"])
        self.assertIn(b"pandas", pq.read_schema(self.data_dir / table_info["data"]).metadata)

        loaded = _load_dataframe(table_info, self.data_dir)

        pd.testing.assert_index_equal(pd.RangeIndex(3), loaded.index)
        self.assertEqual(pd.Int64Dtype(), loaded["value"].dtype)

    def test_load_ignores_malformed_pandas_metadata_without_changing_arrow_table(self) -> None:
        schema = pa.schema(
            [pa.field("value", pa.int64(), nullable=False, metadata={b"units": b"count"})],
            metadata={b"pandas": b"not json", b"other": b"kept"},
        )
        table = pa.Table.from_arrays([pa.array([2**63 - 1])], schema=schema)
        table_info = self._save(table)

        with ParquetLoader(pa.OSFile(str(self.data_dir / table_info["data"]), "r")) as loader:
            loader.validate_with_table_info(table_info)
            assert_frame_equal(
                pd.DataFrame({"value": pd.array([2**63 - 1], dtype="Int64")}), loader.load_as_dataframe()
            )
            self.assertTrue(table.equals(loader.load_as_table(), check_metadata=True))

    def test_serialized_index_is_rejected_by_expected_schema(self) -> None:
        dataframe = pd.DataFrame({"value": [1, 2]}, index=pd.Index([5, 3], name="row"))
        table = pa.Table.from_pandas(dataframe, preserve_index=True)
        table_info = {"data": "index.parquet", "length": 2, "width": 1, "data_type": "int64"}
        table_format = KnownTableFormat.from_table_info(table_info)

        with self.assertRaisesRegex(TableFormatError, "Column count"):
            table_format.save_table(table, self.data_dir)
        self.assertEqual([], list(self.data_dir.iterdir()))

        pq.write_table(table, self.data_dir / table_info["data"])
        with ParquetLoader(pa.OSFile(str(self.data_dir / table_info["data"]), "r")) as loader:
            with self.assertRaisesRegex(SchemaValidationError, "Column count"):
                loader.validate_with_table_info(table_info)

    def test_string_and_float_columns_use_arrow_defaults_not_pandas_metadata(self) -> None:
        for values, dtype in [(["A", None, "B"], "string[pyarrow]"), ([1.5, None, 3.5], "Float64")]:
            with self.subTest(dtype=dtype):
                table = pa.Table.from_pandas(pd.DataFrame({"value": pd.array(values, dtype=dtype)}))
                if pa.types.is_large_string(table["value"].type):
                    table = table.cast(pa.schema([("value", pa.string())], metadata=table.schema.metadata))
                table_info = self._save(table)

                loaded = _load_dataframe(table_info, self.data_dir)

                assert_frame_equal(table.replace_schema_metadata().to_pandas(), loaded)
