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

"""Parquet files must not carry pandas reconstruction metadata; pandas dtypes come from the Geoscience Object schema."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pandas.testing import assert_frame_equal

from evo.objects.exceptions import TableFormatError
from evo.objects.parquet import ParquetLoader
from evo.objects.utils import get_known_format
from evo.objects.utils.data import _table_from_dataframe


def _load_dataframe(table_info: dict, source: Path) -> pd.DataFrame:
    with pa.OSFile(str(source / str(table_info["data"])), mode="r") as parquet_file:
        with ParquetLoader(parquet_file) as loader:
            loader.validate_with_table_info(table_info)
            return loader.load_as_dataframe()


class _BrokenPandasExtensionArrays:
    """Simulate a consumer whose pandas cannot rebuild extension arrays from Arrow.

    pandas only calls `__from_arrow__` when told to by the pandas schema metadata, or by an explicit `types_mapper`.
    """

    def __enter__(self):
        def broken(*_args, **_kwargs):
            raise ModuleNotFoundError("No module named 'pandas.core.arrays.arrow._arrow_utils'")

        self._patches = [
            mock.patch.object(dtype, "__from_arrow__", broken)
            for dtype in (pd.Int32Dtype, pd.Int64Dtype, pd.BooleanDtype, pd.StringDtype)
        ]
        for patch in self._patches:
            patch.start()
        return self

    def __exit__(self, *exc_info):
        for patch in reversed(self._patches):
            patch.stop()


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
        """Guard the premise of these tests: pyarrow always adds pandas metadata when converting a dataframe."""
        table = _table_from_dataframe(pd.DataFrame({"value": pd.array([1, None], dtype="Int64")}))
        self.assertIn(b"pandas", table.schema.metadata)

    def test_save_table_strips_pandas_metadata(self) -> None:
        table = _table_from_dataframe(pd.DataFrame({"value": pd.array([1, None, 3], dtype="Int64")}))
        table = table.replace_schema_metadata({**table.schema.metadata, b"other": b"kept"})

        table_info = self._save(table)

        metadata = self._schema_metadata(table_info)
        self.assertNotIn(b"pandas", metadata)
        self.assertEqual(b"kept", metadata[b"other"])

    def test_saved_table_can_be_read_by_pandas_without_extension_array_support(self) -> None:
        dataframe = pd.DataFrame({"value": pd.array([1, None, 3], dtype="Int64")})
        table_info = self._save(_table_from_dataframe(dataframe))

        with _BrokenPandasExtensionArrays():
            loaded = pq.read_table(self.data_dir / table_info["data"]).to_pandas()

        # Without schema information pandas falls back to float64, but no value is corrupted.
        self.assertEqual([1.0, 3.0], loaded["value"].dropna().tolist())
        self.assertTrue(pd.isna(loaded["value"][1]))

    def test_legacy_pandas_metadata_breaks_pandas_without_extension_array_support(self) -> None:
        """Document the failure mode that stripping the metadata prevents."""
        dataframe = pd.DataFrame({"value": pd.array([1, None, 3], dtype="Int64")})
        legacy_file = self.data_dir / "legacy.parquet"
        pq.write_table(pa.Table.from_pandas(dataframe, preserve_index=False), legacy_file)

        with _BrokenPandasExtensionArrays(), self.assertRaises(ModuleNotFoundError):
            pq.read_table(legacy_file).to_pandas()

    def test_save_table_rejects_pandas_index_columns(self) -> None:
        dataframe = pd.DataFrame({"value": [1.0, 2.0]}, index=pd.Index([5, 3], name="row"))
        table = pa.Table.from_pandas(dataframe, preserve_index=True).select(["value"])
        # Keep the metadata pointing to an index column, which is what a table saved with its index looks like.
        index_metadata = json.loads(pa.Table.from_pandas(dataframe, preserve_index=True).schema.metadata[b"pandas"])
        table = table.replace_schema_metadata({b"pandas": json.dumps(index_metadata).encode()})

        with self.assertRaises(TableFormatError):
            self._save(table)
        self.assertEqual([], list(self.data_dir.iterdir()))

    def test_dataframe_index_is_not_stored(self) -> None:
        dataframe = pd.DataFrame({"value": [1.0, 2.0, 3.0]}).iloc[[2, 0]]

        table = _table_from_dataframe(dataframe)

        self.assertEqual(["value"], table.column_names)
        self.assertEqual([], json.loads(table.schema.metadata[b"pandas"])["index_columns"])

    def test_load_as_dataframe_uses_nullable_dtypes_from_schema(self) -> None:
        cases = [
            (pa.array([1, None, 3], type=pa.int64()), "Int64"),
            (pa.array([1, 2, 3], type=pa.int64()), "Int64"),
            (pa.array([1, None, 3], type=pa.int32()), "Int32"),
            (pa.array([1, 2, 3], type=pa.int32()), "Int32"),
            (pa.array([True, None, False], type=pa.bool_()), "boolean"),
            (pa.array([1.5, None, 3.5], type=pa.float64()), "float64"),
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
        with pa.OSFile(str(self.data_dir / table_info["data"]), mode="r") as parquet_file:
            with ParquetLoader(parquet_file) as loader:
                self.assertIsNone(loader.load_as_table().schema.metadata)
