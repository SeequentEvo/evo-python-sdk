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

import errno
import hashlib
import os
import threading
import unittest
import uuid
from collections.abc import Callable, Iterator
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from io import BytesIO
from multiprocessing import get_context
from pathlib import Path
from typing import BinaryIO
from unittest import mock
from uuid import UUID

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
    KnownTableFormat,
    all_known_formats,
    get_known_format,
)
from evo.objects.utils.tables import (
    _FILE_ACCESS_RETRY_DELAYS,
    _discard_temporary_file,
    _link_without_replacing,
    _publish_without_replacing,
    _rename_without_replacing,
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


_CONCURRENT_WORKERS = 8
_SAVES_PER_WORKER = 10
_CONCURRENT_PROCESSES = 4
_CONCURRENT_TABLE_ROWS = 20_000


def _get_float_table(n_rows: int) -> pa.Table:
    return pa.table({name: pa.array(range(n_rows), type=pa.float64()) for name in "xyz"})


def _save_and_read_in_process(destination: Path, n_rows: int, n_saves: int) -> set[str]:
    table = _get_float_table(n_rows)
    known_format = get_known_format(table)
    names = set()
    for _ in range(n_saves):
        name = known_format.save_table(table, destination)["data"]
        with (destination / name).open("rb") as file:
            assert _get_buffer_digest(file) == name, f"Published file '{name}' has unexpected content"
        names.add(name)
    return names


class TestConcurrentSaveTable(unittest.TestCase):
    """Identical tables share one immutable content-addressed file, which concurrent saves must not disturb."""

    def setUp(self) -> None:
        self.scope = f"{self.__class__.__name__}-{uuid.uuid4()}"
        self.data_dir = CACHE.get_location(ENVIRONMENT, self.scope)
        self.table = _get_float_table(_CONCURRENT_TABLE_ROWS)
        self.known_format = get_known_format(self.table)
        self.expected_digest = _get_table_digest(self.table)
        self.parquet_file = self.data_dir / self.expected_digest

    def tearDown(self) -> None:
        CACHE.clear_cache(ENVIRONMENT, self.scope)

    def assert_only_parquet_file_remains(self) -> None:
        self.assertEqual([self.expected_digest], [path.name for path in self.data_dir.iterdir()])
        with self.parquet_file.open("rb") as file:
            self.assertEqual(self.expected_digest, _get_buffer_digest(file))

    def test_concurrent_identical_saves_and_reads(self) -> None:
        barrier = threading.Barrier(_CONCURRENT_WORKERS)

        def save_and_read() -> set[str]:
            barrier.wait()
            names = set()
            for _ in range(_SAVES_PER_WORKER):
                name = self.known_format.save_table(self.table, self.data_dir)["data"]
                with (self.data_dir / name).open("rb") as file:  # Like an upload of the published file.
                    self.assertEqual(name, _get_buffer_digest(file))
                names.add(name)
            return names

        with ThreadPoolExecutor(_CONCURRENT_WORKERS) as executor:
            futures = [executor.submit(save_and_read) for _ in range(_CONCURRENT_WORKERS)]
            names = set().union(*(future.result() for future in futures))

        self.assertEqual({self.expected_digest}, names)
        self.assert_only_parquet_file_remains()

    def test_concurrent_identical_saves_across_processes(self) -> None:
        with ProcessPoolExecutor(max_workers=_CONCURRENT_PROCESSES, mp_context=get_context("spawn")) as executor:
            futures = [
                executor.submit(_save_and_read_in_process, self.data_dir, _CONCURRENT_TABLE_ROWS, _SAVES_PER_WORKER)
                for _ in range(_CONCURRENT_PROCESSES)
            ]
            for future in futures:
                self.assertEqual({self.expected_digest}, future.result(timeout=120))
        self.assert_only_parquet_file_remains()

    def test_existing_file_open_by_reader_is_not_replaced(self) -> None:
        self.known_format.save_table(self.table, self.data_dir)
        original = self.parquet_file.stat()

        with self.parquet_file.open("rb") as reader:
            self.assertEqual(self.expected_digest, self.known_format.save_table(self.table, self.data_dir)["data"])
            self.assertEqual(self.expected_digest, _get_buffer_digest(reader))

        current = self.parquet_file.stat()
        self.assertEqual((original.st_ino, original.st_mtime_ns), (current.st_ino, current.st_mtime_ns))
        self.assert_only_parquet_file_remains()

    def test_table_info_failure_keeps_shared_file(self) -> None:
        error = RuntimeError("Unexpected failure")
        field_names = mock.MagicMock()
        field_names.__contains__.side_effect = error  # Fails after the parquet file is published.
        with (
            mock.patch.object(self.known_format, "_field_names", field_names),
            self.assertRaises(RuntimeError) as raised,
        ):
            self.known_format.save_table(self.table, self.data_dir)

        self.assertIs(error, raised.exception)
        self.assert_only_parquet_file_remains()


class TestPublishWithoutReplacing(unittest.TestCase):
    def setUp(self) -> None:
        self.scope = f"{self.__class__.__name__}-{uuid.uuid4()}"
        self.data_dir = CACHE.get_location(ENVIRONMENT, self.scope)
        self.source = self.data_dir / "source"
        self.target = self.data_dir / "target"
        self.source.write_bytes(b"new")

    def tearDown(self) -> None:
        CACHE.clear_cache(ENVIRONMENT, self.scope)

    def test_publishes_new_file(self) -> None:
        self.assertTrue(_publish_without_replacing(self.source, self.target))
        self.assertFalse(self.source.exists())
        self.assertEqual(b"new", self.target.read_bytes())

    def test_keeps_existing_file(self) -> None:
        self.target.write_bytes(b"existing")
        self.assertFalse(_publish_without_replacing(self.source, self.target))
        self.assertFalse(self.source.exists())
        self.assertEqual(b"existing", self.target.read_bytes())

    def test_failure_removes_source(self) -> None:
        error = OSError(errno.ENOSPC, "No space left on device")
        with (
            mock.patch("evo.objects.utils.tables._rename_without_replacing", side_effect=error),
            mock.patch("evo.objects.utils.tables._link_without_replacing", side_effect=error),
            self.assertRaises(OSError) as raised,
        ):
            _publish_without_replacing(self.source, self.target)

        self.assertIs(error, raised.exception)
        self.assertFalse(self.source.exists())
        self.assertFalse(self.target.exists())

    def test_link_does_not_replace_existing_file(self) -> None:
        self.target.write_bytes(b"existing")
        with self.assertRaises(FileExistsError):
            _link_without_replacing(self.source, self.target)
        self.assertEqual(b"existing", self.target.read_bytes())

    @parameterized.expand(
        [
            ("unsupported", errno.EOPNOTSUPP),
            ("permission_denied", errno.EACCES),
            ("operation_not_permitted", errno.EPERM),
            ("cross_device", errno.EXDEV),
            ("disk_full", errno.ENOSPC),
        ]
    )
    def test_failed_link_preserves_existing_target(self, _name: str, error_number: int) -> None:
        self.target.write_bytes(b"existing")
        original = self.target.stat()
        error = OSError(error_number, "Hard-link publication failed")

        with (
            mock.patch("evo.objects.utils.tables.os", wraps=os) as platform,
            mock.patch("evo.objects.utils.tables.time.sleep") as sleep,
        ):
            platform.name = "posix"
            platform.link.side_effect = error
            with self.assertRaises(OSError) as raised:
                _publish_without_replacing(self.source, self.target)
            platform.replace.assert_not_called()

        delays = _FILE_ACCESS_RETRY_DELAYS if isinstance(error, PermissionError) else ()
        self.assertEqual([mock.call(delay) for delay in delays], sleep.call_args_list)
        self.assertEqual(len(delays) + 1, platform.link.call_count)
        self.assertIs(error, raised.exception)
        self.assertFalse(self.source.exists())
        self.assertEqual(b"existing", self.target.read_bytes())
        current = self.target.stat()
        self.assertEqual((original.st_ino, original.st_mtime_ns), (current.st_ino, current.st_mtime_ns))

    def test_unsupported_link_does_not_publish_target(self) -> None:
        error = OSError(errno.EOPNOTSUPP, "Hard links are not supported")
        with mock.patch("evo.objects.utils.tables.os", wraps=os) as platform:
            platform.name = "posix"
            platform.link.side_effect = error
            with self.assertRaises(OSError) as raised:
                _publish_without_replacing(self.source, self.target)
            platform.replace.assert_not_called()

        self.assertIs(error, raised.exception)
        self.assertFalse(self.source.exists())
        self.assertFalse(self.target.exists())

    def test_failed_link_preserves_concurrently_published_target(self) -> None:
        error = OSError(errno.EOPNOTSUPP, "Hard links are not supported")

        def fail_link(source: Path, target: Path) -> None:
            target.write_bytes(b"another writer")
            raise error

        with mock.patch("evo.objects.utils.tables.os", wraps=os) as platform:
            platform.name = "posix"
            platform.link.side_effect = fail_link
            with self.assertRaises(OSError) as raised:
                _publish_without_replacing(self.source, self.target)
            platform.replace.assert_not_called()

        self.assertIs(error, raised.exception)
        self.assertFalse(self.source.exists())
        self.assertEqual(b"another writer", self.target.read_bytes())

    @unittest.skipUnless(os.name == "nt", "os.rename only refuses to replace an existing file on Windows")
    def test_rename_does_not_replace_file_open_by_reader(self) -> None:
        self.target.write_bytes(b"existing")
        with self.target.open("rb") as reader:
            with self.assertRaises(FileExistsError):
                _rename_without_replacing(self.source, self.target)
            self.assertEqual(b"existing", reader.read())
        self.assertEqual(b"new", self.source.read_bytes())

    @parameterized.expand([("rename", _rename_without_replacing), ("link", _link_without_replacing)])
    def test_publication_retries_permission_errors(self, operation: str, publish: Callable[[Path, Path], None]) -> None:
        with (
            mock.patch(
                f"evo.objects.utils.tables.os.{operation}", side_effect=[PermissionError, PermissionError, None]
            ) as action,
            mock.patch("time.sleep") as sleep,
        ):
            publish(self.source, self.target)

        self.assertEqual(3, action.call_count)
        self.assertEqual([mock.call(delay) for delay in _FILE_ACCESS_RETRY_DELAYS[:2]], sleep.call_args_list)

    @parameterized.expand([("rename", _rename_without_replacing), ("link", _link_without_replacing)])
    def test_publication_raises_when_access_stays_denied(
        self, operation: str, publish: Callable[[Path, Path], None]
    ) -> None:
        error = PermissionError("Access denied")
        with (
            mock.patch(f"evo.objects.utils.tables.os.{operation}", side_effect=error) as action,
            mock.patch("time.sleep"),
            self.assertRaises(PermissionError) as raised,
        ):
            publish(self.source, self.target)
        self.assertIs(error, raised.exception)
        self.assertEqual(len(_FILE_ACCESS_RETRY_DELAYS) + 1, action.call_count)

    @parameterized.expand([("rename", _rename_without_replacing), ("link", _link_without_replacing)])
    def test_publication_does_not_retry_existing_file(
        self, operation: str, publish: Callable[[Path, Path], None]
    ) -> None:
        with (
            mock.patch(f"evo.objects.utils.tables.os.{operation}", side_effect=FileExistsError) as action,
            mock.patch("time.sleep") as sleep,
            self.assertRaises(FileExistsError),
        ):
            publish(self.source, self.target)
        action.assert_called_once()
        sleep.assert_not_called()

    def test_discard_retries_locked_file(self) -> None:
        unlink = Path.unlink
        attempts = []

        def unlink_once_locked(path: Path, missing_ok: bool = False) -> None:
            attempts.append(path)
            if len(attempts) == 1:
                raise PermissionError
            unlink(path, missing_ok=missing_ok)

        with (
            mock.patch.object(Path, "unlink", autospec=True, side_effect=unlink_once_locked),
            mock.patch("time.sleep") as sleep,
        ):
            _discard_temporary_file(self.source)

        self.assertEqual([self.source, self.source], attempts)
        sleep.assert_called_once_with(_FILE_ACCESS_RETRY_DELAYS[0])
        self.assertFalse(self.source.exists())

    def test_file_that_stays_locked_does_not_fail_publication(self) -> None:
        self.target.write_bytes(b"existing")
        with (
            mock.patch.object(Path, "unlink", side_effect=PermissionError),
            mock.patch("time.sleep"),
            mock.patch("evo.objects.utils.tables.logger") as logger,
        ):
            self.assertFalse(_publish_without_replacing(self.source, self.target))

        logger.warning.assert_called_once()
        self.assertEqual(b"existing", self.target.read_bytes())
