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

import contextlib
import logging
import unittest
from io import BytesIO
from threading import Lock
from unittest import mock

from parameterized import parameterized

from evo.common.exceptions import RetryError
from evo.common.io import ChunkedIOManager, ChunkedIOTracker, ChunkMetadata
from evo.common.io.exceptions import ChunkedIOError
from evo.common.io.interfaces import IDestination, ISource
from evo.common.utils import BackoffLinear, Retry

logger = logging.getLogger(__name__)


class TestChunkedIOTracker(unittest.TestCase):
    FILE_SIZE = 15
    CHUNK_SIZE = 4
    LAST_CHUNK_ID = 3
    LAST_CHUNK_SIZE = 3

    def setUp(self) -> None:
        self.tracker = ChunkedIOTracker(self.FILE_SIZE, self.CHUNK_SIZE)

    def _get_offset(self, i: int) -> int:
        return i * self.CHUNK_SIZE

    def _get_chunk_size(self, i: int) -> int:
        return self.LAST_CHUNK_SIZE if i == self.LAST_CHUNK_ID else self.CHUNK_SIZE

    def test_iter_id(self) -> None:
        for i, chunk_metadata in enumerate(self.tracker):
            self.assertEqual(i, chunk_metadata.id)

    def test_iter_offset(self) -> None:
        for i, chunk_metadata in enumerate(self.tracker):
            expected_offset = self._get_offset(i)
            self.assertEqual(expected_offset, chunk_metadata.offset)

    def test_iter_size(self) -> None:
        for i, chunk_metadata in enumerate(self.tracker):
            expected_size = self._get_chunk_size(i)
            self.assertEqual(expected_size, chunk_metadata.size)

    def test_iter_defaults_to_incomplete(self) -> None:
        for chunk_metadata in self.tracker:
            self.assertFalse(chunk_metadata.completed)

    def _set_complete(self, ids: list[int]) -> None:
        for i in ids:
            meta = ChunkMetadata(id=i, offset=self._get_offset(i), size=self._get_chunk_size(i), completed=False)
            self.tracker.set_complete(meta)

    def test_set_complete(self) -> None:
        ids = [1, 3]
        self._set_complete(ids)
        for meta in self.tracker:
            if meta.id in ids:
                self.assertTrue(meta.completed)
            else:
                self.assertFalse(meta.completed)

    @parameterized.expand(
        [
            ("first chunk", [0], 0.25),
            ("last chunk", [3], 0.25),
            ("odd chunks", [1, 3], 0.5),
            ("even chunks", [0, 2], 0.5),
            ("all chunks", [0, 1, 2, 3], 1.0),
        ]
    )
    def test_get_progress(self, _name: str, ids: list[int], expected_progress: float) -> None:
        self._set_complete(ids)
        actual_progress = self.tracker.get_progress()
        self.assertEqual(expected_progress, actual_progress)


class _TestIOError(ChunkedIOError):
    def __init__(self, message: str, raised_by: _TestIO):
        super().__init__(message)
        self.raised_by = raised_by

    async def recover(self) -> bool:
        try:
            await self.raised_by.renew()
            return True
        except:  # noqa: E722
            return False


class _TestIO(IDestination, ISource, object):
    """IDestination and ISource implementation for unit tests.
    expires_after: read or write operations will raise _TestIOError after the configured number of reads or
    writes. If expires_after is negative, _TestIO never expires and _TestIOError is never raised.
    """

    def __init__(self, content: bytes | None = None, expires_after: int = -1):
        if content is None:
            content = b""
        self._size = len(content)
        self._content = BytesIO(content)
        self._expires_after = expires_after
        self._n = 0
        self._io_lock = Lock()

    async def renew(self) -> None:
        self._n = 0

    @contextlib.contextmanager
    def _get_io_lock(self) -> None:
        with self._io_lock:
            if self._n == self._expires_after:
                raise _TestIOError(message="IO Expired", raised_by=self)
            self._n += 1
            yield

    async def write_chunk(self, offset: int, data: bytes) -> None:
        with self._get_io_lock():
            self._content.seek(offset)
            length = self._content.write(data)
            self._size = max(self._size, offset + length)

    async def read_chunk(self, offset: int, length: int) -> bytes:
        with self._get_io_lock():
            self._content.seek(offset)
            return self._content.read(length)

    async def get_size(self) -> int:
        return self._size

    def get_raw_content(self) -> bytes:
        self._content.seek(0)
        return self._content.read()


class TestChunkedIOManager(unittest.IsolatedAsyncioTestCase):
    FILE_SIZE = 15
    CHUNK_SIZE = 4
    DATA = b"\xaa\xaa\xaa\xaa\xbb\xbb\xbb\xbb\xcc\xcc\xcc\xcc\xdd\xdd\xdd"
    EXPIRES_AFTER = 2
    PART_DATA = DATA[: EXPIRES_AFTER * CHUNK_SIZE]

    def setUp(self) -> None:
        self.manager = ChunkedIOManager(
            retry=Retry(logger, backoff_method=BackoffLinear(0)), chunk_size=self.CHUNK_SIZE, max_workers=1
        )

    async def test_run_til_complete(self) -> None:
        # Create and verify source.
        source = _TestIO(content=self.DATA)
        self.assertEqual(self.DATA, source.get_raw_content())

        # Create and verify destination.
        destination = _TestIO()
        self.assertEqual(b"", destination.get_raw_content())

        # Transfer all data.
        await self.manager.run(source, destination)

        # Verify results.
        self.assertTrue(self.manager.is_complete())
        self.assertEqual(self.DATA, destination.get_raw_content())

    async def test_run_source_expires(self) -> None:
        # Create and verify source.
        source = _TestIO(content=self.DATA, expires_after=self.EXPIRES_AFTER)
        self.assertEqual(self.DATA, source.get_raw_content())

        # Create and verify destination.
        destination = _TestIO()
        self.assertEqual(b"", destination.get_raw_content())

        # Transfer data.
        with self.assertRaises(RetryError), mock.patch.object(source, "renew") as mock_renew:
            await self.manager.run(source, destination)
        self.assertEqual(2, mock_renew.call_count)

        # Verify results.
        self.assertFalse(self.manager.is_complete())
        self.assertEqual(self.PART_DATA, destination.get_raw_content())

    async def test_run_destination_expires(self) -> None:
        # Create and verify source.
        source = _TestIO(content=self.DATA)
        self.assertEqual(self.DATA, source.get_raw_content())

        # Create and verify destination.
        destination = _TestIO(expires_after=self.EXPIRES_AFTER)
        self.assertEqual(b"", destination.get_raw_content())

        # Transfer data.
        with self.assertRaises(RetryError), mock.patch.object(destination, "renew") as mock_renew:
            await self.manager.run(source, destination)
        self.assertEqual(2, mock_renew.call_count)

        # Verify results.
        self.assertFalse(self.manager.is_complete())
        self.assertEqual(self.PART_DATA, destination.get_raw_content())

    async def test_resume(self) -> None:
        # Create and verify source.
        source = _TestIO(content=self.DATA, expires_after=self.EXPIRES_AFTER)
        self.assertEqual(self.DATA, source.get_raw_content())

        # Create and verify destination.
        destination = _TestIO()
        self.assertEqual(b"", destination.get_raw_content())

        # Transfer data.
        with self.assertRaises(RetryError), mock.patch.object(source, "renew") as mock_renew:
            await self.manager.run(source, destination)
        self.assertEqual(2, mock_renew.call_count)

        # Verify results.
        self.assertFalse(self.manager.is_complete())
        self.assertEqual(self.PART_DATA, destination.get_raw_content())

        # Create new source and verify content.
        new_source = _TestIO(content=self.DATA)
        self.assertEqual(self.DATA, source.get_raw_content())

        # Finish data transfer.
        mock_read_chunk = mock.Mock(wraps=new_source.read_chunk)
        with mock.patch.object(new_source, "read_chunk", mock_read_chunk):
            await self.manager.run(new_source, destination)
        self.assertEqual(2, mock_read_chunk.call_count)

        # Verify results.
        self.assertTrue(self.manager.is_complete())
        self.assertEqual(self.DATA, destination.get_raw_content())

    async def test_resume_fails(self) -> None:
        # Create and verify source.
        source = _TestIO(content=self.DATA, expires_after=self.EXPIRES_AFTER)
        self.assertEqual(self.DATA, source.get_raw_content())

        # Create and verify destination.
        destination = _TestIO()
        self.assertEqual(b"", destination.get_raw_content())

        # Transfer data.
        with self.assertRaises(_TestIOError), mock.patch.object(source, "renew") as mock_renew:
            mock_renew.side_effect = ValueError("cannot renew")
            await self.manager.run(source, destination)
        self.assertEqual(1, mock_renew.call_count)

        # Verify results.
        self.assertFalse(self.manager.is_complete())

    async def test_zero_byte_file(self) -> None:
        # Create and verify source.
        source = _TestIO(content=b"", expires_after=self.EXPIRES_AFTER)
        self.assertEqual(b"", source.get_raw_content())

        # Create and verify destination.
        destination = _TestIO()
        self.assertEqual(b"", destination.get_raw_content())

        # Transfer data.
        await self.manager.run(source, destination)

        # Verify results.
        self.assertTrue(self.manager.is_complete())
