#  Copyright (c) 2026 Bentley Systems, Incorporated
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import asyncio
import logging
import weakref
from collections import Counter
from io import BytesIO
from unittest import mock

from parameterized import parameterized

from evo.common import HTTPHeaderDict
from evo.common.exceptions import EvoAPIException, RetryError, TransportError
from evo.common.io import BytesSource, ChunkedIOManager, StorageDestination, TransferLimits
from evo.common.io.exceptions import ChunkedIOError, TransientIOError
from evo.common.io.interfaces import IDestination, ISource
from evo.common.test_tools import TestWithUploadHandler
from evo.common.utils import BackoffLinear, Retry


def retry_policy():
    return Retry(logging.getLogger(__name__), max_attempts=3, backoff_method=BackoffLinear(0))


def http_error(status):
    return EvoAPIException(status, "test failure", None, HTTPHeaderDict())


class TestTransientUploads(TestWithUploadHandler):
    async def test_retry_tracebacks_do_not_retain_chunk_payloads(self):
        references = []

        class Payload(bytearray):
            pass

        class Source(ISource):
            async def get_size(self):
                return 4

            async def read_chunk(self, offset, length):
                data = Payload(b"test")
                references.append(weakref.ref(data))
                return data

        class Destination(IDestination):
            async def write_chunk(self, offset, data):
                raise TransientIOError("temporary")

        with self.assertRaises(RetryError) as caught:
            await ChunkedIOManager(retry=retry_policy(), max_workers=1).run(Source(), Destination())
        self.assertEqual(3, len(caught.exception.exceptions))
        self.assertEqual(3, len(references))
        self.assertTrue(all(reference() is None for reference in references))

    @parameterized.expand([(408,), (429,), (500,), (502,), (503,), (504,)])
    async def test_transient_chunk_failure_preserves_completed_blocks(self, status):
        counts = Counter()
        original = self.handler.handle_put_block

        async def put(block_id, headers, body):
            counts[block_id] += 1
            if body == b"efgh" and counts[block_id] == 1:
                raise http_error(status)
            return await original(block_id, headers, body)

        with mock.patch.object(self.handler, "handle_put_block", side_effect=put):
            async with StorageDestination(self.url_generator.get_new_url, self.transport) as destination:
                manager = ChunkedIOManager(chunk_size=4, max_workers=1, retry=retry_policy())
                await manager.run(BytesSource(BytesIO(b"abcdefgh")), destination)
                await destination.commit(retry_policy())
        self.assertEqual([1, 2], list(counts.values()))
        self.assertEqual(b"abcdefgh", await self.handler.get_committed())
        self.assertEqual(1, self.url_generator.n_calls)

    async def test_transient_commit_retries_without_reuploading_chunks(self):
        original = self.handler.handle_put_block_list
        attempts = 0

        async def commit(headers, body):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise http_error(503)
            return await original(headers, body)

        filename = self.CACHE_DIR / "commit.parquet"
        filename.write_bytes(b"test")
        with (
            mock.patch.object(self.handler, "handle_put_block_list", side_effect=commit),
            mock.patch.object(self.handler, "handle_put_block", wraps=self.handler.handle_put_block) as put,
        ):
            await StorageDestination.upload_file(
                filename, self.url_generator.get_new_url, self.transport, retry=retry_policy()
            )
        self.assertEqual(2, attempts)
        self.assertEqual(1, put.await_count)
        self.assertEqual(b"test", await self.handler.get_committed())

    async def test_transient_url_failure_retries_inside_file_admission(self):
        filename = self.CACHE_DIR / "url.parquet"
        filename.write_bytes(b"test")
        self.transport.transfer_limits = TransferLimits(max_files=1)
        calls = 0

        async def get_url():
            nonlocal calls
            calls += 1
            self.assertTrue(self.transport.transfer_limits.files.locked())
            if calls == 1:
                raise http_error(429)
            return await self.url_generator.get_new_url()

        await StorageDestination.upload_file(filename, get_url, self.transport, retry=retry_policy())
        self.assertEqual(2, calls)
        self.assertEqual(1, self.transport.transfer_limits.files._value)
        self.assertEqual(b"test", await self.handler.get_committed())

    async def test_exhausted_transient_url_failures_raise_original_error(self):
        filename = self.CACHE_DIR / "url-exhausted.parquet"
        filename.write_bytes(b"test")
        calls = 0

        async def get_url():
            nonlocal calls
            calls += 1
            raise http_error(503)

        with self.assertRaises(EvoAPIException) as caught:
            await StorageDestination.upload_file(filename, get_url, self.transport, retry=retry_policy())
        self.assertEqual(503, caught.exception.status)
        self.assertIsInstance(caught.exception.__cause__, RetryError)
        self.assertEqual(3, len(caught.exception.__cause__.exceptions))
        self.assertEqual(3, calls)
        filename.unlink()

    async def test_permanent_url_failure_is_not_retried(self):
        filename = self.CACHE_DIR / "url-permanent.parquet"
        filename.write_bytes(b"test")
        get_url = mock.AsyncMock(side_effect=http_error(404))

        with self.assertRaises(EvoAPIException) as caught:
            await StorageDestination.upload_file(filename, get_url, self.transport, retry=retry_policy())
        self.assertEqual(404, caught.exception.status)
        get_url.assert_awaited_once()
        filename.unlink()

    @parameterized.expand([(400,), (404,), (409,), (422,)])
    async def test_permanent_chunk_failures_are_not_retried(self, status):
        with mock.patch.object(self.handler, "handle_put_block", side_effect=http_error(status)) as put:
            async with StorageDestination(self.url_generator.get_new_url, self.transport) as destination:
                with self.assertRaises(ChunkedIOError) as caught:
                    await ChunkedIOManager(retry=retry_policy()).run(BytesSource(BytesIO(b"test")), destination)
        self.assertNotIsInstance(caught.exception, TransientIOError)
        self.assertEqual(1, put.await_count)

    async def test_exhausted_transport_failures_are_not_retried_again(self):
        error = TransportError("transport attempts exhausted")
        with mock.patch.object(self.handler, "handle_put_block", side_effect=error) as put:
            async with StorageDestination(self.url_generator.get_new_url, self.transport) as destination:
                with self.assertRaises(ChunkedIOError):
                    await ChunkedIOManager(retry=retry_policy()).run(BytesSource(BytesIO(b"test")), destination)
        self.assertEqual(1, put.await_count)

    async def test_persistent_failure_exhausts_budget_and_never_commits(self):
        filename = self.CACHE_DIR / "failed.parquet"
        filename.write_bytes(b"test")
        self.transport.transfer_limits = TransferLimits(max_files=1, max_chunks=1)
        with (
            mock.patch.object(self.handler, "handle_put_block", side_effect=http_error(503)) as put,
            mock.patch.object(
                self.handler, "handle_put_block_list", wraps=self.handler.handle_put_block_list
            ) as commit,
        ):
            with self.assertRaises(RetryError):
                await StorageDestination.upload_file(
                    filename, self.url_generator.get_new_url, self.transport, retry=retry_policy()
                )
        self.assertEqual(3, put.await_count)
        commit.assert_not_awaited()
        self.assertEqual(1, self.transport.transfer_limits.files._value)
        self.assertEqual(1, self.transport.transfer_limits.chunks._value)
        filename.unlink()

    async def test_cancellation_during_backoff_stops_retries(self):
        filename = self.CACHE_DIR / "cancelled.parquet"
        filename.write_bytes(b"test")
        sleeping = asyncio.Event()

        async def sleep(_delay):
            sleeping.set()
            await asyncio.Event().wait()

        with (
            mock.patch.object(self.handler, "handle_put_block", side_effect=http_error(503)) as put,
            mock.patch("evo.common.utils.retry.asyncio.sleep", side_effect=sleep),
        ):
            task = asyncio.create_task(
                StorageDestination.upload_file(
                    filename, self.url_generator.get_new_url, self.transport, retry=retry_policy()
                )
            )
            try:
                await asyncio.wait_for(sleeping.wait(), 2)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        self.assertEqual(1, put.await_count)
        filename.unlink()
