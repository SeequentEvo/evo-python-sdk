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
import warnings
from unittest import IsolatedAsyncioTestCase, mock

from evo.aio import AioTransport
from evo.common import RequestMethod
from evo.common.exceptions import TransportError
from evo.common.io import ChunkedIOManager, HTTPSource, TransferLimits
from evo.common.io.interfaces import IDestination, ISource
from evo.common.io.storage import StorageDestination
from evo.common.io.transfer_limits import file_transfer_slot
from evo.common.test_tools import TestWithDownloadHandler, TestWithUploadHandler


class TestTransferLimits(IsolatedAsyncioTestCase):
    async def test_many_files_share_budget_before_reading(self):
        limits = TransferLimits(max_files=16, max_chunks=3)
        saturated = asyncio.Event()
        release = asyncio.Event()
        reads = 0
        active = 0
        peak = 0

        async def read(offset, length):
            nonlocal reads, active, peak
            reads += 1
            active += 1
            peak = max(peak, active)
            if active == 3:
                saturated.set()
            return b"x" * length

        async def write(offset, data):
            nonlocal active
            try:
                await release.wait()
            finally:
                active -= 1

        source = mock.AsyncMock(spec=ISource)
        source.get_size.return_value = 4
        source.read_chunk.side_effect = read
        destination = mock.AsyncMock(spec=IDestination)
        destination.write_chunk.side_effect = write
        tasks = [
            asyncio.create_task(ChunkedIOManager(chunk_size=4, chunk_limiter=limits.chunks).run(source, destination))
            for _ in range(100)
        ]
        try:
            await asyncio.wait_for(saturated.wait(), 2)
            await asyncio.sleep(0)
            self.assertEqual(3, reads)
            self.assertEqual(3, active)
            release.set()
            await asyncio.wait_for(asyncio.gather(*tasks), 5)
        finally:
            release.set()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        self.assertEqual(100, reads)
        self.assertEqual(3, peak)
        self.assertEqual(0, active)

    async def test_large_file_uses_fixed_workers_and_drains_repeated_cancellation(self):
        limits = TransferLimits(max_chunks=2)
        started = asyncio.Event()
        cleaning = asyncio.Event()
        release_cleanup = asyncio.Event()
        stopped = []
        source = mock.AsyncMock(spec=ISource)
        source.get_size.return_value = 4 * 10000
        source.read_chunk.return_value = b"data"
        destination = mock.AsyncMock(spec=IDestination)

        async def write(offset, data):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaning.set()
                await release_cleanup.wait()
                stopped.append(offset)

        destination.write_chunk.side_effect = write
        before = len(asyncio.all_tasks())
        manager = ChunkedIOManager(chunk_size=4, max_workers=3, chunk_limiter=limits.chunks)
        task = asyncio.create_task(manager.run(source, destination))
        try:
            await asyncio.wait_for(started.wait(), 2)
            self.assertLessEqual(len(asyncio.all_tasks()) - before, 4)
            self.assertEqual(2, source.read_chunk.await_count)
            task.cancel()
            await asyncio.wait_for(cleaning.wait(), 2)
            task.cancel()
            await asyncio.sleep(0)
            self.assertFalse(task.done())
            release_cleanup.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertEqual(2, len(stopped))
            self.assertEqual(2, limits.chunks._value)
        finally:
            release_cleanup.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def test_transport_limits_are_session_owned_and_do_not_gate_api_calls(self):
        transport = AioTransport("test", max_concurrent_transfers=1, max_concurrent_chunks=1, close_grace_period_ms=0)
        other = AioTransport("other", close_grace_period_ms=0)
        with self.assertRaises(TransportError):
            _ = transport.transfer_limits
        async with transport, other:
            first = transport.transfer_limits
            self.assertIsNot(first, other.transfer_limits)
            async with file_transfer_slot(transport), first.chunks:
                async with transport:
                    self.assertIs(first, transport.transfer_limits)
                with mock.patch("evo.aio._helpers.Context.request", new_callable=mock.AsyncMock) as request:
                    await asyncio.wait_for(transport.request(RequestMethod.GET, "https://example.test/metadata"), 2)
                    request.assert_awaited_once()
        async with transport:
            self.assertIsNot(first, transport.transfer_limits)

    async def test_default_and_explicit_http_limits(self):
        from aiohttp import TCPConnector

        for configured, expected in ((None, 100), (4, 4), (0, 0)):
            kwargs = {} if configured is None else {"max_connections": configured}
            transport = AioTransport("test", close_grace_period_ms=0, **kwargs)

            with mock.patch("evo.aio._helpers.aiohttp.TCPConnector", wraps=TCPConnector) as connector:
                async with transport:
                    connector.assert_called_once_with(ssl=None, limit=expected)

    async def test_num_pools_is_a_deprecated_alias_of_max_connections(self):
        from aiohttp import TCPConnector

        with self.assertWarnsRegex(DeprecationWarning, "num_pools") as caught:
            transport = AioTransport("test", num_pools=4, close_grace_period_ms=0)
        self.assertEqual(__file__, caught.filename)
        with mock.patch("evo.aio._helpers.aiohttp.TCPConnector", wraps=TCPConnector) as connector:
            async with transport:
                connector.assert_called_once_with(ssl=None, limit=4)

        with self.assertRaisesRegex(ValueError, "either max_connections or the deprecated num_pools"):
            AioTransport("test", num_pools=4, max_connections=4)

    def test_invalid_limits(self):
        for value in (-1, True, 1.5):
            for option in ("max_connections", "num_pools"):
                with (
                    self.subTest(value=value, option=option),
                    warnings.catch_warnings(),
                    self.assertRaisesRegex(ValueError, option),
                ):
                    warnings.simplefilter("ignore", DeprecationWarning)
                    AioTransport("test", **{option: value})
        for value in (0, -1, True, 1.5):
            for option in ("max_concurrent_transfers", "max_concurrent_chunks"):
                with self.subTest(value=value, option=option), self.assertRaises(ValueError):
                    AioTransport("test", **{option: value})
            with self.subTest(value=value), self.assertRaises(ValueError):
                ChunkedIOManager(max_workers=value)


class TestFileAdmission(TestWithUploadHandler):
    async def test_file_slot_precedes_sas_and_open_and_cancelled_waiters_do_not_start(self):
        self.transport.transfer_limits = TransferLimits(max_files=1, max_chunks=1)
        filename = self.CACHE_DIR / "admission.parquet"
        filename.write_bytes(b"test")
        started = asyncio.Event()
        release = asyncio.Event()
        original = self.handler.handle_put_block

        async def block(*args, **kwargs):
            started.set()
            await release.wait()
            return await original(*args, **kwargs)

        first_url = mock.AsyncMock(side_effect=self.url_generator.get_new_url)
        second_url = mock.AsyncMock(side_effect=self.url_generator.get_new_url)
        with (
            mock.patch.object(self.handler, "handle_put_block", side_effect=block),
            mock.patch("pathlib.Path.open", autospec=True, side_effect=type(filename).open) as opened,
        ):
            first = asyncio.create_task(StorageDestination.upload_file(filename, first_url, self.transport))
            second = None
            try:
                await asyncio.wait_for(started.wait(), 2)
                second = asyncio.create_task(StorageDestination.upload_file(filename, second_url, self.transport))
                await asyncio.sleep(0)
                await asyncio.sleep(0)
                second_url.assert_not_awaited()
                self.assertEqual(1, opened.call_count)
                second.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await second
                release.set()
                await asyncio.wait_for(first, 2)
            finally:
                release.set()
                tasks = [first] + ([second] if second is not None else [])
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
        self.assertEqual(1, self.transport.transfer_limits.files._value)


class TestDownloadAdmission(TestWithDownloadHandler):
    async def test_downloads_share_file_admission_before_requesting_urls(self):
        self.setup_download_handler(b"test")
        self.transport.transfer_limits = TransferLimits(max_files=1, max_chunks=1)
        filename = self.CACHE_DIR / "download-admission.parquet"
        task = None
        try:
            async with file_transfer_slot(self.transport):
                task = asyncio.create_task(
                    HTTPSource.download_file(filename, self.url_generator.get_new_url, self.transport)
                )
                await asyncio.sleep(0)
                await asyncio.sleep(0)
                self.assertEqual(0, self.url_generator.n_calls)
                self.transport.assert_no_requests()
                self.assertFalse(filename.exists())
            await asyncio.wait_for(task, 2)
            self.assertEqual(b"test", filename.read_bytes())
        finally:
            if task is not None:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
