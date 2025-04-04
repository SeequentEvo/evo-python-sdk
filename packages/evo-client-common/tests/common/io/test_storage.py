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

import asyncio
import base64
import logging
import random
import unittest
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from unittest import mock
from urllib.parse import urlencode

from parameterized import parameterized_class

from evo.common import RequestMethod
from evo.common.io.storage import BlockList, StorageBlock, StorageDestination
from evo.common.test_tools import TestWithUploadHandler, utc_datetime
from evo.common.utils import BackoffLinear, Retry

from .common import TEST_DATA, TestIDestination

logger = logging.getLogger(__name__)


@parameterized_class(
    [
        {"byte_offset": 0},
        {"byte_offset": 1},
        {"byte_offset": 1024},
        {"byte_offset": 4 * 1024 * 1024 * 1024 * 1024},
    ]
)
class TestStorageBlock(unittest.TestCase):
    byte_offset: int

    def setUp(self) -> None:
        self.block = StorageBlock(self.byte_offset)

    def test_byte_offset(self) -> None:
        """Test the block byte offset."""
        self.assertEqual(self.byte_offset, self.block.byte_offset)

    def test_id(self) -> None:
        """Test the block id generation."""
        expected_id = base64.b64encode(f"{self.byte_offset:032d}".encode("utf-8")).decode("utf-8")
        self.assertEqual(expected_id, self.block.id)

    def test_str(self) -> None:
        """Test the string representation of a block."""
        expected_str = f"<Latest>{self.block.id}</Latest>"
        self.assertEqual(expected_str, str(self.block))


@parameterized_class(
    [
        {"block_offsets": [0]},
        {"block_offsets": [1]},
        {"block_offsets": [i for i in range(0, 4 * 1024 * 1024, 512 * 1024)]},
    ]
)
class TestBlockList(unittest.IsolatedAsyncioTestCase):
    block_offsets: list[int]

    def setUp(self) -> None:
        self.block_list = BlockList()

    async def asyncSetUp(self) -> None:
        tasks = []
        for offset in random.sample(self.block_offsets, k=len(self.block_offsets)):
            tasks.append(self.block_list.add_block(offset))
        # Add a retry of the last task due to failure
        tasks.append(self.block_list.add_block(offset))
        self.sorted_block_ids = [StorageBlock(offset).id for offset in sorted(self.block_offsets)]
        self.randomized_block_ids = await asyncio.gather(*tasks)

    async def test_empty_list(self) -> None:
        """Test preparing an empty block list."""
        document = await BlockList().prepare()
        self.assertEqual('<?xml version="1.0" encoding="utf-8"?>\n<BlockList>\n</BlockList>', document.decode("utf-8"))

    async def test_populated_list(self) -> None:
        """Test preparing a populated block list."""
        if len(self.block_offsets) > 1:
            assert self.sorted_block_ids != self.randomized_block_ids
        document = await self.block_list.prepare()
        expected = "\n".join(
            [
                '<?xml version="1.0" encoding="utf-8"?>',
                "<BlockList>",
                *(f"  <Latest>{block_id}</Latest>" for block_id in self.sorted_block_ids),
                "</BlockList>",
            ]
        )
        self.assertEqual(expected, document.decode("utf-8"))


class TestStorageDestination(TestIDestination, TestWithUploadHandler):
    destination: StorageDestination

    def setup_destination(self) -> None:
        TestWithUploadHandler.setUp(self)
        self.destination = StorageDestination(self.url_generator.get_new_url, self.transport)

    def assert_put_block(self, offset: int, data: bytes) -> None:
        """Assert that a block was put to the server with the expected request format."""
        block = StorageBlock(offset)
        url = self.url_generator.current_url + "&" + urlencode({"comp": "block", "blockid": block.id})
        self.transport.assert_request_made(
            RequestMethod.PUT,
            url,
            headers={"Content-Length": str(len(data))},
            body=data,
        )

    def assert_put_block_list(self, block_list: bytes, with_url: str | None = None) -> None:
        """Assert that a block list was put to the server with the expected request format."""
        if with_url is None:
            with_url = self.url_generator.current_url
        self.transport.assert_any_request_made(
            RequestMethod.PUT,
            with_url + "&" + urlencode({"comp": "blocklist"}),
            headers={
                "Content-Type": "text/plain; charset=UTF-8",
                "Content-Length": str(len(block_list)),
            },
            body=block_list,
        )

    @asynccontextmanager
    async def ctx_test_write_chunk(self, offset: int, data: bytes) -> AsyncIterator[None]:
        async with self.destination:
            self.transport.assert_no_requests()
            self.assertEqual(1, self.url_generator.n_calls)
            yield
            self.assert_put_block(offset, data)
            self.assertEqual(1, self.url_generator.n_calls)

    async def test_commit(self) -> AsyncIterator[None]:
        """Test committing a file to storage submits the expected block list."""
        async with self.destination:
            chunks_with_offsets = await self.write_whole_file()
            self.transport.reset_mock()
            self.assertEqual(1, self.url_generator.n_calls)

            expected_block_list = BlockList()
            tasks = [expected_block_list.add_block(offset) for offset, _ in chunks_with_offsets]
            await asyncio.gather(*tasks)

            await self.destination.commit()
            self.assert_put_block_list(await expected_block_list.prepare())
            self.assertEqual(1, self.url_generator.n_calls)

        expected_data = b"".join(data for _, data in sorted(chunks_with_offsets))
        actual_data = await self.handler.get_committed()
        self.assertEqual(expected_data, actual_data)

    @mock.patch("evo.common.io.http.datetime", spec=datetime)
    async def test_commit_retries(self, mock_datetime: mock.Mock) -> None:
        """Test that committing a file to storage retries on auth failure."""
        mock_datetime.now.return_value = utc_datetime(2024, 7, 12, hour=12, minute=0)
        async with self.destination:
            chunks_with_offsets = await self.write_whole_file()
            self.transport.reset_mock()
            self.assertEqual(1, self.url_generator.n_calls)

            expected_block_list = BlockList()
            tasks = [expected_block_list.add_block(offset) for offset, _ in chunks_with_offsets]
            await asyncio.gather(*tasks)

            mock_datetime.now.return_value = utc_datetime(2024, 7, 12, hour=12, minute=5)
            expired_url = self.url_generator.current_url
            with self.handler.expired_url(expired_url):
                await self.destination.commit(
                    # Use a retry strategy that doesn't wait between attempts.
                    retry=Retry(logger=logger, max_attempts=3, backoff_method=BackoffLinear(0.0)),
                )

            expected_document = await expected_block_list.prepare()
            self.assert_put_block_list(expected_document, with_url=expired_url)
            self.assert_put_block_list(expected_document, with_url=self.url_generator.current_url)
            self.assertEqual(2, self.url_generator.n_calls)

    async def test_upload_file(self) -> None:
        test_data_file = self.CACHE_DIR / "test_data_upload.csv"
        test_data_file.write_bytes(TEST_DATA)
        await StorageDestination.upload_file(test_data_file, self.url_generator.get_new_url, self.transport)
        uploaded_data = await self.handler.get_committed()
        self.assertEqual(test_data_file.read_bytes(), uploaded_data)


# Delete base test classes to prevent discovery by unittest.
del TestIDestination
