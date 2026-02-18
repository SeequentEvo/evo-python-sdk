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

from unittest import mock

from evo.common.connector import APIConnector
from evo.common.io import HTTPSource
from evo.common.test_tools import TestWithDownloadHandler

from .common import TEST_DATA, AsyncIterator, TestISource, asynccontextmanager


class TestHTTPSource(TestISource, TestWithDownloadHandler):
    def setup_source(self, test_data: bytes) -> None:
        TestWithDownloadHandler.setUp(self)
        self.setup_download_handler(test_data)
        self.source = HTTPSource(self.url_generator.get_new_url, self.transport)

    async def test_context_manager(self) -> None:
        """Test context manager functionality"""
        self.transport.assert_no_requests()
        self.transport.open.assert_not_called()
        self.transport.close.assert_not_called()
        self.assertEqual(0, self.url_generator.n_calls)

        with self.assertRaises(AssertionError):
            await self.source.get_size()

        async with self.source:
            self.assertEqual(2, self.transport.open.call_count)
            self.transport.close.assert_called_once()
            self.assertEqual(1, self.url_generator.n_calls)

            # We currently perform a HEAD request to get the size when the context is entered. The headers in the
            # response also tell us whether the server supports range requests.
            self.assert_head_request_made(self.url_generator.current_url)
            self.transport.reset_mock()

        self.transport.assert_no_requests()
        self.transport.close.assert_called_once()
        self.assertEqual(1, self.url_generator.n_calls)

    @asynccontextmanager
    async def ctx_test_get_size(self) -> AsyncIterator[None]:
        self.transport.assert_no_requests()
        async with self.source:
            # We currently perform a HEAD request to get the size when the context is entered.
            self.assert_head_request_made(self.url_generator.current_url)
            self.transport.reset_mock()

            yield

            # The size is cached after the first request.
            self.transport.assert_no_requests()

    @asynccontextmanager
    async def ctx_test_read_chunk(self, offset: int, length: int) -> AsyncIterator[None]:
        self.transport.assert_no_requests()
        async with self.source:
            self.transport.reset_mock()
            yield
            self.assert_range_request_made(self.url_generator.current_url, offset, length)

    async def test_download_file(self) -> None:
        """Test that the source can download a file."""
        test_data_file = self.CACHE_DIR / "test_data_download.csv"
        self.assertFalse(test_data_file.exists())
        await HTTPSource.download_file(str(test_data_file), self.url_generator.get_new_url, self.transport)
        self.assertTrue(test_data_file.exists())
        self.assertEqual(TEST_DATA, test_data_file.read_bytes())

    @mock.patch("evo.common.io.http.APIConnector", wraps=APIConnector)
    async def test_additional_headers_passed_to_connector(self, mock_api_connector: mock.Mock) -> None:
        """Test that additional headers are passed through to APIConnector."""
        additional_headers = {"X-Custom-Header": "custom-value", "X-Another-Header": "another-value"}
        source = HTTPSource(self.url_generator.get_new_url, self.transport, additional_headers=additional_headers)

        async with source:
            # Verify APIConnector was called with the additional headers
            mock_api_connector.assert_called_once()
            call_kwargs = mock_api_connector.call_args.kwargs
            self.assertEqual(additional_headers, call_kwargs.get("additional_headers"))


# Delete base test classes to prevent discovery by unittest.
del TestISource
