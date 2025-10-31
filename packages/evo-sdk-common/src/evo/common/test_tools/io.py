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
import os
import re
import unittest
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from io import BytesIO
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse

from ..data import HTTPHeaderDict, RequestMethod
from ..io.chunked_io_manager import DEFAULT_CHUNK_SIZE
from .http import AbstractTestRequestHandler, MockResponse, TestTransport
from .storage import TestWithStorage


class UrlGenerator:
    def __init__(self, base_url: str = "https://unittest.localhost/") -> None:
        self._base_url = base_url.removesuffix("/")
        self._n = 0
        self._current_url: str | None = None

    @property
    def n_calls(self) -> int:
        return self._n

    @property
    def current_url(self) -> str:
        assert self._current_url is not None
        return self._current_url

    async def get_new_url(self) -> str:
        self._n += 1
        self._current_url = (
            urlparse(self._base_url)
            ._replace(
                path="file.txt",
                query=urlencode(
                    [
                        ("attempt", self._n),
                        ("oct", self._n),
                        ("dt", datetime.now(timezone.utc).isoformat()),
                        ("blank", ""),
                    ]
                ),
            )
            .geturl()
        )
        return self.current_url


RANGE_HEADER_RE = re.compile(r"bytes=(?P<first>\d+)-(?P<last>\d+)")


class DownloadRequestHandler(AbstractTestRequestHandler):
    def __init__(self, data: bytes) -> None:
        self.data = data
        self._mutex = asyncio.Lock()
        self._buffer = BytesIO(data)
        self.size = len(data)

    async def handle_get(self, headers: HTTPHeaderDict) -> MockResponse:
        """Handle GET range requests for the test data.

        If the request is not a range request with a single range, return a 400 response.

        :param headers: The request headers.

        :return: A response containing the requested data.
        """
        requested = headers.get("Range", "")
        if match := RANGE_HEADER_RE.fullmatch(requested):
            requested_first = int(match.group("first"))
            requested_last = int(match.group("last"))
        else:
            return self.bad_request()

        requested_size = (requested_last - requested_first) + 1
        async with self._mutex:
            actual_first = self._buffer.seek(requested_first, os.SEEK_SET)
            data = self._buffer.read(requested_size)
            actual_last = self._buffer.tell() - 1
        response_headers = {
            "Content-Range": f"bytes {actual_first}-{actual_last}/{self.size}",
            "Content-Length": str(len(data)),
        }
        return MockResponse(status_code=206, body=data, reason="Partial Content", headers=response_headers)

    async def handle_head(self) -> MockResponse:
        """Handle HEAD requests for the test data.

        :return: A response containing the size of the test data and indicating support for range requests.
        """
        return MockResponse(
            status_code=200,
            reason="OK",
            headers={
                "Accept-Ranges": "bytes",
                "Content-Length": str(self.size),
            },
        )

    async def request(
        self,
        method: RequestMethod,
        url: str,
        headers: HTTPHeaderDict | None = None,
        post_params: list[tuple[str, str | bytes]] | None = None,
        body: object | str | bytes | None = None,
        request_timeout: int | float | tuple[int | float, int | float] | None = None,
    ) -> MockResponse:
        match method:
            case RequestMethod.HEAD:
                return await self.handle_head()
            case RequestMethod.GET:
                return await self.handle_get(headers or HTTPHeaderDict())
            case _:
                return self.not_found()


class MultiDownloadRequestHandler(AbstractTestRequestHandler):
    """A request handler that can handle multiple download URLs."""

    def __init__(self, data: dict[str, bytes]) -> None:
        self._handlers: dict[str, DownloadRequestHandler] = {
            url: DownloadRequestHandler(file_data) for url, file_data in data.items()
        }

    async def request(
        self,
        method: RequestMethod,
        url: str,
        headers: HTTPHeaderDict | None = None,
        post_params: list[tuple[str, str | bytes]] | None = None,
        body: object | str | bytes | None = None,
        request_timeout: int | float | tuple[int | float, int | float] | None = None,
    ) -> MockResponse:
        if url not in self._handlers:
            return self.not_found()
        return await self._handlers[url].request(method, url, headers, post_params, body, request_timeout)


class TestWithDownloadHandler(unittest.IsolatedAsyncioTestCase, TestWithStorage):
    def setUp(self) -> None:
        super().setUp()
        if not hasattr(self, "transport"):
            # Allowing for multiple inheritance, where the transport is set in a different setUp method.
            self.transport = TestTransport()
        self.url_generator = UrlGenerator()

    def setup_download_handler(self, data: bytes) -> DownloadRequestHandler:
        self.handler = DownloadRequestHandler(data)
        self.transport.set_request_handler(self.handler)
        return self.handler

    def assert_head_request_made(self, url: str) -> None:
        """Assert that a HEAD request was made to the given URL.

        :param url: The URL the HEAD request should have been made to.
        """
        self.transport.assert_any_request_made(RequestMethod.HEAD, url)

    def assert_range_request_made(self, url: str, offset: int, length: int) -> None:
        """Assert that a range request was made with the given offset and length.

        :param url: The URL the range request should have been made to.
        :param offset: The expected offset of the range request.
        :param length: The expected length of the range request.
        """
        self.transport.assert_any_request_made(
            RequestMethod.GET,
            url,
            headers={
                "Range": f"bytes={offset}-{offset + length - 1}",
            },
        )

    def assert_download_requests(self, url: str, chunk_size: int = DEFAULT_CHUNK_SIZE) -> None:
        """Assert that the correct requests were made to download the file.

        :param url: The URL the file should have been downloaded from.
        :param chunk_size: The expected chunk size for the requests.
        """
        filesize = len(self.handler.data)
        self.assert_head_request_made(url)
        for offset in range(0, filesize, chunk_size):
            self.assert_range_request_made(url, offset, min(chunk_size, filesize - offset))


class StorageDestinationRequestHandler(AbstractTestRequestHandler):
    def __init__(self) -> None:
        self._mutex = asyncio.Lock()
        self._buffer: dict[str, bytes] = {}
        self._block_list: list[str] | None = None
        self._expired_query_params: dict[str, str] | None = None

    def is_expired(self, query_params: dict[str, Any]) -> bool:
        """Check if the URL is expired."""
        if self._expired_query_params is None:
            return False
        current_query_params = {field: query_params[field] for field in self._expired_query_params}
        return self._expired_query_params == current_query_params

    async def get_committed(self) -> bytes:
        """Get the committed data from the buffer.

        :return: The committed data.
        """
        async with self._mutex:
            assert self._block_list is not None
            return b"".join(self._buffer[block_id] for block_id in self._block_list)

    async def handle_put_block(self, block_id: str, headers: HTTPHeaderDict, body: bytes) -> MockResponse:
        """Mock Storage PUT BLOCK operation.

        :param block_id: The block id from the query parameters.
        :param headers: The request headers.
        :param body: The request body.

        :return: A response indicating success.
        """
        async with self._mutex:
            assert block_id not in self._buffer, f"Block {block_id} already exists."
            self._buffer[block_id] = body

        return MockResponse(status_code=201, reason="Created")

    async def handle_put_block_list(self, headers: HTTPHeaderDict, body: bytes) -> MockResponse:
        """Mock Storage PUT BLOCK LIST operation.

        :param headers: The request headers.
        :param body: The request body.

        :return: A response indicating success.
        """
        async with self._mutex:
            assert headers.get("Content-Type") == "text/plain; charset=UTF-8"
            document = body.decode("utf-8")
            assert document.startswith('<?xml version="1.0" encoding="utf-8"?>\n<BlockList>\n') and document.endswith(
                "\n</BlockList>"
            ), f"Invalid XML document\n{document}"
            assert self._block_list is None, "Block list already committed."

            self._block_list = []
            for entry in document.split("\n")[2:-1]:
                assert entry.startswith("  <Latest>") and entry.endswith("</Latest>"), f"Invalid XML row: '{entry}'"
                block_id = entry[10:-9]
                assert block_id in self._buffer, f"Missing block with ID: '{block_id}'"
                self._block_list.append(block_id)

            all_block_ids = set(self._buffer.keys())
            committed_block_ids = set(self._block_list)
            assert all_block_ids == committed_block_ids, f"Missing blocks: {all_block_ids - committed_block_ids}"

        return MockResponse(status_code=201, reason="Created")

    async def handle_put(self, url: str, headers: HTTPHeaderDict, body: bytes) -> MockResponse:
        assert body is not None, "Body is required for PUT requests."
        assert "Content-Length" in headers, "Content-Length header is required for PUT requests."
        assert int(headers.get("Content-Length")) == len(body), "Content-Length header must match body length."
        uri = urlparse(url)
        query = parse_qs(uri.query)

        if self.is_expired(query):  # Mock expired URL.
            return MockResponse(status_code=403, reason="Forbidden")

        match query.get("comp", [""]):
            case ["block"]:
                try:
                    block_id = query["blockid"][0]
                except (KeyError, IndexError):
                    return self.bad_request()
                return await self.handle_put_block(block_id, headers, body)
            case ["blocklist"]:
                return await self.handle_put_block_list(headers, body)
            case _:
                return self.bad_request()

    async def request(
        self,
        method: RequestMethod,
        url: str,
        headers: HTTPHeaderDict | None = None,
        post_params: list[tuple[str, str | bytes]] | None = None,
        body: object | str | bytes | None = None,
        request_timeout: int | float | tuple[int | float, int | float] | None = None,
    ) -> MockResponse:
        match method:
            case RequestMethod.PUT:
                return await self.handle_put(url, headers, body)
            case _:
                return self.not_found()

    @contextmanager
    def expired_url(self, url: str) -> Iterator[None]:
        """Context manager that simulates an expired URL."""
        uri = urlparse(url)
        self._expired_query_params = parse_qs(uri.query)
        yield
        self._expired_query_params = None


class TestWithUploadHandler(unittest.IsolatedAsyncioTestCase, TestWithStorage):
    def setUp(self) -> None:
        super().setUp()
        if not hasattr(self, "transport"):
            # Allowing for multiple inheritance, where the transport is set in a different setUp method.
            self.transport = TestTransport()
        self.handler = StorageDestinationRequestHandler()
        self.transport.set_request_handler(self.handler)
        self.url_generator = UrlGenerator()
