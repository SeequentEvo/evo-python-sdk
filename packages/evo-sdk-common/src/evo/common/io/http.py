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

import asyncio
import os
from collections.abc import Awaitable, Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from types import TracebackType
from typing import Any, ParamSpec, TypeVar
from urllib.parse import parse_qsl, urlparse

from evo import logging

from .._types import PathLike, resolve_path
from ..connector import APIConnector
from ..data import EmptyResponse, HTTPHeaderDict, HTTPResponse, RequestMethod
from ..exceptions import ForbiddenException, UnauthorizedException
from ..interfaces import IFeedback, ITransport
from ..utils import NoFeedback, Retry
from .bytes import BytesDestination
from .chunked_io_manager import ChunkedIOManager
from .exceptions import ChunkedIOError, RenewalError, RenewalTimeoutError
from .interfaces import ISource

logger = logging.getLogger("io.http")

__all__ = [
    "HTTPIOBase",
    "HTTPSource",
    "ResourceAuthorizationError",
]


T = TypeVar("T")
P = ParamSpec("P")

_T_Self = TypeVar("_T_Self", bound="HTTPIOBase")


class ResourceAuthorizationError(ChunkedIOError):
    """Resource authorization failed.

    This error is raised when resource operations fail with a 401 status code. This error usually indicates that
    the resource URL that was provided has expired and needs to be renewed. The error can be recovered by renewing the
    resource URL, which must in turn be supported by the issuing API.
    """

    def __init__(self, message: str, client: HTTPIOBase) -> None:
        """
        :param message: The error message.
        :param client: The client that raised the exception.
        """
        super().__init__(message)
        self.client = client

    async def recover(self) -> bool:
        """Renew the resource URL so that we can retry the operation.

        :return: True if the renewal was successful.
        """
        return await self.client.renew_url()


class HTTPIOBase:
    """Base class for HTTP-based IO operations."""

    _RENEWAL_SECONDS_THRESHOLD = 5 * 60  # Five minutes

    def __init__(
        self,
        url_callback: Callable[[], Awaitable[str]],
        transport: ITransport,
        additional_headers: dict[str, str] | None = None,
    ) -> None:
        """
        :param url_callback: An awaitable callback method that accepts no arguments and returns a new resource url.
            The callback is used to renew the resource url when the current one expires. Resource URLs are considered
            expired when resource operations fail with a 401 status code.
        :param transport: The transport to use for the API calls.
        """
        self.__callback = url_callback
        self.__transport = transport
        self.__mutex = asyncio.Lock()
        self.__connector: APIConnector | None = None
        self.__resource_path: str | None = None
        self.__query_params: list[tuple[str, Any]] = []
        self.__renewal_time: datetime | None = None
        self.__additional_headers = additional_headers

    async def __get_new_url(self) -> str:
        """Get a new resource url.

        :returns: The new resource url.
        """
        url = await self.__callback()
        self.__renewal_time = datetime.now(timezone.utc)
        return url

    def __update_url(self, url: str) -> None:
        """Update the resource url.

        :param url: The new resource url.
        """
        if not url.startswith(self.__connector.base_url):
            raise RenewalError("Renewed resource URL does not match the original resource URL.")
        uri = urlparse(url)
        self.__query_params = parse_qsl(uri.query, keep_blank_values=True)
        self.__resource_path = uri._replace(query="").geturl().removeprefix(self.__connector.base_url)

    async def __aenter__(self: _T_Self) -> _T_Self:
        async with self.__mutex:
            if self.__connector is None:
                url = await self.__get_new_url()
                if not url.startswith("https://"):
                    raise ValueError("Unsupported URL scheme")
                base_url = "https://" + urlparse(url).hostname
                self.__connector = APIConnector(
                    base_url=base_url, transport=self.__transport, additional_headers=self.__additional_headers
                )
                self.__update_url(url)
        await self.__connector.__aenter__()
        return self

    async def __aexit__(
        self, exc_type: type[Exception] | None, exc_val: Exception | None, exc_tb: TracebackType | None
    ) -> None:
        await self.__connector.__aexit__(exc_type, exc_val, exc_tb)

    async def renew_url(self) -> bool:
        """Get a new resource url for resource operations.

        The url_callback may not be thread-safe and should only be called in one thread. This function will not renew
        and will raise RenewalTimeoutError if the renewal was requested within a specified period of the previous
        renewal.

        Exceptions raised by the url_callback are not handled and will be propagated.

        :raises RenewalTimeoutError: if the renewal request was made within the threshold

        :return: True when the renewal is successful
        """
        async with self.__mutex:
            assert self.__connector is not None

            if self.__renewal_time is not None:
                delta = datetime.now(timezone.utc) - self.__renewal_time
                if delta.total_seconds() < self._RENEWAL_SECONDS_THRESHOLD:
                    raise RenewalTimeoutError(
                        f"Renewed the url {delta.total_seconds()} seconds ago, but {self._RENEWAL_SECONDS_THRESHOLD} "
                        "second period required."
                    )

            self.__update_url(await self.__get_new_url())

        return True

    async def _query_resource(
        self,
        method: RequestMethod,
        *,
        query_params: Mapping[str, Any] | None = None,
        header_params: Mapping[str, Any] | None = None,
        body: object | str | bytes | None = None,
        collection_formats: Mapping[str, str] | None = None,
        response_types_map: Mapping[str, type[T]] | None = None,
    ) -> T:
        """Query the resource with the given method and headers.

        :param method: HTTP request method.
        :param query_params: Query parameters to embed in the url.
        :param header_params: Header parameters to be placed in the request header.
        :param body: Body to send with the request.
        :param response_types_map: Mapping of response status codes to response data types. The response will
            be deserialized to the corresponding type.
        """
        all_query_params = self.__query_params.copy()
        if query_params is not None:
            all_query_params.extend(query_params.items())
        async with self.__mutex:
            assert self.__connector is not None
            query = self.__connector.call_api(
                method,
                self.__resource_path,
                query_params=all_query_params,
                header_params=header_params,
                body=body,
                collection_formats=collection_formats,
                response_types_map=response_types_map,
            )

        try:
            return await query
        except (UnauthorizedException, ForbiddenException) as e:
            raise ResourceAuthorizationError(str(e), self) from e
        except Exception as e:
            raise ChunkedIOError(str(e)) from e


class HTTPSource(HTTPIOBase, ISource):
    """HTTP source for reading data in chunks.

    This general-purpose client relies on HTTP range requests to read chunks of data from a remote resource.
    The proliferation of HTTP range request support in modern web servers makes this client suitable for downloading
    large files from a variety of sources.

    For more information on range requests, refer to https://developer.mozilla.org/en-US/docs/Web/HTTP/Range_requests.
    """

    def __init__(
        self,
        url_callback: Callable[[], Awaitable[str]],
        transport: ITransport,
        additional_headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(url_callback, transport, additional_headers)
        self._size: int | None = None

    async def __aenter__(self) -> HTTPSource:
        await super().__aenter__()
        await self.get_size()
        return self

    async def get_size(self) -> int:
        """Get the size of the source data.

        :returns: The size of the source data in bytes.

        :raises ValueError: If the target resource does not support range requests.
        """
        if self._size is None:
            logger.debug("Requesting file size...")
            response = await self._query_resource(
                RequestMethod.HEAD,
                response_types_map={
                    "200": EmptyResponse,  # Use the raw response so that we can access the headers.
                },
            )
            headers = response.headers
            # https://developer.mozilla.org/en-US/docs/Web/HTTP/Range_requests#checking_if_a_server_supports_partial_requests
            if headers.get("Accept-Ranges", None) is None:
                raise ChunkedIOError("Requested resource does not support range requests.")
            self._size = int(headers["Content-Length"])  # Content-Length is the total size of the resource.
        return self._size

    async def read_chunk(self, offset: int, length: int) -> bytes:
        """Read a chunk of data from the source.

        :param offset: The offset from the start of the source data to read from.
        :param length: The length of the data to read.

        :returns: The data read from the source.
        """
        # https://developer.mozilla.org/en-US/docs/Web/HTTP/Range_requests#single_part_ranges
        req_headers = HTTPHeaderDict(range=f"bytes={offset}-{offset + length - 1}")
        logger.debug(f"Requesting chunk {req_headers['Range']}")
        response = await self._query_resource(
            RequestMethod.GET,
            header_params=req_headers,
            response_types_map={
                "206": HTTPResponse,  # Use the raw response so that we can access the headers.
            },
        )
        logger.debug(f"Received chunk {response.headers['Content-Range']}")
        return response.data

    @staticmethod
    async def download_file(
        filename: PathLike,
        url_generator: Callable[[], Awaitable[str]],
        transport: ITransport,
        max_workers: int | None = None,
        retry: Retry | None = None,
        overwrite: bool = False,
        additional_headers: dict[str, str] | None = None,
        fb: IFeedback = NoFeedback,
    ) -> None:
        """Download an HTTP resource to a file with the given filename.

        The url generator MUST generate a resource URL that can be used to access the required resource. The URL
        generator may be called again if the last URL expires (unless Retry is initialised with max_attempts == 0).

        :param filename: file to download to
        :param url_generator: An awaitable callback that accepts no arguments and returns a URL to download from
        :param transport: The transport to use for the API calls.
        :param max_workers: The maximum number of concurrent connections to use for data transfer. If None, the default
            is `min(32, os.cpu_count() + 4)`
        :param retry: A Retry object with a wait strategy. If None, a default Retry is created. If a chunk is successfully
            transferred the attempt counter will be reset.
        :param overwrite: whether to overwrite an existing local file
        :param fb: feedback to track the download, by tracking writes to the file
        :param additional_headers: Additional headers to include in each request.

        :raises FileNameTooLongError: If the filename is too long.
        :raises ValueError: if the file already exists and overwrite is False
        :raises ChunkedIOException: if a non-recoverable exception occurred.
        :raises RetryError: if the maximum number of consecutive attempts have failed.
        """
        dst_path = resolve_path(filename, check_path_length=True)

        if dst_path.exists() and not overwrite:
            raise ValueError(f"file {dst_path} already exists, must specify overwrite")

        if retry is None:
            retry = Retry(logger=logger)
        manager = ChunkedIOManager(retry=retry, max_workers=max_workers)

        async with HTTPSource(url_generator, transport, additional_headers=additional_headers) as source:
            tmp_file = None
            try:
                with NamedTemporaryFile(delete=False, dir=dst_path.parent) as tmp_file:
                    logger.debug(f"Writing temporary file {tmp_file.name}")
                    destination = BytesDestination(tmp_file)
                    await manager.run(source, destination, fb)

                tmp_path = Path(tmp_file.name).resolve()

                logger.debug(f"Renaming {tmp_path.name} to {dst_path.name}")
                tmp_path.replace(dst_path)

            except BaseException:
                if tmp_file is not None:
                    logger.error(
                        f"Removing temporary file '{tmp_file.name}' due to an unhandled exception", exc_info=True
                    )
                    os.unlink(tmp_file.name)
                raise
