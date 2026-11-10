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
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode

import aiohttp
from aiohttp.typedefs import StrOrURL

from evo.common import HTTPHeaderDict, HTTPResponse
from evo.common.exceptions import RetryError, TransportError
from evo.common.utils import Retry

__all__ = ["Config", "Context"]


@dataclass(frozen=True, kw_only=True)
class Config:
    user_agent: str
    """The value to provide in the `User-Agent` header."""

    num_pools: int
    """Maximum simultaneous HTTP connections across all hosts; 0 disables the limit."""

    verify_ssl: bool
    """Verify SSL certificates."""

    retry: Retry
    """Retry handler."""

    proxy: StrOrURL | None
    """Proxy server to use for the request."""

    close_grace_period_ms: int
    """Grace period (in milliseconds) to wait for connections to close gracefully. 250 is used if not set"""

    async def create_context(self) -> Context:
        return Context(
            num_pools=self.num_pools,
            verify_ssl=self.verify_ssl,
            retry_handler=self.retry,
            proxy=self.proxy,
            close_grace_period_ms=self.close_grace_period_ms,
        )


class Context:
    """Inner class to manage the aiohttp session."""

    def __init__(
        self, num_pools: int, verify_ssl: bool, retry_handler: Retry, proxy: StrOrURL | None, close_grace_period_ms: int
    ) -> None:
        """
        :param num_pools: Maximum simultaneous HTTP connections across all hosts; 0 disables the limit.
        :param verify_ssl: Verify SSL certificates.
        :param retry_handler: Retry handler.
        """
        connector = aiohttp.TCPConnector(ssl=None if verify_ssl else False, limit=num_pools)

        # TODO: Debug tracing? https://docs.aiohttp.org/en/stable/tracing_reference.html
        self.__session: aiohttp.ClientSession | None = aiohttp.ClientSession(
            connector=connector, skip_auto_headers=["Accept", "Accept-Encoding"]
        )
        self.__retry_handler = retry_handler
        self.__proxy = proxy
        self._close_grace_period_ms = close_grace_period_ms

    async def close(self) -> None:
        """Close the aiohttp session."""

        session = self.__session
        self.__session = None
        await session.close()

        # Wait for the underlying SSL connections to close
        # https://docs.aiohttp.org/en/stable/client_advanced.html#graceful-shutdown
        # Per comments on https://github.com/aio-libs/aiohttp/issues/1925, this should be fixed in aiohttp 4.0.
        await asyncio.sleep(self._close_grace_period_ms / 1000)

    async def __perform_request(self, **kwargs: Any) -> HTTPResponse:
        """Perform an HTTP request."""

        if self.__session is None:
            raise TransportError("Cannot make a request after the transport has been closed.")

        try:
            async for request_attempt in self.__retry_handler:
                with request_attempt.suppress_errors():
                    async with self.__session.request(allow_redirects=False, **kwargs) as resp:
                        return HTTPResponse(
                            status=resp.status,
                            data=await resp.read(),
                            reason=resp.reason,
                            headers=HTTPHeaderDict(resp.headers),
                        )
        except RetryError as error:
            raise TransportError("Reached maximum number of retries", caused_by=error).with_traceback(
                error.__traceback__
            )

    async def request_form(
        self,
        method: str,
        url: str,
        headers: HTTPHeaderDict,
        fields: list[tuple[str, str | bytes]],
        timeout: aiohttp.ClientTimeout | None,
    ) -> HTTPResponse:
        """Submit a request with urlencoded or multipart form data.

        :param method: HTTP method.
        :param url: Request URL.
        :param headers: Request headers.
        :param fields: Request form data.
        :param timeout: Request timeout.

        :return: The server response.
        """
        match headers.get("Content-Type"):
            case "multipart/form-data":
                data = aiohttp.FormData(fields, quote_fields=False)
            case "application/x-www-form-urlencoded":
                data = urlencode(fields)
            case content_type:
                raise NotImplementedError(f"Unsupported form content type '{content_type}'")

        return await self.__perform_request(
            method=method, url=url, data=data, headers=headers, timeout=timeout, proxy=self.__proxy
        )

    async def request(
        self,
        method: str,
        url: str,
        headers: HTTPHeaderDict,
        body: str | bytes | None,
        timeout: aiohttp.ClientTimeout | None,
    ) -> HTTPResponse:
        """Submit a standard HTTP request.

        :param method: HTTP method.
        :param url: Request URL.
        :param headers: Request headers.
        :param body: Serialized request body.
        :param timeout: Request timeout.

        :return: The server response.
        """
        return await self.__perform_request(
            method=method, url=url, headers=headers, data=body, timeout=timeout, proxy=self.__proxy
        )
