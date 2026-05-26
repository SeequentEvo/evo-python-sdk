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

try:
    import aiohttp
    from aiohttp.client_exceptions import ClientError
    from aiohttp.typedefs import StrOrURL
except ImportError:
    raise ImportError("AioTransport cannot be used because the aiohttp library is not installed.")
import asyncio
import json
import re
from types import TracebackType

from evo.common import HTTPHeaderDict, HTTPResponse, RequestMethod
from evo.common.exceptions import ClientValueError, TransportError
from evo.common.interfaces import ITransport
from evo.common.utils import BackoffIncremental, BackoffMethod, Retry
from evo.logging import getLogger

from ._helpers import Config, Context

__all__ = ["AioTransport"]

logger = getLogger("aio.transport")

_RE_JSON = re.compile(r"json", re.IGNORECASE)


class AioTransport(ITransport):
    """A client for managing concurrent connections to multiple APIs with a common configuration.

    See `evo.common.interfaces.ITransport` for more detail.
    """

    def __init__(
        self,
        user_agent: str,
        max_attempts: int = 3,
        backoff_method: BackoffMethod = BackoffIncremental(2),
        num_pools: int = 4,
        verify_ssl: bool = True,
        proxy: StrOrURL | None = None,
        close_grace_period_ms: int = 250,
    ):
        """
        :param user_agent: The value to provide in the `User-Agent` header.
        :param max_attempts: Configure the number of retries to allow before raising a `MaxRetryError`
            exception.
        :param backoff_method: Configure the retry backoff implementation.
        :param num_pools: Number of connection pools to cache before discarding the least recently used pool.
        :param verify_ssl: Verify SSL certificates. This should not be disabled unless you really absolutely have to,
            and never in production environments.
        :param close_grace_period_ms: Grace period (in milliseconds) to wait for connections to close.
        """
        self.__config = Config(
            user_agent=user_agent,
            num_pools=num_pools,
            verify_ssl=verify_ssl,
            retry=Retry(logger=logger, max_attempts=max_attempts, backoff_method=backoff_method),
            proxy=proxy,
            close_grace_period_ms=close_grace_period_ms,
        )
        self.__context: Context | None = None
        self.__mutex = asyncio.Lock()
        self.__num_handles = 0

    def __n_handles(self) -> str:
        """Helper method that returns a string describing the number of open handles."""
        assert self.__mutex.locked(), "This method should only be called while holding the mutex."
        return f"{self.__num_handles} handle{'' if self.__num_handles in {1, -1} else 's'}"

    async def open(self) -> None:
        logger.debug("Opening transport.")
        async with self.__mutex:
            if self.__context is None:
                logger.debug("Creating new context.")
                self.__context = await self.__config.create_context()
            self.__num_handles += 1
            logger.debug(f"Transport has {self.__n_handles()}.")
        logger.debug("Transport opened.")

    async def close(self) -> None:
        logger.debug("Closing transport.")
        async with self.__mutex:
            self.__num_handles -= 1
            logger.debug(f"{self.__n_handles()} remaining.")
            if self.__num_handles <= 0:
                logger.debug("Closing context.")
                context = self.__context
                self.__context = None
                await context.close()
            assert self.__num_handles >= 0, "Number of close calls is greater than number of open calls."
        logger.debug("Transport closed.")

    async def __aenter__(self) -> AioTransport:
        await self.open()
        return self

    async def __aexit__(
        self, exc_type: type[Exception] | None, exc_val: Exception | None, exc_tb: TracebackType | None
    ) -> None:
        await self.close()

    async def __unwrap(self) -> Context:
        logger.debug("Retrieving context.")
        async with self.__mutex:
            if self.__context is None:
                raise TransportError(
                    "Cannot make a request before the transport has been opened, or after it has been closed."
                )
            else:
                logger.debug("Context retrieved.")
                return self.__context

    async def request(
        self,
        method: RequestMethod,
        url: str,
        headers: HTTPHeaderDict | None = None,
        post_params: list[tuple[str, str | bytes]] | None = None,
        body: object | str | bytes | None = None,
        request_timeout: int | float | tuple[int | float, int | float] | None = None,
    ) -> HTTPResponse:
        if post_params is not None and body is not None:
            raise ClientValueError(msg="HTTP body and post parameters cannot be used at the same time.")

        headers = HTTPHeaderDict(headers or {})
        post_params = post_params or []

        if "User-Agent" not in headers:
            headers["User-Agent"] = self.__config.user_agent

        # Resolve timeout value.
        match request_timeout:
            case int() | float():
                timeout = aiohttp.ClientTimeout(total=request_timeout)
            case (sock_connect, sock_read):
                timeout = aiohttp.ClientTimeout(sock_connect=sock_connect, sock_read=sock_read)
            case _:
                timeout = None

        ctx = await self.__unwrap()
        str_method = str(method)
        try:
            match headers.get("Content-Type"), method, body:
                case "application/x-www-form-urlencoded" | "multipart/form-data", _, _:
                    return await ctx.request_form(
                        str_method,
                        url,
                        headers=headers,
                        fields=post_params,
                        timeout=timeout,
                    )
                case _, _, str() | bytes() | bytearray():
                    # Allow any content-type if the body is already serialized.
                    return await ctx.request(str_method, url, headers=headers, body=body, timeout=timeout)
                case content_type, _, _ if content_type is None or _RE_JSON.search(content_type):
                    # Use JSON as the default body encoding.
                    return await ctx.request(
                        str_method,
                        url,
                        headers=headers,
                        body=json.dumps(body) if body else None,
                        timeout=timeout,
                    )
                case _, _, _:
                    raise TransportError(
                        msg="Cannot prepare a request message with the provided arguments. "
                        "Please check that your arguments match the declared content type."
                    )
        except (ClientError, TimeoutError) as e:
            raise TransportError(msg="Could not complete HTTP request", caused_by=e)
