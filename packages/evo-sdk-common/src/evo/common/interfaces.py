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

import uuid
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING

from pure_interface import Interface

from .data import Environment, HTTPHeaderDict, HTTPResponse, RequestMethod

if TYPE_CHECKING:
    from .connector import APIConnector

__all__ = ["IAuthorizer", "ICache", "IContext", "IFeedback", "ITransport"]


class ITransport(Interface):
    """Interface for HTTP Transport.

    ITransport is responsible for sending HTTP requests and receiving responses. The open and close methods are
    used to manage the connection state. The request method is used to send an HTTP request and receive a response.

    An internal counter should be incremented when open is called and decremented when close is called. The transport
    should only release resources when the counter reaches zero.
    """

    async def open(self) -> None:
        """Open the HTTP transport.

        This method should be called before sending any requests. ITransport implementations should be reentrant,
        meaning that calling open multiple times should not have any side effects. Resources that are consumed by the
        transport should be retained until the close method is called the same number of times as open.

        Reopening a transport object that has previously been closed should work the same way as opening a new transport
        object.
        """
        ...  # pragma: no cover

    async def close(self) -> None:
        """Close the transport.

        This method should be called when the HTTP transport is no longer needed. Because ITransport implementations
        should be reentrant, calling close does not guarantee that the transport is closed immediately. Instead,
        resources should be retained as long as there is an open call that has not been matched by a close call.
        """
        ...  # pragma: no cover

    async def __aenter__(self) -> ITransport:
        """ITransport should be an asynchronous context manager that opens the transport when entered, and closes it when
        exited.

        This allows the transport to be used in an async with statement.
        """
        ...

    async def __aexit__(
        self, exc_type: type[Exception] | None, exc_val: Exception | None, exc_tb: TracebackType | None
    ) -> None:
        """ITransport should be an asynchronous context manager that opens the transport when entered, and closes it when
        exited.

        This allows the transport to be used in an async with statement.
        """
        ...

    async def request(
        self,
        method: RequestMethod,
        url: str,
        headers: HTTPHeaderDict | None = None,
        post_params: list[tuple[str, str | bytes]] | None = None,
        body: object | str | bytes | None = None,
        request_timeout: int | float | tuple[int | float, int | float] | None = None,
    ) -> HTTPResponse:
        """Send an asynchronous request.

        ITransport implementations *MUST NOT* automatically follow redirects. Redirects have special semantics and will
        be handled by individual clients.

        :param method: HTTP request method.
        :param url: HTTP request url.
        :param headers: Http request headers.
        :param body: Request body.
        :param post_params: Request post parameters, `application/x-www-form-urlencoded` and `multipart/form-data`.
        :param request_timeout: Timeout setting for this request. If one number provided, it will be total request
            timeout. It can also be a pair (tuple) of (connection, read) timeouts.

        :return: The HTTP response object.

        :raise ClientValueError: If `body` and `post_params` are both provided.
        :raise TransportError: If the underlying implementation encounters an error.
        """
        ...  # pragma: no cover


class IAuthorizer(Interface):
    """Interface for authorizing HTTP requests.

    IAuthorizer is responsible for providing the necessary headers to authorize HTTP requests. The get_default_headers
    method should return appropriate authorization headers for the target environment. The refresh_token method will be
    called when authorizing a request fails. Implementations should refresh the token and update the default_headers.

    Ideally, an async mutex should be used to ensure that get_default_headers does not return an expired token while
    refresh_token is running.
    """

    async def get_default_headers(self) -> HTTPHeaderDict:
        """Get the default headers for authorizing HTTP requests.

        Depending on the operating environment, different headers may be required to authorize requests. This interface
        allows the authorizer implementation to provide the necessary headers for authorization based on the environment
        the implementation is intended for.

        :return: A header dictionary to be included in the HTTP request by default.
        """
        ...  # pragma: no cover

    async def refresh_token(self) -> bool:
        """Refresh the authorization token.

        This method will be called when the authorization token has expired. Implementations should refresh the token
        and update the default_headers property with the new token.

        :return: True if the token was successfully refreshed, False otherwise.
        """
        ...  # pragma: no cover


class IFeedback(Interface):
    """An interface for providing live feedback to the user."""

    def progress(self, progress: float, message: str | None = None) -> None:
        """Progress the feedback and update the text to message.

        This can raise an exception to cancel the current operation.

        :param progress: A float between 0 and 1 representing the progress of the operation as a percentage.
        :param message: An optional message to display to the user.
        """
        ...  # pragma: no cover


class ICache(Interface):
    """An interface for managing cache directories for service data.

    The cache may be used to store transient binary data, such as files from File API or parquet files from
    the Geoscience Object API.
    """

    root: Path
    """The absolute path to the root cache directory."""

    def get_location(self, environment: Environment, scope: str) -> Path:
        """Get the cache location for the specified environment and scope.

        :param environment: The environment used to determine the cache location.
        :param scope: The scope used to determine the cache location.

        :returns: The absolute path to the cache location.

        :raises StorageFileExistsError: If the location already exists, and it is not a directory.
        """
        ...

    def clear_cache(self, environment: Environment | None = None, scope: str | None = None) -> None:
        """Clear the cache for the specified environment and scope.

        If the environment and the scope is None, clear the entire cache.

        :param environment: The environment of the cache location. If None, clear the entire cache.
        :param scope: The scope of the cache location. If None, clear the entire cache.

        :raises ValueError: If either environment or scope is None, but not both.
        """
        ...


class IContext(Interface):
    """An interface for providing context information for service clients."""

    def get_environment(self) -> Environment:
        """Gets the Environment associated with this context.

        :return: The Environment.
        :raises ContextError: If the context does not have sufficient information to create an Environment.
        """

    def get_org_id(self) -> uuid.UUID:
        """Gets the organization ID associated with this context.

        :return: The organization ID.
        :raises ContextError: If the context does not have an organization ID.
        """

    def get_connector(self) -> APIConnector:
        """Gets the APIConnector associated with this context.

        :return: The APIConnector.
        :raises ContextError: If the context does not have sufficient information to create an APIConnector.
        """

    def get_cache(self) -> ICache | None:
        """Gets the ICache associated with this context, if any.

        :return: The ICache, or None if no cache is associated with this context.
        """
