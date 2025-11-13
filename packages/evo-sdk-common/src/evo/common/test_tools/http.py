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

import unittest
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any
from unittest import mock
from urllib.parse import urljoin

from ..connector import APIConnector
from ..data import HTTPHeaderDict, HTTPResponse, RequestMethod
from ..interfaces import IAuthorizer, ITransport
from .consts import ACCESS_TOKEN, BASE_URL


class TestHTTPHeaderDict(HTTPHeaderDict):
    """A subclass of HTTPHeaderDict that overrides the __repr__ method to provide a more readable output in unit tests.

    The sole benefit of this class is that it does not hide sensitive information when printed, making it easier to
    visually compare the actual and expected values in unit tests.
    """

    __test__ = False  # Prevent unittest from discovering this class as a test case.

    def __repr__(self) -> str:
        repr_data = {key.title(): value for key, value in self.items()}
        return f"{self.__class__.__name__}({repr_data!r})"


class MockResponse(mock.Mock):
    """Fake HTTPResponse object to be returned by a mocked IHttpTransport"""

    def __init__(
        self,
        status_code: int,
        reason: str | None = None,
        headers: Mapping[str, str] | None = None,
        body: bytes | None = None,
        content: str = "",
    ):
        """
        :param status_code: HTTP status code.
        :param reason: Response reason.
        :param headers: Response headers.
        :param body: Response body, as bytes. Body takes precedence over content.
        :param content: Response body, as a string. Content will be encoded using utf-8.
        """
        super().__init__(spec=HTTPResponse)
        self.status = status_code
        self._content = content
        self.headers = TestHTTPHeaderDict(headers)
        self.data = body if body is not None else content.encode("utf-8")
        self.reason = reason
        self.getheader = mock.Mock(side_effect=self._get_header)
        self.getheaders = mock.Mock(return_value=self.headers.copy())

    def _get_header(self, name: str, default: Any | None = None) -> Any:
        return self.headers.get(name, default)


class AbstractTestRequestHandler(ABC):
    """Base class for a test request handler that can be attached to TestTransport."""

    @abstractmethod
    async def request(
        self,
        method: RequestMethod,
        url: str,
        headers: HTTPHeaderDict | None = None,
        post_params: list[tuple[str, str | bytes]] | None = None,
        body: object | str | bytes | None = None,
        request_timeout: int | float | tuple[int | float, int | float] | None = None,
    ) -> MockResponse:
        """Same as ITransport.request(), but returns a MockResponse object.

        TestTransport attaches this handler to the request method, so that it can be used to simulate complex API
        behaviour in tests.

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

    @staticmethod
    def not_found() -> MockResponse:
        return MockResponse(status_code=404, reason="Not Found")

    @staticmethod
    def bad_request() -> MockResponse:
        return MockResponse(status_code=400, reason="Bad Request")


class TestTransport(mock.AsyncMock):
    """Fake ITransport object to be used in tests"""

    open: mock.AsyncMock
    close: mock.AsyncMock
    request: mock.AsyncMock

    def __init__(self, *, base_url: str = BASE_URL) -> None:
        super().__init__(spec=ITransport)
        self._base_url = base_url
        self.request.return_value = MockResponse(status_code=503)

    def _join_hostname(self, path: str) -> str:
        return urljoin(self._base_url, path.lstrip("/"))

    @contextmanager
    def set_http_response(
        self,
        status_code: int,
        content: str = "",
        reason: str | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> Iterator[MockResponse]:
        """Context manager constructor to temporarily use a different MockResponse with the provided parameters.

        :param status_code: HTTP status code.
        :param content: Response content, as a string. Content will be encoded using utf-8.
        :param reason: Response reason.
        :param headers: Response headers.

        :yields: the constructed MockResponse object.
        """
        old_response = self.request.return_value
        self.request.return_value = new_response = MockResponse(
            status_code=status_code, content=content, reason=reason, headers=headers
        )
        old_side_effect = self.request.side_effect
        self.request.side_effect = None
        yield new_response
        self.request.side_effect = old_side_effect
        self.request.return_value = old_response

    def set_request_handler(self, handler: AbstractTestRequestHandler) -> None:
        """Set the request handler for this transport.

        :param handler: The request handler to be used for all requests.
        """
        self.request.side_effect = handler.request

    def assert_request_made(
        self,
        method: RequestMethod,
        path: str = "",
        headers: Mapping[str, str] | None = None,
        post_params: list[tuple[str, str]] | None = None,
        body: object | str | bytes | None = None,
        request_timeout: int | float | tuple[int | float, int | float] | None = None,
    ) -> None:
        """Assert that last call to ITransport.request() used the specified arguments.

        :param method: HTTP request method.
        :param path: The API path, relative to the base_url.
        :param headers: Http request headers.
        :param body: Request json body, for `application/json`.
        :param post_params: Request post parameters, `application/x-www-form-urlencoded` and `multipart/form-data`.
        :param request_timeout: Timeout setting for this request. If one number provided, it will be total request
            timeout. It can also be a pair (tuple) of (connection, read) timeouts.
        """
        headers = TestHTTPHeaderDict(headers or {})
        self.request.assert_called_with(
            method=method,
            url=self._join_hostname(path),
            headers=headers,
            post_params=post_params,
            body=body,
            request_timeout=request_timeout,
        )

    def assert_any_request_made(
        self,
        method: RequestMethod,
        path: str = "",
        headers: Mapping[str, str] | None = None,
        post_params: list[tuple[str, str]] | None = None,
        body: object | str | bytes | None = None,
        request_timeout: int | float | tuple[int | float, int | float] | None = None,
    ) -> None:
        """Assert that any call to ITransport.request() used the specified arguments.

        :param method: HTTP request method.
        :param path: The API path, relative to the base_url.
        :param headers: Http request headers.
        :param body: Request json body, for `application/json`.
        :param post_params: Request post parameters, `application/x-www-form-urlencoded` and `multipart/form-data`.
        :param request_timeout: Timeout setting for this request. If one number provided, it will be total request
            timeout. It can also be a pair (tuple) of (connection, read) timeouts.
        """
        headers = TestHTTPHeaderDict(headers or {})
        self.request.assert_any_call(
            method=method,
            url=self._join_hostname(path),
            headers=headers,
            post_params=post_params,
            body=body,
            request_timeout=request_timeout,
        )

    def assert_n_requests_made(self, n: int) -> None:
        """Assert that there were a certain number of calls to ITransport.request().

        :param n: Number of expected calls.
        """
        assert self.request.await_count == n

    def assert_no_requests(self) -> None:
        """Assert that ITransport.request() has not been called."""
        self.request.assert_not_called()


class TestAuthorizer(mock.AsyncMock):
    """Fake IAuthorizer object to be used in tests"""

    def __init__(self) -> None:
        super().__init__(spec=IAuthorizer)
        self.default_headers = TestHTTPHeaderDict({"Authorization": f"Bearer {ACCESS_TOKEN}"})
        self.refresh_token.return_value = False
        self.get_default_headers.side_effect = lambda: self.default_headers.copy()

    def set_next_access_token(self, token: str) -> None:
        def refresh_token() -> bool:
            self.default_headers = TestHTTPHeaderDict({"Authorization": f"Bearer {token}"})
            self.refresh_token.side_effect = None
            return True

        self.refresh_token.side_effect = refresh_token


class TestWithConnector(unittest.IsolatedAsyncioTestCase):
    """Base unittest class for testing API calls"""

    def setUp(self) -> None:
        self.transport = TestTransport()
        self.authorizer = TestAuthorizer()
        self.connector = APIConnector(BASE_URL, self.transport, self.authorizer)
        self.universal_headers = TestHTTPHeaderDict({})

    def setup_universal_headers(self, headers: Mapping[str, str]) -> None:
        """Set universal headers that are expected to be sent with every request.

        This method should only be called during test setup (i.e., in setUp method).

        :param headers: The headers to be added to every request.
        """
        self.universal_headers = TestHTTPHeaderDict(headers)

    def _get_expected_headers(self, headers: Mapping[str, str] | None) -> TestHTTPHeaderDict:
        return TestHTTPHeaderDict(
            {
                **self.authorizer.default_headers,
                **self.universal_headers,
                **(self.connector._additional_headers or {}),
                **(headers or {}),
            }
        )

    def assert_request_made(
        self,
        method: RequestMethod,
        path: str = "",
        headers: Mapping[str, str] | None = None,
        post_params: list[tuple[str, str]] | None = None,
        body: object | str | bytes | None = None,
        request_timeout: int | float | tuple[int | float, int | float] | None = None,
    ) -> None:
        """Assert that last call to ITransport.request() used the specified arguments.

        :param method: HTTP request method.
        :param path: The API path, relative to the base_url.
        :param headers: Http request headers.
        :param body: Request json body, for `application/json`.
        :param post_params: Request post parameters, `application/x-www-form-urlencoded` and `multipart/form-data`.
        :param request_timeout: Timeout setting for this request. If one number provided, it will be total request
            timeout. It can also be a pair (tuple) of (connection, read) timeouts.
        """
        self.transport.assert_request_made(
            method=method,
            path=path,
            headers=self._get_expected_headers(headers),
            post_params=post_params,
            body=body,
            request_timeout=request_timeout,
        )

    def assert_any_request_made(
        self,
        method: RequestMethod,
        path: str = "",
        headers: Mapping[str, str] | None = None,
        post_params: list[tuple[str, str]] | None = None,
        body: object | str | bytes | None = None,
        request_timeout: int | float | tuple[int | float, int | float] | None = None,
    ) -> None:
        """Assert that any call to ITransport.request() used the specified arguments.

        :param method: HTTP request method.
        :param path: The API path, relative to the base_url.
        :param headers: Http request headers.
        :param body: Request json body, for `application/json`.
        :param post_params: Request post parameters, `application/x-www-form-urlencoded` and `multipart/form-data`.
        :param request_timeout: Timeout setting for this request. If one number provided, it will be total request
            timeout. It can also be a pair (tuple) of (connection, read) timeouts.
        """
        self.transport.assert_any_request_made(
            method=method,
            path=path,
            headers=self._get_expected_headers(headers),
            post_params=post_params,
            body=body,
            request_timeout=request_timeout,
        )
