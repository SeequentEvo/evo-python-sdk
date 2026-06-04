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
import logging
import socket
import threading
import unittest
from collections.abc import Callable, Mapping
from functools import partial
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from typing import Any
from unittest import mock
from unittest.mock import patch
from urllib.parse import ParseResult, urlparse

from aiohttp import ClientResponse, ClientSession, ClientTimeout, TCPConnector, multipart
from parameterized.parameterized import parameterized

from evo.aio.transport import AioTransport
from evo.common import HTTPHeaderDict, HTTPResponse, RequestMethod
from evo.common.exceptions import ClientValueError, RetryError, TransportError
from evo.common.test_tools import long_test
from evo.common.utils.retry import BackoffLinear
from evo.logging import getLogger

getLogger("aio.transport").setLevel(logging.DEBUG)


class EchoHandler(BaseHTTPRequestHandler):
    """Simple HTTP server request handler for testing."""

    def __init__(self, mock_object: mock.Mock, *args: Any, **kwargs: Any) -> None:
        self.__mock_object = mock_object
        super().__init__(*args, **kwargs)

    def __do_status(self) -> bool:
        self.__mock_object(RequestMethod(self.command), self.path)
        match self.path:
            case "/test/retry":  # Trigger a retry.
                return False
            case _:  # Normal response.
                self.send_response(200)

        return True

    def __do_headers(self) -> None:
        for field, value in self.headers.items():
            self.send_header(field, value)
        self.end_headers()

    def __do_body(self) -> None:
        data = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self.wfile.write(data)

    def handle_one_request(self) -> None:
        """Handle a single HTTP request."""
        # This method is copied from the BaseHTTPRequestHandler class and modified for testing purposes.
        try:
            self.raw_requestline = self.rfile.readline(65537)
            if len(self.raw_requestline) > 65536:
                self.requestline = ""
                self.request_version = ""
                self.command = ""
                self.send_error(HTTPStatus.REQUEST_URI_TOO_LONG)
                return
            if not self.raw_requestline:
                self.close_connection = True
                return
            if not self.parse_request():
                # An error code has been sent, just exit
                return

            # Begin: Custom code.
            try:
                if self.__do_status():
                    self.__do_headers()
                    if self.command != "HEAD":
                        self.__do_body()
            except Exception as e:
                self.send_error(500, str(e))
            # End: Custom code.

            self.wfile.flush()  # actually send the response if not already done.
        except TimeoutError as e:
            # a read or a write timed out. Discard this connection
            self.log_error("Request timed out: %r", e)
            self.close_connection = True
            return


def _mock_response() -> mock.AsyncMock:
    """Mock a 200 response."""
    mock_response = mock.AsyncMock(spec=ClientResponse)
    mock_response.status = 200
    mock_response.read.return_value = b""
    mock_response.reason = "OK"
    mock_response.headers = {}
    return mock_response


class MockSession(mock.AsyncMock):
    def __init__(self) -> None:
        super().__init__(spec_set=ClientSession)
        mock_request_ctx = mock.AsyncMock()
        mock_request_ctx.__aenter__.return_value = _mock_response()
        self.request.return_value = mock_request_ctx

    @classmethod
    def klass(cls) -> mock.Mock:
        return mock.Mock(return_value=cls())


class MockConnector(mock.AsyncMock):
    def __init__(self) -> None:
        super().__init__(spec_set=TCPConnector)

    @classmethod
    def klass(cls) -> mock.Mock:
        return mock.Mock(return_value=cls())


def _get_open_port() -> int:
    s = socket.socket(socket.AF_INET, type=socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    _, port = s.getsockname()
    s.close()
    return port


def _start_server(server: HTTPServer) -> None:
    server.serve_forever()


def _stop_server(server: HTTPServer, thread: threading.Thread) -> None:
    server.shutdown()
    thread.join()


def _create_server(port: int, handler: Callable[[Any], BaseHTTPRequestHandler]) -> Callable[[None], None]:
    server = HTTPServer(("localhost", port), handler)
    server_thread = Thread(target=_start_server, args=(server,))
    server_thread.start()
    return partial(_stop_server, server, server_thread)


# The following test suite takes more time due to starting a local web server. We can probably afford to skip these
# tests while running tests locally.
@long_test
class TestTransportRequest(unittest.IsolatedAsyncioTestCase):
    transport = AioTransport("test-client")
    request_handler = mock.Mock()
    port: int
    url: str

    @classmethod
    def setUpClass(cls) -> None:
        cls.port = _get_open_port()
        cls.url = f"http://localhost:{cls.port}/test"
        cls.addClassCleanup(_create_server(cls.port, partial(EchoHandler, cls.request_handler)))

    def tearDown(self) -> None:
        # Reset the mock object after each test.
        self.request_handler.reset_mock()

    async def asyncSetUp(self) -> None:
        await self.transport.open()

    async def asyncTearDown(self) -> None:
        await self.transport.close()

    def resource_url(self, path: str | None = None) -> ParseResult:
        """Get the full URL for a resource."""
        resolved = self.url.rstrip("/")
        if path is not None:
            resolved += "/" + path.lstrip("/")
        return urlparse(resolved)

    async def request_and_assert(
        self,
        method: RequestMethod,
        relpath: str | None = None,
        headers: Mapping[str, str] | None = None,
        post_params: list[tuple[str, str | bytes]] | None = None,
        body: object | str | bytes | None = None,
        request_timeout: int | float | tuple[int | float, int | float] | None = None,
        expect_hits: int = 1,
    ) -> HTTPResponse:
        """Send a request and assert that the server handled the correct number of requests.

        :param method: HTTP method.
        :param relpath: Relative path to the resource.
        :param headers: Request headers.
        :param post_params: Request post parameters.
        :param body: Request body.
        :param request_timeout: Request timeout.
        :param expect_hits: Expected number of hits on the request handler.

        :return: The server response.
        """
        uri = self.resource_url(relpath)
        try:
            response = await self.transport.request(
                method=method,
                url=uri.geturl(),
                headers=HTTPHeaderDict(headers or {}),
                post_params=post_params,
                body=body,
                request_timeout=request_timeout,
            )
            self.assertEqual(expect_hits, self.request_handler.call_count)
            self.request_handler.assert_has_calls([mock.call(method, uri.path)] * expect_hits)
        finally:
            if expect_hits == 0:
                self.request_handler.assert_not_called()
        return response

    async def test_post_params_encoded(self) -> None:
        """Test that post parameters are correctly encoded."""
        post_params = [("text1", "abc"), ("text2", "foo bar £".encode("utf-8"))]
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        response = await self.request_and_assert(RequestMethod.POST, headers=headers, post_params=post_params)
        self.assertEqual(b"text1=abc&text2=foo+bar+%C2%A3", response.data)

    async def test_post_params_form_data(self) -> None:
        """Test that post parameters are correctly encoded as form data."""
        post_params = [("text1", "abc"), ("text2", "foo bar £".encode("utf-8"))]
        headers = {"Content-Type": "multipart/form-data"}
        writer = multipart.MultipartWriter("form-data", boundary="foo")
        # Make a reproducible boundary code
        with mock.patch("aiohttp.multipart.MultipartWriter") as MultiPartWriter:
            MultiPartWriter.return_value = writer
            response = await self.request_and_assert(RequestMethod.POST, headers=headers, post_params=post_params)
        expected = (
            b"--foo\r\n"
            b"Content-Type: text/plain; charset=utf-8\r\n"
            b'Content-Disposition: form-data; name="text1"\r\n'
            b"\r\n"
            b"abc\r\n"
            b"--foo\r\n"
            b"Content-Type: application/octet-stream\r\n"
            b'Content-Disposition: form-data; name="text2"; filename="text2"\r\n'
            b"\r\n"
            b"foo bar \xc2\xa3\r\n"
            b"--foo--\r\n"
        )
        self.assertEqual(expected, response.data)

    async def test_post_params_bad_content_type(self) -> None:
        """Test that an error is raised when the content type is not supported."""
        headers = {"Content-Type": "application"}
        with self.assertRaises(TransportError) as ctx:
            await self.request_and_assert(RequestMethod.POST, headers=headers, expect_hits=0)
        self.assertIsNone(ctx.exception.caused_by)
        expect_msg = (
            "Cannot prepare a request message with the provided arguments. "
            "Please check that your arguments match the declared content type."
        )
        self.assertEqual(expect_msg, str(ctx.exception))

    async def test_post_params_and_body(self) -> None:
        """Test that an error is raised when both post parameters and a body are provided."""
        with self.assertRaises(ClientValueError):
            await self.request_and_assert(
                RequestMethod.GET,
                post_params={},  # type: ignore
                body="body",
                expect_hits=0,
            )

    async def test_head_request(self) -> None:
        """Test sending a HEAD request."""
        await self.request_and_assert(RequestMethod.HEAD)

    async def test_get_request(self) -> None:
        """Test sending a GET request."""
        response = await self.request_and_assert(RequestMethod.GET)
        expected_headers = {"Host": f"localhost:{self.port}", "User-Agent": "test-client"}
        for field, value in expected_headers.items():
            self.assertEqual(value, response.headers[field])

    async def test_no_content_type(self) -> None:
        """Test sending a request without a content type."""
        body = {"Coordinate": {"Latitude": 90, "Longitude": 90}}
        response = await self.request_and_assert(RequestMethod.PUT, body=body)
        self.assertEqual(b'{"Coordinate": {"Latitude": 90, "Longitude": 90}}', response.data)

    async def test_json_content_type(self) -> None:
        """Test sending a request with a JSON body."""
        headers = {"Content-Type": "application/json"}
        body = {"Coordinate": {"Latitude": 90, "Longitude": 90}}
        response = await self.request_and_assert(RequestMethod.PUT, body=body, headers=headers)
        self.assertEqual(b'{"Coordinate": {"Latitude": 90, "Longitude": 90}}', response.data)

    async def test_str_body(self) -> None:
        """Test sending a request with a string body."""
        body = "Header1,Header2\nText,1.2"
        response = await self.request_and_assert(RequestMethod.PUT, body=body)
        self.assertEqual(body.encode("utf-8"), response.data)

    async def test_bytes_body(self) -> None:
        """Test sending a request with a bytes body."""
        body = "foo bar £".encode("utf-8")
        response = await self.request_and_assert(RequestMethod.PUT, body=body)
        self.assertEqual(body, response.data)

    @parameterized.expand(
        [
            (1, None),
            (1.5, None),
            (4, 1),
            (1, 3),
        ]
    )
    @mock.patch("aiohttp.ClientSession", new_callable=MockSession.klass)
    async def test_timeout(
        self, connect_timeout: float, read_timeout: float | None, mock_session_klass: mock.Mock
    ) -> None:
        """Test setting a timeout on the request.

        This test has consistently caused problems testing against a real server due to varying response times. The
        underlying aiohttp.ClientSession object is now mocked to avoid these issues, and we are to assume that the
        timeout implementation in aiohttp is as advertised.
        """
        # Use a transport object that only allows one attempt.
        transport = AioTransport("test-client", max_attempts=1)
        await transport.open()

        # Mock the session object to raise a timeout error.
        mock_session: MockSession = mock_session_klass.return_value
        mock_session.request.side_effect = TimeoutError()

        # Send the request.
        request_timeout = connect_timeout if read_timeout is None else (connect_timeout, read_timeout)
        with self.assertRaises(TransportError) as ctx:
            await transport.request(
                RequestMethod.GET,
                self.url,
                request_timeout=request_timeout,
            )

        # Check the exception.
        cause = ctx.exception.caused_by
        self.assertIsInstance(cause, RetryError)
        (cause,) = cause.exceptions
        self.assertIsInstance(cause, TimeoutError)

        # Check the request call.
        expect_timeout = (
            ClientTimeout(total=connect_timeout)
            if read_timeout is None
            else ClientTimeout(sock_connect=connect_timeout, sock_read=read_timeout)
        )
        mock_session.request.assert_called_once_with(
            allow_redirects=False,
            method="GET",
            url=self.url,
            headers=HTTPHeaderDict({"User-Agent": "test-client"}),
            data=None,
            timeout=expect_timeout,
            proxy=None,
        )

    async def test_retries(self) -> None:
        """Test that the transport retries the request."""
        # Use a transport object that allows 5 attempts with no delay.
        await self.transport.close()
        self.transport = AioTransport("test-client", max_attempts=5, backoff_method=BackoffLinear(0))
        await self.transport.open()

        # Test that the transport retries the request.
        with self.assertRaises(TransportError) as ctx:
            await self.request_and_assert(RequestMethod.GET, relpath="/retry", expect_hits=5)

        # Check the exception.
        cause = ctx.exception.caused_by
        self.assertIsInstance(cause, RetryError)
        self.assertEqual(5, len(cause.exceptions))

    @mock.patch("aiohttp.ClientSession", new_callable=MockSession.klass)
    @mock.patch("aiohttp.TCPConnector", new_callable=MockConnector.klass)
    async def test_open(self, mock_connector_klass: mock.Mock, mock_session_klass: mock.Mock) -> None:
        """Test opening the transport."""
        mock_connector: MockConnector = mock_connector_klass.return_value
        mock_session: MockSession = mock_session_klass.return_value

        # Create a new transport object that picks up the mocked session and connector.
        transport = AioTransport("test-client", max_attempts=4)

        # Check that the transport is not opened yet.
        mock_connector_klass.assert_not_called()
        mock_session_klass.assert_not_called()
        with self.assertRaises(TransportError) as ctx:
            await transport.request(RequestMethod.GET, self.url)
        self.assertIsNone(ctx.exception.caused_by)
        expect_msg = "Cannot make a request before the transport has been opened, or after it has been closed."
        self.assertEqual(expect_msg, str(ctx.exception))

        # Open the transport and check that the session and connector are created.
        await transport.open()
        mock_connector_klass.assert_called_once_with(ssl=None, limit=4)
        mock_session_klass.assert_called_once_with(
            connector=mock_connector, skip_auto_headers=["Accept", "Accept-Encoding"]
        )
        await transport.request(RequestMethod.GET, self.url)
        mock_session.request.assert_called_once()

    @mock.patch("aiohttp.ClientSession", new_callable=MockSession.klass)
    @mock.patch("aiohttp.TCPConnector", new_callable=MockConnector.klass)
    async def test_aenter(self, mock_connector_klass: mock.Mock, mock_session_klass: mock.Mock) -> None:
        """Test entering the transport context."""
        mock_connector: MockConnector = mock_connector_klass.return_value
        mock_session: MockSession = mock_session_klass.return_value

        # Create a new transport object that picks up the mocked session and connector.
        transport = AioTransport("test-client", max_attempts=4)

        # Check that the transport is not opened yet.
        mock_connector_klass.assert_not_called()
        mock_session_klass.assert_not_called()
        with self.assertRaises(TransportError) as ctx:
            await transport.request(RequestMethod.GET, self.url)
        self.assertIsNone(ctx.exception.caused_by)
        expect_msg = "Cannot make a request before the transport has been opened, or after it has been closed."
        self.assertEqual(expect_msg, str(ctx.exception))

        # Open the transport and check that the session and connector are created.
        async with transport:
            mock_connector_klass.assert_called_once_with(ssl=None, limit=4)
            mock_session_klass.assert_called_once_with(
                connector=mock_connector, skip_auto_headers=["Accept", "Accept-Encoding"]
            )
            await transport.request(RequestMethod.GET, self.url)
            mock_session.request.assert_called_once()

    @mock.patch("asyncio.sleep")
    @mock.patch("aiohttp.ClientSession", new_callable=MockSession.klass)
    async def test_close(self, mock_session_klass: mock.Mock, mock_sleep: mock.Mock) -> None:
        """Test closing the transport."""
        mock_session: MockSession = mock_session_klass.return_value

        # Create a new transport object that picks up the mocked session.
        transport = AioTransport("test-client", close_grace_period_ms=20)
        # Open the transport.
        await transport.open()
        await transport.request(RequestMethod.GET, self.url)
        mock_session.request.assert_called_once()

        # Check that the session is not closed yet.
        mock_session.close.assert_not_called()

        # Close the transport and check that the session is closed.
        await transport.close()
        mock_session.close.assert_called_once()
        with self.assertRaises(TransportError) as ctx:
            await transport.request(RequestMethod.GET, self.url)
        self.assertIsNone(ctx.exception.caused_by)
        expect_msg = "Cannot make a request before the transport has been opened, or after it has been closed."
        self.assertEqual(expect_msg, str(ctx.exception))

        # Check that the correct duration of sleep was called during the close process.
        mock_sleep.assert_called_once_with(20 / 1000)

    @mock.patch("aiohttp.ClientSession", new_callable=MockSession.klass)
    async def test_aexit(self, mock_session_klass: mock.Mock) -> None:
        """Test exiting the transport context."""
        mock_session: MockSession = mock_session_klass.return_value

        # Create a new transport object that picks up the mocked session.
        transport = AioTransport("test-client")
        # Open the transport.
        async with transport:
            await transport.request(RequestMethod.GET, self.url)
            mock_session.request.assert_called_once()

            # Check that the session is not closed yet.
            mock_session.close.assert_not_called()

        # Exiting the context should close the session.
        mock_session.close.assert_called_once()
        with self.assertRaises(TransportError) as ctx:
            await transport.request(RequestMethod.GET, self.url)
        self.assertIsNone(ctx.exception.caused_by)
        expect_msg = "Cannot make a request before the transport has been opened, or after it has been closed."
        self.assertEqual(expect_msg, str(ctx.exception))

    @mock.patch("aiohttp.ClientSession", new_callable=MockSession.klass)
    @mock.patch("aiohttp.TCPConnector", new_callable=MockConnector.klass)
    async def test_reentrant(self, mock_connector_klass: mock.Mock, mock_session_klass: mock.Mock) -> None:
        """Test opening and closing the transport multiple times."""
        mock_connector: MockConnector = mock_connector_klass.return_value
        mock_session: MockSession = mock_session_klass.return_value

        # Create a new transport object that picks up the mocked session and connector.
        transport = AioTransport("test-client", max_attempts=4)

        # Check that the transport is not opened yet.
        mock_connector_klass.assert_not_called()
        mock_session_klass.assert_not_called()
        with self.assertRaises(TransportError) as ctx:
            await transport.request(RequestMethod.GET, self.url)
        self.assertIsNone(ctx.exception.caused_by)
        expect_msg = "Cannot make a request before the transport has been opened, or after it has been closed."
        self.assertEqual(expect_msg, str(ctx.exception))

        # Open the transport and check that the session and connector are created.
        await transport.open()
        mock_connector_klass.assert_called_once_with(ssl=None, limit=4)
        mock_session_klass.assert_called_once_with(
            connector=mock_connector, skip_auto_headers=["Accept", "Accept-Encoding"]
        )
        await transport.request(RequestMethod.GET, self.url)
        mock_session.request.assert_called_once()
        mock_session.close.assert_not_called()

        mock_session.request.reset_mock()  # Reset the call count.

        # Open the transport again and check that the session and connector are not created again.
        await transport.open()
        mock_connector_klass.assert_called_once_with(ssl=None, limit=4)
        mock_session_klass.assert_called_once_with(
            connector=mock_connector, skip_auto_headers=["Accept", "Accept-Encoding"]
        )
        await transport.request(RequestMethod.GET, self.url)
        mock_session.request.assert_called_once()
        mock_session.close.assert_not_called()

        mock_session.request.reset_mock()  # Reset the call count.

        # Close the transport and check that the session is not closed.
        await transport.close()
        mock_session.close.assert_not_called()
        await transport.request(RequestMethod.GET, self.url)
        mock_session.request.assert_called_once()

        # Close the transport again and check that the session is closed.
        await transport.close()
        mock_session.close.assert_called_once()
        with self.assertRaises(TransportError) as ctx:
            await transport.request(RequestMethod.GET, self.url)
        self.assertIsNone(ctx.exception.caused_by)
        expect_msg = "Cannot make a request before the transport has been opened, or after it has been closed."
        self.assertEqual(expect_msg, str(ctx.exception))

    @mock.patch("aiohttp.ClientSession", new_callable=MockSession.klass)
    @mock.patch("aiohttp.TCPConnector", new_callable=MockConnector.klass)
    async def test_reentrant_ctx_manager(self, mock_connector_klass: mock.Mock, mock_session_klass: mock.Mock) -> None:
        """Test context manager reentrancy."""
        mock_connector: MockConnector = mock_connector_klass.return_value
        mock_session: MockSession = mock_session_klass.return_value

        # Create a new transport object that picks up the mocked session and connector.
        transport = AioTransport("test-client", max_attempts=4)

        # Check that the transport is not opened yet.
        mock_connector_klass.assert_not_called()
        mock_session_klass.assert_not_called()
        with self.assertRaises(TransportError) as ctx:
            await transport.request(RequestMethod.GET, self.url)
        self.assertIsNone(ctx.exception.caused_by)
        expect_msg = "Cannot make a request before the transport has been opened, or after it has been closed."
        self.assertEqual(expect_msg, str(ctx.exception))

        # Open the transport and check that the session and connector are created.
        async with transport:
            mock_connector_klass.assert_called_once_with(ssl=None, limit=4)
            mock_session_klass.assert_called_once_with(
                connector=mock_connector, skip_auto_headers=["Accept", "Accept-Encoding"]
            )
            await transport.request(RequestMethod.GET, self.url)
            mock_session.request.assert_called_once()
            mock_session.close.assert_not_called()

            mock_session.request.reset_mock()  # Reset the call count.

            # Open the transport again and check that the session and connector are not created again.
            async with transport:
                mock_connector_klass.assert_called_once_with(ssl=None, limit=4)
                mock_session_klass.assert_called_once_with(
                    connector=mock_connector, skip_auto_headers=["Accept", "Accept-Encoding"]
                )
                await transport.request(RequestMethod.GET, self.url)
                mock_session.request.assert_called_once()
                mock_session.close.assert_not_called()

                mock_session.request.reset_mock()  # Reset the call count.

            # Close the transport and check that the session is not closed.
            mock_session.close.assert_not_called()
            await transport.request(RequestMethod.GET, self.url)
            mock_session.request.assert_called_once()

        # Close the transport again and check that the session is closed.
        mock_session.close.assert_called_once()
        with self.assertRaises(TransportError) as ctx:
            await transport.request(RequestMethod.GET, self.url)
        self.assertIsNone(ctx.exception.caused_by)
        expect_msg = "Cannot make a request before the transport has been opened, or after it has been closed."
        self.assertEqual(expect_msg, str(ctx.exception))

    @mock.patch("aiohttp.ClientSession", new_callable=MockSession.klass)
    async def test_proxy(self, mock_session_klass: mock.Mock) -> None:
        """Test creating AioTransport with a proxy uses the proxy in the request."""
        mock_session: MockSession = mock_session_klass.return_value

        # Create a new transport object that picks up the mocked session.
        transport = AioTransport("test-client", proxy="https://example.com:8080")

        # Open the transport and check that the session is created with the proxy.
        await transport.open()
        await transport.request(RequestMethod.GET, self.url)
        mock_session.request.assert_called_once_with(
            allow_redirects=False,
            method="GET",
            url=self.url,
            headers=HTTPHeaderDict({"User-Agent": "test-client"}),
            data=None,
            timeout=None,
            proxy="https://example.com:8080",
        )

    async def test_bg_task_cancelled_during_close(self):
        """Test for cancelling background tasks mid close."""

        transport = AioTransport(user_agent="test-agent")
        await transport.open()

        original_close_method = transport._AioTransport__context.close
        close_finished_event = asyncio.Event()

        async def context_close_wrapper():
            """Wait indefinitely at the end of the coroutine to allow for cancellation."""
            await original_close_method()
            close_finished_event.set()
            await asyncio.Event().wait()

        with patch.object(transport._AioTransport__context, "close", new=context_close_wrapper):
            # Schedule the close coroutine with a wait after the context is closed
            close_task = asyncio.create_task(transport.close())

            # Wait for the original close coroutine to finish
            await close_finished_event.wait()

            # Cancel the future, this should happen before the context_close_wrapper is able to exit
            close_task.cancel()

            # Context should be cleaned up even though we cancelled
            self.assertIsNone(transport._AioTransport__context)

            # Open the transport again
            await transport.open()

            # Make a request
            await transport.request(RequestMethod.GET, self.url)

            # Check that the context was recreated
            self.assertIsNotNone(transport._AioTransport__context)

            # Close the transport
            await transport.close()

            # Check that the context was closed
            self.assertIsNone(transport._AioTransport__context)
