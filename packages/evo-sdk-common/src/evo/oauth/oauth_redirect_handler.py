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
import hashlib
import secrets
import webbrowser
from base64 import urlsafe_b64encode
from types import TracebackType
from urllib.parse import urlencode, urlparse

from evo import logging

from .connector import OAuthConnector
from .data import AccessToken, AnyScopes
from .exceptions import OAuthError

try:
    from aiohttp import web
except ImportError:
    raise ImportError(f"{__name__} could not be used because OAuth dependencies are not installed.")

logger = logging.getLogger("oauth")

__all__ = ["OAuthRedirectHandler"]


class OAuthRedirectHandler:
    """An asynchronous context manager for handling OAuth authorisation redirects. Not thread-safe.

    This context manager starts an HTTP server to handle the OAuth redirect from the server. The user should be
    redirected to the local server to complete the authorisation process. The context manager waits for the
    authorisation to complete and then fetches the access token.
    """

    _REDIRECT_HTML = """
<html>
  <body>
    <h1>Authorization request to Seequent has been completed.</h1>
    <p>You may close this tab or window now.</p>
    <script>setTimeout("window.close()", 2500);</script>
  </body>
</html>
""".encode("UTF-8")

    def __init__(self, oauth_connector: OAuthConnector, redirect_url: str) -> None:
        """
        :param oauth_connector: The OAuth connector to use for authorisation.
        """
        self.__oauth_connector = oauth_connector
        self.__runner: web.AppRunner | None = None  # The HTTP server runner.
        self.__authorisation: asyncio.Future[AccessToken] = asyncio.get_event_loop().create_future()
        self.__redirect_url = redirect_url
        self.__state: str | None = None
        self.__verifier: str | None = None

    @property
    def pending(self) -> bool:
        """Check if the context is still pending.

        True if the context is still pending, False otherwise.

        The context is pending if the token has not been fetched and no errors have occurred.
        """
        return not self.__authorisation.done()

    def __check_server_started(self) -> None:
        """Check if the OAuth server has been started.

        :raises OAuthError: If the server has not been started.
        """
        if self.__runner is None:
            raise OAuthError("OAuth HTTP server not started.")

    async def __aenter__(self) -> OAuthRedirectHandler:
        """Start the OAuth HTTP server."""
        if self.__runner is not None:
            raise OAuthError("OAuth redirect server cannot be reused.")

        logger.debug("Opening connector...")
        await self.__oauth_connector.__aenter__()

        logger.debug("Configuring OAuth HTTP server...")
        app = web.Application(logger=logger)
        uri = urlparse(self.__redirect_url)
        app.add_routes([web.get(uri.path, self.__handle_request)])
        self.__runner = runner = web.AppRunner(app)
        await runner.setup()

        logger.debug("Starting OAuth HTTP server...")
        await web.TCPSite(runner, uri.hostname, uri.port, ssl_context=None).start()

        logger.debug("OAuth HTTP server started.")
        return self

    async def __aexit__(
        self, exc_type: type[Exception] | None, exc_val: Exception | None, exc_tb: TracebackType | None
    ) -> None:
        """Stop the OAuth HTTP server."""
        self.__check_server_started()

        logger.debug("Stopping OAuth HTTP server...")
        await self.__runner.cleanup()
        logger.debug("OAuth HTTP server stopped.")

        logger.debug("Closing connector...")
        await self.__oauth_connector.__aexit__(exc_type, exc_val, exc_tb)

    async def __handle_request(self, request: web.Request) -> web.StreamResponse:
        """Handle the in-browser redirect from the OAuth server.

        This method is called when the user is redirected back to the local server.

        Errors that are encountered by this handler would usually be invisible to other parts of the application, so
        they are logged and stored in the context for later retrieval.
        """
        # This request is already successful. Any further errors will be raised in application code.
        response = web.StreamResponse(status=200, headers={"Content-Type": "text/html"})
        await response.prepare(request)
        await response.write(self._REDIRECT_HTML)
        try:  # Broad exception handling to ensure any errors are logged and stored in the context.
            if "error" in request.query:  # Check for an error response from the OAuth provider.
                title = request.query.getone("error")  # Raises KeyError if `error` is missing.
                detail = request.query.getone("error_description", None)
                raise OAuthError(detail or title)  # Report the more detailed error message if it is available.

            token = await self.get_token(
                request.query.getone("state"),  # Raises KeyError if `state` is missing.
                request.query.getone("code"),  # Raises KeyError if `code` is missing.
            )
            self.__authorisation.set_result(token)
        except Exception as exc:  # noqa: E722
            logger.error("Unable to fetch access token.", exc_info=True)
            self.__authorisation.set_exception(OAuthError(str(exc)).with_traceback(exc.__traceback__))
        finally:  # Always return the response to the web client.
            return response

    async def get_result(self, timeout_seconds: int | float = 60) -> AccessToken:
        """Get the result of the authorisation process.

        This method will block until the authorisation process is complete or the timeout is reached.

        :param timeout_seconds: The maximum time (in seconds) to wait for the authorisation process to complete.

        :return: The access token.

        :raises OAuthError: If the authorisation process times out.
        :raises OAuthError: If an error occurred during the authorisation process.
        """
        self.__check_server_started()
        try:
            return await asyncio.wait_for(self.__authorisation, timeout_seconds)
        except asyncio.TimeoutError:
            raise OAuthError("Timed out waiting for OAuth response.")

    async def login(self, scopes: AnyScopes, timeout_seconds: int | float = 60) -> AccessToken:
        """Authenticate the user and obtain an access token.

        This method will launch a web browser to authenticate the user and obtain an access token.

        :param scopes: The OAuth scopes to request.
        :param timeout_seconds: The maximum time (in seconds) to wait for the authorisation process to complete.

        :return: The access token.

        :raises OAuthError: If the user does not authenticate within the timeout.
        :raises OAuthError: If an error occurred during the authorisation process.
        """
        self.__check_server_started()
        auth_uri = self.create_authorization_url(scopes)
        webbrowser.open(auth_uri)
        return await self.get_result(timeout_seconds)

    def create_authorization_url(self, scopes: AnyScopes) -> str:
        """Create an authorization URL for the given client.

        https://www.oauth.com/oauth2-servers/pkce/authorization-request/

        :param scopes: The scopes to request.

        :return: The authorization URL.
        """
        base_url = self.__oauth_connector.base_uri + self.__oauth_connector.endpoint("authorize")
        challenge, state = self._get_challenge_and_state()
        qs_params = {  # The query string parameters for the authorisation URL.
            "response_type": "code",
            "client_id": self.__oauth_connector.client_id,
            "redirect_uri": self.__redirect_url,
            "scope": str(scopes),
            "state": state,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        }
        return base_url + "?" + urlencode(sorted(qs_params.items()))

    def _get_challenge_and_state(self) -> tuple[str, str]:
        """Generate the PKCE challenge and state for the OAuth request.

        https://www.oauth.com/oauth2-servers/pkce/authorization-request/

        A new state, code verifier, and PKCE challenge are generated each time this method is called to ensure
        that the challenge is unique for each attempt.

        :return: A tuple containing the encoded PKCE challenge and the state.
        """
        self.__state = state = secrets.token_urlsafe(32)  # A unique state to prevent CSRF attacks.
        self.__verifier = verifier = secrets.token_urlsafe(48)  # A unique code verifier for the PKCE challenge.
        code_verifier = verifier.encode("utf-8")
        digest = hashlib.sha256(code_verifier).digest()
        return urlsafe_b64encode(digest).decode("utf-8").rstrip("="), state

    async def get_token(self, state: str, code: str) -> AccessToken:
        """Get an access token from the server.

        https://www.oauth.com/oauth2-servers/access-tokens/authorization-code-request/

        This method should be called by the client's redirect handler with the parameters from the
        authorisation response.

        :param state: The state parameter from the authorisation response.
        :param code: The code parameter from the authorisation response.

        :return: The access token.

        :raises OAuthError: If the state is invalid or the token cannot be fetched.
        """
        logger.debug("Validating state...")
        if state != self.__state:  # The state must match the one generated by this client.
            raise OAuthError("Invalid state.")

        data = {  # The payload to send to the OAuth server in the token request.
            "grant_type": "authorization_code",
            "code": code,
            "code_verifier": self.__verifier,
            "redirect_uri": self.__redirect_url,
        }

        logger.debug("Fetching access token...")
        token = await self.__oauth_connector.fetch_token(data, AccessToken)

        return token
