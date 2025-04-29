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
import contextlib
from collections.abc import AsyncIterator
from typing import Generic, TypeVar

from evo import logging
from evo.common.data import HTTPHeaderDict
from evo.common.exceptions import RetryError
from evo.common.interfaces import IAuthorizer

from .data import AccessToken, DeviceFlowResponse, OAuthScopes, UserAccessToken
from .exceptions import OAuthError, OIDCError
from .oauth_redirect_handler import OAuthRedirectHandler
from .oidc import OIDCConnector

__all__ = [
    "AccessTokenAuthorizer",
    "AuthorizationCodeAuthorizer",
    "ClientCredentialsAuthorizer",
    "DeviceFlowAuthorizer",
]

logger = logging.getLogger("oauth")

T = TypeVar("T", bound=AccessToken)


class _BaseAuthorizer(IAuthorizer, Generic[T]):
    pi_partial_implementation = True  # Suppress warning about missing interface methods.

    def __init__(self, oidc_connector: OIDCConnector, scopes: OAuthScopes = OAuthScopes.default) -> None:
        """
        :param oidc_connector: The OIDC connector to use for fetching tokens.
        :param scopes: The OAuth scopes to request.
        """
        self._mutex = asyncio.Lock()

        self._connector = oidc_connector
        self.__token: T | None = None

        assert isinstance(scopes, OAuthScopes), "Scopes must be an instance of OAuthScopes."
        self._scopes: OAuthScopes = scopes

    def _get_token(self) -> T | None:
        """Get the current access token, or None if the token is not available.

        The mutex should be acquired before setting or getting the token.

        :return: The access token, or None if the token is not available.
        """
        assert self._mutex.locked(), "self._mutex should be acquired before getting the token."
        return self.__token

    def _update_token(self, new_token: T) -> None:
        """Update the access token.

        The mutex should be acquired before setting or getting the token.

        :param new_token: The new access token.
        """
        assert self._mutex.locked(), "self._mutex should be acquired before updating the token."
        self.__token = new_token

    @contextlib.asynccontextmanager
    async def _unwrap_token(self) -> AsyncIterator[T]:
        """A context manager that yields the access token if available.

        :raises OAuthError: If the access token is not available.
        """
        async with self._mutex:
            if self._get_token() is None:
                raise OAuthError("Access token is not available.")
            yield self._get_token()

    async def get_default_headers(self) -> HTTPHeaderDict:
        async with self._unwrap_token() as token:
            return HTTPHeaderDict({"Authorization": f"Bearer {token.access_token}"})


class ClientCredentialsAuthorizer(_BaseAuthorizer[AccessToken]):
    """An OAuth authorizer that uses a client credientials to authorize a client.

    The authorizer will automatically refresh the access token when it expires.
    """

    @contextlib.asynccontextmanager
    async def _unwrap_token(self) -> AsyncIterator[T]:
        """A context manager that yields the access token if available, or fetches a new access token if unavailable.

        :raises OAuthError: If the access token cannot be fetched.
        """
        async with self._mutex:
            if self._get_token() is None:
                self._update_token(await self._fetch_token())
            yield self._get_token()

    async def _fetch_token(self) -> AccessToken:
        """Get an access token from the server.

        https://www.oauth.com/oauth2-servers/access-tokens/client-credentials/

        :return: The access token.

        :raises OAuthError: If token cannot be fetched.
        """
        await self._connector.load_config()
        self._validate_configuration()
        data = {  # The payload to send to the OAuth server in the token request.
            "grant_type": "client_credentials",
            "scope": str(self._scopes),
        }

        logger.debug("Fetching access token...")
        return await self._connector.fetch_token(data, AccessToken)

    async def authorize(self) -> None:
        """Get an access token from the server.

        https://www.oauth.com/oauth2-servers/access-tokens/client-credentials/

        :return: The access token.

        :raises OAuthError: If token cannot be fetched.
        """
        async with self._mutex:
            self._update_token(await self._fetch_token())

    async def refresh_token(self) -> bool:
        """Get an access token from the server.

        https://www.oauth.com/oauth2-servers/access-tokens/client-credentials/

        :return: The access token.

        :raises OAuthError: If token cannot be fetched.
        """
        try:
            await self.authorize()
        except Exception:  # noqa
            logger.exception("Failed to refresh the access token.", exc_info=True)
            return False
        else:
            return True

    def _validate_configuration(self):
        """Validate an OIDC configuration against the issuer.

        https://openid.net/specs/openid-connect-discovery-1_0.html#rfc.section.3

        :raises OIDCError: If the configuration is invalid.
        """
        if (
            self._connector.config.grant_types_supported
            and "client_credentials" not in self._connector.config.grant_types_supported
        ):
            raise OIDCError("Authorization provider does not support the 'client_credentials' grant type.")


class AuthorizationCodeAuthorizer(_BaseAuthorizer[UserAccessToken]):
    """An OAuth authorizer that uses a localhost callback to authenticate the user.

    This authorizer launches a local web browser to authenticate the user and obtain an access token. The user is
    required to log in and authorize the application. The authorizer will then use the access token to authorize
    requests.

    The authorizer will automatically refresh the access token when it expires. The refresh token is stored in the
    session and is used to obtain a new access token. If the refresh token expires, the user will need to log in again.
    A mutex is used to ensure that default headers are not returned while the user is logging in or the access token is
    being refreshed.

    This authorizer does not store the access token between sessions. The user will need to log in each time the
    application is started.
    """

    def __init__(
        self, oidc_connector: OIDCConnector, redirect_url: str, scopes: OAuthScopes = OAuthScopes.default
    ) -> None:
        """
        :param oidc_connector: The OIDC connector to use for fetching tokens.
        :param redirect_url: The local URL to redirect the user back to after authorisation.
        :param scopes: The OAuth scopes to request.
        """
        super().__init__(oidc_connector, scopes)
        self._redirect_url = redirect_url

    def _update_token(self, new_token: UserAccessToken) -> None:
        new_token.validate_id_token(issuer=self._connector.issuer, client_id=self._connector.client_id)
        return super()._update_token(new_token)

    async def _handle_login(self, timeout_seconds: int) -> UserAccessToken:
        """Internal method to handle the login process without acquiring the mutex.

        :param timeout_seconds: The maximum time (in seconds) to wait for the authorisation process to complete.

        :return: The new access token.

        :raises OAuthError: If the user does not authenticate within the timeout.
        """
        async with OAuthRedirectHandler(self._connector, self._redirect_url) as handler:
            self._validate_configuration()
            return await handler.login(scopes=self._scopes, timeout_seconds=timeout_seconds)

    async def login(self, timeout_seconds: int = 180) -> None:
        """Authenticate the user and obtain an access token.

        This method will launch a web browser to authenticate the user and obtain an access token.

        :param timeout_seconds: The maximum time (in seconds) to wait for the authorisation process to complete.

        :raises OAuthError: If the user does not authenticate within the timeout.
        :raises OAuthError: If an error occurred during the authorisation process.
        """
        async with self._mutex:
            new_token = await self._handle_login(timeout_seconds)
            self._update_token(new_token)

    async def refresh_token(self) -> bool:
        async with self._unwrap_token() as old_token, self._connector:
            try:
                if old_token.refresh_token is None:
                    raise OAuthError("Refresh token is missing.")

                data = {  # The payload to send to the OAuth server in the token request.
                    "grant_type": "refresh_token",
                    "refresh_token": old_token.refresh_token,
                }
                logger.debug("Refreshing access token...")
                new_token = await self._connector.fetch_token(data, UserAccessToken)
                self._update_token(new_token)
            except Exception:  # noqa
                logger.exception("Failed to refresh the access token.", exc_info=True)
                return False
            else:
                return True

    def _validate_configuration(self):
        """Validate an OIDC configuration against the issuer.

        https://openid.net/specs/openid-connect-discovery-1_0.html#rfc.section.3

        :raises OIDCError: If the configuration is invalid.
        """
        if "code" not in self._connector.config.response_types_supported:
            raise OIDCError("Authorization provider does not support the 'code' response type.")

        if (
            self._connector.config.grant_types_supported
            and "authorization_code" not in self._connector.config.grant_types_supported
        ):
            raise OIDCError("Authorization provider does not support the 'authorization_code' grant type.")


class DeviceFlowAuthorizer(_BaseAuthorizer[AccessToken]):
    """An OAuth authorizer that uses the device flow to authenticate the user.

    https://datatracker.ietf.org/doc/html/rfc8628

    The device flow is a two-step process. The first step is to begin the device flow and obtain a device code. The
    second step is to wait for the user to authorize the device flow. The access token is then obtained and stored
    in the authorizer.

    After receiving a successful authorization response, the client must display the "user_code" and the
    "verification_uri" to the end user and instruct them to visit the URI in a user agent on another device and
    enter the user code.

    The "verification_uri_complete" value, if available, can be used in any non-textual manner that results in a
    browser being used to open the URI. For example, the URI could be placed in a QR code or NFC tag to save the user
    from manually typing it.

    https://datatracker.ietf.org/doc/html/rfc8628#section-3.3.1
    """

    async def _begin_device_flow(self) -> DeviceFlowResponse:
        """Begin the device flow.

        Send a device authorization request to the authorization server.

        :return: The device authorization response.

        :raises OIDCError: If the authorization provider does not support device flow.
        :raises OAuthError: If the device flow cannot be started.
        """
        await self._connector.load_config()
        self._validate_configuration()
        return await self._connector.begin_device_flow(self._scopes)

    async def _wait_for_authorization(self, flow: DeviceFlowResponse) -> None:
        """Wait for the user to authorize the device flow.

        :param flow: The device authorization response.

        :raises OAuthError: If the user does not authorize the device flow within the required time.
        """
        data = {
            "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
            "device_code": flow.device_code,
        }
        try:
            async for handler in flow._retry:
                with handler.suppress_errors():
                    token = await self._connector.fetch_token(data, AccessToken)
        except RetryError:
            raise OAuthError("Failed to authenticate the user.")

        self._update_token(token)

    @contextlib.asynccontextmanager
    async def login(self, timeout_seconds: int = 180) -> AsyncIterator[DeviceFlowResponse]:
        """A context manager to authenticate the user using the device flow.

        The device flow is a two-step process. The first step is to begin the device flow and obtain a device code. The
        second step is to wait for the user to authorize the device flow. The access token is then obtained and stored
        in the authorizer.

        After receiving a successful authorization response, the client must display the "user_code" and the
        "verification_uri" to the end user and instruct them to visit the URI in a user agent on another device and
        enter the user code.

        This authorizer will wait for the user to authorize the device flow upon exiting the context manager.

        https://datatracker.ietf.org/doc/html/rfc8628#section-3.3

        :param timeout_seconds: The maximum time (in seconds) to wait for the authorisation process to complete.
            The OAuth provider may specify a shorter timeout in the device authorization response, in which case that
            value will be used.

        :yield: The device authorization response.

        :raises OIDCError: If the authorization provider does not support device flow.
        :raises OAuthError: If the device flow cannot be started.
        :raises OAuthError: If the user does not authorize the device flow within the required time.
        """
        async with self._mutex:
            logger.debug("Starting device flow...")
            flow = await self._begin_device_flow()
            # Override the expires_in value with the timeout_seconds value if it is less than the expires_in value.
            flow.expires_in = min(flow.expires_in, timeout_seconds)
            logger.debug(f"Verification URI: {flow.verification_uri}")
            logger.debug(f"User code: {flow.user_code}")
            yield flow
            await self._wait_for_authorization(flow)

    async def refresh_token(self) -> bool:
        logger.exception("Unable to refresh the access token because refresh tokens are not supported for device flow.")
        return False

    def _validate_configuration(self):
        if (
            self._connector.config.grant_types_supported
            and "urn:ietf:params:oauth:grant-type:device_code" not in self._connector.config.grant_types_supported
        ):
            raise OIDCError("Authorization provider does not support the 'device_code' grant type.")


class AccessTokenAuthorizer(IAuthorizer):
    def __init__(self, access_token: str):
        """An OAuth authorizer that uses the provided access token to authorize requests.

        This authorizer does not make any attempt to refresh expired tokens. This is handy if you already have an
        access token and are happy to support refreshing it when it expires outside of this interface.
        """
        self._access_token = access_token

    async def refresh_token(self) -> bool:
        logger.debug("AccessTokenAuthorizer does not support refreshing expired tokens.")
        return False

    async def get_default_headers(self) -> HTTPHeaderDict:
        return HTTPHeaderDict({"Authorization": f"Bearer {self._access_token}"})
