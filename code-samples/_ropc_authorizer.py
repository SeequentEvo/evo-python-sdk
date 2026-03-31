#  Copyright © 2026 Bentley Systems, Incorporated
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Resource Owner Password Credentials (ROPC) authorizer for CI test-user authentication.

This module is intentionally kept in the code-samples directory (not in the SDK
proper) because the ROPC grant is deprecated in OAuth 2.1 and should only be
used for automated testing with dedicated test users.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from typing import TypeVar

from evo import logging
from evo.common.data import HTTPHeaderDict
from evo.common.interfaces import IAuthorizer
from evo.oauth import OAuthConnector
from evo.oauth.data import AccessToken, AnyScopes, EvoScopes, Scopes

logger = logging.getLogger("oauth.ropc")

T = TypeVar("T", bound=AccessToken)


class ResourceOwnerPasswordAuthorizer(IAuthorizer):
    """An OAuth authorizer that uses the Resource Owner Password Credentials (ROPC) grant.

    This authorizer authenticates as a specific user using their username and
    password.  It is intended **only** for automated CI testing with dedicated
    test accounts — do not use it in production or interactive contexts.

    The authorizer will automatically refresh the access token using the refresh
    token when one is available.
    """

    pi_partial_implementation = True  # Suppress warning about missing interface methods.

    def __init__(
        self,
        oauth_connector: OAuthConnector,
        username: str,
        password: str,
        scopes: AnyScopes = EvoScopes.default,
    ) -> None:
        """
        :param oauth_connector: The connector to use for fetching tokens.
        :param username: The test user's username / email.
        :param password: The test user's password.
        :param scopes: The OAuth scopes to request.
        """
        self._mutex = asyncio.Lock()
        self._connector = oauth_connector
        self._username = username
        self._password = password
        self._scopes = Scopes(scopes)
        self.__token: AccessToken | None = None

    # -- token management ---------------------------------------------------

    def _get_token(self) -> AccessToken | None:
        assert self._mutex.locked()
        return self.__token

    def _update_token(self, new_token: AccessToken) -> None:
        assert self._mutex.locked()
        self.__token = new_token

    @contextlib.asynccontextmanager
    async def _unwrap_token(self) -> AsyncIterator[AccessToken]:
        """Yield the current token, fetching one first if necessary."""
        async with self._mutex:
            if self._get_token() is None:
                self._update_token(await self._fetch_token())
            yield self._get_token()  # type: ignore[arg-type]

    # -- ROPC token fetch ---------------------------------------------------

    async def _fetch_token(self) -> AccessToken:
        """Fetch an access token using the ROPC grant."""
        data = {
            "grant_type": "password",
            "username": self._username,
            "password": self._password,
            "scope": str(self._scopes),
        }
        logger.debug("Fetching access token via ROPC grant...")
        return await self._connector.fetch_token(data, AccessToken)

    # -- IAuthorizer interface ----------------------------------------------

    async def get_default_headers(self) -> HTTPHeaderDict:
        async with self._unwrap_token() as token:
            return HTTPHeaderDict({"Authorization": f"Bearer {token.access_token}"})

    async def refresh_token(self) -> bool:
        """Refresh the access token.

        Uses the refresh token if available, otherwise re-authenticates with
        username/password.
        """
        async with self._mutex:
            try:
                old_token = self._get_token()
                if old_token is not None and old_token.refresh_token:
                    data = {
                        "grant_type": "refresh_token",
                        "refresh_token": old_token.refresh_token,
                    }
                    logger.debug("Refreshing ROPC access token via refresh_token...")
                    new_token = await self._connector.fetch_token(data, AccessToken)
                else:
                    logger.debug("No refresh token available, re-authenticating via ROPC...")
                    new_token = await self._fetch_token()
                self._update_token(new_token)
            except Exception:
                logger.exception("Failed to refresh the access token.", exc_info=True)
                return False
            else:
                return True
