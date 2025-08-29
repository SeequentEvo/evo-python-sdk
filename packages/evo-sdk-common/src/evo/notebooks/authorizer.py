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

import contextlib
import json
import warnings
from collections.abc import AsyncIterator
from typing import Generic, TypeVar

from evo import logging, oauth
from evo.oauth.authorizer import _BaseAuthorizer

from .env import DotEnv

__all__ = [
    "AuthorizationCodeAuthorizer",
]

logger = logging.getLogger("notebooks.oauth")

_TOKEN_KEY = "NotebookOAuth.token"

_T_token = TypeVar("_T_token", bound=oauth.AccessToken)


class _OAuthEnv(Generic[_T_token]):
    def __init__(self, env: DotEnv) -> None:
        self.__dotenv = env

    def get_token(self) -> _T_token | None:
        token_str = self.__dotenv.get(_TOKEN_KEY)
        if token_str is None:
            return None

        try:
            token_dict = json.loads(token_str)
            return oauth.AccessToken.model_validate(token_dict)

        except Exception:
            # The token is invalid.
            raise ValueError(f"Invalid token found in the environment file {_TOKEN_KEY}: {token_str!r}")

    def set_token(self, token: _T_token | None) -> None:
        if token is None:
            new_value = None
        else:
            new_value = token.model_dump_json(by_alias=True, exclude_unset=True)
        self.__dotenv.set(_TOKEN_KEY, new_value)


class _NotebookAuthorizerMixin(_BaseAuthorizer[_T_token]):
    pi_partial_implementation = True  # Indicate to pure-interface that this is a mixin.

    _env: _OAuthEnv[_T_token]

    async def reuse_token(self) -> bool:
        """Attempt to reuse an existing token from the environment file.

        :returns: True if a token was found and reused, False otherwise.
        """
        async with self._mutex:
            if (token := self._env.get_token()) is None:
                return False

            if token.is_expired:
                self._env.set_token(None)
                return False

            return True

    def _get_token(self) -> _T_token | None:
        return self._env.get_token()

    def _update_token(self, new_token: _T_token) -> None:
        super()._update_token(new_token)
        self._env.set_token(new_token)


class AuthorizationCodeAuthorizer(_NotebookAuthorizerMixin[oauth.AccessToken], oauth.AuthorizationCodeAuthorizer):
    """An authorization code authorizer for use in Jupyter notebooks.

    This authorizer is not secure, and should only ever be used in Jupyter notebooks. It stores the access token in the
    environment file, which is not secure. It is intended for use in a development environment only. The environment
    file must not be committed to source control.
    """

    def __init__(
        self,
        oauth_connector: oauth.OAuthConnector,
        redirect_url: str,
        scopes: oauth.AnyScopes,
        env: DotEnv,
    ) -> None:
        """
        :param oauth_connector: The OAuth connector to use for authentication.
        :param redirect_url: The local URL to redirect the user back to after authorisation.
        :param scopes: The OAuth scopes to request.
        :param env: The environment to store the OAuth token in.
        """
        warnings.warn(
            "The evo.notebooks.AuthorizationCodeAuthorizer is not secure, and should only ever be used in Jupyter"
            " notebooks in a private environment."
        )
        super().__init__(oauth_connector=oauth_connector, redirect_url=redirect_url, scopes=scopes)
        self._env = _OAuthEnv(env)

    @contextlib.asynccontextmanager
    async def _unwrap_token(self) -> AsyncIterator[oauth.AccessToken]:
        # Overrides the parent implementation so that we can automatically login at startup.
        async with self._mutex:
            if (token := self._env.get_token()) is None:
                token = await self._handle_login(timeout_seconds=60)
                self._update_token(token)
            yield token

    async def refresh_token(self) -> bool:
        succeeded = await oauth.AuthorizationCodeAuthorizer.refresh_token(self)
        if not succeeded:
            # The refresh token has expired. Clear the token from the environment.
            self._env.set_token(None)
        return succeeded
