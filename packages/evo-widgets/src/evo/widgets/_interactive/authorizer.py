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

"""OAuth authorizers for interactive notebook widgets."""

import contextlib
import json
import os
import warnings
from collections.abc import AsyncIterator

from evo import logging, oauth

from .env import DotEnv

__all__ = [
    "InteractiveAuthorizer",
]

logger = logging.getLogger("widgets.oauth")

_TOKEN_KEY = "NotebookOAuth.token"

_LOCAL_HOSTNAMES = {"127.0.0.1", "localhost"}


def _should_warn_insecure_notebook_usage() -> bool:
    """Return True if we should emit the insecure-usage warning.

    The user requested that we only warn when *not* running a local Jupyter
    notebook. Because detecting the actual notebook URL/host is unreliable
    without additional heavy dependencies (and highly environment-specific),
    we use the following conservative policy:

    - If we can clearly see indicators of a remote / hosted notebook
      environment, we return True (emit the warning).
    - Otherwise (local or unknown), we return False and suppress the warning.

    This means we might *not* warn in some edge cases that are actually
    remote, but we will avoid spamming users of typical local notebooks.

    We treat the following as *remote*:
    - Presence of well-known JupyterHub / cloud notebook env vars.

    Anything not clearly identified as remote is treated as "local/unknown"
    and will not trigger the warning.
    """

    env = os.environ

    # Common JupyterHub / cloud indicators. This list is intentionally small
    # and conservative; we only want to flag cases that are very likely to be
    # remote or multi-user.
    cloud_indicators = [
        "JUPYTERHUB_API_URL",  # JupyterHub deployments
        "JUPYTERHUB_SERVICE_PREFIX",
        "JUPYTERHUB_HOST",
        "COLAB_GPU",  # Google Colab
        "VSCODE_CWD",  # VS Code remote / web contexts
        "PAPERMILL_EXECUTION_ENV",  # Some managed notebook runners
    ]

    if any(key in env for key in cloud_indicators):
        return True

    return False


class _OAuthEnv:
    def __init__(self, env: DotEnv) -> None:
        self.__dotenv = env

    def get_token(self) -> oauth.AccessToken | None:
        token_str = self.__dotenv.get(_TOKEN_KEY)
        if token_str is None:
            return None

        try:
            token_dict = json.loads(token_str)
            return oauth.AccessToken.model_validate(token_dict)

        except Exception:
            # The token is invalid.
            raise ValueError(f"Invalid token found in the environment file!")

    def set_token(self, token: oauth.AccessToken | None) -> None:
        if token is None:
            new_value = None
        else:
            new_value = token.model_dump_json(by_alias=True, exclude_unset=True)
        self.__dotenv.set(_TOKEN_KEY, new_value)


class InteractiveAuthorizer(oauth.AuthorizationCodeAuthorizer):
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
        if _should_warn_insecure_notebook_usage():
            warnings.warn(
                "The evo.widgets.InteractiveAuthorizer is not secure, and should only ever be used in Jupyter"
                " notebooks in a private environment."
            )
        super().__init__(oauth_connector=oauth_connector, redirect_url=redirect_url, scopes=scopes)
        self._env: _OAuthEnv = _OAuthEnv(env)

    async def reuse_token(self) -> bool:
        """Attempt to reuse an existing token from the environment file.

        :returns: True if a token was found and reused, False otherwise.
        """
        async with self._mutex:
            if (token := self._get_token()) is None:
                return False

            if token.is_expired:
                self._env.set_token(None)
                return False

            return True

    def _get_token(self) -> oauth.AccessToken | None:
        return self._env.get_token()

    def _update_token(self, new_token: oauth.AccessToken) -> None:
        super()._update_token(new_token)
        self._env.set_token(new_token)

    @contextlib.asynccontextmanager
    async def _unwrap_token(self) -> AsyncIterator[oauth.AccessToken]:
        # Overrides the parent implementation so that we can automatically login at startup.
        async with self._mutex:
            if (token := self._get_token()) is None:
                token = await self._handle_login(timeout_seconds=60)
                self._update_token(token)
            yield token

    async def refresh_token(self) -> bool:
        succeeded = await super().refresh_token()
        if not succeeded:
            # The refresh token has expired. Clear the token from the environment.
            self._env.set_token(None)
        return succeeded
