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

from __future__ import annotations

import base64
import json
import time

from evo.logging import getLogger
from evo.oauth.exceptions import OAuthError

__all__ = [
    "get_browser_access_token",
    "get_browser_config",
]

logger = getLogger("pyodide.browser")

DEFAULT_TOKEN_STORAGE_KEY = "accessToken"
DEFAULT_CONFIG_URL = "/config.json"


def _is_expired(token: str) -> bool:
    """Best-effort expiry check, so an obviously stale token fails fast with a useful message.

    The signature is not verified and opaque tokens are treated as valid; the service remains the authority.
    """
    try:
        _, payload, _ = token.split(".")
        padded = payload + "=" * (-len(payload) % 4)
        expires_at = json.loads(base64.urlsafe_b64decode(padded))["exp"]
    except Exception:
        logger.debug("Could not read an expiry from the access token.", exc_info=True)
        return False
    return float(expires_at) <= time.time()


def get_browser_access_token(storage_key: str = DEFAULT_TOKEN_STORAGE_KEY) -> str:
    """Read the access token that the hosting page has placed in browser local storage.

    The hosting page is expected to store a JSON document containing an `access_token` field.

    :param storage_key: The local storage key the hosting page stores the token under.

    :returns: The access token.

    :raises OAuthError: If no usable, unexpired token is available.
    """
    try:
        from js import localStorage
    except ImportError:
        raise ImportError("Browser credentials are only available when running in a Pyodide runtime.")

    raw_token = localStorage.getItem(storage_key)
    if not raw_token:
        raise OAuthError(f"No access token found in browser local storage under {storage_key!r}. Sign in first.")

    try:
        access_token = json.loads(raw_token)["access_token"]
    except (ValueError, KeyError, TypeError):
        raise OAuthError(f"The value stored under {storage_key!r} is not a valid access token document.")

    if not isinstance(access_token, str) or not access_token:
        raise OAuthError(f"The value stored under {storage_key!r} does not contain an access token.")

    if _is_expired(access_token):
        raise OAuthError("The access token provided by the host page has expired. Sign in again.")

    return access_token


async def get_browser_config(config_url: str = DEFAULT_CONFIG_URL) -> dict:
    """Read the configuration document published by the hosting page.

    :param config_url: The URL of the configuration document.

    :returns: The parsed configuration, or an empty mapping if it is unavailable.
    """
    from pyodide.http import pyfetch

    try:
        response = await pyfetch(config_url)
        if not response.ok:
            raise OAuthError(f"Unexpected status {response.status} fetching {config_url}.")
        return await response.json()
    except Exception:
        logger.debug(f"Could not load host configuration from {config_url}.", exc_info=True)
        return {}
