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
import hashlib
import hmac
import json
import unittest
from base64 import urlsafe_b64encode
from collections.abc import Iterator
from datetime import datetime
from typing import Any, Literal
from unittest import mock
from urllib.parse import urlencode

from aiohttp.client import ClientSession

from evo.common import RequestMethod
from evo.common.test_tools import TestTransport, utc_datetime
from evo.oauth import AnyScopes, EvoScopes, OAuthConnector
from evo.oauth.data import AccessToken

ACCESS_TOKEN = "TestAccessToken"
BASE_URI = "https://auth.unittest.test"
CLIENT_ID = "TestClientID"
CLIENT_SECRET = "TestClientSecret"
REFRESH_TOKEN = "TestRefreshToken"
EXPIRES_IN = 3600
ISSUED_AT = utc_datetime(2024, 4, 1)

REDIRECT_URL = "http://localhost:1234"
STATE_TOKEN = "TestStateToken"
VERIFIER_TOKEN = "TestVerifierToken"
AUTHORIZATION_CODE = "TestAuthorizationCode"
OAUTH_SCOPES = EvoScopes.all_evo


def _jwt_encode(**data: Any) -> str:
    return urlsafe_b64encode(json.dumps(data).encode("utf-8")).decode("utf-8").rstrip("=")


def _get_jwt(issuer: str, subject: str, audience: str | list[str], expiry: int, issued_at: int) -> str:
    header = _jwt_encode(typ="JWT", alg="HS256")
    payload = _jwt_encode(iss=issuer, sub=subject, aud=audience, exp=expiry, iat=issued_at)
    token = f"{header}.{payload}"
    digest = hmac.new(b"secret", token.encode("utf-8"), "sha256").digest()
    return token + "." + urlsafe_b64encode(digest).decode("utf-8").rstrip("=")


def get_user_access_token(
    access_token: str = ACCESS_TOKEN,
    refresh_token: str | None = REFRESH_TOKEN,
    expires_in: int = EXPIRES_IN,
    issued_at: datetime = ISSUED_AT,
) -> AccessToken:
    return AccessToken(
        token_type="Bearer",
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=expires_in,
        issued_at=issued_at,
    )


def get_access_token(
    access_token: str = ACCESS_TOKEN,
    refresh_token: str | None = REFRESH_TOKEN,
    expires_in: int = EXPIRES_IN,
    issued_at: datetime = ISSUED_AT,
) -> AccessToken:
    return AccessToken(
        token_type="Bearer",
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=expires_in,
        issued_at=issued_at,
    )


@contextlib.contextmanager
def patch_urlsafe_tokens(state: str = STATE_TOKEN, verifier: str = VERIFIER_TOKEN) -> Iterator[None]:
    with mock.patch("secrets.token_urlsafe", side_effect=[state, verifier]):
        yield


async def get_redirect(state: str = STATE_TOKEN, code: str = AUTHORIZATION_CODE) -> str:
    async with ClientSession() as session:
        async with session.get(REDIRECT_URL, params={"state": state, "code": code}) as response:
            response.raise_for_status()
            return await response.text()


@contextlib.contextmanager
def patch_webbrowser_open(authenticate: bool) -> Iterator[mock.Mock]:
    """Patch the `webbrowser.open` function and optionally authenticate the user."""

    def webbrowser_open(_url: str) -> None:
        if authenticate:
            asyncio.ensure_future(get_redirect(state=STATE_TOKEN, code=AUTHORIZATION_CODE))

    with mock.patch("webbrowser.open", side_effect=webbrowser_open) as mock_open:
        yield mock_open


class TestWithOAuthConnector(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.transport = TestTransport(base_url=BASE_URI)
        self.connector = OAuthConnector(
            self.transport,
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
            base_uri=BASE_URI,
        )

    def get_expected_auth_url(
        self, state: str = STATE_TOKEN, verifier: str = VERIFIER_TOKEN, scopes: AnyScopes = EvoScopes.default
    ) -> str:
        base_url = self.connector.base_uri + self.connector.endpoint("authorize")

        # urlsafe base64 encoded sha256 hash of the verifier token.
        expect_challenge = (
            urlsafe_b64encode(
                hashlib.sha256(verifier.encode()).digest(),
            )
            .decode()
            .strip("=")
        )

        query = {
            "response_type": "code",
            "client_id": self.connector.client_id,
            "redirect_uri": REDIRECT_URL,
            "state": state,
            "scope": str(scopes),
            "code_challenge": expect_challenge,
            "code_challenge_method": "S256",
        }
        return base_url + "?" + urlencode(sorted(query.items()))

    def assert_fetched_token(self, path: str = "/connect/token", **data: str) -> None:
        """Assert that a token was fetched with the given data"""
        self.transport.assert_any_request_made(
            RequestMethod.POST,
            path,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            post_params=list(data.items()),
        )


class MockOAuthConnector(mock.AsyncMock):
    def __init__(self, base_uri=BASE_URI, client_id=CLIENT_ID, client_secret=CLIENT_SECRET, **kwargs):
        super().__init__(kwargs=kwargs)
        self._base_uri = base_uri
        self._client_id = client_id
        self._client_secret = client_secret
        self._config = None

    @property
    def base_uri(self):
        return BASE_URI

    @property
    def client_id(self):
        return self._client_id

    def endpoint(self, endpoint_type: Literal["authorize", "token"]) -> str:
        """
        Returns the relevant OAuth endpoint by endpoint type. Possible values: "authorize", "token".
        """
        match endpoint_type:
            case "authorize":
                return "/connect/authorize"
            case "token":
                return "/connect/token"


class TestWithMockOAuthConnector(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.transport = TestTransport(base_url=BASE_URI)
        self.connector = MockOAuthConnector(spec=OAuthConnector)

    def get_expected_auth_url(
        self, state: str = STATE_TOKEN, verifier: str = VERIFIER_TOKEN, scopes: AnyScopes = EvoScopes.default
    ) -> str:
        base_url = self.connector.base_uri + "/connect/authorize"

        # urlsafe base64 encoded sha256 hash of the verifier token.
        expect_challenge = (
            urlsafe_b64encode(
                hashlib.sha256(verifier.encode()).digest(),
            )
            .decode()
            .strip("=")
        )

        query = {
            "response_type": "code",
            "client_id": self.connector.client_id,
            "redirect_uri": REDIRECT_URL,
            "state": state,
            "scope": str(scopes),
            "code_challenge": expect_challenge,
            "code_challenge_method": "S256",
        }
        return base_url + "?" + urlencode(sorted(query.items()))

    def assert_fetched_token(self, path: str = "/connect/token", **data: str) -> None:
        """Assert that a token was fetched with the given data"""
        self.transport.assert_any_request_made(
            RequestMethod.POST,
            path,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            post_params=list(data.items()),
        )
