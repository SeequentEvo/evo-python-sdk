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

from unittest import mock

from parameterized import parameterized

from evo.common import HTTPHeaderDict
from evo.common.test_tools import MockResponse, TestHTTPHeaderDict, long_test
from evo.oauth import AccessToken, AnyScopes, AuthorizationCodeAuthorizer, EvoScopes, OAuthError

from ._helpers import (
    AUTHORIZATION_CODE,
    CLIENT_ID,
    CLIENT_SECRET,
    REDIRECT_URL,
    STATE_TOKEN,
    VERIFIER_TOKEN,
    TestWithOAuthConnector,
    get_access_token,
    patch_urlsafe_tokens,
    patch_webbrowser_open,
)

AUTH_HOSTNAME = ...


class TestAuthorizationCodeAuthorizer(TestWithOAuthConnector):
    def setUp(self) -> None:
        super().setUp()
        self.authorizer = AuthorizationCodeAuthorizer(oauth_connector=self.connector, redirect_url=REDIRECT_URL)
        self.first_token = get_access_token(access_token="first_token")
        self.second_token = get_access_token(access_token="second_token")
        self.transport.request.side_effect = [
            MockResponse(status_code=200, content=self.first_token.model_dump_json(by_alias=True, exclude_unset=True)),
            MockResponse(
                status_code=200,
                content=self.second_token.model_dump_json(by_alias=True, exclude_unset=True),
            ),
        ]

    async def _login(self, scopes: AnyScopes | None = None, timeout_seconds: int | None = None) -> None:
        if scopes is not None:
            self.authorizer._scopes = scopes

        if timeout_seconds is not None:
            await self.authorizer.login(timeout_seconds=timeout_seconds)
        else:
            await self.authorizer.login()

    async def login(
        self,
        scopes: AnyScopes | None = None,
        timeout_seconds: int | None = None,
        authenticate: bool = True,
        state: str = STATE_TOKEN,
        verifier: str = VERIFIER_TOKEN,
    ) -> mock.Mock:
        """Log in and return the mock object that was used to patch the `webbrowser.open` function.

        :param scopes: The OAuth scopes to use in this test. If `None`, the default scopes are used.
        :param timeout_seconds: The timeout to use for the login. If `None`, the timeout parameter is not passed to the
            `login` method.
        :param authenticate: Whether to authenticate the user during the login.
        :param state: The state token to use for the login.
        :param verifier: The verifier token to use for the login.
        """
        if authenticate:
            with (
                patch_webbrowser_open(authenticate=True) as mock_open,
                patch_urlsafe_tokens(state=state, verifier=verifier),
            ):
                await self._login(scopes, timeout_seconds)
        else:
            with (
                patch_webbrowser_open(authenticate=False) as mock_open,
                patch_urlsafe_tokens(state=state, verifier=verifier),
                self.assertRaises(OAuthError),
            ):
                await self._login(scopes, timeout_seconds)

        return mock_open

    def assert_token_equals(self, expected_token: AccessToken | None) -> None:
        """Check that the access token is as expected."""
        actual_token: AccessToken | None = self.authorizer._BaseAuthorizer__token
        self.assertEqual(expected_token, actual_token)

    async def test_login_defaults(self) -> None:
        """Test the login method of the AuthorizationCodeAuthorizer."""
        mock_open = await self.login()
        expected_auth_url = self.get_expected_auth_url(
            state=STATE_TOKEN, verifier=VERIFIER_TOKEN, scopes=EvoScopes.default
        )
        mock_open.assert_called_once_with(expected_auth_url)
        self.assert_fetched_token(
            grant_type="authorization_code",
            code=AUTHORIZATION_CODE,
            code_verifier=VERIFIER_TOKEN,
            redirect_uri=REDIRECT_URL,
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
        )
        self.assert_token_equals(self.first_token)

    async def test_login_custom_scopes(self) -> None:
        """Test the login method of the AuthorizationCodeAuthorizer with custom scopes."""
        mock_open = await self.login(scopes=EvoScopes.all_evo)
        expected_auth_url = self.get_expected_auth_url(
            state=STATE_TOKEN, verifier=VERIFIER_TOKEN, scopes=EvoScopes.all_evo
        )
        mock_open.assert_called_once_with(expected_auth_url)
        self.assert_fetched_token(
            grant_type="authorization_code",
            code=AUTHORIZATION_CODE,
            code_verifier=VERIFIER_TOKEN,
            redirect_uri=REDIRECT_URL,
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
        )
        self.assert_token_equals(self.first_token)

    @parameterized.expand(
        [
            ("no delay", 0),
            ("short delay", 1),
            ("longer delay", 5),
        ]
    )
    @long_test
    async def test_login_custom_timeout(self, _label: str, delay: int) -> None:
        """Test the login method of the AuthorizationCodeAuthorizer with a custom timeout."""
        mock_open = await self.login(timeout_seconds=delay, authenticate=False)
        expected_auth_url = self.get_expected_auth_url(
            state=STATE_TOKEN, verifier=VERIFIER_TOKEN, scopes=EvoScopes.default
        )
        mock_open.assert_called_once_with(expected_auth_url)
        self.assert_token_equals(None)

    @parameterized.expand(
        [
            ("no delay", 0),
            ("short delay", 1),
            ("longer delay", 5),
        ]
    )
    @long_test
    async def test_login_custom_scopes_timeout(self, _label: str, delay: int) -> None:
        """Test the login method of the AuthorizationCodeAuthorizer with custom scopes and a custom timeout."""
        mock_open = await self.login(scopes=EvoScopes.all_evo, timeout_seconds=delay, authenticate=False)
        expected_auth_url = self.get_expected_auth_url(
            state=STATE_TOKEN, verifier=VERIFIER_TOKEN, scopes=EvoScopes.all_evo
        )
        mock_open.assert_called_once_with(expected_auth_url)
        self.assert_token_equals(None)

    async def test_login(self) -> None:
        """Test the login method of the AuthorizationCodeAuthorizer."""
        mock_open = await self.login()
        expected_auth_url = self.get_expected_auth_url(
            state=STATE_TOKEN, verifier=VERIFIER_TOKEN, scopes=EvoScopes.default
        )
        mock_open.assert_called_once_with(expected_auth_url)
        self.assert_fetched_token(
            grant_type="authorization_code",
            code=AUTHORIZATION_CODE,
            code_verifier=VERIFIER_TOKEN,
            redirect_uri=REDIRECT_URL,
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
        )
        self.assert_token_equals(self.first_token)

    async def test_refresh_token(self) -> None:
        """Test the refresh_token method of the AuthorizationCodeAuthorizer."""
        await self.login()
        self.transport.request.reset_mock()
        self.assertTrue(await self.authorizer.refresh_token())
        self.assert_fetched_token(
            grant_type="refresh_token",
            refresh_token=self.first_token.refresh_token,
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
        )
        self.assert_token_equals(self.second_token)

    async def test_assert_refresh_token_fails_before_login_fails(self) -> None:
        """Assert that refreshing the token fails before logging in."""
        with self.assertRaises(OAuthError):
            await self.authorizer.refresh_token()
        self.transport.assert_no_requests()

    async def test_refresh_token_fails_with_http_error(self) -> None:
        """Test that the refresh_token method of the AuthorizationCodeAuthorizer fails when an HTTP error occurs."""
        await self.login()
        self.assert_token_equals(self.first_token)
        self.transport.request.reset_mock()
        self.transport.request.side_effect = [MockResponse(status_code=500)]
        self.assertFalse(await self.authorizer.refresh_token())
        self.assert_fetched_token(
            grant_type="refresh_token",
            refresh_token=self.first_token.refresh_token,
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
        )
        self.assert_token_equals(self.first_token)

    async def test_get_default_headers(self) -> None:
        """Test the get_default_headers method of the AuthorizationCodeAuthorizer."""
        await self.login()
        headers = await self.authorizer.get_default_headers()
        self.assertIs(type(headers), HTTPHeaderDict, "The default headers should be an HTTPHeaderDict.")
        self.assertEqual(
            TestHTTPHeaderDict({"Authorization": f"Bearer {self.first_token.access_token}"}),
            TestHTTPHeaderDict(headers),
        )

    async def test_get_default_headers_before_login_fails(self) -> None:
        """Test that the get_default_headers method of the AuthorizationCodeAuthorizer fails before logging in."""
        with self.assertRaises(OAuthError):
            await self.authorizer.get_default_headers()

    async def test_refresh_token_fails_without_refresh_token(self) -> None:
        """Test refreshing an access token fails if the current token does not have a refresh token"""
        with self.assertRaises(OAuthError):
            await self.authorizer.refresh_token()
        self.transport.assert_no_requests()
