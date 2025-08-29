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
from unittest import mock
from urllib.parse import parse_qs, urlparse

from parameterized import parameterized

from evo.common.test_tools import long_test
from evo.oauth import EvoScopes, OAuthError, OAuthRedirectHandler

from ._helpers import (
    AUTHORIZATION_CODE,
    REDIRECT_URL,
    STATE_TOKEN,
    VERIFIER_TOKEN,
    TestWithMockOAuthConnector,
    TestWithOAuthConnector,
    get_access_token,
    get_redirect,
    patch_urlsafe_tokens,
    patch_webbrowser_open,
)


class TestOAuthRedirectHandlerWithMockOAuth(TestWithMockOAuthConnector):
    def setUp(self) -> None:
        super().setUp()
        self.handler = OAuthRedirectHandler(oauth_connector=self.connector, redirect_url=REDIRECT_URL)
        self.handler.get_token = mock.AsyncMock(return_value=get_access_token())

    async def authenticate(self) -> None:
        """Authenticate and return the access token."""
        async with self.handler:
            response_text = await get_redirect(state=STATE_TOKEN, code=AUTHORIZATION_CODE)

        self.handler.get_token.assert_called_once_with(STATE_TOKEN, AUTHORIZATION_CODE)
        self.assertEqual(OAuthRedirectHandler._REDIRECT_HTML.decode("utf-8"), response_text)

    async def test_pending_with_success(self) -> None:
        """Test that the handler is pending until authentication is successful"""
        self.assertTrue(self.handler.pending)
        await self.authenticate()
        self.assertFalse(self.handler.pending)

    async def test_pending_with_failed(self) -> None:
        """Test that the handler is pending until authentication fails"""
        self.assertTrue(self.handler.pending)

        class SomeError(Exception): ...

        self.handler.get_token.side_effect = SomeError("Test error")
        await self.authenticate()
        self.assertFalse(self.handler.pending)

        with self.assertRaises(OAuthError):
            await self.handler.get_result()

    async def test_get_result(self) -> None:
        """Test that the handler returns the access token"""
        with self.assertRaises(OAuthError):
            await self.handler.get_result()

        async with self.handler:
            auth = asyncio.create_task(self.handler.get_result())
            redirect = get_redirect()
            self.assertFalse(auth.done())
            result, _ = await asyncio.gather(auth, redirect)
            self.assertTrue(auth.done())

        self.handler.get_token.assert_called_once_with(STATE_TOKEN, AUTHORIZATION_CODE)
        self.assertEqual(get_access_token(), result)

    @parameterized.expand(
        [
            ("no delay", 0),
            ("short delay", 1),
            ("longer delay", 5),
        ]
    )
    @long_test
    async def test_get_result_timeout(self, _label: str, delay: int) -> None:
        """Test that the handler times out waiting for the result."""
        async with self.handler:
            auth = asyncio.create_task(self.handler.get_result(delay))
            self.assertFalse(auth.done())
            with self.assertRaises(OAuthError):
                await auth
            self.assertTrue(auth.done())

    async def test_reentering_context(self) -> None:
        """Test that re-entering the context raises an error."""
        async with self.handler:
            with self.assertRaises(OAuthError):
                async with self.handler:
                    pass

    async def test_exiting_unentered_context(self) -> None:
        """Test that exiting an unentered context raises an error."""
        with self.assertRaises(OAuthError):
            await self.handler.__aexit__(None, None, None)

    async def test_login(self) -> None:
        """Test logging in with the handler."""
        expect_auth_url = "http://unit.test"
        self.handler.create_authorization_url = mock.Mock(return_value=expect_auth_url)

        async with self.handler:
            with patch_webbrowser_open(authenticate=True) as mock_webbrowser_open:
                result = await self.handler.login(EvoScopes.all_evo)

        mock_webbrowser_open.assert_called_once_with(expect_auth_url)
        self.handler.get_token.assert_called_once_with(STATE_TOKEN, AUTHORIZATION_CODE)
        self.assertEqual(get_access_token(), result)

    @parameterized.expand(
        [
            ("no delay", 0),
            ("short delay", 1),
            ("longer delay", 5),
        ]
    )
    @long_test
    async def test_login_timeout(self, _label: str, delay: int) -> None:
        """Test that the handler times out waiting for the user to authenticate."""
        expect_auth_url = "http://unit.test"
        self.handler.create_authorization_url = mock.Mock(return_value=expect_auth_url)

        async with self.handler:
            with patch_webbrowser_open(authenticate=False) as mock_webbrowser_open:
                auth = asyncio.create_task(self.handler.login(EvoScopes.all_evo, delay))
                self.assertFalse(auth.done())
                with self.assertRaises(OAuthError):
                    await auth
                self.assertTrue(auth.done())

        mock_webbrowser_open.assert_called_once_with(expect_auth_url)
        self.handler.get_token.assert_not_called()

    def test_create_authorization_url(self) -> None:
        """Test creating the authorization URL"""
        expected_url = self.get_expected_auth_url(state=STATE_TOKEN, verifier=VERIFIER_TOKEN, scopes=EvoScopes.default)
        with patch_urlsafe_tokens(state=STATE_TOKEN, verifier=VERIFIER_TOKEN):
            actual_url = self.handler.create_authorization_url(EvoScopes.default)

        # Check the auth URL and query parameters.
        self.assertEqual(expected_url, actual_url)

    def test_create_authorization_url_generates_unique_tokens(self) -> None:
        """Test creating the authorization URL generates unique state and verifier tokens."""

        uri_one = self.handler.create_authorization_url(EvoScopes.default)
        query_one = parse_qs(urlparse(uri_one).query)

        uri_two = self.handler.create_authorization_url(EvoScopes.default)
        query_two = parse_qs(urlparse(uri_two).query)

        self.assertNotEqual(query_one["state"], query_two["state"])
        self.assertNotEqual(query_one["code_challenge"], query_two["code_challenge"])


class TestOAuthRedirectHandler(TestWithOAuthConnector):
    def setUp(self) -> None:
        super().setUp()
        self.handler = OAuthRedirectHandler(oauth_connector=self.connector, redirect_url=REDIRECT_URL)

    async def test_get_token_fails_without_state(self) -> None:
        """Test getting an access token fails if the state token has not been generated"""
        with self.assertRaises(OAuthError):
            await self.handler.get_token(AUTHORIZATION_CODE, VERIFIER_TOKEN)
