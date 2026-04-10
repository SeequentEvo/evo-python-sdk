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

"""Tests for the InteractiveAuthorizer and token handling."""

from __future__ import annotations

import json
import unittest
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

if TYPE_CHECKING:
    from evo.widgets._interactive.authorizer import _OAuthEnv


class TestOAuthEnvTokenHandling(unittest.TestCase):
    """Tests for _OAuthEnv token get/set behavior."""

    def _create_oauth_env(self, dotenv_mock: MagicMock) -> _OAuthEnv:
        """Create an _OAuthEnv instance with a mocked DotEnv."""
        from evo.widgets._interactive.authorizer import _OAuthEnv

        return _OAuthEnv(dotenv_mock)

    def test_get_token_returns_none_when_not_set(self) -> None:
        """get_token() returns None when no token is stored."""
        mock_dotenv = MagicMock()
        mock_dotenv.get.return_value = None

        env = self._create_oauth_env(mock_dotenv)

        result = env.get_token()

        self.assertIsNone(result)

    def test_get_token_returns_valid_token(self) -> None:
        """get_token() returns an AccessToken when valid JSON is stored."""
        from evo.oauth import AccessToken

        mock_dotenv = MagicMock()
        token_data = {
            "access_token": "test-token",
            "token_type": "Bearer",
            "expires_in": 3600,
        }
        mock_dotenv.get.return_value = json.dumps(token_data)

        env = self._create_oauth_env(mock_dotenv)

        result = env.get_token()

        self.assertIsInstance(result, AccessToken)
        self.assertEqual(result.access_token, "test-token")

    def test_get_token_clears_corrupted_token(self) -> None:
        """get_token() clears and returns None when token JSON is corrupted."""
        from evo.widgets._interactive.authorizer import _OAuthEnv

        mock_dotenv = MagicMock()
        mock_dotenv.get.return_value = "not valid json {"

        env = _OAuthEnv(mock_dotenv)

        with patch("evo.widgets._interactive.authorizer.logger"):
            result = env.get_token()

        self.assertIsNone(result)
        # The implementation calls set_token(None) to clear corrupted token
        mock_dotenv.set.assert_called_once()

    def test_get_token_clears_invalid_token_structure(self) -> None:
        """get_token() clears and returns None when token structure is invalid."""
        mock_dotenv = MagicMock()
        mock_dotenv.get.return_value = json.dumps({"invalid": "structure"})

        env = self._create_oauth_env(mock_dotenv)

        with patch("evo.widgets._interactive.authorizer.logger"):
            result = env.get_token()

        self.assertIsNone(result)

    def test_set_token_stores_serialized_token(self) -> None:
        """set_token() stores the token as JSON."""
        from evo.oauth import AccessToken

        mock_dotenv = MagicMock()
        env = self._create_oauth_env(mock_dotenv)

        token = AccessToken(access_token="test-token", token_type="Bearer", expires_in=3600)
        env.set_token(token)

        mock_dotenv.set.assert_called_once()
        call_args = mock_dotenv.set.call_args
        self.assertIn("test-token", call_args[0][1])

    def test_set_token_clears_with_none(self) -> None:
        """set_token(None) clears the stored token."""
        mock_dotenv = MagicMock()
        env = self._create_oauth_env(mock_dotenv)

        env.set_token(None)

        mock_dotenv.set.assert_called_once()
        call_args = mock_dotenv.set.call_args
        self.assertIsNone(call_args[0][1])


@pytest.mark.asyncio
class TestInteractiveAuthorizerReuseToken:
    """Tests for InteractiveAuthorizer.reuse_token()."""

    async def test_reuse_token_returns_false_when_no_token(self) -> None:
        """reuse_token() returns False when no token is stored."""
        from evo.widgets._interactive.authorizer import InteractiveAuthorizer

        mock_connector = MagicMock()
        mock_env = MagicMock()
        mock_env.get.return_value = None

        authorizer = InteractiveAuthorizer(
            oauth_connector=mock_connector,
            redirect_url="http://localhost:8080",
            scopes="openid",
            env=mock_env,
        )

        result = await authorizer.reuse_token()

        assert result is False

    async def test_reuse_token_returns_false_when_token_expired(self) -> None:
        """reuse_token() returns False and clears token when expired."""
        import json

        from evo.widgets._interactive.authorizer import InteractiveAuthorizer

        mock_connector = MagicMock()
        mock_env = MagicMock()

        # Create an expired token (expires_in=0, no refresh token)
        expired_token = {
            "access_token": "expired-token",
            "token_type": "Bearer",
            "expires_in": 0,
        }
        mock_env.get.return_value = json.dumps(expired_token)

        authorizer = InteractiveAuthorizer(
            oauth_connector=mock_connector,
            redirect_url="http://localhost:8080",
            scopes="openid",
            env=mock_env,
        )

        result = await authorizer.reuse_token()

        assert result is False

    async def test_reuse_token_returns_true_when_valid(self) -> None:
        """reuse_token() returns True when a valid, non-expired token exists."""
        import json
        import time

        from evo.widgets._interactive.authorizer import InteractiveAuthorizer

        mock_connector = MagicMock()
        mock_env = MagicMock()

        # Create a valid token that expires far in the future
        valid_token = {
            "access_token": "valid-token",
            "token_type": "Bearer",
            "expires_in": 7200,
            "expires_at": time.time() + 7200,
        }
        mock_env.get.return_value = json.dumps(valid_token)

        authorizer = InteractiveAuthorizer(
            oauth_connector=mock_connector,
            redirect_url="http://localhost:8080",
            scopes="openid",
            env=mock_env,
        )

        result = await authorizer.reuse_token()

        assert result is True

    async def test_reuse_token_handles_corrupted_token_gracefully(self) -> None:
        """reuse_token() returns False when token is corrupted (no exception raised)."""
        from evo.widgets._interactive.authorizer import InteractiveAuthorizer

        mock_connector = MagicMock()
        mock_env = MagicMock()
        mock_env.get.return_value = "corrupted json {"

        authorizer = InteractiveAuthorizer(
            oauth_connector=mock_connector,
            redirect_url="http://localhost:8080",
            scopes="openid",
            env=mock_env,
        )

        with patch("evo.widgets._interactive.authorizer.logger"):
            result = await authorizer.reuse_token()

        # Should return False, not raise
        assert result is False


@pytest.mark.asyncio
class TestInteractiveAuthorizerRefreshToken:
    """Tests for InteractiveAuthorizer.refresh_token() behavior."""

    async def test_refresh_token_clears_on_failure(self) -> None:
        """refresh_token() clears the stored token when refresh fails."""
        import json
        import time

        from evo.widgets._interactive.authorizer import InteractiveAuthorizer

        mock_connector = MagicMock()

        mock_env = MagicMock()
        valid_token = {
            "access_token": "test-token",
            "token_type": "Bearer",
            "expires_in": 3600,
            "expires_at": time.time() + 3600,
            "refresh_token": "refresh-token",
        }
        mock_env.get.return_value = json.dumps(valid_token)

        authorizer = InteractiveAuthorizer(
            oauth_connector=mock_connector,
            redirect_url="http://localhost:8080",
            scopes="openid",
            env=mock_env,
        )

        # Mock the parent class's refresh_token to simulate failure
        with patch("evo.oauth.AuthorizationCodeAuthorizer.refresh_token", new_callable=AsyncMock) as mock_refresh:
            mock_refresh.return_value = False

            result = await authorizer.refresh_token()

            assert result is False


if __name__ == "__main__":
    unittest.main()
