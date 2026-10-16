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

import json
import sys
import unittest
from unittest import mock

from evo.oauth.exceptions import OAuthError

from ._stubs import FakeFetch, FakeLocalStorage, FakeResponse, expired_jwt, install, valid_jwt

install()

from evo.pyodide import _browser, get_browser_access_token, get_browser_config  # noqa: E402


class TestIsExpired(unittest.TestCase):
    def test_expired_token(self) -> None:
        assert _browser._is_expired(expired_jwt())

    def test_valid_token(self) -> None:
        assert not _browser._is_expired(valid_jwt())

    def test_opaque_token_is_assumed_valid(self) -> None:
        assert not _browser._is_expired("an-opaque-token")

    def test_undecodable_payload_is_assumed_valid(self) -> None:
        assert not _browser._is_expired("header.!!!not-base64!!!.signature")

    def test_token_without_exp_claim_is_assumed_valid(self) -> None:
        assert not _browser._is_expired("header.e30.signature")  # Payload is `{}`.


class TestGetBrowserAccessToken(unittest.TestCase):
    def set_storage(self, items: dict[str, str]) -> None:
        patcher = mock.patch.object(sys.modules["js"], "localStorage", FakeLocalStorage(items))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_returns_stored_token(self) -> None:
        token = valid_jwt()
        self.set_storage({"accessToken": json.dumps({"access_token": token})})
        assert get_browser_access_token() == token

    def test_reads_custom_storage_key(self) -> None:
        token = valid_jwt()
        self.set_storage({"evoToken": json.dumps({"access_token": token})})
        assert get_browser_access_token(storage_key="evoToken") == token

    def test_missing_key(self) -> None:
        self.set_storage({})
        with self.assertRaisesRegex(OAuthError, "No access token found"):
            get_browser_access_token()

    def test_empty_value(self) -> None:
        self.set_storage({"accessToken": ""})
        with self.assertRaisesRegex(OAuthError, "No access token found"):
            get_browser_access_token()

    def test_value_is_not_json(self) -> None:
        self.set_storage({"accessToken": "not json"})
        with self.assertRaisesRegex(OAuthError, "not a valid access token document"):
            get_browser_access_token()

    def test_document_is_missing_access_token(self) -> None:
        self.set_storage({"accessToken": json.dumps({"refresh_token": "abc"})})
        with self.assertRaisesRegex(OAuthError, "not a valid access token document"):
            get_browser_access_token()

    def test_access_token_is_not_a_string(self) -> None:
        self.set_storage({"accessToken": json.dumps({"access_token": 1234})})
        with self.assertRaisesRegex(OAuthError, "does not contain an access token"):
            get_browser_access_token()

    def test_expired_token(self) -> None:
        self.set_storage({"accessToken": json.dumps({"access_token": expired_jwt()})})
        with self.assertRaisesRegex(OAuthError, "has expired"):
            get_browser_access_token()

    def test_opaque_token_is_accepted(self) -> None:
        self.set_storage({"accessToken": json.dumps({"access_token": "an-opaque-token"})})
        assert get_browser_access_token() == "an-opaque-token"


class TestGetBrowserConfig(unittest.IsolatedAsyncioTestCase):
    def patch_fetch(self, *responses: FakeResponse, error: Exception | None = None) -> FakeFetch:
        fetch = FakeFetch(*responses, error=error)
        patcher = mock.patch.object(sys.modules["pyodide.http"], "pyfetch", fetch)
        patcher.start()
        self.addCleanup(patcher.stop)
        return fetch

    async def test_returns_parsed_config(self) -> None:
        fetch = self.patch_fetch(
            FakeResponse(status=200, body=json.dumps({"discovery_url": "https://d.test"}).encode())
        )

        assert await get_browser_config() == {"discovery_url": "https://d.test"}
        assert fetch.calls[0][0] == "/config.json"

    async def test_reads_custom_url(self) -> None:
        fetch = self.patch_fetch(FakeResponse(status=200, body=b"{}"))
        await get_browser_config("/custom.json")
        assert fetch.calls[0][0] == "/custom.json"

    async def test_missing_config_returns_empty_mapping(self) -> None:
        self.patch_fetch(FakeResponse(status=404, status_text="Not Found"))
        assert await get_browser_config() == {}

    async def test_unparseable_config_returns_empty_mapping(self) -> None:
        self.patch_fetch(FakeResponse(status=200, body=b"not json"))
        assert await get_browser_config() == {}

    async def test_fetch_failure_returns_empty_mapping(self) -> None:
        self.patch_fetch(error=RuntimeError("network down"))
        assert await get_browser_config() == {}
