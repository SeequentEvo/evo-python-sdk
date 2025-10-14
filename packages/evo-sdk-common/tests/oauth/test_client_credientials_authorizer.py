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

from evo.common.test_tools import MockResponse
from evo.oauth import ClientCredentialsAuthorizer
from evo.oauth.data import AccessToken, EvoScopes

from ._helpers import CLIENT_ID, CLIENT_SECRET, TestWithOAuthConnector, get_access_token


class TestClientCredentialsAuthorizer(TestWithOAuthConnector):
    def setUp(self):
        super().setUp()
        self.authorizer = ClientCredentialsAuthorizer(oauth_connector=self.connector, scopes=EvoScopes.all_evo)
        self.first_token = get_access_token(access_token="first_token")
        self.second_token = get_access_token(access_token="second_token")

        self.transport.request.side_effect = [
            MockResponse(status_code=200, content=self.first_token.model_dump_json(by_alias=True, exclude_unset=True)),
            MockResponse(
                status_code=200,
                content=self.second_token.model_dump_json(by_alias=True, exclude_unset=True),
            ),
        ]

    def assert_token_equals(self, expected_token: AccessToken | None) -> None:
        """Check that the access token is as expected."""
        actual_token: AccessToken | None = self.authorizer._BaseAuthorizer__token
        self.assertEqual(expected_token, actual_token)

    async def test_authorize(self):
        await self.authorizer.authorize()
        self.assert_token_equals(self.first_token)
        self.assert_fetched_token(
            grant_type="client_credentials",
            scope="evo.blocksync evo.discovery evo.file evo.object evo.workspace",
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
        )

    async def test_refresh_token(self):
        await self.authorizer.authorize()

        result = await self.authorizer.refresh_token()

        self.assertTrue(result)
        self.assert_fetched_token(
            grant_type="client_credentials",
            scope="evo.blocksync evo.discovery evo.file evo.object evo.workspace",
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
        )
        self.assert_token_equals(self.second_token)

    async def test_get_default_headers(self):
        await self.authorizer.authorize()

        headers = await self.authorizer.get_default_headers()

        self.assertEqual(headers, {"Authorization": "Bearer first_token"})

    async def test_get_default_headers_authorize_not_explicitly_called(self):
        headers = await self.authorizer.get_default_headers()

        self.assertEqual(headers, {"Authorization": "Bearer first_token"})
