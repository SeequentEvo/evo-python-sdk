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

import unittest

from evo.oauth.authorizer import AccessTokenAuthorizer


class TestAccessTokenAuthorizer(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.authorizer = AccessTokenAuthorizer(access_token="abc-123")

    async def test_get_default_headers(self) -> None:
        headers = await self.authorizer.get_default_headers()
        assert ("Authorization", "Bearer abc-123") in headers.items()

    async def test_refresh_token(self) -> None:
        assert not await self.authorizer.refresh_token()
