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

import json
import unittest
from typing import Literal

from parameterized import parameterized

from evo.aio import AioTransport
from evo.common.data import HTTPHeaderDict, RequestMethod
from evo.common.test_tools import BASE_URL, AbstractTestRequestHandler, MockResponse, TestTransport, long_test
from evo.common.utils import BackoffIncremental, BackoffMethod
from evo.oauth import AccessToken, OAuthConnector, OAuthError

RESPONSE_200_WITH_TOKEN = MockResponse(
    status_code=200,
    content=AccessToken(
        token_type="Bearer",
        access_token="not-a-real-token",
    ).model_dump_json(exclude_unset=True),
    headers={"Content-Type": "application/json"},
    reason="OK",
)
RESPONSE_500 = MockResponse(
    status_code=500,
    content=json.dumps(
        {
            "title": "Internal Server Error",
            "detail": "An unexpected error occurred.",
        },
    ),
    headers={"Content-Type": "application/json"},
    reason="Internal Server Error",
)
RESPONSE_502 = MockResponse(502)
RESPONSE_504 = MockResponse(504)


class IteratingTestRequestHandler(AbstractTestRequestHandler):
    def __init__(self, responses: tuple[MockResponse, ...]) -> None:
        self._responses = iter(responses)

    async def request(
        self,
        method: RequestMethod,
        url: str,
        headers: HTTPHeaderDict | None = None,
        post_params: list[tuple[str, str | bytes]] | None = None,
        body: object | str | bytes | None = None,
        request_timeout: int | float | tuple[int | float, int | float] | None = None,
    ) -> MockResponse:
        return next(self._responses)


class TestOAuthConnector(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.transport = TestTransport()

    @parameterized.expand(
        [
            ("authorize", "/connect/authorize"),
            ("token", "/connect/token"),
        ],
    )
    def test_endpoint(self, endpoint_type: Literal["authorize", "token"], expected: str) -> None:
        transport = AioTransport(user_agent="test")
        connector = OAuthConnector(transport, client_id="test")
        assert connector.endpoint(endpoint_type) == expected

    @long_test
    @parameterized.expand(
        [
            (
                3,
                BackoffIncremental(0.02),
                (
                    RESPONSE_504,
                    RESPONSE_504,
                    RESPONSE_200_WITH_TOKEN,
                ),
                3,
                None,
            ),
            (
                3,
                BackoffIncremental(0.02),
                (
                    RESPONSE_502,
                    RESPONSE_502,
                    RESPONSE_200_WITH_TOKEN,
                ),
                3,
                None,
            ),
            (
                5,
                BackoffIncremental(0.02),
                (
                    RESPONSE_504,
                    RESPONSE_504,
                    RESPONSE_200_WITH_TOKEN,
                ),
                3,
                None,
            ),
            (
                5,
                BackoffIncremental(0.02),
                (
                    RESPONSE_500,
                    RESPONSE_500,
                    RESPONSE_500,
                ),
                1,
                OAuthError,
            ),
        ]
    )
    async def test_fetch_token(
        self,
        max_attempts: int,
        backoff_method: BackoffMethod,
        responses: tuple[MockResponse, ...],
        number_of_requests_that_should_be_made: int,
        raises: type[Exception] | None = None,
    ) -> None:
        self.transport.set_request_handler(IteratingTestRequestHandler(responses))
        connector = OAuthConnector(
            self.transport,
            base_uri=BASE_URL,
            client_id="test",
            max_attempts=max_attempts,
            backoff_method=backoff_method,
        )
        if raises:
            with self.assertRaises(raises):
                async with connector:
                    await connector.fetch_token({}, AccessToken)
        else:
            async with connector:
                await connector.fetch_token({}, AccessToken)
        self.transport.assert_n_requests_made(number_of_requests_that_should_be_made)
