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

import datetime
import json
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from enum import Enum
from typing import Any
from unittest import mock
from uuid import UUID

from parameterized import parameterized
from pydantic import BaseModel

from evo.common import APIConnector, EmptyResponse, HTTPHeaderDict, HTTPResponse, RequestMethod
from evo.common.exceptions import (
    BadRequestException,
    ClientValueError,
    CustomTypedError,
    DefaultTypedError,
    EvoAPIException,
    ForbiddenException,
    GoneException,
    NotFoundException,
    UnauthorizedException,
    UnknownResponseError,
)
from evo.common.test_tools import MockResponse, TestWithConnector, utc_datetime, utc_time


class SampleEnum(Enum):
    STRING_VALUE = "string value"
    INTEGER_VALUE = 1
    FLOAT_VALUE = 1.1
    BYTES_VALUE = b"\xab\xcd"


class SamplePydanticModel(BaseModel):
    str_value: str = "string value"
    int_value: int = 1
    float_value: float = 1.1
    enum_value: SampleEnum = SampleEnum.STRING_VALUE
    datetime_value: datetime.datetime = utc_datetime(2000, 1, 2, 3, 4, 5)
    uuid_value: UUID = UUID(int=1)


class _ResponseType200(BaseModel):
    value: str


class _ResponseType201(BaseModel):
    str_value: str
    int_value: int
    float_value: float
    bool_value: bool
    nested_value: _ResponseType200


class _ResponseType203(_ResponseType200):
    other_value: str


class TestAPIConnector(TestWithConnector):
    def setUp(self) -> None:
        super().setUp()
        self.transport.request.return_value = MockResponse(status_code=200)

    async def test_open(self) -> None:
        """Test that the connector opens the transport when opened."""
        self.transport.open.assert_not_called()
        await self.connector.open()
        self.transport.open.assert_called_once()

    async def test_close(self) -> None:
        """Test that the connector closes the transport when closed."""
        self.transport.close.assert_not_called()
        await self.connector.close()
        self.transport.close.assert_called_once()

    async def test_context_manager(self) -> None:
        """Test that the connector opens and closes the transport when used as a context manager."""
        self.transport.open.assert_not_called()
        self.transport.close.assert_not_called()
        async with self.connector:
            self.transport.open.assert_called_once()
            self.transport.close.assert_not_called()
        self.transport.open.assert_called_once()
        self.transport.close.assert_called_once()

    async def test_call_api_manages_transport(self) -> None:
        """Test that the connector opens and closes the transport when making an API call."""
        self.transport.open.assert_not_called()
        self.transport.close.assert_not_called()
        await self.connector.call_api(method=RequestMethod.GET, resource_path="", response_types_map={"200": str})
        self.transport.open.assert_called_once()
        self.transport.close.assert_called_once()

    async def test_call_api_enters_context_once(self) -> None:
        """Test that the connector uses the same context for retrying an API call."""
        self.authorizer.set_next_access_token("<new-access-token>")

        self.transport.request.side_effect = [
            MockResponse(status_code=401, content="Unauthorized", reason="Unauthorized"),
            MockResponse(status_code=200, content="success"),
        ]
        self.transport.open.assert_not_called()
        self.transport.close.assert_not_called()
        await self.connector.call_api(
            method=RequestMethod.GET,
            resource_path="",
            response_types_map={"200": str},
        )
        self.authorizer.refresh_token.assert_called_once()
        self.transport.open.assert_called_once()
        self.transport.close.assert_called_once()

    @parameterized.expand([(method,) for method in RequestMethod])
    async def test_request_method(self, method: RequestMethod) -> None:
        with self.transport.set_http_response(status_code=200, content="success"):
            result = await self.connector.call_api(method=method, resource_path="", response_types_map={"200": str})
        self.assertEqual("success", result)
        self.assert_request_made(method=method)

    @parameterized.expand(
        [
            (  # Single path param.
                "/service/{single-param}",
                {"single-param": "str-value"},
                "/service/str-value",
            ),
            (  # Multiple path params.
                "/service/{org}/{workspace}/",
                {"org": "org-id", "workspace": UUID(int=1)},
                "/service/org-id/00000000-0000-0000-0000-000000000001/",
            ),
        ]
    )
    async def test_path_params(self, path: str, params: dict[str, str | object], expected_path: str) -> None:
        await self.connector.call_api(
            method=RequestMethod.GET,
            resource_path=path,
            path_params=params,
            response_types_map={"200": bytes},
        )
        self.assert_request_made(method=RequestMethod.GET, path=expected_path)

    @parameterized.expand(
        [
            (
                "Single query param",
                {"param": "value"},
                "?param=value",
            ),
            (
                "Multiple query params in alphanumeric order",
                {"param1": "value1", "param2": "value2", "param3": "value3"},
                "?param1=value1&param2=value2&param3=value3",
            ),
            (
                "Multiple query params in random order",
                {"param3": "value3", "param1": "value1", "param2": "value2"},
                "?param3=value3&param1=value1&param2=value2",
            ),
            (
                "Multi-value query param in alphanumeric order",
                {"param": ["value1", "value2", "value3"]},
                "?param=value1&param=value2&param=value3",
                "multi",
            ),
            (
                "Multi-value query param in random order",
                {"param": ["value3", "value1", "value2"]},
                "?param=value3&param=value1&param=value2",
                "multi",
            ),
            (
                "SSV-value query param in alphanumeric order",
                {"param": ["value1", "value2", "value3"]},
                "?param=value1+value2+value3",
                "ssv",
            ),
            (
                "SSV-value query param in random order",
                {"param": ["value3", "value1", "value2"]},
                "?param=value3+value1+value2",
                "ssv",
            ),
            (
                "Pipe-separated value query param in alphanumeric order",
                {"param": ["value1", "value2", "value3"]},
                "?param=value1%7Cvalue2%7Cvalue3",
                "pipes",
            ),
            (
                "Pipe-separated value query param in random order",
                {"param": ["value3", "value1", "value2"]},
                "?param=value3%7Cvalue1%7Cvalue2",
                "pipes",
            ),
            (
                "CSV-value query param in alphanumeric order",
                {"param": ["value1", "value2", "value3"]},
                "?param=value1%2Cvalue2%2Cvalue3",
                "csv",
            ),
            (
                "CSV-value query param in random order",
                {"param": ["value3", "value1", "value2"]},
                "?param=value3%2Cvalue1%2Cvalue2",
                "csv",
            ),
        ]
    )
    async def test_query_params(
        self,
        _name: str,
        params: dict[str, Any],
        expected_path: str,
        collection_format: str | None = None,
    ) -> None:
        collection_formats = {param: collection_format for param in params.keys()} if collection_format else None
        await self.connector.call_api(
            method=RequestMethod.GET,
            resource_path="",
            query_params=params,
            collection_formats=collection_formats,
            response_types_map={"200": bytes},
        )
        self.assert_request_made(method=RequestMethod.GET, path=expected_path)

    @parameterized.expand(
        [
            (
                "Single header param",
                {"param": "value"},
                {"param": "value"},
            ),
            (
                "Multiple header params",
                {"param1": "value1", "param2": "value2", "param3": "value3"},
                {"param1": "value1", "param2": "value2", "param3": "value3"},
            ),
            (
                "SSV-value header param in alphanumeric order",
                {"param": ["value1", "value2", "value3"]},
                {"param": "value1 value2 value3"},
                "ssv",
            ),
            (
                "SSV-value header param in random order",
                {"param": ["value3", "value1", "value2"]},
                {"param": "value3 value1 value2"},
                "ssv",
            ),
            (
                "Pipe-separated value header param in alphanumeric order",
                {"param": ["value1", "value2", "value3"]},
                {"param": "value1|value2|value3"},
                "pipes",
            ),
            (
                "Pipe-separated value header param in random order",
                {"param": ["value3", "value1", "value2"]},
                {"param": "value3|value1|value2"},
                "pipes",
            ),
            (
                "CSV-value header param in alphanumeric order",
                {"param": ["value1", "value2", "value3"]},
                {"param": "value1,value2,value3"},
                "csv",
            ),
            (
                "CSV-value header param in random order",
                {"param": ["value3", "value1", "value2"]},
                {"param": "value3,value1,value2"},
                "csv",
            ),
            (
                "Single header param as mapping",
                HTTPHeaderDict(param="value"),
                {"param": "value"},
            ),
        ]
    )
    async def test_header_params(
        self,
        _name: str,
        params: Mapping[str, Any],
        expected_headers: Mapping[str, str],
        collection_format: str | None = None,
    ) -> None:
        collection_formats = {param: collection_format for param in params.keys()} if collection_format else None
        await self.connector.call_api(
            method=RequestMethod.GET,
            resource_path="",
            header_params=params,
            collection_formats=collection_formats,
            response_types_map={"200": bytes},
        )
        self.assert_request_made(method=RequestMethod.GET, headers=expected_headers)

    @parameterized.expand(
        [
            (
                "Single header param",
                {"param": "value"},
            ),
            (
                "Multiple header params",
                {"param1": "value1", "param2": "value2", "param3": "value3"},
            ),
        ]
    )
    async def test_additional_headers_in_connector(
        self,
        _name: str,
        headers: Mapping[str, str],
    ) -> None:
        self.connector._additional_headers = HTTPHeaderDict(headers)
        await self.connector.call_api(
            method=RequestMethod.GET,
            resource_path="",
            response_types_map={"200": bytes},
        )
        self.assert_request_made(method=RequestMethod.GET, headers=HTTPHeaderDict(headers))

    async def test_header_params_multi_value_raises_error(self) -> None:
        with self.assertRaises(RuntimeError):
            await self.connector.call_api(
                method=RequestMethod.GET,
                resource_path="",
                header_params={"param": ["value1", "value2", "value3"]},
                collection_formats={"param": "multi"},
                response_types_map={"200": bytes},
            )
            self.transport.request.assert_not_called()

    @parameterized.expand(
        [
            (
                "Single post param",
                {"param": "value"},
                [("param", "value")],
            ),
            (
                "Multiple post params in alphanumeric order",
                {"param1": "value1", "param2": "value2", "param3": "value3"},
                [("param1", "value1"), ("param2", "value2"), ("param3", "value3")],
            ),
            (
                "Multiple post params in random order",
                {"param3": "value3", "param1": "value1", "param2": "value2"},
                [("param3", "value3"), ("param1", "value1"), ("param2", "value2")],
            ),
            (
                "Multi-value post param in alphanumeric order",
                {"param": ["value1", "value2", "value3"]},
                [("param", "value1"), ("param", "value2"), ("param", "value3")],
                "multi",
            ),
            (
                "Multi-value post param in random order",
                {"param": ["value3", "value1", "value2"]},
                [("param", "value3"), ("param", "value1"), ("param", "value2")],
                "multi",
            ),
            (
                "SSV-value post param in alphanumeric order",
                {"param": ["value1", "value2", "value3"]},
                [("param", "value1 value2 value3")],
                "ssv",
            ),
            (
                "SSV-value post param in random order",
                {"param": ["value3", "value1", "value2"]},
                [("param", "value3 value1 value2")],
                "ssv",
            ),
            (
                "Pipe-separated value post param in alphanumeric order",
                {"param": ["value1", "value2", "value3"]},
                [("param", "value1|value2|value3")],
                "pipes",
            ),
            (
                "Pipe-separated value post param in random order",
                {"param": ["value3", "value1", "value2"]},
                [("param", "value3|value1|value2")],
                "pipes",
            ),
            (
                "CSV-value post param in alphanumeric order",
                {"param": ["value1", "value2", "value3"]},
                [("param", "value1,value2,value3")],
                "csv",
            ),
            (
                "CSV-value post param in random order",
                {"param": ["value3", "value1", "value2"]},
                [("param", "value3,value1,value2")],
                "csv",
            ),
        ]
    )
    async def test_post_params(
        self,
        _name: str,
        params: dict[str, Any],
        expected_post_params: list[tuple[str, str]],
        collection_format: str | None = None,
    ) -> None:
        collection_formats = {param: collection_format for param in params.keys()} if collection_format else None
        await self.connector.call_api(
            method=RequestMethod.GET,
            resource_path="",
            post_params=params,
            collection_formats=collection_formats,
            response_types_map={"200": bytes},
        )
        self.assert_request_made(
            method=RequestMethod.GET,
            post_params=expected_post_params,
        )

    async def _parse_response(self, expected_type: Any, response: MockResponse | None = None) -> Any:
        if response is not None:
            self.transport.request.return_value = response
        result = await self.connector.call_api(
            method=RequestMethod.GET,
            resource_path="",
            response_types_map={"200": expected_type},
        )
        self.assert_request_made(method=RequestMethod.GET)
        return result

    @parameterized.expand(
        [
            (
                "empty response",
                "",
                EmptyResponse(status=200, headers=HTTPHeaderDict({"Sample-Header": "sample value"})),
                EmptyResponse,
            ),
            (
                "string",
                "a simple string",
                "a simple string",
                str,
            ),
            (
                "integer",
                "1",
                1,
                int,
            ),
            (
                "float",
                "2.2",
                2.2,
                float,
            ),
            (
                "boolean (true)",
                "true",
                True,
                bool,
            ),
            (
                "boolean (false)",
                "false",
                False,
                bool,
            ),
            (
                "bytes",
                "abcd",
                b"\x61\x62\x63\x64",
                bytes,
            ),
            (
                "time",
                "13:05:04.321+00:00",
                utc_time(hour=13, minute=5, second=4, microsecond=321000),
                datetime.time,
            ),
            (
                "date",
                "2023-05-23",
                datetime.date(year=2023, month=5, day=23),
                datetime.date,
            ),
            (
                "datetime",
                "2023-05-23T13:05:04.321+00:00",
                utc_datetime(
                    year=2023,
                    month=5,
                    day=23,
                    hour=13,
                    minute=5,
                    second=4,
                    microsecond=321000,
                ),
                datetime.datetime,
            ),
            (
                "dict",
                """
                {
                    "string": "a string",
                    "integer": 1,
                    "float": 2.2,
                    "boolean": true,
                    "time": "13:00:00",
                    "date": "2023-05-23",
                    "datetime": "2023-05-23T13:00:00+00:00"
                }
                """,
                {
                    "string": "a string",
                    "integer": 1,
                    "float": 2.2,
                    "boolean": True,
                    "time": "13:00:00",
                    "date": "2023-05-23",
                    "datetime": "2023-05-23T13:00:00+00:00",
                },
                dict,
            ),
            (
                "Basic pydantic model",
                """
                {"value": "basic value"}
                """,
                _ResponseType200(value="basic value"),
                _ResponseType200,
            ),
            (
                "Complex pydantic model",
                """
                {
                    "str_value": "a string",
                    "int_value": 1,
                    "float_value": 2.2,
                    "bool_value": true,
                    "nested_value": {
                        "value": "another string"
                    }
                }
                """,
                _ResponseType201(
                    str_value="a string",
                    int_value=1,
                    float_value=2.2,
                    bool_value=True,
                    nested_value=_ResponseType200(value="another string"),
                ),
                _ResponseType201,
            ),
            (
                "No response type expecting json",
                '{"key": "value"}',
                {"key": "value"},
                None,
            ),
            (
                "No response type expecting string",
                "a simple string",
                "a simple string",
                None,
            ),
        ]
    )
    async def test_response_types(
        self,
        _name: str,
        response_content: str,
        expected_object: Any,
        expected_type: type,
    ):
        mock_response = MockResponse(
            status_code=200, content=response_content, headers={"sample-header": "sample value"}
        )
        actual_obj = await self._parse_response(
            expected_type=expected_type,
            response=mock_response,
        )
        self.assertEqual(expected_object, actual_obj)
        if isinstance(expected_object, EmptyResponse):
            mock_response.getheaders.assert_called_once()

    @parameterized.expand(
        [
            (
                "list of strings",
                '["one", "two", "three", "four"]',
                list[str],
                ["one", "two", "three", "four"],
            ),
            (
                "list of integers",
                "[1, 2, 3, 4]",
                list[int],
                [1, 2, 3, 4],
            ),
            (
                "list of floats",
                "[1.1, 2.2, 3.3, 4.4]",
                list[float],
                [1.1, 2.2, 3.3, 4.4],
            ),
            (
                "list of booleans",
                "[true, true, false, false, true]",
                list[bool],
                [True, True, False, False, True],
            ),
        ]
    )
    async def test_generic_response_types(
        self,
        _name: str,
        response_content: str,
        expected_type: Any,
        expected_value: Any,
    ) -> None:
        actual_value = await self._parse_response(
            expected_type=expected_type,
            response=MockResponse(status_code=200, content=response_content),
        )
        self.assertEqual(expected_value, actual_value)

    @parameterized.expand(
        [
            (
                "matches the more specific union member",
                '{"value": "basic value", "other_value": "extra"}',
                _ResponseType203(value="basic value", other_value="extra"),
            ),
            (
                "falls back to the less specific union member",
                '{"value": "basic value"}',
                _ResponseType200(value="basic value"),
            ),
        ]
    )
    async def test_union_response_type(
        self,
        _name: str,
        response_content: str,
        expected_object: Any,
    ) -> None:
        actual_value = await self._parse_response(
            expected_type=_ResponseType203 | _ResponseType200,
            response=MockResponse(status_code=200, content=response_content),
        )
        self.assertEqual(expected_object, actual_value)
        self.assertIsInstance(actual_value, type(expected_object))

    @parameterized.expand(
        [
            ("Normal response", 200),
            ("Expected generic error response does not raise", 404),
            # The following happens to be the status code used for failed service health checks.
            ("Expected error response does not raise", 503),
        ]
    )
    async def test_raw_response(self, name: str, status_code: int) -> None:
        """Test that the connector always returns the expected response for mapped status codes."""
        with self.transport.set_http_response(status_code=status_code, content=name):
            response = await self.connector.call_api(
                method=RequestMethod.GET,
                resource_path="",
                response_types_map={str(status_code): HTTPResponse},
            )
        self.assert_request_made(method=RequestMethod.GET)
        self.assertIsInstance(response, HTTPResponse)
        self.assertEqual(status_code, response.status)
        self.assertEqual(name, response.data.decode())

    async def test_empty_response_error(self) -> None:
        with (
            self.transport.set_http_response(
                status_code=204, content="some content", headers={"Sample-Header": "sample header value"}
            ),
            self.assertRaises(ClientValueError) as cm,
        ):
            await self.connector.call_api(
                method=RequestMethod.GET,
                resource_path="",
                response_types_map={"204": EmptyResponse},
            )
        expected_message = "Unexpected content with '204' status code"
        actual_message = str(cm.exception)
        self.assertEqual(expected_message, actual_message)

    @parameterized.expand(
        [
            (
                "Invalid response (200)",
                MockResponse(
                    status_code=200,
                    content='{"other_value": "other value"}',
                    reason="OK",
                ),
                ClientValueError,
                "Could not deserialize result: 1 validation error for _ResponseType200",
            ),
            (
                "Unknown response type (203)",
                MockResponse(
                    status_code=203,
                    content="some content",
                    reason="OK",
                ),
                UnknownResponseError,
                "(203) OK\nsome content",
            ),
            (
                "Unknown response type (203) with body",
                MockResponse(
                    status_code=203,
                    content=json.dumps(
                        {
                            "type": "service data",
                            "title": "Service Data Format",
                            "detail": "some data model",
                        }
                    ),
                    reason="OK",
                ),
                UnknownResponseError,
                "(203) OK\n{'type': 'service data', 'title': 'Service Data Format', 'detail': 'some data model'}",
            ),
            (
                "Bad request (400)",
                MockResponse(
                    status_code=400,
                    content=json.dumps({"title": "Bad request title.", "detail": "(optional) additional detail."}),
                    reason="Bad Request",
                ),
                BadRequestException,
                "Error: (400) Bad Request"
                "\nType: about:blank"
                "\nTitle: Bad request title."
                "\nDetail: (optional) additional detail.",
            ),
            (
                "Bad request (401)",
                MockResponse(
                    status_code=401,
                    content=json.dumps({"title": "Unauthorized title."}),
                    reason="Unauthorized",
                ),
                UnauthorizedException,
                "Error: (401) Unauthorized\nType: about:blank\nTitle: Unauthorized title.",
            ),
            (
                "Access denied (403)",
                MockResponse(
                    status_code=403,
                    content=json.dumps({"title": "Access denied title."}),
                    reason="Access Denied",
                ),
                ForbiddenException,
                "Error: (403) Access Denied\nType: about:blank\nTitle: Access denied title.",
            ),
            (
                "Not found (404)",
                MockResponse(
                    status_code=404,
                    content=json.dumps({"title": "Not found title."}),
                    reason="Not Found",
                ),
                NotFoundException,
                "Error: (404) Not Found\nType: about:blank\nTitle: Not found title.",
            ),
            (
                "Conflict (409)",
                MockResponse(
                    status_code=409,
                    content=json.dumps({"title": "Conflict title."}),
                    reason="Conflict",
                ),
                DefaultTypedError,
                "Error: (409) Conflict\nType: about:blank\nTitle: Conflict title.",
            ),
            (
                "Gone (410)",
                MockResponse(
                    status_code=410,
                    content=json.dumps({"title": "Gone title."}),
                    reason="Gone",
                ),
                GoneException,
                "Error: (410) Gone\nType: about:blank\nTitle: Gone title.",
            ),
            (
                "Internal server error (500)",
                MockResponse(
                    status_code=500,
                    content=json.dumps({"title": "Internal server error title."}),
                    reason="Internal Server Error",
                ),
                DefaultTypedError,
                "Error: (500) Internal Server Error\nType: about:blank\nTitle: Internal server error title.",
            ),
            (
                "Bad gateway (502)",
                MockResponse(
                    status_code=502,
                    content=json.dumps({"title": "Bad gateway title."}),
                    reason="Bad Gateway",
                ),
                DefaultTypedError,
                "Error: (502) Bad Gateway\nType: about:blank\nTitle: Bad gateway title.",
            ),
            (
                "Internal server error (500) with specific type id",
                MockResponse(
                    status_code=500,
                    content=json.dumps(
                        {
                            "title": "Internal server error title.",
                            "type": "https://specific.unittest.test/errors/internal-server-error",
                        }
                    ),
                    reason="Internal Server Error",
                ),
                DefaultTypedError,
                "Error: (500) Internal Server Error"
                "\nType: https://specific.unittest.test/errors/internal-server-error"
                "\nTitle: Internal server error title.",
            ),
            (
                "Internal server error (500) with other type id",
                MockResponse(
                    status_code=500,
                    content=json.dumps(
                        {
                            "title": "Internal server error title.",
                            "type": "https://other.unittest.test/errors/other/internal-server-error",
                        }
                    ),
                    reason="Internal Server Error",
                ),
                DefaultTypedError,
                "Error: (500) Internal Server Error"
                "\nType: https://other.unittest.test/errors/other/internal-server-error"
                "\nTitle: Internal server error title.",
            ),
            (
                "Unauthorized (401) with specific type id",
                MockResponse(
                    status_code=401,
                    content=json.dumps(
                        {"title": "Unauthorized title.", "type": "https://specific.unittest.test/errors/unauthorized"}
                    ),
                    reason="Unauthorized",
                ),
                UnauthorizedException,
                "Error: (401) Unauthorized"
                "\nType: https://specific.unittest.test/errors/unauthorized"
                "\nTitle: Unauthorized title.",
            ),
            (
                "Unauthorized (401) with other type id",
                MockResponse(
                    status_code=401,
                    content=json.dumps(
                        {
                            "title": "Unauthorized title.",
                            "type": "https://other.unittest.test/errors/other/unauthorized",
                        }
                    ),
                    reason="Unauthorized",
                ),
                UnauthorizedException,
                "Error: (401) Unauthorized"
                "\nType: https://other.unittest.test/errors/other/unauthorized"
                "\nTitle: Unauthorized title.",
            ),
        ]
    )
    async def test_error_response(
        self,
        _name: str,
        response: MockResponse,
        exc_type: type[EvoAPIException],
        expected_message: str,
    ) -> None:
        with self.assertRaises(exc_type) as cm:
            await self._parse_response(expected_type=_ResponseType200, response=response)
        actual_exc_type = type(cm.exception)
        self.assertIs(exc_type, actual_exc_type)
        actual_message = str(cm.exception)
        self.assertIn(expected_message, actual_message)

    @parameterized.expand(
        [
            # If obj is None return None.
            ("no value", None, None),
            # If obj is an Enum sanitize the value.
            ("enum string value", SampleEnum.STRING_VALUE, "string value"),
            ("enum integer value", SampleEnum.INTEGER_VALUE, 1),
            ("enum float value", SampleEnum.FLOAT_VALUE, 1.1),
            ("enum bytes value", SampleEnum.BYTES_VALUE, b"\xab\xcd"),
            # If obj is a primitive return directly.
            ("string", "string value", "string value"),
            ("integer", 1, 1),
            ("float", 1.1, 1.1),
            ("boolean", True, True),
            ("bytes", b"\xab\xcd", b"\xab\xcd"),
            ("boolean", True, True),
            # If obj is a date or datetime convert to string in iso8601 format.
            ("datetime", True, True),
            # If obj is a UUID convert to string.
            ("uuid", UUID(int=1), "00000000-0000-0000-0000-000000000001"),
            # If obj is a list or tuple, sanitize each element.
            (
                "list",
                [
                    None,
                    SampleEnum.STRING_VALUE,
                    "string value",
                    1,
                    1.1,
                    True,
                    b"\xab\xcd",
                    utc_datetime(2000, 1, 2, 3, 4, 5),
                ],
                [
                    None,
                    "string value",
                    "string value",
                    1,
                    1.1,
                    True,
                    b"\xab\xcd",
                    "2000-01-02T03:04:05+00:00",
                ],
            ),
            (
                "tuple",
                (
                    None,
                    SampleEnum.STRING_VALUE,
                    "string value",
                    1,
                    1.1,
                    True,
                    b"\xab\xcd",
                    utc_datetime(2000, 1, 2, 3, 4, 5),
                ),
                (
                    None,
                    "string value",
                    "string value",
                    1,
                    1.1,
                    True,
                    b"\xab\xcd",
                    "2000-01-02T03:04:05+00:00",
                ),
            ),
            # If obj is a dict, sanitize the dict.
            (
                "dict",
                {
                    "none": None,
                    "enum": SampleEnum.STRING_VALUE,
                    "string": "string value",
                    "int": 1,
                    "float": 1.1,
                    "bool": True,
                    "bytes": b"\xab\xcd",
                    "datetime": utc_datetime(2000, 1, 2, 3, 4, 5),
                },
                {
                    "none": None,
                    "enum": "string value",
                    "string": "string value",
                    "int": 1,
                    "float": 1.1,
                    "bool": True,
                    "bytes": b"\xab\xcd",
                    "datetime": "2000-01-02T03:04:05+00:00",
                },
            ),
            # If obj is an API model, convert to dict.
            (
                "api model",
                SamplePydanticModel(),
                json.loads(SamplePydanticModel().model_dump_json(by_alias=True, exclude_unset=True)),
            ),
        ]
    )
    def test_sanitization(self, _name: str, input_value: Any, expected_value: Any) -> None:
        expected_type = type(expected_value)
        actual_value = APIConnector._sanitize_for_serialization(input_value)
        self.assertIsInstance(actual_value, expected_type)
        self.assertEqual(actual_value, expected_value)

    async def test_refresh_on_auth_error(self) -> None:
        """Test that the connector refreshes the access token and retries the request on an auth error."""
        old_headers = self.authorizer.default_headers.copy()
        self.authorizer.set_next_access_token("<new-access-token>")

        self.transport.request.side_effect = [
            MockResponse(status_code=401, content="Unauthorized", reason="Unauthorized"),
            MockResponse(status_code=200, content="success"),
        ]
        response = await self.connector.call_api(
            method=RequestMethod.GET,
            resource_path="",
            response_types_map={"200": str},
        )
        self.assertEqual("success", response)
        self.authorizer.refresh_token.assert_called_once()
        self.transport.assert_any_request_made(method=RequestMethod.GET, headers=old_headers)

        new_headers = self.authorizer.default_headers.copy()
        self.transport.assert_request_made(method=RequestMethod.GET, headers=new_headers)

    async def test_refresh_on_auth_error_fails(self) -> None:
        """Test that the connector raises an exception and does not retry the request if refreshing the token fails."""
        self.transport.request.side_effect = [
            MockResponse(status_code=401, content="Unauthorized", reason="Unauthorized"),
            MockResponse(status_code=200, content="success"),
        ]
        with self.assertRaises(UnauthorizedException):
            await self.connector.call_api(
                method=RequestMethod.GET,
                resource_path="",
                response_types_map={"200": str},
            )
        self.authorizer.refresh_token.assert_called_once()
        self.transport.request.assert_called_once()
        self.transport.assert_any_request_made(method=RequestMethod.GET, headers=self.authorizer.default_headers.copy())

    @contextmanager
    def __temp_register_custom_error(self) -> Iterator[None]:
        with mock.patch.object(CustomTypedError, "_CustomTypedError__CONCRETE_TYPES", dict()):
            yield

    async def test_custom_typed_error_mapped(self) -> None:
        mock_response = MockResponse(
            status_code=422,
            content=json.dumps(
                {
                    "type": "https://other.unittest.test/errors/other/validation",
                    "title": "Validation Error",
                }
            ),
        )

        with self.__temp_register_custom_error():

            class CustomValidationError(CustomTypedError):
                TYPE_ID = "errors/other/validation"

            with self.assertRaises(CustomValidationError):
                await self._parse_response(expected_type=_ResponseType200, response=mock_response)

    async def test_typed_error_fallback(self) -> None:
        mock_response = MockResponse(
            status_code=422,
            content=json.dumps(
                {
                    "type": "https://other.unittest.test/errors/other/validation",
                    "title": "Validation Error",
                }
            ),
        )

        with self.__temp_register_custom_error():

            class NotTheSameCustomError(CustomTypedError):
                TYPE_ID = "/some/other/error"

            with self.assertRaises(DefaultTypedError):
                await self._parse_response(expected_type=_ResponseType200, response=mock_response)

    def test_duplicate_custom_typed_error(self) -> None:
        def create_duplicate_custom_error_handle() -> None:
            class AnTypedError(CustomTypedError):
                TYPE_ID = "/error-source"

            class ADuplicateTypedError(CustomTypedError):
                TYPE_ID = "/error-source"

        with self.__temp_register_custom_error():
            with self.assertRaises(ValueError):
                create_duplicate_custom_error_handle()

    def test_invalid_type_id_for_custom_typed_error_handle(self) -> None:
        def create_invalid_type_id() -> None:
            class AnInvalidTypedError(CustomTypedError):
                TYPE_ID = 1

        with self.__temp_register_custom_error():
            with self.assertRaises(ValueError):
                create_invalid_type_id()

    def test_nonetype_allowed_for_abstract_typed_error_handle(self) -> None:
        with self.__temp_register_custom_error():

            class AValidBaseTypedError(CustomTypedError):
                TYPE_ID = None

            class AValidTypedError(AValidBaseTypedError):
                TYPE_ID = "/some/path/to/error"

    def test_type_id_must_be_set_for_custom_typed_error_handle(self) -> None:
        with self.__temp_register_custom_error():
            with self.assertRaises(ValueError):

                class NoTypeIDTypedError(CustomTypedError):
                    pass
