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
import unittest
from contextlib import redirect_stdout
from io import StringIO
from uuid import UUID

from evo.common import RequestMethod
from evo.common.exceptions import BadRequestException, GeneralizedTypedError
from evo.common.test_tools import TestWithConnector
from evo.objects.endpoints.api import ObjectsApi
from evo.objects.exceptions import InvalidGeoscienceObjectError


class TestInvalidGeoscienceObjectError(unittest.TestCase):
    def test_validation_paths_and_full_detail_are_preserved(self) -> None:
        content = {
            "type": "https://seequent.com/error-codes/geoscienceobject/schemas/invalid-object",
            "title": "The Geoscience Object Is Invalid.",
            "detail": f"{{'values': '{'x' * 1000}'}} is not valid under any of the given schemas",
            "json_path": "$.collections[0]",
            "schema_path": ["properties", "collections", "items", "oneOf"],
        }
        error = InvalidGeoscienceObjectError(400, "Bad Request", content, None)
        message = str(error)
        self.assertIn("Invalid value at: $.collections[0]", message)
        self.assertIn(f"Failed schema rule: {content['schema_path']}", message)
        self.assertIn("is not valid under any of the given schemas", message)
        self.assertNotIn("x" * 1000, message)
        self.assertIn("catch this exception as exc and run print(exc.detail)", message)
        self.assertEqual(content["detail"], error.detail)
        self.assertIs(content, error.content)
        self.assertIsInstance(error, BadRequestException)

    def test_missing_paths_and_detail_preview_boundaries(self) -> None:
        for length in (499, 500, 501):
            with self.subTest(length=length):
                detail = "x" * length
                content = {
                    "type": "https://seequent.com/error-codes/geoscienceobject/schemas/invalid-object",
                    "title": "Invalid object",
                    "detail": detail,
                }
                message = str(InvalidGeoscienceObjectError(400, "Bad Request", content, None))
                self.assertNotIn("Invalid value at:", message)
                self.assertNotIn("Failed schema rule:", message)
                self.assertEqual(length > 500, "[omitted]" in message)
                if length <= 500:
                    self.assertIn(f"Detail: {detail}", message)

    def test_paths_without_detail(self) -> None:
        content = {"title": "Invalid object", "json_path": "$.name"}
        message = str(InvalidGeoscienceObjectError(400, "Bad Request", content, None))
        self.assertIn("Invalid value at: $.name", message)
        self.assertNotIn("Detail:", message)

    def test_caught_exception_prints_the_full_detail(self) -> None:
        detail = f"{'x' * 1000} is not valid under any of the given schemas"
        error = InvalidGeoscienceObjectError(400, "Bad Request", {"title": "Invalid object", "detail": detail}, None)
        output = StringIO()
        with redirect_stdout(output):
            try:
                raise error
            except BadRequestException as exc:
                print(exc.detail)
        self.assertEqual(f"{detail}\n", output.getvalue())

    def test_registration_preserves_the_global_bad_request_handler(self) -> None:
        self.assertIs(BadRequestException, GeneralizedTypedError.from_status_code(400))


class TestInvalidObjectDispatch(TestWithConnector):
    async def test_generated_api_raises_a_compatible_specialised_error(self) -> None:
        content = {
            "type": "https://seequent.com/error-codes/geoscienceobject/schemas/invalid-object",
            "title": "Invalid object",
            "detail": "Missing required property",
            "json_path": "$.name",
        }
        with (
            self.transport.set_http_response(status_code=400, content=json.dumps(content)),
            self.assertRaises(BadRequestException) as cm,
        ):
            await ObjectsApi(self.connector).get_object(
                org_id=str(UUID(int=1)),
                workspace_id=str(UUID(int=2)),
                objects_path="invalid.json",
            )
        self.assertIs(type(cm.exception), InvalidGeoscienceObjectError)
        self.assertEqual(content, cm.exception.content)
        self.assertIn("Invalid value at: $.name", str(cm.exception))

    async def test_unrelated_bad_requests_remain_generalised(self) -> None:
        with (
            self.transport.set_http_response(status_code=400, content='{"title": "Bad request"}'),
            self.assertRaises(BadRequestException) as cm,
        ):
            await self.connector.call_api(method=RequestMethod.POST, resource_path="")
        self.assertIs(type(cm.exception), BadRequestException)
