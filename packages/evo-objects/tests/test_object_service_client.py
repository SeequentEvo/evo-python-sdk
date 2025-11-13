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

import dataclasses
import datetime
import json
from unittest import mock
from urllib.parse import quote as quote
from uuid import UUID

from dateutil.parser import parse as dateutil_parse

from data import load_test_data
from evo.common import HealthCheckType, Page, RequestMethod, ServiceUser
from evo.common.data import OrderByOperatorEnum
from evo.common.io.exceptions import DataNotFoundError
from evo.common.test_tools import MockResponse, TestWithConnector, TestWithStorage
from evo.common.utils import get_header_metadata
from evo.objects import (
    ObjectAPIClient,
    ObjectDataDownload,
    ObjectDataUpload,
    ObjectMetadata,
    ObjectSchema,
    ObjectVersion,
    SchemaVersion,
)
from evo.objects.data import ObjectOrderByEnum, OrgObjectMetadata, Stage
from evo.objects.exceptions import ObjectAlreadyExistsError, ObjectUUIDError
from evo.objects.utils import ObjectDataClient
from helpers import NoImport, UnloadModule

EMPTY_CONTENT = '{"objects": [], "links": {"next": null, "prev": null}}'
MOCK_VERSION_CONTENT = json.dumps(load_test_data("list_versions.json"))
_MAX_UPLOAD_URLS = 32


class TestObjectAPIClient(TestWithConnector, TestWithStorage):
    def setUp(self) -> None:
        TestWithConnector.setUp(self)
        TestWithStorage.setUp(self)
        self.object_client = ObjectAPIClient(connector=self.connector, environment=self.environment)
        self.setup_universal_headers(get_header_metadata(ObjectAPIClient.__module__))

    @property
    def instance_base_path(self) -> str:
        return f"geoscience-object/orgs/{self.environment.org_id}"

    @property
    def base_path(self) -> str:
        return f"{self.instance_base_path}/workspaces/{self.environment.workspace_id}"

    async def test_check_service_health(self) -> None:
        """Test service health check implementation"""
        with mock.patch("evo.objects.client.api_client.get_service_health", spec_set=True) as mock_get_service_health:
            await self.object_client.get_service_health()
        mock_get_service_health.assert_called_once_with(
            self.connector, "geoscience-object", check_type=HealthCheckType.FULL
        )

    async def test_list_objects_default_args(self) -> None:
        with self.transport.set_http_response(200, EMPTY_CONTENT, headers={"Content-Type": "application/json"}):
            page = await self.object_client.list_objects()
        self.assertIsInstance(page, Page)
        self.assertEqual([], page.items())
        self.assert_request_made(
            method=RequestMethod.GET,
            path=f"{self.base_path}/objects?limit=5000&offset=0",
            headers={"Accept": "application/json"},
        )

    async def test_list_objects_all_args(self) -> None:
        with self.transport.set_http_response(200, EMPTY_CONTENT, headers={"Content-Type": "application/json"}):
            page = await self.object_client.list_objects(limit=20)
        self.assertIsInstance(page, Page)
        self.assertEqual([], page.items())
        self.assert_request_made(
            method=RequestMethod.GET,
            path=f"{self.base_path}/objects?limit=20&offset=0",
            headers={"Accept": "application/json"},
        )

    async def test_list_objects(self) -> None:
        content_0 = load_test_data("list_objects_0.json")
        content_1 = load_test_data("list_objects_1.json")
        responses = [
            MockResponse(
                status_code=200,
                content=json.dumps(content_0),
                headers={"Content-Type": "application/json"},
            ),
            MockResponse(
                status_code=200,
                content=json.dumps(content_1),
                headers={"Content-Type": "application/json"},
            ),
        ]
        self.transport.request.side_effect = responses
        page_one = await self.object_client.list_objects(
            limit=2, order_by={ObjectOrderByEnum.created_at: OrderByOperatorEnum.asc}, schema_id=["test"], deleted=False
        )
        expected_items_page_one = [
            ObjectMetadata(
                environment=self.environment,
                id=UUID("00000000-0000-0000-0000-000000000002"),
                name="m.json",
                created_at=dateutil_parse("2020-01-01 01:30:00+00:00").replace(tzinfo=datetime.timezone.utc),
                created_by=ServiceUser(
                    id=UUID("00000000-0000-0000-0000-000000000010"),
                    name="x y",
                    email="test@example.com",
                ),
                modified_at=dateutil_parse("2020-01-02 01:30:00+00:00").replace(tzinfo=datetime.timezone.utc),
                modified_by=ServiceUser(
                    id=UUID("00000000-0000-0000-0000-000000000010"),
                    name="x y",
                    email="test@example.com",
                ),
                parent="/A",
                schema_id=ObjectSchema("objects", "test", SchemaVersion(1, 2, 3)),
                version_id="2",
                stage=Stage(name="Approved", id=UUID("00000000-0000-0000-0000-000000000999")),
            ),
            ObjectMetadata(
                environment=self.environment,
                id=UUID("00000000-0000-0000-0000-000000000003"),
                name="n.json",
                created_at=dateutil_parse("2020-01-02 01:30:00+00:00").replace(tzinfo=datetime.timezone.utc),
                created_by=ServiceUser(
                    id=UUID("00000000-0000-0000-0000-000000000011"),
                    name=None,
                    email=None,
                ),
                modified_at=dateutil_parse("2020-01-03 01:30:00+00:00").replace(tzinfo=datetime.timezone.utc),
                modified_by=ServiceUser(
                    id=UUID("00000000-0000-0000-0000-000000000011"),
                    name=None,
                    email=None,
                ),
                parent="/A",
                schema_id=ObjectSchema("objects", "test", SchemaVersion(1, 2, 3)),
                version_id="1",
                stage=None,
            ),
        ]
        self.assertIsInstance(page_one, Page)
        self.assertEqual(expected_items_page_one, page_one.items())
        self.assertEqual(0, page_one.offset)
        self.assertEqual(2, page_one.limit)
        self.assertFalse(page_one.is_last)
        self.assert_request_made(
            method=RequestMethod.GET,
            path=f"{self.base_path}/objects?limit=2&offset=0&deleted=False&order_by=asc%3Acreated_at&schema_id=test",
            headers={"Accept": "application/json"},
        )
        self.transport.request.reset_mock()

        page_two = await self.object_client.list_objects(offset=page_one.next_offset, limit=page_one.limit)
        expected_items_page_two = [
            ObjectMetadata(
                environment=self.environment,
                id=UUID("00000000-0000-0000-0000-000000000002"),
                name="o.json",
                created_at=dateutil_parse("2020-01-01 01:30:00+00:00").replace(tzinfo=datetime.timezone.utc),
                created_by=ServiceUser(
                    id=UUID("00000000-0000-0000-0000-000000000010"),
                    name="x y",
                    email="test@example.com",
                ),
                modified_at=dateutil_parse("2020-01-02 01:30:00+00:00").replace(tzinfo=datetime.timezone.utc),
                modified_by=ServiceUser(
                    id=UUID("00000000-0000-0000-0000-000000000011"),
                    name=None,
                    email=None,
                ),
                parent="/B",
                schema_id=ObjectSchema("objects", "test", SchemaVersion(1, 2, 3)),
                version_id="3",
                stage=None,
            ),
        ]
        self.assertIsInstance(page_two, Page)
        self.assertEqual(expected_items_page_two, page_two.items())
        self.assertEqual(2, page_two.offset)
        self.assertEqual(2, page_two.limit)
        self.assertTrue(page_two.is_last)
        self.assert_request_made(
            method=RequestMethod.GET,
            path=f"{self.base_path}/objects?limit=2&offset=2",
            headers={"Accept": "application/json"},
        )

    async def test_list_objects_for_instance(self) -> None:
        content_0 = load_test_data("list_objects_for_instance_0.json")
        content_1 = load_test_data("list_objects_for_instance_1.json")
        responses = [
            MockResponse(
                status_code=200,
                content=json.dumps(content_0),
                headers={"Content-Type": "application/json"},
            ),
            MockResponse(
                status_code=200,
                content=json.dumps(content_1),
                headers={"Content-Type": "application/json"},
            ),
        ]
        self.transport.request.side_effect = responses
        page_one = await self.object_client.list_objects_for_instance(
            limit=2, order_by={ObjectOrderByEnum.created_at: OrderByOperatorEnum.asc}, schema_id=["test"], deleted=False
        )
        expected_items_page_one = [
            OrgObjectMetadata(
                environment=self.environment,
                workspace_id=UUID("00000000-0000-0000-0000-00000000162e"),
                workspace_name="Test Workspace",
                id=UUID("00000000-0000-0000-0000-000000000002"),
                name="m.json",
                created_at=dateutil_parse("2020-01-01 01:30:00+00:00").replace(tzinfo=datetime.timezone.utc),
                created_by=ServiceUser(
                    id=UUID("00000000-0000-0000-0000-000000000010"),
                    name="x y",
                    email="test@example.com",
                ),
                modified_at=dateutil_parse("2020-01-02 01:30:00+00:00").replace(tzinfo=datetime.timezone.utc),
                modified_by=ServiceUser(
                    id=UUID("00000000-0000-0000-0000-000000000010"),
                    name="x y",
                    email="test@example.com",
                ),
                schema_id=ObjectSchema("objects", "test", SchemaVersion(1, 2, 3)),
                stage=Stage(name="Approved", id=UUID("00000000-0000-0000-0000-000000000747")),
            ),
            OrgObjectMetadata(
                environment=self.environment,
                workspace_id=UUID("00000000-0000-0000-0000-00000000162e"),
                workspace_name="Test Workspace",
                id=UUID("00000000-0000-0000-0000-000000000003"),
                name="n.json",
                created_at=dateutil_parse("2020-01-02 01:30:00+00:00").replace(tzinfo=datetime.timezone.utc),
                created_by=ServiceUser(
                    id=UUID("00000000-0000-0000-0000-000000000011"),
                    name=None,
                    email=None,
                ),
                modified_at=dateutil_parse("2020-01-03 01:30:00+00:00").replace(tzinfo=datetime.timezone.utc),
                modified_by=ServiceUser(
                    id=UUID("00000000-0000-0000-0000-000000000011"),
                    name=None,
                    email=None,
                ),
                schema_id=ObjectSchema("objects", "test", SchemaVersion(1, 2, 3)),
                stage=None,
            ),
        ]
        self.assertIsInstance(page_one, Page)
        for item in page_one:
            self.assertEqual(item.environment.workspace_id, item.workspace_id, "workspace_id should match environment")
        self.assertEqual(expected_items_page_one, page_one.items())
        self.assertEqual(0, page_one.offset)
        self.assertEqual(2, page_one.limit)
        self.assertFalse(page_one.is_last)
        self.assert_request_made(
            method=RequestMethod.GET,
            path=f"{self.instance_base_path}/objects?offset=0&limit=2&deleted=False&permitted_workspaces_only=True&order_by=asc%3Acreated_at&schema_id=test",
            headers={"Accept": "application/json"},
        )
        self.transport.request.reset_mock()

        page_two = await self.object_client.list_objects_for_instance(offset=page_one.next_offset, limit=page_one.limit)
        expected_items_page_two = [
            OrgObjectMetadata(
                environment=dataclasses.replace(
                    self.environment, workspace_id=UUID("00000000-0000-0000-0000-0000000004d2")
                ),
                workspace_id=UUID("00000000-0000-0000-0000-0000000004d2"),
                workspace_name="Test Workspace 2",
                id=UUID("00000000-0000-0000-0000-000000000002"),
                name="o.json",
                created_at=dateutil_parse("2020-01-01 01:30:00+00:00").replace(tzinfo=datetime.timezone.utc),
                created_by=ServiceUser(
                    id=UUID("00000000-0000-0000-0000-000000000010"),
                    name="x y",
                    email="test@example.com",
                ),
                modified_at=dateutil_parse("2020-01-02 01:30:00+00:00").replace(tzinfo=datetime.timezone.utc),
                modified_by=ServiceUser(
                    id=UUID("00000000-0000-0000-0000-000000000011"),
                    name=None,
                    email=None,
                ),
                schema_id=ObjectSchema("objects", "test", SchemaVersion(1, 2, 3)),
                stage=None,
            ),
        ]
        self.assertIsInstance(page_two, Page)
        for item in page_two:
            self.assertEqual(item.environment.workspace_id, item.workspace_id, "workspace_id should match environment")
        self.assertEqual(expected_items_page_two, page_two.items())
        self.assertEqual(2, page_two.offset)
        self.assertEqual(2, page_two.limit)
        self.assertTrue(page_two.is_last)
        self.assert_request_made(
            method=RequestMethod.GET,
            path=f"{self.instance_base_path}/objects?offset=2&limit=2&permitted_workspaces_only=True",
            headers={"Accept": "application/json"},
        )

    async def test_list_all_objects(self) -> None:
        content_0 = load_test_data("list_objects_0.json")
        content_1 = load_test_data("list_objects_1.json")
        responses = [
            MockResponse(
                status_code=200,
                content=json.dumps(content_0),
                headers={"Content-Type": "application/json"},
            ),
            MockResponse(
                status_code=200,
                content=json.dumps(content_1),
                headers={"Content-Type": "application/json"},
            ),
        ]
        self.transport.request.side_effect = responses
        all_objects = await self.object_client.list_all_objects(
            limit_per_request=2,
            order_by={ObjectOrderByEnum.created_at: OrderByOperatorEnum.asc},
            schema_id=["test"],
            deleted=False,
        )
        expected_objects = [
            ObjectMetadata(
                environment=self.environment,
                id=UUID("00000000-0000-0000-0000-000000000002"),
                name="m.json",
                created_at=dateutil_parse("2020-01-01 01:30:00+00:00").replace(tzinfo=datetime.timezone.utc),
                created_by=ServiceUser(
                    id=UUID("00000000-0000-0000-0000-000000000010"),
                    name="x y",
                    email="test@example.com",
                ),
                modified_at=dateutil_parse("2020-01-02 01:30:00+00:00").replace(tzinfo=datetime.timezone.utc),
                modified_by=ServiceUser(
                    id=UUID("00000000-0000-0000-0000-000000000010"),
                    name="x y",
                    email="test@example.com",
                ),
                parent="/A",
                schema_id=ObjectSchema("objects", "test", SchemaVersion(1, 2, 3)),
                version_id="2",
                stage=Stage(name="Approved", id=UUID("00000000-0000-0000-0000-000000000999")),
            ),
            ObjectMetadata(
                environment=self.environment,
                id=UUID("00000000-0000-0000-0000-000000000003"),
                name="n.json",
                created_at=dateutil_parse("2020-01-02 01:30:00+00:00").replace(tzinfo=datetime.timezone.utc),
                created_by=ServiceUser(
                    id=UUID("00000000-0000-0000-0000-000000000011"),
                    name=None,
                    email=None,
                ),
                modified_at=dateutil_parse("2020-01-03 01:30:00+00:00").replace(tzinfo=datetime.timezone.utc),
                modified_by=ServiceUser(
                    id=UUID("00000000-0000-0000-0000-000000000011"),
                    name=None,
                    email=None,
                ),
                parent="/A",
                schema_id=ObjectSchema("objects", "test", SchemaVersion(1, 2, 3)),
                version_id="1",
                stage=None,
            ),
            ObjectMetadata(
                environment=self.environment,
                id=UUID("00000000-0000-0000-0000-000000000002"),
                name="o.json",
                created_at=dateutil_parse("2020-01-01 01:30:00+00:00").replace(tzinfo=datetime.timezone.utc),
                created_by=ServiceUser(
                    id=UUID("00000000-0000-0000-0000-000000000010"),
                    name="x y",
                    email="test@example.com",
                ),
                modified_at=dateutil_parse("2020-01-02 01:30:00+00:00").replace(tzinfo=datetime.timezone.utc),
                modified_by=ServiceUser(
                    id=UUID("00000000-0000-0000-0000-000000000011"),
                    name=None,
                    email=None,
                ),
                parent="/B",
                schema_id=ObjectSchema("objects", "test", SchemaVersion(1, 2, 3)),
                version_id="3",
                stage=None,
            ),
        ]
        self.assertEqual(expected_objects, all_objects)
        self.assert_any_request_made(
            method=RequestMethod.GET,
            path=f"{self.base_path}/objects?limit=2&offset=0&deleted=False&order_by=asc%3Acreated_at&schema_id=test",
            headers={"Accept": "application/json"},
        )
        self.assert_any_request_made(
            method=RequestMethod.GET,
            path=f"{self.base_path}/objects?limit=2&offset=2&deleted=False&order_by=asc%3Acreated_at&schema_id=test",
            headers={"Accept": "application/json"},
        )

    async def test_list_versions_by_path(self) -> None:
        object_path = "A/m.json"
        with self.transport.set_http_response(200, MOCK_VERSION_CONTENT, headers={"Content-Type": "application/json"}):
            versions = await self.object_client.list_versions_by_path(object_path)
        expected_versions = [
            ObjectVersion(
                version_id="2022-01-01T01:30:00.0000000Z",
                created_at=datetime.datetime(2022, 1, 1, 1, 30, tzinfo=datetime.timezone.utc),
                created_by=ServiceUser(
                    id=UUID("00000000-0000-0000-0000-000000000011"),
                    name="x z",
                    email="test@example.com",
                ),
                stage=Stage(name="Approved", id=UUID("00000000-0000-0000-0000-000000000123")),
            ),
            ObjectVersion(
                version_id="2020-01-01T01:30:00.0000000Z",
                created_at=datetime.datetime(2020, 1, 1, 1, 30, tzinfo=datetime.timezone.utc),
                created_by=ServiceUser(
                    id=UUID("00000000-0000-0000-0000-000000000010"),
                    name="x y",
                    email="test@example.com",
                ),
                stage=Stage(name="Approved", id=UUID("00000000-0000-0000-0000-000000000123")),
            ),
            ObjectVersion(
                version_id="2010-01-01T01:30:00.0000000Z",
                created_at=datetime.datetime(2010, 1, 1, 1, 30, tzinfo=datetime.timezone.utc),
                created_by=ServiceUser(
                    id=UUID("00000000-0000-0000-0000-000000000012"),
                    name="x w",
                    email="test@example.com",
                ),
                stage=None,
            ),
        ]

        self.assertEqual(versions, expected_versions)
        self.assert_request_made(
            method=RequestMethod.GET,
            path=f"{self.base_path}/objects/path/A/m.json?include_versions=True",
            headers={"Authorization": "Bearer <not-a-real-token>", "Accept": "application/json"},
        )

    async def test_list_versions_by_id(self) -> None:
        object_id = UUID(int=2)
        with self.transport.set_http_response(200, MOCK_VERSION_CONTENT, headers={"Content-Type": "application/json"}):
            versions = await self.object_client.list_versions_by_id(object_id)
        expected_versions = [
            ObjectVersion(
                version_id="2022-01-01T01:30:00.0000000Z",
                created_at=datetime.datetime(2022, 1, 1, 1, 30, tzinfo=datetime.timezone.utc),
                created_by=ServiceUser(
                    id=UUID("00000000-0000-0000-0000-000000000011"),
                    name="x z",
                    email="test@example.com",
                ),
                stage=Stage(name="Approved", id=UUID("00000000-0000-0000-0000-000000000123")),
            ),
            ObjectVersion(
                version_id="2020-01-01T01:30:00.0000000Z",
                created_at=datetime.datetime(2020, 1, 1, 1, 30, tzinfo=datetime.timezone.utc),
                created_by=ServiceUser(
                    id=UUID("00000000-0000-0000-0000-000000000010"),
                    name="x y",
                    email="test@example.com",
                ),
                stage=Stage(name="Approved", id=UUID("00000000-0000-0000-0000-000000000123")),
            ),
            ObjectVersion(
                version_id="2010-01-01T01:30:00.0000000Z",
                created_at=datetime.datetime(2010, 1, 1, 1, 30, tzinfo=datetime.timezone.utc),
                created_by=ServiceUser(
                    id=UUID("00000000-0000-0000-0000-000000000012"),
                    name="x w",
                    email="test@example.com",
                ),
                stage=None,
            ),
        ]

        self.assertEqual(expected_versions, versions)
        self.assert_request_made(
            method=RequestMethod.GET,
            path=f"{self.base_path}/objects/00000000-0000-0000-0000-000000000002?include_versions=True",
            headers={"Authorization": "Bearer <not-a-real-token>", "Accept": "application/json"},
        )

    async def test_prepare_data_upload(self) -> None:
        """Test preparing a single data upload."""
        put_data_response = load_test_data("put_data.json")
        expected_name = put_data_response[0]["name"]
        with self.transport.set_http_response(
            status_code=200,
            content=json.dumps(put_data_response),
            headers={"Content-Type": "application/json"},
        ):
            (upload,) = [upload async for upload in self.object_client.prepare_data_upload([expected_name])]

        self.assertIsInstance(upload, ObjectDataUpload)
        self.assertEqual(expected_name, upload.name)
        self.assertEqual(self.environment, upload.environment)
        self.assert_request_made(
            method=RequestMethod.PUT,
            path=f"{self.base_path}/data",
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            body=[{"name": expected_name}],
        )

    async def test_prepare_data_upload_batches(self) -> None:
        """Test preparing multiple data uploads in batches."""
        put_data_response = load_test_data("put_data_batch.json")
        # The service can only generate _MAX_UPLOAD_URLS upload URLs in a single request.
        batch_1 = put_data_response[:_MAX_UPLOAD_URLS]
        batch_2 = put_data_response[_MAX_UPLOAD_URLS:]

        batch_1_by_name = {
            upload["name"]: upload
            for upload in batch_1
            if upload["exists"] is False  # Pre-existing data is skipped.
        }
        batch_2_by_name = {
            upload["name"]: upload
            for upload in batch_2
            if upload["exists"] is False  # Pre-existing data is skipped.
        }

        aiter_uploads = self.object_client.prepare_data_upload([data["name"] for data in put_data_response])
        with self.transport.set_http_response(
            status_code=200, content=json.dumps(batch_1), headers={"Content-Type": "application/json"}
        ):
            self.transport.assert_no_requests()

            # Awaiting the first result from the first batch should trigger the first request.
            upload = await anext(aiter_uploads)

            self.assert_request_made(
                method=RequestMethod.PUT,
                path=f"{self.base_path}/data",
                headers={"Content-Type": "application/json", "Accept": "application/json"},
                body=[{"name": data["name"]} for data in batch_1],
            )
            self.assertIsInstance(upload, ObjectDataUpload)
            expected = batch_1_by_name.pop(upload.name)
            self.assertEqual(expected["name"], upload.name)
            self.assertEqual(self.environment, upload.environment)

        self.transport.reset_mock()
        while len(batch_1_by_name) > 0:
            upload = await anext(aiter_uploads)
            self.assertIsInstance(upload, ObjectDataUpload)
            expected = batch_1_by_name.pop(upload.name)
            self.assertEqual(expected["name"], upload.name)
            self.assertEqual(self.environment, upload.environment)

        # No more requests should be made until the next batch is requested.
        self.transport.assert_no_requests()

        with self.transport.set_http_response(
            status_code=200, content=json.dumps(batch_2), headers={"Content-Type": "application/json"}
        ):
            upload = await anext(aiter_uploads)
            self.assert_request_made(
                method=RequestMethod.PUT,
                path=f"{self.base_path}/data",
                headers={"Content-Type": "application/json", "Accept": "application/json"},
                body=[{"name": data["name"]} for data in batch_2],
            )
            self.assertIsInstance(upload, ObjectDataUpload)
            expected = batch_2_by_name.pop(upload.name)
            self.assertEqual(expected["name"], upload.name)
            self.assertEqual(self.environment, upload.environment)

        self.transport.reset_mock()
        while len(batch_2_by_name) > 0:
            upload = await anext(aiter_uploads)
            self.assertIsInstance(upload, ObjectDataUpload)
            expected = batch_2_by_name.pop(upload.name)
            self.assertEqual(expected["name"], upload.name)
            self.assertEqual(self.environment, upload.environment)

        # No more uploads should be available.
        with self.assertRaises(StopAsyncIteration):
            await anext(aiter_uploads)

        # No more requests should be made after all uploads are returned.
        self.transport.assert_no_requests()

    async def test_prepare_data_download(self) -> None:
        """Test preparing a single data download."""
        get_object_response = load_test_data("get_object.json")
        expected_id = UUID(get_object_response["object_id"])
        expected_version = get_object_response["version_id"]
        expected_name = get_object_response["links"]["data"][0]["name"]

        with self.transport.set_http_response(
            status_code=200,
            content=json.dumps(get_object_response),
            headers={"Content-Type": "application/json"},
        ):
            (download,) = [
                download
                async for download in self.object_client.prepare_data_download(
                    expected_id, expected_version, [expected_name]
                )
            ]

        self.assertIsInstance(download, ObjectDataDownload)
        self.assertEqual(expected_name, download.name)
        self.assertEqual(expected_id, download.metadata.id)
        self.assertEqual(expected_version, download.metadata.version_id)
        self.assertEqual(self.environment, download.metadata.environment)
        self.assert_request_made(
            method=RequestMethod.GET,
            path=f"{self.base_path}/objects/{expected_id}?version={quote(expected_version)}",
            headers={"Accept": "application/json", "Accept-Encoding": "gzip"},
        )

    async def test_prepare_data_download_multiple(self) -> None:
        """Test preparing multiple data downloads."""
        get_object_response = load_test_data("get_object.json")
        expected_id = UUID(get_object_response["object_id"])
        expected_version = get_object_response["version_id"]
        expected_names = [data["name"] for data in get_object_response["links"]["data"]]
        expected_data_by_name = {data["name"]: data for data in get_object_response["links"]["data"]}

        aiter_downloads = self.object_client.prepare_data_download(expected_id, expected_version, expected_names)
        with self.transport.set_http_response(
            status_code=200,
            content=json.dumps(get_object_response),
            headers={"Content-Type": "application/json"},
        ):
            self.transport.assert_no_requests()

            # Awaiting the first result should trigger the first request.
            download = await anext(aiter_downloads)

            self.assert_request_made(
                method=RequestMethod.GET,
                path=f"{self.base_path}/objects/{expected_id}?version={quote(expected_version)}",
                headers={"Accept": "application/json", "Accept-Encoding": "gzip"},
            )
            self.assertIsInstance(download, ObjectDataDownload)
            expected = expected_data_by_name.pop(download.name)
            self.assertEqual(expected["name"], download.name)
            self.assertEqual(expected_id, download.metadata.id)
            self.assertEqual(expected_version, download.metadata.version_id)
            self.assertEqual(self.environment, download.metadata.environment)

        self.transport.reset_mock()

        while len(expected_data_by_name) > 0:
            download = await anext(aiter_downloads)
            self.assertIsInstance(download, ObjectDataDownload)
            expected = expected_data_by_name.pop(download.name)
            self.assertEqual(expected["name"], download.name)
            self.assertEqual(expected_id, download.metadata.id)
            self.assertEqual(expected_version, download.metadata.version_id)
            self.assertEqual(self.environment, download.metadata.environment)

        # No more downloads should be available.
        with self.assertRaises(StopAsyncIteration):
            await anext(aiter_downloads)

        # No more requests should be made after all downloads are returned.
        self.transport.assert_no_requests()

    async def test_prepare_data_download_missing_data(self) -> None:
        """Test preparing to download missing data."""
        get_object_response = load_test_data("get_object.json")
        expected_id = UUID(get_object_response["object_id"])
        expected_version = get_object_response["version_id"]

        with self.transport.set_http_response(
            status_code=200,
            content=json.dumps(get_object_response),
            headers={"Content-Type": "application/json"},
        ):
            aiter_downloads = self.object_client.prepare_data_download(expected_id, expected_version, ["missing"])

            with self.assertRaises(DataNotFoundError):
                await anext(aiter_downloads)

    def test_get_data_client(self) -> None:
        """Test getting a data client."""
        data_client = self.object_client.get_data_client(self.cache)
        self.assertIsInstance(data_client, ObjectDataClient)

    def test_get_data_client_missing_dependencies(self) -> None:
        """Test getting a data client with missing dependencies."""
        with UnloadModule("evo.objects.client.api_client", "evo.objects.utils.data"), NoImport("pyarrow"):
            from evo.objects.client import ObjectAPIClient

            client = ObjectAPIClient(self.environment, self.connector)
            self.assertFalse(
                any(
                    (
                        hasattr(ObjectAPIClient, "get_data_client"),
                        hasattr(client, "get_data_client"),
                    )
                ),
                "get_data_client should not be available if pyarrow is missing",
            )

    async def test_get_latest_object_versions(self) -> None:
        content = json.dumps(
            [
                {"object_id": str(UUID(int=1)), "version_id": "2023-01-01T00:00:00.00000000Z"},
                {"object_id": str(UUID(int=2)), "version_id": None},
            ]
        )
        object_ids = [UUID(int=1), UUID(int=2)]
        with self.transport.set_http_response(200, content, headers={"Content-Type": "application/json"}):
            versions = await self.object_client.get_latest_object_versions(object_ids)
        self.assertEqual(versions, {UUID(int=1): "2023-01-01T00:00:00.00000000Z", UUID(int=2): None})
        self.assert_request_made(
            method=RequestMethod.PATCH,
            path=f"{self.base_path}/objects",
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            body=["00000000-0000-0000-0000-000000000001", "00000000-0000-0000-0000-000000000002"],
        )

    async def test_get_latest_object_versions_batch(self) -> None:
        version_id = "2023-01-01T00:00:00.00000000Z"
        object_ids = [UUID(int=i) for i in range(10)]
        batch_ids = [object_ids[i : i + 3] for i in range(0, len(object_ids), 3)]
        responses = []
        for batch in batch_ids:
            data = json.dumps([{"object_id": str(batch_id), "version_id": version_id} for batch_id in batch])
            responses.append(MockResponse(status_code=200, headers={"Content-Type": "application/json"}, content=data))
        self.transport.request.side_effect = responses
        versions = await self.object_client.get_latest_object_versions(object_ids, batch_size=3)
        expected_versions = {object_id: version_id for object_id in object_ids}
        self.assertEqual(versions, expected_versions)
        for batch in batch_ids:
            self.assert_any_request_made(
                method=RequestMethod.PATCH,
                path=f"{self.base_path}/objects",
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                },
                body=[str(b_id) for b_id in batch],
            )

    async def test_create_geoscience_object(self) -> None:
        get_object_response = load_test_data("get_object.json")
        new_pointset = {
            "name": "Sample pointset",
            "uuid": None,
            "description": "A sample pointset object",
            "bounding_box": {"min_x": 0.0, "max_x": 0.0, "min_y": 0.0, "max_y": 0.0, "min_z": 0.0, "max_z": 0.0},
            "coordinate_reference_system": {"epsg_code": 2048},
            "locations": {
                "coordinates": {
                    "data": "0000000000000000000000000000000000000000000000000000000000000001",
                    "length": 1,
                    "width": 3,
                    "data_type": "float64",
                }
            },
            "schema": "/objects/pointset/1.0.1/pointset.schema.json",
        }
        new_pointset_without_uuid = new_pointset.copy()
        with self.transport.set_http_response(status_code=201, content=json.dumps(get_object_response)):
            expected_object_path = "A/m.json"
            new_object_metadata = await self.object_client.create_geoscience_object(expected_object_path, new_pointset)

        self.assert_request_made(
            method=RequestMethod.POST,
            path=f"{self.base_path}/objects/path/{expected_object_path}",
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            body=new_pointset_without_uuid,
        )
        expected_uuid = UUID(int=2)
        self.assertEqual("A", new_object_metadata.parent)
        self.assertEqual("m.json", new_object_metadata.name)
        self.assertEqual(expected_uuid, new_pointset["uuid"])
        self.assertEqual(expected_uuid, new_object_metadata.id)
        self.assertEqual("2023-08-03T05:47:18.3402289Z", new_object_metadata.version_id)

    async def test_create_geoscience_object_already_exists(self) -> None:
        already_exists_response = load_test_data("object_already_exists_error.json")
        expected_type = already_exists_response["type"]
        expected_title = already_exists_response["title"]
        expected_detail = already_exists_response["detail"]
        expected_existing_id = UUID(already_exists_response["existing_id"])
        expected_object_path = already_exists_response["object_path"]
        new_pointset = {
            "name": "Sample pointset",
            "uuid": None,
            "description": "A sample pointset object",
            "bounding_box": {"min_x": 0.0, "max_x": 0.0, "min_y": 0.0, "max_y": 0.0, "min_z": 0.0, "max_z": 0.0},
            "coordinate_reference_system": {"epsg_code": 2048},
            "locations": {
                "coordinates": {
                    "data": "0000000000000000000000000000000000000000000000000000000000000001",
                    "length": 1,
                    "width": 3,
                    "data_type": "float64",
                }
            },
            "schema": "/objects/pointset/1.0.1/pointset.schema.json",
        }
        with (
            self.transport.set_http_response(
                status_code=409, content=json.dumps(already_exists_response), reason="Conflict"
            ),
            self.assertRaises(ObjectAlreadyExistsError) as cm,
        ):
            await self.object_client.create_geoscience_object(expected_object_path, new_pointset)

        self.assert_request_made(
            method=RequestMethod.POST,
            path=f"{self.base_path}/objects/path/{expected_object_path}",
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            body=new_pointset,
        )

        self.assertEqual(expected_type, cm.exception.type_)
        self.assertEqual(expected_title, cm.exception.title)
        self.assertEqual(expected_detail, cm.exception.detail)
        self.assertEqual(expected_existing_id, cm.exception.existing_id)
        self.assertEqual(expected_object_path, cm.exception.object_path)

        expected_error_str = (
            f"Error: (409) Conflict"
            f"\nType: {expected_type}"
            f"\nTitle: {expected_title}"
            f"\nDetail: {expected_detail}"
            f"\nPath: {expected_object_path}"
            f"\nExisting ID: {expected_existing_id}"
        )
        actual_error_str = str(cm.exception)
        self.assertEqual(expected_error_str, actual_error_str)

    async def test_create_geoscience_object_with_uuid_fails(self) -> None:
        existing_pointset = {
            "name": "Sample pointset",
            "uuid": "00000000-0000-0000-0000-000000000002",
            "description": "A sample pointset object",
            "bounding_box": {"min_x": 0.0, "max_x": 0.0, "min_y": 0.0, "max_y": 0.0, "min_z": 0.0, "max_z": 0.0},
            "coordinate_reference_system": {"epsg_code": 2048},
            "locations": {
                "coordinates": {
                    "data": "0000000000000000000000000000000000000000000000000000000000000001",
                    "length": 1,
                    "width": 3,
                    "data_type": "float64",
                }
            },
            "schema": "/objects/pointset/1.0.1/pointset.schema.json",
        }
        with self.assertRaises(ObjectUUIDError):
            await self.object_client.create_geoscience_object("A/m.json", existing_pointset)
        self.transport.assert_no_requests()

    async def test_move_geoscience_object(self) -> None:
        get_object_response = load_test_data("get_object.json")
        existing_uuid = UUID(int=2)
        existing_pointset = {
            "name": "Sample pointset",
            "uuid": "00000000-0000-0000-0000-000000000002",
            "description": "A sample pointset object",
            "bounding_box": {
                "min_x": 0.0,
                "max_x": 0.0,
                "min_y": 0.0,
                "max_y": 0.0,
                "min_z": 0.0,
                "max_z": 0.0,
            },
            "coordinate_reference_system": {"epsg_code": 2048},
            "locations": {
                "coordinates": {
                    "data": "0000000000000000000000000000000000000000000000000000000000000001",
                    "length": 1,
                    "width": 3,
                    "data_type": "float64",
                }
            },
            "schema": "/objects/pointset/1.0.1/pointset.schema.json",
        }

        with self.transport.set_http_response(status_code=201, content=json.dumps(get_object_response)):
            expected_object_path = "A/m.json"
            new_object_metadata = await self.object_client.move_geoscience_object(
                expected_object_path, existing_pointset
            )

        self.assert_request_made(
            method=RequestMethod.POST,
            path=f"{self.base_path}/objects/path/{expected_object_path}",
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            body=existing_pointset,
        )
        self.assertEqual("A", new_object_metadata.parent)
        self.assertEqual("m.json", new_object_metadata.name)
        self.assertEqual(existing_uuid, new_object_metadata.id)
        self.assertEqual("2023-08-03T05:47:18.3402289Z", new_object_metadata.version_id)

    async def test_move_geoscience_object_already_exists(self) -> None:
        already_exists_response = load_test_data("object_already_exists_error.json")
        expected_type = already_exists_response["type"]
        expected_title = already_exists_response["title"]
        expected_detail = already_exists_response["detail"]
        expected_existing_id = UUID(already_exists_response["existing_id"])
        expected_object_path = already_exists_response["object_path"]
        existing_pointset = {
            "name": "Sample pointset",
            "uuid": str(UUID(int=2)),
            "description": "A sample pointset object",
            "bounding_box": {
                "min_x": 0.0,
                "max_x": 0.0,
                "min_y": 0.0,
                "max_y": 0.0,
                "min_z": 0.0,
                "max_z": 0.0,
            },
            "coordinate_reference_system": {"epsg_code": 2048},
            "locations": {
                "coordinates": {
                    "data": "0000000000000000000000000000000000000000000000000000000000000001",
                    "length": 1,
                    "width": 3,
                    "data_type": "float64",
                }
            },
            "schema": "/objects/pointset/1.0.1/pointset.schema.json",
        }
        with (
            self.transport.set_http_response(
                status_code=409,
                content=json.dumps(already_exists_response),
                reason="Conflict",
            ),
            self.assertRaises(ObjectAlreadyExistsError) as cm,
        ):
            await self.object_client.move_geoscience_object(expected_object_path, existing_pointset)

        self.assert_request_made(
            method=RequestMethod.POST,
            path=f"{self.base_path}/objects/path/{expected_object_path}",
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            body=existing_pointset,
        )

        self.assertEqual(expected_type, cm.exception.type_)
        self.assertEqual(expected_title, cm.exception.title)
        self.assertEqual(expected_detail, cm.exception.detail)
        self.assertEqual(expected_existing_id, cm.exception.existing_id)
        self.assertEqual(expected_object_path, cm.exception.object_path)

        expected_error_str = (
            f"Error: (409) Conflict"
            f"\nType: {expected_type}"
            f"\nTitle: {expected_title}"
            f"\nDetail: {expected_detail}"
            f"\nPath: {expected_object_path}"
            f"\nExisting ID: {expected_existing_id}"
        )
        actual_error_str = str(cm.exception)
        self.assertEqual(expected_error_str, actual_error_str)

    async def test_move_geoscience_object_without_uuid_fails(self) -> None:
        existing_pointset = {
            "name": "Sample pointset",
            "uuid": None,
            "description": "A sample pointset object",
            "bounding_box": {
                "min_x": 0.0,
                "max_x": 0.0,
                "min_y": 0.0,
                "max_y": 0.0,
                "min_z": 0.0,
                "max_z": 0.0,
            },
            "coordinate_reference_system": {"epsg_code": 2048},
            "locations": {
                "coordinates": {
                    "data": "0000000000000000000000000000000000000000000000000000000000000001",
                    "length": 1,
                    "width": 3,
                    "data_type": "float64",
                }
            },
            "schema": "/objects/pointset/1.0.1/pointset.schema.json",
        }

        with self.assertRaises(ObjectUUIDError):
            expected_object_path = "A/m.json"
            await self.object_client.move_geoscience_object(expected_object_path, existing_pointset)
        self.transport.assert_no_requests()

    async def test_update_geoscience_object(self) -> None:
        get_object_response = load_test_data("get_object.json")
        existing_uuid = UUID(int=2)
        updated_pointset = {
            "name": "Sample pointset",
            "uuid": "00000000-0000-0000-0000-000000000002",
            "description": "A sample pointset object",
            "bounding_box": {"min_x": 0.0, "max_x": 0.0, "min_y": 0.0, "max_y": 0.0, "min_z": 0.0, "max_z": 0.0},
            "coordinate_reference_system": {"epsg_code": 2048},
            "locations": {
                "coordinates": {
                    "data": "0000000000000000000000000000000000000000000000000000000000000001",
                    "length": 1,
                    "width": 3,
                    "data_type": "float64",
                }
            },
            "schema": "/objects/pointset/1.0.1/pointset.schema.json",
        }

        with self.transport.set_http_response(status_code=201, content=json.dumps(get_object_response)):
            new_object_metadata = await self.object_client.update_geoscience_object(updated_pointset)

        self.assert_request_made(
            method=RequestMethod.POST,
            path=f"{self.base_path}/objects/{existing_uuid}",
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            body=updated_pointset,
        )
        self.assertEqual("A", new_object_metadata.parent)
        self.assertEqual("m.json", new_object_metadata.name)
        self.assertEqual(existing_uuid, UUID(updated_pointset["uuid"]))
        self.assertEqual(existing_uuid, new_object_metadata.id)
        self.assertEqual("2023-08-03T05:47:18.3402289Z", new_object_metadata.version_id)

    async def test_update_geoscience_object_without_uuid_fails(self) -> None:
        existing_pointset = {
            "name": "Sample pointset",
            "uuid": None,
            "description": "A sample pointset object",
            "bounding_box": {"min_x": 0.0, "max_x": 0.0, "min_y": 0.0, "max_y": 0.0, "min_z": 0.0, "max_z": 0.0},
            "coordinate_reference_system": {"epsg_code": 2048},
            "locations": {
                "coordinates": {
                    "data": "0000000000000000000000000000000000000000000000000000000000000001",
                    "length": 1,
                    "width": 3,
                    "data_type": "float64",
                }
            },
            "schema": "/objects/pointset/1.0.1/pointset.schema.json",
        }

        with self.assertRaises(ObjectUUIDError):
            await self.object_client.update_geoscience_object(existing_pointset)
        self.transport.assert_no_requests()

    async def test_download_object_by_path(self) -> None:
        get_object_response = load_test_data("get_object.json")
        expected_uuid = UUID(int=2)
        expected_object_dict = {
            "schema": "/objects/pointset/1.0.1/pointset.schema.json",
            "uuid": UUID("00000000-0000-0000-0000-000000000002"),
            "name": "Sample pointset",
            "description": "A sample pointset object",
            "bounding_box": {"min_x": 0.0, "max_x": 0.0, "min_y": 0.0, "max_y": 0.0, "min_z": 0.0, "max_z": 0.0},
            "coordinate_reference_system": {"epsg_code": 2048},
            "locations": {
                "coordinates": {
                    "data": "0000000000000000000000000000000000000000000000000000000000000001",
                    "length": 1,
                    "width": 3,
                    "data_type": "float64",
                }
            },
        }
        expected_path = "A/m.json"
        expected_version = "2023-08-03T05:47:18.3402289Z"
        with self.transport.set_http_response(status_code=200, content=json.dumps(get_object_response)):
            actual_object = await self.object_client.download_object_by_path(expected_path, expected_version)

        self.assert_request_made(
            method=RequestMethod.GET,
            path=f"{self.base_path}/objects/path/{expected_path}?version={quote(expected_version)}",
            headers={"Accept": "application/json", "Accept-Encoding": "gzip"},
        )
        # Check metadata.
        actual_metadata = actual_object.metadata
        self.assertEqual(expected_path, actual_metadata.path)
        self.assertEqual("A", actual_metadata.parent)
        self.assertEqual("m.json", actual_metadata.name)
        self.assertEqual(expected_uuid, actual_metadata.id)
        self.assertEqual(expected_version, actual_metadata.version_id)

        # Check geoscience object.
        self.assertEqual(expected_object_dict, actual_object.as_dict())

    async def test_download_object_by_id(self) -> None:
        get_object_response = load_test_data("get_object.json")
        expected_uuid = UUID(int=2)
        expected_object_dict = {
            "schema": "/objects/pointset/1.0.1/pointset.schema.json",
            "uuid": UUID("00000000-0000-0000-0000-000000000002"),
            "name": "Sample pointset",
            "description": "A sample pointset object",
            "bounding_box": {"min_x": 0.0, "max_x": 0.0, "min_y": 0.0, "max_y": 0.0, "min_z": 0.0, "max_z": 0.0},
            "coordinate_reference_system": {"epsg_code": 2048},
            "locations": {
                "coordinates": {
                    "data": "0000000000000000000000000000000000000000000000000000000000000001",
                    "length": 1,
                    "width": 3,
                    "data_type": "float64",
                }
            },
        }
        expected_path = "A/m.json"
        expected_version = "2023-08-03T05:47:18.3402289Z"
        with self.transport.set_http_response(status_code=200, content=json.dumps(get_object_response)):
            actual_object = await self.object_client.download_object_by_id(expected_uuid, expected_version)

        self.assert_request_made(
            method=RequestMethod.GET,
            path=f"{self.base_path}/objects/{expected_uuid}?version={quote(expected_version)}",
            headers={"Accept": "application/json", "Accept-Encoding": "gzip"},
        )
        # Check metadata.
        actual_metadata = actual_object.metadata
        self.assertEqual(expected_path, actual_metadata.path)
        self.assertEqual("A", actual_metadata.parent)
        self.assertEqual("m.json", actual_metadata.name)
        self.assertEqual(expected_uuid, actual_metadata.id)
        self.assertEqual(expected_version, actual_metadata.version_id)

        # Check geoscience object.
        self.assertEqual(expected_object_dict, actual_object.as_dict())
        self.assertIs(actual_object.schema, actual_metadata.schema_id)

    async def test_download_object_alternate_validation(self) -> None:
        """Test downloading an object with alternate representation of confusable types.

        E.g.:
            - floats as `1` instead of `1.0`
            - integers as `1.0` instead of `1`
            - data (parquet file) ID as UUID instead of sha256 checksum
        """
        get_object_response = load_test_data("get_object_validator_check.json")
        expected_uuid = UUID(int=2)
        expected_object_dict = {
            "schema": "/objects/pointset/1.0.1/pointset.schema.json",
            "uuid": UUID("00000000-0000-0000-0000-000000000002"),
            "name": "Sample pointset",
            "description": "A sample pointset object with alternate representations of confusable types",
            "bounding_box": {"min_x": 0, "max_x": 1, "min_y": 0, "max_y": 1, "min_z": 0, "max_z": 1},
            "coordinate_reference_system": "unspecified",
            "locations": {
                "coordinates": {
                    "data": "00000000-0000-0000-0000-000000000003",
                    "length": 1.0,
                    "width": 3.0,
                    "data_type": "float64",
                }
            },
        }
        expected_path = "A/m.json"
        expected_version = "2023-08-03T05:47:18.3402289Z"
        with self.transport.set_http_response(status_code=200, content=json.dumps(get_object_response)):
            actual_object = await self.object_client.download_object_by_path(expected_path, expected_version)

        self.assert_request_made(
            method=RequestMethod.GET,
            path=f"{self.base_path}/objects/path/{expected_path}?version={quote(expected_version)}",
            headers={"Accept": "application/json", "Accept-Encoding": "gzip"},
        )
        # Check metadata.
        actual_metadata = actual_object.metadata
        self.assertEqual(expected_path, actual_metadata.path)
        self.assertEqual("A", actual_metadata.parent)
        self.assertEqual("m.json", actual_metadata.name)
        self.assertEqual(expected_uuid, actual_metadata.id)
        self.assertEqual(expected_version, actual_metadata.version_id)

        # Check geoscience object.
        self.assertEqual(expected_object_dict, actual_object.as_dict())
        self.assertIs(actual_object.schema, actual_metadata.schema_id)

    async def test_delete_object_by_path(self) -> None:
        expected_path = "A/m.json"
        with self.transport.set_http_response(status_code=204):
            actual_object = await self.object_client.delete_object_by_path(expected_path)

        self.assert_request_made(
            method=RequestMethod.DELETE,
            path=f"{self.base_path}/objects/path/{expected_path}",
            headers={"Accept-Encoding": "gzip"},
        )
        self.assertIsNone(actual_object)

    async def test_delete_object_by_id(self) -> None:
        expected_uuid = UUID(int=2)
        with self.transport.set_http_response(status_code=204):
            actual_object = await self.object_client.delete_object_by_id(expected_uuid)

        self.assert_request_made(
            method=RequestMethod.DELETE,
            path=f"{self.base_path}/objects/{expected_uuid}",
            headers={"Accept-Encoding": "gzip"},
        )
        self.assertIsNone(actual_object)

    async def test_restore_geoscience_object(self) -> None:
        expected_uuid = UUID(int=2)
        with self.transport.set_http_response(status_code=204):
            result = await self.object_client.restore_geoscience_object(expected_uuid)
            # The service returns no content on a successful restore without rename.
            assert result is None

        self.assert_request_made(
            method=RequestMethod.POST,
            path=f"{self.base_path}/objects/{expected_uuid}?deleted=False",
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
        )

    async def test_restore_geoscience_object_with_rename(self) -> None:
        get_object_response = load_test_data("get_object.json")
        expected_uuid = UUID(int=2)
        # Given a server response that is a 303 redirect with the updated (post-rename) object metadata...
        with self.transport.set_http_response(status_code=303, content=json.dumps(get_object_response)):
            # ...the restored object metadata should be returned.
            restored_object_metadata = await self.object_client.restore_geoscience_object(expected_uuid)
            assert restored_object_metadata is not None

        self.assert_request_made(
            method=RequestMethod.POST,
            path=f"{self.base_path}/objects/{expected_uuid}?deleted=False",
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
        )
        self.assertEqual("A", restored_object_metadata.parent)
        self.assertEqual("m.json", restored_object_metadata.name)
        self.assertEqual(expected_uuid, restored_object_metadata.id)
        self.assertEqual("2023-08-03T05:47:18.3402289Z", restored_object_metadata.version_id)

    async def test_list_stages(self) -> None:
        list_stages_response = load_test_data("list_stages.json")

        with self.transport.set_http_response(
            200, json.dumps(list_stages_response), headers={"Content-Type": "application/json"}
        ):
            stages = await self.object_client.list_stages()

        self.assert_request_made(
            method=RequestMethod.GET,
            path=f"{self.instance_base_path}/stages",
            headers={"Accept": "application/json"},
        )

        self.assertListEqual(
            [
                Stage(id=UUID(int=1), name="Approved"),
                Stage(id=UUID(int=2), name="Experimental"),
                Stage(id=UUID(int=3), name="In Review"),
                Stage(id=UUID(int=4), name="Peer Review"),
                Stage(id=UUID(int=5), name="Preliminary Update"),
                Stage(id=UUID(int=6), name="Resource Ready"),
            ],
            stages,
        )

    async def test_set_stage(self) -> None:
        object_id = UUID(int=0)
        version_id = 1
        stage_id = UUID(int=2)

        with self.transport.set_http_response(status_code=204):
            await self.object_client.set_stage(object_id, version_id, stage_id)

        self.assert_request_made(
            method=RequestMethod.PATCH,
            path=f"{self.base_path}/objects/{object_id}/metadata?version_id={version_id}",
            headers={"Content-Type": "application/json"},
            body={"stage_id": str(stage_id)},
        )
