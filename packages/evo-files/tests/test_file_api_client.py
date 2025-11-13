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
from unittest import mock
from uuid import UUID

from data import load_test_data
from evo.common import Environment, HealthCheckType, Page, RequestMethod, ServiceUser
from evo.common.test_tools import BASE_URL, ORG, WORKSPACE_ID, MockResponse, TestWithConnector, utc_datetime
from evo.common.utils import get_header_metadata
from evo.files import FileAPIClient, FileAPIDownload, FileAPIUpload, FileMetadata, FileVersion


class TestFileApiClient(TestWithConnector):
    def setUp(self) -> None:
        TestWithConnector.setUp(self)
        self.environment = Environment(hub_url=BASE_URL, org_id=ORG.id, workspace_id=WORKSPACE_ID)
        self.file_api_client = FileAPIClient(connector=self.connector, environment=self.environment)
        self.setup_universal_headers(get_header_metadata(FileAPIClient.__module__))

    @property
    def base_path(self) -> str:
        return f"file/v2/orgs/{self.environment.org_id}/workspaces/{self.environment.workspace_id}"

    async def test_check_service_health(self) -> None:
        """Test service health check implementation"""
        with mock.patch("evo.files.client.get_service_health", spec_set=True) as mock_get_service_health:
            await self.file_api_client.get_service_health()
        mock_get_service_health.assert_called_once_with(self.connector, "file", check_type=HealthCheckType.FULL)

    async def test_list_files_default_args(self) -> None:
        empty_content = load_test_data("list_files_empty.json")
        with self.transport.set_http_response(
            200,
            json.dumps(empty_content),
            headers={"Content-Type": "application/json"},
        ):
            page = await self.file_api_client.list_files()
        self.assertIsInstance(page, Page)
        self.assertEqual([], page.items())
        self.assert_request_made(
            method=RequestMethod.GET,
            path=f"{self.base_path}/files?limit=5000&offset=0",
            headers={"Accept": "application/json"},
        )

    async def test_list_files_all_args(self) -> None:
        empty_content = load_test_data("list_files_empty.json")
        with self.transport.set_http_response(
            200,
            json.dumps(empty_content),
            headers={"Content-Type": "application/json"},
        ):
            page = await self.file_api_client.list_files(limit=20, offset=10, name="x.csv")
        self.assertIsInstance(page, Page)
        self.assertEqual([], page.items())
        self.assert_request_made(
            method=RequestMethod.GET,
            path=f"{self.base_path}/files?limit=20&offset=10&file_name=x.csv",
            headers={"Accept": "application/json"},
        )

    async def test_list_files(self) -> None:
        content_0 = load_test_data("list_files_0.json")
        content_1 = load_test_data("list_files_1.json")
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
        page_one = await self.file_api_client.list_files(limit=2)
        expected_files_page_one = [
            FileMetadata(
                environment=self.environment,
                id=UUID(int=2),
                name="x.csv",
                created_by=ServiceUser(
                    id=UUID(int=16),
                    name="x y",
                    email="test@example.com",
                ),
                created_at=utc_datetime(2020, 1, 1, 1, 30),
                modified_by=ServiceUser(
                    id=UUID(int=16),
                    name="x y",
                    email="test@example.com",
                ),
                modified_at=utc_datetime(2020, 1, 1, 1, 30),
                parent="/A/",
                size=11,
                version_id="2",
            ),
            FileMetadata(
                environment=self.environment,
                id=UUID(int=3),
                name="y.csv",
                created_by=ServiceUser(
                    id=UUID(int=17),
                    name=None,
                    email=None,
                ),
                created_at=utc_datetime(2020, 1, 2, 1, 30),
                modified_by=ServiceUser(
                    id=UUID(int=17),
                    name=None,
                    email=None,
                ),
                modified_at=utc_datetime(2020, 1, 2, 1, 30),
                parent="/A/",
                size=12,
                version_id="1",
            ),
        ]
        self.assertIsInstance(page_one, Page)
        self.assertEqual(expected_files_page_one, page_one.items())
        self.assertEqual(0, page_one.offset)
        self.assertEqual(2, page_one.limit)
        self.assertFalse(page_one.is_last)
        self.assert_request_made(
            method=RequestMethod.GET,
            path=f"{self.base_path}/files?limit=2&offset=0",
            headers={"Accept": "application/json"},
        )
        self.transport.request.reset_mock()

        page_two = await self.file_api_client.list_files(offset=page_one.next_offset, limit=page_one.limit)
        expected_files_page_two = [
            FileMetadata(
                environment=self.environment,
                id=UUID(int=4),
                name="z.csv",
                created_by=ServiceUser(
                    id=UUID(int=16),
                    name="x y",
                    email="test@example.com",
                ),
                created_at=utc_datetime(2020, 1, 1, 1, 30),
                modified_by=ServiceUser(
                    id=UUID(int=16),
                    name="x y",
                    email="test@example.com",
                ),
                modified_at=utc_datetime(2020, 1, 1, 1, 30),
                parent="/B/",
                size=13,
                version_id="3",
            ),
        ]
        self.assertIsInstance(page_two, Page)
        self.assertEqual(expected_files_page_two, page_two.items())
        self.assertEqual(2, page_two.offset)
        self.assertEqual(2, page_two.limit)
        self.assertTrue(page_two.is_last)
        self.assert_request_made(
            method=RequestMethod.GET,
            path=f"{self.base_path}/files?limit=2&offset=2",
            headers={"Accept": "application/json"},
        )

    async def test_list_all_files(self) -> None:
        content_0 = load_test_data("list_files_0.json")
        content_1 = load_test_data("list_files_1.json")
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
        file_list = await self.file_api_client.list_all_files(limit_per_request=2)
        expected_files = [
            FileMetadata(
                environment=self.environment,
                id=UUID(int=2),
                name="x.csv",
                created_by=ServiceUser(
                    id=UUID(int=16),
                    name="x y",
                    email="test@example.com",
                ),
                created_at=utc_datetime(2020, 1, 1, 1, 30),
                modified_by=ServiceUser(
                    id=UUID(int=16),
                    name="x y",
                    email="test@example.com",
                ),
                modified_at=utc_datetime(2020, 1, 1, 1, 30),
                parent="/A/",
                size=11,
                version_id="2",
            ),
            FileMetadata(
                environment=self.environment,
                id=UUID(int=3),
                name="y.csv",
                created_by=ServiceUser(
                    id=UUID(int=17),
                    name=None,
                    email=None,
                ),
                created_at=utc_datetime(2020, 1, 2, 1, 30),
                modified_by=ServiceUser(
                    id=UUID(int=17),
                    name=None,
                    email=None,
                ),
                modified_at=utc_datetime(2020, 1, 2, 1, 30),
                parent="/A/",
                size=12,
                version_id="1",
            ),
            FileMetadata(
                environment=self.environment,
                id=UUID(int=4),
                name="z.csv",
                created_by=ServiceUser(
                    id=UUID(int=16),
                    name="x y",
                    email="test@example.com",
                ),
                created_at=utc_datetime(2020, 1, 1, 1, 30),
                modified_by=ServiceUser(
                    id=UUID(int=16),
                    name="x y",
                    email="test@example.com",
                ),
                modified_at=utc_datetime(2020, 1, 1, 1, 30),
                parent="/B/",
                size=13,
                version_id="3",
            ),
        ]

        self.assertEqual(expected_files, file_list)
        self.assert_any_request_made(
            method=RequestMethod.GET,
            path=f"{self.base_path}/files?limit=2&offset=0",
            headers={"Accept": "application/json"},
        )
        self.assert_any_request_made(
            method=RequestMethod.GET,
            path=f"{self.base_path}/files?limit=2&offset=2",
            headers={"Accept": "application/json"},
        )

    async def test_get_file_by_path(self) -> None:
        get_file_response = load_test_data("get_file.json")
        path = "points.csv"
        expected_metadata = FileMetadata(
            environment=self.environment,
            id=UUID(int=6),
            name="points.csv",
            created_by=ServiceUser(
                id=UUID(int=16),
                name="x y",
                email="test@example.com",
            ),
            created_at=utc_datetime(2020, 1, 1, 1, 30),
            modified_by=ServiceUser(
                id=UUID(int=16),
                name="x y",
                email="test@example.com",
            ),
            modified_at=utc_datetime(2020, 1, 1, 1, 30),
            parent="/",
            size=10,
            version_id="1",
        )

        with self.transport.set_http_response(
            status_code=200,
            content=json.dumps(get_file_response),
            headers={"Content-Type": "application/json"},
        ):
            actual_metadata = await self.file_api_client.get_file_by_path(path)

        self.assert_request_made(
            method=RequestMethod.GET,
            path=f"{self.base_path}/files/path/{path}",
            headers={"Accept": "application/json"},
        )
        self.assertEqual(expected_metadata, actual_metadata)

    async def test_prepare_download_by_path(self) -> None:
        get_file_response = load_test_data("get_file.json")
        path = "points.csv"
        expected_metadata = FileMetadata(
            environment=self.environment,
            id=UUID(int=6),
            name="points.csv",
            created_by=ServiceUser(
                id=UUID(int=16),
                name="x y",
                email="test@example.com",
            ),
            created_at=utc_datetime(2020, 1, 1, 1, 30),
            modified_by=ServiceUser(
                id=UUID(int=16),
                name="x y",
                email="test@example.com",
            ),
            modified_at=utc_datetime(2020, 1, 1, 1, 30),
            parent="/",
            size=10,
            version_id="1",
        )

        with self.transport.set_http_response(
            status_code=200,
            content=json.dumps(get_file_response),
            headers={"Content-Type": "application/json"},
        ):
            download = await self.file_api_client.prepare_download_by_path(path)

        self.assertIsInstance(download, FileAPIDownload)
        self.assertEqual(expected_metadata, download.metadata)
        self.assert_request_made(
            method=RequestMethod.GET,
            path=f"{self.base_path}/files/path/{path}",
            headers={"Accept": "application/json"},
        )

    async def test_get_file_by_id(self) -> None:
        get_file_response = load_test_data("get_file.json")
        file_id = UUID(int=6)
        expected_metadata = FileMetadata(
            environment=self.environment,
            id=file_id,
            name="points.csv",
            created_by=ServiceUser(
                id=UUID(int=16),
                name="x y",
                email="test@example.com",
            ),
            created_at=utc_datetime(2020, 1, 1, 1, 30),
            modified_by=ServiceUser(
                id=UUID(int=16),
                name="x y",
                email="test@example.com",
            ),
            modified_at=utc_datetime(2020, 1, 1, 1, 30),
            parent="/",
            size=10,
            version_id="1",
        )

        with self.transport.set_http_response(
            status_code=200,
            content=json.dumps(get_file_response),
            headers={"Content-Type": "application/json"},
        ):
            actual_metadata = await self.file_api_client.get_file_by_id(file_id)

        self.assert_request_made(
            method=RequestMethod.GET,
            path=f"{self.base_path}/files/{file_id}",
            headers={"Accept": "application/json"},
        )
        self.assertEqual(expected_metadata, actual_metadata)

    async def test_prepare_download_by_id(self) -> None:
        get_file_response = load_test_data("get_file.json")
        file_id = UUID(int=6)
        expected_metadata = FileMetadata(
            environment=self.environment,
            id=file_id,
            name="points.csv",
            created_by=ServiceUser(
                id=UUID(int=16),
                name="x y",
                email="test@example.com",
            ),
            created_at=utc_datetime(2020, 1, 1, 1, 30),
            modified_by=ServiceUser(
                id=UUID(int=16),
                name="x y",
                email="test@example.com",
            ),
            modified_at=utc_datetime(2020, 1, 1, 1, 30),
            parent="/",
            size=10,
            version_id="1",
        )

        with self.transport.set_http_response(
            status_code=200,
            content=json.dumps(get_file_response),
            headers={"Content-Type": "application/json"},
        ):
            download = await self.file_api_client.prepare_download_by_id(file_id)

        self.assertIsInstance(download, FileAPIDownload)
        self.assertEqual(expected_metadata, download.metadata)
        self.assert_request_made(
            method=RequestMethod.GET,
            path=f"{self.base_path}/files/{file_id}",
            headers={"Accept": "application/json"},
        )

    async def test_list_versions_by_path(self) -> None:
        file_path = "points.csv"
        list_versions_response = load_test_data("list_versions.json")
        with self.transport.set_http_response(
            200,
            json.dumps(list_versions_response),
            headers={"Content-Type": "application/json"},
        ):
            versions = await self.file_api_client.list_versions_by_path(file_path)
        expected_versions = [
            FileVersion(
                version_id="3",
                created_at=utc_datetime(2020, 1, 3, 1, 30),
                created_by=ServiceUser(
                    id=UUID(int=16),
                    name="x y",
                    email="test@example.com",
                ),
            ),
            FileVersion(
                version_id="2",
                created_at=utc_datetime(2020, 1, 2, 1, 30),
                created_by=ServiceUser(
                    id=UUID(int=17),
                    name="x z",
                    email="test@example.com",
                ),
            ),
            FileVersion(
                version_id="1",
                created_at=utc_datetime(2020, 1, 1, 1, 30),
                created_by=ServiceUser(
                    id=UUID("00000000-0000-0000-0000-000000000012"),
                    name="x w",
                    email="test@example.com",
                ),
            ),
        ]

        self.assertEqual(versions, expected_versions)
        self.assert_request_made(
            method=RequestMethod.GET,
            path=f"{self.base_path}/files/path/points.csv?include_versions=True",
            headers={"Accept": "application/json"},
        )

    async def test_list_versions_by_id(self) -> None:
        file_id = UUID(int=6)
        list_versions_response = load_test_data("list_versions.json")
        with self.transport.set_http_response(
            200,
            json.dumps(list_versions_response),
            headers={"Content-Type": "application/json"},
        ):
            versions = await self.file_api_client.list_versions_by_id(file_id)
        expected_versions = [
            FileVersion(
                version_id="3",
                created_at=utc_datetime(2020, 1, 3, 1, 30),
                created_by=ServiceUser(
                    id=UUID(int=16),
                    name="x y",
                    email="test@example.com",
                ),
            ),
            FileVersion(
                version_id="2",
                created_at=utc_datetime(2020, 1, 2, 1, 30),
                created_by=ServiceUser(
                    id=UUID(int=17),
                    name="x z",
                    email="test@example.com",
                ),
            ),
            FileVersion(
                version_id="1",
                created_at=utc_datetime(2020, 1, 1, 1, 30),
                created_by=ServiceUser(
                    id=UUID("00000000-0000-0000-0000-000000000012"),
                    name="x w",
                    email="test@example.com",
                ),
            ),
        ]

        self.assertEqual(versions, expected_versions)
        self.assert_request_made(
            method=RequestMethod.GET,
            path=f"{self.base_path}/files/{file_id}?include_versions=True",
            headers={"Accept": "application/json"},
        )

    async def test_prepare_upload_by_path(self) -> None:
        upsert_file_response = load_test_data("upsert_file.json")
        with self.transport.set_http_response(
            status_code=200,
            content=json.dumps(upsert_file_response),
            headers={"Content-Type": "application/json"},
        ):
            upload = await self.file_api_client.prepare_upload_by_path("points.csv")

        self.assertIsInstance(upload, FileAPIUpload)
        self.assertEqual(upsert_file_response["file_id"], str(upload.file_id))
        self.assertEqual(upsert_file_response["version_id"], upload.version_id)
        self.assert_request_made(
            method=RequestMethod.PUT,
            path=f"{self.base_path}/files/path/points.csv",
            headers={"Accept": "application/json"},
        )

    async def test_update_file_by_id(self) -> None:
        update_file_response = load_test_data("update_file.json")
        file_id = UUID(int=5)
        with self.transport.set_http_response(
            status_code=200,
            content=json.dumps(update_file_response),
            headers={"Content-Type": "application/json"},
        ):
            upload = await self.file_api_client.prepare_upload_by_id(file_id)

        self.assertIsInstance(upload, FileAPIUpload)
        self.assertEqual(update_file_response["file_id"], str(upload.file_id))
        self.assertEqual(update_file_response["version_id"], upload.version_id)
        self.assert_request_made(
            method=RequestMethod.PUT,
            path=f"{self.base_path}/files/{file_id}",
            headers={"Accept": "application/json"},
        )

    async def test_delete_file_by_path(self) -> None:
        path = "points.csv"
        with self.transport.set_http_response(
            status_code=204,
            content="",
            headers={"Content-Type": "application/json"},
        ):
            await self.file_api_client.delete_file_by_path(path)
        self.assert_request_made(
            method=RequestMethod.DELETE,
            path=f"{self.base_path}/files/path/{path}",
        )

    async def test_delete_file_by_id(self) -> None:
        file_id = UUID(int=6)
        with self.transport.set_http_response(
            status_code=204,
            content="",
            headers={"Content-Type": "application/json"},
        ):
            await self.file_api_client.delete_file_by_id(file_id)
        self.assert_request_made(
            method=RequestMethod.DELETE,
            path=f"{self.base_path}/files/{file_id}",
        )
