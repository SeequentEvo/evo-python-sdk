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

from evo.common import HealthCheckType, RequestMethod
from evo.common.test_tools import BASE_URL, MockResponse, TestHTTPHeaderDict, TestWithConnector, utc_datetime
from evo.common.utils import get_header_metadata
from evo.workspaces import (
    BasicWorkspace,
    OrderByOperatorEnum,
    ServiceUser,
    User,
    UserRole,
    Workspace,
    WorkspaceAPIClient,
    WorkspaceOrderByEnum,
    WorkspaceRole,
)

from ..data import load_test_data

ORG_UUID = UUID(int=0)
USER_ID = UUID(int=2)
BASE_PATH = f"/workspace/orgs/{ORG_UUID}"

TEST_USER = ServiceUser(id=USER_ID, name="Test User", email="test.user@unit.test")


def _test_workspace(ws_id: UUID, name: str) -> Workspace:
    """Factory method to create test workspace objects."""
    return Workspace(
        id=ws_id,
        display_name=name.title(),
        description=name.lower(),
        user_role=WorkspaceRole.owner,
        org_id=ORG_UUID,
        hub_url=BASE_URL,
        created_at=utc_datetime(2020, 1, 1),
        created_by=TEST_USER,
        updated_at=utc_datetime(2020, 1, 1),
        updated_by=TEST_USER,
    )


def _test_basic_workspace(ws_id: UUID, name: str) -> BasicWorkspace:
    """Factory method to create test basic workspace objects."""
    return BasicWorkspace(
        id=ws_id,
        display_name=name.title(),
    )


TEST_WORKSPACE_A = _test_workspace(UUID(int=0xA), "Test Workspace A")
TEST_WORKSPACE_B = _test_workspace(UUID(int=0xB), "Test Workspace B")
TEST_WORKSPACE_C = _test_workspace(UUID(int=0xC), "Test Workspace C")
TEST_BASIC_WORKSPACE_A = _test_basic_workspace(UUID(int=0xA), "Test Workspace A")
TEST_BASIC_WORKSPACE_B = _test_basic_workspace(UUID(int=0xB), "Test Workspace B")
TEST_BASIC_WORKSPACE_C = _test_basic_workspace(UUID(int=0xC), "Test Workspace C")


class TestWorkspaceClient(TestWithConnector):
    def setUp(self) -> None:
        super().setUp()
        self.workspace_client = WorkspaceAPIClient(connector=self.connector, org_id=ORG_UUID)
        self.setup_universal_headers(get_header_metadata(WorkspaceAPIClient.__module__))

    async def test_get_service_health(self) -> None:
        with mock.patch("evo.workspaces.client.get_service_health") as mock_get_service_health:
            await self.workspace_client.get_service_health()
        mock_get_service_health.assert_called_once_with(self.connector, "workspace", check_type=HealthCheckType.FULL)

    def _empty_content(self) -> str:
        data = """{"results": [], "links": {"first": "http://firstlink", "last": "http://lastlink",
                "next": null, "previous": null, "count": 0, "total": 0}}"""
        return data

    async def test_list_workspaces_default_args(self):
        with self.transport.set_http_response(200, self._empty_content(), headers={"Content-Type": "application/json"}):
            workspaces = await self.workspace_client.list_workspaces()
        self.assert_request_made(
            method=RequestMethod.GET,
            path=f"{BASE_PATH}/workspaces?offset=0",
            headers={"Accept": "application/json"},
        )
        self.assertEqual([], workspaces.items())

    async def test_list_workspaces_all_args(self):
        with self.transport.set_http_response(200, self._empty_content(), headers={"Content-Type": "application/json"}):
            workspaces = await self.workspace_client.list_workspaces(offset=10, limit=20)
        self.assert_request_made(
            method=RequestMethod.GET,
            path=f"{BASE_PATH}/workspaces?limit=20&offset=10",
            headers={"Accept": "application/json"},
        )
        self.assertEqual([], workspaces.items())

    async def test_delete_workspace_call(self):
        with self.transport.set_http_response(204):
            response = await self.workspace_client.delete_workspace(workspace_id=TEST_WORKSPACE_A.id)
        self.assert_request_made(method=RequestMethod.DELETE, path=f"{BASE_PATH}/workspaces/{TEST_WORKSPACE_A.id}")
        self.assertIsNone(response, "Delete workspace response should be None")

    async def test_create_workspace(self):
        with self.transport.set_http_response(201, json.dumps(load_test_data("new_workspace.json"))):
            new_workspace = await self.workspace_client.create_workspace(
                name="Test Workspace",
                description="test workspace",
            )
        self.assert_request_made(
            method=RequestMethod.POST,
            path=f"{BASE_PATH}/workspaces",
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            body={
                "bounding_box": None,
                "default_coordinate_system": "",
                "description": "test workspace",
                "labels": None,
                "name": "Test Workspace",
            },
        )
        self.assertEqual(TEST_WORKSPACE_A, new_workspace)

    async def test_update_workspace(self):
        with self.transport.set_http_response(200, json.dumps(load_test_data("new_workspace.json"))):
            updated_workspace = await self.workspace_client.update_workspace(
                workspace_id=TEST_WORKSPACE_A.id,
                name="Test Workspace",
            )
        self.assert_request_made(
            method=RequestMethod.PATCH,
            path=f"{BASE_PATH}/workspaces/{TEST_WORKSPACE_A.id}",
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            body={
                "name": "Test Workspace",
            },
        )
        self.assertEqual(TEST_WORKSPACE_A, updated_workspace)

    async def test_assign_user_role(self) -> None:
        with self.transport.set_http_response(
            201,
            json.dumps(
                {
                    "user_id": str(USER_ID),
                    "role": "owner",
                }
            ),
        ):
            response = await self.workspace_client.assign_user_role(
                workspace_id=TEST_WORKSPACE_A.id,
                user_id=USER_ID,
                role=WorkspaceRole.owner,
            )
        self.assert_request_made(
            method=RequestMethod.POST,
            path=f"{BASE_PATH}/workspaces/{TEST_WORKSPACE_A.id}/users",
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            body={
                "user_id": str(USER_ID),
                "role": "owner",
            },
        )
        self.assertEqual(response, UserRole(user_id=USER_ID, role=WorkspaceRole.owner))

    async def test_get_current_user_role(self) -> None:
        with self.transport.set_http_response(
            200,
            json.dumps(
                {
                    "user_id": str(USER_ID),
                    "role": "owner",
                }
            ),
        ):
            response = await self.workspace_client.get_current_user_role(workspace_id=TEST_WORKSPACE_A.id)
        self.assert_request_made(
            method=RequestMethod.GET,
            path=f"{BASE_PATH}/workspaces/{TEST_WORKSPACE_A.id}/current-user-role",
            headers={"Accept": "application/json"},
        )
        self.assertEqual(response, UserRole(user_id=USER_ID, role=WorkspaceRole.owner))

    async def test_list_user_roles(self) -> None:
        with self.transport.set_http_response(
            200,
            json.dumps(
                {
                    "results": [
                        {
                            "user_id": str(USER_ID),
                            "role": "owner",
                            "full_name": "Test User",
                            "email": "test@example.com",
                        },
                    ],
                    "links": {"self": "dummy-link.com"},
                }
            ),
        ):
            response = await self.workspace_client.list_user_roles(workspace_id=TEST_WORKSPACE_A.id)

        self.assert_request_made(
            method=RequestMethod.GET,
            path=f"{BASE_PATH}/workspaces/{TEST_WORKSPACE_A.id}/users",
            headers={"Accept": "application/json"},
        )
        self.assertEqual(
            response,
            [
                User(user_id=USER_ID, role=WorkspaceRole.owner, full_name="Test User", email="test@example.com"),
            ],
        )

    async def test_list_workspaces(self) -> None:
        content = load_test_data("list_workspaces_0.json")
        with self.transport.set_http_response(200, json.dumps(content), headers={"Content-Type": "application/json"}):
            workspaces = await self.workspace_client.list_workspaces(limit=2)
        self.assert_request_made(
            method=RequestMethod.GET,
            path=f"{BASE_PATH}/workspaces?limit=2&offset=0",
            headers={"Accept": "application/json"},
        )
        self.assertEqual(1, self.transport.request.call_count, "One requests should be made.")
        self.assertEqual([TEST_WORKSPACE_A, TEST_WORKSPACE_B], workspaces.items())

    async def test_list_all_workspaces(self) -> None:
        content_0 = load_test_data("list_workspaces_0.json")
        content_1 = load_test_data("list_workspaces_1.json")
        responses = [
            MockResponse(status_code=200, content=json.dumps(content_0), headers={"Content-Type": "application/json"}),
            MockResponse(status_code=200, content=json.dumps(content_1), headers={"Content-Type": "application/json"}),
        ]
        self.transport.request.side_effect = responses
        expected_workspaces = [TEST_WORKSPACE_A, TEST_WORKSPACE_B, TEST_WORKSPACE_C]
        actual_workspaces = await self.workspace_client.list_all_workspaces(limit=2)

        self.assertEqual(2, self.transport.request.call_count, "Two requests should be made.")
        self.assert_any_request_made(
            method=RequestMethod.GET,
            path=f"{BASE_PATH}/workspaces?limit=2&offset=0",
            headers=TestHTTPHeaderDict({"Accept": "application/json"}),
        )
        self.assert_any_request_made(
            method=RequestMethod.GET,
            path=f"{BASE_PATH}/workspaces?limit=2&offset=2",
            headers=TestHTTPHeaderDict({"Accept": "application/json"}),
        )
        self.assertEqual(expected_workspaces, actual_workspaces)

    async def test_list_workspaces_sorted_by_display_name(self) -> None:
        content_0 = load_test_data("list_workspaces_0.json")
        content_1 = load_test_data("list_workspaces_1.json")

        # Shuffle the workspaces in the response content.
        results_0: list = content_0["results"]
        results_1: list = content_1["results"]
        results_0.insert(2, results_1.pop(0))
        results_1.insert(0, results_0.pop(0))

        responses = [
            MockResponse(status_code=200, content=json.dumps(content_0), headers={"Content-Type": "application/json"}),
            MockResponse(status_code=200, content=json.dumps(content_1), headers={"Content-Type": "application/json"}),
        ]
        self.transport.request.side_effect = responses
        expected_workspaces = [TEST_WORKSPACE_A, TEST_WORKSPACE_B, TEST_WORKSPACE_C]
        actual_workspaces = await self.workspace_client.list_all_workspaces(limit=2)

        self.assertEqual(2, self.transport.request.call_count, "Two requests should be made.")
        self.assert_any_request_made(
            method=RequestMethod.GET,
            path=f"{BASE_PATH}/workspaces?limit=2&offset=0",
            headers=TestHTTPHeaderDict({"Accept": "application/json"}),
        )
        self.assert_any_request_made(
            method=RequestMethod.GET,
            path=f"{BASE_PATH}/workspaces?limit=2&offset=2",
            headers=TestHTTPHeaderDict({"Accept": "application/json"}),
        )
        self.assertEqual(expected_workspaces, actual_workspaces)

    async def test_list_workspaces_summary_default_args(self):
        with self.transport.set_http_response(200, self._empty_content(), headers={"Content-Type": "application/json"}):
            workspaces = await self.workspace_client.list_workspaces_summary()
        self.assert_request_made(
            method=RequestMethod.GET,
            path=f"{BASE_PATH}/workspaces/summary",
            headers={"Accept": "application/json"},
        )
        self.assertEqual([], workspaces.items())

    async def test_list_workspaces_summary_all_args(self) -> None:
        for order_by in [
            {WorkspaceOrderByEnum.name: OrderByOperatorEnum.asc},
            {"name": "asc"},
        ]:
            with self.transport.set_http_response(
                200, self._empty_content(), headers={"Content-Type": "application/json"}
            ):
                workspaces = await self.workspace_client.list_workspaces_summary(
                    offset=10,
                    limit=20,
                    order_by=order_by,
                    filter_created_by=USER_ID,
                    created_at=str(utc_datetime(2020, 1, 1)),
                    updated_at=str(utc_datetime(2020, 1, 1)),
                    name="Test Workspace A",
                    deleted=False,
                    filter_user_id=USER_ID,
                )
            self.assert_request_made(
                method=RequestMethod.GET,
                path=f"{BASE_PATH}/workspaces/summary?"
                f"limit=20&offset=10&order_by=asc%3Aname&filter%5Bcreated_by%5D=00000000-0000-0000-0000-000000000002&"
                f"created_at=2020-01-01+00%3A00%3A00%2B00%3A00&updated_at=2020-01-01+00%3A00%3A00%2B00%3A00&"
                f"name=Test+Workspace+A&deleted=False&filter%5Buser_id%5D=00000000-0000-0000-0000-000000000002",
                headers={"Accept": "application/json"},
            )
            self.assertEqual([], workspaces.items())

    async def test_list_workspaces_summary(self) -> None:
        content = load_test_data("list_workspaces_summary.json")
        with self.transport.set_http_response(200, json.dumps(content), headers={"Content-Type": "application/json"}):
            workspaces = await self.workspace_client.list_workspaces_summary()
        self.assert_request_made(
            method=RequestMethod.GET,
            path=f"{BASE_PATH}/workspaces/summary",
            headers={"Accept": "application/json"},
        )
        self.assertEqual(1, self.transport.request.call_count, "One requests should be made.")
        self.assertEqual([TEST_BASIC_WORKSPACE_A, TEST_BASIC_WORKSPACE_B, TEST_BASIC_WORKSPACE_C], workspaces.items())

    async def test_paginated_list_workspaces_summary(self) -> None:
        content = load_test_data("list_workspaces_summary_paginated_0.json")
        content_2 = load_test_data("list_workspaces_summary_paginated_1.json")
        with self.transport.set_http_response(200, json.dumps(content), headers={"Content-Type": "application/json"}):
            workspaces_page_1 = await self.workspace_client.list_workspaces_summary(limit=2, offset=0)

        with self.transport.set_http_response(200, json.dumps(content_2), headers={"Content-Type": "application/json"}):
            workspaces_page_2 = await self.workspace_client.list_workspaces_summary(limit=2, offset=2)

        self.assert_any_request_made(
            method=RequestMethod.GET,
            path=f"{BASE_PATH}/workspaces/summary?limit=2&offset=0",
            headers={"Accept": "application/json"},
        )
        self.assert_any_request_made(
            method=RequestMethod.GET,
            path=f"{BASE_PATH}/workspaces/summary?limit=2&offset=2",
            headers={"Accept": "application/json"},
        )
        self.assertEqual(2, self.transport.request.call_count, "Two requests should be made.")
        self.assertEqual([TEST_BASIC_WORKSPACE_A, TEST_BASIC_WORKSPACE_B], workspaces_page_1.items())
        self.assertEqual([TEST_BASIC_WORKSPACE_C], workspaces_page_2.items())
