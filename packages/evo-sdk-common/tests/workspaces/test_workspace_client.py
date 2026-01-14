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

from evo.common import HealthCheckType, RequestMethod, StaticContext
from evo.common.exceptions import ContextError
from evo.common.test_tools import BASE_URL, MockResponse, TestHTTPHeaderDict, TestWithConnector, utc_datetime
from evo.common.utils import get_header_metadata
from evo.workspaces import (
    AddedInstanceUsers,
    BasicWorkspace,
    InstanceRole,
    InstanceRoleWithPermissions,
    InstanceUser,
    InstanceUserInvitation,
    InstanceUserWithEmail,
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


def _test_instance_role(role_id: UUID, name: str) -> InstanceRole:
    """Factory method to create test instance role objects."""
    return InstanceRole(
        role_id=role_id,
        name=name.title(),
        description=name.lower(),
    )


def _test_instance_role_with_permissions(role_id: UUID, name: str) -> InstanceRoleWithPermissions:
    """Factory method to create test instance role objects."""
    return InstanceRoleWithPermissions(
        role_id=role_id, name=name.title(), description=name.lower(), permissions=[name.lower() + " permission"]
    )


def _test_instance_user(user_id: UUID, role_name: str, role_id: int) -> InstanceUser:
    """Factory method to create test instance user objects."""
    return InstanceUser(user_id=user_id, roles=[_test_instance_role(UUID(int=role_id), role_name)])


def _test_instance_user_with_email(
    user_id: UUID, email: str, full_name: str, role_name: str, role_id: int
) -> InstanceUserWithEmail:
    """Factory method to create test instance user objects."""
    return InstanceUserWithEmail(
        user_id=user_id, email=email, full_name=full_name, roles=[_test_instance_role(UUID(int=role_id), role_name)]
    )


def _test_instance_user_invitation(
    invitation_id: UUID, email: str, status: str, role_name: str, role_id: int
) -> InstanceUserInvitation:
    """Factory method to create test instance user invitation objects."""
    return InstanceUserInvitation(
        email=email,
        invitation_id=invitation_id,
        invited_at=utc_datetime(2026, 1, 1, 12, 0, 0),
        expiration_date=utc_datetime(2026, 1, 15, 12, 0, 0),
        invited_by="admin.user@bentley.com",
        status=status,
        roles=[_test_instance_role(UUID(int=role_id), role_name)],
    )


TEST_WORKSPACE_A = _test_workspace(UUID(int=0xA), "Test Workspace A")
TEST_WORKSPACE_B = _test_workspace(UUID(int=0xB), "Test Workspace B")
TEST_WORKSPACE_C = _test_workspace(UUID(int=0xC), "Test Workspace C")
TEST_BASIC_WORKSPACE_A = _test_basic_workspace(UUID(int=0xA), "Test Workspace A")
TEST_BASIC_WORKSPACE_B = _test_basic_workspace(UUID(int=0xB), "Test Workspace B")
TEST_BASIC_WORKSPACE_C = _test_basic_workspace(UUID(int=0xC), "Test Workspace C")

INSTANCE_USER_1 = _test_instance_user_with_email(UUID(int=1), "test.user1@gmail.com", "User 1", "Evo Owner", 3)
INSTANCE_USER_2 = _test_instance_user_with_email(UUID(int=2), "test.user2@gmail.com", "User 2", "Evo Admin", 2)
INSTANCE_USER_3 = _test_instance_user_with_email(UUID(int=3), "test.user3@gmail.com", "User 3", "Evo User", 1)
INVITATION_1 = _test_instance_user_invitation(UUID(int=1), "external.user1@gmail.com", "Pending", "Evo User", 1)
INVITATION_2 = _test_instance_user_invitation(UUID(int=2), "external.user2@gmail.com", "Accepted", "Evo Admin", 2)
INVITATION_3 = _test_instance_user_invitation(UUID(int=3), "external.user3@gmail.com", "Pending", "Evo User", 1)

INSTANCE_USER_ROLE = _test_instance_role_with_permissions(UUID(int=1), "Evo User")
INSTANCE_ADMIN_ROLE = _test_instance_role_with_permissions(UUID(int=2), "Evo Admin")


class TestWorkspaceClient(TestWithConnector):
    def setUp(self) -> None:
        super().setUp()
        self.workspace_client = WorkspaceAPIClient(connector=self.connector, org_id=ORG_UUID)
        self.setup_universal_headers(get_header_metadata(WorkspaceAPIClient.__module__))

    def test_from_context(self):
        client = WorkspaceAPIClient.from_context(StaticContext(connector=self.connector, org_id=ORG_UUID))
        self.assertIsInstance(client, WorkspaceAPIClient)

    def test_from_context_missing_org_id(self):
        with self.assertRaises(ContextError):
            WorkspaceAPIClient.from_context(StaticContext(connector=self.connector))

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

    async def test_list_instance_users(self) -> None:
        content_1 = load_test_data("instance_users_page_1.json")
        content_2 = load_test_data("instance_users_page_2.json")

        with self.transport.set_http_response(200, json.dumps(content_1), headers={"Content-Type": "application/json"}):
            users_page_1 = await self.workspace_client.list_instance_users(limit=2, offset=0)

        with self.transport.set_http_response(200, json.dumps(content_2), headers={"Content-Type": "application/json"}):
            users_page_2 = await self.workspace_client.list_instance_users(limit=2, offset=2)

        self.assert_any_request_made(
            method=RequestMethod.GET,
            path=f"{BASE_PATH}/members?limit=2&offset=0",
            headers={"Accept": "application/json"},
        )
        self.assert_any_request_made(
            method=RequestMethod.GET,
            path=f"{BASE_PATH}/members?limit=2&offset=2",
            headers={"Accept": "application/json"},
        )
        self.assertEqual(2, self.transport.request.call_count, "Two requests should be made.")
        self.assertEqual([INSTANCE_USER_1, INSTANCE_USER_2], users_page_1.items())
        self.assertEqual([INSTANCE_USER_3], users_page_2.items())

    async def test_list_instance_user_invitations(self) -> None:
        content = load_test_data("invitations_page_1.json")

        with self.transport.set_http_response(200, json.dumps(content), headers={"Content-Type": "application/json"}):
            invitations = await self.workspace_client.list_instance_user_invitations(limit=2, offset=0)

        self.assert_request_made(
            method=RequestMethod.GET,
            path=f"{BASE_PATH}/members/invitations?limit=2&offset=0",
            headers={"Accept": "application/json"},
        )
        self.assertEqual([INVITATION_1, INVITATION_2], invitations.items())

    async def test_add_users_to_instance(self) -> None:
        add_users_content = load_test_data("add_instance_users.json")

        with self.transport.set_http_response(
            201,
            json.dumps(add_users_content),
            headers={"Content-Type": "application/json"},
        ):
            response = await self.workspace_client.add_users_to_instance(
                users={
                    INSTANCE_USER_2.email: [INSTANCE_USER_2.roles[0].role_id],
                    INVITATION_1.email: [INVITATION_1.roles[0].role_id],
                }
            )
        self.assert_request_made(
            method=RequestMethod.POST,
            path=f"{BASE_PATH}/members",
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            body={
                "users": [
                    {
                        "email": INSTANCE_USER_2.email,
                        "roles": [str(INSTANCE_USER_2.roles[0].role_id)],
                    },
                    {
                        "email": INVITATION_1.email,
                        "roles": [str(INVITATION_1.roles[0].role_id)],
                    },
                ]
            },
        )

        self.assertEqual(
            response,
            AddedInstanceUsers(
                members=[INSTANCE_USER_2],
                invitations=[INVITATION_1],
            ),
        )

    async def test_delete_instance_user_invitation(self) -> None:
        with self.transport.set_http_response(204):
            response = await self.workspace_client.delete_instance_user_invitation(
                invitation_id=INVITATION_1.invitation_id
            )
        self.assert_request_made(
            method=RequestMethod.DELETE,
            path=f"{BASE_PATH}/members/invitations/{INVITATION_1.invitation_id}",
        )
        self.assertIsNone(response, "Delete instance user invitation response should be None")

    async def test_remove_instance_user(self) -> None:
        with self.transport.set_http_response(204):
            response = await self.workspace_client.remove_instance_user(user_id=INSTANCE_USER_1.user_id)
        self.assert_request_made(
            method=RequestMethod.DELETE,
            path=f"{BASE_PATH}/members/{INSTANCE_USER_1.user_id}",
        )
        self.assertIsNone(response, "Remove instance user response should be None")

    async def test_update_instance_user_roles(self) -> None:
        update_users_content = load_test_data("update_instance_user.json")
        with self.transport.set_http_response(
            200,
            json.dumps(update_users_content),
            headers={"Content-Type": "application/json"},
        ):
            response = await self.workspace_client.update_instance_user_roles(
                user_id=INSTANCE_USER_1.user_id,
                roles=[INSTANCE_ADMIN_ROLE.role_id],
            )
        self.assert_request_made(
            method=RequestMethod.PATCH,
            path=f"{BASE_PATH}/members/{INSTANCE_USER_1.user_id}",
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            body={
                "user_id": str(INSTANCE_USER_1.user_id),
                "roles": [str(INSTANCE_ADMIN_ROLE.role_id)],
            },
        )
        self.assertEqual(
            response,
            InstanceUser(
                user_id=INSTANCE_USER_1.user_id,
                roles=[_test_instance_role(INSTANCE_ADMIN_ROLE.role_id, "Evo Admin")],
            ),
        )
