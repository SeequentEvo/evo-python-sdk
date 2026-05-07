import json

from evo.common import RequestMethod
from evo.common.test_tools import MockResponse, TestWithConnector
from evo.common.utils import get_header_metadata
from evo.workspaces import (
    BulkUserRoleAssignment,
    BulkUserRoleAssignmentRequest,
    User,
    UserRole,
    WorkspaceAPIClient,
    WorkspaceRole,
)

from ...data import load_test_data
from ..consts import (
    ADMIN_BASE_PATH,
    ORG_UUID,
    USER_ID,
)
from ..data import (
    INSTANCE_USER_1,
    TEST_WORKSPACE_A,
    TEST_WORKSPACE_B,
)


class TestWorkspaceClientAdminEndpoints(TestWithConnector):
    def setUp(self) -> None:
        super().setUp()
        self.workspace_client = WorkspaceAPIClient(connector=self.connector, org_id=ORG_UUID)
        self.setup_universal_headers(get_header_metadata(WorkspaceAPIClient.__module__))

    async def test_admin_bulk_assign_roles(self):
        request = BulkUserRoleAssignmentRequest(
            role_assignments=[
                BulkUserRoleAssignment(
                    workspace_id=TEST_WORKSPACE_A.id,
                    role=WorkspaceRole.editor,
                    user_id=INSTANCE_USER_1.user_id,
                )
            ]
        )
        with self.transport.set_http_response(200):
            response = await self.workspace_client.admin_bulk_assign_roles(request)
        self.assert_request_made(
            method=RequestMethod.POST,
            path=f"{ADMIN_BASE_PATH}/action/bulk_assign_roles",
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            body={
                "role_assignments": [
                    {
                        "role": "editor",
                        "user_id": str(INSTANCE_USER_1.user_id),
                        "workspace_id": str(TEST_WORKSPACE_A.id),
                    }
                ]
            },
        )
        self.assertIsNone(response, "Bulk assign roles response should be None")

    async def test_admin_list_user_workspaces(self):
        content = load_test_data("list_user_workspaces_admin.json")
        with self.transport.set_http_response(200, json.dumps(content), headers={"Content-Type": "application/json"}):
            workspaces = await self.workspace_client.admin_list_user_workspaces(user_id=INSTANCE_USER_1.user_id)
        self.assert_request_made(
            method=RequestMethod.GET,
            path=f"{ADMIN_BASE_PATH}/users/{INSTANCE_USER_1.user_id}/workspaces?limit=100&offset=0",
            headers={"Accept": "application/json"},
        )
        self.assertEqual(1, self.transport.request.call_count, "One requests should be made.")
        self.assertEqual([TEST_WORKSPACE_A, TEST_WORKSPACE_B], workspaces.items())

    async def test_admin_list_workspaces(self):
        content = load_test_data("list_workspaces_0.json")
        with self.transport.set_http_response(200, json.dumps(content), headers={"Content-Type": "application/json"}):
            workspaces = await self.workspace_client.admin_list_workspaces()
            self.assert_request_made(
                method=RequestMethod.GET,
                path=f"{ADMIN_BASE_PATH}/workspaces?limit=100&offset=0",
                headers={"Accept": "application/json"},
            )
            self.assertEqual(1, self.transport.request.call_count, "One requests should be made.")
            self.assertEqual([TEST_WORKSPACE_A, TEST_WORKSPACE_B], workspaces.items())

    async def test_admin_get_workspace(self):
        content = load_test_data("get_workspace.json")
        with self.transport.set_http_response(
            200,
            json.dumps(content),
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
        ):
            workspace = await self.workspace_client.admin_get_workspace(workspace_id=TEST_WORKSPACE_A.id)

        self.assert_request_made(
            method=RequestMethod.GET,
            path=f"{ADMIN_BASE_PATH}/workspaces/{TEST_WORKSPACE_A.id}?deleted=False",
            headers={"Accept": "application/json"},
        )
        self.assertEqual(TEST_WORKSPACE_A, workspace)

    async def test_admin_get_thumbnail(self):
        thumbnail_bytes: bytearray = load_test_data("thumbnail.jpg")
        self.transport.request.return_value = MockResponse(
            status_code=200,
            body=thumbnail_bytes,
            headers={"Content-Type": "image/jpeg"},
        )
        response = await self.workspace_client.admin_get_thumbnail(workspace_id=TEST_WORKSPACE_A.id)
        self.assert_request_made(
            method=RequestMethod.GET,
            path=f"{ADMIN_BASE_PATH}/workspaces/{TEST_WORKSPACE_A.id}/thumbnail",
            headers={"accept": "image/jpeg"},
        )
        self.assertEqual(response, thumbnail_bytes)

    async def test_admin_list_users(self):
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
            response = await self.workspace_client.admin_list_users(workspace_id=TEST_WORKSPACE_A.id)
        self.assert_request_made(
            method=RequestMethod.GET,
            path=f"{ADMIN_BASE_PATH}/workspaces/{TEST_WORKSPACE_A.id}/users",
            headers={"Accept": "application/json"},
        )
        self.assertEqual(
            response,
            [
                User(user_id=USER_ID, role=WorkspaceRole.owner, full_name="Test User", email="test@example.com"),
            ],
        )

    async def test_admin_assign_user_role(self):
        with self.transport.set_http_response(
            201,
            json.dumps(
                {
                    "user_id": str(USER_ID),
                    "role": "owner",
                }
            ),
        ):
            response = await self.workspace_client.admin_assign_user_role(
                workspace_id=TEST_WORKSPACE_A.id,
                user_id=USER_ID,
                role=WorkspaceRole.owner,
            )
        self.assert_request_made(
            method=RequestMethod.POST,
            path=f"{ADMIN_BASE_PATH}/workspaces/{TEST_WORKSPACE_A.id}/users",
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

    async def test_admin_remove_user_from_workspace(self):
        with self.transport.set_http_response(204):
            response = await self.workspace_client.admin_remove_user_from_workspace(TEST_WORKSPACE_A.id, USER_ID)

        self.assert_request_made(
            method=RequestMethod.DELETE,
            path=f"{ADMIN_BASE_PATH}/workspaces/{TEST_WORKSPACE_A.id}/users/{USER_ID}",
        )
        self.assertIsNone(response, "delete response should be empty.")
