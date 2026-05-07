import json

from parameterized import parameterized

from evo.common import RequestMethod
from evo.common.test_tools import TestWithConnector
from evo.common.utils import get_header_metadata
from evo.workspaces import (
    User,
    UserRole,
    WorkspaceAPIClient,
    WorkspaceRole,
)

from ..consts import (
    BASE_PATH,
    ORG_UUID,
    USER_ID,
)
from ..data import (
    TEST_WORKSPACE_A,
)


class TestWorkspaceClientUserEndpoints(TestWithConnector):
    def setUp(self) -> None:
        super().setUp()
        self.workspace_client = WorkspaceAPIClient(connector=self.connector, org_id=ORG_UUID)
        self.setup_universal_headers(get_header_metadata(WorkspaceAPIClient.__module__))

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

    @parameterized.expand(
        [
            None,
            USER_ID,
        ]
    )
    async def test_list_user_roles(self, user_id_filter) -> None:
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
            response = await self.workspace_client.list_user_roles(
                workspace_id=TEST_WORKSPACE_A.id, filter_user_id=user_id_filter
            )

        expected_path = f"{BASE_PATH}/workspaces/{TEST_WORKSPACE_A.id}/users"
        if user_id_filter:
            expected_path += f"?user_id={user_id_filter}"

        self.assert_request_made(
            method=RequestMethod.GET,
            path=expected_path,
            headers={"Accept": "application/json"},
        )
        self.assertEqual(
            response,
            [
                User(user_id=USER_ID, role=WorkspaceRole.owner, full_name="Test User", email="test@example.com"),
            ],
        )

    async def test_remove_user_from_workspace(self):
        with self.transport.set_http_response(204):
            response = await self.workspace_client.delete_user_role(TEST_WORKSPACE_A.id, USER_ID)

        self.assert_request_made(
            method=RequestMethod.DELETE,
            path=f"{BASE_PATH}/workspaces/{TEST_WORKSPACE_A.id}/users/{USER_ID}",
        )
        self.assertIsNone(response, "delete response should be empty.")
