import json

from evo.common import RequestMethod
from evo.common.test_tools import TestWithConnector
from evo.common.utils import get_header_metadata
from evo.workspaces import (
    AddedInstanceUsers,
    InstanceUser,
    WorkspaceAPIClient,
)

from ...data import load_test_data
from ..consts import (
    BASE_PATH,
    ORG_UUID,
)
from ..data import (
    INSTANCE_ADMIN_ROLE,
    INSTANCE_USER_1,
    INSTANCE_USER_2,
    INSTANCE_USER_3,
    INSTANCE_USER_ROLE,
    INVITATION_1,
    INVITATION_2,
)
from ..helpers import (
    make_instance_role,
)


class TestWorkspaceClientInstanceUserEndpoints(TestWithConnector):
    def setUp(self) -> None:
        super().setUp()
        self.workspace_client = WorkspaceAPIClient(connector=self.connector, org_id=ORG_UUID)
        self.setup_universal_headers(get_header_metadata(WorkspaceAPIClient.__module__))

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
                roles=[make_instance_role(INSTANCE_ADMIN_ROLE.role_id, "Evo Admin")],
            ),
        )

    async def test_list_instance_user_roles(self):
        content = load_test_data("list_instance_roles.json")
        with self.transport.set_http_response(
            200,
            json.dumps(content),
            headers={"Content-Type": "application/json"},
        ):
            response = await self.workspace_client.list_instance_roles()

        self.assert_request_made(
            method=RequestMethod.GET,
            path=f"{BASE_PATH}/members/roles",
            headers={"Accept": "application/json"},
        )
        self.assertEqual(response, [INSTANCE_USER_ROLE, INSTANCE_ADMIN_ROLE])
