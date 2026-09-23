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
from uuid import UUID

from evo.common import RequestMethod
from evo.common.test_tools import TestWithConnector, utc_datetime
from evo.common.utils import get_header_metadata
from evo.workspaces import (
    InstanceGroup,
    InstanceGroupInvitation,
    InstanceGroupMember,
    InstanceRole,
    UpdatedInstanceGroupMembers,
    WorkspaceAPIClient,
)

from ...data import load_test_data
from ..consts import BASE_PATH, ORG_UUID

IMS_GROUP_ID = str(UUID(int=0xA))
GROUP_ID = UUID(int=0xB)
MEMBER = InstanceGroupMember(user_id=UUID(int=2), email="test.user@example.com", full_name="Test User")
ROLE = InstanceRole(role_id=UUID(int=1), name="Evo User", description="evo user")
INVITATION = InstanceGroupInvitation(
    invitation_id=UUID(int=3),
    email="invited.user@example.com",
    invited_at=utc_datetime(2026, 1, 1, 12, 0, 0),
    expiration_date=utc_datetime(2026, 1, 15, 12, 0, 0),
    invited_by="admin.user@example.com",
    status="Pending",
)
PREVIEW_HEADER = {"Accept": "application/json", "API-Preview": "opt-in"}


class TestWorkspaceClientInstanceGroupEndpoints(TestWithConnector):
    def setUp(self) -> None:
        super().setUp()
        self.workspace_client = WorkspaceAPIClient(connector=self.connector, org_id=ORG_UUID)
        self.setup_universal_headers(get_header_metadata(WorkspaceAPIClient.__module__))

    async def test_list_instance_groups(self) -> None:
        content = load_test_data("instance_groups.json")
        with self.transport.set_http_response(200, json.dumps(content), headers={"Content-Type": "application/json"}):
            response = await self.workspace_client.list_instance_groups(api_preview="opt-in")

        self.assert_request_made(
            method=RequestMethod.GET,
            path=f"{BASE_PATH}/members/groups",
            headers=PREVIEW_HEADER,
        )
        self.assertEqual(
            response,
            [
                InstanceGroup(
                    group_id=GROUP_ID,
                    name="Instance Group",
                    description="Instance group description",
                    ims_groups=[IMS_GROUP_ID],
                    members=[MEMBER],
                    roles=[ROLE],
                )
            ],
        )

    async def test_update_instance_group_members(self) -> None:
        content = load_test_data("update_instance_group_members.json")
        with self.transport.set_http_response(200, json.dumps(content), headers={"Content-Type": "application/json"}):
            response = await self.workspace_client.update_instance_group_members(
                GROUP_ID, ims_groups=[IMS_GROUP_ID], api_preview="opt-in"
            )

        self.assert_request_made(
            method=RequestMethod.PATCH,
            path=f"{BASE_PATH}/members/groups/{GROUP_ID}",
            headers={"Content-Type": "application/json", **PREVIEW_HEADER},
            body={"ims_groups": [IMS_GROUP_ID]},
        )
        self.assertEqual(
            response,
            UpdatedInstanceGroupMembers(
                group_id=GROUP_ID,
                name="Instance Group",
                description="Instance group description",
                ims_groups=[IMS_GROUP_ID],
                members=[MEMBER],
                invitations=[INVITATION],
            ),
        )
