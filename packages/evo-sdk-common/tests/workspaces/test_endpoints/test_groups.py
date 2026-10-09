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
from evo.common.test_tools import TestWithConnector
from evo.common.utils import get_header_metadata
from evo.workspaces import ImsGroup, ImsGroupDetail, ImsUser, WorkspaceAPIClient

from ...data import load_test_data
from ..consts import BASE_PATH, ORG_UUID

GROUP_ID = UUID(int=0xA)
PREVIEW_HEADER = {"Accept": "application/json", "API-Preview": "opt-in"}
IMS_USER = ImsUser(email="test.user@example.com", full_name="Test User")
IMS_GROUP = ImsGroup(group_id=str(GROUP_ID), name="Test Group", is_federated_group=False)
IMS_GROUP_DETAIL = ImsGroupDetail(
    group_id=GROUP_ID,
    name="Test Group",
    description="Test group description",
    is_federated_group=False,
)


class TestWorkspaceClientGroupEndpoints(TestWithConnector):
    def setUp(self) -> None:
        super().setUp()
        self.workspace_client = WorkspaceAPIClient(connector=self.connector, org_id=ORG_UUID)
        self.setup_universal_headers(get_header_metadata(WorkspaceAPIClient.__module__))

    async def test_get_ims_users(self) -> None:
        content = load_test_data("ims_users.json")
        with self.transport.set_http_response(200, json.dumps(content), headers={"Content-Type": "application/json"}):
            response = await self.workspace_client.get_ims_users("test.user@example.com")

        self.assert_request_made(
            method=RequestMethod.GET,
            path=f"{BASE_PATH}/ims/users?email=test.user%40example.com",
            headers={"Accept": "application/json"},
        )
        self.assertEqual(response, [IMS_USER])

    async def test_get_group_description(self) -> None:
        content = load_test_data("ims_group_detail.json")
        with self.transport.set_http_response(200, json.dumps(content), headers={"Content-Type": "application/json"}):
            response = await self.workspace_client.get_group_description(GROUP_ID, api_preview="opt-in")

        self.assert_request_made(
            method=RequestMethod.GET,
            path=f"{BASE_PATH}/ims/groups/{GROUP_ID}",
            headers=PREVIEW_HEADER,
        )
        self.assertEqual(response, IMS_GROUP_DETAIL)

    async def test_list_ims_group_members(self) -> None:
        content = load_test_data("ims_users.json")
        with self.transport.set_http_response(200, json.dumps(content), headers={"Content-Type": "application/json"}):
            response = await self.workspace_client.list_ims_group_members(GROUP_ID, api_preview="opt-in")

        self.assert_request_made(
            method=RequestMethod.GET,
            path=f"{BASE_PATH}/ims/groups/{GROUP_ID}/members",
            headers=PREVIEW_HEADER,
        )
        self.assertEqual(response, [IMS_USER])

    async def test_list_ims_groups(self) -> None:
        content = load_test_data("ims_groups.json")
        with self.transport.set_http_response(200, json.dumps(content), headers={"Content-Type": "application/json"}):
            response = await self.workspace_client.list_ims_groups(api_preview="opt-in")

        self.assert_request_made(
            method=RequestMethod.GET,
            path=f"{BASE_PATH}/ims/groups",
            headers=PREVIEW_HEADER,
        )
        self.assertEqual(response, [IMS_GROUP])
