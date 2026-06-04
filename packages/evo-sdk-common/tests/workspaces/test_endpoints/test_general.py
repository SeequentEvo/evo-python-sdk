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

from unittest import mock

from evo.common import HealthCheckType, StaticContext
from evo.common.exceptions import ContextError
from evo.common.test_tools import TestWithConnector
from evo.common.utils import get_header_metadata
from evo.workspaces import (
    WorkspaceAPIClient,
)

from ..consts import (
    ORG_UUID,
)


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
