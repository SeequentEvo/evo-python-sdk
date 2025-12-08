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

from __future__ import annotations

import json

from parameterized import parameterized

from data import load_test_data
from evo.common import Environment, StaticContext
from evo.common.data import RequestMethod
from evo.common.test_tools import BASE_URL, ORG, WORKSPACE_ID, TestWithConnector
from evo.common.utils.version import get_header_metadata
from evo.objects.client.api_client import ObjectAPIClient
from evo.objects.typed._utils import create_geoscience_object


class TestCreateGeoscienceObject(TestWithConnector):
    def setUp(self) -> None:
        TestWithConnector.setUp(self)
        self.environment = Environment(hub_url=BASE_URL, org_id=ORG.id, workspace_id=WORKSPACE_ID)
        self.context = StaticContext.from_environment(
            environment=self.environment,
            connector=self.connector,
        )
        self.setup_universal_headers(get_header_metadata(ObjectAPIClient.__module__))

    @property
    def instance_base_path(self) -> str:
        return f"geoscience-object/orgs/{self.environment.org_id}"

    @property
    def base_path(self) -> str:
        return f"{self.instance_base_path}/workspaces/{self.environment.workspace_id}"

    @parameterized.expand(
        [
            (None, "Sample%20pointset.json"),
            ("path/to/parent", "path/to/parent/Sample%20pointset.json"),
            ("path/to/parent/", "path/to/parent/Sample%20pointset.json"),
        ]
    )
    async def test_create_geoscience_object(self, parent: str | None, expected_object_path: str):
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
            await create_geoscience_object(self.context, new_pointset, parent=parent)

        self.assert_request_made(
            method=RequestMethod.POST,
            path=f"{self.base_path}/objects/path/{expected_object_path}",
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            body=new_pointset_without_uuid,
        )
