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

from parameterized import param, parameterized

from evo.common.data import RequestMethod
from evo.common.test_tools import MockResponse, TestWithConnector
from evo.discovery import DiscoveryAPIClient, Hub, Organization

from ..data import load_test_data


def _sample_data_as_response_content(sample_data: dict) -> str:
    return json.dumps({"discovery": sample_data})


def _get_service_access_services(org: dict, hub: dict, service_access: list[dict]) -> list[str]:
    return next(
        service_access
        for service_access in service_access
        if service_access["hub_code"] == hub["code"] and service_access["org_id"] == org["id"]
    )["services"]


def _sample_data_as_expected_orgs(sample_data: dict) -> list[Organization]:
    all_hubs = sorted(sample_data["hubs"], key=lambda h: h["display_name"])
    org_data = sorted(sample_data["organizations"], key=lambda o: o["display_name"])
    service_access = {(sa["org_id"], sa["hub_code"]) for sa in sample_data["service_access"]}
    return [
        Organization(
            id=UUID(org["id"]),
            display_name=org["display_name"],
            hubs=tuple(
                Hub(
                    url=hub["url"],
                    code=hub["code"],
                    display_name=hub["display_name"],
                    services=tuple(_get_service_access_services(org, hub, sample_data["service_access"])),
                )
                for hub in all_hubs
                if (org["id"], hub["code"]) in service_access
            ),
        )
        for org in org_data
    ]


class TestDiscoveryAPIClient(TestWithConnector):
    def setUp(self) -> None:
        super().setUp()
        self.discovery_client = DiscoveryAPIClient(self.connector)
        self.transport.request.return_value = MockResponse(status_code=500)

    async def test_list_organizations_default_service_code(self) -> None:
        """Test a successful get organizations request with the default service code."""
        test_data = load_test_data("successful_service_discovery.json")
        test_data_text = json.dumps(test_data)
        with self.transport.set_http_response(
            status_code=200, content=test_data_text, headers={"Content-Type": "application/json"}
        ):
            await self.discovery_client.list_organizations()
        expected_path = "/evo/identity/v2/discovery?service=evo"
        self.assert_request_made(method=RequestMethod.GET, path=expected_path)

    async def test_list_organizations_with_custom_service_codes(self) -> None:
        """Test a successful get organizations request with custom service codes."""
        test_data = load_test_data("successful_service_discovery.json")
        test_data_text = json.dumps(test_data)
        with self.transport.set_http_response(
            status_code=200, content=test_data_text, headers={"Content-Type": "application/json"}
        ):
            await self.discovery_client.list_organizations(["service0", "service1", "service2"])
        expected_path = "/evo/identity/v2/discovery?service=service0&service=service1&service=service2"
        self.assert_request_made(method=RequestMethod.GET, path=expected_path)

    @parameterized.expand([param(**scenario) for scenario in load_test_data("service_discovery_data.json")])
    async def test_list_organizations(self, scenario: str, sample_data: dict) -> None:
        expected_orgs = _sample_data_as_expected_orgs(sample_data)
        with self.transport.set_http_response(
            status_code=200,
            content=_sample_data_as_response_content(sample_data),
            headers={"Content-Type": "application/json"},
        ):
            actual_orgs = await self.discovery_client.list_organizations()
        self.assertListEqual(expected_orgs, actual_orgs)

        # Test that the organizations are sorted alphanumerically.
        self.assertListEqual(sorted(actual_orgs, key=lambda o: o.display_name), actual_orgs)

        # Test that the hubs are sorted alphanumerically.
        for org in actual_orgs:
            self.assertListEqual(sorted(org.hubs, key=lambda h: h.display_name), list(org.hubs))

    def test_organization_is_hashable(self) -> None:
        """Test that the Organization dataclass is hashable."""
        org = Organization(
            id=UUID("12345678-1234-5678-1234-567812345678"),
            display_name="Hashable org name",
            hubs=(
                Hub(
                    url="https://example.com",
                    code="test_hub",
                    display_name="Test Hub",
                    services=("service1", "service2"),
                ),
            ),
        )
        try:
            hash_value = hash(org)
            self.assertIsInstance(hash_value, int)
        except TypeError:
            self.fail("Organization dataclass is not hashable")
