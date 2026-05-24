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

from collections.abc import Sequence
from uuid import UUID

from pydantic import BaseModel, field_validator

from evo.common import APIConnector, RequestMethod

from .data import CentralInstance, Hub, Organization

__all__ = ["DiscoveryAPIClient"]


class _ServiceDiscoveryHub(BaseModel):
    url: str
    code: str
    display_name: str


class _ServiceDiscoveryOrganization(BaseModel):
    id: UUID
    display_name: str


class _ServiceDiscoveryAccess(BaseModel):
    hub_code: str
    org_id: UUID
    services: list[str]


class _ServiceDiscoveryAccountInstanceHub(BaseModel):
    code: str


class _ServiceDiscoveryAccountInstanceCentral(BaseModel):
    id: UUID
    display_name: str
    name: str
    host: str
    organization_name: str


class _ServiceDiscoveryAccountInstance(BaseModel):
    id: UUID
    hub: _ServiceDiscoveryAccountInstanceHub
    central: _ServiceDiscoveryAccountInstanceCentral | None = None


class _ServiceDiscoverAccount(BaseModel):
    instances: list[_ServiceDiscoveryAccountInstance]


class _ServiceDiscoveryResult(BaseModel):
    organizations: list[_ServiceDiscoveryOrganization]
    hubs: list[_ServiceDiscoveryHub]
    service_access: list[_ServiceDiscoveryAccess]
    accounts: list[_ServiceDiscoverAccount] | None = None

    @field_validator("organizations", "hubs")
    def _sort_by_display_name(cls, v: list) -> list:
        # Sort organizations and hubs alphanumerically by display name.
        return sorted(v, key=lambda i: i.display_name)

    def __contains__(self, item: tuple[_ServiceDiscoveryOrganization, _ServiceDiscoveryHub]) -> bool:
        assert isinstance(item, tuple) and len(item) == 2, "Expected a tuple containing organization and hub."
        org, hub = item
        assert isinstance(org, _ServiceDiscoveryOrganization), (
            "Expected organization to be an instance of _ServiceDiscoveryOrganization."
        )
        assert isinstance(hub, _ServiceDiscoveryHub), "Expected hub to be an instance of _ServiceDiscoveryHub."

        for access in self.service_access:
            if access.org_id == org.id and access.hub_code == hub.code:
                return True
        return False

    def get_service_access_services(self, org: _ServiceDiscoveryOrganization, hub: _ServiceDiscoveryHub) -> list[str]:
        assert (org, hub) in self, "Expected a service access entry for the specified organization and hub."
        return next(
            service_access
            for service_access in self.service_access
            if service_access.hub_code == hub.code and service_access.org_id == org.id
        ).services


class _DiscoveryResult(BaseModel):
    discovery: _ServiceDiscoveryResult


class DiscoveryAPIClient:
    """Simple client for interacting with the Discovery API."""

    def __init__(self, connector: APIConnector) -> None:
        """
        :param connector: The API connector to use for making requests.
        """
        self._connector = connector

    @staticmethod
    def _central_instance_for_org(discovery_result: _ServiceDiscoveryResult, org_id: UUID) -> CentralInstance | None:
        if discovery_result.accounts is None:
            return None

        instance = next((i for acc in discovery_result.accounts for i in acc.instances if i.id == org_id), None)
        if instance is None or instance.central is None:
            return None

        return CentralInstance(
            id=instance.central.id,
            display_name=instance.central.display_name,
            name=instance.central.name,
            host=instance.central.host,
            organization_name=instance.central.organization_name,
        )

    async def list_organizations(self, service_codes: Sequence[str] = ("evo",)) -> list[Organization]:
        """Get organizations with access to the specified services.

        :param service_codes: The service codes to use in the query.
        """
        result = await self._connector.call_api(
            RequestMethod.GET,
            "/evo/identity/v2/discovery",
            query_params={"service": service_codes},
            collection_formats={"service": "multi"},
            response_types_map={"200": _DiscoveryResult},
        )
        discovered: _DiscoveryResult = result.discovery

        return [
            Organization(
                id=org.id,
                display_name=org.display_name,
                hubs=tuple(
                    Hub(
                        url=hub.url,
                        code=hub.code,
                        display_name=hub.display_name,
                        services=tuple(discovered.get_service_access_services(org, hub)),
                    )
                    for hub in discovered.hubs
                    if (org, hub) in discovered
                ),
                central=self._central_instance_for_org(discovered, org.id),
            )
            for org in discovered.organizations
        ]
