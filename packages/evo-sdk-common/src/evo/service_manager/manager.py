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

import asyncio
from typing import Any, TypeVar
from uuid import UUID

from evo import logging
from evo.common import APIConnector, BaseAPIClient, Environment
from evo.common.context import IContext
from evo.common.exceptions import SelectionError
from evo.common.interfaces import IAuthorizer, ITransport
from evo.discovery import DiscoveryAPIClient, Hub, Organization
from evo.workspaces import Workspace, WorkspaceAPIClient

__all__ = ["ServiceManager"]

logger = logging.getLogger("common.utils.manager")


# Generic type variable for the client factory method.
T_client = TypeVar("T_client", bound=BaseAPIClient)


class _State:
    """A simple state manager for managing the current selection of organizations, hubs, and workspaces."""

    def __init__(
        self,
        organizations: list[Organization],
        *,
        workspaces: list[Workspace] | None = None,
        selected_org_id: UUID | None = None,
        selected_hub_code: str | None = None,
        selected_workspace_id: UUID | None = None,
    ) -> None:
        """
        :param organizations: A list of organizations.
        :param workspaces: A list of workspaces.
        :param selected_org_id: The ID of the selected organization.
        :param selected_hub_code: The code of the selected hub.
        :param selected_workspace_id: The ID of the selected workspace.

        :raises SelectionError: If the selected organization, hub, or workspace is invalid.
        """
        self._organizations_by_id = {org.id: org for org in organizations}
        self._select_org(selected_org_id)
        self._select_hub(selected_hub_code)

        if workspaces is not None and not isinstance(self.organization, Organization):
            raise SelectionError("Cannot list workspaces without an organization.")

        if workspaces is not None and not isinstance(self.hub, Hub):
            raise SelectionError("Cannot list workspaces without a hub.")

        self._workspaces_by_id = {ws.id: ws for ws in workspaces} if workspaces is not None else {}
        self._select_workspace(selected_workspace_id)

    def _select_org(self, org_id: UUID | None) -> None:
        if org_id is not None and org_id not in self._organizations_by_id:
            raise SelectionError(f"Invalid organization ID: {org_id!r}")
        else:
            self._selected_org_id = org_id

        org = self.organization
        self._hubs_by_code = {hub.code: hub for hub in org.hubs} if isinstance(org, Organization) else {}

    def _select_hub(self, hub_code: str | None) -> None:
        if hub_code is not None and not isinstance(self.organization, Organization):
            raise SelectionError("Cannot select a hub without an organization.")
        elif hub_code is not None and hub_code not in self._hubs_by_code:
            raise SelectionError(f"Invalid hub code: {hub_code!r}")
        else:
            self._selected_hub_code = hub_code

    def _select_workspace(self, workspace_id: UUID | None) -> None:
        if workspace_id is not None and not isinstance(self.hub, Hub):
            raise SelectionError("Cannot select a workspace without a hub.")
        elif workspace_id is not None and workspace_id not in self._workspaces_by_id:
            raise SelectionError(f"Invalid workspace ID: {workspace_id!r}")
        else:
            self._selected_workspace_id = workspace_id

    @property
    def organizations(self) -> list[Organization]:
        """Get a list of organizations sorted by display name."""
        return sorted(self._organizations_by_id.values(), key=lambda org: org.display_name)

    @property
    def organization(self) -> Organization | None:
        """Get the currently selected organization."""
        return self._organizations_by_id[self._selected_org_id] if self._selected_org_id is not None else None

    @property
    def hubs(self) -> list[Hub]:
        """Get a list of hubs sorted by display name."""
        return sorted(self._hubs_by_code.values(), key=lambda hub: hub.display_name)

    @property
    def hub(self) -> Hub | None:
        """Get the currently selected hub."""
        return self._hubs_by_code[self._selected_hub_code] if self._selected_hub_code is not None else None

    @property
    def workspaces(self) -> list[Workspace]:
        """Get a list of workspaces sorted by display name."""
        return sorted(self._workspaces_by_id.values(), key=lambda ws: ws.display_name)

    @property
    def workspace(self) -> Workspace | None:
        """Get the currently selected workspace."""
        return self._workspaces_by_id[self._selected_workspace_id] if self._selected_workspace_id is not None else None

    def with_organization(self, org_id: UUID | None) -> _State:
        """Create a new state with the specified organization selected.

        :param org_id: The ID of the organization to select, or `None` to clear the current selection.

        :returns: The new state, or the current state if the organization was not changed.

        :raises SelectionError: If the organization is invalid.
        """
        logger.debug("Selecting organization")
        if org_id == getattr(self.organization, "id", None):
            logger.debug("Organization was not changed.")
            return self
        else:
            logger.debug("Organization was changed.")
            return _State(self.organizations, selected_org_id=org_id)

    def with_hub(self, hub_code: str | None) -> _State:
        """Create a new state with the specified hub selected.

        :param hub_code: The code of the hub to select, or `None` to clear the current selection.

        :returns: The new state, or the current state if the hub was not changed.

        :raises SelectionError: If the hub is invalid.
        """
        logger.debug("Selecting hub")
        if hub_code == getattr(self.hub, "code", None):
            logger.debug("Hub was not changed.")
            return self
        else:
            logger.debug("Hub was changed.")
            return _State(
                self.organizations,
                selected_org_id=self.organization.id,
                selected_hub_code=hub_code,
            )

    def with_workspace(self, workspace_id: UUID | None) -> _State:
        """Create a new state with the specified workspace selected.

        :param workspace_id: The ID of the workspace to select, or `None` to clear the current selection.

        :returns: The new state, or the current state if the workspace was not changed.

        :raises SelectionError: If the workspace is invalid.
        """
        logger.debug("Selecting workspace")
        if workspace_id == getattr(self.workspace, "id", None):
            logger.debug("Workspace was not changed.")
            return self
        else:
            logger.debug("Workspace was changed.")
            return _State(
                self.organizations,
                workspaces=self.workspaces,
                selected_org_id=self.organization.id,
                selected_hub_code=self.hub.code,
                selected_workspace_id=workspace_id,
            )

    def update_organizations(self, organizations: list[Organization]) -> _State:
        """Update the organizations in the state.

        This method will re-select the currently selected organization, hub, and workspace if they are still valid.

        :param organizations: The new list of organizations.

        :returns: The new state with the updated organizations.
        """
        logger.debug("Updating organizations")
        new_state = _State(organizations)
        if self._selected_org_id in new_state._organizations_by_id:
            logger.debug("Re-selecting organization")
            new_state._select_org(self._selected_org_id)

            if self._selected_hub_code in new_state._hubs_by_code:
                logger.debug("Re-selecting hub")
                new_state._select_hub(self._selected_hub_code)

                # Copy the workspaces from the old state to the new state.
                new_state._workspaces_by_id = self._workspaces_by_id.copy()

                if self._selected_workspace_id in new_state._workspaces_by_id:
                    logger.debug("Re-selecting workspace")
                    new_state._select_workspace(self._selected_workspace_id)

        logger.debug("Organizations updated.")
        return new_state

    def update_workspaces(self, workspaces: list[Workspace]) -> _State:
        """Update the workspaces in the state.

        This method will re-select the currently selected workspace if it is still valid.

        :param workspaces: The new list of workspaces.

        :returns: The new state with the updated workspaces.
        """
        logger.debug("Updating workspaces")
        new_state = _State(
            self.organizations,
            workspaces=workspaces,
            selected_org_id=self._selected_org_id,
            selected_hub_code=self._selected_hub_code,
        )

        if self._selected_workspace_id in new_state._workspaces_by_id:
            logger.debug("Re-selecting workspace")
            new_state._select_workspace(self._selected_workspace_id)

        logger.debug("Workspaces updated.")
        return new_state

    def without_workspaces(self) -> _State:
        """Create a new state without any workspaces selected.

        :returns: The new state without any workspaces selected.
        """
        logger.debug("Removing workspaces")
        return _State(
            self.organizations,
            selected_org_id=self._selected_org_id,
            selected_hub_code=self._selected_hub_code,
        )


class ServiceManager(IContext):
    """A simple service manager for managing the current selection of organizations, hubs, and workspaces."""

    def __init__(self, transport: ITransport, authorizer: IAuthorizer, discovery_url: str) -> None:
        """
        :param transport: The transport to use for API requests.
        :param authorizer: The authorizer to use for API requests.
        :param discovery_url: The URL of the discovery service.
        """
        self._transport = transport
        self._authorizer = authorizer
        self._discovery_url = discovery_url

        # The state mutex is only required for longer-running operations (e.g., refreshing organizations) to prevent
        # concurrent updates to the state. Read-only operations (e.g., listing organizations) are considered safe
        # because each state instance is not mutated.
        self.__state_mutex = asyncio.Lock()
        self.__state = _State([])

    def _get_connector(self, base_url: str) -> APIConnector:
        return APIConnector(base_url, self._transport, self._authorizer)

    async def refresh_organizations(self) -> None:
        """Refresh the list of organizations.

        This method will update the current state with the latest list of organizations from the discovery service.

        If an error occurs while refreshing the organizations, the state will be reset.
        """
        logger.debug("Refreshing organizations")
        async with self.__state_mutex, self._get_connector(self._discovery_url) as connector:
            client = DiscoveryAPIClient(connector)
            try:
                self.__state = self.__state.update_organizations(await client.list_organizations())
            except Exception:
                logger.exception("Failed to refresh organizations.", exc_info=True)
                self.__state = _State([])
                raise

    def list_organizations(self) -> list[Organization]:
        """Get a list of organizations sorted by display name.

        If organizations have not been refreshed, this method will return an empty list.

        :returns: The list of organizations.
        """
        return self.__state.organizations

    def get_current_organization(self) -> Organization | None:
        """Get the currently selected organization.

        :returns: The currently selected organization, or `None` if no organization is selected.
        """
        return self.__state.organization

    def set_current_organization(self, org_id: UUID | None) -> None:
        """Set the currently selected organization.

        :param org_id: The ID of the organization to select, or `None` to clear the current selection.

        :raises SelectionError: If the organization is invalid.
        """
        self.__state = self.__state.with_organization(org_id)

    def list_hubs(self) -> list[Hub]:
        """Get a list of hubs sorted by display name.

        If no organization is currently selected, this method will return an empty list.

        :returns: The list of hubs.
        """
        return self.__state.hubs

    def get_current_hub(self) -> Hub | None:
        """Get the currently selected hub.

        :returns: The currently selected hub, or `None` if no hub is selected.
        """
        return self.__state.hub

    def set_current_hub(self, hub_code: str) -> None:
        """Set the currently selected hub.

        :param hub_code: The code of the hub to select.

        :raises SelectionError: If the hub is invalid.
        """
        self.__state = self.__state.with_hub(hub_code)

    async def refresh_workspaces(self) -> None:
        """Refresh the list of workspaces.

        This method will update the current state with the latest list of workspaces from the workspace service.

        If an error occurs while refreshing the workspaces, the list of workspaces will be reset.
        """
        logger.debug("Refreshing workspaces")
        async with self.__state_mutex:
            if not isinstance(org := self.get_current_organization(), Organization):
                return  # Cannot refresh workspaces without an organization.

            if not isinstance(hub := self.get_current_hub(), Hub):
                return  # Cannot refresh workspaces without a hub.

            async with self._get_connector(hub.url) as connector:
                client = WorkspaceAPIClient(connector, org.id)
                try:
                    self.__state = self.__state.update_workspaces(await client.list_all_workspaces())
                except Exception:
                    logger.exception("Failed to refresh workspaces.", exc_info=True)
                    self.__state = self.__state.without_workspaces()
                    raise

    def list_workspaces(self) -> list[Workspace]:
        """Get a list of workspaces sorted by display name.

        If workspaces have not been refreshed, this method will return an empty list.

        :returns: The list of workspaces.
        """
        return self.__state.workspaces

    def get_current_workspace(self) -> Workspace | None:
        """Get the currently selected workspace.

        :returns: The currently selected workspace, or `None` if no workspace is selected.
        """
        return self.__state.workspace

    def set_current_workspace(self, workspace_id: UUID) -> None:
        """Set the currently selected workspace.

        :param workspace_id: The ID of the workspace to select.

        :raises SelectionError: If the workspace is invalid.
        """
        self.__state = self.__state.with_workspace(workspace_id)

    def get_connector(self) -> APIConnector:
        """Get an API connector for the currently selected hub.

        :returns: The API connector.

        :raises SelectionError: If no organization or hub is currently selected.
        """
        if not isinstance(self.get_current_organization(), Organization):
            raise SelectionError("No organization is currently selected.")

        if not isinstance(hub := self.get_current_hub(), Hub):
            raise SelectionError("No hub is currently selected.")

        return self._get_connector(hub.url)

    def get_environment(self) -> Environment:
        """Get an environment with the currently selected organization, hub, and workspace.

        :returns: The environment.

        :raises SelectionError: If no organization, hub, or workspace is currently selected.
        """
        if not isinstance(org := self.get_current_organization(), Organization):
            raise SelectionError("No organization is currently selected.")
        if not isinstance(hub := self.get_current_hub(), Hub):
            raise SelectionError("No hub is currently selected.")
        if not isinstance(ws := self.get_current_workspace(), Workspace):
            raise SelectionError("No workspace is currently selected.")
        return Environment(hub_url=hub.url, org_id=org.id, workspace_id=ws.id)

    def get_org_id(self) -> UUID:
        """Gets the organization ID associated with this context.

        :return: The organization ID.
        :raises SelectionError: If no organization is currently selected.
        """
        if not isinstance(org := self.get_current_organization(), Organization):
            raise SelectionError("No organization is currently selected.")
        return org.id

    def get_cache(self) -> None:
        """Gets the cache of this context.

        Always None, as ServiceManager does not have a cache.
        """
        return None

    def create_client(self, client_class: type[T_client], *args: Any, **kwargs: Any) -> T_client:
        """Create a client for the currently selected workspace.

        :param client_class: The class of the client to create.

        :returns: The new client.

        :raises SelectionError: If no organization, hub, or workspace is currently selected.
        """
        return client_class(self.get_environment(), self.get_connector(), *args, **kwargs)
