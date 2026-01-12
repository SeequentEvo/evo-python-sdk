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

from datetime import timezone
from typing import Literal, TypeAlias
from uuid import UUID

from pydantic import ValidationError
from pydantic.type_adapter import TypeAdapter

from evo.common import APIConnector, HealthCheckType, IContext, Page, ServiceHealth, ServiceUser
from evo.common.utils import get_service_health, parse_order_by

from .data import (
    BasicWorkspace,
    BoundingBox,
    Coordinate,
    InstanceUser,
    InstanceUserInvitation,
    InstanceUserRole,
    InstanceUserRoleWithPermissions,
    InstanceUserWithEmail,
    OrderByOperatorEnum,
    User,
    UserRole,
    Workspace,
    WorkspaceOrderByEnum,
    WorkspaceRole,
)
from .endpoints.api import AdminApi, GeneralApi, InstanceUsersApi, ThumbnailsApi, WorkspacesApi
from .endpoints.models import (
    AddInstanceUsersRequest,
    BaseInstanceUserResponse,
    BasicWorkspaceResponse,
    CreateWorkspaceRequest,
    GeometryTypeEnum,
    Label,
    ListInstanceRolesResponse,
    ListInstanceUserInvitationsResponse,
    RoleEnum,
    UpdateInstanceUserRolesRequest,
    UpdateWorkspaceRequest,
    UserRoleMapping,
    WorkspaceRoleOptionalResponse,
    WorkspaceRoleRequiredResponse,
)
from .endpoints.models import BoundingBox as PydanticBoundingBox
from .endpoints.models import Coordinate as PydanticCoordinate
from .endpoints.models import User as PydanticUser
from .endpoints.models import UserRole as PydanticUserRole

WorkspaceOrderByLiteral: TypeAlias = Literal["name", "created_at", "updated_at", "user_role"]
OrderByOperatorLiteral: TypeAlias = Literal["asc", "desc"]


class WorkspaceAPIClient:
    """
    The Workspace Service API client.
    """

    def __init__(self, connector: APIConnector, org_id: UUID) -> None:
        self._connector = connector
        self._org_id = org_id
        self._workspaces_api = WorkspacesApi(connector)
        self._admin_api = AdminApi(connector)
        self._general_api = GeneralApi(connector)
        self._thumbnails_api = ThumbnailsApi(connector)
        self._instance_users_api = InstanceUsersApi(connector)

    @classmethod
    def from_context(cls, context: IContext) -> WorkspaceAPIClient:
        """Create a WorkspaceAPIClient from the given context.

        The context must have a hub_url and org_id set.

        :param context: The context to create the client from.
        :return: A WorkspaceAPIClient instance.
        """
        return cls(connector=context.get_connector(), org_id=context.get_org_id())

    async def get_service_health(self, check_type: HealthCheckType = HealthCheckType.FULL) -> ServiceHealth:
        """Get the health of the workspace service.
        :param check_type: The type of health check to perform.
        :return: A ServiceHealth object.
        :raises EvoAPIException: If the API returns an unexpected status code.
        :raises ClientValueError: If the response is not a valid service health check response.
        """
        return await get_service_health(self._connector, "workspace", check_type=check_type)

    @staticmethod
    def __parse_bounding_box(model: PydanticBoundingBox) -> BoundingBox:
        def convert_coordinate(pydantic_coordinate: PydanticCoordinate) -> Coordinate:
            return Coordinate(latitude=pydantic_coordinate.root[1], longitude=pydantic_coordinate.root[0])

        coordinates = [[convert_coordinate(coord) for coord in coord_list] for coord_list in model.coordinates]

        return BoundingBox(coordinates=coordinates, type=str(model.type.value))

    def __parse_workspace_model(
        self, model: WorkspaceRoleOptionalResponse | WorkspaceRoleRequiredResponse
    ) -> Workspace:
        bounding_box = None
        if model.bounding_box:
            bounding_box = self.__parse_bounding_box(model.bounding_box)

        return Workspace(
            id=model.id,
            org_id=self._org_id,
            hub_url=self._connector.base_url,
            display_name=model.name,
            description=model.description,
            user_role=WorkspaceRole[str(model.current_user_role.value)] if model.current_user_role else None,
            created_at=model.created_at,
            created_by=ServiceUser.from_model(model.created_by),
            updated_at=model.updated_at,
            updated_by=ServiceUser.from_model(model.updated_by),
            bounding_box=bounding_box,
            default_coordinate_system=model.default_coordinate_system,
            labels=model.labels,
        )

    @staticmethod
    def __parse_workspace_basic_model(model: BasicWorkspaceResponse) -> BasicWorkspace:
        return BasicWorkspace(
            id=model.id,
            display_name=model.name,
        )

    @staticmethod
    def __parse_user_role_model(model: PydanticUserRole) -> UserRole:
        return UserRole(user_id=model.user_id, role=WorkspaceRole[str(model.role.value)])

    @staticmethod
    def __parse_user_model(model: PydanticUser) -> User:
        return User(
            user_id=model.user_id,
            role=WorkspaceRole[str(model.role.value)],
            email=model.email,
            full_name=model.full_name,
        )

    @staticmethod
    def _parse_instance_user_model(model: BaseInstanceUserResponse) -> InstanceUser:
        return InstanceUser(
            user_id=model.id,
            roles=[
                InstanceUserRole(role_id=role.id, name=role.name, description=role.description) for role in model.roles
            ],
        )

    @staticmethod
    def _parse_instance_user_with_email_model(model: BaseInstanceUserResponse) -> InstanceUserWithEmail:
        return InstanceUserWithEmail(
            email=model.email,
            full_name=model.full_name,
            user_id=model.id,
            roles=[
                InstanceUserRole(role_id=role.id, name=role.name, description=role.description) for role in model.roles
            ],
        )

    def _parse_instance_user_invitation_model(
        self, model: ListInstanceUserInvitationsResponse
    ) -> InstanceUserInvitation:
        return InstanceUserInvitation(
            email=model.email,
            invitation_id=model.id,
            roles=[
                InstanceUserRole(role_id=role.id, name=role.name, description=role.description) for role in model.roles
            ],
            invited_at=model.created_date.replace(tzinfo=timezone.utc),
            expiration_date=model.expiration_date.replace(tzinfo=timezone.utc),
            invited_by=model.invited_by_email,
            status=model.status,
        )

    def _parse_instance_user_role_model(self, model: ListInstanceRolesResponse) -> InstanceUserRoleWithPermissions:
        return InstanceUserRoleWithPermissions(
            role_id=model.id, name=model.name, description=model.description, permissions=model.permissions
        )

    async def list_user_roles(
        self,
        workspace_id: UUID,
        filter_user_id: UUID | None = None,
    ) -> list[User]:
        response = await self._workspaces_api.list_user_roles(
            org_id=str(self._org_id),
            workspace_id=str(workspace_id),
            filter_user_id=str(filter_user_id) if filter_user_id else None,
        )

        return [self.__parse_user_model(item) for item in response.results]

    async def get_current_user_role(
        self,
        workspace_id: UUID,
    ) -> UserRole:
        response = await self._workspaces_api.get_current_user_role(
            org_id=str(self._org_id),
            workspace_id=str(workspace_id),
        )

        return self.__parse_user_role_model(response)

    async def assign_user_role(
        self,
        workspace_id: UUID,
        user_id: UUID,
        role: WorkspaceRole,
    ) -> UserRole:
        assign_role_request = PydanticUserRole(user_id=user_id, role=RoleEnum[str(role.name)])
        response = await self._workspaces_api.assign_user_role(
            org_id=str(self._org_id),
            workspace_id=str(workspace_id),
            assign_role_request=assign_role_request,
        )

        return self.__parse_user_role_model(response)

    async def delete_user_role(
        self,
        workspace_id: UUID,
        user_id: UUID,
    ) -> None:
        await self._workspaces_api.delete_user_role(
            org_id=str(self._org_id),
            workspace_id=str(workspace_id),
            user_id=str(user_id),
        )

    async def list_workspaces(
        self,
        limit: int | None = None,
        offset: int | None = None,
        order_by: dict[WorkspaceOrderByEnum, OrderByOperatorEnum]
        | dict[WorkspaceOrderByLiteral, OrderByOperatorLiteral]
        | None = None,
        filter_created_by: UUID | None = None,
        created_at: str | None = None,
        updated_at: str | None = None,
        name: str | None = None,
        deleted: bool | None = None,
        filter_user_id: UUID | None = None,
    ) -> Page[Workspace]:
        parsed_order_by = parse_order_by(order_by)  # type: ignore
        if not offset:
            offset = 0
        response = await self._workspaces_api.list_workspaces(
            org_id=str(self._org_id),
            limit=limit,
            offset=offset,
            order_by=parsed_order_by,
            filter_created_by=str(filter_created_by) if filter_created_by else None,
            created_at=created_at,
            updated_at=updated_at,
            name=name,
            deleted=deleted,
            filter_user_id=str(filter_user_id) if filter_user_id else None,
        )

        return Page(
            offset=offset,
            limit=limit,
            total=response.links.total,
            items=[self.__parse_workspace_model(item) for item in response.results],
        )

    async def list_all_workspaces(
        self,
        limit: int | None = None,
        offset: int | None = None,
        order_by: dict[WorkspaceOrderByEnum, OrderByOperatorEnum] | None = None,
        filter_created_by: UUID | None = None,
        created_at: str | None = None,
        updated_at: str | None = None,
        name: str | None = None,
        deleted: bool | None = None,
        filter_user_id: UUID | None = None,
    ) -> list[Workspace]:
        workspaces: list[Workspace] = []
        if offset is None:
            offset = 0
        if limit is None:
            limit = 50

        while True:
            workspace_page = await self.list_workspaces(
                limit=limit,
                offset=offset,
                order_by=order_by,
                filter_created_by=filter_created_by,
                created_at=created_at,
                updated_at=updated_at,
                name=name,
                deleted=deleted,
                filter_user_id=filter_user_id,
            )
            workspaces += workspace_page.items()
            offset += limit
            if offset >= workspace_page.total:
                break

        return sorted(workspaces, key=lambda x: x.display_name)

    async def list_workspaces_summary(
        self,
        limit: int | None = None,
        offset: int | None = None,
        order_by: dict[WorkspaceOrderByEnum, OrderByOperatorEnum]
        | dict[WorkspaceOrderByLiteral, OrderByOperatorLiteral]
        | None = None,
        filter_created_by: UUID | None = None,
        created_at: str | None = None,
        updated_at: str | None = None,
        name: str | None = None,
        deleted: bool | None = None,
        filter_user_id: UUID | None = None,
    ) -> Page[BasicWorkspace]:
        """
        Get an optionally paginated list of basic workspaces with optional filtering and sorting.

        This method provides faster performance than list_workspaces() or list_all_workspaces()
        by returning BasicWorkspace objects with minimal data instead of full Workspace objects.
        """
        parsed_order_by = parse_order_by(order_by)

        response = await self._workspaces_api.list_workspaces_summary(
            org_id=str(self._org_id),
            limit=limit,
            offset=offset,
            order_by=parsed_order_by,
            filter_created_by=str(filter_created_by) if filter_created_by else None,
            created_at=created_at,
            updated_at=updated_at,
            name=name,
            deleted=deleted,
            filter_user_id=str(filter_user_id) if filter_user_id else None,
        )

        return Page(
            offset=offset or 0,
            limit=limit or response.links.total,
            total=response.links.total,
            items=[self.__parse_workspace_basic_model(item) for item in response.results],
        )

    async def get_workspace(self, workspace_id: UUID, deleted: bool = False) -> Workspace:
        response = await self._workspaces_api.get_workspace(
            org_id=str(self._org_id), workspace_id=str(workspace_id), deleted=deleted
        )
        return self.__parse_workspace_model(response)

    async def delete_workspace(
        self,
        workspace_id: UUID,
    ) -> None:
        """Delete workspaces by workspace id.

        :param workspace_id: The workspace id to delete.

        :returns: An empty response.
        """
        await self._workspaces_api.delete_workspace(
            org_id=str(self._org_id),
            workspace_id=str(workspace_id),
        )

    async def create_workspace(
        self,
        name: str,
        bounding_box_coordinates: list[tuple[float, float]] | None = None,
        default_coordinate_system: str | None = None,
        description: str | None = None,
        labels: list[str] | None = None,
    ) -> Workspace:
        """Create a new workspace.

        :param name: The name of the workspace.
        :param bounding_box_coordinates: The coordinates list corresponding to the bounding box of the workspace.
            The coordinates list should be in the format of [[longitude, latitude], [longitude, latitude], ...]
        :param default_coordinate_system: The default coordinate system of the workspace.
        :param description: The description of the workspace.
        :param labels: The labels of the workspace.

        :returns: The created workspace response.
        """
        # apply validation on the values
        if description is None:
            description = ""
        if default_coordinate_system is None:
            default_coordinate_system = ""
        if labels is not None and len(labels) > 20:
            raise ValueError("The labels field must contain 20 or fewer items.")

        if bounding_box_coordinates is not None:
            try:
                bounding_box = PydanticBoundingBox(
                    type=GeometryTypeEnum.Polygon,
                    coordinates=[bounding_box_coordinates],
                )
            except ValidationError as e:
                raise ValueError(
                    "Invalid bounding box coordinates! Ensure that the bounding box coordinate is in the format of [[longitude, latitude], [longitude, latitude], ...]"
                ) from e
        else:
            bounding_box = None

        create_workspace_request = CreateWorkspaceRequest(
            name=name,
            bounding_box=bounding_box,
            default_coordinate_system=default_coordinate_system,
            description=description,
            labels=labels,
        )
        model = await self._workspaces_api.create_workspace(
            org_id=str(self._org_id), create_workspace_request=create_workspace_request
        )
        return self.__parse_workspace_model(model)

    async def update_workspace(
        self,
        workspace_id: UUID,
        name: str | None = None,
        bounding_box_coordinates: list[tuple[float, float]] | None = None,
        default_coordinate_system: str | None = None,
        description: str | None = None,
        labels: list[str] | None = None,
    ) -> Workspace:
        """Update an existing workspace.

        :param workspace_id: The workspace id to update.
        :param name: The name of the workspace.
        :param bounding_box_coordinates: The coordinates list corresponding to the bounding box of the workspace.
            The coordinates list should be in the format of [[longitude, latitude], [longitude, latitude], ...]
        :param default_coordinate_system: The default coordinate system of the workspace.
        :param description: The description of the workspace.
        :param labels: The labels of the workspace.

        :returns: The updated workspace response.
        """
        # apply validation on the values
        update_workspace_request = UpdateWorkspaceRequest()
        if description is not None:
            update_workspace_request.description = description
        if default_coordinate_system is not None:
            update_workspace_request.default_coordinate_system = default_coordinate_system
        if name is not None:
            update_workspace_request.name = name
        if labels is not None:
            if len(labels) > 20:
                raise ValueError("The labels field must contain 20 or fewer items.")
            labels_adapter = TypeAdapter(list[Label])
            update_workspace_request.labels = labels_adapter.validate_python(labels)

        if bounding_box_coordinates is not None:
            try:
                bounding_box = PydanticBoundingBox(
                    type=GeometryTypeEnum.Polygon,
                    coordinates=[bounding_box_coordinates],
                )
                update_workspace_request.bounding_box = bounding_box
            except ValidationError as e:
                raise ValueError(
                    "Invalid bounding box coordinates! Ensure that the bounding box coordinate is in the format of [[longitude, latitude], [longitude, latitude], ...]"
                ) from e
        model = await self._workspaces_api.update_workspace(
            org_id=str(self._org_id), workspace_id=str(workspace_id), update_workspace_request=update_workspace_request
        )
        return self.__parse_workspace_model(model)

    async def list_instance_users(
        self, limit: int | None = None, offset: int | None = None
    ) -> Page[InstanceUserWithEmail]:
        """
        Returns a page of the list of instance users.

        :param limit: The maximum number of users to return.
        :param offset: The offset for pagination.

        :returns: A page of instance users with email addresses.
        """

        if offset is None:
            offset = 0
        if limit is None:
            limit = 50

        response = await self._instance_users_api.list_instance_users(
            org_id=str(self._org_id), limit=limit, offset=offset
        )

        if response.links.next:
            total = -1
        else:
            total = len(response.results) + offset

        return Page(
            offset=offset,
            limit=limit,
            total=total,
            items=[self._parse_instance_user_with_email_model(item) for item in response.results],
        )

    async def list_all_instance_users(
        self, limit: int | None = None, offset: int | None = None
    ) -> list[InstanceUserWithEmail]:
        """
        Returns the complete list of instance users.
        :param limit: The maximum number of users to return.
        :param offset: The offset for pagination.

        :returns: A list of instance users with email addresses.
        """
        instance_users: list[InstanceUserWithEmail] = []
        if offset is None:
            offset = 0
        if limit is None:
            limit = 50

        while True:
            instance_user_page = await self.list_instance_users(
                limit=limit,
                offset=offset,
            )
            instance_users += instance_user_page.items()
            offset += limit
            if instance_user_page.total != -1:
                break

        return sorted(instance_users, key=lambda x: x.email)

    async def add_users_to_instance(
        self, users: dict[str, list[UUID]]
    ) -> list[InstanceUserWithEmail | InstanceUserInvitation]:
        """
        Adds users to the instance.

        :param users: A dictionary mapping of user emails to their roles.
        :returns: A list of the users that were added and/or the invitations that were sent.
        """

        add_instance_users_request = AddInstanceUsersRequest(
            users=[UserRoleMapping(email=email, roles=roles) for email, roles in users.items()]
        )

        response = await self._instance_users_api.add_instance_users(
            org_id=str(self._org_id), add_instance_users_request=add_instance_users_request
        )

        result: list[InstanceUserWithEmail | InstanceUserInvitation] = []
        result.extend([self._parse_instance_user_invitation_model(item) for item in response.invitations])
        result.extend([self._parse_instance_user_with_email_model(item) for item in response.members])

        return result

    async def list_instance_user_invitations(
        self, limit: int | None = None, offset: int | None = None
    ) -> Page[InstanceUserInvitation]:
        """
        Returns a page of the list of instance user invitations.

        :param limit: The maximum number of invitations to return.
        :param offset: The offset for pagination.
        :returns: A page of instance user invitations.
        """

        if offset is None:
            offset = 0
        if limit is None:
            limit = 50
        response = await self._instance_users_api.list_instance_user_invitations(
            org_id=str(self._org_id), limit=limit, offset=offset
        )

        if response.links.next:
            total = -1
        else:
            total = len(response.results) + offset

        return Page(
            offset=offset,
            limit=limit,
            total=total,
            items=[self._parse_instance_user_invitation_model(item) for item in response.results],
        )

    async def list_all_instance_user_invitations(
        self, limit: int | None = None, offset: int | None = None
    ) -> list[InstanceUserInvitation]:
        """
        Returns the complete list of instance user invitations.

        :param limit: The maximum number of invitations to fetch per request.
        :param offset: The offset for pagination.
        :returns: A list of instance user invitations.
        """
        instance_user_invitations: list[InstanceUserInvitation] = []
        if offset is None:
            offset = 0
        if limit is None:
            limit = 50

        while True:
            instance_user_invitation_page = await self.list_instance_user_invitations(
                limit=limit,
                offset=offset,
            )
            instance_user_invitations += instance_user_invitation_page.items()
            offset += limit
            if instance_user_invitation_page.total != -1:
                break

        return sorted(instance_user_invitations, key=lambda x: x.email)

    async def delete_instance_user_invitation(self, invitation_id: UUID) -> None:
        """
        Deletes an instance user invitation.
        :param invitation_id: The ID of the invitation to delete.
        """
        await self._instance_users_api.delete_instance_user_invitation(
            org_id=str(self._org_id), invitation_id=str(invitation_id)
        )

    async def list_instance_user_roles(self) -> list[InstanceUserRoleWithPermissions]:
        """
        Returns the list of roles available in the instance.
        :returns: A list of instance user roles with their permissions.
        """

        response = await self._instance_users_api.list_instance_user_roles(org_id=str(self._org_id))
        return [self._parse_instance_user_role_model(item) for item in response.roles]

    async def remove_instance_user(self, user_id: UUID) -> None:
        """
        Removes a user from the instance.
        :param user_id: The ID of the user to remove.
        """
        await self._instance_users_api.remove_instance_user(org_id=str(self._org_id), user_id=str(user_id))

    async def update_instance_user_roles(self, user_id: UUID, roles: list[UUID]) -> InstanceUser:
        """
        Updates the roles of an instance user.

        :param user_id: The ID of the user to update.
        :param roles: The new roles to assign to the user.
        :returns: The updated instance user.
        """
        update_instance_user_roles_request = UpdateInstanceUserRolesRequest(user_id=user_id, roles=roles)
        response = await self._instance_users_api.update_instance_user_roles(
            org_id=str(self._org_id),
            user_id=str(user_id),
            update_instance_user_roles_request=update_instance_user_roles_request,
        )
        return self._parse_instance_user_model(response)
