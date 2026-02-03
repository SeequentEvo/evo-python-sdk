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


from datetime import timezone
from uuid import UUID

from evo.common import ServiceUser

from .data import (
    AddedInstanceUsers,
    BasicWorkspace,
    BoundingBox,
    Coordinate,
    InstanceRole,
    InstanceRoleWithPermissions,
    InstanceUser,
    InstanceUserInvitation,
    InstanceUserWithEmail,
    User,
    UserRole,
    Workspace,
    WorkspaceRole,
)
from .endpoints.models import (
    AddInstanceUsersResponse,
    BaseInstanceUserResponse,
    BasicWorkspaceResponse,
    ListInstanceRolesResponse,
    ListInstanceUserInvitationsResponse,
    WorkspaceRoleOptionalResponse,
    WorkspaceRoleRequiredResponse,
)
from .endpoints.models import BoundingBox as PydanticBoundingBox
from .endpoints.models import Coordinate as PydanticCoordinate
from .endpoints.models import User as PydanticUser
from .endpoints.models import UserRole as PydanticUserRole

__all__ = [
    "bounding_box",
    "instance_user_invitation_model",
    "instance_user_model",
    "instance_user_role_model",
    "instance_user_with_email_model",
    "user_model",
    "user_role_model",
    "workspace_basic_model",
    "workspace_model",
]


def bounding_box(model: PydanticBoundingBox) -> BoundingBox:
    """
    Parse a BoundingBox from the generated model.

    :param model: The model returned by the generated code.
    :return: A BoundingBox instance.
    """

    def convert_coordinate(pydantic_coordinate: PydanticCoordinate) -> Coordinate:
        return Coordinate(latitude=pydantic_coordinate.root[1], longitude=pydantic_coordinate.root[0])

    coordinates = [[convert_coordinate(coord) for coord in coord_list] for coord_list in model.coordinates]

    return BoundingBox(coordinates=coordinates, type=str(model.type.value))


def workspace_model(
    model: WorkspaceRoleOptionalResponse | WorkspaceRoleRequiredResponse, org_id: UUID, base_url: str
) -> Workspace:
    """
    Parse a Workspace from the generated model.
    :param model: The model returned by the generated code.
    :param org_id: The organization ID.
    :param base_url: The base URL of the hub.
    :return: A Workspace instance.
    """
    parsed_bounding_box = None
    if model.bounding_box:
        parsed_bounding_box = bounding_box(model.bounding_box)

    return Workspace(
        id=model.id,
        org_id=org_id,
        hub_url=base_url,
        display_name=model.name,
        description=model.description,
        user_role=WorkspaceRole[str(model.current_user_role.value)] if model.current_user_role else None,
        created_at=model.created_at.replace(tzinfo=timezone.utc),
        created_by=ServiceUser.from_model(model.created_by),
        updated_at=model.updated_at.replace(tzinfo=timezone.utc),
        updated_by=ServiceUser.from_model(model.updated_by),
        bounding_box=parsed_bounding_box,
        default_coordinate_system=model.default_coordinate_system,
        labels=model.labels,
    )


def workspace_basic_model(model: BasicWorkspaceResponse) -> BasicWorkspace:
    """
    Parse a BasicWorkspace from the generated model.
    :param model: The model returned by the generated code.
    :return: A BasicWorkspace instance.
    """
    return BasicWorkspace(
        id=model.id,
        display_name=model.name,
    )


def user_role_model(model: PydanticUserRole) -> UserRole:
    """
    Parse a UserRole from the generated model.
    :param model: The model returned by the generated code.
    :return: A UserRole instance.
    """
    return UserRole(user_id=model.user_id, role=WorkspaceRole[str(model.role.value)])


def user_model(model: PydanticUser) -> User:
    """
    Parse a User from the generated model.
    :param model: The model returned by the generated code.
    :return: A User instance.
    """
    return User(
        user_id=model.user_id,
        role=WorkspaceRole[str(model.role.value)],
        email=model.email,
        full_name=model.full_name,
    )


def instance_user_model(model: BaseInstanceUserResponse) -> InstanceUser:
    """
    Parse an InstanceUser from the generated model.
    :param model: The model returned by the generated code.
    :return: An InstanceUser instance.
    """
    return InstanceUser(
        user_id=model.id,
        roles=[InstanceRole(role_id=role.id, name=role.name, description=role.description) for role in model.roles],
    )


def instance_user_with_email_model(model: BaseInstanceUserResponse) -> InstanceUserWithEmail:
    """
    Parse an InstanceUserWithEmail from the generated model.
    :param model: The model returned by the generated code.
    :return: An InstanceUserWithEmail instance.
    """
    return InstanceUserWithEmail(
        email=model.email,
        full_name=model.full_name,
        user_id=model.id,
        roles=[InstanceRole(role_id=role.id, name=role.name, description=role.description) for role in model.roles],
    )


def instance_user_invitation_model(model: ListInstanceUserInvitationsResponse) -> InstanceUserInvitation:
    """
    Parse an InstanceUserInvitation from the generated model.
    :param model: The model returned by the generated code.
    :return: An InstanceUserInvitation instance.
    """
    return InstanceUserInvitation(
        email=model.email,
        invitation_id=model.id,
        roles=[InstanceRole(role_id=role.id, name=role.name, description=role.description) for role in model.roles],
        invited_at=model.created_date.replace(tzinfo=timezone.utc),
        expiration_date=model.expiration_date.replace(tzinfo=timezone.utc),
        invited_by=model.invited_by_email,
        status=model.status,
    )


def instance_user_role_model(model: ListInstanceRolesResponse) -> InstanceRoleWithPermissions:
    """
    Parse an InstanceUserRoleWithPermissions from the generated model.
    :param model: The model returned by the generated code.
    :return: An InstanceUserRoleWithPermissions instance.
    """
    return InstanceRoleWithPermissions(
        role_id=model.id, name=model.name, description=model.description, permissions=model.permissions
    )


def add_instance_user_model(model: AddInstanceUsersResponse):
    return AddedInstanceUsers(
        members=[instance_user_with_email_model(user) for user in model.members],
        invitations=[instance_user_invitation_model(invitation) for invitation in model.invitations],
    )
