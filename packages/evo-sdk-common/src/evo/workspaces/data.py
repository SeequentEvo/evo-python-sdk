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

import enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import NamedTuple
from uuid import UUID

from evo.common import Environment, ServiceUser
from evo.common.data import OrderByOperatorEnum

from .exceptions import UserPermissionTypeError

__all__ = [
    "BasicWorkspace",
    "BoundingBox",
    "Coordinate",
    "OrderByOperatorEnum",
    "User",
    "UserPermission",
    "UserRole",
    "Workspace",
    "WorkspaceOrderByEnum",
    "WorkspaceRole",
]


class WorkspaceOrderByEnum(str, enum.Enum):
    name = "name"
    created_at = "created_at"
    updated_at = "updated_at"
    user_role = "user_role"


class UserPermission(enum.Flag):
    """Effective user permissions within a workspace."""

    read = enum.auto()
    """The user can read data within the workspace."""

    write = enum.auto()
    """The user can write data within the workspace."""

    manage = enum.auto()
    """The user can manage the workspace itself."""


class WorkspaceRole(enum.Flag):
    """The role of the user within a workspace."""

    viewer = UserPermission.read.value
    """A read-only user."""

    editor = UserPermission.write.value | viewer
    """A user that can read and write data."""

    owner = UserPermission.manage.value | editor
    """A user that can read and write, and manage the workspace."""

    def __contains__(self, item: UserPermission | WorkspaceRole) -> bool:
        if isinstance(item, (UserPermission, WorkspaceRole)):
            return (item.value & self.value) == item.value
        else:
            raise UserPermissionTypeError(
                f"The required permission is invalid, "
                f"expected UserPermission or WorkspaceRole but got type {type(item).__name__}"
            )


@dataclass(frozen=True, kw_only=True)
class UserRole:
    """The role of a user in a workspace."""

    user_id: UUID
    """The ID of the user."""

    role: WorkspaceRole
    """The role of the user in the workspace."""


@dataclass(frozen=True, kw_only=True)
class User(UserRole):
    """The info of a user in a workspace."""

    email: str | None = None
    """The email of the user."""

    full_name: str | None = None
    """The full name of the user."""


class Coordinate(NamedTuple):
    latitude: float
    longitude: float


@dataclass(frozen=True, kw_only=True)
class BoundingBox:
    coordinates: list[list[Coordinate]]
    type: str


@dataclass(frozen=True, kw_only=True)
class BasicWorkspace:
    id: UUID
    """The workspace UUID."""

    display_name: str
    """The workspace display name."""


@dataclass(frozen=True, kw_only=True)
class Workspace(BasicWorkspace):
    """Metadata about a workspace environment."""

    description: str | None
    """The workspace description."""

    user_role: WorkspaceRole | None
    """The role of the current user in the workspace."""

    org_id: UUID
    """UUID of the organization the workspace belongs to."""

    hub_url: str
    """The URL of the hub the workspace resides in."""

    created_at: datetime
    """A timestamp representing when the workspace was created."""

    created_by: ServiceUser
    """The info of the user that created the workspace."""

    updated_at: datetime
    """A timestamp representing when the workspace was updated last."""

    updated_by: ServiceUser
    """The info of the user that updated the workspace."""

    bounding_box: BoundingBox | None = None
    """The bounding box of the workspace."""

    default_coordinate_system: str = ""
    """The default coordinate system of the workspace."""

    labels: list[str] = field(default_factory=list)
    """The labels associated with the workspace."""

    def user_has_permission(self, required_permission: UserPermission | WorkspaceRole) -> bool:
        """Test whether the current user has at least the given permission for this workspace.

        :param required_permission: the minimum role or permission the user must have

        :returns: whether the user has sufficient permissions for this workspace - any `user_role` with all the required
          permissions will return True.
        """
        return required_permission in self.user_role

    def get_environment(self) -> Environment:
        """Return an environment that can be used to interact with the workspace."""
        return Environment(hub_url=self.hub_url, org_id=self.org_id, workspace_id=self.id)

    def __hash__(self) -> int:
        return hash(self.id)
