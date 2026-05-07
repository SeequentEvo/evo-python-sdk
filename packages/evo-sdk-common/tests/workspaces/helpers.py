from uuid import UUID

from evo.common.test_tools import BASE_URL, utc_datetime
from evo.workspaces import (
    BasicWorkspace,
    BoundingBox,
    InstanceRole,
    InstanceRoleWithPermissions,
    InstanceUser,
    InstanceUserInvitation,
    InstanceUserWithEmail,
    Workspace,
    WorkspaceRole,
)

from .consts import (
    ORG_UUID,
    TEST_USER,
)


def _make_instance_user(user_id: UUID, role_name: str, role_id: int) -> InstanceUser:
    """Factory method to create test instance user objects."""
    return InstanceUser(user_id=user_id, roles=[make_instance_role(UUID(int=role_id), role_name)])


def make_workspace(ws_id: UUID, name: str, bounding_box: BoundingBox | None = None) -> Workspace:
    """Factory method to create test workspace objects."""
    return Workspace(
        id=ws_id,
        display_name=name.title(),
        description=name.lower(),
        user_role=WorkspaceRole.owner,
        org_id=ORG_UUID,
        hub_url=BASE_URL,
        created_at=utc_datetime(2020, 1, 1),
        created_by=TEST_USER,
        updated_at=utc_datetime(2020, 1, 1),
        updated_by=TEST_USER,
        bounding_box=bounding_box,
    )


def make_basic_workspace(ws_id: UUID, name: str) -> BasicWorkspace:
    """Factory method to create test basic workspace objects."""
    return BasicWorkspace(
        id=ws_id,
        display_name=name.title(),
    )


def make_instance_role(role_id: UUID, name: str) -> InstanceRole:
    """Factory method to create test instance role objects."""
    return InstanceRole(
        role_id=role_id,
        name=name.title(),
        description=name.lower(),
    )


def make_instance_role_with_permissions(role_id: UUID, name: str) -> InstanceRoleWithPermissions:
    """Factory method to create test instance role objects."""
    return InstanceRoleWithPermissions(
        role_id=role_id, name=name.title(), description=name.lower(), permissions=[name.lower() + " permission"]
    )


def make_instance_user_with_email(
    user_id: UUID, email: str, full_name: str, role_name: str, role_id: int
) -> InstanceUserWithEmail:
    """Factory method to create test instance user objects."""
    return InstanceUserWithEmail(
        user_id=user_id, email=email, full_name=full_name, roles=[make_instance_role(UUID(int=role_id), role_name)]
    )


def make_instance_user_invitation(
    invitation_id: UUID, email: str, status: str, role_name: str, role_id: int
) -> InstanceUserInvitation:
    """Factory method to create test instance user invitation objects."""
    return InstanceUserInvitation(
        email=email,
        invitation_id=invitation_id,
        invited_at=utc_datetime(2026, 1, 1, 12, 0, 0),
        expiration_date=utc_datetime(2026, 1, 15, 12, 0, 0),
        invited_by="admin.user@bentley.com",
        status=status,
        roles=[make_instance_role(UUID(int=role_id), role_name)],
    )
