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

from uuid import UUID

from evo.workspaces import (
    BoundingBox,
    Coordinate,
)

from .helpers import (
    make_basic_workspace,
    make_instance_role_with_permissions,
    make_instance_user_invitation,
    make_instance_user_with_email,
    make_workspace,
)

TEST_WORKSPACE_A = make_workspace(UUID(int=0xA), "Test Workspace A")
TEST_WORKSPACE_B = make_workspace(UUID(int=0xB), "Test Workspace B")
TEST_WORKSPACE_C = make_workspace(UUID(int=0xC), "Test Workspace C")
TEST_WORKSPACE_D = make_workspace(
    UUID(int=0xD),
    "Test Workspace D",
    bounding_box=BoundingBox(
        coordinates=[
            [
                Coordinate(longitude=100, latitude=0),
                Coordinate(longitude=101, latitude=0),
                Coordinate(longitude=101, latitude=1),
                Coordinate(longitude=100, latitude=1),
                Coordinate(longitude=100, latitude=0),
            ]
        ],
        type="Polygon",
    ),
)

TEST_BASIC_WORKSPACE_A = make_basic_workspace(UUID(int=0xA), "Test Workspace A")
TEST_BASIC_WORKSPACE_B = make_basic_workspace(UUID(int=0xB), "Test Workspace B")
TEST_BASIC_WORKSPACE_C = make_basic_workspace(UUID(int=0xC), "Test Workspace C")

INSTANCE_USER_1 = make_instance_user_with_email(UUID(int=1), "test.user1@gmail.com", "User 1", "Evo Owner", 3)
INSTANCE_USER_2 = make_instance_user_with_email(UUID(int=2), "test.user2@gmail.com", "User 2", "Evo Admin", 2)
INSTANCE_USER_3 = make_instance_user_with_email(UUID(int=3), "test.user3@gmail.com", "User 3", "Evo User", 1)
INVITATION_1 = make_instance_user_invitation(UUID(int=1), "external.user1@gmail.com", "Pending", "Evo User", 1)
INVITATION_2 = make_instance_user_invitation(UUID(int=2), "external.user2@gmail.com", "Accepted", "Evo Admin", 2)
INVITATION_3 = make_instance_user_invitation(UUID(int=3), "external.user3@gmail.com", "Pending", "Evo User", 1)

INSTANCE_USER_ROLE = make_instance_role_with_permissions(UUID(int=1), "Evo User")
INSTANCE_ADMIN_ROLE = make_instance_role_with_permissions(UUID(int=2), "Evo Admin")
