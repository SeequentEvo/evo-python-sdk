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

from evo.common import ServiceUser

ORG_UUID = UUID(int=0)
USER_ID = UUID(int=2)
BASE_PATH = f"/workspace/orgs/{ORG_UUID}"
ADMIN_BASE_PATH = f"/workspace/admin/orgs/{ORG_UUID}"
TEST_USER = ServiceUser(id=USER_ID, name="Test User", email="test.user@unit.test")
