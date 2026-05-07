from uuid import UUID

from evo.common import ServiceUser

ORG_UUID = UUID(int=0)
USER_ID = UUID(int=2)
BASE_PATH = f"/workspace/orgs/{ORG_UUID}"
ADMIN_BASE_PATH = f"/workspace/admin/orgs/{ORG_UUID}"
TEST_USER = ServiceUser(id=USER_ID, name="Test User", email="test.user@unit.test")
