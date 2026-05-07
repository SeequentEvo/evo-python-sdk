from evo.common import RequestMethod
from evo.common.test_tools import MockResponse, TestWithConnector
from evo.common.utils import get_header_metadata
from evo.workspaces import (
    WorkspaceAPIClient,
)

from ...data import load_test_data
from ..consts import (
    BASE_PATH,
    ORG_UUID,
)
from ..data import (
    TEST_WORKSPACE_A,
)


class TestWorkspaceClientThumbnailEndpoints(TestWithConnector):
    def setUp(self) -> None:
        super().setUp()
        self.workspace_client = WorkspaceAPIClient(connector=self.connector, org_id=ORG_UUID)
        self.setup_universal_headers(get_header_metadata(WorkspaceAPIClient.__module__))

    async def test_get_thumbnail(self):
        thumbnail_bytes: bytearray = load_test_data("thumbnail.jpg")
        self.transport.request.return_value = MockResponse(
            status_code=200,
            body=thumbnail_bytes,
            headers={"Content-Type": "image/jpeg"},
        )
        response = await self.workspace_client.get_thumbnail(workspace_id=TEST_WORKSPACE_A.id)
        self.assert_request_made(
            method=RequestMethod.GET,
            path=f"{BASE_PATH}/workspaces/{TEST_WORKSPACE_A.id}/thumbnail",
            headers={"accept": "image/jpeg"},
        )
        self.assertEqual(response, thumbnail_bytes)

    async def test_put_thumbnail(self):
        thumbnail_bytes: bytearray = load_test_data("thumbnail.jpg")
        with self.transport.set_http_response(204):
            response = await self.workspace_client.put_thumbnail(
                workspace_id=TEST_WORKSPACE_A.id, thumbnail=thumbnail_bytes
            )
        self.assert_request_made(
            method=RequestMethod.PUT,
            path=f"{BASE_PATH}/workspaces/{TEST_WORKSPACE_A.id}/thumbnail",
            headers={"Content-Type": "image/jpeg"},
            body=thumbnail_bytes,
        )
        self.assertIsNone(response, "Put thumbnail response should be None")

    async def test_delete_thumbnail(self):
        with self.transport.set_http_response(204):
            response = await self.workspace_client.delete_thumbnail(workspace_id=TEST_WORKSPACE_A.id)
        self.assert_request_made(
            method=RequestMethod.DELETE,
            path=f"{BASE_PATH}/workspaces/{TEST_WORKSPACE_A.id}/thumbnail",
        )
        self.assertIsNone(response, "Delete thumbnail response should be None")
