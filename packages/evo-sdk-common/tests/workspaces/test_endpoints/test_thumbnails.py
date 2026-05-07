from parameterized import parameterized

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

    @parameterized.expand(
        [
            ("jpeg", "thumbnail.jpg", "image/jpeg"),
            ("png", "thumbnail_2.png", "image/png"),
        ]
    )
    async def test_get_thumbnail(self, _name: str, filename: str, content_type: str):
        thumbnail_bytes: bytearray = load_test_data(filename)
        self.transport.request.return_value = MockResponse(
            status_code=200,
            body=thumbnail_bytes,
            headers={"Content-Type": content_type},
        )
        response = await self.workspace_client.get_thumbnail(workspace_id=TEST_WORKSPACE_A.id)
        self.assert_request_made(
            method=RequestMethod.GET,
            path=f"{BASE_PATH}/workspaces/{TEST_WORKSPACE_A.id}/thumbnail",
            headers={"accept": "image/jpeg, image/png"},
        )
        self.assertEqual(response, thumbnail_bytes)

    @parameterized.expand(
        [
            ("jpeg", "thumbnail.jpg", "image/jpeg"),
            ("png", "thumbnail_2.png", "image/png"),
        ]
    )
    async def test_put_thumbnail(self, _name: str, filename: str, content_type: str):
        thumbnail_bytes: bytearray = load_test_data(filename)
        with self.transport.set_http_response(204):
            response = await self.workspace_client.put_thumbnail(
                workspace_id=TEST_WORKSPACE_A.id, thumbnail=thumbnail_bytes
            )
        self.assert_request_made(
            method=RequestMethod.PUT,
            path=f"{BASE_PATH}/workspaces/{TEST_WORKSPACE_A.id}/thumbnail",
            headers={"Content-Type": content_type},
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
