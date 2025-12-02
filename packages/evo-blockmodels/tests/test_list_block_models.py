import json
import uuid
from datetime import datetime, timezone
from uuid import uuid4

from evo.blockmodels import BlockModelAPIClient
from evo.blockmodels.data import RegularGridDefinition, Version
from evo.common import Environment
from evo.common.test_tools import (
    BASE_URL,
    ORG,
    WORKSPACE_ID,
    MockResponse,
    TestWithConnector,
    TestWithStorage,
)


class TestListBlockModels(TestWithConnector, TestWithStorage):
    def setUp(self) -> None:
        TestWithConnector.setUp(self)
        TestWithStorage.setUp(self)
        self.environment = Environment(hub_url=BASE_URL, org_id=ORG.id, workspace_id=WORKSPACE_ID)
        self.client = BlockModelAPIClient(connector=self.connector, environment=self.environment)

    def make_bm(self, name: str):
        return {
            "bbox": {
                "x_minmax": {"max": 1.0, "min": 0},
                "y_minmax": {"max": 1.0, "min": 0},
                "z_minmax": {"max": 1.0, "min": 0},
            },
            "block_rotation": [{"angle": 0, "axis": "x"}],
            "bm_uuid": str(uuid4()),
            "coordinate_reference_system": "string",
            "created_at": str(datetime.now(timezone.utc)),
            "created_by": {"email": "c@example.com", "id": str(uuid4()), "name": "creator"},
            "description": "string",
            "fill_subblocks": False,
            "geoscience_object_id": str(uuid4()),
            "last_updated_at": str(datetime.now(timezone.utc)),
            "last_updated_by": {"email": "u@example.com", "id": str(uuid4()), "name": "updater"},
            "model_origin": {"x": 0, "y": 0, "z": 0},
            "name": name,
            "normalized_rotation": [0.0, 0.0, 0.0],
            "org_uuid": str(uuid4()),
            "size_options": {
                "block_size": {"x": 1.0, "y": 1.0, "z": 1.0},
                "model_type": "regular",
                "n_blocks": {"nx": 1, "ny": 1, "nz": 1},
            },
            "workspace_id": str(uuid4()),
        }

    def make_version(self, version_id: int, version_uuid: str):
        return json.loads(
            json.dumps(
                {
                    "version_id": version_id,
                    "version_uuid": version_uuid,
                    "bm_uuid": str(uuid.uuid4()),
                    "created_at": str(datetime.now(timezone.utc)),
                    "created_by": {"email": "c@example.com", "id": str(uuid.uuid4()), "name": "creator"},
                    "comment": f"Version {version_id}",
                    "bbox": None,
                    "base_version_id": None,
                    "parent_version_id": version_id - 1 if version_id > 1 else None,
                    "geoscience_version_id": str(version_id),
                    "mapping": {"columns": []},
                }
            )
        )

    async def test_list_block_models_converts_endpoint_models_to_dataclass(self) -> None:
        # Prepare a fake endpoint BlockModel
        endpoint_bm = self.make_bm("Test BM")

        with self.transport.set_http_response(
            200,
            json.dumps({"count": 1, "limit": 0, "offset": 0, "results": [endpoint_bm], "total": 1}),
            headers={"Content-Type": "application/json"},
        ):
            result = await self.client.list_block_models()

        self.assertEqual(len(result), 1)
        bm = result[0]
        # Ensure converted dataclass has expected identity and grid definition
        self.assertEqual(str(bm.id), endpoint_bm["bm_uuid"])
        self.assertEqual(bm.name, "Test BM")
        self.assertIsInstance(bm.grid_definition, RegularGridDefinition)
        self.assertEqual(bm.grid_definition.n_blocks, [1, 1, 1])

    async def test_list_block_models_empty_list_returns_empty(self) -> None:
        with self.transport.set_http_response(
            200,
            json.dumps({"count": 1, "limit": 0, "offset": 0, "results": [], "total": 1}),
            headers={"Content-Type": "application/json"},
        ):
            result = await self.client.list_block_models()
        self.assertEqual(result, [])

    async def test_list_all_block_models_paginates_and_returns_all(self) -> None:
        # create three endpoint models
        bm1 = self.make_bm("pg-1")
        bm2 = self.make_bm("pg-2")
        bm3 = self.make_bm("pg-3")
        responses = [
            MockResponse(
                status_code=200,
                content=json.dumps({"count": 2, "limit": 2, "offset": 0, "results": [bm1, bm2], "total": 3}),
                headers={"Content-Type": "application/json"},
            ),
            MockResponse(
                status_code=200,
                content=json.dumps({"count": 1, "limit": 2, "offset": 2, "results": [bm3], "total": 3}),
                headers={"Content-Type": "application/json"},
            ),
        ]
        self.transport.request.side_effect = responses

        result = await self.client.list_all_block_models(page_limit=2)
        self.assertEqual([r.name for r in result], ["pg-1", "pg-2", "pg-3"])

    async def test_list_versions_returns_versions(self) -> None:
        bm_id = uuid.uuid4()
        v1 = self.make_version(1, str(uuid.uuid4()))
        v2 = self.make_version(2, str(uuid.uuid4()))
        with self.transport.set_http_response(
            200,
            json.dumps(
                {"count": 2, "limit": 100, "offset": 0, "results": [v2, v1], "total": 2, "referenced_units": []}
            ),
            headers={"Content-Type": "application/json"},
        ):
            result = await self.client.list_versions(bm_id)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], Version)
        self.assertIsInstance(result[1], Version)
        self.assertEqual(result[0].version_id, 2)
        self.assertEqual(result[1].version_id, 1)

    async def test_list_versions_empty_returns_empty(self) -> None:
        bm_id = uuid.uuid4()
        with self.transport.set_http_response(
            200,
            json.dumps({"count": 0, "limit": 100, "offset": 0, "results": [], "total": 0, "referenced_units": []}),
            headers={"Content-Type": "application/json"},
        ):
            result = await self.client.list_versions(bm_id)
        self.assertEqual(result, [])

    async def test_list_all_versions_returns_all_versions(self) -> None:
        bm_id = uuid.uuid4()
        v1 = self.make_version(1, str(uuid.uuid4()))
        v2 = self.make_version(2, str(uuid.uuid4()))
        responses = [
            MockResponse(
                status_code=200,
                content=json.dumps(
                    {"count": 2, "limit": 2, "offset": 0, "results": [v2, v1], "total": 2, "referenced_units": []}
                ),
                headers={"Content-Type": "application/json"},
            ),
        ]
        self.transport.request.side_effect = responses

        result = await self.client.list_all_versions(bm_id, page_limit=2)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], Version)
        self.assertIsInstance(result[1], Version)
        self.assertEqual(result[0].version_id, 2)
        self.assertEqual(result[1].version_id, 1)

    async def test_list_all_versions_paginates_across_pages(self) -> None:
        bm_id = uuid.uuid4()
        v1 = self.make_version(1, str(uuid.uuid4()))
        v2 = self.make_version(2, str(uuid.uuid4()))
        v3 = self.make_version(3, str(uuid.uuid4()))
        responses = [
            MockResponse(
                status_code=200,
                content=json.dumps(
                    {"count": 2, "limit": 2, "offset": 0, "results": [v3, v2], "total": 3, "referenced_units": []}
                ),
                headers={"Content-Type": "application/json"},
            ),
            MockResponse(
                status_code=200,
                content=json.dumps(
                    {"count": 1, "limit": 2, "offset": 2, "results": [v1], "total": 3, "referenced_units": []}
                ),
                headers={"Content-Type": "application/json"},
            ),
        ]
        self.transport.request.side_effect = responses

        result = await self.client.list_all_versions(bm_id, page_limit=2)
        self.assertEqual(len(result), 3)
        self.assertEqual([v.version_id for v in result], [3, 2, 1])

    async def test_list_all_versions_empty_returns_empty(self) -> None:
        bm_id = uuid.uuid4()
        with self.transport.set_http_response(
            200,
            json.dumps({"count": 0, "limit": 100, "offset": 0, "results": [], "total": 0, "referenced_units": []}),
            headers={"Content-Type": "application/json"},
        ):
            result = await self.client.list_all_versions(bm_id)
        self.assertEqual(result, [])
