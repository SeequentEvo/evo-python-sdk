import json
import uuid
from datetime import datetime, timezone
from unittest import mock
from uuid import uuid4

from evo.blockmodels import BlockModelAPIClient
from evo.blockmodels.data import ListingVersion, RegularGridDefinition
from evo.blockmodels.endpoints.models import MissingColumnPolicy
from evo.blockmodels.endpoints.models import UnitType as EndpointUnitType
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
        self.preview_client = BlockModelAPIClient(connector=self.connector, environment=self.environment, preview=True)

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

    def make_version(self, version_id: int, version_uuid: str, mapping: dict | None = None):
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
                    "mapping": mapping if mapping is not None else {"columns": []},
                }
            )
        )

    @staticmethod
    def make_group(
        group_uuid: str,
        title: str,
        parent_group_uuid: str | None = None,
        is_hidden: bool = False,
        tags: dict | None = None,
    ) -> dict:
        group = {
            "group_uuid": group_uuid,
            "title": title,
            "parent_group_uuid": parent_group_uuid,
            "is_hidden": is_hidden,
            "resolved_missing_column_policy": "USE_PREVIOUS",
        }
        if tags is not None:
            group["tags"] = tags
        return group

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
        self.assertIsInstance(result[0], ListingVersion)
        self.assertIsInstance(result[1], ListingVersion)
        self.assertEqual(result[0].version_id, 2)
        self.assertEqual(result[1].version_id, 1)

    async def test_list_versions_accepts_unknown_referenced_unit_type(self) -> None:
        bm_id = uuid.uuid4()
        version = self.make_version(1, str(uuid.uuid4()))
        referenced_units = [
            {
                "conversion_factor": 1.0,
                "description": "A future unit",
                "symbol": "future",
                "unit_id": "future",
                "unit_type": "FUTURE_UNIT_TYPE",
            },
        ]
        captured_units = []
        list_block_model_versions = self.client._versions_api.list_block_model_versions

        async def capture_referenced_units(*args, **kwargs):
            response = await list_block_model_versions(*args, **kwargs)
            captured_units.extend(response.referenced_units)
            return response

        with (
            self.transport.set_http_response(
                200,
                json.dumps(
                    {
                        "count": 1,
                        "limit": 100,
                        "offset": 0,
                        "results": [version],
                        "total": 1,
                        "referenced_units": referenced_units,
                    }
                ),
                headers={"Content-Type": "application/json"},
            ),
            mock.patch.object(
                self.client._versions_api,
                "list_block_model_versions",
                side_effect=capture_referenced_units,
            ),
        ):
            result = await self.client.list_versions(bm_id)

        self.assertEqual(len(result), 1)
        self.assertIs(captured_units[0].unit_type, EndpointUnitType.UNKNOWN)

    async def test_list_versions_with_preview_sends_header(self) -> None:
        bm_id = uuid.uuid4()
        v1 = self.make_version(1, str(uuid.uuid4()))
        with self.transport.set_http_response(
            200,
            json.dumps({"count": 1, "limit": 100, "offset": 0, "results": [v1], "total": 1, "referenced_units": []}),
            headers={"Content-Type": "application/json"},
        ):
            await self.preview_client.list_versions(bm_id)

        request_headers = self.transport.request.call_args.kwargs["headers"]
        self.assertEqual(request_headers["Api-Preview"], "opt-in")

    async def test_list_versions_empty_returns_empty(self) -> None:
        bm_id = uuid.uuid4()
        with self.transport.set_http_response(
            200,
            json.dumps({"count": 0, "limit": 100, "offset": 0, "results": [], "total": 0, "referenced_units": []}),
            headers={"Content-Type": "application/json"},
        ):
            result = await self.client.list_versions(bm_id)
        self.assertEqual(result, [])

    async def test_get_version_returns_version_with_tags(self) -> None:
        from evo.blockmodels.data import Version

        bm_id = uuid.uuid4()
        version_uuid = uuid.uuid4()
        version = self.make_version(2, str(version_uuid))
        version["mapping"]["columns"] = [
            {
                "col_id": str(uuid.uuid4()),
                "data_type": "Float64",
                "title": "Au",
                "unit_id": "g/t",
                "tags": {"source": "assay"},
            }
        ]
        with self.transport.set_http_response(
            200,
            json.dumps(version),
            headers={"Content-Type": "application/json"},
        ):
            result = await self.client.get_version(bm_id, version_uuid)

        self.assertIsInstance(result, Version)
        self.assertEqual(result.version_id, 2)
        self.assertEqual(result.columns[0].tags, {"source": "assay"})

    async def test_get_version_with_preview_sends_header(self) -> None:
        bm_id = uuid.uuid4()
        version_uuid = uuid.uuid4()
        version = self.make_version(1, str(version_uuid))
        with self.transport.set_http_response(
            200,
            json.dumps(version),
            headers={"Content-Type": "application/json"},
        ):
            await self.preview_client.get_version(bm_id, version_uuid)

        request_headers = self.transport.request.call_args.kwargs["headers"]
        self.assertEqual(request_headers["Api-Preview"], "opt-in")

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
        self.assertIsInstance(result[0], ListingVersion)
        self.assertIsInstance(result[1], ListingVersion)
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

    async def test_list_all_block_models_deleted_parameter(self) -> None:
        bm = self.make_bm("deleted-bm")
        with self.transport.set_http_response(
            200,
            json.dumps({"count": 1, "limit": 100, "offset": 0, "results": [bm], "total": 1}),
            headers={"Content-Type": "application/json"},
        ):
            result = await self.client.list_all_block_models(deleted=True)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].name, "deleted-bm")

    async def test_list_versions_returns_listing_groups(self) -> None:
        bm_id = uuid.uuid4()
        parent_uuid = str(uuid.uuid4())
        child_uuid = str(uuid.uuid4())
        version = self.make_version(
            1,
            str(uuid.uuid4()),
            mapping={
                "columns": [],
                "groups": [
                    self.make_group(parent_uuid, "Assays"),
                    self.make_group(child_uuid, "Internal", parent_group_uuid=parent_uuid, is_hidden=True),
                ],
            },
        )
        with self.transport.set_http_response(
            200,
            json.dumps(
                {"count": 1, "limit": 100, "offset": 0, "results": [version], "total": 1, "referenced_units": []}
            ),
            headers={"Content-Type": "application/json"},
        ):
            result = await self.client.list_versions(bm_id)

        groups = result[0].groups
        self.assertEqual(len(groups), 2)
        self.assertEqual(str(groups[0].group_uuid), parent_uuid)
        self.assertEqual(groups[0].title, "Assays")
        self.assertIsNone(groups[0].parent_group_uuid)
        self.assertFalse(groups[0].is_hidden)
        self.assertEqual(str(groups[1].group_uuid), child_uuid)
        self.assertEqual(groups[1].title, "Internal")
        self.assertEqual(str(groups[1].parent_group_uuid), parent_uuid)
        self.assertTrue(groups[1].is_hidden)

    async def test_get_version_returns_resolved_groups_with_tags(self) -> None:
        bm_id = uuid.uuid4()
        version_uuid = uuid.uuid4()
        group_uuid = str(uuid.uuid4())
        version = self.make_version(
            2,
            str(version_uuid),
            mapping={
                "columns": [],
                "groups": [
                    {
                        "group_uuid": group_uuid,
                        "title": "Assays",
                        "parent_group_uuid": None,
                        "is_hidden": False,
                        "missing_column_policy": "REJECT",
                        "resolved_missing_column_policy": "REJECT",
                        "tags": {"source": "assay"},
                    }
                ],
            },
        )
        with self.transport.set_http_response(
            200,
            json.dumps(version),
            headers={"Content-Type": "application/json"},
        ):
            result = await self.client.get_version(bm_id, version_uuid)

        self.assertEqual(len(result.groups), 1)
        self.assertEqual(str(result.groups[0].group_uuid), group_uuid)
        self.assertEqual(result.groups[0].tags, {"source": "assay"})
        self.assertEqual(result.groups[0].resolved_missing_column_policy, MissingColumnPolicy.REJECT)

    async def test_list_versions_preserves_column_group_assignment(self) -> None:
        bm_id = uuid.uuid4()
        group_uuid = str(uuid.uuid4())
        version = self.make_version(
            1,
            str(uuid.uuid4()),
            mapping={
                "columns": [
                    {
                        "col_id": str(uuid.uuid4()),
                        "data_type": "Float64",
                        "title": "Au",
                        "unit_id": "g/t",
                        "group_uuid": group_uuid,
                    },
                    {"col_id": str(uuid.uuid4()), "data_type": "Float64", "title": "Ag", "unit_id": "g/t"},
                ],
                "groups": [self.make_group(group_uuid, "Assays")],
            },
        )
        with self.transport.set_http_response(
            200,
            json.dumps(
                {"count": 1, "limit": 100, "offset": 0, "results": [version], "total": 1, "referenced_units": []}
            ),
            headers={"Content-Type": "application/json"},
        ):
            result = await self.client.list_versions(bm_id)

        columns = result[0].columns
        self.assertEqual(str(columns[0].group_uuid), group_uuid)
        self.assertIsNone(columns[1].group_uuid)

    async def test_get_version_preserves_column_group_assignment(self) -> None:
        bm_id = uuid.uuid4()
        version_uuid = uuid.uuid4()
        group_uuid = str(uuid.uuid4())
        version = self.make_version(
            1,
            str(version_uuid),
            mapping={
                "columns": [
                    {
                        "col_id": str(uuid.uuid4()),
                        "data_type": "Float64",
                        "title": "Au",
                        "unit_id": "g/t",
                        "group_uuid": group_uuid,
                    },
                    {"col_id": str(uuid.uuid4()), "data_type": "Float64", "title": "Ag", "unit_id": "g/t"},
                ],
                "groups": [self.make_group(group_uuid, "Assays")],
            },
        )
        with self.transport.set_http_response(
            200,
            json.dumps(version),
            headers={"Content-Type": "application/json"},
        ):
            result = await self.client.get_version(bm_id, version_uuid)

        self.assertEqual(str(result.columns[0].group_uuid), group_uuid)
        self.assertIsNone(result.columns[1].group_uuid)

    async def test_get_version_without_groups_returns_empty_groups(self) -> None:
        bm_id = uuid.uuid4()
        version_uuid = uuid.uuid4()
        version = self.make_version(
            1,
            str(version_uuid),
            mapping={
                "columns": [{"col_id": str(uuid.uuid4()), "data_type": "Float64", "title": "Au", "unit_id": "g/t"}]
            },
        )
        with self.transport.set_http_response(
            200,
            json.dumps(version),
            headers={"Content-Type": "application/json"},
        ):
            result = await self.client.get_version(bm_id, version_uuid)

        self.assertEqual(result.groups, [])

    async def test_list_all_versions_preserves_groups_across_pages(self) -> None:
        bm_id = uuid.uuid4()
        group_uuid = str(uuid.uuid4())
        v2 = self.make_version(2, str(uuid.uuid4()))
        v1 = self.make_version(
            1,
            str(uuid.uuid4()),
            mapping={"columns": [], "groups": [self.make_group(group_uuid, "Assays")]},
        )
        self.transport.request.side_effect = [
            MockResponse(
                status_code=200,
                content=json.dumps(
                    {"count": 1, "limit": 1, "offset": 0, "results": [v2], "total": 2, "referenced_units": []}
                ),
                headers={"Content-Type": "application/json"},
            ),
            MockResponse(
                status_code=200,
                content=json.dumps(
                    {"count": 1, "limit": 1, "offset": 1, "results": [v1], "total": 2, "referenced_units": []}
                ),
                headers={"Content-Type": "application/json"},
            ),
        ]

        result = await self.client.list_all_versions(bm_id, page_limit=1)
        self.assertEqual(result[0].groups, [])
        self.assertEqual(len(result[1].groups), 1)
        self.assertEqual(str(result[1].groups[0].group_uuid), group_uuid)
        self.assertEqual(result[1].groups[0].title, "Assays")
