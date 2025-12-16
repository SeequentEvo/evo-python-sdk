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

import uuid
from datetime import datetime
from typing import Iterable
from unittest import mock

import pyarrow

from evo.blockmodels import BlockModelAPIClient
from evo.blockmodels.data import (
    BaseGridDefinition,
    FlexibleGridDefinition,
    FullySubBlockedGridDefinition,
    OctreeGridDefinition,
    RegularGridDefinition,
)
from evo.blockmodels.endpoints import models
from evo.blockmodels.endpoints.models import JobResponse, JobStatus, RotationAxis
from evo.blockmodels.exceptions import CacheNotConfiguredException, JobFailedException
from evo.common import ServiceUser
from evo.common.data import HTTPHeaderDict, RequestMethod
from evo.common.test_tools import BASE_URL, MockResponse, TestWithConnector, TestWithStorage
from evo.common.utils import get_header_metadata
from utils import JobPollingRequestHandler

BM_UUID = uuid.uuid4()
GOOSE_UUID = uuid.uuid4()
GOOSE_VERSION_ID = "2"
DATE = datetime(2021, 1, 1)
MODEL_USER = models.UserInfo(email="test@test.com", name="Test User", id=uuid.uuid4())
USER = ServiceUser.from_model(MODEL_USER)
BM_BBOX = models.BBoxXYZ(
    x_minmax=models.FloatRange(min=0, max=10),
    y_minmax=models.FloatRange(min=0, max=10),
    z_minmax=models.FloatRange(min=0, max=10),
)
GRID_DEFINITION = RegularGridDefinition(
    model_origin=[0, 0, 0], rotations=[(RotationAxis.x, 20)], n_blocks=[10, 10, 10], block_size=[1, 1, 1]
)
SUBBLOCKED_GRID_DEFINITION = FullySubBlockedGridDefinition(
    model_origin=[0, 0, 0],
    rotations=[],
    n_parent_blocks=[10, 10, 10],
    parent_block_size=[1, 1, 1],
    n_subblocks_per_parent=[5, 1, 2],
)


def _mock_create_result(environment, size_option=None) -> models.BlockModelAndJobURL:
    if size_option is None:
        size_option = models.SizeOptionsRegular(
            model_type="regular",
            n_blocks=models.Size3D(nx=10, ny=10, nz=10),
            block_size=models.BlockSize(x=1, y=1, z=1),
        )
    return models.BlockModelAndJobURL(
        bbox=BM_BBOX,
        block_rotation=[models.Rotation(axis=RotationAxis.x, angle=20)],
        bm_uuid=BM_UUID,
        name="Test BM",
        description="Test Block Model",
        coordinate_reference_system="EPSG:4326",
        size_unit_id="m",
        workspace_id=environment.workspace_id,
        org_uuid=environment.org_id,
        model_origin=models.Location(x=0, y=0, z=0),
        normalized_rotation=[0, 20, 0],
        size_options=size_option,
        geoscience_object_id=GOOSE_UUID,
        created_at=DATE,
        created_by=MODEL_USER,
        last_updated_at=DATE,
        last_updated_by=MODEL_USER,
        job_url=f"{BASE_URL}/jobs/{uuid.uuid4()}",
    )


def _mock_version(
    version_id: int, version_uuid: uuid.UUID, goose_version_id: str, bbox=None, columns: Iterable[models.Column] = ()
) -> models.Version:
    return models.Version(
        base_version_id=None if version_id == 1 else version_id - 1,
        bbox=bbox,
        bm_uuid=BM_UUID,
        comment="",
        created_at=DATE,
        created_by=MODEL_USER,
        geoscience_version_id=goose_version_id,
        mapping=models.Mapping(columns=list(columns)),
        parent_version_id=version_id - 1,
        version_id=version_id,
        version_uuid=version_uuid,
    )


FIRST_VERSION = _mock_version(1, uuid.uuid4(), "2")

UPDATE_RESULT = models.UpdateWithUrl(
    changes=models.UpdateDataLiteOutput(
        # We don't look at these values, so we can just set them to empty
        columns=models.UpdateColumnsLiteOutput(new=[], update=[], rename=[], delete=[])
    ),
    version_uuid=FIRST_VERSION.version_uuid,
    job_uuid=uuid.uuid4(),
    job_url=f"{BASE_URL}/jobs/{uuid.uuid4()}",
    upload_url=f"{BASE_URL}/upload/{uuid.uuid4()}",
)

INITIAL_DATA = pyarrow.table(
    {"i": [1, 2, 3], "j": [4, 5, 6], "k": [7, 8, 9], "col1": ["A", "B", "B"], "col2": [4.5, 5.3, 6.2]}
)
SUBBLOCKED_INITIAL_DATA = pyarrow.table(
    {
        "x": [0.5, 1.3, 1.5],
        "y": [1.5, 2.5, 2.5],
        "z": [4.5, 2.25, 2.25],
        "dx": [1.0, 0.2, 0.2],
        "dy": [1.0, 1.0, 1.0],
        "dz": [1.0, 0.5, 0.5],
        "col1": ["A", "B", "B"],
        "col2": [4.5, 5.3, 6.2],
    }
)


class CreateRequestHandler(JobPollingRequestHandler):
    def __init__(
        self,
        create_result: models.BlockModelAndJobURL,
        job_response: JobResponse,
        update_result: models.UpdateWithUrl | None = None,
        update_job_response: JobResponse | None = None,
        pending_request: int = 0,
    ) -> None:
        super().__init__(job_response, pending_request)
        self._create_result = create_result
        self._update_result = update_result
        self._update_job_response = update_job_response

    async def request(
        self,
        method: RequestMethod,
        url: str,
        headers: HTTPHeaderDict | None = None,
        post_params: list[tuple[str, str | bytes]] | None = None,
        body: object | str | bytes | None = None,
        request_timeout: int | float | tuple[int | float, int | float] | None = None,
    ) -> MockResponse:
        match method:
            case RequestMethod.POST:
                return MockResponse(status_code=201, content=self._create_result.model_dump_json())
            case RequestMethod.PATCH:
                if self._update_result is None:
                    return self.not_found()
                self._job_response = self._update_job_response
                return MockResponse(status_code=202, content=self._update_result.model_dump_json())
            case RequestMethod.GET:
                return self.job_poll()
            case _:
                return self.not_found()


class TestCreateBlockModel(TestWithConnector, TestWithStorage):
    def setUp(self) -> None:
        TestWithConnector.setUp(self)
        TestWithStorage.setUp(self)
        self.bms_client = BlockModelAPIClient(connector=self.connector, environment=self.environment, cache=self.cache)
        self.bms_client_without_cache = BlockModelAPIClient(connector=self.connector, environment=self.environment)
        self.setup_universal_headers(get_header_metadata(BlockModelAPIClient.__module__))

    @property
    def base_path(self) -> str:
        return f"blockmodel/orgs/{self.environment.org_id}/workspaces/{self.environment.workspace_id}"

    def _assert_create_request(self, definition: BaseGridDefinition) -> None:
        match definition:
            case RegularGridDefinition():
                size_option = models.SizeOptionsRegular(
                    model_type="regular",
                    n_blocks=models.Size3D(
                        nx=definition.n_blocks[0], ny=definition.n_blocks[1], nz=definition.n_blocks[2]
                    ),
                    block_size=models.BlockSize(
                        x=definition.block_size[0], y=definition.block_size[1], z=definition.block_size[2]
                    ),
                )
            case FullySubBlockedGridDefinition():
                size_option = models.SizeOptionsFullySubBlocked(
                    model_type="fully-sub-blocked",
                    n_parent_blocks=models.Size3D(
                        nx=definition.n_parent_blocks[0],
                        ny=definition.n_parent_blocks[1],
                        nz=definition.n_parent_blocks[2],
                    ),
                    n_subblocks_per_parent=models.RegularSubblocks(
                        nx=definition.n_subblocks_per_parent[0],
                        ny=definition.n_subblocks_per_parent[1],
                        nz=definition.n_subblocks_per_parent[2],
                    ),
                    parent_block_size=models.BlockSize(
                        x=definition.parent_block_size[0],
                        y=definition.parent_block_size[1],
                        z=definition.parent_block_size[2],
                    ),
                )
            case FlexibleGridDefinition():
                size_option = models.SizeOptionsFlexible(
                    model_type="flexible",
                    n_parent_blocks=models.Size3D(
                        nx=definition.n_parent_blocks[0],
                        ny=definition.n_parent_blocks[1],
                        nz=definition.n_parent_blocks[2],
                    ),
                    n_subblocks_per_parent=models.RegularSubblocks(
                        nx=definition.n_subblocks_per_parent[0],
                        ny=definition.n_subblocks_per_parent[1],
                        nz=definition.n_subblocks_per_parent[2],
                    ),
                    parent_block_size=models.BlockSize(
                        x=definition.parent_block_size[0],
                        y=definition.parent_block_size[1],
                        z=definition.parent_block_size[2],
                    ),
                )
            case OctreeGridDefinition():
                size_option = models.SizeOptionsOctree(
                    model_type="variable-octree",
                    n_parent_blocks=models.Size3D(
                        nx=definition.n_parent_blocks[0],
                        ny=definition.n_parent_blocks[1],
                        nz=definition.n_parent_blocks[2],
                    ),
                    n_subblocks_per_parent=models.OctreeSubblocks(
                        nx=definition.n_subblocks_per_parent[0],
                        ny=definition.n_subblocks_per_parent[1],
                        nz=definition.n_subblocks_per_parent[2],
                    ),
                    parent_block_size=models.BlockSize(
                        x=definition.parent_block_size[0],
                        y=definition.parent_block_size[1],
                        z=definition.parent_block_size[2],
                    ),
                )
            case _:
                raise ValueError("Unexpected grid definition type")

        self.assert_any_request_made(
            method=RequestMethod.POST,
            path=f"{self.base_path}/block-models",
            body=models.CreateData(
                name="Test BM",
                description="Test Block Model",
                size_options=size_option,
                block_rotation=[models.Rotation(axis=axis, angle=angle) for axis, angle in definition.rotations],
                model_origin=models.Location(
                    x=definition.model_origin[0],
                    y=definition.model_origin[1],
                    z=definition.model_origin[2],
                ),
                coordinate_reference_system="EPSG:4326",
                size_unit_id="m",
                object_path="test/path",
            ).model_dump(mode="json", exclude_unset=True),
            headers={
                "Authorization": "Bearer <not-a-real-token>",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )

    async def test_create_block_model(self) -> None:
        self.transport.set_request_handler(
            CreateRequestHandler(
                create_result=_mock_create_result(self.environment),
                job_response=JobResponse(
                    job_status=JobStatus.COMPLETE,
                    payload=FIRST_VERSION,
                ),
            )
        )
        bm, version = await self.bms_client.create_block_model(
            name="Test BM",
            description="Test Block Model",
            grid_definition=GRID_DEFINITION,
            object_path="test/path",
            coordinate_reference_system="EPSG:4326",
            size_unit_id="m",
        )
        self.assertEqual(bm.id, BM_UUID)
        self.assertEqual(bm.geoscience_object_id, GOOSE_UUID)
        self.assertEqual(bm.name, "Test BM")
        self.assertEqual(bm.description, "Test Block Model")
        self.assertEqual(bm.grid_definition.model_origin, [0, 0, 0])
        self.assertEqual(bm.grid_definition.rotations, [(RotationAxis.x, 20)])
        self.assertIsInstance(bm.grid_definition, RegularGridDefinition)
        self.assertEqual(bm.grid_definition.n_blocks, [10, 10, 10])
        self.assertEqual(bm.grid_definition.block_size, [1, 1, 1])
        self.assertEqual(bm.coordinate_reference_system, "EPSG:4326")
        self.assertEqual(bm.size_unit_id, "m")
        self.assertEqual(bm.bbox, BM_BBOX)
        self.assertEqual(bm.created_at, DATE)
        self.assertEqual(bm.created_by, USER)
        self.assertEqual(bm.last_updated_at, DATE)
        self.assertEqual(bm.last_updated_by, USER)

        self.assertEqual(version.bm_uuid, BM_UUID)
        self.assertEqual(version.version_id, 1)
        self.assertEqual(version.version_uuid, FIRST_VERSION.version_uuid)
        self.assertEqual(version.parent_version_id, 0)
        self.assertEqual(version.base_version_id, None)
        self.assertEqual(version.geoscience_version_id, "2")
        self.assertEqual(version.bbox, None)
        self.assertEqual(version.comment, "")
        self.assertEqual(version.created_at, DATE)
        self.assertEqual(version.created_by, USER)
        self.assertEqual(version.columns, [])

        self._assert_create_request(GRID_DEFINITION)

    async def test_create_block_model_with_data(self) -> None:
        second_version = _mock_version(
            2,
            uuid.uuid4(),
            "3",
            models.BBox(
                i_minmax=models.IntRange(min=1, max=3),
                j_minmax=models.IntRange(min=4, max=6),
                k_minmax=models.IntRange(min=7, max=9),
            ),
            columns=[
                models.Column(col_id=str(uuid.uuid4()), title="col1", data_type=models.DataType.Utf8),
                models.Column(col_id=str(uuid.uuid4()), title="col2", data_type=models.DataType.Float64),
            ],
        )

        self.transport.set_request_handler(
            CreateRequestHandler(
                create_result=_mock_create_result(self.environment),
                job_response=JobResponse(
                    job_status=JobStatus.COMPLETE,
                    payload=FIRST_VERSION,
                ),
                update_result=UPDATE_RESULT,
                update_job_response=JobResponse(
                    job_status=JobStatus.COMPLETE,
                    payload=second_version,
                ),
            )
        )
        with (
            mock.patch("evo.common.io.upload.StorageDestination") as mock_destination,
        ):
            mock_destination.upload_file = mock.AsyncMock()
            bm, version = await self.bms_client.create_block_model(
                name="Test BM",
                description="Test Block Model",
                grid_definition=GRID_DEFINITION,
                object_path="test/path",
                coordinate_reference_system="EPSG:4326",
                size_unit_id="m",
                initial_data=INITIAL_DATA,
                units={"col2": "g/t"},
            )
            mock_destination.upload_file.assert_called_once()

            # Assert that the correct columns are part of the update
            expected_update_body = models.UpdateDataLiteInput(
                columns=models.UpdateColumnsLiteInput(
                    new=[
                        models.ColumnLite(
                            title="col1",
                            data_type=models.DataType.Utf8,
                            unit_id=None,
                        ),
                        models.ColumnLite(
                            title="col2",
                            data_type=models.DataType.Float64,
                            unit_id="g/t",
                        ),
                    ],
                    update=[],
                    rename=[],
                    delete=[],
                ),
                update_type=models.UpdateType.replace,
                geometry_change=None,
            )
            self.assert_any_request_made(
                method=RequestMethod.PATCH,
                path=f"{self.base_path}/block-models/{BM_UUID}/blocks",
                body=expected_update_body.model_dump(mode="json", exclude_unset=True),
                headers={
                    "Authorization": "Bearer <not-a-real-token>",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
            )
        self.assertEqual(bm.id, BM_UUID)

        self.assertEqual(version.bm_uuid, BM_UUID)
        self.assertEqual(version.version_id, 2)
        self.assertEqual(version.version_uuid, second_version.version_uuid)
        self.assertEqual(version.parent_version_id, 1)
        self.assertEqual(version.base_version_id, 1)
        self.assertEqual(version.geoscience_version_id, "3")
        self.assertEqual(version.bbox, second_version.bbox)
        self.assertEqual(version.comment, "")
        self.assertEqual(version.created_at, DATE)
        self.assertEqual(version.created_by, USER)
        self.assertEqual(version.columns, second_version.mapping.columns)

        self._assert_create_request(GRID_DEFINITION)

    async def test_create_subblocked_model_with_data(self) -> None:
        second_version = _mock_version(
            2,
            uuid.uuid4(),
            "3",
            models.BBox(
                i_minmax=models.IntRange(min=1, max=3),
                j_minmax=models.IntRange(min=4, max=6),
                k_minmax=models.IntRange(min=7, max=9),
            ),
            columns=[
                models.Column(col_id=str(uuid.uuid4()), title="col1", data_type=models.DataType.Utf8),
                models.Column(col_id=str(uuid.uuid4()), title="col2", data_type=models.DataType.Float64),
            ],
        )

        self.transport.set_request_handler(
            CreateRequestHandler(
                create_result=_mock_create_result(
                    self.environment,
                    size_option=models.SizeOptionsFullySubBlocked(
                        model_type="fully-sub-blocked",
                        n_parent_blocks=models.Size3D(nx=10, ny=10, nz=10),
                        n_subblocks_per_parent=models.RegularSubblocks(nx=5, ny=1, nz=2),
                        parent_block_size=models.BlockSize(x=1, y=1, z=1),
                    ),
                ),
                job_response=JobResponse(
                    job_status=JobStatus.COMPLETE,
                    payload=FIRST_VERSION,
                ),
                update_result=UPDATE_RESULT,
                update_job_response=JobResponse(
                    job_status=JobStatus.COMPLETE,
                    payload=second_version,
                ),
            )
        )
        with (
            mock.patch("evo.common.io.upload.StorageDestination") as mock_destination,
        ):
            mock_destination.upload_file = mock.AsyncMock()
            bm, version = await self.bms_client.create_block_model(
                name="Test BM",
                description="Test Block Model",
                grid_definition=SUBBLOCKED_GRID_DEFINITION,
                object_path="test/path",
                coordinate_reference_system="EPSG:4326",
                size_unit_id="m",
                initial_data=SUBBLOCKED_INITIAL_DATA,
                units={"col2": "g/t"},
            )
            mock_destination.upload_file.assert_called_once()

            # Assert that the correct columns are part of the update
            expected_update_body = models.UpdateDataLiteInput(
                columns=models.UpdateColumnsLiteInput(
                    new=[
                        models.ColumnLite(
                            title="col1",
                            data_type=models.DataType.Utf8,
                            unit_id=None,
                        ),
                        models.ColumnLite(
                            title="col2",
                            data_type=models.DataType.Float64,
                            unit_id="g/t",
                        ),
                    ],
                    update=[],
                    rename=[],
                    delete=[],
                ),
                update_type=models.UpdateType.replace,
                geometry_change=True,
            )
            self.assert_any_request_made(
                method=RequestMethod.PATCH,
                path=f"{self.base_path}/block-models/{BM_UUID}/blocks",
                body=expected_update_body.model_dump(mode="json", exclude_unset=True),
                headers={
                    "Authorization": "Bearer <not-a-real-token>",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
            )
        self.assertEqual(bm.id, BM_UUID)

        self.assertEqual(version.bm_uuid, BM_UUID)
        self.assertEqual(version.version_id, 2)
        self.assertEqual(version.version_uuid, second_version.version_uuid)
        self.assertEqual(version.parent_version_id, 1)
        self.assertEqual(version.base_version_id, 1)
        self.assertEqual(version.geoscience_version_id, "3")
        self.assertEqual(version.bbox, second_version.bbox)
        self.assertEqual(version.comment, "")
        self.assertEqual(version.created_at, DATE)
        self.assertEqual(version.created_by, USER)
        self.assertEqual(version.columns, second_version.mapping.columns)

        self.assertIsInstance(bm.grid_definition, FullySubBlockedGridDefinition)
        self.assertEqual(bm.grid_definition.n_parent_blocks, [10, 10, 10])
        self.assertEqual(bm.grid_definition.parent_block_size, [1, 1, 1])
        self.assertEqual(bm.grid_definition.n_subblocks_per_parent, [5, 1, 2])

        self._assert_create_request(SUBBLOCKED_GRID_DEFINITION)

    async def test_create_block_model_no_cache(self) -> None:
        with self.assertRaises(CacheNotConfiguredException):
            await self.bms_client_without_cache.create_block_model(
                name="Test BM",
                description="Test Block Model",
                grid_definition=GRID_DEFINITION,
                object_path="test/path",
                coordinate_reference_system="EPSG:4326",
                size_unit_id="m",
                initial_data=INITIAL_DATA,
            )

    async def test_create_block_model_job_failed(self) -> None:
        self.transport.set_request_handler(
            CreateRequestHandler(
                create_result=_mock_create_result(self.environment),
                job_response=JobResponse(
                    job_status=JobStatus.COMPLETE,
                    payload=FIRST_VERSION,
                ),
                update_result=UPDATE_RESULT,
                update_job_response=JobResponse(
                    job_status=JobStatus.FAILED,
                    payload=models.JobErrorPayload(
                        detail="Update Job failed",
                        status=500,
                        title="Update Job failed",
                        type="https://seequent.com/error-codes/block-model-service/job/internal-error",
                    ),
                ),
            )
        )
        with (
            mock.patch("evo.common.io.upload.StorageDestination") as mock_destination,
        ):
            mock_destination.upload_file = mock.AsyncMock()
            with self.assertRaises(JobFailedException):
                await self.bms_client.create_block_model(
                    name="Test BM",
                    description="Test Block Model",
                    grid_definition=GRID_DEFINITION,
                    object_path="test/path",
                    coordinate_reference_system="EPSG:4326",
                    size_unit_id="m",
                    initial_data=INITIAL_DATA,
                )
            mock_destination.upload_file.assert_called_once()

    async def test_create_flexible_block_model(self) -> None:
        self.transport.set_request_handler(
            CreateRequestHandler(
                create_result=_mock_create_result(
                    self.environment,
                    size_option=models.SizeOptionsFlexible(
                        model_type="flexible",
                        n_parent_blocks=models.Size3D(nx=10, ny=10, nz=10),
                        n_subblocks_per_parent=models.RegularSubblocks(nx=5, ny=1, nz=2),
                        parent_block_size=models.BlockSize(x=1, y=1, z=1),
                    ),
                ),
                job_response=JobResponse(
                    job_status=JobStatus.COMPLETE,
                    payload=FIRST_VERSION,
                ),
            )
        )
        grid_definition = FlexibleGridDefinition(
            model_origin=[0, 0, 0],
            rotations=[],
            n_parent_blocks=[10, 10, 10],
            parent_block_size=[1, 1, 1],
            n_subblocks_per_parent=[5, 1, 2],
        )
        bm, version = await self.bms_client.create_block_model(
            name="Test BM",
            description="Test Block Model",
            grid_definition=grid_definition,
            object_path="test/path",
            coordinate_reference_system="EPSG:4326",
            size_unit_id="m",
        )
        self.assertEqual(bm.id, BM_UUID)
        self.assertIsInstance(bm.grid_definition, FlexibleGridDefinition)
        self.assertEqual(bm.grid_definition.n_parent_blocks, [10, 10, 10])
        self.assertEqual(bm.grid_definition.parent_block_size, [1, 1, 1])
        self.assertEqual(bm.grid_definition.n_subblocks_per_parent, [5, 1, 2])

        self._assert_create_request(grid_definition)

    async def test_create_octree_block_model(self) -> None:
        self.transport.set_request_handler(
            CreateRequestHandler(
                create_result=_mock_create_result(
                    self.environment,
                    size_option=models.SizeOptionsOctree(
                        model_type="variable-octree",
                        n_parent_blocks=models.Size3D(nx=10, ny=10, nz=10),
                        n_subblocks_per_parent=models.OctreeSubblocks(nx=2, ny=8, nz=4),
                        parent_block_size=models.BlockSize(x=1, y=1, z=1),
                    ),
                ),
                job_response=JobResponse(
                    job_status=JobStatus.COMPLETE,
                    payload=FIRST_VERSION,
                ),
            )
        )
        grid_definition = OctreeGridDefinition(
            model_origin=[0, 0, 0],
            rotations=[],
            n_parent_blocks=[10, 10, 10],
            parent_block_size=[1, 1, 1],
            n_subblocks_per_parent=[2, 8, 4],
        )
        bm, version = await self.bms_client.create_block_model(
            name="Test BM",
            description="Test Block Model",
            grid_definition=grid_definition,
            object_path="test/path",
            coordinate_reference_system="EPSG:4326",
            size_unit_id="m",
        )

        self.assertEqual(bm.id, BM_UUID)
        self.assertIsInstance(bm.grid_definition, OctreeGridDefinition)
        self.assertEqual(bm.grid_definition.n_parent_blocks, [10, 10, 10])
        self.assertEqual(bm.grid_definition.parent_block_size, [1, 1, 1])
        self.assertEqual(bm.grid_definition.n_subblocks_per_parent, [2, 8, 4])

        self._assert_create_request(grid_definition)
