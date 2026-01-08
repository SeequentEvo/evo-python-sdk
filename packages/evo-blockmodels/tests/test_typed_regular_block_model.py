#  Copyright © 2026 Bentley Systems, Incorporated
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import json
import uuid
from datetime import datetime
from typing import Iterable
from unittest import mock

import pandas as pd
import pyarrow

from evo.blockmodels import RegularBlockModel, RegularBlockModelData
from evo.blockmodels.endpoints import models
from evo.blockmodels.endpoints.models import JobResponse, JobStatus, RotationAxis
from evo.blockmodels.typed import Point3, Size3d, Size3i
from evo.common import ServiceUser, StaticContext
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


def _mock_create_result(environment) -> models.BlockModelAndJobURL:
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
        size_options=models.SizeOptionsRegular(
            model_type="regular",
            n_blocks=models.Size3D(nx=10, ny=10, nz=10),
            block_size=models.BlockSize(x=1, y=1, z=1),
        ),
        geoscience_object_id=GOOSE_UUID,
        created_at=DATE,
        created_by=MODEL_USER,
        last_updated_at=DATE,
        last_updated_by=MODEL_USER,
        job_url=f"{BASE_URL}/jobs/{uuid.uuid4()}",
    )


def _mock_block_model(environment) -> models.BlockModel:
    return models.BlockModel(
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
        size_options=models.SizeOptionsRegular(
            model_type="regular",
            n_blocks=models.Size3D(nx=10, ny=10, nz=10),
            block_size=models.BlockSize(x=1, y=1, z=1),
        ),
        geoscience_object_id=GOOSE_UUID,
        created_at=DATE,
        created_by=MODEL_USER,
        last_updated_at=DATE,
        last_updated_by=MODEL_USER,
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
        columns=models.UpdateColumnsLiteOutput(new=[], update=[], rename=[], delete=[])
    ),
    version_uuid=FIRST_VERSION.version_uuid,
    job_uuid=uuid.uuid4(),
    job_url=f"{BASE_URL}/jobs/{uuid.uuid4()}",
    upload_url=f"{BASE_URL}/upload/{uuid.uuid4()}",
)

SECOND_VERSION = _mock_version(
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


class CreateTypedBlockModelRequestHandler(JobPollingRequestHandler):
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
            case RequestMethod.POST if url.endswith("/block-models"):
                return MockResponse(status_code=201, content=self._create_result.model_dump_json())
            case RequestMethod.POST if url.endswith("/uploaded"):
                job_url, _ = url.rsplit("/", 1)
                return MockResponse(status_code=201, content=json.dumps({"job_url": job_url}))
            case RequestMethod.PATCH:
                if self._update_result is None:
                    return self.not_found()
                self._job_response = self._update_job_response
                return MockResponse(status_code=202, content=self._update_result.model_dump_json())
            case RequestMethod.GET:
                return self.job_poll()
            case _:
                return self.not_found()


class UpdateTypedBlockModelRequestHandler(JobPollingRequestHandler):
    def __init__(
        self,
        update_result: models.UpdateWithUrl,
        job_response: JobResponse,
        pending_request: int = 0,
    ) -> None:
        super().__init__(job_response, pending_request)
        self._update_result = update_result

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
            case RequestMethod.POST if url.endswith("/uploaded"):
                job_url, _ = url.rsplit("/", 1)
                return MockResponse(status_code=201, content=json.dumps({"job_url": job_url}))
            case RequestMethod.PATCH:
                return MockResponse(status_code=202, content=self._update_result.model_dump_json())
            case RequestMethod.GET:
                return self.job_poll()
            case _:
                return self.not_found()


class TestRegularBlockModelCreate(TestWithConnector, TestWithStorage):
    def setUp(self) -> None:
        TestWithConnector.setUp(self)
        TestWithStorage.setUp(self)
        self.setup_universal_headers(get_header_metadata("evo.blockmodels.client"))
        self._context = StaticContext.from_environment(
            environment=self.environment,
            connector=self.connector,
            cache=self.cache,
        )

    @property
    def context(self):
        return self._context

    @property
    def base_path(self) -> str:
        return f"blockmodel/orgs/{self.environment.org_id}/workspaces/{self.environment.workspace_id}"

    async def test_create_without_data(self) -> None:
        """Test creating a block model without initial data."""
        self.transport.set_request_handler(
            CreateTypedBlockModelRequestHandler(
                create_result=_mock_create_result(self.environment),
                job_response=JobResponse(
                    job_status=JobStatus.COMPLETE,
                    payload=FIRST_VERSION,
                ),
            )
        )

        data = RegularBlockModelData(
            name="Test BM",
            description="Test Block Model",
            origin=Point3(0, 0, 0),
            n_blocks=Size3i(10, 10, 10),
            block_size=Size3d(1.0, 1.0, 1.0),
            rotations=[(RotationAxis.x, 20)],
            crs="EPSG:4326",
            size_unit_id="m",
        )

        block_model = await RegularBlockModel.create(self.context, data, path="test/path")

        self.assertEqual(block_model.id, BM_UUID)
        self.assertEqual(block_model.name, "Test BM")
        self.assertEqual(block_model.description, "Test Block Model")
        self.assertEqual(block_model.origin, Point3(0, 0, 0))
        self.assertEqual(block_model.n_blocks, Size3i(10, 10, 10))
        self.assertEqual(block_model.block_size, Size3d(1.0, 1.0, 1.0))
        self.assertEqual(block_model.rotations, [(RotationAxis.x, 20)])
        self.assertTrue(block_model.cell_data.empty)

    async def test_create_with_data(self) -> None:
        """Test creating a block model with initial cell data."""
        self.transport.set_request_handler(
            CreateTypedBlockModelRequestHandler(
                create_result=_mock_create_result(self.environment),
                job_response=JobResponse(
                    job_status=JobStatus.COMPLETE,
                    payload=FIRST_VERSION,
                ),
                update_result=UPDATE_RESULT,
                update_job_response=JobResponse(
                    job_status=JobStatus.COMPLETE,
                    payload=SECOND_VERSION,
                ),
            )
        )

        cell_data = pd.DataFrame({
            "i": [1, 2, 3],
            "j": [4, 5, 6],
            "k": [7, 8, 9],
            "col1": ["A", "B", "B"],
            "col2": [4.5, 5.3, 6.2],
        })

        data = RegularBlockModelData(
            name="Test BM",
            description="Test Block Model",
            origin=Point3(0, 0, 0),
            n_blocks=Size3i(10, 10, 10),
            block_size=Size3d(1.0, 1.0, 1.0),
            rotations=[(RotationAxis.x, 20)],
            crs="EPSG:4326",
            size_unit_id="m",
            cell_data=cell_data,
            units={"col2": "g/t"},
        )

        with mock.patch("evo.common.io.upload.StorageDestination") as mock_destination:
            mock_destination.upload_file = mock.AsyncMock()
            block_model = await RegularBlockModel.create(self.context, data, path="test/path")
            mock_destination.upload_file.assert_called_once()

        self.assertEqual(block_model.id, BM_UUID)
        self.assertEqual(block_model.name, "Test BM")
        self.assertEqual(block_model.version.version_id, 2)
        self.assertEqual(len(block_model.cell_data), 3)
        self.assertIn("col1", block_model.cell_data.columns)
        self.assertIn("col2", block_model.cell_data.columns)


class TestRegularBlockModelGet(TestWithConnector, TestWithStorage):
    def setUp(self) -> None:
        TestWithConnector.setUp(self)
        TestWithStorage.setUp(self)
        self.setup_universal_headers(get_header_metadata("evo.blockmodels.client"))
        self._context = StaticContext.from_environment(
            environment=self.environment,
            connector=self.connector,
            cache=self.cache,
        )

    @property
    def context(self):
        return self._context

    async def test_get_block_model(self) -> None:
        """Test retrieving an existing block model."""
        from evo.blockmodels import BlockModelAPIClient
        from evo.blockmodels.data import BlockModel as BlockModelData, RegularGridDefinition

        test_df = pd.DataFrame({
            "x": [0.5, 1.5, 2.5],
            "y": [0.5, 1.5, 2.5],
            "z": [0.5, 1.5, 2.5],
            "col1": ["A", "B", "C"],
        })
        test_table = pyarrow.Table.from_pandas(test_df)

        with (
            mock.patch.object(BlockModelAPIClient, "get_block_model") as mock_get_bm,
            mock.patch.object(BlockModelAPIClient, "query_block_model_as_table") as mock_query,
            mock.patch.object(BlockModelAPIClient, "list_versions") as mock_list_versions,
        ):
            # Setup mock return values
            mock_metadata = BlockModelData(
                environment=self.environment,
                id=BM_UUID,
                name="Test BM",
                description="Test Block Model",
                created_at=DATE,
                created_by=USER,
                grid_definition=RegularGridDefinition(
                    model_origin=[0, 0, 0],
                    rotations=[(RotationAxis.x, 20)],
                    n_blocks=[10, 10, 10],
                    block_size=[1.0, 1.0, 1.0],
                ),
                coordinate_reference_system="EPSG:4326",
                size_unit_id="m",
                bbox=BM_BBOX,
                last_updated_at=DATE,
                last_updated_by=USER,
                geoscience_object_id=GOOSE_UUID,
            )
            mock_get_bm.return_value = mock_metadata
            mock_query.return_value = test_table
            mock_list_versions.return_value = [
                self._create_version(1, FIRST_VERSION.version_uuid),
            ]

            block_model = await RegularBlockModel.get(self.context, BM_UUID)

        self.assertEqual(block_model.id, BM_UUID)
        self.assertEqual(block_model.name, "Test BM")
        self.assertEqual(block_model.origin, Point3(0, 0, 0))
        self.assertEqual(block_model.n_blocks, Size3i(10, 10, 10))
        self.assertEqual(block_model.block_size, Size3d(1.0, 1.0, 1.0))
        self.assertEqual(len(block_model.cell_data), 3)

    def _create_version(self, version_id: int, version_uuid: uuid.UUID):
        """Helper to create a Version object for testing."""
        from evo.blockmodels.data import Version
        return Version(
            bm_uuid=BM_UUID,
            version_id=version_id,
            version_uuid=version_uuid,
            created_at=DATE,
            created_by=USER,
            comment="",
            bbox=None,
            base_version_id=None if version_id == 1 else version_id - 1,
            parent_version_id=version_id - 1,
            columns=[],
            geoscience_version_id=str(version_id + 1),
        )

    async def test_get_block_model_with_version(self) -> None:
        """Test retrieving a specific version of a block model."""
        from evo.blockmodels import BlockModelAPIClient
        from evo.blockmodels.data import BlockModel as BlockModelData, RegularGridDefinition

        version_uuid = uuid.uuid4()

        test_df = pd.DataFrame({
            "x": [0.5, 1.5, 2.5],
            "y": [0.5, 1.5, 2.5],
            "z": [0.5, 1.5, 2.5],
            "col1": ["A", "B", "C"],
        })
        test_table = pyarrow.Table.from_pandas(test_df)

        with (
            mock.patch.object(BlockModelAPIClient, "get_block_model") as mock_get_bm,
            mock.patch.object(BlockModelAPIClient, "query_block_model_as_table") as mock_query,
            mock.patch.object(BlockModelAPIClient, "list_versions") as mock_list_versions,
        ):
            mock_metadata = BlockModelData(
                environment=self.environment,
                id=BM_UUID,
                name="Test BM",
                description="Test Block Model",
                created_at=DATE,
                created_by=USER,
                grid_definition=RegularGridDefinition(
                    model_origin=[0, 0, 0],
                    rotations=[(RotationAxis.x, 20)],
                    n_blocks=[10, 10, 10],
                    block_size=[1.0, 1.0, 1.0],
                ),
                coordinate_reference_system="EPSG:4326",
                size_unit_id="m",
                bbox=BM_BBOX,
                last_updated_at=DATE,
                last_updated_by=USER,
                geoscience_object_id=GOOSE_UUID,
            )
            mock_get_bm.return_value = mock_metadata
            mock_query.return_value = test_table
            mock_list_versions.return_value = [
                self._create_version(2, version_uuid),
                self._create_version(1, FIRST_VERSION.version_uuid),
            ]

            block_model = await RegularBlockModel.get(self.context, BM_UUID, version_id=version_uuid)

        self.assertEqual(block_model.id, BM_UUID)
        self.assertEqual(block_model.version.version_uuid, version_uuid)


class TestRegularBlockModelUpdateAttributes(TestWithConnector, TestWithStorage):
    def setUp(self) -> None:
        TestWithConnector.setUp(self)
        TestWithStorage.setUp(self)
        self.setup_universal_headers(get_header_metadata("evo.blockmodels.client"))
        self._context = StaticContext.from_environment(
            environment=self.environment,
            connector=self.connector,
            cache=self.cache,
        )

    @property
    def context(self):
        return self._context

    async def test_update_attributes(self) -> None:
        """Test updating attributes on an existing block model."""
        self.transport.set_request_handler(
            UpdateTypedBlockModelRequestHandler(
                update_result=UPDATE_RESULT,
                job_response=JobResponse(
                    job_status=JobStatus.COMPLETE,
                    payload=SECOND_VERSION,
                ),
            )
        )

        # Create a mock RegularBlockModel instance
        from evo.blockmodels import BlockModelAPIClient
        from evo.blockmodels.data import BlockModel as BlockModelData, RegularGridDefinition, Version

        client = BlockModelAPIClient.from_context(self.context)
        metadata = BlockModelData(
            environment=self.environment,
            id=BM_UUID,
            name="Test BM",
            description="Test Block Model",
            created_at=DATE,
            created_by=USER,
            grid_definition=RegularGridDefinition(
                model_origin=[0, 0, 0],
                rotations=[(RotationAxis.x, 20)],
                n_blocks=[10, 10, 10],
                block_size=[1.0, 1.0, 1.0],
            ),
            coordinate_reference_system="EPSG:4326",
            size_unit_id="m",
            bbox=BM_BBOX,
            last_updated_at=DATE,
            last_updated_by=USER,
            geoscience_object_id=GOOSE_UUID,
        )
        version = Version(
            bm_uuid=BM_UUID,
            version_id=1,
            version_uuid=FIRST_VERSION.version_uuid,
            created_at=DATE,
            created_by=USER,
            comment="",
            bbox=None,
            base_version_id=None,
            parent_version_id=0,
            columns=[],
            geoscience_version_id="2",
        )
        cell_data = pd.DataFrame({
            "i": [1, 2, 3],
            "j": [4, 5, 6],
            "k": [7, 8, 9],
        })

        block_model = RegularBlockModel(
            client=client,
            metadata=metadata,
            version=version,
            cell_data=cell_data,
        )

        # Update with new columns
        new_data = pd.DataFrame({
            "i": [1, 2, 3],
            "j": [4, 5, 6],
            "k": [7, 8, 9],
            "col1": ["A", "B", "B"],
            "col2": [4.5, 5.3, 6.2],
        })

        with mock.patch("evo.common.io.upload.StorageDestination") as mock_destination:
            mock_destination.upload_file = mock.AsyncMock()
            new_version = await block_model.update_attributes(
                new_data,
                new_columns=["col1", "col2"],
                units={"col2": "g/t"},
            )
            mock_destination.upload_file.assert_called_once()

        self.assertEqual(new_version.version_id, 2)
        self.assertEqual(block_model.version.version_id, 2)
        self.assertIn("col1", block_model.cell_data.columns)
        self.assertIn("col2", block_model.cell_data.columns)


class TestTypedTypes(TestWithConnector):
    """Test the typed type classes."""

    def test_point3(self) -> None:
        """Test Point3 named tuple."""
        p = Point3(1.0, 2.0, 3.0)
        self.assertEqual(p.x, 1.0)
        self.assertEqual(p.y, 2.0)
        self.assertEqual(p.z, 3.0)

    def test_size3i(self) -> None:
        """Test Size3i named tuple."""
        s = Size3i(10, 20, 30)
        self.assertEqual(s.nx, 10)
        self.assertEqual(s.ny, 20)
        self.assertEqual(s.nz, 30)
        self.assertEqual(s.total_size, 6000)

    def test_size3d(self) -> None:
        """Test Size3d named tuple."""
        s = Size3d(1.5, 2.5, 3.5)
        self.assertEqual(s.dx, 1.5)
        self.assertEqual(s.dy, 2.5)
        self.assertEqual(s.dz, 3.5)

    def test_bounding_box_from_origin_and_size(self) -> None:
        """Test BoundingBox.from_origin_and_size class method."""
        from evo.blockmodels.typed import BoundingBox

        bbox = BoundingBox.from_origin_and_size(
            origin=Point3(0, 0, 0),
            size=Size3i(10, 20, 30),
            cell_size=Size3d(1.0, 2.0, 3.0),
        )
        self.assertEqual(bbox.x_min, 0)
        self.assertEqual(bbox.x_max, 10)
        self.assertEqual(bbox.y_min, 0)
        self.assertEqual(bbox.y_max, 40)
        self.assertEqual(bbox.z_min, 0)
        self.assertEqual(bbox.z_max, 90)

