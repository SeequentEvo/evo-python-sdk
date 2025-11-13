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

import json
import uuid
from datetime import datetime
from typing import Iterable
from unittest import mock

import pyarrow

from evo.blockmodels import BlockModelAPIClient
from evo.blockmodels.endpoints import models
from evo.blockmodels.endpoints.models import JobResponse, JobStatus
from evo.blockmodels.exceptions import CacheNotConfiguredException, JobFailedException, MissingColumnInTable
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


UPDATE_RESULT = models.UpdateWithUrl(
    changes=models.UpdateDataLiteOutput(
        # We don't look at these values, so we can just set them to empty
        columns=models.UpdateColumnsLiteOutput(new=[], update=[], rename=[], delete=[])
    ),
    version_uuid=uuid.uuid4(),
    job_uuid=uuid.uuid4(),
    job_url=f"{BASE_URL}/jobs/{uuid.uuid4()}",
    upload_url=f"{BASE_URL}/upload/{uuid.uuid4()}",
)

NEW_DATA = pyarrow.table(
    {"i": [1, 2, 3], "j": [4, 5, 6], "k": [7, 8, 9], "col1": ["A", "B", "B"], "col2": [4.5, 5.3, 6.2]}
)

UPDATED_VERSION = _mock_version(
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


class UpdateRequestHandler(JobPollingRequestHandler):
    def __init__(
        self,
        update_result: models.UpdateWithUrl | None = None,
        job_response: JobResponse | None = None,
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


class TestUpdateBlockModel(TestWithConnector, TestWithStorage):
    def setUp(self) -> None:
        TestWithConnector.setUp(self)
        TestWithStorage.setUp(self)
        self.bms_client = BlockModelAPIClient(connector=self.connector, environment=self.environment, cache=self.cache)
        self.bms_client_without_cache = BlockModelAPIClient(connector=self.connector, environment=self.environment)
        self.setup_universal_headers(get_header_metadata(BlockModelAPIClient.__module__))

    @property
    def base_path(self) -> str:
        return f"blockmodel/orgs/{self.environment.org_id}/workspaces/{self.environment.workspace_id}"

    async def test_add_new_columns(self) -> None:
        self.transport.set_request_handler(
            UpdateRequestHandler(
                update_result=UPDATE_RESULT,
                job_response=JobResponse(
                    job_status=JobStatus.COMPLETE,
                    payload=UPDATED_VERSION,
                ),
            )
        )
        with (
            mock.patch("evo.common.io.upload.StorageDestination") as mock_destination,
        ):
            mock_destination.upload_file = mock.AsyncMock()
            version = await self.bms_client.add_new_columns(
                BM_UUID,
                NEW_DATA,
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
        self.assertEqual(version.bm_uuid, BM_UUID)
        self.assertEqual(version.version_id, 2)
        self.assertEqual(version.version_uuid, UPDATED_VERSION.version_uuid)
        self.assertEqual(version.parent_version_id, 1)
        self.assertEqual(version.base_version_id, 1)
        self.assertEqual(version.geoscience_version_id, "3")
        self.assertEqual(version.bbox, UPDATED_VERSION.bbox)
        self.assertEqual(version.comment, "")
        self.assertEqual(version.created_at, DATE)
        self.assertEqual(version.created_by, USER)
        self.assertEqual(version.columns, UPDATED_VERSION.mapping.columns)

    async def test_add_new_columns_no_cache(self) -> None:
        with self.assertRaises(CacheNotConfiguredException):
            await self.bms_client_without_cache.add_new_columns(
                BM_UUID,
                NEW_DATA,
            )

    async def test_add_new_columns_job_failed(self) -> None:
        self.transport.set_request_handler(
            UpdateRequestHandler(
                update_result=UPDATE_RESULT,
                job_response=JobResponse(
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
                await self.bms_client.add_new_columns(
                    BM_UUID,
                    NEW_DATA,
                )
            mock_destination.upload_file.assert_called_once()

    async def test_update_columns(self) -> None:
        self.transport.set_request_handler(
            UpdateRequestHandler(
                update_result=UPDATE_RESULT,
                job_response=JobResponse(
                    job_status=JobStatus.COMPLETE,
                    payload=UPDATED_VERSION,
                ),
            )
        )
        with (
            mock.patch("evo.common.io.upload.StorageDestination") as mock_destination,
        ):
            mock_destination.upload_file = mock.AsyncMock()
            version = await self.bms_client.update_block_model_columns(
                BM_UUID,
                NEW_DATA,
                new_columns=["col2"],
                update_columns={"col1"},
                delete_columns={"col3"},
                units={"col2": "g/t"},
            )
            mock_destination.upload_file.assert_called_once()

            # Assert that the correct columns are part of the update
            expected_update_body = models.UpdateDataLiteInput(
                columns=models.UpdateColumnsLiteInput(
                    new=[
                        models.ColumnLite(
                            title="col2",
                            data_type=models.DataType.Float64,
                            unit_id="g/t",
                        ),
                    ],
                    update=["col1"],
                    rename=[],
                    delete=["col3"],
                ),
                update_type=models.UpdateType.replace,
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
        self.assertEqual(version.bm_uuid, BM_UUID)
        self.assertEqual(version.version_id, 2)
        self.assertEqual(version.version_uuid, UPDATED_VERSION.version_uuid)
        self.assertEqual(version.parent_version_id, 1)
        self.assertEqual(version.base_version_id, 1)
        self.assertEqual(version.geoscience_version_id, "3")
        self.assertEqual(version.bbox, UPDATED_VERSION.bbox)
        self.assertEqual(version.comment, "")
        self.assertEqual(version.created_at, DATE)
        self.assertEqual(version.created_by, USER)
        self.assertEqual(version.columns, UPDATED_VERSION.mapping.columns)

    async def test_update_columns_no_cache(self) -> None:
        with self.assertRaises(CacheNotConfiguredException):
            await self.bms_client_without_cache.update_block_model_columns(
                BM_UUID,
                NEW_DATA,
                new_columns=["col2"],
            )

    async def test_update_columns_missing_in_data(self) -> None:
        with self.assertRaises(MissingColumnInTable):
            await self.bms_client.update_block_model_columns(
                BM_UUID,
                NEW_DATA,
                new_columns=["non_existent_column"],
            )

    async def test_update_columns_job_failed(self) -> None:
        self.transport.set_request_handler(
            UpdateRequestHandler(
                update_result=UPDATE_RESULT,
                job_response=JobResponse(
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
                await self.bms_client.update_block_model_columns(
                    BM_UUID,
                    NEW_DATA,
                    new_columns=["col2"],
                )
            mock_destination.upload_file.assert_called_once()
