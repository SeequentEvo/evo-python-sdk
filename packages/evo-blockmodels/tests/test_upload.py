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
from unittest import mock

from evo.blockmodels import BlockModelAPIClient
from evo.blockmodels.endpoints import models
from evo.blockmodels.endpoints.models import (
    JobErrorPayload,
    JobResponse,
    JobStatus,
)
from evo.blockmodels.exceptions import JobFailedException
from evo.common.data import HTTPHeaderDict, RequestMethod
from evo.common.test_tools import MockResponse, TestWithConnector, TestWithStorage
from utils import JobPollingRequestHandler

DATE = datetime(2021, 1, 1)
MODEL_USER = models.UserInfo(email="test@test.com", name="Test User", id=uuid.uuid4())

MOCK_VERSION = models.Version(
    base_version_id=None,
    bbox=None,
    bm_uuid=uuid.uuid4(),
    comment="uploaded",
    created_at=DATE,
    created_by=MODEL_USER,
    geoscience_version_id="1",
    mapping=models.Mapping(columns=[]),
    parent_version_id=0,
    version_id=1,
    version_uuid=uuid.uuid4(),
)


class UploadNotifyRequestHandler(JobPollingRequestHandler):
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
            case RequestMethod.GET:
                return self.job_poll()
            case _:
                return self.not_found()


class TestUploadBlockModel(TestWithConnector, TestWithStorage):
    def setUp(self) -> None:
        TestWithConnector.setUp(self)
        TestWithStorage.setUp(self)
        self.bms_client = BlockModelAPIClient(connector=self.connector, environment=self.environment, cache=self.cache)

    async def test_upload_block_model(self) -> None:
        bm_uuid = uuid.uuid4()
        job_uuid = uuid.uuid4()
        upload_url = "http://upload.example.com/data"

        self.transport.set_request_handler(
            UploadNotifyRequestHandler(
                job_response=JobResponse(
                    job_status=JobStatus.COMPLETE,
                    payload=MOCK_VERSION,
                ),
            )
        )
        with mock.patch("evo.common.io.upload.StorageDestination") as mock_destination:
            mock_destination.upload_file = mock.AsyncMock()
            version = await self.bms_client.upload_block_model(
                bm_id=bm_uuid,
                job_uuid=job_uuid,
                upload_url=upload_url,
                filename="test_data.parquet",
            )
            mock_destination.upload_file.assert_called_once()

        self.assertEqual(version.bm_uuid, MOCK_VERSION.bm_uuid)
        self.assertEqual(version.version_id, MOCK_VERSION.version_id)
        self.assertEqual(version.version_uuid, MOCK_VERSION.version_uuid)
        self.assertEqual(version.comment, "uploaded")

    async def test_upload_block_model_job_failed(self) -> None:
        bm_uuid = uuid.uuid4()
        job_uuid = uuid.uuid4()
        upload_url = "http://upload.example.com/data"

        self.transport.set_request_handler(
            UploadNotifyRequestHandler(
                job_response=JobResponse(
                    job_status=JobStatus.FAILED,
                    payload=JobErrorPayload(
                        detail="Upload Job failed",
                        status=500,
                        title="Upload Job failed",
                        type="https://seequent.com/error-codes/block-model-service/job/internal-error",
                    ),
                ),
            )
        )
        with mock.patch("evo.common.io.upload.StorageDestination") as mock_destination:
            mock_destination.upload_file = mock.AsyncMock()
            with self.assertRaises(JobFailedException):
                await self.bms_client.upload_block_model(
                    bm_id=bm_uuid,
                    job_uuid=job_uuid,
                    upload_url=upload_url,
                    filename="test_data.parquet",
                )
            mock_destination.upload_file.assert_called_once()

    async def test_upload_block_model_with_pending_job(self) -> None:
        bm_uuid = uuid.uuid4()
        job_uuid = uuid.uuid4()
        upload_url = "http://upload.example.com/data"

        self.transport.set_request_handler(
            UploadNotifyRequestHandler(
                job_response=JobResponse(
                    job_status=JobStatus.COMPLETE,
                    payload=MOCK_VERSION,
                ),
                pending_request=2,
            )
        )
        with mock.patch("evo.common.io.upload.StorageDestination") as mock_destination:
            mock_destination.upload_file = mock.AsyncMock()
            version = await self.bms_client.upload_block_model(
                bm_id=bm_uuid,
                job_uuid=job_uuid,
                upload_url=upload_url,
                filename="test_data.parquet",
            )
            mock_destination.upload_file.assert_called_once()

        self.assertEqual(version.bm_uuid, MOCK_VERSION.bm_uuid)
        self.assertEqual(version.version_id, MOCK_VERSION.version_id)
        # 1 notify upload, 2 pending polls, 1 complete poll
        self.assertEqual(self.transport.request.call_count, 4)
