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
from unittest import mock

import pyarrow

from evo.blockmodels import BlockModelAPIClient
from evo.blockmodels.endpoints.models import (
    BBox,
    IntRange,
    JobErrorPayload,
    JobResponse,
    JobStatus,
    Mapping,
    QueryDownload,
    QueryResult,
)
from evo.blockmodels.exceptions import CacheNotConfiguredException, JobFailedException
from evo.common import HealthCheckType
from evo.common.data import HTTPHeaderDict, RequestMethod
from evo.common.test_tools import BASE_URL, AbstractTestRequestHandler, MockResponse, TestWithConnector, TestWithStorage
from evo.common.utils import NoFeedback

TABLE = pyarrow.table({"x": [1.0, 2.0], "y": [3.0, 4.0], "z": [5.0, 6.0], "col1": [1, 2], "col2": [3, 4]})
BBOX = BBox(
    i_minmax=IntRange(min=1, max=2),
    j_minmax=IntRange(min=3, max=4),
    k_minmax=IntRange(min=5, max=6),
)


class QueryRequestHandler(AbstractTestRequestHandler):
    def __init__(self, query_result: QueryResult, job_response: JobResponse, pending_request: int = 0) -> None:
        self._query_result = query_result
        self._job_response = job_response
        self._pending_requests = pending_request

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
                return MockResponse(status_code=200, content=self._query_result.model_dump_json())
            case RequestMethod.GET:
                if self._pending_requests > 0:
                    self._pending_requests -= 1
                    job_response = JobResponse(job_status=JobStatus.PROCESSING)
                else:
                    job_response = self._job_response
                return MockResponse(status_code=200, content=job_response.model_dump_json())
            case _:
                return self.not_found()


class TestBlockModelAPIClient(TestWithConnector, TestWithStorage):
    def setUp(self) -> None:
        TestWithConnector.setUp(self)
        TestWithStorage.setUp(self)
        self.bms_client = BlockModelAPIClient(connector=self.connector, environment=self.environment, cache=self.cache)
        self.bms_client_without_cache = BlockModelAPIClient(connector=self.connector, environment=self.environment)

    @property
    def base_path(self) -> str:
        return f"blockmodel/orgs/{self.environment.org_id}/workspaces/{self.environment.workspace_id}"

    async def test_check_service_health(self) -> None:
        """Test service health check implementation"""
        with mock.patch("evo.blockmodels.client.get_service_health", spec_set=True) as mock_get_service_health:
            await self.bms_client.get_service_health()
        mock_get_service_health.assert_called_once_with(self.connector, "blockmodel", check_type=HealthCheckType.FULL)

    async def test_query_block_model(self) -> None:
        bm_uuid = uuid.uuid4()
        version_uuid = uuid.uuid4()
        column_uuid = uuid.uuid4()
        job_uuid = uuid.uuid4()

        download_url = "http://data.com/"

        query_result = QueryResult(
            bm_uuid=bm_uuid,
            version_id=1,
            version_uuid=version_uuid,
            bbox=BBOX,
            mapping=Mapping(columns=[]),
            columns=["col1", str(column_uuid)],
            job_url=BASE_URL + self.base_path + f"blockmodel/{bm_uuid}/jobs/{job_uuid}",
        )
        job_response = JobResponse(
            job_status=JobStatus.COMPLETE,
            payload=QueryDownload(download_url=download_url),
        )
        self.transport.set_request_handler(QueryRequestHandler(query_result, job_response, pending_request=1))

        expected_filename = self.cache.get_location(environment=self.environment, scope="blockmodel") / str(job_uuid)
        with (
            mock.patch("pyarrow.parquet.read_table") as read_table,
            mock.patch("evo.common.io.download.HTTPSource", autospec=True) as mock_source,
        ):

            async def _mock_download_file_side_effect(*args, **kwargs):
                actual_download_url = await kwargs["url_generator"]()
                self.assertEqual(expected_filename, kwargs["filename"])
                self.assertEqual(download_url, actual_download_url)
                self.assertIs(self.transport, kwargs["transport"])
                self.assertIs(NoFeedback, kwargs["fb"])

            mock_source.download_file.side_effect = _mock_download_file_side_effect

            read_table.return_value = TABLE

            result = await self.bms_client.query_block_model_as_table(
                bm_id=bm_uuid,
                columns=["col1", column_uuid],
                bbox=BBOX,
                version_uuid=version_uuid,
            )
        read_table.assert_called_once_with(expected_filename)
        self.assertEqual(TABLE, result)

        # 1 query, 1 pending job status, and 1 complete job status
        self.assertEqual(self.transport.request.call_count, 3)

    async def test_query_block_model_job_failed(self) -> None:
        bm_uuid = uuid.uuid4()
        version_uuid = uuid.uuid4()
        column_uuid = uuid.uuid4()
        job_uuid = uuid.uuid4()

        query_result = QueryResult(
            bm_uuid=bm_uuid,
            version_id=1,
            version_uuid=version_uuid,
            bbox=BBOX,
            mapping=Mapping(columns=[]),
            columns=["col1", str(column_uuid)],
            job_url=BASE_URL + self.base_path + f"blockmodel/{bm_uuid}/jobs/{job_uuid}",
        )
        job_response = JobResponse(
            job_status=JobStatus.FAILED,
            payload=JobErrorPayload(
                detail="Query Job failed",
                status=500,
                title="Query Job failed",
                type="https://seequent.com/error-codes/block-model-service/job/internal-error",
            ),
        )
        self.transport.set_request_handler(QueryRequestHandler(query_result, job_response, pending_request=1))

        with self.assertRaises(JobFailedException):
            await self.bms_client.query_block_model_as_table(
                bm_id=bm_uuid,
                columns=["col1", column_uuid],
                bbox=BBOX,
                version_uuid=version_uuid,
            )

        # 1 query, 1 pending job status, and 1 failed job status
        self.assertEqual(self.transport.request.call_count, 3)

    async def test_query_block_model_no_cache(self) -> None:
        bm_uuid = uuid.uuid4()
        version_uuid = uuid.uuid4()
        column_uuid = uuid.uuid4()

        with self.assertRaises(CacheNotConfiguredException):
            await self.bms_client_without_cache.query_block_model_as_table(
                bm_id=bm_uuid,
                columns=["col1", column_uuid],
                bbox=BBOX,
                version_uuid=version_uuid,
            )

        self.transport.assert_no_requests()
