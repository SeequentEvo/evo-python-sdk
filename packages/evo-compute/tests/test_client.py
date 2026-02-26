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
import textwrap
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from uuid import UUID

from evo.common import RequestMethod
from evo.common.exceptions import UnknownResponseError
from evo.common.test_tools import ORG as TEST_ORG
from evo.common.test_tools import MockResponse, TestWithConnector
from evo.common.utils import get_header_metadata
from parameterized import parameterized
from pydantic import BaseModel

from data import load_test_data
from evo.compute import JobClient, JobProgress, JobStatusEnum
from evo.compute.exceptions import JobError, JobPendingError

TEST_TOPIC = "test"
TEST_TASK = "job-client"
TEST_JOB_ID = UUID(int=1234)


class PydanticResult(BaseModel):
    foo: str
    baz: int


@dataclass(frozen=True, kw_only=True)
class DataclassResult:
    foo: str
    baz: int


class TestJobClient(TestWithConnector):
    def setUp(self) -> None:
        super().setUp()
        self.job = JobClient(
            connector=self.connector,
            org_id=TEST_ORG.id,
            topic=TEST_TOPIC,
            task=TEST_TASK,
            job_id=TEST_JOB_ID,
        )
        self.setup_universal_headers(get_header_metadata(JobClient.__module__))

    @property
    def task_path(self) -> str:
        return f"/compute/orgs/{TEST_ORG.id}/{self.job.topic}/{self.job.task}"

    @property
    def job_url(self) -> str:
        return self.connector.base_url.rstrip("/") + self.task_path + f"/{self.job.id}/status"

    def test_id(self) -> None:
        """Test that the job ID is exposed."""
        self.assertEqual(TEST_JOB_ID, self.job.id)

    def test_topic(self) -> None:
        """Test that the topic is exposed."""
        self.assertEqual(TEST_TOPIC, self.job.topic)

    def test_task(self) -> None:
        """Test that the task is exposed."""
        self.assertEqual(TEST_TASK, self.job.task)

    def test_url(self) -> None:
        """Test that the job URL is correctly constructed."""
        self.assertEqual(self.job_url, self.job.url)

    def test_repr(self) -> None:
        """Test that the job is represented as expected."""
        self.assertEqual(self.job_url, repr(self.job))

    def assert_jobs_equal(self, job1: JobClient, job2: JobClient) -> None:
        """Assert that two jobs are equal, but not the same object."""
        self.assertIsNot(job1, job2)
        self.assertEqual(job1.id, job2.id)
        self.assertEqual(job1.topic, job2.topic)
        self.assertEqual(job1.task, job2.task)
        self.assertEqual(job1.url, job2.url)

    def test_from_url(self) -> None:
        """Test that a job can be constructed from a URL."""
        job = JobClient.from_url(self.connector, self.job_url)
        self.assert_jobs_equal(self.job, job)

    async def test_submit(self) -> None:
        """Test that a job can be submitted."""
        with self.transport.set_http_response(status_code=303, headers={"Location": self.job_url}):
            job = await JobClient.submit(
                connector=self.connector,
                org_id=TEST_ORG.id,
                topic=TEST_TOPIC,
                task=TEST_TASK,
                parameters={"foo": "bar"},
            )
        self.assert_request_made(
            RequestMethod.POST,
            self.task_path,
            headers={"Content-Type": "application/json"},
            body={"parameters": {"foo": "bar"}},
        )
        self.assert_jobs_equal(self.job, job)

    @contextmanager
    def set_status_response(self, http_status: int, data_file: str) -> Iterator[JobProgress]:
        """Set the response for a status request and yield the expected status."""
        response_data = load_test_data(data_file)
        response_data.pop("results", None)

        if response_data.get("error") is not None:
            error = JobError(
                status=response_data["error"].get("status"),
                reason=None,
                content=response_data["error"],
                headers=None,
            )
        else:
            error = None
        expected_status = JobProgress(
            status=JobStatusEnum(response_data["status"]),
            progress=response_data.get("progress"),
            message=response_data.get("message"),
            error=error,
        )

        response_json = json.dumps(response_data)
        with self.transport.set_http_response(
            status_code=http_status,
            content=response_json,
            headers={"Content-Type": "application/json"},
        ):
            yield expected_status

            self.assert_request_made(
                RequestMethod.GET,
                self.task_path + f"/{self.job.id}/status",
                headers={"Accept": "application/json"},
            )

    @parameterized.expand(
        [
            ("requested", 202, "job-response-requested.json"),
            ("in_progress", 202, "job-response-in-progress.json"),
            ("cancelling", 202, "job-response-cancelling.json"),
            ("cancelled", 200, "job-response-cancelled.json"),
            ("failed", 200, "job-response-failed.json"),
            ("succeeded", 200, "job-response-succeeded.json"),
        ]
    )
    async def test_get_status(self, _label: str, http_status: int, data_file: str) -> None:
        """Test retrieving the status of a job."""
        with self.set_status_response(http_status, data_file) as expected_status:
            reported_status = await self.job.get_status()
        self.assertEqual(expected_status.status, reported_status.status)
        self.assertEqual(expected_status.progress, reported_status.progress)
        self.assertEqual(expected_status.message, reported_status.message)
        # JobError does not implement __eq__, so compare their string representations.
        self.assertEqual(str(expected_status.error), str(reported_status.error))

    def test_job_progress_failed_str(self) -> None:
        """Test the string representation of a failed JobProgress."""
        progress = JobProgress(
            status=JobStatusEnum.failed,
            progress=None,
            message="Job failed due to error",
            error=JobError(
                status=422,
                reason=None,
                content={
                    "title": "Unprocessable Entity",
                    "type": "https://example.com/errors/422",
                    "detail": "Invalid parameters",
                },
                headers=None,
            ),
        )
        expected_str = """\
        [failed] > Job failed due to error
        Error: (422)
        Type: https://example.com/errors/422
        Title: Unprocessable Entity
        Detail: Invalid parameters"""
        self.assertEqual(textwrap.dedent(expected_str), str(progress))

    @contextmanager
    def set_result_response(self, http_status: int, data_file: str) -> Iterator[dict | None]:
        """Set the response for a result request and yield the expected results."""
        response_data = load_test_data(data_file)
        response_json = json.dumps(response_data)
        with self.transport.set_http_response(
            status_code=http_status,
            content=response_json,
            headers={"Content-Type": "application/json"},
        ):
            yield response_data.get("results")
            self.assert_request_made(
                RequestMethod.GET,
                self.task_path + f"/{self.job.id}",
                headers={"Accept": "application/json"},
            )

    @parameterized.expand(
        [
            ("job requested", 202, "job-response-requested.json", JobPendingError),
            ("job in progress", 202, "job-response-in-progress.json", JobPendingError),
            ("job cancelling", 202, "job-response-cancelling.json", JobPendingError),
            ("job failed", 200, "job-response-failed.json", JobError),
            # It is unclear what the actual expected response is for a cancelled job.
            # For now we assume there are no results and no error.
            ("job cancelled", 200, "job-response-cancelled.json", UnknownResponseError),
        ]
    )
    async def test_get_results_errors(
        self, _label: str, http_status: int, data_file: str, exc_type: type[Exception]
    ) -> None:
        """Test that errors are raised when requesting results for a job in an invalid state."""
        with self.set_result_response(http_status, data_file):
            with self.assertRaises(exc_type):
                await self.job.get_results()

    async def test_get_results(self) -> None:
        """Test that results can be retrieved for a job."""
        with self.set_result_response(200, "job-response-succeeded.json") as expected_results:
            results = await self.job.get_results()
        self.assertEqual(expected_results, results)

    async def test_get_results_twice(self) -> None:
        with self.set_result_response(200, "job-response-succeeded.json") as expected_results:
            first_results = await self.job.get_results()
        self.assertEqual(expected_results, first_results)

        self.transport.reset_mock()
        with self.transport.set_http_response(status_code=500):  # Should not be called.
            second_results = await self.job.get_results()
        self.transport.assert_no_requests()
        self.assertEqual(first_results, second_results)
        self.assertIsNot(first_results, second_results)  # Ensure a new object is returned.

    async def test_get_results_as_pydantic(self) -> None:
        job = JobClient.from_url(self.connector, self.job_url, result_type=PydanticResult)

        with self.set_result_response(200, "job-response-succeeded.json") as expected_results:
            results = await job.get_results()

        expected_results = PydanticResult.model_validate(expected_results)
        self.assertIsInstance(results, PydanticResult)
        self.assertEqual(expected_results, results)

    async def test_get_results_as_dataclass(self) -> None:
        job = JobClient.from_url(self.connector, self.job_url, result_type=DataclassResult)

        with self.set_result_response(200, "job-response-succeeded.json") as expected_results:
            results = await job.get_results()

        expected_results = DataclassResult(**expected_results)
        self.assertIsInstance(results, DataclassResult)
        self.assertEqual(expected_results, results)

    async def test_cancel(self) -> None:
        """Test that a job can be cancelled."""
        with self.transport.set_http_response(status_code=204):
            await self.job.cancel()
        self.assert_request_made(RequestMethod.DELETE, self.task_path + f"/{self.job.id}")

    @contextmanager
    def set_job_states(self, *data_files: str) -> Iterator[dict | None]:
        """Set the responses for a sequence of job status requests and yield the final results."""
        all_responses = [
            MockResponse(
                status_code=202,
                headers={"Content-Type": "application/json"},
                content=json.dumps(load_test_data(data_file)),
            )
            for data_file in data_files
        ]
        final_response = load_test_data(data_files[-1])
        all_responses.append(
            MockResponse(
                status_code=200,
                headers={"Content-Type": "application/json"},
                content=json.dumps(final_response),
            )
        )
        old_side_effect = self.transport.request.side_effect
        self.transport.request.side_effect = all_responses
        try:
            yield final_response.get("results")
        finally:
            self.transport.request.side_effect = old_side_effect

    async def test_wait_for_result(self) -> None:
        """Test that a job can be waited on for results."""
        with self.set_job_states(
            "job-response-requested.json",
            "job-response-in-progress.json",
            "job-response-succeeded.json",
        ) as expected_results:
            results = await self.job.wait_for_results(polling_interval_seconds=0.0)

        self.assert_any_request_made(
            RequestMethod.GET,
            self.task_path + f"/{self.job.id}/status",
            headers={"Accept": "application/json"},
        )
        self.assert_any_request_made(
            RequestMethod.GET,
            self.task_path + f"/{self.job.id}",
            headers={"Accept": "application/json"},
        )

        self.assertEqual(expected_results, results)

    async def test_wait_for_result_failed(self) -> None:
        """Test waiting for a job that fails."""
        with self.set_job_states(
            "job-response-requested.json",
            "job-response-in-progress.json",
            "job-response-failed.json",
        ):
            with self.assertRaises(JobError):
                await self.job.wait_for_results(polling_interval_seconds=0.0)

        self.assert_any_request_made(
            RequestMethod.GET,
            self.task_path + f"/{self.job.id}/status",
            headers={"Accept": "application/json"},
        )
        self.assert_any_request_made(
            RequestMethod.GET,
            self.task_path + f"/{self.job.id}",
            headers={"Accept": "application/json"},
        )

    async def test_wait_for_result_cancelled(self) -> None:
        """Test waiting for a job that gets cancelled."""
        with self.set_job_states(
            "job-response-requested.json",
            "job-response-in-progress.json",
            "job-response-cancelling.json",
            "job-response-cancelled.json",
        ):
            with self.assertRaises(UnknownResponseError):
                await self.job.wait_for_results(polling_interval_seconds=0.0)

        self.assert_any_request_made(
            RequestMethod.GET,
            self.task_path + f"/{self.job.id}/status",
            headers={"Accept": "application/json"},
        )
        self.assert_any_request_made(
            RequestMethod.GET,
            self.task_path + f"/{self.job.id}",
            headers={"Accept": "application/json"},
        )
