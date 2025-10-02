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

from evo.blockmodels.endpoints.models import JobResponse, JobStatus
from evo.common.test_tools import AbstractTestRequestHandler, MockResponse


class JobPollingRequestHandler(AbstractTestRequestHandler):
    def __init__(self, job_response: JobResponse, pending_request: int = 0) -> None:
        self._job_response = job_response
        self._pending_requests = pending_request

    def job_poll(self) -> MockResponse:
        if self._pending_requests > 0:
            self._pending_requests -= 1
            job_response = JobResponse(job_status=JobStatus.PROCESSING)
        else:
            job_response = self._job_response
        return MockResponse(status_code=200, content=job_response.model_dump_json())
