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

from uuid import UUID

from evo.common.exceptions import EvoClientException

from .endpoints.models import JobErrorPayload


class JobFailedException(EvoClientException):
    """Exception raised when a job fails.

    :param job_id: ID of the job that failed.
    :param error_payload: The error payload of the failed job.
    """

    def __init__(self, job_id: UUID, error_payload: JobErrorPayload):
        self.job_id = job_id
        self.error_payload = error_payload
        super().__init__(f"Job {job_id} failed: {self.error_payload.title}")


class UnknownJobPayload(EvoClientException):
    def __init__(self, job_id: UUID, error_message: str):
        self.job_id = job_id
        super().__init__(error_message)


class CacheNotConfiguredException(EvoClientException):
    """Exception raised when the cache is not configured."""


class MissingColumnInTable(EvoClientException):
    """Exception raised when the provided table is missing some columns."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message
