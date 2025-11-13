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

from __future__ import annotations

from enum import Enum

from pydantic import AnyUrl, BaseModel, ConfigDict, Field, StrictInt, StrictStr


class JobStatusCompleted(Enum):
    failed = "failed"
    canceled = "canceled"
    succeeded = "succeeded"


class JobStatusOngoing(Enum):
    requested = "requested"
    in_progress = "in progress"
    canceling = "canceling"


class JobStatusEnum(Enum):
    """
    Enum representing the status of a job.
    """

    requested = "requested"
    in_progress = "in progress"
    succeeded = "succeeded"
    failed = "failed"
    cancelling = "cancelling"
    cancelled = "cancelled"


class CompletedJobLinks(BaseModel):
    results: AnyUrl


class OngoingJobLinks(BaseModel):
    cancel: AnyUrl


class ExecuteTaskRequest(BaseModel):
    parameters: dict[str, StrictStr]


class OngoingJobResponse(BaseModel):
    status: JobStatusOngoing


class Error(BaseModel):
    model_config = ConfigDict(
        extra="allow",
    )

    status: StrictInt
    """The status code of the error."""

    type_: StrictStr = Field(alias="type")
    """The type of the error."""

    title: StrictStr
    """The title of the error."""

    detail: StrictStr | None = None
    """A message describing the error."""


class JobStatusResponse(BaseModel):
    model_config = ConfigDict(
        extra="allow",
    )

    status: JobStatusEnum
    """The status of the job."""

    progress: StrictInt | None = Field(None, ge=0, le=100)
    """A number between 0 and 100 representing the progress of the job."""

    message: StrictStr | None = None
    """A message describing the current progress of the job."""

    error: Error | None = None
    """An error that occurred during the job."""


class CompletedJobResponse(BaseModel):
    model_config = ConfigDict(
        extra="allow",
    )

    status: JobStatusEnum
    """The status of the job."""

    results: dict | None = None
    """The results of the job."""

    error: Error | None = None
    """An error that occurred during the job."""
