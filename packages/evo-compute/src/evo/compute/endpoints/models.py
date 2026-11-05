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
from typing import Any

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


# Discovery endpoint models.


class TaskResource(BaseModel):
    """A single compute task advertised by the discovery endpoint."""

    model_config = ConfigDict(
        extra="allow",
    )

    topic: StrictStr
    """The topic the task belongs to (e.g. ``geostatistics``)."""

    name: StrictStr
    """The task name within the topic (e.g. ``kriging``)."""

    key: StrictStr | None = None
    """A stable identifier for the task, if provided."""

    display_name: StrictStr | None = None
    """A human-friendly name for the task."""

    version: StrictStr | None = None
    """The task version."""

    feature_flag: StrictStr | None = None
    """The api-preview feature flag gating the task, when it is a preview task."""

    description: StrictStr | None = None
    """A description of the task."""

    parameters: dict[str, Any] = Field(default_factory=dict)
    """The JSON Schema for the task's parameters (full schema when fetched with ``details=true``)."""

    results: dict[str, Any] | None = None
    """The JSON Schema for the task's results, when advertised."""


class DiscoveryResponse(BaseModel):
    """The paginated response from ``GET /compute/orgs/{org_id}/tasks``."""

    model_config = ConfigDict(
        extra="allow",
    )

    results: list[TaskResource] = Field(default_factory=list)
    """The advertised tasks."""

    limit: StrictInt | None = None
    """The requested page size."""

    offset: StrictInt | None = None
    """The page offset."""

    total: StrictInt | None = None
    """The total number of tasks available."""

    count: StrictInt | None = None
    """The number of tasks returned in this page."""
