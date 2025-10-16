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

from dataclasses import dataclass

from .endpoints.models import JobStatusEnum

__all__ = [
    "JobProgress",
    "JobStatusEnum",
]


@dataclass(frozen=True, kw_only=True)
class JobProgress:
    status: JobStatusEnum
    """The status of the job."""

    progress: int | None = None
    """A number between 0 and 100 representing the progress of the job."""

    message: str | None = None
    """A message describing the current progress of the job."""

    def __str__(self) -> str:
        msg = f"[{self.status.value}]"
        if self.progress is not None:
            msg += f" {self.progress}%"
        if self.message is not None:
            msg += f" > {self.message}"
        return msg
