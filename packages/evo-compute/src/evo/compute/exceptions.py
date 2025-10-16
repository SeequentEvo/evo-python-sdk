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

import copy

from evo.common.exceptions import BaseTypedError, EvoClientException


class JobError(BaseTypedError):
    """Raised when a job fails."""

    def copy(self) -> JobError:
        """Return a copy of the error.

        Job result responses are cached so that they can be retrieved multiple times without
        re-fetching. This method is used to create a copy of an error result so that it can be
        raised from a cached copy without modifying the original.

        :return: A copy of the error.
        """
        return JobError(
            status=self.status,
            reason=self.reason,
            content=copy.deepcopy(self.content),
            headers=self.headers.copy(),
        )


class JobPendingError(EvoClientException):
    """Raised when results are requested but the job is still pending."""

    def __init__(self, url: str, status: str) -> None:
        super().__init__(f"Job at {url} is still pending with status: {status}")
