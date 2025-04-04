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

from ..exceptions import EvoClientException

__all__ = [
    "ChunkedIOError",
    "ChunkedIOException",
    "RenewalError",
    "RenewalTimeoutError",
]


class ChunkedIOException(EvoClientException):
    """Base exception for unrecoverable errors that are expected within io.

    These exceptions should be caught by client code.
    """


class ChunkedIOError(ChunkedIOException):
    """Base exception for recoverable errors that are handled within chunked_io.

    All ChunkedIOError subtypes are considered recoverable and should implement a recover() method that prepares the
    `evo.common.io.interfaces.ISource` or `evo.common.io.interfaces.IDestination` that raised
    the error for a retry attempt.

    :ivar raised_by: The source or destination that raised the exception.
    """

    def __init__(self, message: str) -> None:
        """
        :param message: The error message.
        """
        super().__init__(message)

    async def recover(self) -> bool:
        """Run the steps required to recover from the original error and prepare for a retry attempt."""
        return False  # ChunkedIOError is not recoverable by default.


class RenewalError(EvoClientException):
    """RenewalError is raised when renewing the url fails."""


class RenewalTimeoutError(RenewalError):
    """RenewalTimeoutError is raised when an attempt to renew a url is made within a specified threshold."""


class DataExistsError(EvoClientException):
    """Raised when requesting an upload url for a binary resource that already exists in the target service."""


class DataNotFoundError(EvoClientException):
    """Raised when requesting a download url for a binary resource that does not exist in the target service."""
