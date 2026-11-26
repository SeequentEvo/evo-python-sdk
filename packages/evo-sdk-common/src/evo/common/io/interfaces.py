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

from abc import abstractmethod
from typing import Protocol, runtime_checkable

__all__ = [
    "IDestination",
    "ISource",
]


@runtime_checkable
class IDestination(Protocol):
    """A local or remote destination for managed file IO.

    IDestination implementations should raise a subtype of ChunkedIOError when a recoverable error occurs. The specific
    error will depend on the implementation and must be capable of recovering the IDestination from the failure state.
    """

    @abstractmethod
    async def write_chunk(self, offset: int, data: bytes) -> None:
        """Write raw data to the destination at the provided offset"""


@runtime_checkable
class ISource(Protocol):
    """A local or remote source for managed file IO.

    ISource implementations should raise a subtype of ChunkedIOError when a recoverable error occurs. The specific
    error will depend on the implementation and must be capable of recovering the ISource from the failure state.
    """

    @abstractmethod
    async def get_size(self) -> int:
        """Get the size of the source data"""

    @abstractmethod
    async def read_chunk(self, offset: int, length: int) -> bytes:
        """Read <length> bytes of raw data from the source, starting at the given offset"""
