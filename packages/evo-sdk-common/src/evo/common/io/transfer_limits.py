#  Copyright (c) 2026 Bentley Systems, Incorporated
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Transport-owned admission controls for file transfers and buffered chunks."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from ..interfaces import ITransport


class TransferLimits:
    """Limits shared by file uploads and downloads within one transport session.

    Custom transports can opt in by exposing a ``transfer_limits`` property returning
    this object while open. Transports without that property retain per-file limiting.
    Chunk slots bound payload count, not total process memory or cached disk usage.
    """

    def __init__(self, max_files: int = 16, max_chunks: int = 16) -> None:
        for name, value in (("max_files", max_files), ("max_chunks", max_chunks)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.files = asyncio.BoundedSemaphore(max_files)
        self.chunks = asyncio.BoundedSemaphore(max_chunks)


@asynccontextmanager
async def file_transfer_slot(transport: ITransport) -> AsyncIterator[TransferLimits | None]:
    """Keep the transport session alive while waiting for and using file admission."""
    async with transport:
        limits = getattr(transport, "transfer_limits", None)
        if limits is None:
            yield None
        elif isinstance(limits, TransferLimits):
            async with limits.files:
                yield limits
        else:
            raise TypeError("transport.transfer_limits must be a TransferLimits instance")
