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

import asyncio
import traceback
from collections.abc import Iterator
from dataclasses import dataclass
from threading import Lock

from evo import logging

from ..interfaces import IFeedback
from ..utils import NoFeedback, Retry, RetryHandler
from .exceptions import ChunkedIOError
from .interfaces import IDestination, ISource

logger = logging.getLogger("io.manager")

__all__ = [
    "DEFAULT_CHUNK_SIZE",
    "ChunkMetadata",
    "ChunkedIOManager",
    "ChunkedIOTracker",
]

DEFAULT_CHUNK_SIZE = 4 * 1024 * 1024
"""The default max size of each chunk to be transferred."""


@dataclass(frozen=True)
class ChunkMetadata:
    """Metadata for a chunk of data to be transferred."""

    id: int
    """The chunk ID, starting from 0."""

    offset: int
    """The byte offset where this chunk starts, where 0 is the beginning of the file."""

    size: int
    """The size of this chunk in bytes. The final chunk will usually be smaller than the chunk_size."""

    completed: bool
    """Whether this chunk has been successfully transferred."""


class ChunkedIOTracker:
    """Tracks the progress of chunked data transfer."""

    def __init__(self, file_size: int, chunk_size: int):
        """
        :param file_size: The total size of the file to be transferred.
        :param chunk_size: The maximum size of each chunk to be transferred.
        """
        self._chunk_size = chunk_size
        self._total_size = file_size
        self._n_chunks, rem = divmod(file_size, chunk_size)
        if rem:
            self._n_chunks += 1
        self._n_chunks_completed = 0
        self._chunk_status: list[bool] = [False] * self._n_chunks
        self._lock = Lock()

    def __iter__(self) -> Iterator[ChunkMetadata]:
        """Iterate over the chunks of data to be transferred.

        :return: An iterator of `evo.common.io.ChunkMetadata` objects.
        """
        for i, is_complete in enumerate(self._chunk_status):
            this_offset = i * self._chunk_size
            this_chunk_size = min(self._chunk_size, self._total_size - this_offset)
            yield ChunkMetadata(id=i, offset=this_offset, size=this_chunk_size, completed=is_complete)

    def set_complete(self, chunk: ChunkMetadata) -> None:
        """Mark a chunk as completed (successfully transferred).

        :param chunk: Metadata for the chunk that has been transferred.
        """
        with self._lock:
            self._chunk_status[chunk.id] = True
            self._n_chunks_completed += 1

    def get_progress(self) -> float:
        """Get the percentage of total chunks that have been transferred.

        :return: A float between 0 and 1 representing the progress, rounded to 2 decimal places.
        """
        if self._n_chunks == 0:
            return 1.0
        return round(self._n_chunks_completed / self._n_chunks, 2)

    def is_complete(self) -> bool:
        """Check if all chunks have been successfully transferred.

        :return: True if all chunks have been successfully transferred.
        """
        return self._n_chunks_completed == self._n_chunks


class ChunkedIOManager:
    """Manager for robust multithreaded data transfer from `evo.common.io.interfaces.ISource` to
    `evo.common.io.interfaces.IDestination`.

    This class is designed to transfer large files in chunks, with the ability to retry failed transfers. The progress
    of the transfer can be reported to a `evo.common.interfaces.IFeedback` object.
    """

    def __init__(
        self,
        message: str = "",
        retry: Retry | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        max_workers: int | None = None,
        chunk_limiter: asyncio.Semaphore | None = None,
    ):
        """
        :param message: The message to update feedback progress with.
        :param retry: A Retry object with a wait strategy. If None, a default Retry is created. If a chunk is
            successfully transferred the attempt counter will be reset.
        :param chunk_size: The maximum size of each chunk to be transferred. The final chunk will usually be smaller
            than the chunk_size.
        :param max_workers: Maximum concurrent chunk workers for this file. Defaults to four.
        :param chunk_limiter: Optional transport-owned budget acquired before reading each chunk.
        """
        self._message = message
        self._retry = Retry(logger=logger) if retry is None else retry
        self._chunk_size = chunk_size
        self._tracker: ChunkedIOTracker | None = None

        max_workers = 4 if max_workers is None else max_workers
        for name, value in (("max_workers", max_workers), ("chunk_size", chunk_size)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self._max_workers = max_workers
        self._chunk_limiter = chunk_limiter or asyncio.BoundedSemaphore(max_workers)

    async def _transfer(self, source: ISource, destination: IDestination, chunk: ChunkMetadata) -> float:
        async with self._chunk_limiter:
            data = await source.read_chunk(chunk.offset, chunk.size)
            try:
                await destination.write_chunk(chunk.offset, data)
            except BaseException as error:
                # Retry history must not keep request payloads alive after releasing a chunk slot.
                pending = [error]
                seen = set()
                while pending:
                    failure = pending.pop()
                    if id(failure) in seen:
                        continue
                    seen.add(id(failure))
                    traceback.clear_frames(failure.__traceback__)
                    pending.extend(e for e in (failure.__cause__, failure.__context__) if e is not None)
                raise
            finally:
                # Exceptions retain traceback frames: release payloads before returning a budget slot.
                del data
            self._tracker.set_complete(chunk)
            return self._tracker.get_progress()

    async def _attempt_transfer(
        self, source: ISource, destination: IDestination, fb: IFeedback, retry_handler: RetryHandler
    ) -> None:
        if self._tracker is None:
            self._tracker = ChunkedIOTracker(await source.get_size(), self._chunk_size)
        last_progress = self._tracker.get_progress()
        fb.progress(last_progress, self._message)

        chunks = (chunk for chunk in self._tracker if not chunk.completed)

        async def worker() -> None:
            nonlocal last_progress
            for chunk in chunks:
                this_progress = await self._transfer(source, destination, chunk)
                if this_progress != last_progress:
                    fb.progress(this_progress, self._message)
                    last_progress = this_progress
                retry_handler.reset_counter()

        transfers = []
        group = None
        try:
            for _ in range(self._max_workers):
                transfers.append(asyncio.create_task(worker()))
            group = asyncio.gather(*transfers)
            await asyncio.shield(group)
        finally:
            for transfer in transfers:
                if not transfer.done():
                    transfer.cancel()
            cleanup = asyncio.gather(*transfers, *([group] if group is not None else []), return_exceptions=True)
            cancelled = False
            while not cleanup.done():
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError:
                    cancelled = True
            if cancelled:
                raise asyncio.CancelledError

    async def run(self, source: ISource, destination: IDestination, fb: IFeedback | None = None) -> None:
        """Transfer all data from source to destination.

        Recoverable transfer failures must be raised as a subclass of `evo.common.io.exceptions.ChunkedIOError`,
        this will retry recoverable transfers up to the max_attempts specified in Retry after calling the error's
        recover() method. The attempt counter is reset whenever a chunk is successfully transferred.

        If the maximum number of attempts is reached, `evo.common.exceptions.RetryError` will be
        raised with the last error.

        :param source: The source to read data from.
        :param destination: The destination to write data to.
        :param fb: A feedback object for reporting transfer progress.
        """
        fb = NoFeedback if fb is None else IFeedback.adapt(fb, allow_implicit=True)
        async for handler in self._retry:
            with handler.suppress_errors(ChunkedIOError):
                await self._attempt_transfer(source, destination, fb, handler)

            if isinstance(handler.exception, ChunkedIOError):
                fb.progress(self._tracker.get_progress() if self._tracker is not None else 0.0, "Retrying...")
                if not await handler.exception.recover():
                    raise handler.exception

    def is_complete(self) -> bool:
        """Check if all chunks have been successfully transferred.

        :return: True if all chunks have been successfully transferred.
        """
        return self._tracker is not None and self._tracker.is_complete()
