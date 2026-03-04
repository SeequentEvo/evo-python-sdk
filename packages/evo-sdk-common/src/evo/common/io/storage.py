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

import asyncio
import base64
import os
from collections.abc import Awaitable, Callable

from evo import logging
from evo.common.interfaces import ITransport

from .._types import PathLike, resolve_path
from ..data import EmptyResponse, RequestMethod
from ..interfaces import IFeedback
from ..utils import Retry
from .bytes import BytesSource
from .chunked_io_manager import ChunkedIOManager
from .exceptions import ChunkedIOError
from .http import HTTPIOBase
from .interfaces import IDestination

logger = logging.getLogger("io.storage")

__all__ = [
    "StorageDestination",
]


class StorageBlock:
    """Metadata about a block in storage."""

    def __init__(self, byte_offset: int) -> None:
        """Create a new block with the given byte offset."""
        self._offset = byte_offset

    @property
    def byte_offset(self) -> int:
        """The byte offset of the block."""
        return self._offset

    @property
    def id(self) -> str:
        """The block id of the block."""
        block_index = f"{self._offset:032d}"
        block_id_bytes = base64.b64encode(block_index.encode("utf-8"))
        return block_id_bytes.decode("utf-8")

    def __str__(self) -> str:
        return f"<Latest>{self.id}</Latest>"


class BlockList:
    """A list of blocks to be committed to storage."""

    def __init__(self) -> None:
        self._mutex = asyncio.Lock()
        self._sealed = False
        self._blocks: list[StorageBlock] = []

    async def add_block(self, byte_offset: int) -> str:
        """Add a block with the given byte offset to the list.

        :param byte_offset: The byte offset of the block.

        :return: The block id of the created block.
        """
        new_block = StorageBlock(byte_offset)
        async with self._mutex:
            assert not self._sealed, "Cannot add block to sealed block list."
            existing_blocks = {block.byte_offset: block for block in self._blocks}
            if byte_offset in existing_blocks.keys():
                # Remove the old block for this offset, duplicates are not allowed.
                self._blocks.remove(existing_blocks[byte_offset])
            self._blocks.append(new_block)
        return new_block.id

    async def prepare(self) -> bytes:
        """Prepare the block list for committing to storage.

        :return: The XML representation of the block list as bytes.
        """
        async with self._mutex:
            self._sealed = True
            return "\n".join(
                [
                    '<?xml version="1.0" encoding="utf-8"?>',
                    "<BlockList>",
                    *(f"  {block}" for block in sorted(self._blocks, key=lambda b: b.byte_offset)),
                    "</BlockList>",
                ]
            ).encode("utf-8")


class StorageDestination(HTTPIOBase, IDestination):
    """`evo.common.io.interfaces.IDestination` implementation for uploading to storage."""

    def __init__(
        self,
        url_callback: Callable[[], Awaitable[str]],
        transport: ITransport,
        additional_headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(url_callback, transport, additional_headers)
        self._block_list = BlockList()
        self._committed = False

    async def write_chunk(self, offset: int, data: bytes) -> None:
        """Write a chunk of data to storage.

        :param offset: Byte number to start writing from.
        :param data: Bytes to write.

        :raises AssertionError: if the data is already committed
        """
        logger.debug(f"Writing {len(data)} at offset {offset}")
        if self._committed:
            raise AssertionError("Cannot write chunk because data is already committed.")
        block_id = await self._block_list.add_block(offset)
        logger.debug(f"Staging block at offset {offset:032d} with id {block_id}")
        await self._query_resource(
            RequestMethod.PUT,
            query_params={"comp": "block", "blockid": block_id},
            header_params={"Content-Length": str(len(data))},
            body=data,
            response_types_map={"201": EmptyResponse},
        )
        logger.debug(f"Staging {block_id} complete")

    async def _commit(self) -> None:
        payload = await self._block_list.prepare()
        logger.debug(f"Committing blocks:\n{payload.decode('utf-8')}")
        await self._query_resource(
            RequestMethod.PUT,
            query_params={"comp": "blocklist"},
            header_params={
                "Content-Type": "text/plain; charset=UTF-8",
                "Content-Length": str(len(payload)),
            },
            body=payload,
            response_types_map={"201": EmptyResponse},
        )

    async def commit(self, retry: Retry | None = None) -> None:
        """Commit all staged blocks to storage.

        :param retry: A Retry object with a wait strategy. If a chunk is successfully transferred the attempt counter
            will be reset.

        :raises ChunkedIOException: if a non-recoverable exception occurred. ChunkedIOException should be caught by
            client code and raised as either ProcessingError or ApplicationError for a user-friendly presentation.
        :raises RetryError: if the maximum number of consecutive attempts have failed. RetryError is a
            subclass of ChunkedIOException.
        """
        logger.debug("Committing data...")
        assert not self._committed, "Cannot commit data because data is already committed."

        if retry is None:
            retry = Retry(logger=logger)

        async for handler in retry:
            with handler.suppress_errors(ChunkedIOError):
                await self._commit()

            if isinstance(handler.exception, ChunkedIOError):
                if not await handler.exception.recover():
                    raise handler.exception

        self._committed = True
        logger.debug("The data was successfully committed.")

    @staticmethod
    async def upload_file(
        filename: PathLike,
        url_generator: Callable[[], Awaitable[str]],
        transport: ITransport,
        max_workers: int | None = None,
        retry: Retry | None = None,
        additional_headers: dict[str, str] | None = None,
        fb: IFeedback | None = None,
    ) -> None:
        """Upload a file with the given filename to the given Storage URL.

        The url generator MUST generate a url that contains a valid SAS (shared access signature) token. The url
        generator may be called again if the last url expires (unless Retry is initialised with max_attempts == 0).

        :param filename: file to upload
        :param url_generator: An awaitable callback that accepts no arguments and returns a URL to download from
        :param transport: The transport to use for the API calls.
        :param max_workers: The maximum number of concurrent connections to use for data transfer. If None, the default
            is `min(32, os.cpu_count() + 4)`
        :param retry: A Retry object with a wait strategy. If None, a default Retry is created. If a chunk is successfully
            transferred the attempt counter will be reset.
        :param additional_headers: Additional headers to include in each request.
        :param fb: feedback to track the upload, by tracking reads from the file only

        :raises ValueError: if the file to upload does not exist
        :raises ChunkedIOException: if a non-recoverable exception occurred.
        :raises RetryError: if the maximum number of consecutive attempts have failed. RetryError is a subclass
            of ChunkedIOException.
        """
        src_path = resolve_path(filename)

        if not src_path.is_file():
            raise ValueError(f"file {src_path} does not exist")

        if retry is None:
            retry = Retry(logger=logger)

        manager = ChunkedIOManager(message="Uploading", retry=retry, max_workers=max_workers)
        async with StorageDestination(
            url_callback=url_generator, transport=transport, additional_headers=additional_headers
        ) as destination:
            size = os.path.getsize(src_path)
            with src_path.open("rb") as input_:
                source = BytesSource(input_, size)
                await manager.run(source, destination, fb)
            await destination.commit(retry=retry)
