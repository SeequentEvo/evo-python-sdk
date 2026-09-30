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
from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Generic, TypeVar
from weakref import WeakValueDictionary

from evo import logging

from .._types import PathLike
from ..data import ResourceMetadata
from ..interfaces import ICache, IFeedback, ITransport
from ..utils import NoFeedback, Retry
from .http import HTTPSource

logger = logging.getLogger("io.download")

T = TypeVar("T", bound=ResourceMetadata)

_cache_download_locks: WeakValueDictionary[Path, asyncio.Lock] = WeakValueDictionary()


class Download(ABC, Generic[T]):
    """A base class for referencing binary data that needs to be downloaded.

    Each SDK that provides binary data from an API is expected to extend this class, providing the specific implementation
    for generating download URLs and caching the downloaded data.
    """

    @property
    @abstractmethod
    def label(self) -> str:
        """The label of the resource to be downloaded."""
        ...

    @property
    @abstractmethod
    def metadata(self) -> T:
        """The metadata of the resource to be downloaded."""
        ...

    @abstractmethod
    def _get_cache_location(self, cache: ICache) -> Path:
        """Generate the cache location for the resource to be downloaded.

        :param cache: The cache to resolve the cache location.

        :returns: The cache location.
        """
        ...

    @abstractmethod
    async def get_download_url(self) -> str:
        """Generate a URL that will be used to download the resource.

        This method may be called multiple times to generate a new URL if the last URL expires.

        :returns: The download URL.
        """
        ...

    @contextmanager
    def _downloading(self) -> Iterator[None]:
        """A context manager that logs the start and end of the download process."""
        logger.debug(f"Attempting to download data (label: {self.label})")
        yield
        logger.debug(f"Data downloaded successfully (label: {self.label})")

    async def download_to_path(
        self,
        filename: PathLike,
        transport: ITransport,
        max_workers: int | None = None,
        retry: Retry | None = None,
        overwrite: bool = False,
        fb: IFeedback = NoFeedback,
    ) -> None:
        """Download an HTTP resource to a file with the given filename.

        The url generator MUST generate a resource URL that can be used to access the required resource. The URL
        generator may be called again if the last URL expires (unless Retry is initialised with max_attempts == 0).

        :param filename: file to download to
        :param transport: The transport to use for the API calls.
        :param max_workers: The maximum number of concurrent connections to use for data transfer. If None, the default
            is `min(32, os.cpu_count() + 4)`
        :param retry: A Retry object with a wait strategy. If None, a default Retry is created. If a chunk is successfully
            transferred the attempt counter will be reset.
        :param overwrite: whether to overwrite an existing local file
        :param fb: feedback to track the download, by tracking writes to the file

        :raises FileNameTooLongError: If the filename is too long.
        :raises ValueError: if the file already exists and overwrite is False
        :raises ChunkedIOException: if a non-recoverable exception occurred.
        :raises RetryError: if the maximum number of consecutive attempts have failed.
        """
        with self._downloading():
            await HTTPSource.download_file(
                filename=filename,
                url_generator=self.get_download_url,
                transport=transport,
                max_workers=max_workers,
                retry=retry,
                overwrite=overwrite,
                fb=fb,
            )

    async def download_to_cache(
        self,
        cache: ICache,
        transport: ITransport,
        max_workers: int | None = None,
        retry: Retry | None = None,
        overwrite: bool = False,
        fb: IFeedback = NoFeedback,
    ) -> Path:
        """Download an HTTP resource to a file with the given filename.

        The url generator MUST generate a resource URL that can be used to access the required resource. The URL
        generator may be called again if the last URL expires (unless Retry is initialised with max_attempts == 0).

        :param cache: The cache to store the downloaded file.
        :param transport: The transport to use for the API calls.
        :param max_workers: The maximum number of concurrent connections to use for data transfer. If None, the default
            is `min(32, os.cpu_count() + 4)`
        :param retry: A Retry object with a wait strategy. If None, a default Retry is created. If a chunk is successfully
            transferred the attempt counter will be reset.
        :param overwrite: whether to overwrite an existing local file
        :param fb: feedback to track the download, by tracking writes to the file

        :raises FileNameTooLongError: If the filename is too long.
        :raises ValueError: if the file already exists and overwrite is False
        :raises ChunkedIOException: if a non-recoverable exception occurred.
        :raises RetryError: if the maximum number of consecutive attempts have failed.
        """
        location = self._get_cache_location(cache).resolve()
        # Serialize same-process downloads for this path. The weak-value dictionary drops the entry once no active
        # caller or waiter holds the lock.
        lock = _cache_download_locks.setdefault(location, asyncio.Lock())
        async with lock:
            if not location.exists() or overwrite:
                await self.download_to_path(
                    filename=location, transport=transport, max_workers=max_workers, retry=retry, fb=fb
                )
            else:
                logger.debug(f"Skipping download because data already in cache (label: {self.label})")
                fb.progress(1.0)
        return location
