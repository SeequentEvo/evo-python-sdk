#  Copyright © 2026 Bentley Systems, Incorporated
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Concurrency shared by nested table-serialization operations."""

import asyncio
from collections.abc import Coroutine, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, TypeVar

_T = TypeVar("_T")
_MAX_CONCURRENT_UPLOADS = 8
_max_concurrent_uploads: ContextVar[int] = ContextVar("max_concurrent_uploads", default=_MAX_CONCURRENT_UPLOADS)
_upload_semaphore: ContextVar[asyncio.Semaphore | None] = ContextVar("object_upload_semaphore", default=None)


def get_max_concurrent_uploads() -> int:
    """Return the configured limit for newly created upload semaphores."""
    return _max_concurrent_uploads.get()


@contextmanager
def upload_concurrency(max_concurrent: int = _MAX_CONCURRENT_UPLOADS) -> Iterator[None]:
    """Limit concurrent table serialization and uploads within this scope.

    Applies to typed-object operations and direct ObjectDataClient uploads, including
    clients created internally. Each slot is held from Parquet serialization through
    upload and retries. Settings are inherited by child asyncio tasks, not unrelated tasks.
    Clients created in this scope retain this limit for subsequent direct uploads.

    :param max_concurrent: Maximum number of concurrent tables. Defaults to eight.
    :raises ValueError: If max_concurrent is not a positive integer.
    """
    if isinstance(max_concurrent, bool) or not isinstance(max_concurrent, int) or max_concurrent < 1:
        raise ValueError("max_concurrent must be a positive integer")
    semaphore = asyncio.Semaphore(max_concurrent)
    limit_token = _max_concurrent_uploads.set(max_concurrent)
    semaphore_token = _upload_semaphore.set(semaphore)
    try:
        yield
    finally:
        _upload_semaphore.reset(semaphore_token)
        _max_concurrent_uploads.reset(limit_token)


def get_upload_semaphore() -> asyncio.Semaphore | None:
    """Return the serialization's shared limit, if a serialization is active."""
    return _upload_semaphore.get()


async def gather_uploads(*coroutines: Coroutine[Any, Any, _T]) -> list[_T]:
    """Collect ordered results, cancelling and draining siblings on failure or cancellation."""
    token = None
    tasks: list[asyncio.Task[_T]] = []
    try:
        if _upload_semaphore.get() is None:
            token = _upload_semaphore.set(asyncio.Semaphore(get_max_concurrent_uploads()))
        for coroutine in coroutines:
            tasks.append(asyncio.create_task(coroutine))
        # Unlike gather(), this does not cancel children before our own cleanup runs.
        for completed in asyncio.as_completed(tasks):
            await completed
        return [task.result() for task in tasks]
    except BaseException:
        for coroutine in coroutines[len(tasks) :]:
            coroutine.close()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    finally:
        if token is not None:
            _upload_semaphore.reset(token)
