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
from collections.abc import Coroutine
from contextvars import ContextVar
from typing import Any, TypeVar

_T = TypeVar("_T")
_MAX_CONCURRENT_UPLOADS = 8
_upload_semaphore: ContextVar[asyncio.Semaphore | None] = ContextVar("object_upload_semaphore", default=None)


def get_upload_semaphore() -> asyncio.Semaphore | None:
    """Return the serialization's shared limit, if a serialization is active."""
    return _upload_semaphore.get()


async def gather_uploads(*coroutines: Coroutine[Any, Any, _T]) -> list[_T]:
    """Collect ordered results, cancelling and draining siblings on failure or cancellation."""
    token = None
    tasks: list[asyncio.Task[_T]] = []
    try:
        if _upload_semaphore.get() is None:
            token = _upload_semaphore.set(asyncio.Semaphore(_MAX_CONCURRENT_UPLOADS))
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
