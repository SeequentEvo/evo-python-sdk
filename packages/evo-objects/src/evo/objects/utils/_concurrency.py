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

"""Concurrency shared by table serialization and uploads on each event loop."""

import asyncio
from collections.abc import Callable, Coroutine, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, ParamSpec, TypeVar
from weakref import ReferenceType, WeakKeyDictionary, ref

from evo import logging

_T = TypeVar("_T")
_P = ParamSpec("_P")
_DEFAULT_MAX_CONCURRENCY = 4  # Downloads and uploads share the same default.
logger = logging.getLogger("object.uploads")


class _UploadScope:
    """Share a limit across clients without retaining closed event loops.

    Contended semaphores retain their loop, so both loop keys and semaphore values
    must be weak references. Active work keeps its semaphore alive.
    """

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.semaphores: WeakKeyDictionary[asyncio.AbstractEventLoop, ReferenceType[asyncio.Semaphore]] = (
            WeakKeyDictionary()
        )

    def get_semaphore(self) -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        reference = self.semaphores.get(loop)
        semaphore = reference() if reference is not None else None
        if semaphore is None:
            semaphore = asyncio.Semaphore(self.limit)
            self.semaphores[loop] = ref(semaphore)
        return semaphore


_upload_scope: ContextVar[_UploadScope] = ContextVar(
    "object_upload_scope", default=_UploadScope(_DEFAULT_MAX_CONCURRENCY)
)


@contextmanager
def upload_concurrency(max_concurrent: int = _DEFAULT_MAX_CONCURRENCY) -> Iterator[None]:
    """Limit concurrent table preparation and uploads within this scope.

    Applies to typed-object operations and direct ObjectDataClient uploads, including
    clients created internally. Conversion/serialization and each upload attempt acquire a slot;
    backoff sleeps do not. Settings are inherited by child asyncio tasks, not unrelated
    tasks. Each event loop has its own semaphore, so a scope can span asyncio.run calls.
    Outside an explicit scope, all clients and operations on a loop share the default limit.

    :param max_concurrent: Maximum number of concurrent preparation or upload operations. Defaults to four.
    :raises ValueError: If max_concurrent is not a positive integer.
    """
    if isinstance(max_concurrent, bool) or not isinstance(max_concurrent, int) or max_concurrent < 1:
        raise ValueError("max_concurrent must be a positive integer")
    token = _upload_scope.set(_UploadScope(max_concurrent))
    try:
        yield
    finally:
        _upload_scope.reset(token)


def get_upload_semaphore() -> asyncio.Semaphore:
    """Return the shared semaphore; only leaf work may hold a slot, never a parent awaiting child uploads."""
    return _upload_scope.get().get_semaphore()


async def run_upload_preparation(function: Callable[_P, _T], *args: _P.args, **kwargs: _P.kwargs) -> _T:
    """Convert and serialize in a bounded worker, draining it before releasing a cancelled slot."""
    async with get_upload_semaphore():
        worker = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
        try:
            return await asyncio.shield(worker)
        except asyncio.CancelledError:
            # Running threads cannot be cancelled; finish their cache writes before returning.
            await asyncio.gather(worker, return_exceptions=True)
            raise


async def gather_uploads(*coroutines: Coroutine[Any, Any, _T]) -> list[_T]:
    """Collect ordered results, cancelling and draining siblings on failure or cancellation."""
    tasks: list[asyncio.Task[_T]] = []
    try:
        for coroutine in coroutines:
            tasks.append(asyncio.create_task(coroutine))
        # Unlike gather(), this does not cancel children before our own cleanup runs.
        for completed in asyncio.as_completed(tasks):
            await completed
        return [task.result() for task in tasks]
    except BaseException as error:
        for coroutine in coroutines[len(tasks) :]:
            coroutine.close()
        for task in tasks:
            task.cancel()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception) and result is not error:
                logger.debug("Additional upload failure", exc_info=(type(result), result, result.__traceback__))
        raise
