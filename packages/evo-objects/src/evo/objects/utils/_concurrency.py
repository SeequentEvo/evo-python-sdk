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

"""Off-loop preparation and cancellation-safe concurrent operations."""

import asyncio
from collections.abc import Callable, Coroutine
from typing import Any, ParamSpec, TypeVar

from evo import logging

_T = TypeVar("_T")
_P = ParamSpec("_P")
logger = logging.getLogger("object.uploads")


async def _drain_tasks(*tasks: asyncio.Future[_T]) -> list[_T | BaseException]:
    cleanup = asyncio.gather(*tasks, return_exceptions=True)
    while not cleanup.done():
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            # Repeated cancellation must not interrupt cleanup or abandon a cache writer.
            continue
    return cleanup.result()


async def run_upload_preparation(function: Callable[_P, _T], *args: _P.args, **kwargs: _P.kwargs) -> _T:
    """Run synchronous preparation in the default executor, draining active work on cancellation."""
    worker = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        for result in await _drain_tasks(worker):
            if isinstance(result, Exception):
                logger.debug(
                    "Preparation failed during cancellation", exc_info=(type(result), result, result.__traceback__)
                )
        raise


async def gather_cancelling(*coroutines: Coroutine[Any, Any, _T]) -> list[_T]:
    """Collect ordered results, cancelling and draining siblings on failure or cancellation."""
    tasks: list[asyncio.Task[_T]] = []
    group = None
    try:
        for coroutine in coroutines:
            tasks.append(asyncio.create_task(coroutine))
        group = asyncio.gather(*tasks)
        return list(await asyncio.shield(group))
    except BaseException as error:
        for coroutine in coroutines[len(tasks) :]:
            coroutine.close()
        for task in tasks:
            if not task.done():
                task.cancel()
        results = await _drain_tasks(*tasks)
        if group is not None:
            await _drain_tasks(group)
        for result in results:
            if isinstance(result, Exception) and result is not error:
                logger.debug("Additional upload failure", exc_info=(type(result), result, result.__traceback__))
        raise
