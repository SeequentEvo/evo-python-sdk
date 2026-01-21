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

"""Generic utilities for running multiple compute tasks concurrently."""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, TypeVar

from evo.common import IContext
from evo.common.interfaces import IFeedback
from evo.common.utils import NoFeedback, split_feedback

__all__ = [
    "run_multiple",
]


TParams = TypeVar("TParams")
TResult = TypeVar("TResult")


async def run_multiple(
    context: IContext,
    parameters: list[TParams],
    run_fn: Callable[[IContext, TParams], Awaitable[TResult]],
    *,
    fb: IFeedback = NoFeedback,
) -> list[TResult]:
    """
    Run multiple compute tasks concurrently with aggregated feedback.

    This is a generic utility for running multiple instances of any compute task
    in parallel. Progress is aggregated across all tasks.

    Args:
        context: The context providing connector and org_id
        parameters: List of parameter objects for each task instance
        run_fn: The async function to run for each parameter set.
                Should have signature: async (context, params) -> result
        fb: Feedback interface for progress updates

    Returns:
        List of results in the same order as the input parameters

    Example:
        >>> from evo.compute.tasks import run_kriging, KrigingParameters
        >>> from evo.compute.tasks.common import run_multiple
        >>>
        >>> results = await run_multiple(
        ...     context,
        ...     param_sets,
        ...     run_kriging,
        ...     fb=FeedbackWidget("Running scenarios"),
        ... )
    """
    if len(parameters) == 0:
        return []

    total = len(parameters)

    # Split feedback across tasks to aggregate total progress linearly
    per_task_fb = split_feedback(fb, [1.0] * total)

    # Wrapper that returns (index, result) for robust mapping
    async def _run_one(i: int, param: TParams) -> tuple[int, TResult]:
        res = await run_fn(context, param)
        return i, res

    tasks = [asyncio.create_task(_run_one(i, param)) for i, param in enumerate(parameters)]

    results: list[TResult | None] = [None] * total

    done_count = 0
    for fut in asyncio.as_completed(tasks):
        try:
            i, res = await fut
            results[i] = res
            done_count += 1
            percent = done_count / total
            msg = f"{done_count}/{total} tasks completed ({int(percent * 100)}%)"
            per_task_fb[i].progress(1.0, msg)
        except Exception:
            done_count += 1
            percent = done_count / total
            msg = f"{done_count}/{total} tasks completed ({int(percent * 100)}%) with errors"
            fb.progress(percent, msg)
            # Cancel remaining to fail fast
            for t in tasks:
                t.cancel()
            raise

    # Type assertion: all results should be populated
    return [r for r in results if r is not None]

