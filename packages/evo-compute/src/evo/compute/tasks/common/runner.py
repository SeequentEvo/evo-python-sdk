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

"""Task runner registry for dispatching tasks based on parameter types.

This module provides a registry-based system for running compute tasks. Each task
type registers its parameter class and runner function, allowing the unified `run()`
function to dispatch to the correct runner based on the parameter type.

This enables running multiple different task types together in a single call.

Example:
    >>> from evo.compute.tasks import run
    >>> from evo.compute.tasks.kriging import KrigingParameters
    >>> from evo.compute.tasks.simulation import SimulationParameters  # future
    >>>
    >>> # Run mixed task types together
    >>> results = await run(manager, [
    ...     KrigingParameters(...),
    ...     SimulationParameters(...),
    ...     KrigingParameters(...),
    ... ], preview=True)
"""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, TypeVar

from evo.common import IContext
from evo.common.interfaces import IFeedback
from evo.common.utils import NoFeedback, split_feedback

__all__ = [
    "TaskRegistry",
    "get_task_runner",
    "register_task_runner",
    "run_tasks",
]


# Type for task results
TResult = TypeVar("TResult")

# Type for runner functions: async (context, params, *, preview) -> result
RunnerFunc = Callable[..., Awaitable[Any]]


class TaskRegistry:
    """Registry mapping parameter types to their runner functions.

    This is a singleton that stores the mapping from parameter class types
    to their corresponding async runner functions.
    """

    _instance: "TaskRegistry | None" = None
    _runners: dict[type, RunnerFunc]

    def __new__(cls) -> "TaskRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._runners = {}
        return cls._instance

    def register(self, param_type: type, runner: RunnerFunc) -> None:
        """Register a runner function for a parameter type.

        Args:
            param_type: The parameter class (e.g., KrigingParameters)
            runner: Async function with signature (context, params) -> result
        """
        self._runners[param_type] = runner

    def get_runner(self, param_type: type) -> RunnerFunc | None:
        """Get the runner function for a parameter type.

        Args:
            param_type: The parameter class to look up

        Returns:
            The registered runner function, or None if not found
        """
        return self._runners.get(param_type)

    def get_runner_for_params(self, params: Any) -> RunnerFunc:
        """Get the runner function for a parameter instance.

        Args:
            params: A parameter object instance

        Returns:
            The registered runner function

        Raises:
            TypeError: If no runner is registered for the parameter type
        """
        param_type = type(params)
        runner = self._runners.get(param_type)
        if runner is None:
            registered = ", ".join(t.__name__ for t in self._runners.keys())
            raise TypeError(
                f"No task runner registered for parameter type '{param_type.__name__}'. "
                f"Registered types: {registered or 'none'}"
            )
        return runner

    def clear(self) -> None:
        """Clear all registered runners (mainly for testing)."""
        self._runners.clear()


# Global registry instance
_registry = TaskRegistry()


def register_task_runner(param_type: type, runner: RunnerFunc) -> None:
    """Register a task runner function for a parameter type.

    This function is called by task modules to register their runners.

    Args:
        param_type: The parameter class (e.g., KrigingParameters)
        runner: Async function with signature (context, params) -> result

    Example:
        >>> from evo.compute.tasks.common.runner import register_task_runner
        >>>
        >>> async def _run_kriging(context, params):
        ...     # implementation
        ...     pass
        >>>
        >>> register_task_runner(KrigingParameters, _run_kriging)
    """
    _registry.register(param_type, runner)


def get_task_runner(param_type: type) -> RunnerFunc | None:
    """Get the registered runner for a parameter type.

    Args:
        param_type: The parameter class to look up

    Returns:
        The registered runner function, or None if not found
    """
    return _registry.get_runner(param_type)


async def run_tasks(
    context: IContext,
    parameters: list[Any],
    *,
    fb: IFeedback = NoFeedback,
    preview: bool = False,
) -> list[Any]:
    """Run multiple tasks concurrently, dispatching based on parameter types.

    This function looks up the appropriate runner for each parameter based on
    its type, allowing different task types to be run together.

    Args:
        context: The context providing connector and org_id
        parameters: List of parameter objects (can be mixed types)
        fb: Feedback interface for progress updates
        preview: If True, sets the ``API-Preview: opt-in`` header on requests.
            Required for tasks that are still in preview. Defaults to False.

    Returns:
        List of results in the same order as the input parameters

    Raises:
        TypeError: If any parameter type doesn't have a registered runner

    Example:
        >>> # Run mixed task types
        >>> results = await run_tasks(manager, [
        ...     KrigingParameters(...),
        ...     SimulationParameters(...),  # future task type
        ... ], preview=True)
    """
    if len(parameters) == 0:
        return []

    total = len(parameters)

    # Validate all parameters have registered runners upfront
    runners = []
    for params in parameters:
        runner = _registry.get_runner_for_params(params)
        runners.append(runner)

    # Split feedback across tasks
    per_task_fb = split_feedback(fb, [1.0] * total)

    async def _run_one(i: int, params: Any, runner: RunnerFunc, task_fb: IFeedback) -> tuple[int, Any]:
        result = await runner(context, params, preview=preview)
        task_fb.progress(1.0)
        return i, result

    tasks = [
        asyncio.create_task(_run_one(i, params, runner, per_task_fb[i]))
        for i, (params, runner) in enumerate(zip(parameters, runners))
    ]

    results: list[Any | None] = [None] * total

    done_count = 0
    for fut in asyncio.as_completed(tasks):
        try:
            i, res = await fut
            results[i] = res
            done_count += 1
            fb.progress(done_count / total, f"Running {done_count}/{total}...")
        except Exception:
            done_count += 1
            # Cancel remaining to fail fast
            for t in tasks:
                t.cancel()
            raise

    fb.progress(1.0, f"Completed {total}/{total}")

    return [r for r in results if r is not None]
