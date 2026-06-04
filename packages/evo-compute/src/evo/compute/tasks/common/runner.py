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

"""Task runner framework for dispatching compute tasks.

This module provides a class-based system for defining and running compute tasks.
Each task type is defined by subclassing :class:`TaskRunner` with specific parameter
and result types. Subclasses are automatically registered so that the unified
``run()`` function can dispatch to the correct runner based on the parameter type.

Defining a new task runner is as simple as:

.. code-block:: python

    class KrigingRunner(
        TaskRunner[KrigingParameters, KrigingResult],
        topic="geostatistics",
        task="kriging",
    ): ...

Advanced users can run tasks directly:

.. code-block:: python

    result = await KrigingRunner(context, params, preview=True)

Notebook users use the unified ``run()`` function which dispatches automatically:

.. code-block:: python

    from evo.compute.tasks import run

    result = await run(manager, params, preview=True)
"""

from __future__ import annotations

import asyncio
import inspect
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator
from typing import Any, ClassVar, Generic, TypeVar, get_args, get_origin

from evo.common import IContext
from evo.common.interfaces import IFeedback
from evo.common.utils import NoFeedback, Retry, split_feedback
from pydantic import BaseModel

from evo.compute.client import JobClient

__all__ = [
    "TParams",
    "TResult",
    "TResultModel",
    "TaskRegistry",
    "TaskRunner",
    "run_tasks",
]


TParams = TypeVar("TParams", bound=BaseModel)
TResultModel = TypeVar("TResultModel", bound=BaseModel)
TResult = TypeVar("TResult")


def _get_generic_args(cls: type) -> tuple[type, type, type] | None:
    """Extract (TParams, TResultModel, TResult) from a TaskRunner subclass's generic bases."""
    for base in inspect.getmro(cls):
        for orig_base in getattr(base, "__orig_bases__", ()):
            if get_origin(orig_base) is TaskRunner:
                args = get_args(orig_base)
                if len(args) == 3:
                    return args[0], args[1], args[2]
    return None


class TaskRegistry:
    """Registry mapping parameter types to their TaskRunner subclasses.

    This is a singleton that stores the mapping from parameter class types
    to their corresponding TaskRunner subclass.
    """

    _instance: TaskRegistry | None = None
    _runners: dict[type[TParams], type[TaskRunner[TParams, TResultModel, TResult]]]

    def __new__(cls) -> TaskRegistry:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._runners = {}
        return cls._instance

    def register(self, param_type: type[TParams], runner_cls: type[TaskRunner[TParams, TResultModel, TResult]]) -> None:
        """Register a TaskRunner subclass for a parameter type.

        Args:
            param_type: The parameter class (e.g., KrigingParameters).
            runner_cls: The TaskRunner subclass that handles this parameter type.
        """
        self._runners[param_type] = runner_cls

    def get_runner(self, param_type: type[TParams]) -> type[TaskRunner[TParams, TResultModel, TResult]] | None:
        """Get the TaskRunner subclass for a parameter type.

        Args:
            param_type: The parameter class to look up.

        Returns:
            The registered TaskRunner subclass, or None if not found.
        """
        return self._runners.get(param_type)

    def get_runner_for_params(self, params: TParams) -> type[TaskRunner[TParams, TResultModel, TResult]]:
        """Get the TaskRunner subclass for a parameter instance.

        Args:
            params: A parameter object instance.

        Returns:
            The registered TaskRunner subclass.

        Raises:
            TypeError: If no runner is registered for the parameter type.
        """
        param_type = type(params)
        runner_cls = self._runners.get(param_type)
        if runner_cls is None:
            registered = ", ".join(t.__name__ for t in self._runners.keys())
            raise TypeError(
                f"No task runner registered for parameter type '{param_type.__name__}'. "
                f"Registered types: {registered or 'none'}"
            )
        return runner_cls

    def clear(self) -> None:
        """Clear all registered runners (mainly for testing)."""
        self._runners.clear()


# Global registry instance
_registry = TaskRegistry()


class TaskRunner(ABC, Generic[TParams, TResultModel, TResult]):
    """Base class for compute task runners.

    Subclass with concrete ``TParams`` and ``TResult`` type arguments and provide
    ``topic`` and ``task`` as class keyword arguments.  The subclass is
    automatically registered with the :class:`TaskRegistry`.

    Instances are awaitable: ``await runner`` submits the job, waits for
    completion, and returns the typed result with the execution context attached.

    Example — defining a runner::

        class KrigingRunner(
            TaskRunner[KrigingParameters, KrigingResult],
            topic="geostatistics",
            task="kriging",
        ): ...

    Example — advanced direct usage::

        result = await KrigingRunner(context, params, preview=True)
        df = await result.to_dataframe()
    """

    topic: ClassVar[str]
    """The compute topic (e.g. ``"geostatistics"``)."""

    task: ClassVar[str]
    """The task name within the topic (e.g. ``"kriging"``)."""

    params_type: ClassVar[type[TParams]]
    """The Pydantic parameters model, extracted automatically from the generic args."""

    result_model_type: ClassVar[type[TResultModel]]
    """The Pydantic result model, extracted automatically from the generic args."""

    result_type: ClassVar[type[TResult]]
    """The result type returned by the runner, typically a wrapper around the result model with convenience methods."""

    def __init_subclass__(cls, *, topic: str = "", task: str = "", **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        # Skip registration for partially-specialised intermediates that don't
        # provide topic/task (e.g. abstract base helpers).
        if not topic or not task:
            return

        cls.topic = topic
        cls.task = task

        generic_args = _get_generic_args(cls)
        if generic_args is None:
            raise TypeError(
                f"{cls.__name__} must specify generic type arguments: "
                f"class {cls.__name__}(TaskRunner[MyParams, MyResultModel, MyResult], topic=..., task=...)"
            )

        cls.params_type, cls.result_model_type, cls.result_type = generic_args

        # Auto-register with the global registry
        _registry.register(cls.params_type, cls)

    def __init__(
        self,
        context: IContext,
        params: TParams,
        *,
        preview: bool = False,
        polling_interval_seconds: float = 0.5,
        retry: Retry | None = None,
        fb: IFeedback = NoFeedback,
    ) -> None:
        self._context = context
        self._params = params
        self._preview = preview
        self._polling_interval_seconds = polling_interval_seconds
        self._retry = retry
        self._fb = fb

    @abstractmethod
    async def _get_result(self, raw_result: TResultModel) -> TResult: ...

    async def __call__(self) -> TResult:
        """Submit the task, wait for completion, and return the typed result."""
        connector = self._context.get_connector()
        org_id = self._context.get_org_id()

        job = await JobClient.submit(
            connector=connector,
            org_id=org_id,
            topic=self.topic,
            task=self.task,
            parameters=self._params.model_dump(mode="json", by_alias=True, exclude_none=True),
            result_type=self.result_model_type,
            preview=self._preview,
        )

        result = await job.wait_for_results(
            polling_interval_seconds=self._polling_interval_seconds,
            retry=self._retry,
            fb=self._fb,
        )

        return await self._get_result(result)

    def __await__(self) -> Generator[Any, None, TResult]:
        """Make the runner directly awaitable: ``result = await Runner(ctx, params)``."""
        return self().__await__()


class _CompletionTrackingFeedback(IFeedback):
    """Wraps a sub-task feedback to suppress per-job messages and report aggregate completion."""

    def __init__(self, inner: IFeedback, on_complete: Callable[[], str]) -> None:
        self._inner = inner
        self._on_complete = on_complete
        self._done = False

    def progress(self, progress: float, message: str | None = None) -> None:
        if progress >= 1.0 and not self._done:
            self._done = True
            self._inner.progress(1.0, self._on_complete())
        else:
            # Forward progress but suppress per-job messages (e.g. "Waiting on remote job...")
            self._inner.progress(progress)


async def run_tasks(
    context: IContext,
    parameters: list[TParams],
    *,
    fb: IFeedback = NoFeedback,
    preview: bool = False,
) -> list[TResult]:
    """Run multiple tasks concurrently, dispatching based on parameter types.

    This function looks up the appropriate runner for each parameter based on
    its type, allowing different task types to be run together.

    Args:
        context: The context providing connector and org_id.
        parameters: List of parameter objects (can be mixed types).
        fb: Feedback interface for progress updates.
        preview: If True, sets the ``API-Preview: opt-in`` header on requests.
            Required for tasks that are still in preview. Defaults to False.

    Returns:
        List of results in the same order as the input parameters.

    Raises:
        TypeError: If any parameter type doesn't have a registered runner.

    Example:
        >>> results = await run_tasks(manager, [
        ...     KrigingParameters(...),
        ...     SimulationParameters(...),  # future task type
        ... ], preview=True)
    """
    if len(parameters) == 0:
        return []

    total = len(parameters)

    # Validate all parameters have registered runners upfront
    runner_classes = []
    for params in parameters:
        runner_cls = _registry.get_runner_for_params(params)
        runner_classes.append(runner_cls)

    # Split feedback across tasks — each sub-feedback maps its 0→1 progress
    # to an equal slice of the parent's 0→1 range.
    per_task_fb = split_feedback(fb, [1.0] * total)

    # Track completion count for aggregate messages
    done_count = 0

    def _on_task_complete() -> str:
        nonlocal done_count
        done_count += 1
        if done_count >= total:
            return f"Completed {total}/{total}"
        return f"Completed {done_count}/{total}..."

    # Wrap each sub-feedback to suppress per-job messages and show completion count
    tracked_fbs = [_CompletionTrackingFeedback(sub_fb, _on_task_complete) for sub_fb in per_task_fb]
    tasks = [
        runner_cls(context, params, preview=preview, fb=task_fb)
        for runner_cls, params, task_fb in zip(runner_classes, parameters, tracked_fbs, strict=True)
    ]

    return await asyncio.gather(*tasks)
