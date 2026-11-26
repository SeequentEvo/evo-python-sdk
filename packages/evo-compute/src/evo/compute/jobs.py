"""Async lifecycle access to a submitted compute job.

``TaskJob`` keeps the platform job and its async result converter together. Submission
does not wait; callers can inspect or cancel the job before collecting hydrated results.
Feedback supplied by the client is used while waiting unless ``results`` overrides it.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Generic, TypeVar
from uuid import UUID

from evo.common import IContext
from evo.common.interfaces import IFeedback
from evo.common.utils import NoFeedback, Retry

from .client import JobClient
from .data import JobProgress

__all__ = [
    "TaskJob",
]

TResult = TypeVar("TResult")
TMapped = TypeVar("TMapped")


class TaskJob(Generic[TResult]):
    """A task accepted by the platform, with async status, cancellation and results.

    :param job: The submitted platform job.
    :param convert_result: Async conversion from the platform payload to the result type.
    :param fb: Where progress is reported while waiting for results.
    """

    def __init__(
        self,
        job: JobClient[dict],
        convert_result: Callable[[dict[str, Any]], Awaitable[TResult]],
        fb: IFeedback = NoFeedback,
    ) -> None:
        self._job = job
        self._convert_result = convert_result
        self._fb = fb

    @staticmethod
    async def from_url(
        context: IContext,
        url: str,
        *,
        preview: bool = False,
        fb: IFeedback = NoFeedback,
    ) -> TaskJob[Any]:
        """Reopen a saved status URL with discovery hydration and native result conversion.

        The URL must belong to the context's hub and organization. Preview is not
        encoded in the URL, so preview jobs require ``preview=True`` again.
        Discovery determines the result type; a URL alone cannot describe it statically.
        """
        from .engine import ComputeClient

        return await ComputeClient(context, fb=fb)._restore_job(url, preview=preview)

    def map(self, convert: Callable[[TResult], Awaitable[TMapped]]) -> TaskJob[TMapped]:
        """Convert results asynchronously while preserving the same job and wait options.

        Conversion runs only when ``results`` is awaited. Conversion errors propagate
        to the caller, just like errors from the platform.
        """

        async def mapped(payload: dict[str, Any]) -> TMapped:
            return await convert(await self._convert_result(payload))

        return TaskJob(self._job, mapped, self._fb)

    @property
    def id(self) -> UUID:
        """The platform's job identifier."""
        return self._job.id

    @property
    def url(self) -> str:
        """The status URL that ``TaskJob.from_url`` can reopen."""
        return self._job.url

    @property
    def topic(self) -> str:
        """The topic, as the platform spells it."""
        return self._job.topic

    @property
    def task(self) -> str:
        """The task name, as the platform spells it."""
        return self._job.task

    def __repr__(self) -> str:
        return f"<compute job {self.topic}.{self.task} {self.id}>"

    async def status(self) -> JobProgress:
        """Ask the platform for the current status without waiting for completion."""
        return await self._job.get_status()

    async def cancel(self) -> None:
        """Request cancellation; poll ``status`` to observe its completion."""
        await self._job.cancel()

    async def results(
        self,
        *,
        fb: IFeedback | None = None,
        polling_interval_seconds: float = 5.0,
        retry: Retry | None = None,
    ) -> TResult:
        """Wait for completion and convert the payload to the declared result type.

        :param fb: Override the client's feedback for this wait.
        :param polling_interval_seconds: Delay between status checks.
        :param retry: Retry strategy for failed status checks; ``None`` uses the default.

        :raises JobError: If the platform job failed.
        """
        payload = await self._job.wait_for_results(
            polling_interval_seconds=polling_interval_seconds,
            retry=retry,
            fb=self._fb if fb is None else fb,
        )
        return await self._convert_result(payload)
