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
import copy
import json
import re
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any, Generic, TypeVar
from uuid import UUID

from evo.common import APIConnector, HTTPResponse
from evo.common.exceptions import UnknownResponseError
from evo.common.interfaces import IFeedback
from evo.common.utils import NoFeedback, Retry
from pydantic import TypeAdapter, ValidationError

from evo import logging

from .data import JobProgress, JobStatusEnum
from .endpoints import JobsApi, TasksApi
from .endpoints.models import CompletedJobResponse
from .exceptions import JobError, JobPendingError

logger = logging.getLogger("compute.client")

__all__ = [
    "JobClient",
]

T_Result = TypeVar("T_Result")


@contextmanager
def _validating_pydantic_model(ctx: HTTPResponse) -> Iterator[None]:
    """Context manager to handle Pydantic model validation errors.

    :param ctx: The HTTP response context.

    :raises UnknownResponseError: If the managed context raises a pydantic ValidationError.
    """
    try:
        yield
    except ValidationError as e:
        logger.error(f"Failed to validate response: {e}", exc_info=e)

        content = ctx.data
        try:
            content = content.decode("utf-8")  # Raises UnicodeDecodeError.
            content = json.loads(content)  # Raises json.JSONDecodeError.
        except (UnicodeDecodeError, json.JSONDecodeError):
            pass  # content will fall back to the most specific valid value.

        raise UnknownResponseError(
            status=ctx.status,
            reason=ctx.reason,
            content=content,
            headers=ctx.headers,
        ) from e


RE_STATUS_URL = re.compile(r"compute/orgs/(?P<org_id>[^/]+)/(?P<topic>[^/]+)/(?P<task>[^/]+)/(?P<job_id>[^/]+)/status")


class JobClient(Generic[T_Result]):
    """Client for managing a submitted compute task (a job)."""

    def __init__(
        self,
        connector: APIConnector,
        org_id: UUID,
        topic: str,
        task: str,
        job_id: UUID,
        result_type: type[T_Result] = dict,
    ) -> None:
        """
        :param connector: The API connector to use.
        :param org_id: The organization ID.
        :param topic: The topic of the task.
        :param task: The task to be executed.
        :param job_id: The job ID.
        :param result_type: The type to validate the result against.
        """
        self._connector = connector
        self._org_id = org_id
        self._topic = topic
        self._task = task
        self._job_id = job_id
        self._url = connector.base_url + f"compute/orgs/{self._org_id}/{self._topic}/{self._task}/{self._job_id}/status"
        self._type_adapter = TypeAdapter(result_type)
        self._mutex = asyncio.Lock()
        self._results: T_Result | JobError | None = None

    @property
    def id(self) -> UUID:
        """The job id."""
        return self._job_id

    @property
    def topic(self) -> str:
        """The topic of the task."""
        return self._topic

    @property
    def task(self) -> str:
        """The task."""
        return self._task

    @property
    def url(self) -> str:
        """The status URL of the job."""
        return self._url

    def __repr__(self) -> str:
        return self.url

    @staticmethod
    def from_url(connector: APIConnector, url: str, result_type: type[T_Result] = dict) -> JobClient[T_Result]:
        """Create a job client from a status URL.

        The URL hostname must match the connector base URL.

        :param connector: The API connector to use.
        :param url: The status URL of a submitted job.
        :param result_type: The type to validate the result against.

        :return: A client for managing the referenced job.
        """
        if not url.startswith(connector.base_url):
            raise ValueError("The job URL does not match the connector base URL.")

        if match := RE_STATUS_URL.fullmatch(url.removeprefix(connector.base_url)):
            path_params = match.groupdict()
            for key in path_params:
                if key.endswith("_id"):
                    try:
                        path_params[key] = UUID(path_params[key])
                    except ValueError:
                        raise ValueError(f"Invalid {key.removesuffix('_id')} ID in URL: {url}") from None

        return JobClient(connector=connector, **path_params, result_type=result_type)

    @staticmethod
    async def submit(
        connector: APIConnector,
        org_id: UUID,
        topic: str,
        task: str,
        parameters: Mapping[str, Any],
        result_type: type[T_Result] = dict,
    ) -> JobClient[T_Result]:
        """Trigger an asynchronous task within a specific topic with the given parameters.

        :param connector: The API connector to use.
        :param topic: The topic of the task.
        :param task: The task to be executed.
        :param parameters: The parameters for the task.
        :param result_type: The type to validate the result against.

        :return: The job that was created.

        :raises UnknownResponseError: If the Location header is missing or invalid.
        """
        async with connector:
            response = await TasksApi(connector).execute_task(
                org_id=str(org_id),
                topic=topic,
                task=task,
                execute_task_request={"parameters": dict(parameters)},
            )

        # Location header is the status endpoint of the created job.
        # i.e., compute/orgs/{org_id}/{topic}/{task}/{job_id}/status
        try:
            job_url = response.headers["Location"]
            job_url = connector.base_url + job_url.removeprefix(connector.base_url).removeprefix("/")
            return JobClient.from_url(connector, job_url, result_type)
        except (KeyError, ValueError):
            raise UnknownResponseError(
                status=response.status, reason=response.reason, content=None, headers=response.headers
            )

    async def get_status(self) -> JobProgress:
        """Get the status of the job.

        :return: The job progress.
        """
        async with self._connector:
            response = await JobsApi(self._connector).get_job_status(
                org_id=self._org_id,
                topic=self._topic,
                task=self._task,
                job_id=self._job_id,
            )

        if response.error:
            error = JobError(
                status=response.error.status,
                reason=None,
                content=response.error.model_dump(by_alias=True, exclude_unset=True, exclude_defaults=True),
                headers=None,
            )
        else:
            error = None

        return JobProgress(
            status=JobStatusEnum(response.status.value),
            progress=response.progress,
            message=response.message,
            error=error,
        )

    async def _get_results(self) -> T_Result | JobError:
        """Get the results of the job if they are not already cached.

        The results are cached after the first successful call.

        :return: The job results or error.

        :raises UnknownResponseError: If the response or results fail validation.
        :raises JobPendingError: If the job is still pending.
        """
        async with self._mutex:
            if self._results is None:
                # Request the results from the API.
                async with self._connector:
                    response = await JobsApi(self._connector).get_job_results(
                        org_id=self._org_id,
                        topic=self._topic,
                        task=self._task,
                        job_id=self._job_id,
                    )

                with _validating_pydantic_model(response):
                    job = CompletedJobResponse.model_validate_json(response.data)

                if response.status == 202:
                    # The job is still pending, raise an error.
                    raise JobPendingError(url=self.url, status=job.status.value)

                if error := job.error:
                    # The job failed, save the error.
                    self._results = JobError(
                        status=error.status,
                        reason=None,
                        content=error.model_dump(by_alias=True, exclude_unset=True, exclude_defaults=True),
                        headers=response.headers,
                    )
                else:
                    # The job succeeded, deserialize and save the results.
                    with _validating_pydantic_model(response):
                        # Nitpick: job.results _may_ in fact be `None`, in which case the type adapter will
                        # raise ValidationError and trigger the error handling in the context manager.
                        # At the time of writing, it is unclear whether there are any valid cases where both
                        # the error and results are missing when the job is in a completed state.
                        self._results = self._type_adapter.validate_python(job.results)

            # Return a copy of the cached results.
            if isinstance(self._results, JobError):
                return self._results.copy()
            else:
                return copy.deepcopy(self._results)

    async def get_results(self) -> T_Result:
        """Get the results of the job.

        :return: The results.

        :raises UnknownResponseError: If the response or results fail validation.
        :raises JobPendingError: If the job is still pending.
        :raises JobError: If the job failed.
        """
        if isinstance(results := await self._get_results(), JobError):
            raise results.with_traceback(None)
        return results

    async def cancel(self) -> None:
        """Cancel the job."""
        async with self._connector:
            await JobsApi(self._connector).cancel_job(
                org_id=self._org_id,
                topic=self._topic,
                task=self._task,
                job_id=self._job_id,
            )

    async def wait_for_results(
        self, polling_interval_seconds: float = 0.5, retry: Retry | None = None, fb: IFeedback = NoFeedback
    ) -> T_Result:
        """Wait for the job to complete and return the results.

        :param polling_interval_seconds: The interval in seconds between status checks.
        :param retry: A Retry object with a wait strategy. If None, a default Retry is created.
        :param fb: The feedback object to use.

        :return: The results.

        :raises UnknownResponseError: If the response or results fail validation.
        :raises JobPendingError: If the job is still pending.
        :raises JobError: If the job failed.
        """
        if retry is None:
            retry = Retry(logger)

        latest_progress = 0.0
        latest_message = "Waiting on remote job..."

        while True:
            async for handler in retry:
                with handler.suppress_errors():
                    latest = await self.get_status()

            if latest.status in (JobStatusEnum.succeeded, JobStatusEnum.failed, JobStatusEnum.cancelled):
                break

            # Task API returns progress as an integer between 0 and 100
            latest_progress = latest_progress if latest.progress is None else latest.progress * 0.01
            latest_message = latest_message if latest.message is None else latest.message

            fb.progress(latest_progress, latest_message)
            await asyncio.sleep(polling_interval_seconds)

        fb.progress(1.0, "Fetching results...")
        return await self.get_results()
