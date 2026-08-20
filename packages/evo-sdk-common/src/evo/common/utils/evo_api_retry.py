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

import asyncio
import contextlib
import logging
import typing as tp
from collections.abc import Set
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from random import random

from evo.common.exceptions import EvoAPIException, RetryError
from evo.common.utils.retry import BackoffMethod


class EvoAPIRetryHandler:
    """Handler for a single retry attempt in EvoAPIRetry."""

    def __init__(self, logger: logging.Logger, attempt: int):
        self.logger = logger
        self.attempt = attempt
        self.exception: Exception | None = None

    @contextlib.contextmanager
    def suppress_errors(
        self, excs: type[BaseException] | tuple[type[BaseException], ...] | None = None
    ) -> tp.Generator[None, tp.Any, tp.Any]:
        """Suppress errors raised during a retry attempt.

        :param excs: Additional exception types to suppress
        """
        try:
            yield
        except Exception as exc:
            if isinstance(exc, EvoAPIException) or (excs is not None and isinstance(exc, excs)):
                self.exception = exc
            else:
                raise

    def set_exception(self, exc: Exception) -> None:
        """Set the exception handled during the retry attempt.

        :param exc: The exception to set.
        """
        self.exception = exc

    @property
    def succeeded(self) -> bool:
        return self.exception is None

    @property
    def failed(self) -> bool:
        return self.exception is not None


def _parse_retry_after(logger: logging.Logger, retry_after_str: str | None) -> float | None:
    if retry_after_str is None or retry_after_str == "":
        return None
    try:
        return float(retry_after_str)
    except ValueError:
        try:
            retry_after = parsedate_to_datetime(retry_after_str)
            if retry_after.tzinfo is None:
                retry_after = retry_after.replace(tzinfo=timezone.utc)
            return (retry_after - datetime.now(timezone.utc)).total_seconds()
        except (TypeError, ValueError, IndexError):
            logger.info("Failed to parse Retry-After header: %s", repr(retry_after_str))
            return None


class EvoAPIRetry:
    """EvoAPIException-aware retry implementation

    .. note:: Retrying requests that have different outcomes each time they are called can lead to unexpected results such
              as duplicate transactions or data corruption. Although operations such as GET, PUT and DELETE are generally safe to retry,
              it is the responsibility of the caller to ensure that retrying is safe. Consult the API documentation or contact
              the API provider for guidance on which operations are safe to retry.

    Usage::
        retry = EvoAPIRetry(logger=logging.getLogger(__name__), max_attempts=3, backoff_method=BackoffLinear(1))
        async for handler in retry(): # mandatory to call the retry object
            # do some things
            ...
            with handler.suppress_errors(): # mandatory to suppress EvoAPIException
                # make request
                ...
            if handler.failed:
                # do some cleanup
                ...
    """

    def __init__(
        self,
        logger: logging.Logger,
        max_attempts: int,
        backoff_method: BackoffMethod,
        statuses: Set[int] = frozenset({429, 503}),
    ) -> None:
        """Initialise a EvoAPIRetry object used when retrying after failures.

        :param logger: Logger instance for logging retry attempts.
        :param max_attempts: Maximum number of times to retry.
        :param backoff_method: Backoff method to apply.
        :param statuses: HTTP status codes that should trigger a retry.
        """
        if max_attempts < 1:
            raise ValueError("max_attempts must be greater than 0")
        if len(statuses) == 0:
            raise ValueError("statuses must contain at least one status code")

        self._statuses = statuses
        self._logger = logger
        self._max_attempts = max_attempts
        self._backoff_method = backoff_method

    async def _recover(self, handler: EvoAPIRetryHandler) -> None:
        """Recover from a failed attempt, applying backoff and jitter before the next attempt."""

        retry_after: float | None = None
        if isinstance(handler.exception, EvoAPIException) and handler.exception.status in self._statuses:
            headers = handler.exception.headers if handler.exception.headers else {}
            retry_after = _parse_retry_after(self._logger, headers.get("Retry-After", "").strip())

        delay = self._backoff_method.get_backoff_time(handler.attempt)
        if retry_after is not None:
            delay = max(delay, retry_after)

        delay += random()  # jitter
        self._logger.debug(f"Waiting {delay}s")
        await asyncio.sleep(delay)

    async def __call__(self) -> tp.AsyncGenerator[EvoAPIRetryHandler, None]:
        """Returns an async generator that yields EvoAPIRetryHandler objects for each retry attempt."""
        errors: list[Exception] = []

        for attempt in range(1, self._max_attempts + 1):
            handler = EvoAPIRetryHandler(self._logger, attempt)
            yield handler

            if handler.succeeded:
                break
            else:
                assert handler.exception is not None
                errors.append(handler.exception)
                if attempt < self._max_attempts:
                    await self._recover(handler)
        else:
            raise RetryError("Retry failed", errors)
