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
import contextlib
import logging
from abc import ABC, abstractmethod

from ..exceptions import RetryError

__all__ = [
    "BackoffExponential",
    "BackoffIncremental",
    "BackoffLinear",
    "BackoffMethod",
    "Retry",
    "RetryHandler",
]


class BackoffMethod(ABC):
    """Abstract base class for backoff methods.

    The only state is the configuration for the backoff method. One instance may be used for multiple requests.
    """

    def __init__(self, backoff_factor: int | float, max_delay: int | float = -1) -> None:
        """
        :param backoff_factor: The backoff factor to apply. This informs the size of the delay between attempts.
        :param max_delay: The maximum delay to apply. If the calculated delay is greater than this value, the maximum
            delay is used instead. If this value is negative, no maximum delay is applied.
        """
        self._backoff_factor = backoff_factor
        self._max_delay = max_delay

    @abstractmethod
    def _calculate_backoff_time(self, attempt_number: int) -> int | float:
        """Calculate the backoff time as a function of the number of attempts.

        :param attempt_number: The number of attempts that have been made.

        returns: Time to backoff for the current number of attempts
        """
        ...

    def get_backoff_time(self, attempt_number: int) -> int | float:
        """Get the backoff time as a function of the number of attempts.

        :param attempt_number: The number of attempts that have been made.

        returns: Time to backoff for the current number of attempts, or the maximum delay if the calculated delay is
            greater.
        """
        t = self._calculate_backoff_time(attempt_number)
        return t if self._max_delay < 0 else min(t, self._max_delay)


class BackoffLinear(BackoffMethod):
    """Linear retry delay."""

    def _calculate_backoff_time(self, attempt_number: int) -> int | float:
        return self._backoff_factor


class BackoffIncremental(BackoffMethod):
    """Incremental retry delay."""

    def _calculate_backoff_time(self, attempt_number: int) -> int | float:
        return self._backoff_factor * attempt_number


class BackoffExponential(BackoffMethod):
    """Exponential retry delay."""

    def _calculate_backoff_time(self, attempt_number: int) -> int | float:
        return self._backoff_factor * (2**attempt_number)


class _RetryIterator:
    def __init__(self, logger: logging.Logger, max_attempts: int, backoff_method: BackoffMethod) -> None:
        self.__logger = logger
        self.__max_attempts = max_attempts
        self.__backoff_method = backoff_method
        self.__current_handler: RetryHandler | None = None
        self.__errors = []

    def add_error(self, error: Exception) -> None:
        self.__logger.error(error)
        self.__logger.debug(f"{self.__current_handler} failed")
        self.__errors.append(error)
        if self.__current_handler._number >= self.__max_attempts:
            raise RetryError("Retry failed", self.__errors)

    async def __anext__(self) -> RetryHandler:
        if self.__current_handler is None:
            # Use a 1-based sequence for readability.
            next_handler = RetryHandler(self, 1)
        elif self.__current_handler.succeeded:
            raise StopAsyncIteration
        elif self.__current_handler._number >= self.__max_attempts:
            raise RetryError("Retry failed", self.__errors)
        else:
            # Calculate delay based on previous attempt.
            delay = self.__backoff_method.get_backoff_time(self.__current_handler._number)
            self.__logger.debug(f"Waiting {delay}s")
            await asyncio.sleep(delay)
            next_handler = self.__current_handler._next()

        self.__current_handler = next_handler

        return next_handler


class RetryHandler:
    """RetryHandler for retry attempts.

    A RetryHandler is created by a Retry object and is used to handle a single retry attempt. This handler should be
    used to suppress errors raised during the retry attempt, by using the suppress_errors context manager. If no errors
    are suppressed, the retry attempt is considered to have succeeded.
    """

    def __init__(self, iterator: _RetryIterator, attempt_number: int) -> None:
        """
        :param iterator: The iterator that created this handler.
        :param attempt_number: The number of preceding attempts.
        """
        self.__iterator = iterator
        self.__number = attempt_number
        self.__exception: Exception | None = None

    @contextlib.contextmanager
    def suppress_errors(self, excs: type[BaseException] | tuple[type[BaseException, ...]] | None = None) -> None:
        """Context manager to suppress errors raised during a retry attempt.

        :param excs: The exception types to suppress, defaults to all exceptions.
        """
        try:
            yield
        except Exception as exc:
            if excs is None or isinstance(exc, excs):
                self.set_exception(exc)
            else:
                raise

    def set_exception(self, exc: Exception) -> None:
        """Set the exception handled during the retry attempt.
        :param exc: The exception to set.
        """
        self.__exception = exc
        self.__iterator.add_error(exc)

    @property
    def exception(self) -> Exception | None:
        """The exception raised during the retry attempt, if any."""
        return self.__exception

    def reset_counter(self) -> None:
        """Reset the attempt counter."""
        self.__number = 1

    @property
    def succeeded(self) -> bool:
        """Whether the retry attempt succeeded, determined by whether an exception was suppressed."""
        return self.__exception is None

    @property
    def failed(self) -> bool:
        """Whether the retry attempt failed."""
        return self.__exception is not None

    @property
    def _number(self) -> int:
        return self.__number

    def _next(self) -> RetryHandler:
        return RetryHandler(self.__iterator, self.__number + 1)

    def __str__(self) -> str:
        return f"Retry attempt #{self.__number}"


class Retry:
    """AsyncIterable retry implementation for handling exceptions in retryable logic.

    A single Retry object can be used to handle multiple requests. The only state in this class is the configuration.

    Usage::
        retry = Retry(logger=logging.getLogger(__name__), max_attempts=3, backoff_method=BackoffLinear(1))
        async for handler in retry:
            # do some things
            ...
            with handler.suppress_errors():
                # make request
                ...
            if handler.failed:
                # do some cleanup
                ...
    """

    def __init__(
        self,
        logger: logging.Logger,
        max_attempts: int = 3,
        backoff_method: BackoffMethod = BackoffExponential(backoff_factor=2),
    ) -> None:
        """Initialise a Retry object used when retrying after failures.

        :param logger: Logger instance for logging retry attempts.
        :param max_attempts: Maximum number of times to retry.
        :param backoff_method: Backoff method to apply.
        """
        if max_attempts < 1:
            raise ValueError("max_attempts must be greater than 0")
        self.__logger = logger
        self.__max_attempts = max_attempts
        self.__backoff_method = backoff_method

    def __aiter__(self) -> _RetryIterator:
        return _RetryIterator(self.__logger, self.__max_attempts, self.__backoff_method)
