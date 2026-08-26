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

import logging
import unittest
from datetime import datetime, timedelta, timezone
from unittest import mock

from evo.common.data import HTTPHeaderDict
from evo.common.exceptions import EvoAPIException, RetryError
from evo.common.utils.evo_api_retry import EvoAPIRetry, EvoAPIRetryHandler
from evo.common.utils.retry import BackoffIncremental

logger = logging.getLogger(__name__)


class _TestException(Exception): ...


class TestEvoAPIRetryHandler(unittest.TestCase):
    def test_suppress_evo_api_exception(self) -> None:
        handler = EvoAPIRetryHandler(logger, 1)
        exception = EvoAPIException(503, "Service unavailable", None, None)

        with handler.suppress_errors():
            raise exception

        self.assertIs(exception, handler.exception)
        self.assertFalse(handler.succeeded)
        self.assertTrue(handler.failed)

    def test_suppress_additional_exception(self) -> None:
        handler = EvoAPIRetryHandler(logger, 1)
        exception = _TestException("Expected exception")

        with handler.suppress_errors(_TestException):
            raise exception

        self.assertIs(exception, handler.exception)

    def test_unexpected_exception_is_not_suppressed(self) -> None:
        handler = EvoAPIRetryHandler(logger, 1)

        with self.assertRaises(_TestException):
            with handler.suppress_errors():
                raise _TestException("Unexpected exception")

        self.assertTrue(handler.succeeded)

    def test_set_exception(self) -> None:
        handler = EvoAPIRetryHandler(logger, 1)
        exception = _TestException("Expected exception")

        handler.set_exception(exception)

        self.assertIs(exception, handler.exception)
        self.assertTrue(handler.failed)


class TestEvoAPIRetry(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.retry = EvoAPIRetry(logger, max_attempts=5, backoff_method=BackoffIncremental(1))

    async def test_successful_attempt(self) -> None:
        with mock.patch("asyncio.sleep", spec_set=True) as mock_sleep:
            async for _ in self.retry():
                pass

        mock_sleep.assert_not_called()

    @mock.patch("evo.common.utils.evo_api_retry.random", return_value=0)
    @mock.patch("asyncio.sleep", spec_set=True)
    async def test_max_attempts(self, mock_sleep: mock.MagicMock, mock_random: mock.MagicMock) -> None:
        with self.assertRaises(RetryError):
            async for handler in self.retry():
                with handler.suppress_errors():
                    raise EvoAPIException(503, "Service unavailable", None, None)

        self.assertEqual(4, mock_sleep.call_count)  # 5 attempts == 4 sleeps.
        mock_sleep.assert_has_calls([mock.call(1), mock.call(2), mock.call(3), mock.call(4)])
        self.assertEqual(4, mock_random.call_count)

    @mock.patch("evo.common.utils.evo_api_retry.random", return_value=0)
    @mock.patch("asyncio.sleep", spec_set=True)
    async def test_retry_after_sets_minimum_delay(
        self, mock_sleep: mock.MagicMock, mock_random: mock.MagicMock
    ) -> None:
        retry = EvoAPIRetry(logger, max_attempts=2, backoff_method=BackoffIncremental(1))

        async for handler in retry():
            with handler.suppress_errors():
                if handler.attempt == 1:
                    raise EvoAPIException(429, "Too many requests", None, HTTPHeaderDict({"Retry-After": "3"}))

        mock_sleep.assert_called_once_with(3)
        mock_random.assert_called_once()

    async def test_retry_after_date_sets_minimum_delay(self) -> None:
        retry = EvoAPIRetry(logger, max_attempts=2, backoff_method=BackoffIncremental(1))
        now = datetime(2026, 8, 20, 12, 0, 0, tzinfo=timezone.utc)
        retry_at = datetime(2100, 1, 1, 0, 0, 0, tzinfo=timezone(-timedelta(hours=5)))

        with (
            mock.patch("asyncio.sleep", spec_set=True) as mock_sleep,
            mock.patch("evo.common.utils.evo_api_retry.datetime", wraps=datetime) as mock_datetime,
            mock.patch("evo.common.utils.evo_api_retry.random", return_value=0) as mock_random,
        ):
            mock_datetime.now.side_effect = lambda timezone_=None: (
                now if timezone_ is None else now.astimezone(timezone_)
            )

            async for handler in retry():
                with handler.suppress_errors():
                    if handler.attempt == 1:
                        raise EvoAPIException(
                            429,
                            "Too many requests",
                            None,
                            HTTPHeaderDict({"Retry-After": "Fri, 01 Jan 2100 00:00:00 EST"}),
                        )

        mock_sleep.assert_called_once_with((retry_at - now).total_seconds())
        mock_random.assert_called_once()

    @mock.patch("evo.common.utils.evo_api_retry.random", return_value=0)
    @mock.patch("asyncio.sleep", spec_set=True)
    async def test_retry_after_is_ignored_for_unconfigured_status(
        self, mock_sleep: mock.MagicMock, mock_random: mock.MagicMock
    ) -> None:
        retry = EvoAPIRetry(logger, max_attempts=2, backoff_method=BackoffIncremental(1), statuses=frozenset({503}))

        async for handler in retry():
            with handler.suppress_errors():
                if handler.attempt == 1:
                    raise EvoAPIException(429, "Too many requests", None, HTTPHeaderDict({"Retry-After": "3"}))

        mock_sleep.assert_called_once_with(1)
        mock_random.assert_called_once()

    def test_invalid_max_attempts(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_attempts must be greater than 0"):
            EvoAPIRetry(logger, max_attempts=0, backoff_method=BackoffIncremental(1))

    def test_empty_statuses(self) -> None:
        with self.assertRaisesRegex(ValueError, "statuses must contain at least one status code"):
            EvoAPIRetry(logger, max_attempts=1, backoff_method=BackoffIncremental(1), statuses=frozenset())
