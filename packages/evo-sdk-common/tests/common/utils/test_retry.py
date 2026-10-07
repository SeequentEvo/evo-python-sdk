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

import itertools
import logging
import time
import unittest
from unittest import mock

from evo.common.exceptions import RetryError
from evo.common.test_tools import long_test
from evo.common.utils import BackoffExponential, BackoffIncremental, BackoffLinear, Retry

logger = logging.getLogger(__name__)


class _TestException1(Exception): ...


class _TestException2(Exception): ...


class _TestException3(Exception): ...


class TestBackoffMethods(unittest.TestCase):
    def test_exponential_backoff(self) -> None:
        for factor in range(-5, 5):
            backoff = BackoffExponential(backoff_factor=factor)
            for attempt_number in range(1, 5):
                self.assertEqual(factor * (2**attempt_number), backoff.get_backoff_time(attempt_number))

    def test_max_exponential_backoff(self) -> None:
        backoff = BackoffExponential(backoff_factor=1, max_delay=4)
        self.assertEqual(2, backoff.get_backoff_time(1))
        self.assertEqual(4, backoff.get_backoff_time(10))

    def test_incremental_backoff(self) -> None:
        for factor in range(-5, 5):
            backoff = BackoffIncremental(backoff_factor=factor)
            for attempt_number in range(1, 5):
                self.assertEqual(factor * attempt_number, backoff.get_backoff_time(attempt_number))

    def test_max_incremental_backoff(self) -> None:
        backoff = BackoffIncremental(backoff_factor=1, max_delay=4)
        self.assertEqual(1, backoff.get_backoff_time(1))
        self.assertEqual(4, backoff.get_backoff_time(10))

    def test_linear_backoff(self) -> None:
        for factor in range(-5, 5):
            backoff = BackoffLinear(backoff_factor=factor)
            for attempt_number in range(1, 5):
                self.assertEqual(factor, backoff.get_backoff_time(attempt_number))

    def test_max_linear_backoff(self) -> None:
        backoff = BackoffLinear(backoff_factor=1, max_delay=4)
        self.assertEqual(1, backoff.get_backoff_time(10))

        # This is a bit of a weird case, but possible ¯\_(ツ)_/¯.
        backoff = BackoffLinear(backoff_factor=10, max_delay=4)
        self.assertEqual(4, backoff.get_backoff_time(10))


class TestRetry(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.retry = Retry(logger, max_attempts=5, backoff_method=BackoffIncremental(1))

    @mock.patch("asyncio.sleep", spec_set=True)
    async def test_successful_attempt(self, mock_sleep: mock.MagicMock) -> None:
        async for _ in self.retry:
            pass

        mock_sleep.assert_not_called()

    @mock.patch("asyncio.sleep", spec_set=True)
    async def test_max_attempts(self, mock_sleep: mock.MagicMock) -> None:
        with self.assertRaises(RetryError):
            async for handler in self.retry:
                with handler.suppress_errors():
                    raise Exception("Test exception")

        self.assertEqual(4, mock_sleep.call_count)  # 5 attempts == 4 sleeps.
        mock_sleep.assert_has_calls([mock.call(1), mock.call(2), mock.call(3), mock.call(4)])

    @mock.patch("asyncio.sleep", spec_set=True)
    async def test_reset_counter(self, mock_sleep: mock.MagicMock) -> None:
        i = 0
        with self.assertRaises(RetryError):
            async for handler in self.retry:
                if (i := i + 1) == 5:
                    handler.reset_counter()

                with handler.suppress_errors():
                    raise Exception("Test exception")

        self.assertEqual(4 + 4, mock_sleep.call_count)
        mock_sleep.assert_has_calls(
            [
                mock.call(1),
                mock.call(2),
                mock.call(3),
                mock.call(4),
                # reset_counter().
                mock.call(1),
                mock.call(2),
                mock.call(3),
                mock.call(4),
            ]
        )

    @mock.patch("asyncio.sleep", spec_set=True)
    async def test_suppress_specific_error(self, mock_sleep: mock.MagicMock) -> None:
        with self.assertRaises(RetryError):
            async for handler in self.retry:
                with handler.suppress_errors(_TestException1):
                    raise _TestException1("Expected exception")

        self.assertEqual(4, mock_sleep.call_count)  # 5 attempts == 4 sleeps.
        mock_sleep.assert_has_calls([mock.call(1), mock.call(2), mock.call(3), mock.call(4)])

    @mock.patch("asyncio.sleep", spec_set=True)
    async def test_suppress_specific_errors(self, mock_sleep: mock.MagicMock) -> None:
        errors = itertools.cycle([_TestException1("Expected exception 1"), _TestException2("Expected exception 2")])
        with self.assertRaises(RetryError):
            async for handler in self.retry:
                with handler.suppress_errors((_TestException1, _TestException2)):
                    raise next(errors)

        self.assertEqual(4, mock_sleep.call_count)  # 5 attempts == 4 sleeps.
        mock_sleep.assert_has_calls([mock.call(1), mock.call(2), mock.call(3), mock.call(4)])

    @mock.patch("asyncio.sleep", spec_set=True)
    async def test_suppress_specific_error_unexpected_not_suppressed(self, mock_sleep: mock.MagicMock) -> None:
        with self.assertRaises(_TestException2):
            async for handler in self.retry:
                with handler.suppress_errors(_TestException1):
                    raise _TestException2("Unexpected exception")

        mock_sleep.assert_not_called()

    @mock.patch("asyncio.sleep", spec_set=True)
    async def test_suppress_specific_errors_unexpected_not_suppressed(self, mock_sleep: mock.MagicMock) -> None:
        with self.assertRaises(_TestException3):
            async for handler in self.retry:
                with handler.suppress_errors((_TestException1, _TestException2)):
                    raise _TestException3("Unexpected exception")

        mock_sleep.assert_not_called()

    @mock.patch("asyncio.sleep", spec_set=True)
    async def test_handler_properties(self, mock_sleep: mock.MagicMock) -> None:
        errors = iter(
            [
                _TestException1("Attempt 1"),
                _TestException1("Attempt 2"),
                _TestException1("Attempt 3"),
                _TestException1("Attempt 4"),
                _TestException1("Attempt 5"),
            ]
        )
        with self.assertRaises(RetryError):
            async for handler in self.retry:
                # No exception suppressed yet.
                self.assertIsNone(handler.exception)
                self.assertTrue(handler.succeeded)
                self.assertFalse(handler.failed)

                this_error = next(errors)
                with handler.suppress_errors():
                    raise this_error

                # Exception suppressed.
                self.assertIs(this_error, handler.exception)
                self.assertFalse(handler.succeeded)
                self.assertTrue(handler.failed)

        self.assertEqual(4, mock_sleep.call_count)  # 5 attempts == 4 sleeps.
        mock_sleep.assert_has_calls([mock.call(1), mock.call(2), mock.call(3), mock.call(4)])

    @long_test
    async def test_actual_delay(self) -> None:
        expect_end = time.perf_counter() + 1 + 2 + 3 + 4
        with self.assertRaises(RetryError):
            async for handler in self.retry:
                with handler.suppress_errors():
                    raise Exception("Test exception")

        actual_end = time.perf_counter()
        self.assertAlmostEqual(expect_end, actual_end, delta=0.1)
