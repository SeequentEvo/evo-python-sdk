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
import inspect
import threading
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from unittest import IsolatedAsyncioTestCase
from unittest.mock import patch

from parameterized import parameterized

from evo.objects.utils._concurrency import gather_cancelling, run_upload_preparation


class TestUploadConcurrency(IsolatedAsyncioTestCase):
    async def test_gather_overlaps_work_and_preserves_input_order(self):
        started = set()
        all_started = asyncio.Event()
        second_finished = asyncio.Event()

        async def work(index):
            started.add(index)
            if len(started) == 2:
                all_started.set()
            await all_started.wait()
            if index == 0:
                await second_finished.wait()
            else:
                second_finished.set()
            return index

        self.assertEqual([0, 1], await asyncio.wait_for(gather_cancelling(work(0), work(1)), timeout=2))

    async def test_failure_cancels_and_drains_nested_uploads(self):
        started = asyncio.Event()
        release = asyncio.Event()
        cleaned_up = set()
        failure = RuntimeError("upload failed")

        async def upload(index):
            try:
                started.set()
                await release.wait()
            finally:
                await asyncio.sleep(0)
                cleaned_up.add(index)

        async def group(start):
            return await gather_cancelling(*(upload(index) for index in range(start, start + 3)))

        async def fail():
            await started.wait()
            raise failure

        with self.assertRaises(RuntimeError) as caught:
            await gather_cancelling(group(0), group(3), fail())
        self.assertIs(failure, caught.exception)
        self.assertEqual(set(range(6)), cleaned_up)

    async def test_caller_cancellation_waits_for_nested_cleanup(self):
        started = asyncio.Event()
        cleaned_up = set()

        async def upload(index):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                await asyncio.sleep(0)
                cleaned_up.add(index)

        async def group(start):
            return await gather_cancelling(*(upload(index) for index in range(start, start + 3)))

        task = asyncio.create_task(gather_cancelling(group(0), group(3)))
        try:
            await asyncio.wait_for(started.wait(), timeout=2)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertEqual(set(range(6)), cleaned_up)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    @parameterized.expand([("cancelled", False), ("failed", True)])
    async def test_repeated_cancellation_does_not_interrupt_sibling_cleanup(self, _name, fail):
        started = asyncio.Event()
        cleaning = asyncio.Event()
        release = asyncio.Event()
        finished = asyncio.Event()
        failure = RuntimeError("upload failed")

        async def upload():
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaning.set()
                await release.wait()
                finished.set()

        async def fail_or_wait():
            await started.wait()
            if fail:
                raise failure
            await asyncio.Event().wait()

        task = asyncio.create_task(gather_cancelling(upload(), fail_or_wait()))
        try:
            await asyncio.wait_for(started.wait(), timeout=2)
            if not fail:
                task.cancel()
            await asyncio.wait_for(cleaning.wait(), timeout=2)
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0)
            self.assertFalse(task.done())
            release.set()
            with self.assertRaises(RuntimeError if fail else asyncio.CancelledError) as caught:
                await task
            if fail:
                self.assertIs(failure, caught.exception)
            self.assertTrue(finished.is_set())
        finally:
            release.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def test_empty_group_returns_empty_list(self):
        self.assertEqual([], await gather_cancelling())

    async def test_preparation_uses_default_executor_and_propagates_context(self):
        context = ContextVar("preparation_test")
        context.set("inherited")
        main_thread = threading.get_ident()
        loop = asyncio.get_running_loop()
        executor = ThreadPoolExecutor(max_workers=1)
        loop.set_default_executor(executor)

        def worker():
            return threading.get_ident(), context.get()

        thread, value = await run_upload_preparation(worker)
        self.assertNotEqual(main_thread, thread)
        self.assertEqual("inherited", value)

    @parameterized.expand([("success", False), ("failure", True)])
    async def test_repeated_cancellation_drains_preparation(self, _name, fail):
        started = asyncio.Event()
        release = threading.Event()
        finished = threading.Event()
        loop = asyncio.get_running_loop()

        def worker():
            loop.call_soon_threadsafe(started.set)
            try:
                if not release.wait(timeout=10):
                    raise TimeoutError("Test did not release preparation")
                if fail:
                    raise ValueError("preparation failed")
                return 42
            finally:
                finished.set()

        task = asyncio.create_task(run_upload_preparation(worker))
        try:
            await asyncio.wait_for(started.wait(), timeout=2)
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0)
            self.assertFalse(task.done())
            with (
                self.assertLogs("evo.object.uploads", level="DEBUG")
                if fail
                else self.assertNoLogs("evo.object.uploads", level="DEBUG")
            ):
                release.set()
                with self.assertRaises(asyncio.CancelledError):
                    await task
            self.assertTrue(finished.is_set())
        finally:
            release.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def test_preparation_failure_is_propagated(self):
        failure = ValueError("invalid input")

        def worker():
            raise failure

        with self.assertRaises(ValueError) as caught:
            await run_upload_preparation(worker)
        self.assertIs(failure, caught.exception)

    async def test_secondary_failures_are_logged_without_replacing_the_first(self):
        errors = [RuntimeError("first"), ValueError("second")]

        async def fail(error):
            raise error

        with self.assertLogs("evo.object.uploads", level="DEBUG") as logs:
            with self.assertRaises(RuntimeError) as caught:
                await gather_cancelling(*(fail(error) for error in errors))
        self.assertIs(errors[0], caught.exception)
        self.assertEqual(1, len(logs.records))
        self.assertIs(errors[1], logs.records[0].exc_info[1])

    async def test_task_creation_failure_closes_unscheduled_coroutines_and_drains_tasks(self):
        async def upload():
            await asyncio.sleep(0)

        coroutines = [upload() for _ in range(3)]
        tasks = []
        create_task = asyncio.create_task

        def schedule(coroutine):
            if tasks:
                raise RuntimeError("task factory failed")
            task = create_task(coroutine)
            tasks.append(task)
            return task

        with patch("evo.objects.utils._concurrency.asyncio.create_task", side_effect=schedule):
            with self.assertRaisesRegex(RuntimeError, "task factory failed"):
                await gather_cancelling(*coroutines)

        self.assertTrue(all(task.done() for task in tasks))
        self.assertTrue(all(inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED for coro in coroutines))
