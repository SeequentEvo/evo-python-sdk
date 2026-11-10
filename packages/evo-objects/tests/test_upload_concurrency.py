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
from contextlib import nullcontext
from unittest import IsolatedAsyncioTestCase
from unittest.mock import patch

from parameterized import parameterized

from evo.objects.utils import upload_concurrency
from evo.objects.utils._concurrency import gather_uploads, get_max_concurrent_uploads, get_upload_semaphore


class TestUploadConcurrency(IsolatedAsyncioTestCase):
    @parameterized.expand([("default", None), ("serial", 1), ("lower", 3), ("higher", 12)])
    async def test_nested_groups_share_limit_preserve_order_and_reset_scope(self, _name, configured_limit):
        limit = configured_limit if configured_limit is not None else 8
        release = asyncio.Event()
        limit_reached = asyncio.Event()
        active = 0
        peak = 0
        semaphores = []

        async def upload(index):
            nonlocal active, peak
            semaphore = get_upload_semaphore()
            semaphores.append(semaphore)
            async with semaphore:
                active += 1
                peak = max(peak, active)
                if active == limit:
                    limit_reached.set()
                try:
                    await release.wait()
                    return index
                finally:
                    active -= 1

        async def group(start):
            return await gather_uploads(*(upload(index) for index in range(start, start + 12)))

        async def run():
            return await gather_uploads(group(0), group(12), group(24))

        with upload_concurrency(configured_limit) if configured_limit is not None else nullcontext():
            task = asyncio.create_task(run())
        try:
            await asyncio.wait_for(limit_reached.wait(), timeout=2)
            self.assertEqual(limit, active)
            release.set()
            result = await asyncio.wait_for(task, timeout=2)
        finally:
            release.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

        self.assertEqual([list(range(0, 12)), list(range(12, 24)), list(range(24, 36))], result)
        self.assertEqual(limit, peak)
        self.assertEqual(0, active)
        self.assertTrue(all(semaphore is semaphores[0] for semaphore in semaphores))
        self.assertIsNot(get_upload_semaphore(), semaphores[0])

    @parameterized.expand([(value,) for value in (0, -1, True, False, 1.5, "3", None)])
    async def test_invalid_configuration_leaves_scope_unchanged(self, value):
        with self.assertRaisesRegex(ValueError, "max_concurrent must be a positive integer"):
            with upload_concurrency(value):
                self.fail("Invalid configuration was accepted")
        self.assertEqual(8, get_max_concurrent_uploads())
        self.assertIsNone(get_upload_semaphore())

    async def test_nested_configuration_restores_scope_after_failure(self):
        with upload_concurrency(3):
            outer = get_upload_semaphore()
            self.assertEqual(3, get_max_concurrent_uploads())
            with self.assertRaisesRegex(RuntimeError, "failed"):
                with upload_concurrency(1):
                    self.assertEqual(1, get_max_concurrent_uploads())
                    self.assertIsNot(outer, get_upload_semaphore())
                    raise RuntimeError("failed")
            self.assertEqual(3, get_max_concurrent_uploads())
            self.assertIs(outer, get_upload_semaphore())
        self.assertEqual(8, get_max_concurrent_uploads())
        self.assertIsNone(get_upload_semaphore())

    async def test_configuration_is_isolated_between_tasks(self):
        ready = asyncio.Event()
        release = asyncio.Event()

        async def configured():
            with upload_concurrency(2):
                ready.set()
                await release.wait()
                return get_max_concurrent_uploads()

        task = asyncio.create_task(configured())
        try:
            await asyncio.wait_for(ready.wait(), timeout=2)
            self.assertEqual(8, get_max_concurrent_uploads())
            self.assertIsNone(get_upload_semaphore())
            release.set()
            self.assertEqual(2, await asyncio.wait_for(task, timeout=2))
        finally:
            release.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def test_failure_cancels_and_drains_nested_uploads_and_waiters(self):
        started = asyncio.Event()
        release = asyncio.Event()
        cleaned_up = set()
        failure = RuntimeError("upload failed")

        async def upload(index):
            try:
                async with get_upload_semaphore():
                    started.set()
                    await release.wait()
            finally:
                await asyncio.sleep(0)
                cleaned_up.add(index)

        async def group(start):
            return await gather_uploads(*(upload(index) for index in range(start, start + 12)))

        async def fail():
            await started.wait()
            raise failure

        try:
            with self.assertRaises(RuntimeError) as caught:
                await gather_uploads(group(0), group(12), fail())
            self.assertIs(failure, caught.exception)
            self.assertEqual(set(range(24)), cleaned_up)
        finally:
            release.set()

    async def test_caller_cancellation_waits_for_nested_cleanup(self):
        started = asyncio.Event()
        release = asyncio.Event()
        cleaned_up = set()

        async def upload(index):
            started.set()
            try:
                await release.wait()
            finally:
                await asyncio.sleep(0)
                cleaned_up.add(index)

        async def group(start):
            return await gather_uploads(*(upload(index) for index in range(start, start + 3)))

        async def run():
            return await gather_uploads(group(0), group(3))

        task = asyncio.create_task(run())
        try:
            await asyncio.wait_for(started.wait(), timeout=2)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertEqual(set(range(6)), cleaned_up)
        finally:
            release.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def test_scope_is_reset_after_success_and_failure(self):
        semaphores = []

        async def upload(*, fail=False):
            semaphores.append(get_upload_semaphore())
            if fail:
                raise ValueError("invalid input")

        await gather_uploads(upload())
        self.assertIsNot(get_upload_semaphore(), semaphores[-1])
        with self.assertRaisesRegex(ValueError, "invalid input"):
            await gather_uploads(upload(fail=True))
        self.assertIsNot(get_upload_semaphore(), semaphores[-1])
        self.assertIsNot(semaphores[0], semaphores[1])

    async def test_empty_group_returns_empty_list(self):
        self.assertEqual([], await gather_uploads())

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
                await gather_uploads(*coroutines)

        self.assertTrue(all(task.done() for task in tasks))
        self.assertTrue(all(inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED for coro in coroutines))
        self.assertIsNone(get_upload_semaphore())

    async def test_semaphore_creation_failure_closes_all_coroutines(self):
        async def upload():
            return 1

        coroutines = [upload(), upload()]
        with patch("evo.objects.utils._concurrency.asyncio.Semaphore", side_effect=RuntimeError("no semaphore")):
            with self.assertRaisesRegex(RuntimeError, "no semaphore"):
                await gather_uploads(*coroutines)

        self.assertTrue(all(inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED for coro in coroutines))
        self.assertIsNone(get_upload_semaphore())
