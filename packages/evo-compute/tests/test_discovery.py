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

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from contextlib import contextmanager
from unittest import mock

from evo.common import RequestMethod
from evo.common.test_tools import ORG as TEST_ORG
from evo.common.test_tools import MockResponse, TestWithConnector
from evo.common.utils import get_header_metadata

from data import load_test_data
from evo.compute import DiscoveryClient, TaskResource
from evo.compute.discovery import DEFAULT_CACHE_TTL_SECONDS
from evo.compute.endpoints.api import DiscoveryApi
from evo.compute.endpoints.models import DiscoveryResponse


class FakeClock:
    """A controllable monotonic clock, so the cache TTL can be exercised without sleeping."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeContext:
    """Minimal duck-typed IContext exposing just what ``from_context`` uses."""

    def __init__(self, connector, org_id) -> None:
        self._connector = connector
        self._org_id = org_id

    def get_connector(self):
        return self._connector

    def get_org_id(self):
        return self._org_id


class TestDiscoveryClient(TestWithConnector):
    def setUp(self) -> None:
        super().setUp()
        self.clock = FakeClock()
        clock_patch = mock.patch("evo.compute.discovery.time", mock.Mock(monotonic=self.clock))
        clock_patch.start()
        self.addCleanup(clock_patch.stop)
        self.client = DiscoveryClient(connector=self.connector, org_id=TEST_ORG.id)
        self.setup_universal_headers(get_header_metadata(DiscoveryApi.__module__))
        self.catalogue = load_test_data("discovery-tasks.json")

    @property
    def tasks_path(self) -> str:
        return f"/compute/orgs/{TEST_ORG.id}/tasks?details=true&limit=100&offset=0"

    @contextmanager
    def set_discovery_response(self, data: dict | None = None) -> Iterator[None]:
        content = json.dumps(self.catalogue if data is None else data)
        with self.transport.set_http_response(
            status_code=200, content=content, headers={"Content-Type": "application/json"}
        ):
            yield

    def _page_response(self, page: dict) -> MockResponse:
        return MockResponse(status_code=200, content=json.dumps(page), headers={"Content-Type": "application/json"})

    # -- request shape ----------------------------------------------------- #

    async def test_list_tasks_requests_details_and_pagination(self) -> None:
        """The catalogue is fetched with ``details=true`` and pagination params."""
        with self.set_discovery_response():
            await self.client.list_tasks()
        self.assert_request_made(
            RequestMethod.GET,
            self.tasks_path,
            headers={"Accept": "application/json"},
        )

    async def test_list_tasks_parses_task_resource(self) -> None:
        """Results deserialize to ``TaskResource`` with the full parameter schema."""
        with self.set_discovery_response():
            tasks = await self.client.list_tasks()
        declustering = next(task for task in tasks if task.name == "declustering")
        self.assertIsInstance(declustering, TaskResource)
        self.assertEqual("geostatistics", declustering.topic)
        self.assertEqual("0.3.1", declustering.version)
        self.assertIn("source", declustering.parameters.get("properties", {}))

    async def test_lists_every_task(self) -> None:
        """The whole catalogue is returned, including feature_flag tasks."""
        with self.set_discovery_response():
            tasks = await self.client.list_tasks()
        names = {task.name for task in tasks}
        self.assertEqual({"declustering", "normal-score", "kriging", "obj-import"}, names)

    # -- pagination -------------------------------------------------------- #

    async def test_fetches_all_pages(self) -> None:
        """``_fetch`` follows pagination until the reported total is reached."""
        results = self.catalogue["results"]
        self.transport.request.side_effect = [
            self._page_response({"limit": 2, "offset": 0, "total": 3, "count": 2, "results": results[:2]}),
            self._page_response({"limit": 2, "offset": 2, "total": 3, "count": 1, "results": results[2:3]}),
        ]
        tasks = await self.client.list_tasks()
        self.transport.assert_n_requests_made(2)
        self.assertEqual(3, len(tasks))

    async def test_stops_when_total_is_missing(self) -> None:
        """A page without a ``total`` stops pagination instead of looping forever."""
        results = self.catalogue["results"]
        self.transport.request.side_effect = [
            self._page_response({"limit": 2, "offset": 0, "total": None, "count": 2, "results": results[:2]}),
        ]
        tasks = await self.client.list_tasks()
        self.transport.assert_n_requests_made(1)
        self.assertEqual(2, len(tasks))

    # -- caching / TTL ----------------------------------------------------- #

    async def test_second_call_within_ttl_is_cached(self) -> None:
        """A second call within the TTL is served from the cache (no new request)."""
        with self.set_discovery_response():
            await self.client.list_tasks()
            self.transport.assert_n_requests_made(1)
            self.clock.advance(DEFAULT_CACHE_TTL_SECONDS - 1)
            await self.client.list_tasks()
        self.transport.assert_n_requests_made(1)

    async def test_cache_expiry_triggers_refetch(self) -> None:
        """Once the TTL elapses, the catalogue is re-fetched."""
        with self.set_discovery_response():
            await self.client.list_tasks()
            self.clock.advance(DEFAULT_CACHE_TTL_SECONDS + 1)
            await self.client.list_tasks()
        self.transport.assert_n_requests_made(2)

    async def test_force_refresh_bypasses_cache(self) -> None:
        """``list_tasks(force_refresh=True)`` always re-fetches."""
        with self.set_discovery_response():
            await self.client.list_tasks()
            await self.client.list_tasks(force_refresh=True)
        self.transport.assert_n_requests_made(2)

    async def test_accessor_force_refresh_bypasses_cache(self) -> None:
        """``force_refresh`` is wired through the convenience accessors."""
        with self.set_discovery_response():
            await self.client.get_topics()
            await self.client.get_topics(force_refresh=True)
        self.transport.assert_n_requests_made(2)

    async def test_returned_list_is_a_copy(self) -> None:
        """Mutating the returned list must not corrupt the internal cache."""
        with self.set_discovery_response():
            first = await self.client.list_tasks()
            first.clear()
            second = await self.client.list_tasks()
        self.assertTrue(second)

    async def test_concurrent_cold_calls_fetch_once(self) -> None:
        """A burst of concurrent calls on a cold cache triggers a single fetch."""
        calls = 0
        first_call_started = asyncio.Event()
        release = asyncio.Event()

        async def slow_list_tasks(**kwargs: object) -> DiscoveryResponse:
            nonlocal calls
            calls += 1
            first_call_started.set()
            await release.wait()
            return DiscoveryResponse.model_validate(self.catalogue)

        with mock.patch.object(self.client._discovery_api, "list_tasks", new=slow_list_tasks):
            gathered = asyncio.gather(
                self.client.list_tasks(),
                self.client.list_tasks(),
                self.client.list_tasks(),
            )
            await first_call_started.wait()
            release.set()
            results = await gathered

        self.assertEqual(1, calls)
        self.assertTrue(all(results))

    # -- convenience helpers ----------------------------------------------- #

    async def test_get_topics(self) -> None:
        with self.set_discovery_response():
            topics = await self.client.get_topics()
        self.assertEqual(["converter", "geostatistics"], topics)

    async def test_get_topic_tasks(self) -> None:
        with self.set_discovery_response():
            tasks = await self.client.get_topic_tasks("geostatistics")
        self.assertEqual({"declustering", "normal-score", "kriging"}, {task.name for task in tasks})

    async def test_get_task_found_and_missing(self) -> None:
        with self.set_discovery_response():
            found = await self.client.get_task("geostatistics", "declustering")
            missing = await self.client.get_task("geostatistics", "does-not-exist")
        self.assertIsNotNone(found)
        assert found is not None  # for type-checkers
        self.assertEqual("declustering", found.name)
        self.assertIsNone(missing)

    # -- construction ------------------------------------------------------ #

    async def test_from_context(self) -> None:
        client = DiscoveryClient.from_context(FakeContext(self.connector, TEST_ORG.id))
        self.assertEqual(TEST_ORG.id, client.org_id)
        with self.set_discovery_response():
            tasks = await client.list_tasks()
        self.assertTrue(tasks)
