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

"""Discovery client for the compute task catalogue.

``DiscoveryClient`` lists the tasks available to an organization, with the full parameter
and result schemas for each. The catalogue is paginated by the service; this client fetches
every page and caches the combined result with a time-to-live. The convenience accessors
(:meth:`get_topics`, :meth:`get_topic_tasks`, :meth:`get_task`) read from that cache.
"""

from __future__ import annotations

import asyncio
import time
from uuid import UUID

from evo.common import APIConnector, IContext

from .endpoints.api import DiscoveryApi
from .endpoints.models import TaskResource

__all__ = [
    "DiscoveryClient",
]

DEFAULT_CACHE_TTL_SECONDS = 300.0
"""Default time-to-live for the cached task catalogue, in seconds."""

_PAGE_SIZE = 100
"""Number of tasks requested per page when fetching the catalogue."""


class DiscoveryClient:
    """List the compute task catalogue for an organization.

    Returned tasks are independent copies, so editing them does not affect the cache.

    :param connector: The API connector to use.
    :param org_id: The organization ID.
    :param cache_ttl_seconds: How long a fetched catalogue is cached before the next call
        re-fetches it. Pass ``force_refresh=True`` to any accessor to re-fetch sooner.
    """

    def __init__(
        self,
        connector: APIConnector,
        org_id: UUID,
        *,
        cache_ttl_seconds: float = DEFAULT_CACHE_TTL_SECONDS,
    ) -> None:
        self._connector = connector
        self._org_id = org_id
        self._cache_ttl_seconds = cache_ttl_seconds
        self._discovery_api = DiscoveryApi(connector)
        self._mutex = asyncio.Lock()
        self._cache: list[TaskResource] = []
        self._cache_expiry: float | None = None

    @classmethod
    def from_context(
        cls,
        context: IContext,
        *,
        cache_ttl_seconds: float = DEFAULT_CACHE_TTL_SECONDS,
    ) -> DiscoveryClient:
        """Create a DiscoveryClient from any real ``evo.common.IContext``.

        :param context: The context to create the client from (provides the connector and org id).
        :param cache_ttl_seconds: The catalogue cache time-to-live, in seconds.

        :return: A DiscoveryClient instance.
        """
        return cls(
            context.get_connector(),
            context.get_org_id(),
            cache_ttl_seconds=cache_ttl_seconds,
        )

    @property
    def org_id(self) -> UUID:
        """The organization ID."""
        return self._org_id

    def __repr__(self) -> str:
        return f"DiscoveryClient(org_id={self._org_id!r})"

    def _cache_is_valid(self) -> bool:
        return self._cache_expiry is not None and time.monotonic() < self._cache_expiry

    async def _fetch(self) -> None:
        """Fetch every page of the catalogue and refresh the cache."""
        tasks: list[TaskResource] = []
        offset = 0
        async with self._connector:
            while True:
                page = await self._discovery_api.list_tasks(
                    org_id=str(self._org_id),
                    details=True,
                    limit=_PAGE_SIZE,
                    offset=offset,
                )
                tasks.extend(page.results)
                offset += len(page.results)
                # Stop on an empty page, an unreported total, or once the reported total is reached.
                if not page.results or page.total is None or offset >= page.total:
                    break
        self._cache = tasks
        self._cache_expiry = time.monotonic() + self._cache_ttl_seconds

    async def _catalogue(self, force_refresh: bool) -> list[TaskResource]:
        """Return the cached catalogue, fetching it first if it is stale or forced."""
        async with self._mutex:
            if force_refresh or not self._cache_is_valid():
                await self._fetch()
            return self._cache

    async def list_tasks(self, *, force_refresh: bool = False) -> list[TaskResource]:
        """Return the full catalogue of discoverable tasks.

        Served from the cache while it is within its TTL; pass ``force_refresh=True`` to
        re-fetch immediately.

        :param force_refresh: Re-fetch the catalogue even if the cache is still valid.

        :return: Every advertised task.
        """
        return [task.model_copy(deep=True) for task in await self._catalogue(force_refresh)]

    async def get_topics(self, *, force_refresh: bool = False) -> list[str]:
        """Return the sorted list of topics that have discoverable tasks.

        :param force_refresh: Re-fetch the catalogue even if the cache is still valid.
        """
        return sorted({task.topic for task in await self._catalogue(force_refresh)})

    async def get_topic_tasks(self, topic: str, *, force_refresh: bool = False) -> list[TaskResource]:
        """Return the tasks advertised within a single topic.

        :param topic: The task topic.
        :param force_refresh: Re-fetch the catalogue even if the cache is still valid.

        :return: The tasks in the topic (empty if the topic is not advertised).
        """
        return [task.model_copy(deep=True) for task in await self._catalogue(force_refresh) if task.topic == topic]

    async def get_task(self, topic: str, name: str, *, force_refresh: bool = False) -> TaskResource | None:
        """Return a single task by topic and name, or ``None`` if it is not advertised.

        :param topic: The task topic.
        :param name: The task name as advertised by the platform (e.g. ``kriging``).
        :param force_refresh: Re-fetch the catalogue even if the cache is still valid.

        :return: The matching task, or ``None``.
        """
        for task in await self._catalogue(force_refresh):
            if task.topic == topic and task.name == name:
                return task.model_copy(deep=True)
        return None
