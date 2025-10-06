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
import uuid
from pathlib import Path

from evo.common import APIConnector, Environment, ICache
from evo.common.io import Download, Upload
from evo.common.io.exceptions import RenewalError

from ._utils import extract_payload
from .endpoints.api import JobsApi
from .endpoints.models import JobPendingPayload, JobStatus, QueryDownload, QueryResult
from .exceptions import UnknownJobPayload

_CACHE_SCOPE = "blockmodel"


class BlockModelDownload(Download[QueryResult]):
    def __init__(
        self,
        connector: APIConnector,
        environment: Environment,
        query_result: QueryResult,
        job_id: uuid.UUID,
        initial_url: str | None = None,
    ) -> None:
        self._jobs_api = JobsApi(connector)
        self._environment = environment
        self._query_result = query_result
        self._job_id = job_id
        self._mutex = asyncio.Lock()
        self._initial_url = initial_url

    @property
    def label(self) -> str:
        """The label of the resource to be downloaded."""
        return "Block Model"

    @property
    def metadata(self) -> QueryResult:
        """The metadata of the resource to be downloaded."""
        return self._query_result

    def _get_cache_location(self, cache: ICache) -> Path:
        """Generate the cache location for the resource to be downloaded.

        :param cache: The cache to resolve the cache location.

        :returns: The cache location.
        """
        return cache.get_location(environment=self._environment, scope=_CACHE_SCOPE) / str(self._job_id)

    async def get_download_url(self) -> str:
        """Generate a URL that will be used to download the resource.

        This method may be called multiple times to generate a new URL if the last URL expires.

        :returns: The download URL.
        """
        async with self._mutex:
            if (url := self._initial_url) is not None:
                # Clear the initial URL so it can't be reused.
                self._initial_url = None
                return url

        response = await self._jobs_api.get_job_status(
            job_id=str(self._job_id),
            workspace_id=str(self._environment.workspace_id),
            org_id=str(self._environment.org_id),
            bm_id=str(self._query_result.bm_uuid),
        )
        if response.job_status != JobStatus.COMPLETE:
            raise RenewalError("Job is not complete")
        try:
            payload = extract_payload(self._job_id, response, QueryDownload)
        except UnknownJobPayload as e:
            raise RenewalError(str(e)) from e
        return str(payload.download_url)


class BlockModelUpload(Upload):
    def __init__(
        self,
        connector: APIConnector,
        environment: Environment,
        bm_uuid: uuid.UUID,
        job_id: uuid.UUID,
        initial_url: str | None = None,
    ) -> None:
        self._jobs_api = JobsApi(connector)
        self._environment = environment
        self._bm_uuid = bm_uuid
        self._job_id = job_id
        self._mutex = asyncio.Lock()
        self._initial_url = initial_url

    @property
    def label(self) -> str:
        """The label of the resource to be uploaded."""
        return "Block Model"

    async def get_upload_url(self) -> str:
        """Generate a URL that will be used to upload the resource.

        This method may be called multiple times to generate a new URL if the last URL expires.

        :returns: The upload URL.

        :raises DataExistsError: if the resource already exists in the target service.
        """
        async with self._mutex:
            if (url := self._initial_url) is not None:
                # Clear the initial URL so it can't be reused.
                self._initial_url = None
                return url

        response = await self._jobs_api.get_job_status(
            job_id=str(self._job_id),
            workspace_id=str(self._environment.workspace_id),
            org_id=str(self._environment.org_id),
            bm_id=str(self._bm_uuid),
        )
        if response.job_status != JobStatus.PENDING_UPLOAD:
            raise RenewalError("Job is not in the pending upload state")
        try:
            payload = extract_payload(self._job_id, response, JobPendingPayload)
        except UnknownJobPayload as e:
            raise RenewalError(str(e)) from e
        return str(payload.upload_url)


def get_cache_location_for_upload(cache: ICache, environment: Environment, job_id: uuid.UUID) -> Path:
    return cache.get_location(environment=environment, scope=_CACHE_SCOPE) / str(job_id)
