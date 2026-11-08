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
"""Endpoint wrapper for listing an organization's compute tasks.

Wraps ``GET /compute/orgs/{org_id}/tasks``, returning the task catalogue with optional
pagination, filtering, and (with ``details=true``) the full parameter and result schemas
for each task.
"""

from typing import Any

from evo.common.connector import APIConnector
from evo.common.data import RequestMethod
from evo.common.utils import get_header_metadata

from ..models import DiscoveryResponse

__all__ = ["DiscoveryApi"]


class DiscoveryApi:
    """API client for the compute task discovery endpoint.

    :param connector: Client for communicating with the API.
    """

    def __init__(self, connector: APIConnector):
        self.connector = connector

    async def list_tasks(
        self,
        org_id: str,
        details: bool = False,
        limit: int | None = None,
        offset: int | None = None,
        name: str | None = None,
        topic: str | None = None,
        key: str | None = None,
        additional_headers: dict[str, str] | None = None,
        request_timeout: int | float | tuple[int | float, int | float] | None = None,
    ) -> DiscoveryResponse:
        """List the compute tasks discoverable by an organization.

        Wraps ``GET /compute/orgs/{org_id}/tasks``. The endpoint is paginated; use ``limit``
        and ``offset`` to page through the results.

        :param org_id: The organization identifier.
            Format: `uuid`
        :param details: When ``True``, request the full parameter/result schemas for each
            task (``details=true``) rather than a summary listing.
        :param limit: Maximum number of tasks to return in the page.
        :param offset: Index of the first task to return.
        :param name: Only return tasks with this name.
        :param topic: Only return tasks within this topic.
        :param key: Only return the task with this key.
        :param additional_headers: (optional) Additional headers to send with the request.
        :param request_timeout: (optional) Timeout setting for this request. If one number is
            provided, it will be the total request timeout. It can also be a pair (tuple) of
            (connection, read) timeouts.

        :return: A page of the discovery response containing the advertised tasks.

        :raise evo.common.exceptions.BadRequestException: If the server responds with HTTP status 400.
        :raise evo.common.exceptions.UnauthorizedException: If the server responds with HTTP status 401.
        :raise evo.common.exceptions.ForbiddenException: If the server responds with HTTP status 403.
        :raise evo.common.exceptions.NotFoundException: If the server responds with HTTP status 404.
        :raise evo.common.exceptions.BaseTypedError: If the server responds with any other HTTP status between
            400 and 599, and the body of the response contains a descriptive `type` parameter.
        :raise evo.common.exceptions.EvoAPIException: If the server responds with any other HTTP status between 400
            and 599, and the body of the response does not contain a `type` parameter.
        :raise evo.common.exceptions.UnknownResponseError: For other HTTP status codes with no corresponding response
            type in `response_types_map`.
        """
        _path_params = {
            "org_id": org_id,
        }

        # Serialize the bool as lowercase "true"/"false"; urlencode would emit Python's "True".
        _query_params: dict[str, Any] = {
            "details": str(details).lower(),
        }
        if limit is not None:
            _query_params["limit"] = limit
        if offset is not None:
            _query_params["offset"] = offset
        if name is not None:
            _query_params["name"] = name
        if topic is not None:
            _query_params["topic"] = topic
        if key is not None:
            _query_params["key"] = key

        _header_params = {
            "Accept": "application/json",
        } | get_header_metadata(__name__)
        if additional_headers is not None:
            _header_params.update(additional_headers)

        _response_types_map = {
            "200": DiscoveryResponse,
        }

        return await self.connector.call_api(
            method=RequestMethod.GET,
            resource_path="/compute/orgs/{org_id}/tasks",
            path_params=_path_params,
            query_params=_query_params,
            header_params=_header_params,
            response_types_map=_response_types_map,
            request_timeout=request_timeout,
        )
