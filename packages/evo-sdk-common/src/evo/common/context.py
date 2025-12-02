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

import sys
import uuid
from logging import getLogger
from typing import Any, Mapping, overload

from .connector import APIConnector
from .data import Environment
from .exceptions import ContextError
from .interfaces import IAuthorizer, ICache, ITransport

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

__all__ = ["EvoContext"]

logger = getLogger(__name__)


class EvoContext:
    """Context for Evo SDKs.

    This class includes the following information:
    - Transport and authorizer, required to connect to Evo APIs.
    - Optional cache, which will be used for caching downloads/uploads of data, and other temporary data.
    - Hub, organization, and workspace values, which will be used to determine under which hub/organization/workspace
    operations are performed within. Which of these values are required depends on the specific SDK that is used.
    """

    @overload
    def __init__(
        self,
        *,
        transport: ITransport,
        authorizer: IAuthorizer,
        additional_headers: Mapping[str, Any] | None = None,
        cache: ICache | None = None,
        hub_url: str | None = None,
        org_id: uuid.UUID | None = None,
        workspace_id: uuid.UUID | None = None,
    ):
        """Initializes an EvoContext with transport and authorizer.

        :param transport: The transport to use for API calls.
        :param authorizer: The authorizer to use for API calls.
        :param additional_headers: Additional headers to include in API calls.
        :param cache: Optional cache to use for caching data.
        :param hub_url: Optional hub URL to use for API calls.
        :param org_id: Optional organization ID to use for API calls.
        :param workspace_id: Optional workspace ID to use for API calls.
        """

    @overload
    def __init__(
        self,
        *,
        connector: APIConnector,
        cache: ICache | None = None,
        org_id: uuid.UUID | None = None,
        workspace_id: uuid.UUID | None = None,
    ):
        """Initializes an EvoContext with an APIConnector.

        :param connector: The APIConnector to use for API calls.
        :param cache: Optional cache to use for caching data.
        :param org_id: Optional organization ID to use for API calls.
        :param workspace_id: Optional workspace ID to use for API calls.
        """

    def __init__(
        self,
        *,
        transport: ITransport | None = None,
        authorizer: IAuthorizer | None = None,
        connector: APIConnector | None = None,
        additional_headers: Mapping[str, Any] | None = None,
        cache: ICache | None = None,
        hub_url: str | None = None,
        org_id: uuid.UUID | None = None,
        workspace_id: uuid.UUID | None = None,
    ):
        if connector is not None:
            if transport is not None or authorizer is not None or additional_headers is not None or hub_url is not None:
                raise ValueError(
                    "If 'connector' is provided, 'transport', 'authorizer', 'additional_headers', and 'hub_url' must not be provided."
                )
            self._connector = connector
            hub_url = connector.base_url
            self._transport = None
            self._authorizer = None
            self._additional_headers = None
        else:
            if transport is None or authorizer is None:
                raise ValueError("'transport' and 'authorizer' must be provided if 'connector' is not provided.")
            self._connector = None
            self._transport = transport
            self._authorizer = authorizer
            self._additional_headers = additional_headers
        self._cache = cache
        self._hub_url = hub_url
        self._org_id = org_id
        self._workspace_id = workspace_id

    @property
    def cache(self) -> ICache | None:
        """The cache used by this context, if any."""
        return self._cache

    @property
    def hub_url(self) -> str | None:
        """The hub URL of this context, if set."""
        return self._hub_url

    @property
    def org_id(self) -> uuid.UUID | None:
        """The organization ID of this context, if set."""
        return self._org_id

    @property
    def workspace_id(self) -> uuid.UUID | None:
        """The workspace ID of this context, if set."""
        return self._workspace_id

    def get_connector(self) -> APIConnector:
        """Gets the APIConnector for this context."""
        if self._connector is not None:
            return self._connector
        hub_url = self._hub_url
        if hub_url is None:
            raise ContextError("Can't determine hub URL for connector. Context must have the hub URL set.")
        return APIConnector(
            base_url=hub_url,
            transport=self._transport,
            authorizer=self._authorizer,
            additional_headers=self._additional_headers,
        )

    def get_environment(self) -> Environment:
        """Gets the Environment of this context."""
        if self._hub_url is None:
            raise ContextError("Can't determine hub URL for the environment. Context must have a hub URL set.")
        if self._org_id is None:
            raise ContextError(
                "Can't determine organization for environment. Context must have an organization ID set."
            )
        if self._workspace_id is None:
            raise ContextError("Can't determine workspace for environment. Context must have a workspace ID set.")
        return Environment(
            hub_url=self._hub_url,
            org_id=self._org_id,
            workspace_id=self._workspace_id,
        )

    @classmethod
    def from_environment(
        cls,
        environment: Environment,
        connector: APIConnector,
        cache: ICache | None = None,
    ) -> Self:
        """Constructs an EvoContext from an Environment and an APIConnector."""
        if connector.base_url != environment.hub_url.rstrip("/") + "/":
            raise ContextError("The connector's base URL does not match the environment's hub URL.")
        return cls(
            connector=connector,
            cache=cache,
            org_id=environment.org_id,
            workspace_id=environment.workspace_id,
        )

    def with_cache(self, cache: ICache | None) -> Self:
        """Creates a copy of this context with the specified cache.

        :param cache: The cache to use in the new context.
        :return: A new EvoContext with the specified cache.
        """
        if self._connector is None:
            return EvoContext(
                transport=self._transport,
                authorizer=self._authorizer,
                additional_headers=self._additional_headers,
                cache=cache,
                hub_url=self._hub_url,
                org_id=self._org_id,
                workspace_id=self._workspace_id,
            )
        else:
            return EvoContext(
                connector=self._connector,
                cache=cache,
                org_id=self._org_id,
                workspace_id=self._workspace_id,
            )
