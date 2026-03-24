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

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import UUID

if TYPE_CHECKING:
    from evo.common import APIConnector, ICache


@dataclass
class _Environment:
    """Environment configuration matching ServiceManagerWidget.get_environment() return type."""

    hub_url: str
    org_id: UUID
    workspace_id: UUID | None = None


class _CIManager:
    """CI-compatible manager that mimics ServiceManagerWidget interface."""

    def __init__(
        self,
        connector: "APIConnector",
        hub_url: str,
        org_id: str,
        workspace_id: str | None = None,
        cache: "ICache | None" = None,
    ) -> None:
        self._connector = connector
        self._hub_url = hub_url
        self._org_id = UUID(org_id)
        self._workspace_id = UUID(workspace_id) if workspace_id else None
        self._cache = cache

    def get_connector(self) -> "APIConnector":
        return self._connector

    def get_environment(self) -> _Environment:
        return _Environment(
            hub_url=self._hub_url,
            org_id=self._org_id,
            workspace_id=self._workspace_id,
        )

    @property
    def cache(self) -> "ICache | None":
        return self._cache


async def _create_ci_manager(
    cache_location: str = "./notebook-data",
) -> _CIManager:
    """Create an auth manager for CI using service app credentials.

    Requires the following environment variables:
    - EVO_CLIENT_ID
    - EVO_CLIENT_SECRET
    - EVO_ORG_ID
    - EVO_HUB_URL
    """
    from evo.aio import AioTransport
    from evo.common import APIConnector
    from evo.common.utils import Cache
    from evo.oauth import ClientCredentialsAuthorizer, EvoScopes, OAuthConnector

    required = {"EVO_CLIENT_ID", "EVO_CLIENT_SECRET", "EVO_ORG_ID", "EVO_HUB_URL"}
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(sorted(missing))}")

    client_id = os.environ["EVO_CLIENT_ID"]
    client_secret = os.environ["EVO_CLIENT_SECRET"]
    org_id = os.environ["EVO_ORG_ID"]
    hub_url = os.environ["EVO_HUB_URL"]

    transport = AioTransport(user_agent="Evo SDK CI/1.0")
    authorizer = ClientCredentialsAuthorizer(
        oauth_connector=OAuthConnector(
            transport=transport,
            client_id=client_id,
            client_secret=client_secret,
        ),
        scopes=EvoScopes.all_evo,
    )
    connector = APIConnector(base_url=hub_url, transport=transport, authorizer=authorizer)
    cache = Cache(cache_location, mkdir=True)

    return _CIManager(
        connector=connector,
        hub_url=hub_url,
        org_id=org_id,
        workspace_id=os.environ.get("EVO_WORKSPACE_ID"),
        cache=cache,
    )


async def get_manual_auth(
    client_id: str = "<your-client-id>",
    redirect_url: str = "<your-redirect-url>",
    org_id: str = "<your-organization-id>",
    hub_url: str = "<your-hub-url>",
    user_agent: str = "Evo Python SDK Notebook",
):
    """Get auth components for notebooks that build transport/authorizer manually.

    In CI (when EVO_CLIENT_SECRET is set), uses service app credentials.
    Otherwise, uses interactive browser-based AuthorizationCodeAuthorizer.

    Returns:
        Tuple of (transport, authorizer, org_id, hub_url)
    """
    from evo.aio import AioTransport

    is_ci = bool(os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS")) and bool(os.environ.get("EVO_CLIENT_SECRET"))

    if is_ci:
        from evo.oauth import ClientCredentialsAuthorizer, EvoScopes, OAuthConnector

        transport = AioTransport(user_agent=user_agent)
        authorizer = ClientCredentialsAuthorizer(
            oauth_connector=OAuthConnector(
                transport=transport,
                client_id=os.environ["EVO_CLIENT_ID"],
                client_secret=os.environ["EVO_CLIENT_SECRET"],
            ),
            scopes=EvoScopes.all_evo,
        )
        return transport, authorizer, os.environ["EVO_ORG_ID"], os.environ["EVO_HUB_URL"]

    # Interactive authentication
    from evo.oauth import AuthorizationCodeAuthorizer, EvoScopes, OAuthConnector

    transport = AioTransport(user_agent=user_agent)
    authorizer = AuthorizationCodeAuthorizer(
        oauth_connector=OAuthConnector(transport=transport, client_id=client_id),
        redirect_url=redirect_url,
        scopes=EvoScopes.all_evo | EvoScopes.offline_access,
    )
    await authorizer.login()

    return transport, authorizer, org_id, hub_url
