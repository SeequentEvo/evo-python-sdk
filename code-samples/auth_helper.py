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

    def get_org_id(self) -> UUID:
        return self._org_id

    def get_cache(self) -> "ICache | None":
        return self._cache

    @property
    def cache(self) -> "ICache | None":
        return self._cache


_REQUIRED_ENV_VARS = ("EVO_CLIENT_ID", "EVO_CLIENT_SECRET", "EVO_ORG_ID", "EVO_HUB_URL")


def _get_required_env(*keys: str) -> dict[str, str]:
    """Return a dict of environment variable values, raising if any are missing."""
    missing = [k for k in keys if not os.environ.get(k)]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(sorted(missing))}")
    return {k: os.environ[k] for k in keys}


def _create_service_auth(user_agent: str = "Evo SDK CI/1.0"):
    """Create transport and authorizer using service app credentials from environment."""
    from evo.aio import AioTransport
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
    return transport, authorizer


async def _create_ci_manager(
    cache_location: str = "./notebook-data",
) -> _CIManager:
    """Create an auth manager for CI using service app credentials.

    Requires the following environment variables:
    - EVO_CLIENT_ID
    - EVO_CLIENT_SECRET
    - EVO_ORG_ID
    - EVO_HUB_URL
    - EVO_WORKSPACE_ID (injected automatically by the test fixture)
    """
    from evo.common import APIConnector
    from evo.common.utils import Cache

    env = _get_required_env(*_REQUIRED_ENV_VARS)
    transport, authorizer = _create_service_auth()
    connector = APIConnector(base_url=env["EVO_HUB_URL"], transport=transport, authorizer=authorizer)
    cache = Cache(cache_location, mkdir=True)

    workspace_id = os.environ.get("EVO_WORKSPACE_ID")
    if not workspace_id:
        raise RuntimeError(
            "EVO_WORKSPACE_ID is not set. This value is injected automatically by the test fixture "
            "and should not be set manually."
        )

    return _CIManager(
        connector=connector,
        hub_url=env["EVO_HUB_URL"],
        org_id=env["EVO_ORG_ID"],
        workspace_id=workspace_id,
        cache=cache,
    )


def _is_ci() -> bool:
    """Return True if running in a CI environment with service credentials."""
    return bool(os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS")) and bool(os.environ.get("EVO_CLIENT_SECRET"))


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
    if _is_ci():
        transport, authorizer = _create_service_auth(user_agent=user_agent)
        return transport, authorizer, os.environ["EVO_ORG_ID"], os.environ["EVO_HUB_URL"]

    from evo.aio import AioTransport
    from evo.oauth import AuthorizationCodeAuthorizer, EvoScopes, OAuthConnector

    transport = AioTransport(user_agent=user_agent)
    authorizer = AuthorizationCodeAuthorizer(
        oauth_connector=OAuthConnector(transport=transport, client_id=client_id),
        redirect_url=redirect_url,
        scopes=EvoScopes.all_evo | EvoScopes.offline_access,
    )
    await authorizer.login()

    return transport, authorizer, org_id, hub_url
