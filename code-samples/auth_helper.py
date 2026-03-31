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


_COMMON_REQUIRED_ENV_VARS = ("EVO_ORG_ID", "EVO_HUB_URL")
_USER_AUTH_ENV_VARS = ("EVO_USERNAME", "EVO_PASSWORD", "EVO_CLIENT_ID")


def _get_required_env(*keys: str) -> dict[str, str]:
    """Return a dict of environment variable values, raising if any are missing."""
    missing = [k for k in keys if not os.environ.get(k)]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(sorted(missing))}")
    return {k: os.environ[k] for k in keys}


def _has_user_credentials() -> bool:
    """Return True if test user credentials (ROPC) are available in the environment."""
    return all(os.environ.get(k) for k in _USER_AUTH_ENV_VARS)


def _create_user_auth(user_agent: str = "Evo SDK CI/1.0"):
    """Create transport and authorizer using test user credentials (ROPC) from environment.

    Uses the Resource Owner Password Credentials grant to authenticate as a
    test user.  Requires EVO_USERNAME, EVO_PASSWORD, and EVO_CLIENT_ID.
    """
    from _ropc_authorizer import ResourceOwnerPasswordAuthorizer

    from evo.aio import AioTransport
    from evo.oauth import EvoScopes, OAuthConnector

    transport = AioTransport(user_agent=user_agent)
    authorizer = ResourceOwnerPasswordAuthorizer(
        oauth_connector=OAuthConnector(
            transport=transport,
            client_id=os.environ["EVO_CLIENT_ID"],
        ),
        username=os.environ["EVO_USERNAME"],
        password=os.environ["EVO_PASSWORD"],
        scopes=EvoScopes.all_evo | EvoScopes.offline_access,
    )
    return transport, authorizer


async def _create_ci_manager(
    cache_location: str = "./notebook-data",
) -> _CIManager:
    """Create an auth manager for CI using test user credentials (ROPC).

    Requires the following environment variables:
    - EVO_USERNAME
    - EVO_PASSWORD
    - EVO_CLIENT_ID
    - EVO_ORG_ID
    - EVO_HUB_URL

    Optional:
    - EVO_WORKSPACE_ID (if not set, the first available workspace is used)
    """
    from evo.common import APIConnector
    from evo.common.utils import Cache

    env = _get_required_env(*_COMMON_REQUIRED_ENV_VARS)
    transport, authorizer = _create_user_auth()
    connector = APIConnector(base_url=env["EVO_HUB_URL"], transport=transport, authorizer=authorizer)
    cache = Cache(cache_location, mkdir=True)

    workspace_id = os.environ.get("EVO_WORKSPACE_ID")
    if not workspace_id:
        from evo.workspaces import WorkspaceAPIClient

        ws_client = WorkspaceAPIClient(connector=connector, org_id=UUID(env["EVO_ORG_ID"]))
        workspaces = await ws_client.list_all_workspaces()
        if not workspaces:
            raise RuntimeError("No workspaces found for the given organization. Set EVO_WORKSPACE_ID explicitly.")
        workspace_id = str(workspaces[0].id)

    return _CIManager(
        connector=connector,
        hub_url=env["EVO_HUB_URL"],
        org_id=env["EVO_ORG_ID"],
        workspace_id=workspace_id,
        cache=cache,
    )


def _is_ci() -> bool:
    """Return True if running in a CI environment with test user credentials."""
    is_ci_env = bool(os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"))
    return is_ci_env and _has_user_credentials()


async def get_manual_auth(
    client_id: str = "<your-client-id>",
    redirect_url: str = "<your-redirect-url>",
    org_id: str = "<your-organization-id>",
    hub_url: str = "<your-hub-url>",
    user_agent: str = "Evo Python SDK Notebook",
):
    """Get auth components for notebooks that build transport/authorizer manually.

    In CI, uses test user credentials (ROPC).
    Otherwise, uses interactive browser-based AuthorizationCodeAuthorizer.

    Returns:
        Tuple of (transport, authorizer, org_id, hub_url)
    """
    if _is_ci():
        transport, authorizer = _create_user_auth(user_agent=user_agent)
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
