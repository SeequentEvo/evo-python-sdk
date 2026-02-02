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

"""URL generation utilities for BlockSync Portal."""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

if TYPE_CHECKING:
    from evo.common import Environment

__all__ = [
    "get_blocksync_base_url",
    "get_blocksync_block_model_url",
    "get_blocksync_block_model_url_from_environment",
]


def get_blocksync_base_url(hub_url: str) -> str:
    """Determine the BlockSync base URL from an API hub URL.

    :param hub_url: The hub URL (e.g., "https://350mt.api.integration.seequent.com").
    :return: The BlockSync base URL (e.g., "https://blocksync.integration.seequent.com").
    """
    if "int" in hub_url or "integration" in hub_url or "qa" in hub_url:
        return "https://blocksync.integration.seequent.com"
    else:
        return "https://blocksync.seequent.com"


def get_blocksync_block_model_url(
    org_id: str | UUID,
    workspace_id: str | UUID,
    block_model_id: str | UUID,
    hub_url: str,
) -> str:
    """Generate the BlockSync Portal URL for a block model.

    Uses the format: /{org_id}/redirect?ws={workspace_id}&bm={block_model_id}

    :param org_id: The organization ID.
    :param workspace_id: The workspace ID.
    :param block_model_id: The block model ID.
    :param hub_url: The hub URL to determine the environment.
    :return: The complete BlockSync block model URL.
    """
    base_url = get_blocksync_base_url(hub_url)
    return (
        f"{base_url}/{str(org_id).lower()}"
        f"/redirect?ws={str(workspace_id).lower()}&bm={str(block_model_id).lower()}"
    )


def get_blocksync_block_model_url_from_environment(
    environment: "Environment",
    block_model_id: str | UUID,
) -> str:
    """Generate the BlockSync Portal URL from an Environment object.

    :param environment: The environment containing org_id, workspace_id, and hub_url.
    :param block_model_id: The block model ID.
    :return: The complete BlockSync block model URL.
    """
    return get_blocksync_block_model_url(
        org_id=environment.org_id,
        workspace_id=environment.workspace_id,
        block_model_id=block_model_id,
        hub_url=environment.hub_url,
    )

