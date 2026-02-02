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

"""URL generation utilities for Evo Portal and Viewer links."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

if TYPE_CHECKING:
    from evo.common import Environment

__all__ = [
    "get_evo_base_url",
    "get_hub_code",
    "get_portal_url",
    "get_portal_url_from_environment",
    "get_portal_url_from_reference",
    "get_viewer_url",
    "get_viewer_url_from_environment",
    "get_viewer_url_from_reference",
    "parse_object_reference_url",
    "serialize_object_reference",
]


def get_evo_base_url(hub_url: str) -> str:
    """Determine the Evo base URL from an API hub URL.

    :param hub_url: The hub URL (e.g., "https://350mt.api.integration.seequent.com").
    :return: The Evo base URL (e.g., "https://evo.integration.seequent.com").
    """
    parsed = urlparse(hub_url)
    hostname = parsed.hostname or ""

    if "integration" in hostname:
        return "https://evo.integration.seequent.com"
    elif "test" in hostname:
        return "https://evo.test.seequent.com"
    else:
        return "https://evo.seequent.com"


def get_hub_code(hub_url: str) -> str:
    """Extract the hub code from a hub URL.

    :param hub_url: The hub URL (e.g., "https://350mt.api.integration.seequent.com").
    :return: The hub code (e.g., "350mt").
    :raises ValueError: If the hub code cannot be extracted.
    """
    parsed = urlparse(hub_url)
    hostname_parts = parsed.hostname.split('.') if parsed.hostname else []
    if len(hostname_parts) < 1:
        raise ValueError(f"Invalid URL: cannot extract hub code from hostname '{parsed.hostname}'")
    return hostname_parts[0]


def get_portal_url(
    org_id: str,
    workspace_id: str,
    object_id: str,
    hub_url: str,
) -> str:
    """Generate the Evo Portal URL for a geoscience object.

    Uses the new URL format: /{org_id}/data/{workspace_id}/objects/{object_id}

    :param org_id: The organization ID.
    :param workspace_id: The workspace ID.
    :param object_id: The object ID.
    :param hub_url: The hub URL to determine the environment.
    :return: The complete portal URL.
    """
    evo_base_url = get_evo_base_url(hub_url)
    return f"{evo_base_url}/{org_id}/data/{workspace_id}/objects/{object_id}"


def get_viewer_url(
    org_id: str,
    workspace_id: str,
    object_ids: str | list[str],
    hub_url: str,
) -> str:
    """Generate the Evo Viewer URL for one or more geoscience objects.

    Uses the format: /{org_id}/workspaces/{hub_code}/{workspace_id}/viewer?id={ids}
    Multiple objects are comma-separated in the id parameter.

    :param org_id: The organization ID.
    :param workspace_id: The workspace ID.
    :param object_ids: Single object ID or list of object IDs.
    :param hub_url: The hub URL to determine the environment and hub code.
    :return: The complete viewer URL.
    """
    evo_base_url = get_evo_base_url(hub_url)
    hub_code = get_hub_code(hub_url)

    if isinstance(object_ids, list):
        ids_param = ",".join(str(oid) for oid in object_ids)
    else:
        ids_param = str(object_ids)

    return f"{evo_base_url}/{org_id}/workspaces/{hub_code}/{workspace_id}/viewer?id={ids_param}"


def get_portal_url_from_environment(environment: Environment, object_id: str) -> str:
    """Generate the Evo Portal URL from an Environment object.

    :param environment: The environment containing org_id, workspace_id, and hub_url.
    :param object_id: The object ID.
    :return: The complete portal URL.
    """
    return get_portal_url(
        org_id=str(environment.org_id),
        workspace_id=str(environment.workspace_id),
        object_id=str(object_id),
        hub_url=environment.hub_url,
    )


def get_viewer_url_from_environment(
    environment: Environment,
    object_ids: str | list[str],
) -> str:
    """Generate the Evo Viewer URL from an Environment object.

    :param environment: The environment containing org_id, workspace_id, and hub_url.
    :param object_ids: Single object ID or list of object IDs.
    :return: The complete viewer URL.
    """
    return get_viewer_url(
        org_id=str(environment.org_id),
        workspace_id=str(environment.workspace_id),
        object_ids=object_ids,
        hub_url=environment.hub_url,
    )


def serialize_object_reference(value: Any) -> str:
    """Serialize an object reference to a string URL.

    Supports ObjectReference, string URLs, and various typed object classes.

    :param value: The value to serialize.
    :return: String URL of the object reference.
    :raises TypeError: If the value type is not supported.
    """
    if isinstance(value, str):
        return value

    # Check for ObjectReference (it's a str subclass with extra attributes)
    if hasattr(value, "hub_url") and hasattr(value, "org_id"):
        return str(value)

    # Check for typed objects with metadata.url (e.g., PointSet, BlockModel, Regular3DGrid)
    if hasattr(value, "metadata") and hasattr(value.metadata, "url"):
        return str(value.metadata.url)

    # Check for DownloadedObject or ObjectMetadata with url attribute
    if hasattr(value, "url"):
        return str(value.url)

    raise TypeError(f"Cannot serialize object reference of type {type(value)}")


def parse_object_reference_url(object_reference: str) -> tuple[str, str, str, str]:
    """Parse an object reference URL into its components.

    :param object_reference: A geoscience object reference URL.
    :return: Tuple of (org_id, workspace_id, object_id, hub_url).
    :raises ValueError: If the URL format is invalid.
    """
    parsed = urlparse(object_reference)

    # Reconstruct hub_url from the hostname
    hub_url = f"{parsed.scheme}://{parsed.hostname}"

    # Extract org_id, workspace_id, and object_id from path
    # Path format: /geoscience-object/orgs/{org_id}/workspaces/{workspace_id}/objects/{object_id}
    path_pattern = r'/geoscience-object/orgs/([^/]+)/workspaces/([^/]+)/objects/([^/?]+)'
    match = re.match(path_pattern, parsed.path)

    if not match:
        raise ValueError(f"Invalid object reference URL: path '{parsed.path}' does not match expected format")

    org_id = match.group(1)
    workspace_id = match.group(2)
    object_id = match.group(3)

    return org_id, workspace_id, object_id, hub_url


def get_portal_url_from_reference(object_reference: str) -> str:
    """Generate the Evo Portal URL from an object reference URL.

    :param object_reference: A geoscience object reference URL.
    :return: The complete portal URL.
    """
    org_id, workspace_id, object_id, hub_url = parse_object_reference_url(object_reference)
    return get_portal_url(org_id, workspace_id, object_id, hub_url)


def get_viewer_url_from_reference(object_reference: str) -> str:
    """Generate the Evo Viewer URL from an object reference URL.

    :param object_reference: A geoscience object reference URL.
    :return: The complete viewer URL.
    """
    org_id, workspace_id, object_id, hub_url = parse_object_reference_url(object_reference)
    return get_viewer_url(org_id, workspace_id, object_id, hub_url)

