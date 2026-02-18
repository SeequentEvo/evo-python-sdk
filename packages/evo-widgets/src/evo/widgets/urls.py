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

"""URL generation utilities for Evo Portal and Viewer links.

This module provides functions to generate URLs for viewing objects in the Evo Portal and Viewer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from evo.objects import ObjectReference

if TYPE_CHECKING:
    from evo.common.interfaces import IContext

__all__ = [
    "get_evo_base_url",
    "get_hub_code",
    "get_portal_url",
    "get_portal_url_for_object",
    "get_portal_url_from_reference",
    "get_viewer_url",
    "get_viewer_url_for_object",
    "get_viewer_url_for_objects",
    "get_viewer_url_from_reference",
    "serialize_object_reference",
]


def get_evo_base_url(hub_url: str) -> str:
    """Determine the Evo base URL from an API hub URL.

    :param hub_url: The hub URL (e.g., "https://350mt.api.seequent.com").
    :return: The Evo base URL (e.g., "https://evo.seequent.com").
    """
    return "https://evo.seequent.com"


def get_hub_code(hub_url: str) -> str:
    """Extract the hub code from a hub URL.

    :param hub_url: The hub URL (e.g., "https://350mt.api.seequent.com").
    :return: The hub code (e.g., "350mt").
    :raises ValueError: If the hub code cannot be extracted.
    """
    parsed = urlparse(hub_url)
    hostname_parts = parsed.hostname.split(".") if parsed.hostname else []
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


def get_portal_url_for_object(obj: Any) -> str:
    """Generate the Evo Portal URL for an object.

    :param obj: An object with a `metadata` attribute containing `environment` and `id`.
    :return: The complete portal URL.
    :raises AttributeError: If the object doesn't have the required metadata.
    """
    environment = obj.metadata.environment
    return get_portal_url(
        org_id=str(environment.org_id),
        workspace_id=str(environment.workspace_id),
        object_id=str(obj.metadata.id),
        hub_url=environment.hub_url,
    )


def get_viewer_url_for_object(obj: Any) -> str:
    """Generate the Evo Viewer URL for an object.

    :param obj: An object with a `metadata` attribute containing `environment` and `id`.
    :return: The complete viewer URL.
    :raises AttributeError: If the object doesn't have the required metadata.
    """
    environment = obj.metadata.environment
    return get_viewer_url(
        org_id=str(environment.org_id),
        workspace_id=str(environment.workspace_id),
        object_ids=str(obj.metadata.id),
        hub_url=environment.hub_url,
    )


def get_viewer_url_for_objects(context: IContext, objects: list[Any]) -> str:
    """Generate the Evo Viewer URL for a list of objects.

    This is a convenience function that extracts object IDs from various object types
    and generates a viewer URL to view them together.

    :param context: The context (e.g., manager) containing the environment information.
    :param objects: List of objects to view. Supports typed objects (e.g., PointSet, Regular3DGrid),
        ObjectReference, DownloadedObject, ObjectMetadata, or string object IDs.
    :return: The complete viewer URL for viewing all objects together.
    :raises ValueError: If the objects list is empty.
    :raises TypeError: If an object type is not supported.

    Example::

        from evo.widgets import get_viewer_url_for_objects

        # View multiple objects together
        url = get_viewer_url_for_objects(manager, [pointset, grid, blockmodel])
        print(url)  # Opens all objects in the viewer
    """
    if not objects:
        raise ValueError("At least one object is required")

    object_ids: list[str] = []
    for obj in objects:
        if isinstance(obj, str):
            # Assume it's already an object ID
            object_ids.append(obj)
        elif hasattr(obj, "metadata") and hasattr(obj.metadata, "id"):
            # Typed objects (e.g., PointSet, Regular3DGrid)
            object_ids.append(str(obj.metadata.id))
        elif hasattr(obj, "id"):
            # DownloadedObject or ObjectMetadata
            object_ids.append(str(obj.id))
        else:
            raise TypeError(f"Cannot extract object ID from type {type(obj)}")

    environment = context.get_environment()
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

    # Check for typed objects with metadata.url (e.g., PointSet, BlockModel, Regular3DGrid)
    if hasattr(value, "metadata") and hasattr(value.metadata, "url"):
        return str(value.metadata.url)

    # Check for DownloadedObject or ObjectMetadata with url attribute
    if hasattr(value, "url"):
        return str(value.url)

    raise TypeError(f"Cannot serialize object reference of type {type(value)}")


def get_portal_url_from_reference(object_reference: str) -> str:
    """Generate the Evo Portal URL from an object reference URL.

    :param object_reference: A geoscience object reference URL.
    :return: The complete portal URL.
    """
    ref = ObjectReference(object_reference)
    return get_portal_url(ref.org_id, ref.workspace_id, ref.object_id, ref.hub_url)


def get_viewer_url_from_reference(object_reference: str) -> str:
    """Generate the Evo Viewer URL from an object reference URL.

    :param object_reference: A geoscience object reference URL.
    :return: The complete viewer URL.
    """
    ref = ObjectReference(object_reference)
    return get_viewer_url(ref.org_id, ref.workspace_id, ref.object_id, ref.hub_url)
