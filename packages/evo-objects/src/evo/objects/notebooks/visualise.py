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

"""Visualization utilities for geoscience objects."""

from __future__ import annotations

from typing import Any, Sequence

from evo.common.styles.html import STYLESHEET, build_title
from evo.common.urls import get_viewer_url_from_environment

__all__ = [
    "visualise_objects",
]


def visualise_objects(
    objects: Sequence[Any],
    label: str = "View in Evo Viewer",
) -> None:
    """Display a clickable link to view multiple objects together in the Evo Viewer.

    This function generates a viewer URL with comma-separated object IDs, allowing
    multiple objects to be visualized in the same scene.

    In a Jupyter environment, this renders a styled HTML link. Outside of notebooks,
    this function does nothing.

    :param objects: Sequence of typed objects (PointSet, BlockModel, Variogram, etc.)
        All objects must be from the same workspace.
    :param label: Label text for the link (default: "View in Evo Viewer").

    :raises ValueError: If no objects are provided or if objects are from different workspaces.

    Example:
        ```python
        from evo.objects.notebooks import visualise_objects
        from evo.objects.typed import PointSet, BlockModel

        pointset = await PointSet.from_reference(manager, pointset_ref)
        block_model = await BlockModel.from_reference(manager, bm_ref)

        # Display a link to view both objects together
        visualise_objects([pointset, block_model])

        # With custom label
        visualise_objects([pointset, block_model], label="View drilling data with block model")
        ```
    """
    try:
        from IPython.display import HTML, display
    except ImportError:
        return  # Not in a notebook environment

    if not objects:
        raise ValueError("At least one object is required")

    # Extract object IDs and validate they're from the same workspace
    object_ids: list[str] = []
    environment = None

    for obj in objects:
        if not hasattr(obj, "metadata") or not hasattr(obj.metadata, "id"):
            raise TypeError(f"Object {obj} does not have expected metadata structure")

        obj_env = obj.metadata.environment
        if environment is None:
            environment = obj_env
        elif (
            str(obj_env.org_id) != str(environment.org_id)
            or str(obj_env.workspace_id) != str(environment.workspace_id)
        ):
            raise ValueError("All objects must be from the same workspace")

        object_ids.append(str(obj.metadata.id))

    # Generate the viewer URL with comma-separated IDs
    viewer_url = get_viewer_url_from_environment(environment, object_ids)

    # Build styled HTML with the link
    html = f"""
    {STYLESHEET}
    <div class="evo">
        {build_title(label, [("Open Viewer", viewer_url)])}
        <div style="font-size: 12px; color: #666;">
            Viewing {len(object_ids)} object{"s" if len(object_ids) > 1 else ""}
        </div>
    </div>
    """
    display(HTML(html))

