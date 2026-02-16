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

"""HTML formatters for Evo SDK objects.

This module provides HTML formatter functions for various Evo SDK types.
These formatters are registered with IPython when the extension is loaded.
"""

from __future__ import annotations

from typing import Any

from .html import (
    STYLESHEET,
    build_nested_table,
    build_table_row,
    build_table_row_vtop,
    build_title,
)
from .urls import get_portal_url_for_object, get_viewer_url_for_object

__all__ = [
    "format_attributes_collection",
    "format_base_object",
]


def format_base_object(obj: Any) -> str:
    """Format a BaseObject (or subclass) as HTML.

    This formatter handles any typed geoscience object (PointSet, Regular3DGrid, etc.)
    by extracting metadata and rendering it as a styled HTML table with Portal/Viewer links.

    :param obj: A typed geoscience object with `as_dict()`, `metadata`, and `_sub_models` attributes.
    :return: HTML string representation.
    """
    doc = obj.as_dict()

    # Get basic info
    name = doc.get("name", "Unnamed")
    schema = doc.get("schema", "Unknown")
    obj_id = doc.get("uuid", "Unknown")

    # Build title links for viewer and portal
    try:
        portal_url = get_portal_url_for_object(obj)
        viewer_url = get_viewer_url_for_object(obj)
        title_links = [("Portal", portal_url), ("Viewer", viewer_url)]
    except (AttributeError, TypeError):
        title_links = None

    # Build metadata rows
    rows: list[tuple[str, str]] = [
        ("Object ID:", str(obj_id)),
        ("Schema:", schema),
    ]

    # Add tags if present
    if tags := doc.get("tags"):
        tags_str = ", ".join(f"{k}: {v}" for k, v in tags.items())
        rows.append(("Tags:", tags_str))

    # Add bounding box if present (as nested table)
    if bbox := doc.get("bounding_box"):
        bbox_rows = [
            ["<strong>X:</strong>", bbox.get('min_x', 0), bbox.get('max_x', 0)],
            ["<strong>Y:</strong>", bbox.get('min_y', 0), bbox.get('max_y', 0)],
            ["<strong>Z:</strong>", bbox.get('min_z', 0), bbox.get('max_z', 0)],
        ]
        bbox_table = build_nested_table(["", "Min", "Max"], bbox_rows)
        rows.append(("Bounding box:", bbox_table))

    # Add CRS if present
    if crs := doc.get("coordinate_reference_system"):
        crs_str = f"EPSG:{crs.get('epsg_code')}" if isinstance(crs, dict) else str(crs)
        rows.append(("CRS:", crs_str))

    # Build datasets section - add as rows to the main table
    sub_models = getattr(obj, '_sub_models', [])
    for dataset_name in sub_models:
        dataset = getattr(obj, dataset_name, None)
        if dataset and hasattr(dataset, 'attributes') and len(dataset.attributes) > 0:
            # Build attribute rows
            attr_rows = []
            for attr in dataset.attributes:
                attr_info = attr.as_dict()
                attr_name = attr_info.get("name", "Unknown")
                attr_type = attr_info.get("attribute_type", "Unknown")
                attr_rows.append([attr_name, attr_type])

            attrs_table = build_nested_table(["Attribute", "Type"], attr_rows)
            rows.append((f"{dataset_name}:", attrs_table))

    # Build unified table with all rows
    table_rows = []
    for label, value in rows:
        if label in ("Bounding box:",) or label.endswith(":") and isinstance(value, str) and "<table" in value:
            table_rows.append(build_table_row_vtop(label, value))
        else:
            table_rows.append(build_table_row(label, value))

    html = STYLESHEET
    html += '<div class="evo">'
    html += build_title(name, title_links)
    if table_rows:
        html += f'<table>{"".join(table_rows)}</table>'
    html += '</div>'

    return html


def format_attributes_collection(obj: Any) -> str:
    """Format an Attributes collection as HTML.

    This formatter renders a collection of attributes as a styled table
    showing name and type for each attribute.

    :param obj: An Attributes object that is iterable and has `as_dict()` on items.
    :return: HTML string representation.
    """
    if len(obj) == 0:
        return f'{STYLESHEET}<div class="evo">No attributes available.</div>'

    # Get all attribute info dictionaries
    attr_infos = [attr.as_dict() for attr in obj]

    # Build data rows with headers
    headers = ["Name", "Type"]
    rows = []
    for info in attr_infos:
        attribute_type = info["attribute_type"]
        if attribute_type != "category":
            attribute_str = f"{info['attribute_type']} ({info['values']['data_type']})"
        else:
            attribute_str = attribute_type
        rows.append([info["name"], attribute_str])

    # Use nested table for a clean header/row structure
    table_html = build_nested_table(headers, rows)
    return f'{STYLESHEET}<div class="evo">{table_html}</div>'

