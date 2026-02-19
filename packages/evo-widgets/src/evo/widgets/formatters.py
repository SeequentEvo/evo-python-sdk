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
    "format_variogram",
]


def _get_base_metadata(obj: Any) -> tuple[str, list[tuple[str, str]] | None, list[tuple[str, str]]]:
    """Extract common metadata from a geoscience object.

    :param obj: A typed geoscience object with `as_dict()` and `metadata` attributes.
    :return: A tuple of (name, title_links, rows) where:
        - name: The object name
        - title_links: List of (label, url) tuples for Portal/Viewer links, or None
        - rows: List of (label, value) tuples for Object ID, Schema, and Tags
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

    return name, title_links, rows


def _build_html_from_rows(
    name: str,
    title_links: list[tuple[str, str]] | None,
    rows: list[tuple[str, str]],
    extra_content: str = "",
) -> str:
    """Build HTML output from formatted rows.

    :param name: The object name for the title.
    :param title_links: List of (label, url) tuples for title links, or None.
    :param rows: List of (label, value) tuples for the table.
    :param extra_content: Additional HTML content to append after the table.
    :return: Complete HTML string with stylesheet.
    """
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
        html += f"<table>{''.join(table_rows)}</table>"
    html += extra_content
    html += "</div>"

    return html


def format_base_object(obj: Any) -> str:
    """Format a BaseObject (or subclass) as HTML.

    This formatter handles any typed geoscience object (PointSet, Regular3DGrid, etc.)
    by extracting metadata and rendering it as a styled HTML table with Portal/Viewer links.

    :param obj: A typed geoscience object with `as_dict()`, `metadata`, and `_sub_models` attributes.
    :return: HTML string representation.
    """
    doc = obj.as_dict()
    name, title_links, rows = _get_base_metadata(obj)

    # Add bounding box if present (as nested table)
    if bbox := doc.get("bounding_box"):
        bbox_rows = [
            ["<strong>X:</strong>", bbox.get("min_x", 0), bbox.get("max_x", 0)],
            ["<strong>Y:</strong>", bbox.get("min_y", 0), bbox.get("max_y", 0)],
            ["<strong>Z:</strong>", bbox.get("min_z", 0), bbox.get("max_z", 0)],
        ]
        bbox_table = build_nested_table(["", "Min", "Max"], bbox_rows)
        rows.append(("Bounding box:", bbox_table))

    # Add CRS if present
    if crs := doc.get("coordinate_reference_system"):
        crs_str = f"EPSG:{crs.get('epsg_code')}" if isinstance(crs, dict) else str(crs)
        rows.append(("CRS:", crs_str))

    # Build datasets section - add as rows to the main table
    sub_models = getattr(obj, "_sub_models", [])
    for dataset_name in sub_models:
        dataset = getattr(obj, dataset_name, None)
        if dataset and hasattr(dataset, "attributes") and len(dataset.attributes) > 0:
            # Build attribute rows
            attr_rows = []
            for attr in dataset.attributes:
                attr_info = attr.as_dict()
                attr_name = attr_info.get("name", "Unknown")
                attr_type = attr_info.get("attribute_type", "Unknown")
                attr_rows.append([attr_name, attr_type])

            attrs_table = build_nested_table(["Attribute", "Type"], attr_rows)
            rows.append((f"{dataset_name}:", attrs_table))

    return _build_html_from_rows(name, title_links, rows)


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


def format_variogram(obj: Any) -> str:
    """Format a Variogram object as HTML.

    This formatter renders a variogram with its properties and structures
    as a styled HTML table with Portal/Viewer links.

    :param obj: A Variogram object with `as_dict()`, `metadata`, `sill`, `nugget`,
        `structures`, and other variogram-specific attributes.
    :return: HTML string representation.
    """
    doc = obj.as_dict()
    name, title_links, rows = _get_base_metadata(obj)

    # Add variogram specific rows
    sill = getattr(obj, "sill", doc.get("sill", 0))
    nugget = getattr(obj, "nugget", doc.get("nugget", 0))
    is_rotation_fixed = getattr(obj, "is_rotation_fixed", doc.get("is_rotation_fixed", False))

    rows.append(("Sill:", f"{sill:.4g}"))
    rows.append(("Nugget:", f"{nugget:.4g}"))
    rows.append(("Rotation Fixed:", str(is_rotation_fixed)))

    # Add optional fields
    attribute = getattr(obj, "attribute", doc.get("attribute"))
    domain = getattr(obj, "domain", doc.get("domain"))
    modelling_space = getattr(obj, "modelling_space", doc.get("modelling_space"))
    data_variance = getattr(obj, "data_variance", doc.get("data_variance"))

    if attribute:
        rows.append(("Attribute:", attribute))
    if domain:
        rows.append(("Domain:", domain))
    if modelling_space:
        rows.append(("Modelling Space:", modelling_space))
    if data_variance is not None:
        rows.append(("Data Variance:", f"{data_variance:.4g}"))

    # Build structures section
    extra_content = ""
    structures = getattr(obj, "structures", doc.get("structures", []))
    if structures:
        struct_rows = []
        for i, struct in enumerate(structures):
            vtype = struct.get("variogram_type", "unknown")
            contribution = struct.get("contribution", 0)

            # Calculate standardized sill (% of variance)
            standardized_sill = round(contribution / sill, 2) if sill != 0 else 0.0

            # Extract anisotropy info
            anisotropy = struct.get("anisotropy", {})
            ranges = anisotropy.get("ellipsoid_ranges", {})
            rotation = anisotropy.get("rotation", {})

            range_str = (
                f"({ranges.get('major', 0):.1f}, {ranges.get('semi_major', 0):.1f}, {ranges.get('minor', 0):.1f})"
            )
            # Rotation order: dip, dip_az, pitch
            rot_str = f"({rotation.get('dip', 0):.1f}°, {rotation.get('dip_azimuth', 0):.1f}°, {rotation.get('pitch', 0):.1f}°)"

            struct_rows.append(
                [
                    f"{i + 1}",
                    vtype,
                    f"{contribution:.4g}",
                    f"{standardized_sill:.2f}",
                    range_str,
                    rot_str,
                ]
            )

        structures_table = build_nested_table(
            ["#", "Type", "Contribution", "Std. Sill", "Ranges (maj, semi, min)", "Rotation (dip, dip_az, pitch)"],
            struct_rows,
        )
        extra_content = (
            f'<div style="margin-top: 8px;"><strong>Structures ({len(structures)}):</strong></div>{structures_table}'
        )

    return _build_html_from_rows(name, title_links, rows, extra_content)
