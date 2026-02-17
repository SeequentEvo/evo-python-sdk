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
from .urls import get_blocksync_block_model_url_from_environment, get_portal_url_for_object, get_viewer_url_for_object

__all__ = [
    "format_attributes_collection",
    "format_base_object",
    "format_block_model",
    "format_block_model_attributes",
    "format_block_model_version",
    "format_report",
    "format_report_result",
    "format_variogram",
]


def _get_base_metadata(
    obj: Any,
    extra_links: list[tuple[str, str]] | None = None,
) -> tuple[str, list[tuple[str, str]] | None, list[tuple[str, str]]]:
    """Extract common metadata from a geoscience object.

    :param obj: A typed geoscience object with `as_dict()` and `metadata` attributes.
    :param extra_links: Optional additional links to include after Portal/Viewer links.
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
        if extra_links:
            title_links.extend(extra_links)
    except (AttributeError, TypeError):
        title_links = extra_links if extra_links else None

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


def _format_bounding_box(bbox: dict[str, Any]) -> str:
    """Format a bounding box as a nested HTML table.

    :param bbox: Dictionary with min_x, max_x, min_y, max_y, min_z, max_z keys.
    :return: HTML string for the bounding box table.
    """
    bbox_rows = [
        ["<strong>X:</strong>", f"{bbox.get('min_x', 0):.2f}", f"{bbox.get('max_x', 0):.2f}"],
        ["<strong>Y:</strong>", f"{bbox.get('min_y', 0):.2f}", f"{bbox.get('max_y', 0):.2f}"],
        ["<strong>Z:</strong>", f"{bbox.get('min_z', 0):.2f}", f"{bbox.get('max_z', 0):.2f}"],
    ]
    return build_nested_table(["", "Min", "Max"], bbox_rows)


def _format_crs(crs: Any) -> str:
    """Format a coordinate reference system.

    :param crs: CRS dict with epsg_code or ogc_wkt, or a string.
    :return: Formatted CRS string.
    """
    if isinstance(crs, dict):
        return f"EPSG:{crs.get('epsg_code')}" if crs.get("epsg_code") else str(crs)
    return str(crs)


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
        rows.append(("Bounding box:", _format_bounding_box(bbox)))

    # Add CRS if present
    if crs := doc.get("coordinate_reference_system"):
        rows.append(("CRS:", _format_crs(crs)))

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


def format_block_model_version(obj: Any) -> str:
    """Format a block model Version object as HTML.

    This formatter renders a block model version with its metadata,
    bounding box, and column information as a styled HTML table.

    :param obj: A Version object from evo.blockmodels.data.
    :return: HTML string representation.
    """
    # Build columns table
    col_rows = [[col.title, col.data_type.value, col.unit_id or "-"] for col in obj.columns]
    columns_html = build_nested_table(["Title", "Type", "Unit"], col_rows)

    # Build bbox table
    bbox_html = "-"
    if obj.bbox:
        bbox_rows = [
            ["i", obj.bbox.i_minmax.min, obj.bbox.i_minmax.max],
            ["j", obj.bbox.j_minmax.min, obj.bbox.j_minmax.max],
            ["k", obj.bbox.k_minmax.min, obj.bbox.k_minmax.max],
        ]
        bbox_html = build_nested_table(["Axis", "Min", "Max"], bbox_rows)

    # Build table rows
    rows_html = "".join(
        [
            build_table_row("Version ID", str(obj.version_id)),
            build_table_row("Version UUID", str(obj.version_uuid)),
            build_table_row("Block Model UUID", str(obj.bm_uuid)),
            build_table_row("Parent Version", str(obj.parent_version_id) if obj.parent_version_id else "-"),
            build_table_row("Base Version", str(obj.base_version_id) if obj.base_version_id else "-"),
            build_table_row("Created At", obj.created_at.strftime("%Y-%m-%d %H:%M:%S")),
            build_table_row("Created By", obj.created_by.name or obj.created_by.email or str(obj.created_by.id)),
            build_table_row("Comment", obj.comment if obj.comment else "-"),
            build_table_row_vtop("Bounding Box", bbox_html),
            build_table_row_vtop("Columns", columns_html),
        ]
    )

    html = f"""{STYLESHEET}
<div class="evo">
{build_title("📦 Block Model Version")}
<table>
{rows_html}
</table>
</div>
"""
    return html


def format_report_result(obj: Any) -> str:
    """Format a ReportResult object as HTML.

    This formatter renders a block model report result with its data
    as a styled HTML table.

    :param obj: A ReportResult object from evo.blockmodels.typed.report.
    :return: HTML string representation.
    """
    df = obj.to_dataframe()

    # Build the result table with alternating row colors
    headers = list(df.columns)
    header_html = "".join([f"<th>{h}</th>" for h in headers])

    rows_html = []
    for i, (_, row) in enumerate(df.iterrows()):
        row_class = 'class="alt-row"' if i % 2 == 1 else ""
        cells = "".join([f"<td>{v if v is not None and v == v else ''}</td>" for v in row])
        rows_html.append(f"<tr {row_class}>{cells}</tr>")

    subtitle = f'<div class="subtitle">Created: {obj.created_at.strftime("%Y-%m-%d %H:%M:%S")} | Rows: {len(df)}</div>'

    html = f"""{STYLESHEET}
<div class="evo">
{build_title("📊 Report Result (Version " + str(obj.version_id) + ")")}
{subtitle}
<table class="nested">
    <tr>{header_html}</tr>
    {"".join(rows_html)}
</table>
</div>
"""
    return html


def format_report(obj: Any) -> str:
    """Format a Report object as HTML.

    This formatter renders a block model report specification with its
    columns, categories, and BlockSync link as a styled HTML table.

    :param obj: A Report object from evo.blockmodels.typed.report.
    :return: HTML string representation.
    """
    from .urls import get_blocksync_report_url, get_hub_code

    # Get environment info for BlockSync URL
    environment = obj._context.get_environment()
    hub_code = get_hub_code(environment.hub_url)
    blocksync_url = get_blocksync_report_url(
        org_id=environment.org_id,
        hub_code=hub_code,
        workspace_id=environment.workspace_id,
        block_model_id=obj._block_model_uuid,
        report_id=obj.id,
    )

    # Build column info table
    columns_html = ""
    if obj._specification.columns:
        col_rows = []
        for i, col in enumerate(obj._specification.columns):
            row_class = 'class="alt-row"' if i % 2 == 1 else ""
            col_rows.append(
                f"<tr {row_class}><td>{col.label}</td><td>{col.aggregation}</td><td>{col.output_unit_id}</td></tr>"
            )
        columns_html = f"""
        <div style="margin-top: 8px;"><strong>Columns:</strong></div>
        <table class="nested">
            <tr><th>Label</th><th>Aggregation</th><th>Unit</th></tr>
            {"".join(col_rows)}
        </table>
        """

    # Build category info table
    categories_html = ""
    if obj._specification.categories:
        cat_rows = []
        for i, cat in enumerate(obj._specification.categories):
            row_class = 'class="alt-row"' if i % 2 == 1 else ""
            values_str = ", ".join(cat.values) if cat.values else "(all)"
            cat_rows.append(f"<tr {row_class}><td>{cat.label}</td><td>{values_str}</td></tr>")
        categories_html = f"""
        <div style="margin-top: 8px;"><strong>Categories:</strong></div>
        <table class="nested">
            <tr><th>Label</th><th>Values</th></tr>
            {"".join(cat_rows)}
        </table>
        """

    # Build main info table rows
    block_model_display = (
        f"{obj._block_model_name} ({obj._block_model_uuid})" if obj._block_model_name else str(obj._block_model_uuid)
    )

    rows: list[tuple[str, str]] = [
        ("Report ID:", str(obj.id)),
        ("Block Model:", block_model_display),
        ("Revision:", str(obj.revision)),
    ]

    # Add last run if available
    if hasattr(obj._specification, "last_result_created_at") and obj._specification.last_result_created_at:
        rows.append(("Last run:", obj._specification.last_result_created_at.strftime("%Y-%m-%d %H:%M:%S")))

    # Build table rows HTML
    table_rows_html = "".join([build_table_row(label, value) for label, value in rows])

    html = f"""{STYLESHEET}
<div class="evo">
{build_title(f"📊 {obj.name}", [("BlockSync", blocksync_url)])}
<table>
{table_rows_html}
</table>
{categories_html}
{columns_html}
</div>
"""
    return html


def format_block_model_attributes(obj: Any) -> str:
    """Format a BlockModelAttributes collection as HTML.

    This formatter renders a collection of block model attributes as a styled table
    showing name, type and unit for each attribute.

    :param obj: A BlockModelAttributes object that is iterable.
    :return: HTML string representation.
    """
    if len(obj) == 0:
        return f'{STYLESHEET}<div class="evo">No attributes available.</div>'

    headers = ["Name", "Type", "Unit"]
    rows = [[attr.name, attr.attribute_type, attr.unit or ""] for attr in obj]
    table_html = build_nested_table(headers, rows)
    return f'{STYLESHEET}<div class="evo">{table_html}</div>'


def format_block_model(obj: Any) -> str:
    """Format a BlockModel (from evo.objects.typed) as HTML.

    This formatter renders a block model reference with its metadata, geometry,
    bounding box, and attributes as a styled HTML table with Portal/Viewer/BlockSync links.

    :param obj: A BlockModel object from evo.objects.typed.block_model_ref.
    :return: HTML string representation.
    """
    doc = obj.as_dict()

    # Build BlockSync link
    try:
        blocksync_url = get_blocksync_block_model_url_from_environment(
            environment=obj._obj.metadata.environment,
            block_model_id=obj.block_model_uuid,
        )
        extra_links = [("BlockSync", blocksync_url)]
    except (AttributeError, TypeError):
        extra_links = None

    # Get common metadata with BlockSync link
    name, title_links, rows = _get_base_metadata(obj, extra_links=extra_links)

    # Add Block Model UUID
    rows.append(("Block Model UUID:", str(obj.block_model_uuid)))

    # Add geometry info
    geom = obj.geometry
    geom_rows = [
        ["<strong>Origin:</strong>", f"({geom.origin.x:.2f}, {geom.origin.y:.2f}, {geom.origin.z:.2f})"],
        ["<strong>N Blocks:</strong>", f"({geom.n_blocks.nx}, {geom.n_blocks.ny}, {geom.n_blocks.nz})"],
        [
            "<strong>Block Size:</strong>",
            f"({geom.block_size.dx:.2f}, {geom.block_size.dy:.2f}, {geom.block_size.dz:.2f})",
        ],
    ]
    if geom.rotation:
        geom_rows.append(
            [
                "<strong>Rotation:</strong>",
                f"({geom.rotation[0]:.2f}, {geom.rotation[1]:.2f}, {geom.rotation[2]:.2f})",
            ]
        )
    geom_table = build_nested_table(["Property", "Value"], geom_rows)
    rows.append(("Geometry:", geom_table))

    # Add bounding box if present
    if bbox := doc.get("bounding_box"):
        rows.append(("Bounding Box:", _format_bounding_box(bbox)))

    # Add CRS if present
    if crs := doc.get("coordinate_reference_system"):
        rows.append(("CRS:", _format_crs(crs)))

    # Build the table rows
    table_rows = []
    for label, value in rows:
        if label in ("Bounding Box:", "Geometry:"):
            table_rows.append(build_table_row_vtop(label, value))
        else:
            table_rows.append(build_table_row(label, value))

    html = STYLESHEET
    html += '<div class="evo">'
    html += build_title(name, title_links)
    html += f"<table>{''.join(table_rows)}</table>"

    # Build attributes section
    attrs = obj.attributes
    if attrs and len(attrs) > 0:
        attr_rows = [[attr.name, attr.attribute_type, attr.unit or ""] for attr in attrs]
        attrs_table = build_nested_table(["Name", "Type", "Unit"], attr_rows)
        html += f'<div style="margin-top: 8px;"><strong>Attributes ({len(attrs)}):</strong></div>{attrs_table}'

    html += "</div>"
    return html
