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

"""Shared HTML styles for Jupyter notebook representations across all Evo SDK packages."""

# CSS Stylesheet for all HTML representations (objects, task results, etc.)
STYLESHEET = """
<style>
    .evo {
        border: 1px solid #ccc;
        border-radius: 3px;
        padding: 16px;
        margin: 8px 0;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        font-size: 13px;
        display: inline-block;
        max-width: 800px;
        background-color: var(--jp-layout-color1, #fff);
    }
    .evo .title {
        font-size: 15px;
        font-weight: 600;
        margin-bottom: 12px;
        color: var(--jp-ui-font-color1, #111);
        display: flex;
        align-items: baseline;
        gap: 8px;
    }
    .evo .title-links {
        font-size: 12px;
        font-weight: normal;
    }
    .evo .title-links a {
        color: #0066cc !important;
        text-decoration: none;
    }
    .evo .title-links a:hover {
        text-decoration: underline;
    }
    .evo table {
        border-collapse: collapse;
        width: auto;
        margin-bottom: 8px;
        table-layout: auto;
    }
    .evo td.label {
        padding: 3px 8px 3px 0;
        font-weight: 600;
        white-space: nowrap;
        vertical-align: top;
        color: var(--jp-ui-font-color1, #333);
        text-align: left;
        width: 0.1%;
    }
    .evo td.label-vtop {
        padding: 3px 8px 3px 0;
        font-weight: 600;
        white-space: nowrap;
        vertical-align: top;
        color: var(--jp-ui-font-color1, #333);
        text-align: left;
        width: 0.1%;
    }
    .evo td.value {
        padding: 3px 0;
        color: var(--jp-ui-font-color1, #111);
        text-align: left;
        width: auto;
    }
    .evo table.nested {
        border-collapse: collapse;
        font-size: 12px;
        margin-bottom: 0;
        width: auto;
    }
    .evo table.nested th {
        padding: 3px 12px 3px 0;
        text-align: left;
        font-weight: 600;
        color: var(--jp-ui-font-color1, #333);
    }
    .evo table.nested th.right {
        text-align: right;
        padding-right: 0;
    }
    .evo table.nested td {
        padding: 3px 12px 3px 0;
        color: var(--jp-ui-font-color1, #111);
        text-align: left;
    }
    .evo table.nested td.right {
        text-align: right;
        padding-right: 0;
    }
    .evo table.nested tr.alt-row {
        background-color: var(--jp-layout-color2, #f5f5f5);
    }
    .evo .section {
        margin-top: 8px;
    }
    .evo .section-heading {
        font-weight: 600;
        margin-bottom: 6px;
        color: var(--jp-ui-font-color1, #333);
    }
    .evo .indent {
        margin-left: 16px;
    }
    
    /* Task result specific styles */
    .evo .attr-highlight {
        background: #e3f2fd;
        padding: 2px 8px;
        border-radius: 3px;
        font-family: monospace;
        font-weight: 600;
        color: #1565c0;
    }
    .evo .message {
        background: #e8f5e9;
        padding: 6px 10px;
        border-radius: 3px;
        color: #2e7d32;
        margin-bottom: 12px;
        font-size: 12px;
    }
    .evo .success {
        color: #2e7d32;
    }
    .evo .subtitle {
        font-size: 12px;
        color: #666;
        margin-bottom: 8px;
    }
</style>
"""


def build_container(content: str, css_class: str = "evo") -> str:
    """Wrap content in a styled container div.
    
    :param content: HTML content to wrap.
    :param css_class: CSS class for the container (default: "evo").
    :return: Wrapped HTML string.
    """
    return f'{STYLESHEET}<div class="{css_class}">{content}</div>'


def build_title(text: str, links: list[tuple[str, str]] | None = None) -> str:
    """Create a styled title div.
    
    :param text: Title text.
    :param links: Optional list of (label, url) tuples for links next to the title.
    :return: HTML string.
    """
    if links:
        link_html = ' | '.join([f'<a href="{url}" target="_blank">{label}</a>' for label, url in links])
        return f'<div class="title"><span>{text}</span><span class="title-links">{link_html}</span></div>'
    return f'<div class="title">{text}</div>'


def build_table_row(label: str, value: str, is_last: bool = False) -> str:
    """Create a table row with label and value.
    
    :param label: Label text.
    :param value: Value text (can contain HTML).
    :param is_last: If True, don't add bottom border.
    :return: HTML string.
    """
    return f'<tr><td class="label">{label}</td><td class="value">{value}</td></tr>'


def build_table_row_vtop(label: str, value: str, is_last: bool = False) -> str:
    """Create a table row with label and value (label top-aligned).
    
    :param label: Label text.
    :param value: Value text (can contain HTML).
    :param is_last: If True, don't add bottom border.
    :return: HTML string.
    """
    return f'<tr><td class="label-vtop">{label}</td><td class="value">{value}</td></tr>'


def build_section_divider(title: str) -> str:
    """Create a section divider with title.
    
    :param title: Section title.
    :return: HTML string.
    """
    return f'<div class="section"><div class="section-heading">{title}</div>'


def build_table(rows: list[tuple[str, str]]) -> str:
    """Build an HTML table from rows of (label, value) tuples.
    
    :param rows: List of (label, value) tuples.
    :return: HTML string for a complete table.
    """
    table_rows = [build_table_row(label, value) for label, value in rows]
    return f'<table>{"".join(table_rows)}</table>'


def build_nested_table(headers: list[str], rows: list[list[str]], css_class: str = "") -> str:
    """Build a nested HTML table with headers and data rows.
    
    :param headers: List of header strings.
    :param rows: List of data rows, where each row is a list of cell values.
    :param css_class: Additional CSS classes to add to the table (optional).
    :return: HTML string for a nested table.
    """
    class_attr = f' class="nested {css_class}"' if css_class else ' class="nested"'
    
    # Build header row
    header_cells = []
    for i, header in enumerate(headers):
        header_cells.append(f'<th>{header}</th>')
    
    # Build data rows
    data_rows = []
    for row in rows:
        cells = []
        for i, cell in enumerate(row):
            # Format numeric values
            if isinstance(cell, (int, float)):
                formatted_cell = f"{cell:.2f}"
            else:
                formatted_cell = str(cell)
            cells.append(f'<td>{formatted_cell}</td>')
        data_rows.append(f'<tr>{"".join(cells)}</tr>')
    
    return (
        f'<table{class_attr}>'
        f'<tr>{"".join(header_cells)}</tr>'
        f'{"".join(data_rows)}'
        f'</table>'
    )


def build_object_html(title: str, rows: list[tuple[str, str]], extra_content: str = "") -> str:
    """Build a complete object HTML representation.
    
    :param title: Object title/name.
    :param rows: List of (label, value) tuples for the main properties table.
    :param extra_content: Additional HTML content to append after the table (optional).
    :return: Complete HTML string with stylesheet.
    """
    html_parts = [
        STYLESHEET,
        '<div class="evo">',
        build_title(title),
        build_table(rows),
        extra_content,
        '</div>'
    ]
    return ''.join(html_parts)
