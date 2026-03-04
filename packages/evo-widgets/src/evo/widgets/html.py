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

from pathlib import Path

# Load CSS from external file
_CSS_PATH = Path(__file__).parent / "assets" / "styles.css"
_CSS_CONTENT = _CSS_PATH.read_text(encoding="utf-8")

# CSS Stylesheet wrapped in style tags for HTML representations
STYLESHEET = f"<style>\n{_CSS_CONTENT}</style>\n"


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
        link_html = " | ".join([f'<a href="{url}" target="_blank">{label}</a>' for label, url in links])
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
    return f"<table>{''.join(table_rows)}</table>"


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
        header_cells.append(f"<th>{header}</th>")

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
            cells.append(f"<td>{formatted_cell}</td>")
        data_rows.append(f"<tr>{''.join(cells)}</tr>")

    return f"<table{class_attr}><tr>{''.join(header_cells)}</tr>{''.join(data_rows)}</table>"


def build_object_html(title: str, rows: list[tuple[str, str]], extra_content: str = "") -> str:
    """Build a complete object HTML representation.

    :param title: Object title/name.
    :param rows: List of (label, value) tuples for the main properties table.
    :param extra_content: Additional HTML content to append after the table (optional).
    :return: Complete HTML string with stylesheet.
    """
    html_parts = [STYLESHEET, '<div class="evo">', build_title(title), build_table(rows), extra_content, "</div>"]
    return "".join(html_parts)
