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

"""Tests for HTML style utilities."""

import unittest

from evo.common.styles.html import (
    build_container,
    build_nested_table,
    build_object_html,
    build_section_divider,
    build_table,
    build_table_row,
    build_table_row_vtop,
    build_title,
)


class TestBuildContainer(unittest.TestCase):
    """Tests for build_container function."""

    def test_default_css_class(self):
        """Container uses 'evo' class by default."""
        result = build_container("content")
        self.assertIn(
            '<div class="evo">content</div>',
            result,
        )

    def test_custom_css_class(self):
        """Container uses custom class when provided."""
        result = build_container("content", css_class="custom")
        self.assertIn(
            '<div class="custom">content</div>',
            result,
        )

    def test_includes_stylesheet(self):
        """Container includes the stylesheet."""
        result = build_container("content")
        self.assertIn("<style>", result)


class TestBuildTitle(unittest.TestCase):
    """Tests for build_title function."""

    def test_simple_title(self):
        """Simple title without links."""
        result = build_title("My Title")
        self.assertEqual(result, '<div class="title">My Title</div>')

    def test_title_with_links(self):
        """Title with links displays properly."""
        links = [("Portal", "https://example.com"), ("Viewer", "https://viewer.com")]
        result = build_title("Object", links)
        self.assertIn(
            '<div class="title">'
            "<span>Object</span>"
            '<span class="title-links">'
            '<a href="https://example.com" target="_blank">Portal</a>'
            " | "
            '<a href="https://viewer.com" target="_blank">Viewer</a>'
            "</span>"
            "</div>",
            result,
        )

    def test_links_open_in_new_tab(self):
        """Links have target='_blank' attribute."""
        links = [("Link", "https://example.com")]
        result = build_title("Title", links)
        self.assertIn(
            '<a href="https://example.com" target="_blank">Link</a>',
            result,
        )


class TestBuildTableRow(unittest.TestCase):
    """Tests for build_table_row function."""

    def test_basic_row(self):
        """Basic row with label and value."""
        result = build_table_row("Name", "Test Object")
        self.assertEqual(
            result,
            '<tr><td class="label">Name</td><td class="value">Test Object</td></tr>',
        )

    def test_row_with_html_value(self):
        """Row can contain HTML in value."""
        result = build_table_row("Link", '<a href="#">Click</a>')
        self.assertEqual(
            result,
            '<tr><td class="label">Link</td><td class="value"><a href="#">Click</a></td></tr>',
        )


class TestBuildTableRowVtop(unittest.TestCase):
    """Tests for build_table_row_vtop function."""

    def test_uses_vtop_class(self):
        """Vtop row uses label-vtop class."""
        result = build_table_row_vtop("Name", "Value")
        self.assertEqual(
            result,
            '<tr><td class="label-vtop">Name</td><td class="value">Value</td></tr>',
        )


class TestBuildSectionDivider(unittest.TestCase):
    """Tests for build_section_divider function."""

    def test_creates_section_with_heading(self):
        """Section divider creates section opening with heading (no closing tag)."""
        result = build_section_divider("Details")
        self.assertEqual(
            result,
            '<div class="section"><div class="section-heading">Details</div>',
        )


class TestBuildTable(unittest.TestCase):
    """Tests for build_table function."""

    def test_builds_table_from_rows(self):
        """Table builds from list of tuples."""
        rows = [("Name", "Test"), ("Type", "PointSet")]
        result = build_table(rows)
        self.assertEqual(
            result,
            "<table>"
            '<tr><td class="label">Name</td><td class="value">Test</td></tr>'
            '<tr><td class="label">Type</td><td class="value">PointSet</td></tr>'
            "</table>",
        )

    def test_empty_rows(self):
        """Empty rows produces empty table."""
        result = build_table([])
        self.assertEqual(result, "<table></table>")


class TestBuildNestedTable(unittest.TestCase):
    """Tests for build_nested_table function."""

    def test_builds_table_with_headers(self):
        """Nested table includes headers."""
        headers = ["Column", "Type"]
        rows = [["grade", "float"]]
        result = build_nested_table(headers, rows)
        self.assertEqual(
            result,
            '<table class="nested">'
            "<tr><th>Column</th><th>Type</th></tr>"
            "<tr><td>grade</td><td>float</td></tr>"
            "</table>",
        )

    def test_builds_multiple_data_rows(self):
        """Nested table includes multiple data rows."""
        headers = ["Name", "Value"]
        rows = [["a", "1"], ["b", "2"]]
        result = build_nested_table(headers, rows)
        self.assertEqual(
            result,
            '<table class="nested">'
            "<tr><th>Name</th><th>Value</th></tr>"
            "<tr><td>a</td><td>1</td></tr>"
            "<tr><td>b</td><td>2</td></tr>"
            "</table>",
        )

    def test_formats_numeric_values(self):
        """Numeric values are formatted with 2 decimal places."""
        headers = ["Value"]
        rows: list[list] = [[3.14159], [42]]
        result = build_nested_table(headers, rows)
        self.assertEqual(
            result,
            '<table class="nested">'
            "<tr><th>Value</th></tr>"
            "<tr><td>3.14</td></tr>"
            "<tr><td>42.00</td></tr>"
            "</table>",
        )

    def test_custom_css_class(self):
        """Custom CSS class is appended."""
        result = build_nested_table(["H"], [["v"]], css_class="extra")
        self.assertEqual(
            result,
            '<table class="nested extra">'
            "<tr><th>H</th></tr>"
            "<tr><td>v</td></tr>"
            "</table>",
        )


class TestBuildObjectHtml(unittest.TestCase):
    """Tests for build_object_html function."""

    def test_builds_complete_html(self):
        """Object HTML includes title and table rows."""
        rows = [("Name", "My Object"), ("Type", "PointSet")]
        result = build_object_html("My Object", rows)
        self.assertIn(
            '<div class="title">My Object</div>'
            "<table>"
            '<tr><td class="label">Name</td><td class="value">My Object</td></tr>'
            '<tr><td class="label">Type</td><td class="value">PointSet</td></tr>'
            "</table>",
            result,
        )

    def test_includes_extra_content(self):
        """Extra content is appended after the table."""
        rows = [("Name", "Test")]
        extra = '<div class="section">Extra Section</div>'
        result = build_object_html("Test", rows, extra_content=extra)
        self.assertIn(
            "</table>"
            '<div class="section">Extra Section</div>'
            "</div>",
            result,
        )


if __name__ == "__main__":
    unittest.main()
