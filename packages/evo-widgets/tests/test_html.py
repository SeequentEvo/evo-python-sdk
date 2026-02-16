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

"""Tests for evo.widgets.html module."""

import unittest

from evo.widgets.html import (
    STYLESHEET,
    build_container,
    build_nested_table,
    build_object_html,
    build_table,
    build_table_row,
    build_table_row_vtop,
    build_title,
)


class TestStylesheet(unittest.TestCase):
    """Tests for the STYLESHEET constant."""

    def test_stylesheet_contains_evo_class(self):
        """Test that STYLESHEET contains the .evo class definition."""
        self.assertIn(".evo", STYLESHEET)
        self.assertIn("<style>", STYLESHEET)
        self.assertIn("</style>", STYLESHEET)


class TestBuildContainer(unittest.TestCase):
    """Tests for the build_container function."""

    def test_builds_container_with_default_class(self):
        """Test building a container with default CSS class."""
        result = build_container("content")
        self.assertEqual(
            result,
            f'{STYLESHEET}<div class="evo">content</div>',
        )

    def test_builds_container_with_custom_class(self):
        """Test building a container with custom CSS class."""
        result = build_container("content", css_class="custom")
        self.assertEqual(
            result,
            f'{STYLESHEET}<div class="custom">content</div>',
        )


class TestBuildTitle(unittest.TestCase):
    """Tests for the build_title function."""

    def test_builds_title_without_links(self):
        """Test building a title without links."""
        result = build_title("My Title")
        self.assertEqual(
            result,
            '<div class="title">My Title</div>',
        )

    def test_builds_title_with_links(self):
        """Test building a title with links."""
        links = [("Portal", "https://portal.example.com"), ("Viewer", "https://viewer.example.com")]
        result = build_title("My Title", links)
        self.assertEqual(
            result,
            '<div class="title">'
            "<span>My Title</span>"
            '<span class="title-links">'
            '<a href="https://portal.example.com" target="_blank">Portal</a>'
            " | "
            '<a href="https://viewer.example.com" target="_blank">Viewer</a>'
            "</span>"
            "</div>",
        )


class TestBuildTableRow(unittest.TestCase):
    """Tests for the build_table_row function."""

    def test_builds_table_row(self):
        """Test building a table row."""
        result = build_table_row("Name:", "Test Object")
        self.assertEqual(
            result,
            '<tr><td class="label">Name:</td><td class="value">Test Object</td></tr>',
        )

    def test_builds_table_row_vtop(self):
        """Test building a table row with vertical-top alignment."""
        result = build_table_row_vtop("Attributes:", "<table>...</table>")
        self.assertEqual(
            result,
            '<tr><td class="label-vtop">Attributes:</td><td class="value"><table>...</table></td></tr>',
        )


class TestBuildTable(unittest.TestCase):
    """Tests for the build_table function."""

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

    def test_builds_empty_table(self):
        """Table builds with no rows."""
        result = build_table([])
        self.assertEqual(result, "<table></table>")


class TestBuildNestedTable(unittest.TestCase):
    """Tests for the build_nested_table function."""

    def test_builds_nested_table(self):
        """Test building a nested table with headers and rows."""
        headers = ["Name", "Type"]
        rows = [["grade", "scalar"], ["rock_type", "category"]]
        result = build_nested_table(headers, rows)
        self.assertEqual(
            result,
            '<table class="nested">'
            "<tr><th>Name</th><th>Type</th></tr>"
            "<tr><td>grade</td><td>scalar</td></tr>"
            "<tr><td>rock_type</td><td>category</td></tr>"
            "</table>",
        )

    def test_formats_numeric_values(self):
        """Test that numeric values are formatted correctly."""
        headers = ["Label", "Min", "Max"]
        rows = [["X:", 0.0, 100.5]]
        result = build_nested_table(headers, rows)
        self.assertEqual(
            result,
            '<table class="nested">'
            "<tr><th>Label</th><th>Min</th><th>Max</th></tr>"
            "<tr><td>X:</td><td>0.00</td><td>100.50</td></tr>"
            "</table>",
        )

    def test_builds_nested_table_with_custom_class(self):
        """Test building a nested table with custom CSS class."""
        headers = ["Col"]
        rows = [["val"]]
        result = build_nested_table(headers, rows, css_class="extra")
        self.assertEqual(
            result,
            '<table class="nested extra">'
            "<tr><th>Col</th></tr>"
            "<tr><td>val</td></tr>"
            "</table>",
        )


class TestBuildObjectHtml(unittest.TestCase):
    """Tests for the build_object_html function."""

    def test_builds_complete_object_html(self):
        """Test building a complete object HTML representation."""
        rows = [("ID:", "12345"), ("Type:", "PointSet")]
        result = build_object_html("My Object", rows)
        self.assertEqual(
            result,
            f"{STYLESHEET}"
            '<div class="evo">'
            '<div class="title">My Object</div>'
            "<table>"
            '<tr><td class="label">ID:</td><td class="value">12345</td></tr>'
            '<tr><td class="label">Type:</td><td class="value">PointSet</td></tr>'
            "</table>"
            "</div>",
        )

    def test_builds_object_html_with_extra_content(self):
        """Test building object HTML with extra content."""
        rows = [("Name:", "Test")]
        result = build_object_html("Title", rows, extra_content="<div>Extra</div>")
        self.assertEqual(
            result,
            f"{STYLESHEET}"
            '<div class="evo">'
            '<div class="title">Title</div>'
            "<table>"
            '<tr><td class="label">Name:</td><td class="value">Test</td></tr>'
            "</table>"
            "<div>Extra</div>"
            "</div>",
        )


if __name__ == "__main__":
    unittest.main()

