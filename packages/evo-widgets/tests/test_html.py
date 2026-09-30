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
    build_section_divider,
    build_table,
    build_table_row,
    build_table_row_vtop,
    build_title,
    markup,
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
        """Test building a container with default class."""
        self.assertEqual(build_container("content"), f'{STYLESHEET}<div class="evo">content</div>')

    def test_builds_container_with_custom_class(self):
        """Test building a container with custom class."""
        self.assertEqual(build_container("content", css_class="custom"), f'{STYLESHEET}<div class="custom">content</div>')

    def test_escapes_container_content_and_class(self):
        """Container content and class values cannot break out into markup."""
        result = build_container("<content>", css_class='custom" onclick="alert(1)')
        self.assertEqual(
            result,
            f'{STYLESHEET}<div class="custom&quot; onclick=&quot;alert(1)">&lt;content&gt;</div>',
        )

    def test_accepts_explicit_container_markup(self):
        """Explicitly trusted container content remains markup."""
        self.assertEqual(build_container(markup("<strong>content</strong>")), f'{STYLESHEET}<div class="evo"><strong>content</strong></div>')


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

    def test_escapes_title_and_link_values(self):
        """Document-derived title and link text cannot become markup."""
        result = build_title('<title>', [("<link>", 'https://example.com/?q="x"')])
        self.assertEqual(
            result,
            '<div class="title"><span>&lt;title&gt;</span><span class="title-links">'
            '<a href="https://example.com/?q=&quot;x&quot;" target="_blank">&lt;link&gt;</a>'
            "</span></div>",
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
        result = build_table_row_vtop("Attributes:", markup("<table>...</table>"))
        self.assertEqual(
            result,
            '<tr><td class="label-vtop">Attributes:</td><td class="value"><table>...</table></td></tr>',
        )

    def test_escapes_text_in_table_row_vtop(self):
        """Top-aligned rows escape untrusted labels and values."""
        result = build_table_row_vtop('<label>', '<script>alert("x")</script>')
        self.assertEqual(
            result,
            '<tr><td class="label-vtop">&lt;label&gt;</td><td class="value">&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt;</td></tr>',
        )

    def test_escapes_text_and_accepts_explicit_markup(self):
        """Text is escaped while package-created markup remains HTML."""
        result = build_table_row('<label>', '<script>alert("x")</script>')
        self.assertEqual(
            result,
            '<tr><td class="label">&lt;label&gt;</td><td class="value">&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt;</td></tr>',
        )


class TestBuildSectionDivider(unittest.TestCase):
    """Tests for the build_section_divider function."""

    def test_builds_section_divider(self):
        """Test building a section divider."""
        self.assertEqual(
            build_section_divider("Details"),
            '<div class="section"><div class="section-heading">Details</div>',
        )

    def test_escapes_section_divider_title(self):
        """Section divider titles are treated as text."""
        self.assertEqual(
            build_section_divider("<title>"),
            '<div class="section"><div class="section-heading">&lt;title&gt;</div>',
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
            '<table class="nested extra"><tr><th>Col</th></tr><tr><td>val</td></tr></table>',
        )

    def test_escapes_nested_table_custom_class(self):
        """Custom classes cannot break out of the table class attribute."""
        result = build_nested_table(["Col"], [["val"]], css_class='extra" onclick="alert(1)')
        self.assertEqual(
            result,
            '<table class="nested extra&quot; onclick=&quot;alert(1)"><tr><th>Col</th></tr><tr><td>val</td></tr></table>',
        )

    def test_escapes_nested_table_leaf_values(self):
        """Nested table structure remains intact while document values are escaped."""
        result = build_nested_table(["<header>"], [["<value>", markup("<strong>bold</strong>")]])
        self.assertEqual(
            result,
            '<table class="nested"><tr><th>&lt;header&gt;</th></tr><tr><td>&lt;value&gt;</td><td><strong>bold</strong></td></tr></table>',
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
        result = build_object_html("Title", rows, extra_content=markup("<div>Extra</div>"))
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

    def test_escapes_object_html_extra_content(self):
        """Object extra content is text unless explicitly marked as HTML."""
        result = build_object_html("Title", [], extra_content="<div>Extra</div>")
        self.assertEqual(
            result,
            f'{STYLESHEET}<div class="evo"><div class="title">Title</div><table></table>&lt;div&gt;Extra&lt;/div&gt;</div>',
        )


if __name__ == "__main__":
    unittest.main()
