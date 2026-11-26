#  Copyright © 2026 Bentley Systems, Incorporated
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Tests for offline per-topic Markdown task documentation generation."""

from __future__ import annotations

import re
import unittest

from evo.compute._docgen import _CODE_SPAN_SPLIT_RE, render_topic_docs
from evo.compute._stubgen import DEFAULT_SNAPSHOT_DIR, load_snapshot
from evo.compute.engine import _normalise


def _count_unescaped_pipes_outside_code(line: str) -> int:
    """Count table-column separator pipes (pipes outside backtick spans and not preceded by backslash)."""
    parts = _CODE_SPAN_SPLIT_RE.split(line)
    total = 0
    for idx in range(0, len(parts), 2):
        total += len(re.findall(r"(?<!\\)\|", parts[idx]))
    return total


class TestTopicDocGen(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.pages = render_topic_docs(DEFAULT_SNAPSHOT_DIR)
        cls.tasks = load_snapshot(DEFAULT_SNAPSHOT_DIR)

    def test_renders_every_topic_and_task_in_snapshot(self) -> None:
        expected_topics = sorted({task.topic for task in self.tasks})
        self.assertEqual(expected_topics, sorted(self.pages.keys()))

        for task in self.tasks:
            page = self.pages[task.topic]
            task_attr = _normalise(task.name)
            self.assertIn(f"### {task_attr}\n", page)
            self.assertIn(f"Task: `{task.topic}/{task.name}`", page)

    def test_includes_parameters_hydrated_results_and_nested_shapes(self) -> None:
        geostats = self.pages["geostatistics"]
        self.assertIn("await client.geostatistics.declustering.run(", geostats)
        self.assertIn("Supported schemas:", geostats)
        self.assertIn("await result.target.load()", geostats)
        self.assertIn("#### Nested Shapes", geostats)
        self.assertIn("await node.attribute.load()", geostats)

    def test_portal_and_mdx_formatting_invariants(self) -> None:
        for topic, content in self.pages.items():
            with self.subTest(topic=topic):
                self.assertTrue(
                    content.startswith("[GitHub source](https://github.com/SeequentEvo/evo-python-sdk/"),
                    "Topic page must start with auto-generated GitHub source marker",
                )
                self.assertNotIn("<a href=", content, "Relative HTML anchor tags must be converted to full links")

                lines = content.splitlines()
                expected_pipes: int | None = None
                for line_no, line in enumerate(lines, 1):
                    if line.startswith("## "):
                        self.assertNotIn(
                            ".",
                            line,
                            f"{topic}:{line_no}: H2 headings must not contain dots (Dev Portal class-splitting rule)",
                        )
                    if line.startswith("|"):
                        pipes = _count_unescaped_pipes_outside_code(line)
                        if expected_pipes is None:
                            expected_pipes = pipes
                        else:
                            self.assertEqual(
                                expected_pipes,
                                pipes,
                                f"{topic}:{line_no}: inconsistent GFM table column count in {line!r}",
                            )
                    else:
                        expected_pipes = None


if __name__ == "__main__":
    unittest.main()
