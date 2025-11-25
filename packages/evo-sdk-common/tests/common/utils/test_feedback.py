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

import unittest
from unittest import mock

from parameterized import parameterized

from evo.common.interfaces import IFeedback
from evo.common.utils import NoFeedback, PartialFeedback, iter_with_fb, split_feedback


class TestFeedback(unittest.TestCase):
    def setUp(self) -> None:
        self.parent_fb = mock.Mock(spec=IFeedback)

    def test_partial_feedback_full_range(self) -> None:
        fb = PartialFeedback(self.parent_fb, 0, 1)

        for p in [0.0, 0.0001, 0.1, 0.4999, 0.5, 0.5001, 0.9, 0.9999, 1.0]:
            fb.progress(p)
            self.parent_fb.progress.assert_called_once_with(p, None)
            self.parent_fb.reset_mock()

    @parameterized.expand(
        [
            ("first ten percent", 0.0, 0.1, [0.0, 0.0, 0.01, 0.05, 0.05, 0.05, 0.09, 0.1, 0.1]),
            ("middle of the range", 0.4, 0.6, [0.4, 0.4, 0.42, 0.5, 0.5, 0.5, 0.58, 0.6, 0.6]),
            ("last ten percent", 0.9, 1.0, [0.9, 0.9, 0.91, 0.95, 0.95, 0.95, 0.99, 1.0, 1.0]),
            ("very small part", 0.002, 0.003, [0.002, 0.002, 0.0021, 0.0025, 0.0025, 0.0025, 0.0029, 0.003, 0.003]),
        ]
    )
    def test_partial_feedback_partial_range(
        self, _name: str, start: float, end: float, expected_values: list[float]
    ) -> None:
        fb = PartialFeedback(self.parent_fb, start, end)

        for p, expected_p in zip([0.0, 0.0001, 0.1, 0.4999, 0.5, 0.5001, 0.9, 0.9999, 1.0], expected_values):
            fb.progress(p)
            self.parent_fb.progress.assert_called_once_with(expected_p, None)
            self.parent_fb.reset_mock()

    def test_iter_with_fb(self) -> None:
        n_elements = 10
        elements = [object() for _ in range(n_elements)]
        parent_fb: IFeedback = self.parent_fb
        for i, (element, fb) in enumerate(iter_with_fb(elements, parent_fb)):
            self.assertIs(elements[i], element)
            fb_part = i / n_elements

            with self.subTest("iter_with_fb updates progress"):
                if i > 0:  # FB is updated after each action.
                    self.parent_fb.progress.assert_called_once_with(fb_part)

            self.parent_fb.reset_mock()
            fb_part = round(fb_part + 0.5 / n_elements, ndigits=4)

            with self.subTest("fb from iter_with_fb updates progress"):
                fb.progress(0.5, "message")
                self.parent_fb.progress.assert_called_once_with(fb_part, "message")

            self.parent_fb.reset_mock()

        # FB is updated after each action.
        self.parent_fb.progress.assert_called_once_with(1.0)

    def test_iter_with_no_fb(self) -> None:
        n_elements = 10
        elements = [object() for _ in range(n_elements)]
        for i, (element, fb) in enumerate(iter_with_fb(elements)):
            self.assertIs(elements[i], element)
            self.assertIs(NoFeedback, fb)

    def test_iter_with_fb_no_elements(self) -> None:
        for _ in iter_with_fb([], self.parent_fb):
            raise AssertionError("Unreachable")

        self.parent_fb.progress.assert_not_called()

    def test_split_feedback_no_elements(self) -> None:
        self.assertEqual(split_feedback(self.parent_fb, []), [])

    @parameterized.expand(
        [
            ("weighted", [1, 2, 3], [0.0, 0.1667, 0.6667, 9 / 12, 0.8333]),
            ("equal weights", [1, 1, 1], [0.0, 0.1667, 1 / 2, 0.6667, 9 / 12]),
            ("zero weight", [0, 0, 0], [0.0, 0.1667, 1 / 2, 0.6667, 9 / 12]),
            ("mixed weights", [12, 0, 3], [0.0, 0.0, 1 / 5, 3 / 5, 3 / 5]),
        ]
    )
    def test_split_feedback(
        self,
        _name: str,
        weights: list[float],
        expected_progress_values: list[float],
    ) -> None:
        fb_1, fb_2, fb_3 = split_feedback(self.parent_fb, weights)

        fb_1.progress(0.0)
        self.parent_fb.progress.assert_called_once_with(expected_progress_values[0], None)
        self.parent_fb.reset_mock()

        fb_2.progress(0.5, "halfway")
        self.parent_fb.progress.assert_called_once_with(expected_progress_values[1], "halfway")
        self.parent_fb.reset_mock()

        fb_3.progress(1.0)
        self.parent_fb.progress.assert_called_once_with(expected_progress_values[2], None)
        self.parent_fb.reset_mock()

        fb_1.progress(0.5, "halfway in first")
        self.parent_fb.progress.assert_called_once_with(expected_progress_values[3], "halfway in first")
        self.parent_fb.reset_mock()

        fb_2.progress(0.75)
        self.parent_fb.progress.assert_called_once_with(expected_progress_values[4], None)
        self.parent_fb.reset_mock()
