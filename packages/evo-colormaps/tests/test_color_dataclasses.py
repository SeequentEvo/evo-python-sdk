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

from evo.colormaps.data import CategoryColormap, ContinuousColormap, DiscreteColormap, _validate_colors


class TestValidateColors(unittest.TestCase):
    def test_valid_colors(self):
        colors = [[255, 240, 219], [238, 217, 196], [217, 185, 155]]
        try:
            _validate_colors(colors)
        except ValueError:
            self.fail("_validate_colors raised ValueError unexpectedly!")

    def test_valid_colours_for_category(self):
        colors = [[253, 240, 219]]
        _validate_colors(colors, minimum=1)

    def test_colors_out_of_range(self):
        colors = [[256, 240, 219], [238, 217, 196], [217, 185, 155]]
        with self.assertRaises(ValueError):
            _validate_colors(colors)

    def test_colors_incorrect_length(self):
        colors = [[255, 240], [238, 217, 196], [217, 185, 155]]
        with self.assertRaises(ValueError):
            _validate_colors(colors)

    def test_colors_too_many_entries(self):
        colors = [[255, 240, 219]] * 1025
        with self.assertRaises(ValueError):
            _validate_colors(colors)

    def test_colors_too_few_entries(self):
        colors = [[255, 240, 219]]
        with self.assertRaises(ValueError):
            _validate_colors(colors)

    def test_colors_wrong_numeric_data_type(self):
        colors = [[255, 240, 219], [238, 217, 196], [217, 185, 155], [1.0, 0.5, 0.2]]
        with self.assertRaises(ValueError):
            _validate_colors(colors)

    def test_colors_wrong_data_type(self):
        colors = [[255, 240, 219], [238, 217, 196], [217, 185, 155], ["a", "b", "c"]]
        with self.assertRaises(ValueError):
            _validate_colors(colors)

    @parameterized.expand(
        [
            ("continuous", 2, 1024, ContinuousColormap),
            ("discrete", 1, 1024, DiscreteColormap),
            ("category", 1, 10000, CategoryColormap),
        ]
    )
    def test_colormap_creation_validates(self, _, minimum_colors, maximum_colours, colormap_dataclass):
        kwargs = {param: mock.MagicMock() for param in colormap_dataclass.__dataclass_fields__ if param != "colors"}
        with mock.patch("evo.colormaps.data._validate_colors") as mock_validate_colors:
            colors = mock.MagicMock()
            colormap_dataclass(colors=colors, **kwargs)
            mock_validate_colors.assert_called_once_with(colors, minimum=minimum_colors, maximum=maximum_colours)
