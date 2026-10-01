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

import pytest

from evo.common.typed import Point3, Size3d, Size3i, as_point3, as_size3d, as_size3i


class ArrayLike:
    """Minimal NumPy-style array interface, without importing NumPy."""

    def __init__(self, values, ndim=1):
        self.values = values
        self.ndim = ndim

    def tolist(self):
        return self.values


@pytest.mark.parametrize(
    ("convert", "values", "expected"),
    [
        (as_point3, [1.0, 2.0, 3.0], Point3(1.0, 2.0, 3.0)),
        (as_size3i, [1, 2, 3], Size3i(1, 2, 3)),
        (as_size3d, [1.0, 2.0, 3.0], Size3d(1.0, 2.0, 3.0)),
    ],
)
def test_array_like_inputs_without_numpy(convert, values, expected):
    assert convert(ArrayLike(values)) == expected
    assert convert(expected) is expected


@pytest.mark.parametrize("ndim", [0, 2])
def test_array_like_requires_one_dimension(ndim):
    with pytest.raises(ValueError, match="one-dimensional"):
        as_point3(ArrayLike([1, 2, 3], ndim=ndim))


def test_array_like_requires_three_values():
    with pytest.raises(ValueError, match="exactly three"):
        as_size3d(ArrayLike([1, 2]))
