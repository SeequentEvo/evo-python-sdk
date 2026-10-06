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

from evo.common.typed import Point3, Size3d, Size3i


class ArrayLike:
    """Minimal NumPy-style array interface, without importing NumPy."""

    def __init__(self, values, ndim=1):
        self.values = values
        self.ndim = ndim
        self.converted = False

    def __len__(self):
        return len(self.values)

    def tolist(self):
        self.converted = True
        return self.values


@pytest.mark.parametrize(
    ("convert", "values", "expected"),
    [
        (Point3.from_array_like, [1, 2, 3], Point3(1.0, 2.0, 3.0)),
        (Size3i.from_array_like, [1, 2, 3], Size3i(1, 2, 3)),
        (Size3d.from_array_like, [1, 2, 3], Size3d(1.0, 2.0, 3.0)),
    ],
)
def test_array_like_inputs_without_numpy(convert, values, expected):
    result = convert(ArrayLike(values))
    assert result == expected
    assert all(type(item) is type(reference) for item, reference in zip(result, expected))
    assert convert(expected) == expected


@pytest.mark.parametrize("ndim", [0, 2])
def test_array_like_requires_one_dimension(ndim):
    with pytest.raises(ValueError, match="one-dimensional"):
        Point3.from_array_like(ArrayLike([1, 2, 3], ndim=ndim))


def test_array_like_requires_three_values():
    value = ArrayLike([1, 2])
    with pytest.raises(ValueError, match="exactly three"):
        Size3d.from_array_like(value)
    assert not value.converted


@pytest.mark.parametrize(
    ("convert", "values"),
    [
        (Point3.from_array_like, [1, float("inf"), 3]),
        (Point3.from_array_like, [1, True, 3]),
        (Size3d.from_array_like, [1, 0, 3]),
        (Size3i.from_array_like, [1, 2.0, 3]),
        (Size3i.from_array_like, [1, -2, 3]),
    ],
)
def test_array_like_rejects_invalid_elements(convert, values):
    with pytest.raises(ValueError, match="value must contain three"):
        convert(values)


@pytest.mark.parametrize(
    ("convert", "value"),
    [
        (Point3.from_array_like, Point3(float("nan"), 2, 3)),
        (Size3d.from_array_like, Size3d(1, 0, 3)),
        (Size3i.from_array_like, Size3i(1, 2.0, 3)),
    ],
)
def test_array_like_rejects_invalid_existing_instances(convert, value):
    with pytest.raises(ValueError, match="value must contain three"):
        convert(value)
