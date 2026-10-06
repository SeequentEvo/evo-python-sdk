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

import numpy as np
import pandas as pd
import pytest

from evo.objects.typed import Point3, Regular3DGridData, Size3d, Size3i
from evo.objects.typed.regular_masked_grid import RegularMasked3DGridData
from evo.objects.typed.tensor_grid import Tensor3DGridData


def _create_grid(cls, **geometry):
    kwargs = {"name": "Test", "origin": [1, 2, 3], "size": [2, 2, 2]}
    if cls is Tensor3DGridData:
        kwargs.update(cell_sizes_x=np.ones(2), cell_sizes_y=np.ones(2), cell_sizes_z=np.ones(2))
    else:
        kwargs["cell_size"] = [0.5, 1.5, 2.5]
        if cls is RegularMasked3DGridData:
            kwargs["mask"] = np.ones(8, dtype=bool)
    kwargs.update(geometry)
    return cls(**kwargs)


@pytest.mark.parametrize(
    ("origin", "size"),
    [
        ([1, 2, 3], [2, 2, 2]),
        ((1, 2, 3), (2, 2, 2)),
        (np.array([1, 2, 3]), np.array([2, 2, 2])),
        (Point3(1, 2, 3), Size3i(2, 2, 2)),
    ],
)
@pytest.mark.parametrize("grid_class", [Regular3DGridData, RegularMasked3DGridData, Tensor3DGridData])
def test_grid_geometry_normalizes_before_data_validation(grid_class, origin, size):
    data = _create_grid(grid_class, origin=origin, size=size, cell_data=pd.DataFrame({"value": range(8)}))
    assert type(data.origin) is Point3
    assert type(data.size) is Size3i
    assert data.origin == Point3(1, 2, 3)
    assert data.size == Size3i(2, 2, 2)


@pytest.mark.parametrize(
    "cell_size", [[0.5, 1.5, 2.5], (0.5, 1.5, 2.5), np.array([0.5, 1.5, 2.5]), Size3d(0.5, 1.5, 2.5)]
)
@pytest.mark.parametrize("grid_class", [Regular3DGridData, RegularMasked3DGridData])
def test_regular_grid_cell_size_normalizes(grid_class, cell_size):
    data = _create_grid(grid_class, cell_size=cell_size)
    assert type(data.cell_size) is Size3d
    assert data.cell_size == Size3d(0.5, 1.5, 2.5)


@pytest.mark.parametrize("grid_class", [Regular3DGridData, RegularMasked3DGridData, Tensor3DGridData])
@pytest.mark.parametrize(
    ("field", "value"),
    [("origin", Point3(float("nan"), 2, 3)), ("size", Size3i(0, 2, 2))],
)
def test_grid_rejects_invalid_named_geometry(grid_class, field, value):
    with pytest.raises(ValueError, match="value must contain three"):
        _create_grid(grid_class, **{field: value})


@pytest.mark.parametrize("grid_class", [Regular3DGridData, RegularMasked3DGridData, Tensor3DGridData])
@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("origin", [1, 2], ValueError),
        ("origin", np.array([1, 2]), ValueError),
        ("origin", np.array([[1, 2, 3]]), ValueError),
        ("origin", np.array(1), ValueError),
        ("origin", np.array(["1", "2", "3"]), ValueError),
        ("origin", [float("nan"), 2, 3], ValueError),
        ("size", [2, 2], ValueError),
        ("size", np.array([2, 2.5, 2]), ValueError),
        ("size", np.array([2, True, 2], dtype=object), ValueError),
        ("size", [2, 0, 2], ValueError),
        ("size", [2, 1.5, 2], ValueError),
        ("size", [2, True, 2], ValueError),
    ],
)
def test_grid_rejects_invalid_geometry(grid_class, field, value, error):
    with pytest.raises(error, match="value must"):
        _create_grid(grid_class, **{field: value})


@pytest.mark.parametrize("grid_class", [Regular3DGridData, RegularMasked3DGridData])
@pytest.mark.parametrize(
    "cell_size", [[1, 2], [1, 0, 3], [1, float("inf"), 3], np.array([1, 2]), np.array([1, np.inf, 3]), Size3d(0, 1, 1)]
)
def test_regular_grid_rejects_invalid_cell_size(grid_class, cell_size):
    with pytest.raises(ValueError, match="value must"):
        _create_grid(grid_class, cell_size=cell_size)
