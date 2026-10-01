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

"""Common geometry types shared across Evo SDK packages.

These types provide a lightweight, dependency-free representation of common
3D geometry primitives used by block models, grids, and other spatial objects.
"""

from __future__ import annotations

from math import isfinite
from numbers import Integral, Real
from typing import NamedTuple

__all__ = [
    "BoundingBox",
    "Point3",
    "Size3d",
    "Size3i",
    "as_point3",
    "as_size3d",
    "as_size3i",
]


class Point3(NamedTuple):
    """A 3D point defined by X, Y, and Z coordinates."""

    x: float
    y: float
    z: float


class Size3d(NamedTuple):
    """A 3D size defined by dx, dy, and dz dimensions."""

    dx: float
    dy: float
    dz: float


class Size3i(NamedTuple):
    """A 3D size defined by nx, ny, and nz integer dimensions."""

    nx: int
    ny: int
    nz: int

    @property
    def total_size(self) -> int:
        """The total size (number of elements) represented by this Size3i."""
        return self.nx * self.ny * self.nz


def _three_values(value: object, name: str) -> tuple:
    # Accept NumPy-style arrays without requiring NumPy in evo-sdk-common.
    if hasattr(value, "ndim") and hasattr(value, "tolist"):
        if value.ndim != 1:
            raise ValueError(f"{name} must be a one-dimensional array of exactly three values")
        value = value.tolist()
    if not isinstance(value, (tuple, list)):
        raise TypeError(f"{name} must be a three-value list, tuple, or one-dimensional array")
    if len(value) != 3:
        raise ValueError(f"{name} must have exactly three values")
    return tuple(value)


def as_point3(value: Point3 | tuple[float, float, float] | list[float]) -> Point3:
    """Convert a list, tuple, or 1D array of finite coordinates; preserve a Point3."""
    if isinstance(value, Point3):
        return value
    coordinates = _three_values(value, "origin")
    if not all(isinstance(v, Real) and not isinstance(v, bool) and isfinite(v) for v in coordinates):
        raise ValueError("origin must contain three finite real numbers")
    return Point3(*coordinates)


def as_size3i(value: Size3i | tuple[int, int, int] | list[int], *, name: str = "n_blocks") -> Size3i:
    """Convert a list, tuple, or 1D array of positive counts; preserve a Size3i."""
    if isinstance(value, Size3i):
        return value
    counts = _three_values(value, name)
    if not all(isinstance(v, Integral) and not isinstance(v, bool) and v > 0 for v in counts):
        raise ValueError(f"{name} must contain three positive integers")
    return Size3i(*counts)


def as_size3d(value: Size3d | tuple[float, float, float] | list[float], *, name: str = "block_size") -> Size3d:
    """Convert a list, tuple, or 1D array of positive finite sizes; preserve a Size3d."""
    if isinstance(value, Size3d):
        return value
    sizes = _three_values(value, name)
    if not all(isinstance(v, Real) and not isinstance(v, bool) and isfinite(v) and v > 0 for v in sizes):
        raise ValueError(f"{name} must contain three positive finite real numbers")
    return Size3d(*sizes)


class BoundingBox(NamedTuple):
    """An axis-aligned bounding box defined by minimum and maximum coordinates."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float

    @classmethod
    def from_origin_and_size(cls, origin: Point3, size: Size3i, cell_size: Size3d) -> BoundingBox:
        """Create a bounding box from an origin point and grid dimensions.

        :param origin: The origin point of the grid.
        :param size: The number of cells in each dimension.
        :param cell_size: The size of each cell in each dimension.
        :return: A BoundingBox enclosing the grid.
        """
        return cls(
            x_min=origin.x,
            x_max=origin.x + size.nx * cell_size.dx,
            y_min=origin.y,
            y_max=origin.y + size.ny * cell_size.dy,
            z_min=origin.z,
            z_max=origin.z + size.nz * cell_size.dz,
        )
