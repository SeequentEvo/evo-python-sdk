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
from typing import NamedTuple, Protocol, TypeAlias, TypeVar, runtime_checkable

_T_co = TypeVar("_T_co", covariant=True)


@runtime_checkable
class _ArrayLike1D(Protocol[_T_co]):
    """Array-like input convertible to a Python list without importing NumPy."""

    @property
    def ndim(self) -> int: ...

    def __len__(self) -> int: ...

    def tolist(self) -> list[_T_co]: ...


FloatArrayLike3: TypeAlias = tuple[float, float, float] | list[float] | _ArrayLike1D[float]
IntArrayLike3: TypeAlias = tuple[int, int, int] | list[int] | _ArrayLike1D[int]

__all__ = [
    "BoundingBox",
    "FloatArrayLike3",
    "IntArrayLike3",
    "Point3",
    "Size3d",
    "Size3i",
]


class Point3(NamedTuple):
    """A 3D point defined by X, Y, and Z coordinates."""

    x: float
    y: float
    z: float

    @classmethod
    def from_array_like(cls, value: Point3 | FloatArrayLike3) -> Point3:
        """Validate three finite coordinates, preserving existing points."""
        if isinstance(value, cls):
            return value
        return cls(*_validate_array_like(value, kind=Real, positive=False))


class Size3d(NamedTuple):
    """A 3D size defined by dx, dy, and dz dimensions."""

    dx: float
    dy: float
    dz: float

    @classmethod
    def from_array_like(cls, value: Size3d | FloatArrayLike3) -> Size3d:
        """Validate three positive finite dimensions, preserving existing sizes."""
        if isinstance(value, cls):
            return value
        return cls(*_validate_array_like(value, kind=Real, positive=True))


class Size3i(NamedTuple):
    """A 3D size defined by nx, ny, and nz integer dimensions."""

    nx: int
    ny: int
    nz: int

    @classmethod
    def from_array_like(cls, value: Size3i | IntArrayLike3) -> Size3i:
        """Validate three positive integer counts, preserving existing sizes."""
        if isinstance(value, cls):
            return value
        return cls(*_validate_array_like(value, kind=Integral, positive=True))

    @property
    def total_size(self) -> int:
        """The total size (number of elements) represented by this Size3i."""
        return self.nx * self.ny * self.nz


def _is_valid_number(value: object, kind: type[Integral] | type[Real], *, positive: bool) -> bool:
    if not isinstance(value, kind) or isinstance(value, bool):
        return False
    try:
        if not isfinite(value):
            return False
        if positive and not float(value) > 0:
            return False
    except TypeError:
        return False
    return True


def _validate_array_like(value: object, *, kind: type[Integral] | type[Real], positive: bool) -> tuple:
    # Accept NumPy-style arrays without requiring NumPy in evo-sdk-common.
    if isinstance(value, _ArrayLike1D):
        if value.ndim != 1:
            raise ValueError("value must be a one-dimensional array of exactly three values")
        if len(value) != 3:
            raise ValueError("value must have exactly three values")
        value = value.tolist()
    if not isinstance(value, (tuple, list)):
        raise TypeError("value must be a three-value list, tuple, or one-dimensional array")
    if len(value) != 3:
        raise ValueError("value must have exactly three values")
    cast_to = int if kind is Integral else float
    result = []
    for item in value:
        if not _is_valid_number(item, kind, positive=positive):
            description = "integers" if kind is Integral else "finite real numbers"
            raise ValueError(f"value must contain three {'positive ' if positive else ''}{description}")
        result.append(cast_to(item))
    return tuple(result)


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
