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

"""Type definitions for typed block model access."""

from __future__ import annotations

from typing import NamedTuple

__all__ = [
    "BoundingBox",
    "Point3",
    "Size3d",
    "Size3i",
]


class Point3(NamedTuple):
    """A 3D point defined by X, Y, and Z coordinates."""

    x: float
    y: float
    z: float


class Size3d(NamedTuple):
    """A 3D size defined by dx, dy, and dz dimensions (floats)."""

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
        """Return the total number of cells in the grid."""
        return self.nx * self.ny * self.nz


class BoundingBox(NamedTuple):
    """An axis-aligned bounding box defined by minimum and maximum coordinates."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float

    @classmethod
    def from_origin_and_size(
        cls, origin: Point3, size: Size3i, cell_size: Size3d
    ) -> BoundingBox:
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

