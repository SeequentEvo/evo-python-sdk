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

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any, NamedTuple, overload

import numpy as np
import numpy.typing as npt
import pydantic

__all__ = [
    "BoundingBox",
    "CoordinateReferenceSystem",
    "EpsgCode",
    "Point3",
    "Rotation",
    "Size3d",
    "Size3i",
]


class EpsgCode(int):
    """An integer representing an EPSG code."""

    def __new__(cls, value: int | str) -> EpsgCode:
        if isinstance(value, str):
            try:
                value = int(value)
            except ValueError as ve:
                raise ValueError(f"Cannot convert '{value}' to an integer EPSG code") from ve

        if not (1024 <= value <= 32767):
            raise ValueError(f"EPSG code must be between 1024 and 32767, got {value}")

        return int.__new__(cls, value)

    def __repr__(self) -> str:
        return f"EpsgCode({int(self)})"

    def __str__(self):
        return f"EPSG:{int(self)}"


def _dump_crs(value: EpsgCode | str | None) -> Any:
    if value is None:
        return "unspecified"
    elif isinstance(value, EpsgCode):
        return {"epsg_code": value}
    elif isinstance(value, str):
        return {"ogc_wkt": value}
    else:
        raise ValueError("coordinate_reference_system must be an EpsgCode, str, or None")


def _load_crs(value: Any) -> EpsgCode | str | None:
    if value == "unspecified":
        return None
    elif isinstance(value, dict):
        if "epsg_code" in value:
            return EpsgCode(value["epsg_code"])
        elif "ogc_wkt" in value:
            return value["ogc_wkt"]
        else:
            raise ValueError("Invalid CRS dictionary format")
    else:
        raise ValueError("Invalid CRS format")


CoordinateReferenceSystem = Annotated[
    EpsgCode | str | None, pydantic.PlainValidator(_load_crs), pydantic.PlainSerializer(_dump_crs)
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


@dataclass(frozen=True)
class BoundingBox:
    """A bounding box defined by minimum and maximum coordinates."""

    min_x: float
    min_y: float
    max_x: float
    max_y: float
    min_z: float
    max_z: float

    @property
    def min(self) -> Point3:
        """The minimum point of the bounding box."""
        return Point3(self.min_x, self.min_y, self.min_z)

    @property
    def max(self) -> Point3:
        """The maximum point of the bounding box."""
        return Point3(self.max_x, self.max_y, self.max_z)

    @overload
    @classmethod
    def from_points(cls, x: npt.ArrayLike, y: npt.ArrayLike, z: npt.ArrayLike, /) -> BoundingBox:
        """Create a BoundingBox that encompasses the given points."""

    @overload
    @classmethod
    def from_points(cls, points: npt.ArrayLike, /) -> BoundingBox:
        """Create a BoundingBox that encompasses the given points."""

    @classmethod
    def from_points(cls, *args) -> BoundingBox:
        if len(args) == 1:
            points = np.array(args[0])
            if points.ndim != 2 or points.shape[1] != 3:
                raise ValueError("Points array must be of shape (N, 3)")

            x = points[:, 0]
            y = points[:, 1]
            z = points[:, 2]
        elif len(args) == 3:
            x, y, z = args
            x = np.array(x)
            y = np.array(y)
            z = np.array(z)
            if x.ndim != 1 or y.ndim != 1 or z.ndim != 1:
                raise ValueError("x, y, and z must be 1-dimensional arrays")
            if x.shape != y.shape or x.shape != z.shape:
                raise ValueError("x, y, and z arrays must have the same shape")
        else:
            raise ValueError("from_points() accepts either a single (N, 3) array or three 1D arrays for x, y, and z")

        return cls(
            min_x=x.min(),
            min_y=y.min(),
            min_z=z.min(),
            max_x=x.max(),
            max_y=y.max(),
            max_z=z.max(),
        )


@dataclass(frozen=True)
class Rotation:
    """A rotation defined by dip azimuth, dip, and pitch angles."""

    dip_azimuth: float
    dip: float
    pitch: float

    def as_rotation_matrix(self) -> np.ndarray:
        """Convert the rotation to a rotation matrix.

        The rotation matrix is n the pre-multiplication convention, meaning that
        to rotate a vector `v`, you compute `v_rotated = R @ v`.
        """

        # Clockwise rotation around Z (dip azimuth)
        dip_azimuth_radians = np.radians(self.dip_azimuth)
        dip_azimuth_rotation_matrix = np.array(
            [
                [np.cos(dip_azimuth_radians), np.sin(dip_azimuth_radians), 0],
                [-np.sin(dip_azimuth_radians), np.cos(dip_azimuth_radians), 0],
                [0, 0, 1],
            ]
        )

        # Clockwise rotation around X (dip)
        dip_radians = np.radians(self.dip)
        dip_rotation_matrix = np.array(
            [[1, 0, 0], [0, np.cos(dip_radians), np.sin(dip_radians)], [0, -np.sin(dip_radians), np.cos(dip_radians)]]
        )

        # Clockwise rotation around Z (pitch)
        pitch_radians = np.radians(self.pitch)
        pitch_rotation_matrix = np.array(
            [
                [np.cos(pitch_radians), np.sin(pitch_radians), 0],
                [-np.sin(pitch_radians), np.cos(pitch_radians), 0],
                [0, 0, 1],
            ]
        )

        # Combined intrinsic rotations: dip_azimuth -> dip -> pitch
        return dip_azimuth_rotation_matrix @ dip_rotation_matrix @ pitch_rotation_matrix
