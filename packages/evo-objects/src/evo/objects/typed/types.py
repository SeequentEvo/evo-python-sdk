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
from pydantic_core import core_schema

__all__ = [
    "BoundingBox",
    "CoordinateReferenceSystem",
    "Ellipsoid",
    "EllipsoidRanges",
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

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: Any) -> core_schema.CoreSchema:
        return core_schema.no_info_after_validator_function(
            cls,
            core_schema.int_schema(),
            serialization=core_schema.plain_serializer_function_ser_schema(int),
        )


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

    @property
    def total_size(self) -> int:
        """The total size (number of elements) represented by this Size3i."""
        return self.nx * self.ny * self.nz


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

    @classmethod
    def from_extent(
        cls,
        origin: Point3,
        extent: Size3d,
        rotation: Rotation | None = None,
    ) -> BoundingBox:
        """Create a BoundingBox from an origin point and extent.

        :param origin: The origin point of the box.
        :param extent: The extent (dx, dy, dz) of the box.
        :param rotation: Optional rotation to apply to the box.
        :return: The bounding box that encompasses the rotated box.
        """
        if rotation is not None:
            rotation_matrix = rotation.as_rotation_matrix()
        else:
            rotation_matrix = np.eye(3)

        corners = np.array(
            [
                [0, 0, 0],
                [extent.dx, 0, 0],
                [0, extent.dy, 0],
                [0, 0, extent.dz],
                [extent.dx, extent.dy, 0],
                [extent.dx, 0, extent.dz],
                [0, extent.dy, extent.dz],
                [extent.dx, extent.dy, extent.dz],
            ]
        )
        rotated_corners = rotation_matrix @ corners.T
        return cls.from_points(
            rotated_corners[0, :] + origin.x,
            rotated_corners[1, :] + origin.y,
            rotated_corners[2, :] + origin.z,
        )

    @classmethod
    def from_regular_grid(
        cls,
        origin: Point3,
        size: Size3i,
        cell_size: Size3d,
        rotation: Rotation | None = None,
    ) -> BoundingBox:
        """Create a BoundingBox for a regular 3D grid.

        :param origin: The origin point of the grid.
        :param size: The number of cells in each dimension.
        :param cell_size: The size of each cell in each dimension.
        :param rotation: Optional rotation to apply to the grid.
        :return: The bounding box that encompasses the grid.
        """
        extent = Size3d(
            dx=size.nx * cell_size.dx,
            dy=size.ny * cell_size.dy,
            dz=size.nz * cell_size.dz,
        )
        return cls.from_extent(origin, extent, rotation)


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


@dataclass
class EllipsoidRanges:
    """The ranges (semi-axes lengths) of an ellipsoid."""

    major: float
    semi_major: float
    minor: float

    def __init__(self, major: float, semi_major: float, minor: float):
        self.major = major
        self.semi_major = semi_major
        self.minor = minor

    def to_dict(self) -> dict[str, Any]:
        return {"major": self.major, "semi_major": self.semi_major, "minor": self.minor}

    def scaled(self, factor: float) -> "EllipsoidRanges":
        return EllipsoidRanges(
            major=self.major * factor,
            semi_major=self.semi_major * factor,
            minor=self.minor * factor,
        )


@dataclass
class Ellipsoid:
    """An ellipsoid defining a spatial region."""

    ranges: EllipsoidRanges
    rotation: Rotation | None = None

    def __init__(self, ranges: EllipsoidRanges, rotation: Rotation | None = None):
        self.ranges = ranges
        self.rotation = rotation or Rotation(0, 0, 0)

    def to_dict(self) -> dict[str, Any]:
        rot = self.rotation or Rotation(0, 0, 0)
        return {
            "ellipsoid_ranges": self.ranges.to_dict(),
            "rotation": {"dip_azimuth": rot.dip_azimuth, "dip": rot.dip, "pitch": rot.pitch},
        }

    def scaled(self, factor: float) -> "Ellipsoid":
        rot = self.rotation or Rotation(0, 0, 0)
        return Ellipsoid(
            ranges=self.ranges.scaled(factor),
            rotation=Rotation(
                dip_azimuth=rot.dip_azimuth,
                dip=rot.dip,
                pitch=rot.pitch,
            ),
        )

    def surface_points(
        self,
        center: tuple[float, float, float] = (0, 0, 0),
        n_points: int = 20,
    ) -> tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.floating[Any]], npt.NDArray[np.floating[Any]]]:
        """Generate surface mesh points for 3D visualization."""
        rot = self.rotation or Rotation(0, 0, 0)
        rot_matrix = rot.as_rotation_matrix()

        u = np.linspace(0, 2 * np.pi, n_points)
        v = np.linspace(0, np.pi, n_points)
        u, v = np.meshgrid(u, v)

        # Leapfrog convention: major=X, semi_major=Y, minor=Z
        x = self.ranges.major * np.cos(u) * np.sin(v)       # major along X
        y = self.ranges.semi_major * np.sin(u) * np.sin(v)  # semi_major along Y
        z = self.ranges.minor * np.cos(v)                    # minor along Z

        points = np.array([x.flatten(), y.flatten(), z.flatten()])
        rotated = rot_matrix @ points

        return rotated[0] + center[0], rotated[1] + center[1], rotated[2] + center[2]

    def wireframe_points(
        self,
        center: tuple[float, float, float] = (0, 0, 0),
        n_points: int = 30,
    ) -> tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.floating[Any]], npt.NDArray[np.floating[Any]]]:
        """Generate wireframe points for 3D visualization.

        - Major axis along X
        - Semi-major axis along Y
        - Minor axis along Z (up)
        """
        rot = self.rotation or Rotation(0, 0, 0)
        rot_matrix = rot.as_rotation_matrix()
        theta = np.linspace(0, 2 * np.pi, n_points)

        # Pre-allocate arrays for 3 planes, each with n_points + 1 (for NaN separator)
        total_points = 3 * (n_points + 1)
        all_x = np.empty(total_points)
        all_y = np.empty(total_points)
        all_z = np.empty(total_points)

        # XY plane (major-semi_major): horizontal slice
        x = self.ranges.major * np.cos(theta)       # major along X
        y = self.ranges.semi_major * np.sin(theta)  # semi_major along Y
        z = np.zeros_like(theta)
        rotated = rot_matrix @ np.array([x, y, z])
        all_x[:n_points] = rotated[0] + center[0]
        all_y[:n_points] = rotated[1] + center[1]
        all_z[:n_points] = rotated[2] + center[2]
        all_x[n_points] = np.nan
        all_y[n_points] = np.nan
        all_z[n_points] = np.nan

        # XZ plane (major-minor): vertical slice along major axis
        x = self.ranges.major * np.cos(theta)       # major along X
        y = np.zeros_like(theta)
        z = self.ranges.minor * np.sin(theta)       # minor along Z
        rotated = rot_matrix @ np.array([x, y, z])
        offset = n_points + 1
        all_x[offset:offset + n_points] = rotated[0] + center[0]
        all_y[offset:offset + n_points] = rotated[1] + center[1]
        all_z[offset:offset + n_points] = rotated[2] + center[2]
        all_x[offset + n_points] = np.nan
        all_y[offset + n_points] = np.nan
        all_z[offset + n_points] = np.nan

        # YZ plane (semi_major-minor): vertical slice along semi_major axis
        x = np.zeros_like(theta)
        y = self.ranges.semi_major * np.cos(theta)  # semi_major along Y
        z = self.ranges.minor * np.sin(theta)       # minor along Z
        rotated = rot_matrix @ np.array([x, y, z])
        offset = 2 * (n_points + 1)
        all_x[offset:offset + n_points] = rotated[0] + center[0]
        all_y[offset:offset + n_points] = rotated[1] + center[1]
        all_z[offset:offset + n_points] = rotated[2] + center[2]
        all_x[offset + n_points] = np.nan
        all_y[offset + n_points] = np.nan
        all_z[offset + n_points] = np.nan

        return all_x, all_y, all_z

