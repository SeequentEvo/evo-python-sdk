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

"""Ellipsoid and rotation primitives for spatial operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = [
    "Ellipsoid",
    "EllipsoidRanges",
    "Rotation",
]


def _rotation_matrix(dip_azimuth: float, dip: float, pitch: float) -> NDArray[np.floating[Any]]:
    """Create a 3D rotation matrix from Geoscience object convention angles."""
    az = np.radians(dip_azimuth)
    d = np.radians(dip)
    p = np.radians(pitch)

    rz1 = np.array([
        [np.cos(az), np.sin(az), 0],
        [-np.sin(az), np.cos(az), 0],
        [0, 0, 1],
    ])

    rx = np.array([
        [1, 0, 0],
        [0, np.cos(d), -np.sin(d)],
        [0, np.sin(d), np.cos(d)],
    ])

    rz2 = np.array([
        [np.cos(p), np.sin(p), 0],
        [-np.sin(p), np.cos(p), 0],
        [0, 0, 1],
    ])

    return rz2 @ rx @ rz1


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
class Rotation:
    """The rotation of an ellipsoid using Leapfrog/Geoscience Object convention."""

    dip_azimuth: float = 0.0
    dip: float = 0.0
    pitch: float = 0.0

    def __init__(self, dip_azimuth: float = 0.0, dip: float = 0.0, pitch: float = 0.0):
        self.dip_azimuth = dip_azimuth
        self.dip = dip
        self.pitch = pitch

    def to_dict(self) -> dict[str, Any]:
        return {"dip_azimuth": self.dip_azimuth, "dip": self.dip, "pitch": self.pitch}


@dataclass
class Ellipsoid:
    """An ellipsoid defining a spatial region."""

    ranges: EllipsoidRanges
    rotation: Rotation | None = None

    def __init__(self, ranges: EllipsoidRanges, rotation: Rotation | None = None):
        self.ranges = ranges
        self.rotation = rotation or Rotation()

    def to_dict(self) -> dict[str, Any]:
        return {
            "ellipsoid_ranges": self.ranges.to_dict(),
            "rotation": self.rotation.to_dict() if self.rotation else Rotation().to_dict(),
        }

    def scaled(self, factor: float) -> "Ellipsoid":
        return Ellipsoid(
            ranges=self.ranges.scaled(factor),
            rotation=Rotation(
                dip_azimuth=self.rotation.dip_azimuth if self.rotation else 0.0,
                dip=self.rotation.dip if self.rotation else 0.0,
                pitch=self.rotation.pitch if self.rotation else 0.0,
            ),
        )

    def surface_points(
        self,
        center: tuple[float, float, float] = (0, 0, 0),
        n_points: int = 20,
    ) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
        """Generate surface mesh points for 3D visualization."""
        rot = self.rotation or Rotation()
        rot_matrix = _rotation_matrix(rot.dip_azimuth, rot.dip, rot.pitch)

        u = np.linspace(0, 2 * np.pi, n_points)
        v = np.linspace(0, np.pi, n_points)
        u, v = np.meshgrid(u, v)

        x = self.ranges.major * np.cos(u) * np.sin(v)
        y = self.ranges.semi_major * np.sin(u) * np.sin(v)
        z = self.ranges.minor * np.cos(v)

        points = np.array([x.flatten(), y.flatten(), z.flatten()])
        rotated = rot_matrix @ points

        return rotated[0] + center[0], rotated[1] + center[1], rotated[2] + center[2]

    def wireframe_points(
        self,
        center: tuple[float, float, float] = (0, 0, 0),
        n_points: int = 30,
    ) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
        """Generate wireframe points for 3D visualization."""
        rot = self.rotation or Rotation()
        rot_matrix = _rotation_matrix(rot.dip_azimuth, rot.dip, rot.pitch)
        theta = np.linspace(0, 2 * np.pi, n_points)

        all_x: list[float] = []
        all_y: list[float] = []
        all_z: list[float] = []

        # XY plane
        x = self.ranges.major * np.cos(theta)
        y = self.ranges.semi_major * np.sin(theta)
        z = np.zeros_like(theta)
        points = np.array([x, y, z])
        rotated = rot_matrix @ points
        all_x.extend((rotated[0] + center[0]).tolist() + [np.nan])
        all_y.extend((rotated[1] + center[1]).tolist() + [np.nan])
        all_z.extend((rotated[2] + center[2]).tolist() + [np.nan])

        # XZ plane
        x = self.ranges.major * np.cos(theta)
        y = np.zeros_like(theta)
        z = self.ranges.minor * np.sin(theta)
        points = np.array([x, y, z])
        rotated = rot_matrix @ points
        all_x.extend((rotated[0] + center[0]).tolist() + [np.nan])
        all_y.extend((rotated[1] + center[1]).tolist() + [np.nan])
        all_z.extend((rotated[2] + center[2]).tolist() + [np.nan])

        # YZ plane
        x = np.zeros_like(theta)
        y = self.ranges.semi_major * np.cos(theta)
        z = self.ranges.minor * np.sin(theta)
        points = np.array([x, y, z])
        rotated = rot_matrix @ points
        all_x.extend((rotated[0] + center[0]).tolist() + [np.nan])
        all_y.extend((rotated[1] + center[1]).tolist() + [np.nan])
        all_z.extend((rotated[2] + center[2]).tolist() + [np.nan])

        return np.array(all_x), np.array(all_y), np.array(all_z)

