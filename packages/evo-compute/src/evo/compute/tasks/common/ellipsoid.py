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

"""Ellipsoid and rotation primitives for spatial search operations."""

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
    """Create a 3D rotation matrix from Leapfrog convention angles."""
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
    """The ranges (semi-axes lengths) of an ellipsoid.

    Used to define the spatial extent of search neighborhoods in geostatistical
    operations like kriging, simulation, and other estimation techniques.

    Example:
        >>> ranges = EllipsoidRanges(major=200.0, semi_major=150.0, minor=100.0)
    """

    major: float
    """The major axis length of the ellipsoid (largest extent)."""

    semi_major: float
    """The semi-major axis length of the ellipsoid (intermediate extent)."""

    minor: float
    """The minor axis length of the ellipsoid (smallest extent)."""

    def __init__(self, major: float, semi_major: float, minor: float):
        self.major = major
        self.semi_major = semi_major
        self.minor = minor

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "major": self.major,
            "semi_major": self.semi_major,
            "minor": self.minor,
        }

    def scaled(self, factor: float) -> "EllipsoidRanges":
        """Return a new EllipsoidRanges scaled by the given factor.

        Example:
            >>> ranges = EllipsoidRanges(major=100, semi_major=50, minor=25)
            >>> scaled = ranges.scaled(2.0)  # Double the size
            >>> scaled.major
            200.0
        """
        return EllipsoidRanges(
            major=self.major * factor,
            semi_major=self.semi_major * factor,
            minor=self.minor * factor,
        )


@dataclass
class Rotation:
    """The rotation of an ellipsoid using Leapfrog convention.

    Defines the orientation of an ellipsoid in 3D space using three sequential
    rotations: dip azimuth (about Z), dip (about X'), and pitch (about Z'').

    Example:
        >>> rotation = Rotation(dip_azimuth=45.0, dip=30.0, pitch=0.0)
        >>> # Or use defaults (no rotation):
        >>> rotation = Rotation()
    """

    dip_azimuth: float = 0.0
    """First rotation, about the z-axis, in degrees (0-360)."""

    dip: float = 0.0
    """Second rotation, about the x-axis, in degrees (0-90)."""

    pitch: float = 0.0
    """Third rotation, about the z-axis, in degrees."""

    def __init__(self, dip_azimuth: float = 0.0, dip: float = 0.0, pitch: float = 0.0):
        self.dip_azimuth = dip_azimuth
        self.dip = dip
        self.pitch = pitch

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "dip_azimuth": self.dip_azimuth,
            "dip": self.dip,
            "pitch": self.pitch,
        }


@dataclass
class Ellipsoid:
    """An ellipsoid defining a spatial search region.

    Combines ranges (semi-axes lengths) with rotation to define an oriented
    ellipsoid in 3D space. Used for neighborhood searches in geostatistical
    operations.

    Example:
        >>> ellipsoid = Ellipsoid(
        ...     ranges=EllipsoidRanges(major=200.0, semi_major=150.0, minor=100.0),
        ...     rotation=Rotation(dip_azimuth=45.0, dip=30.0, pitch=0.0),
        ... )
        >>>
        >>> # Generate mesh for 3D visualization
        >>> x, y, z = ellipsoid.surface_points(center=(100, 200, 50))
    """

    ranges: EllipsoidRanges
    """The ranges (semi-axes lengths) of the ellipsoid."""

    rotation: Rotation | None = None
    """The rotation of the ellipsoid. Defaults to no rotation if not specified."""

    def __init__(self, ranges: EllipsoidRanges, rotation: Rotation | None = None):
        self.ranges = ranges
        self.rotation = rotation or Rotation()

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "ellipsoid_ranges": self.ranges.to_dict(),
            "rotation": self.rotation.to_dict() if self.rotation else Rotation().to_dict(),
        }

    def scaled(self, factor: float) -> "Ellipsoid":
        """Return a new Ellipsoid scaled by the given factor.

        The rotation is preserved, only the ranges are scaled.

        Example:
            >>> ell = Ellipsoid(EllipsoidRanges(100, 50, 25), Rotation(45, 30, 0))
            >>> search_ell = ell.scaled(2.0)  # Create search ellipsoid 2x the variogram range
            >>> search_ell.ranges.major
            200.0
        """
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
        """Generate surface mesh points for 3D visualization.

        Returns flattened x, y, z arrays suitable for Plotly Mesh3d or similar.

        Args:
            center: Center point (x, y, z) for the ellipsoid.
            n_points: Number of points in each parametric direction.

        Returns:
            Tuple of (x, y, z) as 1D numpy arrays.

        Example with Plotly:
            >>> import plotly.graph_objects as go
            >>> x, y, z = ellipsoid.surface_points(center=(100, 200, 50))
            >>> mesh = go.Mesh3d(x=x, y=y, z=z, alphahull=0, opacity=0.3, color='blue')
            >>> fig = go.Figure(data=[mesh])
            >>> fig.show()
        """
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

        x_out = rotated[0] + center[0]
        y_out = rotated[1] + center[1]
        z_out = rotated[2] + center[2]

        return x_out, y_out, z_out

    def wireframe_points(
        self,
        center: tuple[float, float, float] = (0, 0, 0),
        n_points: int = 30,
    ) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
        """Generate wireframe points for 3D visualization.

        Returns x, y, z arrays with NaN separators between line segments,
        suitable for Plotly Scatter3d with mode='lines'.

        Args:
            center: Center point (x, y, z) for the ellipsoid.
            n_points: Number of points per circle.

        Returns:
            Tuple of (x, y, z) as 1D numpy arrays with NaN separators.

        Example with Plotly:
            >>> import plotly.graph_objects as go
            >>> x, y, z = ellipsoid.wireframe_points(center=(100, 200, 50))
            >>> line = go.Scatter3d(x=x, y=y, z=z, mode='lines', line=dict(color='blue'))
            >>> fig = go.Figure(data=[line])
            >>> fig.show()
        """
        rot = self.rotation or Rotation()
        rot_matrix = _rotation_matrix(rot.dip_azimuth, rot.dip, rot.pitch)

        theta = np.linspace(0, 2 * np.pi, n_points)

        all_x: list[float] = []
        all_y: list[float] = []
        all_z: list[float] = []

        # XY plane (major-semi_major)
        x = self.ranges.major * np.cos(theta)
        y = self.ranges.semi_major * np.sin(theta)
        z = np.zeros_like(theta)
        points = np.array([x, y, z])
        rotated = rot_matrix @ points
        all_x.extend((rotated[0] + center[0]).tolist() + [np.nan])
        all_y.extend((rotated[1] + center[1]).tolist() + [np.nan])
        all_z.extend((rotated[2] + center[2]).tolist() + [np.nan])

        # XZ plane (major-minor)
        x = self.ranges.major * np.cos(theta)
        y = np.zeros_like(theta)
        z = self.ranges.minor * np.sin(theta)
        points = np.array([x, y, z])
        rotated = rot_matrix @ points
        all_x.extend((rotated[0] + center[0]).tolist() + [np.nan])
        all_y.extend((rotated[1] + center[1]).tolist() + [np.nan])
        all_z.extend((rotated[2] + center[2]).tolist() + [np.nan])

        # YZ plane (semi_major-minor)
        x = np.zeros_like(theta)
        y = self.ranges.semi_major * np.cos(theta)
        z = self.ranges.minor * np.sin(theta)
        points = np.array([x, y, z])
        rotated = rot_matrix @ points
        all_x.extend((rotated[0] + center[0]).tolist() + [np.nan])
        all_y.extend((rotated[1] + center[1]).tolist() + [np.nan])
        all_z.extend((rotated[2] + center[2]).tolist() + [np.nan])

        return np.array(all_x), np.array(all_y), np.array(all_z)
