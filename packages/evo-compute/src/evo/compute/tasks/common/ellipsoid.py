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
from typing import Any

__all__ = [
    "Ellipsoid",
    "EllipsoidRanges",
    "Rotation",
]


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

