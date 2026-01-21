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

"""Ellipsoid and variogram data generation for visualization.

This module provides classes and functions to generate mesh data for 3D ellipsoids
and curve data for 2D variogram plots. The output is numpy arrays that can be
used with any plotting library (Plotly, K3D, matplotlib, etc.).

Example with Plotly - Variogram Ellipsoid with PointSet
-------------------------------------------------------
>>> from evo.objects.notebooks import Ellipsoid
>>> import plotly.graph_objects as go
>>>
>>> # Get ellipsoid from variogram structure
>>> ell = Ellipsoid.from_variogram(variogram)
>>> pts = await source_pointset.to_dataframe()
>>> center = (pts['x'].mean(), pts['y'].mean(), pts['z'].mean())
>>> x, y, z = ell.surface_points(center=center)
>>>
>>> # Create transparent mesh
>>> mesh = go.Mesh3d(x=x, y=y, z=z, alphahull=0, opacity=0.3, color='blue')
>>> scatter = go.Scatter3d(x=pts['x'], y=pts['y'], z=pts['z'], mode='markers',
...                        marker=dict(size=2, color=pts['grade']))
>>> fig = go.Figure(data=[mesh, scatter])
>>> fig.show()

Example with Search Ellipsoid
-----------------------------
>>> from evo.objects.notebooks import Ellipsoid
>>>
>>> # From kriging search ellipsoid parameters
>>> search_ell = Ellipsoid.from_search_ellipsoid(search_ellipsoid)
>>> x, y, z = search_ell.surface_points(center=center)
>>> search_mesh = go.Mesh3d(x=x, y=y, z=z, alphahull=0, opacity=0.2, color='gold')
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = [
    "Ellipsoid",
    "EllipsoidData",
    "VariogramCurveData",
    "generate_ellipsoid_mesh",
    "generate_ellipsoid_wireframe",
    "generate_variogram_curves",
]


@dataclass
class EllipsoidData:
    """Data for rendering a 3D ellipsoid.

    This dataclass contains numpy arrays that can be used with any 3D plotting
    library (Plotly, K3D, matplotlib, etc.).

    Attributes:
        x: X coordinates. For wireframe: 1D array with NaN separators between segments.
           For mesh: 2D array for surface grid.
        y: Y coordinates (same shape as x).
        z: Z coordinates (same shape as x).
        ranges: The (major, semi_major, minor) axis lengths used to generate this ellipsoid.
        label: Optional label for the ellipsoid (e.g., "Variogram Structure 1").

    Example with Plotly:
        >>> data = generate_ellipsoid_wireframe(ranges=(200, 150, 100))
        >>> import plotly.graph_objects as go
        >>> fig = go.Figure(data=[go.Scatter3d(x=data.x, y=data.y, z=data.z, mode='lines')])
        >>> fig.show()

    Example with K3D:
        >>> data = generate_ellipsoid_wireframe(ranges=(200, 150, 100))
        >>> import k3d
        >>> import numpy as np
        >>> mask = ~np.isnan(data.x)
        >>> vertices = np.column_stack([data.x[mask], data.y[mask], data.z[mask]]).astype(np.float32)
        >>> plot = k3d.plot()
        >>> plot += k3d.line(vertices, color=0x0000ff)
        >>> plot.display()
    """

    x: NDArray[np.floating[Any]]
    y: NDArray[np.floating[Any]]
    z: NDArray[np.floating[Any]]
    ranges: tuple[float, float, float]
    label: str | None = None


@dataclass
class VariogramCurveData:
    """Data for rendering a 2D variogram curve.

    This dataclass contains numpy arrays for plotting a variogram model curve
    in one of the principal directions.

    Attributes:
        distance: Lag distances (x-axis values).
        semivariance: Semivariance γ(h) values (y-axis values).
        direction: Direction label ("major", "semi_major", or "minor").
        range_value: The effective range in this direction.
        sill: The variogram sill value.

    Example with Plotly:
        >>> major, semi_maj, minor = generate_variogram_curves(variogram)
        >>> import plotly.graph_objects as go
        >>> fig = go.Figure()
        >>> fig.add_trace(go.Scatter(x=major.distance, y=major.semivariance, name='Major'))
        >>> fig.show()

    Example with matplotlib:
        >>> major, semi_maj, minor = generate_variogram_curves(variogram)
        >>> import matplotlib.pyplot as plt
        >>> plt.plot(major.distance, major.semivariance, label='Major')
        >>> plt.xlabel('Distance')
        >>> plt.ylabel('Semivariance')
        >>> plt.legend()
        >>> plt.show()
    """

    distance: NDArray[np.floating[Any]]
    semivariance: NDArray[np.floating[Any]]
    direction: str
    range_value: float
    sill: float


@runtime_checkable
class VariogramLike(Protocol):
    """Protocol for variogram-like objects."""

    @property
    def sill(self) -> float: ...

    @property
    def nugget(self) -> float: ...

    @property
    def structures(self) -> list[dict[str, Any]]: ...


class Ellipsoid:
    """A 3D ellipsoid for visualization.

    This class provides methods to generate mesh surface points for 3D visualization.
    It can be created from variogram structures or kriging search ellipsoids.

    Attributes:
        ranges: Tuple of (major, semi_major, minor) axis lengths.
        rotation: Tuple of (dip_azimuth, dip, pitch) angles in degrees.
        label: Optional label for the ellipsoid.

    Example:
        >>> from evo.objects.notebooks import Ellipsoid
        >>> import plotly.graph_objects as go
        >>>
        >>> # Create from variogram
        >>> ell = Ellipsoid.from_variogram(variogram)
        >>> x, y, z = ell.surface_points(center=(100, 200, 50))
        >>> mesh = go.Mesh3d(x=x, y=y, z=z, alphahull=0, opacity=0.3)
        >>> fig = go.Figure(data=[mesh])
        >>> fig.show()
    """

    def __init__(
        self,
        ranges: tuple[float, float, float],
        rotation: tuple[float, float, float] = (0, 0, 0),
        label: str | None = None,
    ):
        """Create an ellipsoid.

        Args:
            ranges: Tuple of (major, semi_major, minor) axis lengths.
            rotation: Tuple of (dip_azimuth, dip, pitch) angles in degrees.
            label: Optional label for the ellipsoid.
        """
        self.ranges = ranges
        self.rotation = rotation
        self.label = label

    @classmethod
    def from_variogram(
        cls,
        variogram: VariogramLike | dict[str, Any],
        structure_index: int = 0,
    ) -> "Ellipsoid":
        """Create an ellipsoid from a variogram structure.

        Args:
            variogram: A Variogram object or dict with 'structures'.
            structure_index: Which structure to use (default: 0, the first/largest).

        Returns:
            Ellipsoid configured with the variogram's anisotropy ranges and rotation.

        Example:
            >>> ell = Ellipsoid.from_variogram(variogram)
            >>> x, y, z = ell.surface_points(center=(0, 0, 0))
        """
        if isinstance(variogram, dict):
            structures = variogram.get("structures", [])
        else:
            structures = variogram.structures

        if not structures:
            raise ValueError("Variogram has no structures")

        if structure_index >= len(structures):
            raise ValueError(f"structure_index {structure_index} out of range (max {len(structures) - 1})")

        struct = structures[structure_index]
        anisotropy = struct.get("anisotropy", {})
        ranges_dict = anisotropy.get("ellipsoid_ranges", {})
        rotation_dict = anisotropy.get("rotation", {})

        ranges = (
            ranges_dict.get("major", 1.0),
            ranges_dict.get("semi_major", 1.0),
            ranges_dict.get("minor", 1.0),
        )
        rotation = (
            rotation_dict.get("dip_azimuth", 0.0),
            rotation_dict.get("dip", 0.0),
            rotation_dict.get("pitch", 0.0),
        )

        vtype = struct.get("variogram_type", "unknown")
        contribution = struct.get("contribution", 0)
        label = f"{vtype} (C={contribution:.2g})"

        return cls(ranges=ranges, rotation=rotation, label=label)

    @classmethod
    def from_search_ellipsoid(cls, search_ellipsoid: Any) -> "Ellipsoid":
        """Create an ellipsoid from a kriging search ellipsoid.

        Args:
            search_ellipsoid: An Ellipsoid object from evo.compute.tasks or a dict
                with 'ranges' and 'rotation' keys.

        Returns:
            Ellipsoid configured for visualization.

        Example:
            >>> from evo.compute.tasks import Ellipsoid as SearchEllipsoid, EllipsoidRanges, Rotation
            >>> search = SearchEllipsoid(
            ...     ranges=EllipsoidRanges(major=200, semi_major=150, minor=100),
            ...     rotation=Rotation(dip_azimuth=0, dip=0, pitch=0),
            ... )
            >>> ell = Ellipsoid.from_search_ellipsoid(search)
            >>> x, y, z = ell.surface_points(center=(0, 0, 0))
        """
        if hasattr(search_ellipsoid, "ranges"):
            # It's an Ellipsoid object from evo.compute.tasks
            ranges_obj = search_ellipsoid.ranges
            rotation_obj = search_ellipsoid.rotation
            ranges = (
                getattr(ranges_obj, "major", 1.0),
                getattr(ranges_obj, "semi_major", 1.0),
                getattr(ranges_obj, "minor", 1.0),
            )
            rotation = (
                getattr(rotation_obj, "dip_azimuth", 0.0),
                getattr(rotation_obj, "dip", 0.0),
                getattr(rotation_obj, "pitch", 0.0),
            )
        elif isinstance(search_ellipsoid, dict):
            ranges_dict = search_ellipsoid.get("ranges", {})
            rotation_dict = search_ellipsoid.get("rotation", {})
            ranges = (
                ranges_dict.get("major", 1.0),
                ranges_dict.get("semi_major", 1.0),
                ranges_dict.get("minor", 1.0),
            )
            rotation = (
                rotation_dict.get("dip_azimuth", 0.0),
                rotation_dict.get("dip", 0.0),
                rotation_dict.get("pitch", 0.0),
            )
        else:
            raise ValueError("search_ellipsoid must be an Ellipsoid object or dict")

        return cls(ranges=ranges, rotation=rotation, label="Search Ellipsoid")

    def surface_points(
        self,
        center: tuple[float, float, float] = (0, 0, 0),
        n_points: int = 20,
    ) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
        """Generate surface mesh points for the ellipsoid.

        Returns flattened x, y, z arrays suitable for Plotly Mesh3d.

        Args:
            center: Center point (x, y, z) for the ellipsoid.
            n_points: Number of points in each parametric direction.

        Returns:
            Tuple of (x, y, z) as 1D numpy arrays.

        Example with Plotly:
            >>> x, y, z = ell.surface_points(center=(100, 200, 50))
            >>> mesh = go.Mesh3d(x=x, y=y, z=z, alphahull=0, opacity=0.3, color='blue')
        """
        major, semi_major, minor = self.ranges
        rot_matrix = _rotation_matrix(*self.rotation)

        # Parametric ellipsoid surface
        u = np.linspace(0, 2 * np.pi, n_points)
        v = np.linspace(0, np.pi, n_points)
        u, v = np.meshgrid(u, v)

        # Ellipsoid coordinates (centered at origin)
        x = major * np.cos(u) * np.sin(v)
        y = semi_major * np.sin(u) * np.sin(v)
        z = minor * np.cos(v)

        # Apply rotation
        points = np.array([x.flatten(), y.flatten(), z.flatten()])
        rotated = rot_matrix @ points

        # Translate to center
        x_out = rotated[0] + center[0]
        y_out = rotated[1] + center[1]
        z_out = rotated[2] + center[2]

        return x_out, y_out, z_out

    def wireframe_points(
        self,
        center: tuple[float, float, float] = (0, 0, 0),
        n_points: int = 30,
    ) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
        """Generate wireframe points for the ellipsoid.

        Returns x, y, z arrays with NaN separators between line segments,
        suitable for Plotly Scatter3d with mode='lines'.

        Args:
            center: Center point (x, y, z) for the ellipsoid.
            n_points: Number of points per circle.

        Returns:
            Tuple of (x, y, z) as 1D numpy arrays with NaN separators.

        Example with Plotly:
            >>> x, y, z = ell.wireframe_points(center=(100, 200, 50))
            >>> line = go.Scatter3d(x=x, y=y, z=z, mode='lines', line=dict(color='blue'))
        """
        major, semi_major, minor = self.ranges
        rot_matrix = _rotation_matrix(*self.rotation)

        theta = np.linspace(0, 2 * np.pi, n_points)

        all_x: list[float] = []
        all_y: list[float] = []
        all_z: list[float] = []

        # XY plane (major-semi_major)
        x = major * np.cos(theta)
        y = semi_major * np.sin(theta)
        z = np.zeros_like(theta)
        points = np.array([x, y, z])
        rotated = rot_matrix @ points
        all_x.extend((rotated[0] + center[0]).tolist() + [np.nan])
        all_y.extend((rotated[1] + center[1]).tolist() + [np.nan])
        all_z.extend((rotated[2] + center[2]).tolist() + [np.nan])

        # XZ plane (major-minor)
        x = major * np.cos(theta)
        y = np.zeros_like(theta)
        z = minor * np.sin(theta)
        points = np.array([x, y, z])
        rotated = rot_matrix @ points
        all_x.extend((rotated[0] + center[0]).tolist() + [np.nan])
        all_y.extend((rotated[1] + center[1]).tolist() + [np.nan])
        all_z.extend((rotated[2] + center[2]).tolist() + [np.nan])

        # YZ plane (semi_major-minor)
        x = np.zeros_like(theta)
        y = semi_major * np.cos(theta)
        z = minor * np.sin(theta)
        points = np.array([x, y, z])
        rotated = rot_matrix @ points
        all_x.extend((rotated[0] + center[0]).tolist() + [np.nan])
        all_y.extend((rotated[1] + center[1]).tolist() + [np.nan])
        all_z.extend((rotated[2] + center[2]).tolist() + [np.nan])

        return np.array(all_x), np.array(all_y), np.array(all_z)


def _rotation_matrix(dip_azimuth: float, dip: float, pitch: float) -> NDArray[np.floating[Any]]:
    """Create a 3D rotation matrix from Leapfrog convention angles.

    The rotation follows the Leapfrog/Geoscience Object convention:
    1. Rotate about Z-axis by dip_azimuth (clockwise from North/+Y)
    2. Rotate about X-axis by dip
    3. Rotate about Z-axis by pitch

    Args:
        dip_azimuth: Azimuth of dip direction in degrees (0-360), measured clockwise from North.
        dip: Dip angle in degrees (0-180).
        pitch: Pitch/rake angle in degrees (0-360).

    Returns:
        3x3 rotation matrix as numpy array.
    """
    az = np.radians(dip_azimuth)
    d = np.radians(dip)
    p = np.radians(pitch)

    rz1 = np.array(
        [
            [np.cos(az), np.sin(az), 0],
            [-np.sin(az), np.cos(az), 0],
            [0, 0, 1],
        ]
    )

    rx = np.array(
        [
            [1, 0, 0],
            [0, np.cos(d), -np.sin(d)],
            [0, np.sin(d), np.cos(d)],
        ]
    )

    rz2 = np.array(
        [
            [np.cos(p), np.sin(p), 0],
            [-np.sin(p), np.cos(p), 0],
            [0, 0, 1],
        ]
    )

    return rz2 @ rx @ rz1


def _evaluate_structure(
    structure_type: str,
    h: NDArray[np.floating[Any]],
    contribution: float,
    range_val: float,
    alpha: int | None = None,
) -> NDArray[np.floating[Any]]:
    """Evaluate a variogram structure model."""
    h_norm = h / range_val if range_val > 0 else h

    if structure_type == "spherical":
        gamma = np.where(
            h_norm < 1,
            contribution * (1.5 * h_norm - 0.5 * h_norm**3),
            contribution,
        )
    elif structure_type == "exponential":
        gamma = contribution * (1 - np.exp(-3 * h_norm))
    elif structure_type == "gaussian":
        gamma = contribution * (1 - np.exp(-3 * h_norm**2))
    elif structure_type == "cubic":
        gamma = np.where(
            h_norm < 1,
            contribution * (7 * h_norm**2 - 8.75 * h_norm**3 + 3.5 * h_norm**5 - 0.75 * h_norm**7),
            contribution,
        )
    elif structure_type == "linear":
        gamma = contribution * h_norm
    elif structure_type == "spheroidal":
        if alpha is None:
            alpha = 3
        gamma = np.where(
            h_norm < 1,
            contribution * (1 - (1 - h_norm**2) ** (alpha / 2)),
            contribution,
        )
    elif structure_type == "generalisedcauchy":
        if alpha is None:
            alpha = 3
        gamma = contribution * (1 - (1 + h_norm**2) ** (-alpha / 2))
    else:
        gamma = np.full_like(h, contribution)

    return gamma


def generate_ellipsoid_wireframe(
    ranges: tuple[float, float, float],
    rotation: tuple[float, float, float] = (0, 0, 0),
    n_points: int = 30,
) -> EllipsoidData:
    """Generate wireframe data for a 3D ellipsoid.

    Creates coordinate arrays for rendering an ellipsoid as a wireframe.
    The wireframe consists of circles in the three principal planes.

    Args:
        ranges: Tuple of (major, semi_major, minor) axis lengths.
        rotation: Tuple of (dip_azimuth, dip, pitch) angles in degrees.
            Uses Leapfrog/Geoscience Object convention.
        n_points: Number of points per circle for smoothness.

    Returns:
        EllipsoidData with x, y, z coordinate arrays.
        NaN values separate different line segments.

    Example with Plotly:
        >>> data = generate_ellipsoid_wireframe(ranges=(200, 150, 100), rotation=(45, 30, 0))
        >>> import plotly.graph_objects as go
        >>> fig = go.Figure(data=[go.Scatter3d(x=data.x, y=data.y, z=data.z, mode='lines')])
        >>> fig.update_layout(scene=dict(aspectmode='cube'))
        >>> fig.show()

    Example with K3D:
        >>> data = generate_ellipsoid_wireframe(ranges=(200, 150, 100))
        >>> import k3d
        >>> import numpy as np
        >>> # K3D needs vertices without NaN, so filter them out
        >>> mask = ~np.isnan(data.x)
        >>> vertices = np.column_stack([data.x[mask], data.y[mask], data.z[mask]]).astype(np.float32)
        >>> plot = k3d.plot()
        >>> plot += k3d.line(vertices, color=0x0000ff, width=0.5)
        >>> plot.display()
    """
    major, semi_major, minor = ranges
    rot_matrix = _rotation_matrix(*rotation)

    theta = np.linspace(0, 2 * np.pi, n_points)

    all_x: list[float] = []
    all_y: list[float] = []
    all_z: list[float] = []

    # XY plane (major-semi_major)
    x = major * np.cos(theta)
    y = semi_major * np.sin(theta)
    z = np.zeros_like(theta)
    points = np.array([x, y, z])
    rotated = rot_matrix @ points
    all_x.extend(rotated[0].tolist() + [np.nan])
    all_y.extend(rotated[1].tolist() + [np.nan])
    all_z.extend(rotated[2].tolist() + [np.nan])

    # XZ plane (major-minor)
    x = major * np.cos(theta)
    y = np.zeros_like(theta)
    z = minor * np.sin(theta)
    points = np.array([x, y, z])
    rotated = rot_matrix @ points
    all_x.extend(rotated[0].tolist() + [np.nan])
    all_y.extend(rotated[1].tolist() + [np.nan])
    all_z.extend(rotated[2].tolist() + [np.nan])

    # YZ plane (semi_major-minor)
    x = np.zeros_like(theta)
    y = semi_major * np.cos(theta)
    z = minor * np.sin(theta)
    points = np.array([x, y, z])
    rotated = rot_matrix @ points
    all_x.extend(rotated[0].tolist() + [np.nan])
    all_y.extend(rotated[1].tolist() + [np.nan])
    all_z.extend(rotated[2].tolist() + [np.nan])

    return EllipsoidData(
        x=np.array(all_x),
        y=np.array(all_y),
        z=np.array(all_z),
        ranges=ranges,
    )


def generate_ellipsoid_mesh(
    ranges: tuple[float, float, float],
    rotation: tuple[float, float, float] = (0, 0, 0),
    n_points: int = 20,
) -> EllipsoidData:
    """Generate surface mesh data for a 3D ellipsoid.

    Creates 2D coordinate arrays suitable for surface rendering.

    Args:
        ranges: Tuple of (major, semi_major, minor) axis lengths.
        rotation: Tuple of (dip_azimuth, dip, pitch) angles in degrees.
        n_points: Number of points in each direction for mesh resolution.

    Returns:
        EllipsoidData with x, y, z as 2D arrays for surface mesh.

    Example with Plotly:
        >>> data = generate_ellipsoid_mesh(ranges=(200, 150, 100))
        >>> import plotly.graph_objects as go
        >>> fig = go.Figure(data=[go.Surface(x=data.x, y=data.y, z=data.z, opacity=0.5)])
        >>> fig.show()

    Example with K3D:
        >>> data = generate_ellipsoid_mesh(ranges=(200, 150, 100), n_points=30)
        >>> import k3d
        >>> import numpy as np
        >>> # Create vertices and indices for K3D mesh
        >>> vertices = np.column_stack([data.x.flatten(), data.y.flatten(), data.z.flatten()]).astype(np.float32)
        >>> # Generate triangle indices from the grid
        >>> n = data.x.shape[0]
        >>> indices = []
        >>> for i in range(n - 1):
        ...     for j in range(n - 1):
        ...         idx = i * n + j
        ...         indices.extend([idx, idx + 1, idx + n])
        ...         indices.extend([idx + 1, idx + n + 1, idx + n])
        >>> indices = np.array(indices, dtype=np.uint32)
        >>> plot = k3d.plot()
        >>> plot += k3d.mesh(vertices, indices, color=0x0000ff, opacity=0.5)
        >>> plot.display()
    """
    major, semi_major, minor = ranges
    rot_matrix = _rotation_matrix(*rotation)

    u = np.linspace(0, 2 * np.pi, n_points)
    v = np.linspace(0, np.pi, n_points)
    u, v = np.meshgrid(u, v)

    x = major * np.cos(u) * np.sin(v)
    y = semi_major * np.sin(u) * np.sin(v)
    z = minor * np.cos(v)

    shape = x.shape
    points = np.array([x.flatten(), y.flatten(), z.flatten()])
    rotated = rot_matrix @ points

    return EllipsoidData(
        x=rotated[0].reshape(shape),
        y=rotated[1].reshape(shape),
        z=rotated[2].reshape(shape),
        ranges=ranges,
    )


def generate_variogram_curves(
    variogram: VariogramLike | dict[str, Any],
    max_lag: float | None = None,
    n_points: int = 200,
) -> tuple[VariogramCurveData, VariogramCurveData, VariogramCurveData]:
    """Generate variogram curve data for the three principal directions.

    Calculates the variogram model along the major, semi-major, and minor
    axis directions. Each direction uses the corresponding range from the
    anisotropy ellipsoid.

    Args:
        variogram: A Variogram object or dict with 'sill', 'nugget', and 'structures'.
        max_lag: Maximum lag distance. If None, uses 1.2x the maximum range.
        n_points: Number of points for smooth curves.

    Returns:
        Tuple of (major_curve, semi_major_curve, minor_curve) as VariogramCurveData.

    Example with Plotly:
        >>> major, semi_maj, minor = generate_variogram_curves(variogram)
        >>> import plotly.graph_objects as go
        >>> fig = go.Figure()
        >>> fig.add_trace(go.Scatter(x=minor.distance, y=minor.semivariance,
        ...                          name='Minor', line=dict(color='blue')))
        >>> fig.add_trace(go.Scatter(x=semi_maj.distance, y=semi_maj.semivariance,
        ...                          name='Semi-major', line=dict(color='green')))
        >>> fig.add_trace(go.Scatter(x=major.distance, y=major.semivariance,
        ...                          name='Major', line=dict(color='red')))
        >>> fig.update_layout(xaxis_title='Distance', yaxis_title='Semivariance')
        >>> fig.show()

    Example with matplotlib:
        >>> major, semi_maj, minor = generate_variogram_curves(variogram)
        >>> import matplotlib.pyplot as plt
        >>> plt.figure(figsize=(10, 6))
        >>> plt.plot(minor.distance, minor.semivariance, 'b-', label='Minor', linewidth=2)
        >>> plt.plot(semi_maj.distance, semi_maj.semivariance, 'g-', label='Semi-major', linewidth=2)
        >>> plt.plot(major.distance, major.semivariance, 'r-', label='Major', linewidth=2)
        >>> plt.axhline(y=major.sill, color='gray', linestyle='--', label=f'Sill={major.sill:.2f}')
        >>> plt.xlabel('Distance')
        >>> plt.ylabel('Semivariance')
        >>> plt.legend()
        >>> plt.grid(True, alpha=0.3)
        >>> plt.show()
    """
    if isinstance(variogram, dict):
        sill = variogram["sill"]
        nugget = variogram.get("nugget", 0.0)
        structures = variogram["structures"]
    else:
        sill = variogram.sill
        nugget = variogram.nugget
        structures = variogram.structures

    max_ranges = {"major": 0.0, "semi_major": 0.0, "minor": 0.0}
    for struct in structures:
        anisotropy = struct.get("anisotropy", {})
        ranges = anisotropy.get("ellipsoid_ranges", {})
        for direction in max_ranges:
            max_ranges[direction] = max(max_ranges[direction], ranges.get(direction, 0))

    if max_lag is None:
        max_lag = max(max_ranges.values()) * 1.2 if max(max_ranges.values()) > 0 else 100.0

    h = np.linspace(0, max_lag, n_points)

    results = []
    for direction in ["major", "semi_major", "minor"]:
        gamma = np.full_like(h, nugget, dtype=float)

        for struct in structures:
            vtype = struct.get("variogram_type", "unknown")
            contribution = struct.get("contribution", 0)
            alpha = struct.get("alpha")
            anisotropy = struct.get("anisotropy", {})
            ranges = anisotropy.get("ellipsoid_ranges", {})
            range_val = ranges.get(direction, 1.0)

            gamma += _evaluate_structure(vtype, h, contribution, range_val, alpha)

        results.append(
            VariogramCurveData(
                distance=h.copy(),
                semivariance=gamma,
                direction=direction,
                range_value=max_ranges[direction],
                sill=sill,
            )
        )

    return results[0], results[1], results[2]
