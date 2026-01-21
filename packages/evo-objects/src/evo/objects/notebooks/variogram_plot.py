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

"""Interactive visualization of variogram models using Plotly.

This module provides functions to visualize variogram models in Jupyter notebooks,
including:
- 3D anisotropy ellipsoids showing spatial correlation ranges and orientation
- 2D directional variogram curves showing semivariance vs lag distance

All visualizations use Plotly for interactivity and support both light and dark modes.

Requires: pip install evo-objects[viz]
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

try:
    import numpy as np
    import plotly.graph_objects as go
except ImportError as e:
    raise ImportError(
        "Variogram visualization requires plotly and numpy. Install with: pip install evo-objects[viz]"
    ) from e

__all__ = [
    "plot_variogram",
    "plot_variogram_2d",
    "plot_variogram_3d",
    "plot_variogram_ellipsoids",
    "plot_variogram_model",
]

# Color scheme for directional variograms - bright colors that work in both light and dark mode
DIRECTION_COLORS = {
    "minor": "#1E90FF",  # Dodger blue - visible on both backgrounds
    "semi_major": "#32CD32",  # Lime green - visible on both backgrounds
    "major": "#FF6347",  # Tomato red - visible on both backgrounds
}

# Color palette for structures (Plotly qualitative - works in both modes)
STRUCTURE_COLORS = [
    "#636EFA",  # blue
    "#EF553B",  # red
    "#00CC96",  # green
    "#AB63FA",  # purple
    "#FFA15A",  # orange
    "#19D3F3",  # cyan
    "#FF6692",  # pink
    "#B6E880",  # lime
]

NUGGET_COLOR = "#888888"  # medium gray - visible on both backgrounds


@runtime_checkable
class VariogramLike(Protocol):
    """Protocol for variogram-like objects."""

    @property
    def sill(self) -> float: ...

    @property
    def nugget(self) -> float: ...

    @property
    def structures(self) -> list[dict[str, Any]]: ...


def _get_variogram_data(
    variogram: VariogramLike | dict[str, Any],
) -> tuple[float, float, list[dict[str, Any]], str | None]:
    """Extract sill, nugget, structures, and attribute from a variogram object or dict."""
    if isinstance(variogram, dict):
        return (
            variogram["sill"],
            variogram.get("nugget", 0.0),
            variogram["structures"],
            variogram.get("attribute"),
        )
    else:
        attr = getattr(variogram, "attribute", None)
        return variogram.sill, variogram.nugget, variogram.structures, attr


def _rotation_matrix(dip_azimuth: float, dip: float, pitch: float) -> np.ndarray:
    """Create a 3D rotation matrix from Leapfrog convention angles.

    The rotation follows the Leapfrog/Geoscience Object convention:
    1. Rotate about Z-axis by dip_azimuth (clockwise from North/+Y)
    2. Rotate about X-axis by dip
    3. Rotate about Z-axis by pitch

    Args:
        dip_azimuth: Azimuth of dip direction in degrees (0-360), measured clockwise from North
        dip: Dip angle in degrees (0-180)
        pitch: Pitch/rake angle in degrees (0-360)

    Returns:
        3x3 rotation matrix
    """
    # Convert to radians
    az = np.radians(dip_azimuth)
    d = np.radians(dip)
    p = np.radians(pitch)

    # Rotation about Z-axis (dip azimuth) - clockwise
    rz1 = np.array(
        [
            [np.cos(az), np.sin(az), 0],
            [-np.sin(az), np.cos(az), 0],
            [0, 0, 1],
        ]
    )

    # Rotation about X-axis (dip)
    rx = np.array(
        [
            [1, 0, 0],
            [0, np.cos(d), -np.sin(d)],
            [0, np.sin(d), np.cos(d)],
        ]
    )

    # Rotation about Z-axis (pitch)
    rz2 = np.array(
        [
            [np.cos(p), np.sin(p), 0],
            [-np.sin(p), np.cos(p), 0],
            [0, 0, 1],
        ]
    )

    # Combined rotation: first az, then dip, then pitch
    return rz2 @ rx @ rz1


def _generate_ellipsoid_wireframe(
    ranges: tuple[float, float, float],
    rotation: np.ndarray,
    n_points: int = 30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate wireframe points for an ellipsoid.

    Args:
        ranges: (major, semi_major, minor) axis lengths
        rotation: 3x3 rotation matrix
        n_points: Number of points per circle

    Returns:
        Tuple of (x, y, z) coordinate arrays for wireframe lines
    """
    major, semi_major, minor = ranges

    # Generate points for three principal circles
    theta = np.linspace(0, 2 * np.pi, n_points)

    circles = []

    # XY plane circle (major x semi_major)
    xy_circle = np.array(
        [
            major * np.cos(theta),
            semi_major * np.sin(theta),
            np.zeros_like(theta),
        ]
    )
    circles.append(xy_circle)

    # XZ plane circle (major x minor)
    xz_circle = np.array(
        [
            major * np.cos(theta),
            np.zeros_like(theta),
            minor * np.sin(theta),
        ]
    )
    circles.append(xz_circle)

    # YZ plane circle (semi_major x minor)
    yz_circle = np.array(
        [
            np.zeros_like(theta),
            semi_major * np.cos(theta),
            minor * np.sin(theta),
        ]
    )
    circles.append(yz_circle)

    # Add latitude circles at different heights
    for z_frac in [-0.7, -0.3, 0.3, 0.7]:
        z_val = minor * z_frac
        # Radius at this height for an ellipsoid
        r_scale = np.sqrt(1 - z_frac**2)
        lat_circle = np.array(
            [
                major * r_scale * np.cos(theta),
                semi_major * r_scale * np.sin(theta),
                np.full_like(theta, z_val),
            ]
        )
        circles.append(lat_circle)

    # Apply rotation to all circles
    all_x, all_y, all_z = [], [], []
    for circle in circles:
        rotated = rotation @ circle
        # Add NaN to create line breaks between circles
        all_x.extend(rotated[0].tolist() + [np.nan])
        all_y.extend(rotated[1].tolist() + [np.nan])
        all_z.extend(rotated[2].tolist() + [np.nan])

    return np.array(all_x), np.array(all_y), np.array(all_z)


def _generate_ellipsoid_surface(
    ranges: tuple[float, float, float],
    rotation: np.ndarray,
    n_points: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate surface mesh points for an ellipsoid.

    Args:
        ranges: (major, semi_major, minor) axis lengths
        rotation: 3x3 rotation matrix
        n_points: Number of points in each direction

    Returns:
        Tuple of (x, y, z) coordinate arrays for surface mesh
    """
    major, semi_major, minor = ranges

    # Parametric ellipsoid
    u = np.linspace(0, 2 * np.pi, n_points)
    v = np.linspace(0, np.pi, n_points)
    u, v = np.meshgrid(u, v)

    # Ellipsoid coordinates
    x = major * np.cos(u) * np.sin(v)
    y = semi_major * np.sin(u) * np.sin(v)
    z = minor * np.cos(v)

    # Apply rotation
    shape = x.shape
    points = np.array([x.flatten(), y.flatten(), z.flatten()])
    rotated = rotation @ points

    return (
        rotated[0].reshape(shape),
        rotated[1].reshape(shape),
        rotated[2].reshape(shape),
    )


def _evaluate_structure(
    structure_type: str,
    h: np.ndarray,
    contribution: float,
    range_val: float,
    alpha: int | None = None,
) -> np.ndarray:
    """Evaluate a variogram structure model.

    Args:
        structure_type: Type of variogram model (spherical, exponential, gaussian, etc.)
        h: Lag distances (normalized by range)
        contribution: Contribution of this structure to total variance
        range_val: Range parameter
        alpha: Shape parameter for spheroidal and generalised cauchy models

    Returns:
        Semivariance values
    """
    # Normalize lag by range
    h_norm = h / range_val if range_val > 0 else h

    if structure_type == "spherical":
        # Spherical model: reaches sill at range
        gamma = np.where(
            h_norm < 1,
            contribution * (1.5 * h_norm - 0.5 * h_norm**3),
            contribution,
        )
    elif structure_type == "exponential":
        # Exponential model: approaches sill asymptotically (practical range ≈ 3*range)
        gamma = contribution * (1 - np.exp(-3 * h_norm))
    elif structure_type == "gaussian":
        # Gaussian model: smooth near origin, approaches sill asymptotically
        gamma = contribution * (1 - np.exp(-3 * h_norm**2))
    elif structure_type == "cubic":
        # Cubic model: smooth polynomial transition
        gamma = np.where(
            h_norm < 1,
            contribution * (7 * h_norm**2 - 8.75 * h_norm**3 + 3.5 * h_norm**5 - 0.75 * h_norm**7),
            contribution,
        )
    elif structure_type == "linear":
        # Linear model: unbounded, increases indefinitely
        gamma = contribution * h_norm
    elif structure_type == "spheroidal":
        # Spheroidal model: generalization of spherical with alpha parameter
        # Using simplified approximation
        if alpha is None:
            alpha = 3
        gamma = np.where(
            h_norm < 1,
            contribution * (1 - (1 - h_norm**2) ** (alpha / 2)),
            contribution,
        )
    elif structure_type == "generalisedcauchy":
        # Generalised Cauchy model: long-range correlation
        if alpha is None:
            alpha = 3
        gamma = contribution * (1 - (1 + h_norm**2) ** (-alpha / 2))
    else:
        # Unknown type - treat as nugget-like (instant jump to contribution)
        gamma = np.full_like(h, contribution)

    return gamma


def plot_variogram_ellipsoids(
    variogram: VariogramLike | dict[str, Any],
    *,
    surface: bool = False,
    show_axes: bool = True,
    title: str | None = None,
) -> go.Figure:
    """Plot 3D anisotropy ellipsoids for a variogram model.

    Creates an interactive 3D visualization showing the spatial correlation
    ellipsoids for each structure in the variogram. The ellipsoid axes represent
    the correlation ranges in each direction, and the orientation shows how
    spatial correlation varies with direction.

    Args:
        variogram: A Variogram object or dict with 'structures' containing
            anisotropy definitions (ellipsoid_ranges and rotation)
        surface: If True, render as semi-transparent surfaces instead of wireframes
        show_axes: If True, show axis labels and grid
        title: Optional title for the plot

    Returns:
        Plotly Figure object that can be displayed in Jupyter or saved

    Example:
        >>> from evo.objects.notebooks import plot_variogram_ellipsoids
        >>> fig = plot_variogram_ellipsoids(variogram)
        >>> fig.show()
    """
    sill, nugget, structures, _attribute = _get_variogram_data(variogram)

    fig = go.Figure()

    max_range = 0.0

    for i, struct in enumerate(structures):
        color = STRUCTURE_COLORS[i % len(STRUCTURE_COLORS)]
        vtype = struct.get("variogram_type", "unknown")
        contribution = struct.get("contribution", 0)

        anisotropy = struct.get("anisotropy", {})
        ranges_dict = anisotropy.get("ellipsoid_ranges", {})
        rotation_dict = anisotropy.get("rotation", {})

        ranges = (
            ranges_dict.get("major", 1),
            ranges_dict.get("semi_major", 1),
            ranges_dict.get("minor", 1),
        )
        max_range = max(max_range, max(ranges))

        rotation = _rotation_matrix(
            rotation_dict.get("dip_azimuth", 0),
            rotation_dict.get("dip", 0),
            rotation_dict.get("pitch", 0),
        )

        if surface:
            x, y, z = _generate_ellipsoid_surface(ranges, rotation)
            fig.add_trace(
                go.Surface(
                    x=x,
                    y=y,
                    z=z,
                    colorscale=[[0, color], [1, color]],
                    showscale=False,
                    opacity=0.3,
                    name=f"Structure {i + 1}: {vtype}",
                )
            )
            # Add wireframe on top for clarity
            xw, yw, zw = _generate_ellipsoid_wireframe(ranges, rotation)
            fig.add_trace(
                go.Scatter3d(
                    x=xw,
                    y=yw,
                    z=zw,
                    mode="lines",
                    line={"color": color, "width": 2},
                    name=f"Structure {i + 1}: {vtype} (wireframe)",
                    showlegend=False,
                )
            )
        else:
            x, y, z = _generate_ellipsoid_wireframe(ranges, rotation)
            fig.add_trace(
                go.Scatter3d(
                    x=x,
                    y=y,
                    z=z,
                    mode="lines",
                    line={"color": color, "width": 2},
                    name=f"Structure {i + 1}: {vtype} (C={contribution:.3g})",
                )
            )

        # Add principal axis arrows
        axis_scale = 1.1
        axes = rotation @ np.diag(ranges) * axis_scale
        origin = np.zeros(3)

        for j, (axis_name, axis_range) in enumerate(zip(["Major", "Semi-major", "Minor"], ranges, strict=False)):
            end = axes[:, j]
            fig.add_trace(
                go.Scatter3d(
                    x=[origin[0], end[0]],
                    y=[origin[1], end[1]],
                    z=[origin[2], end[2]],
                    mode="lines",
                    line={"color": color, "width": 4},
                    name=f"S{i + 1} {axis_name}: {axis_range:.1f}",
                    showlegend=False,
                    hoverinfo="name",
                )
            )

    # Set axis properties
    axis_range = [-max_range * 1.2, max_range * 1.2]
    axis_props = {
        "range": axis_range,
        "showgrid": show_axes,
        "showticklabels": show_axes,
        "title": "" if not show_axes else None,
    }

    fig.update_layout(
        title=title or "Variogram Anisotropy Ellipsoids",
        scene={
            "xaxis": {**axis_props, "title": "X (East)" if show_axes else ""},
            "yaxis": {**axis_props, "title": "Y (North)" if show_axes else ""},
            "zaxis": {**axis_props, "title": "Z (Up)" if show_axes else ""},
            "aspectmode": "cube",
        },
        legend={"yanchor": "top", "y": 0.99, "xanchor": "left", "x": 0.01},
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
    )

    return fig


# Alias for backwards compatibility
plot_variogram_3d = plot_variogram_ellipsoids


def _create_directional_variogram_figure(
    h: np.ndarray,
    gamma: np.ndarray,
    sill: float,
    direction_label: str,
    color: str,
    attribute: str | None,
    max_lag: float,
    height: int = 450,
    width: int = 600,
) -> go.Figure:
    """Create a single directional variogram figure.

    Args:
        h: Lag distances
        gamma: Semivariance values
        sill: Variogram sill value
        direction_label: Label for the direction (e.g., "Minor", "Semi-major", "Major")
        color: Color for the curve
        attribute: Attribute name for title
        max_lag: Maximum lag distance
        height: Figure height in pixels
        width: Figure width in pixels

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    # Plot the variogram curve
    fig.add_trace(
        go.Scatter(
            x=h,
            y=gamma,
            mode="lines",
            line={"color": color, "width": 3},
            name=f"{direction_label} Axis",
            hovertemplate=f"{direction_label}<br>Distance: %{{x:.1f}}<br>γ: %{{y:.2f}}<extra></extra>",
        )
    )

    # Add horizontal dashed line at sill
    fig.add_trace(
        go.Scatter(
            x=[0, max_lag],
            y=[sill, sill],
            mode="lines",
            line={"color": color, "width": 1.5, "dash": "dash"},
            name=f"Total Sill: {sill:.3g}",
            hoverinfo="skip",
        )
    )

    # Build title
    title_text = f"{direction_label} Axis Variogram"
    if attribute:
        title_text += f" for {attribute}"

    fig.update_layout(
        title=title_text,
        xaxis_title="Distance",
        yaxis_title="(Semi-)Variogram",
        xaxis={
            "rangemode": "tozero",
            "range": [0, max_lag],
            "showgrid": True,
            "gridwidth": 1,
            "gridcolor": "rgba(128, 128, 128, 0.3)",
        },
        yaxis={
            "rangemode": "tozero",
            "range": [0, max(gamma.max(), sill) * 1.15],
            "showgrid": True,
            "gridwidth": 1,
            "gridcolor": "rgba(128, 128, 128, 0.3)",
        },
        legend={
            "yanchor": "top",
            "y": 0.99,
            "xanchor": "left",
            "x": 0.01,
        },
        height=height,
        width=width,
        template="plotly",
    )

    return fig


def plot_variogram_2d(
    variogram: VariogramLike | dict[str, Any],
    *,
    max_lag: float | None = None,
    n_points: int = 200,
    title: str | None = None,
    height: int = 500,
    width: int = 700,
) -> tuple[go.Figure, go.Figure, go.Figure, go.Figure]:
    """Plot 2D directional variogram curves showing semivariance vs lag distance.

    Creates four Plotly figures:
    1. Combined plot with all three directional curves (Minor, Semi-major, Major)
    2. Minor axis variogram (fastest to reach sill)
    3. Semi-major axis variogram
    4. Major axis variogram (slowest to reach sill)

    Each direction uses the corresponding range from the anisotropy ellipsoid.

    Args:
        variogram: A Variogram object or dict with 'sill', 'nugget', and 'structures'
        max_lag: Maximum lag distance to plot. If None, uses 1.2x the maximum range
        n_points: Number of points for smooth curves
        title: Optional title for the combined plot
        height: Figure height in pixels
        width: Figure width in pixels

    Returns:
        Tuple of (combined_fig, minor_fig, semi_major_fig, major_fig)

    Example:
        >>> from evo.objects.notebooks import plot_variogram_2d
        >>> combined, minor, semi_maj, major = plot_variogram_2d(variogram)
        >>> combined.show()
        >>> minor.show()
    """
    sill, nugget, structures, attribute = _get_variogram_data(variogram)

    # Find the maximum range across all structures for each direction
    max_ranges = {"major": 0.0, "semi_major": 0.0, "minor": 0.0}
    for struct in structures:
        anisotropy = struct.get("anisotropy", {})
        ranges = anisotropy.get("ellipsoid_ranges", {})
        for direction in max_ranges:
            max_ranges[direction] = max(max_ranges[direction], ranges.get(direction, 0))

    # Determine max lag for the plot
    if max_lag is None:
        max_lag = max(max_ranges.values()) * 1.2 if max(max_ranges.values()) > 0 else 100.0

    h = np.linspace(0, max_lag, n_points)

    # Direction labels and colors
    directions = [
        ("minor", "Minor", DIRECTION_COLORS["minor"]),
        ("semi_major", "Semi-major", DIRECTION_COLORS["semi_major"]),
        ("major", "Major", DIRECTION_COLORS["major"]),
    ]

    # Calculate gamma for each direction and store
    direction_data = {}
    for direction_key, direction_label, color in directions:
        gamma = np.full_like(h, nugget, dtype=float)
        for struct in structures:
            vtype = struct.get("variogram_type", "unknown")
            contribution = struct.get("contribution", 0)
            alpha = struct.get("alpha")
            anisotropy = struct.get("anisotropy", {})
            ranges = anisotropy.get("ellipsoid_ranges", {})
            range_val = ranges.get(direction_key, 1.0)
            gamma += _evaluate_structure(vtype, h, contribution, range_val, alpha)

        direction_data[direction_key] = {
            "gamma": gamma,
            "label": direction_label,
            "color": color,
            "max_lag": max_ranges[direction_key] * 1.2 if max_ranges[direction_key] > 0 else max_lag,
        }

    # Create combined figure
    fig_combined = go.Figure()
    for direction_key, direction_label, color in directions:
        data = direction_data[direction_key]
        fig_combined.add_trace(
            go.Scatter(
                x=h,
                y=data["gamma"],
                mode="lines",
                line={"color": color, "width": 3},
                name=data["label"],
                hovertemplate=f"{data['label']}<br>Distance: %{{x:.1f}}<br>γ: %{{y:.2f}}<extra></extra>",
            )
        )

    # Set combined figure title
    plot_title = title
    if not plot_title:
        plot_title = f"Variogram for {attribute}" if attribute else "Variogram Model"

    fig_combined.update_layout(
        title=plot_title,
        xaxis_title="Distance",
        yaxis_title="(Semi-)Variogram",
        xaxis={
            "rangemode": "tozero",
            "range": [0, max_lag],
            "showgrid": True,
            "gridwidth": 1,
            "gridcolor": "rgba(128, 128, 128, 0.3)",
        },
        yaxis={
            "rangemode": "tozero",
            "range": [0, sill * 1.15],
            "showgrid": True,
            "gridwidth": 1,
            "gridcolor": "rgba(128, 128, 128, 0.3)",
        },
        legend={
            "yanchor": "bottom",
            "y": 0.01,
            "xanchor": "right",
            "x": 0.99,
        },
        height=height,
        width=width,
        template="plotly",
    )

    # Create individual directional figures
    fig_minor = _create_directional_variogram_figure(
        h=h,
        gamma=direction_data["minor"]["gamma"],
        sill=sill,
        direction_label="Minor",
        color=DIRECTION_COLORS["minor"],
        attribute=attribute,
        max_lag=direction_data["minor"]["max_lag"],
    )

    fig_semi_major = _create_directional_variogram_figure(
        h=h,
        gamma=direction_data["semi_major"]["gamma"],
        sill=sill,
        direction_label="Semi-major",
        color=DIRECTION_COLORS["semi_major"],
        attribute=attribute,
        max_lag=direction_data["semi_major"]["max_lag"],
    )

    fig_major = _create_directional_variogram_figure(
        h=h,
        gamma=direction_data["major"]["gamma"],
        sill=sill,
        direction_label="Major",
        color=DIRECTION_COLORS["major"],
        attribute=attribute,
        max_lag=direction_data["major"]["max_lag"],
    )

    return fig_combined, fig_minor, fig_semi_major, fig_major


# Legacy alias for backwards compatibility
def plot_variogram_model(
    variogram: VariogramLike | dict[str, Any],
    *,
    max_lag: float | None = None,
    n_points: int = 200,
    show_structures: bool = True,
    title: str | None = None,
) -> tuple[go.Figure, go.Figure, go.Figure, go.Figure]:
    """Plot 2D variogram model curves (legacy function).

    This function is kept for backwards compatibility. Consider using
    `plot_variogram_2d()` for a cleaner directional variogram plot.
    """
    return plot_variogram_2d(variogram, max_lag=max_lag, n_points=n_points, title=title)


def plot_variogram(
    variogram: VariogramLike | dict[str, Any],
    *,
    surface: bool = False,
    max_lag: float | None = None,
    title: str | None = None,
) -> tuple[go.Figure, go.Figure, go.Figure, go.Figure, go.Figure]:
    """Plot variogram visualization with 2D directional curves and 3D ellipsoids.

    Returns five separate Plotly figures:
    1. Combined 2D plot with all three directional curves
    2. Minor axis variogram
    3. Semi-major axis variogram
    4. Major axis variogram
    5. 3D anisotropy ellipsoids (interactive)

    This combined view helps geostatisticians understand both the directional
    anisotropy and the overall variogram model behavior.

    Args:
        variogram: A Variogram object or dict with variogram model parameters
        surface: If True, render 3D ellipsoids as semi-transparent surfaces
        max_lag: Maximum lag distance for the 2D plots
        title: Optional base title for the plots

    Returns:
        Tuple of (combined_2d, minor_2d, semi_major_2d, major_2d, ellipsoids_3d)

    Example:
        >>> from evo.objects.notebooks import plot_variogram
        >>> combined, minor, semi_maj, major, ellipsoids = plot_variogram(variogram)
        >>> combined.show()
        >>> minor.show()
        >>> ellipsoids.show()
    """
    sill, nugget, structures, attribute = _get_variogram_data(variogram)

    # Generate 2D directional variogram plots
    title_2d = title or (f"Variogram for {attribute}" if attribute else "Variogram Model")
    fig_combined, fig_minor, fig_semi_major, fig_major = plot_variogram_2d(variogram, max_lag=max_lag, title=title_2d)

    # Generate 3D ellipsoid plot
    title_3d = "Anisotropy Ellipsoids" + (f" - {attribute}" if attribute else "")
    fig_3d = plot_variogram_ellipsoids(variogram, surface=surface, title=title_3d)

    return fig_combined, fig_minor, fig_semi_major, fig_major, fig_3d
