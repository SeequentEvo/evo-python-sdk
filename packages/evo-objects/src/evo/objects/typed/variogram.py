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

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from pydantic import TypeAdapter

from evo.common.styles.html import STYLESHEET, build_nested_table, build_table_row, build_title
from evo.objects import SchemaVersion

from ._property import SchemaProperty
from .base import BaseObjectData, ConstructableObject

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .ellipsoid import Ellipsoid

__all__ = [
    "Anisotropy",
    "CubicStructure",
    "EllipsoidRanges",
    "ExponentialStructure",
    "GaussianStructure",
    "GeneralisedCauchyStructure",
    "LinearStructure",
    "SphericalStructure",
    "SpheroidalStructure",
    "Variogram",
    "VariogramCurveData",
    "VariogramData",
    "VariogramRotation",
    "VariogramStructure",
]


@dataclass(frozen=True, kw_only=True)
class EllipsoidRanges:
    """Ellipsoid ranges defining the spatial extent of correlation in each direction."""

    major: float
    """Range in the major (longest) direction."""

    semi_major: float
    """Range in the semi-major (intermediate) direction."""

    minor: float
    """Range in the minor (shortest) direction."""

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary format for the schema."""
        return {
            "major": self.major,
            "semi_major": self.semi_major,
            "minor": self.minor,
        }


@dataclass(frozen=True, kw_only=True)
class VariogramRotation:
    """Rotation angles for variogram anisotropy using Leapfrog convention."""

    dip_azimuth: float = 0.0
    """Azimuth of the dip direction in degrees (0-360)."""

    dip: float = 0.0
    """Dip angle in degrees (0-90)."""

    pitch: float = 0.0
    """Pitch/rake angle in degrees."""

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary format for the schema."""
        return {
            "dip_azimuth": self.dip_azimuth,
            "dip": self.dip,
            "pitch": self.pitch,
        }


@dataclass(frozen=True, kw_only=True)
class Anisotropy:
    """Anisotropy definition combining ellipsoid ranges and rotation."""

    ellipsoid_ranges: EllipsoidRanges
    """The ranges of spatial correlation in each direction."""

    rotation: VariogramRotation = field(default_factory=lambda: VariogramRotation())
    """The rotation of the anisotropy ellipsoid."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format for the schema."""
        return {
            "ellipsoid_ranges": self.ellipsoid_ranges.to_dict(),
            "rotation": self.rotation.to_dict(),
        }


@dataclass(frozen=True, kw_only=True)
class VariogramStructure:
    """Base class for variogram structures."""

    contribution: float
    """The contribution of this structure to the total variance."""

    anisotropy: Anisotropy
    """The anisotropy definition for this structure."""

    variogram_type: str = field(init=False)
    """The type of variogram structure (set by subclasses)."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format for the schema."""
        return {
            "variogram_type": self.variogram_type,
            "contribution": self.contribution,
            "anisotropy": self.anisotropy.to_dict(),
        }

    def to_ellipsoid(self) -> "Ellipsoid":
        """Convert this structure's anisotropy to an Ellipsoid for visualization or search.

        Returns an Ellipsoid from evo.compute.tasks that can be used for:
        - 3D visualization with surface_points() or wireframe_points()
        - Creating search ellipsoids via scaled()
        - Kriging search neighborhoods

        Example:
            >>> # Get ellipsoid from variogram structure
            >>> var_ell = variogram.structures[0].to_ellipsoid()
            >>>
            >>> # Create search ellipsoid scaled by 2x
            >>> search_ell = var_ell.scaled(2.0)
            >>>
            >>> # Visualize with Plotly
            >>> x, y, z = var_ell.surface_points(center=(100, 200, 50))
            >>> mesh = go.Mesh3d(x=x, y=y, z=z, alphahull=0, opacity=0.3)
        """
        from .ellipsoid import Ellipsoid, EllipsoidRanges as EllipsoidRangesObj, Rotation

        ranges = self.anisotropy.ellipsoid_ranges
        rotation = self.anisotropy.rotation

        return Ellipsoid(
            ranges=EllipsoidRangesObj(
                major=ranges.major,
                semi_major=ranges.semi_major,
                minor=ranges.minor,
            ),
            rotation=Rotation(
                dip_azimuth=rotation.dip_azimuth,
                dip=rotation.dip,
                pitch=rotation.pitch,
            ),
        )


@dataclass(frozen=True, kw_only=True)
class SphericalStructure(VariogramStructure):
    """Spherical variogram structure.

    The spherical model is one of the most commonly used variogram models.
    It reaches its sill at a finite range.
    """

    variogram_type: str = field(default="spherical", init=False)


@dataclass(frozen=True, kw_only=True)
class ExponentialStructure(VariogramStructure):
    """Exponential variogram structure.

    The exponential model approaches its sill asymptotically.
    The practical range is about 3 times the effective range parameter.
    """

    variogram_type: str = field(default="exponential", init=False)


@dataclass(frozen=True, kw_only=True)
class GaussianStructure(VariogramStructure):
    """Gaussian variogram structure.

    The Gaussian model has a parabolic behavior near the origin,
    indicating very smooth spatial variation.
    """

    variogram_type: str = field(default="gaussian", init=False)


@dataclass(frozen=True, kw_only=True)
class CubicStructure(VariogramStructure):
    """Cubic variogram structure.

    The cubic model provides smooth transitions and is bounded.
    """

    variogram_type: str = field(default="cubic", init=False)


@dataclass(frozen=True, kw_only=True)
class LinearStructure(VariogramStructure):
    """Linear variogram structure.

    The linear model has no sill and increases indefinitely.
    Useful for modeling trends or unbounded variability.
    """

    variogram_type: str = field(default="linear", init=False)


@dataclass(frozen=True, kw_only=True)
class SpheroidalStructure(VariogramStructure):
    """Spheroidal variogram structure.

    The spheroidal model is a generalization of the spherical model
    with a shape parameter (alpha) that controls the curvature.
    """

    alpha: Literal[3, 5, 7, 9]
    """Shape factor of the spheroidal model. Valid values: 3, 5, 7, or 9."""

    variogram_type: str = field(default="spheroidal", init=False)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format for the schema."""
        return {
            "variogram_type": self.variogram_type,
            "contribution": self.contribution,
            "anisotropy": self.anisotropy.to_dict(),
            "alpha": self.alpha,
        }


@dataclass(frozen=True, kw_only=True)
class GeneralisedCauchyStructure(VariogramStructure):
    """Generalised Cauchy variogram structure.

    The Generalised Cauchy model allows for long-range correlation with
    a shape parameter (alpha) that controls the decay behavior.
    """

    alpha: Literal[3, 5, 7, 9]
    """Shape factor of the Cauchy model. Valid values: 3, 5, 7, or 9."""

    variogram_type: str = field(default="generalisedcauchy", init=False)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format for the schema."""
        return {
            "variogram_type": self.variogram_type,
            "contribution": self.contribution,
            "anisotropy": self.anisotropy.to_dict(),
            "alpha": self.alpha,
        }


def _convert_structures(structures: list[VariogramStructure | dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert a list of structures to dictionary format."""
    result = []
    for struct in structures:
        if isinstance(struct, VariogramStructure):
            result.append(struct.to_dict())
        else:
            # Already a dict, pass through
            result.append(struct)
    return result


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
    """

    distance: "NDArray[np.floating[Any]]"
    semivariance: "NDArray[np.floating[Any]]"
    direction: str
    range_value: float
    sill: float


def _evaluate_structure(
    structure_type: str,
    h: "NDArray[np.floating[Any]]",
    contribution: float,
    range_val: float,
    alpha: int | None = None,
) -> "NDArray[np.floating[Any]]":
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


@dataclass(kw_only=True, frozen=True)
class VariogramData(BaseObjectData):
    """Data for creating a Variogram.

    A variogram is a geostatistical model describing spatial correlation structure.
    The variogram model is defined by the nugget and multiple structures using the
    leapfrog-convention rotation.

    Note:
        When using a variogram with kriging tasks, the following fields should be set:
        - `modelling_space`: Set to "data" for original units or "normalscore" for gaussian space
        - `data_variance`: Should match the sill value for non-normalized data

    Example using typed structures (recommended):
        >>> data = VariogramData(
        ...     name="My Variogram",
        ...     sill=1.0,
        ...     nugget=0.1,
        ...     is_rotation_fixed=True,
        ...     modelling_space="data",  # Required for kriging
        ...     data_variance=1.0,       # Required for kriging
        ...     structures=[
        ...         SphericalStructure(
        ...             contribution=0.9,
        ...             anisotropy=Anisotropy(
        ...                 ellipsoid_ranges=EllipsoidRanges(major=200, semi_major=150, minor=100),
        ...                 rotation=VariogramRotation(dip_azimuth=0, dip=0, pitch=0),
        ...             ),
        ...         ),
        ...     ],
        ...     attribute="grade",
        ... )
    """

    sill: float
    """The variance of the variogram. Must be within a very small tolerance of the nugget
    plus the sum of all structure's contributions."""

    is_rotation_fixed: bool
    """Boolean value specifying whether all structure's rotations are the same."""

    structures: list[VariogramStructure | dict[str, Any]]
    """A list of at least one mathematical model, which are parameterised to represent
    the spatial structure of the variogram model. Can use typed classes like
    SphericalStructure, ExponentialStructure, GaussianStructure, CubicStructure,
    or raw dictionaries."""

    nugget: float = 0.0
    """The variance between two samples separated by near-zero lag distance, representing
    the randomness present. When plotted, this value is the y-intercept."""

    data_variance: float | None = None
    """The variance of the data, if different from the sill value, this is used for
    normalising or rescaling the variogram."""

    modelling_space: Literal["data", "normalscore"] | None = None
    """The modelling space the variogram model was fitted in - either 'data' for original
    units or 'normalscore' for gaussian space."""

    domain: str | None = None
    """The domain the variogram is modelled for."""

    attribute: str | None = None
    """The attribute the variogram is modelled for."""

    def get_structures_as_dicts(self) -> list[dict[str, Any]]:
        """Get structures as a list of dictionaries for serialization."""
        return _convert_structures(self.structures)


class Variogram(ConstructableObject[VariogramData]):
    """A GeoscienceObject representing a variogram.

    The variogram describes the spatial correlation structure of a variable,
    used in geostatistical modeling and kriging interpolation.
    """

    _data_class = VariogramData

    sub_classification = "variogram"
    # Note: Using v1.1.0 for compatibility with kriging task runtime
    # v1.2.0 adds spheroidal and linear structures, but the task runtime
    # currently only supports v1.1.0 schemas
    creation_schema_version = SchemaVersion(major=1, minor=1, patch=0)

    @classmethod
    async def _data_to_dict(cls, data: VariogramData, context: Any) -> dict[str, Any]:
        """Convert VariogramData to a dictionary for creating the Geoscience Object.

        Overrides the base implementation to handle typed structure conversion.
        """
        # Convert structures to dicts BEFORE calling super() to avoid Pydantic
        # serialization warnings (it expects dict but gets dataclass objects)
        converted_structures = data.get_structures_as_dicts()

        # Create a modified data object with pre-converted structures
        # This avoids the warning from TypeAdapter.dump_python()
        modified_data = replace(data, structures=converted_structures)

        # Get base dict from parent class (now with dict structures)
        result = await super()._data_to_dict(modified_data, context)

        return result

    sill: float = SchemaProperty("sill", TypeAdapter(float))
    """The variance of the variogram."""

    is_rotation_fixed: bool = SchemaProperty("is_rotation_fixed", TypeAdapter(bool))
    """Boolean value specifying whether all structure's rotations are the same."""

    structures: list[dict[str, Any]] = SchemaProperty("structures", TypeAdapter(list[dict[str, Any]]))
    """List of variogram structures (exponential, gaussian, spherical, etc.)."""

    nugget: float = SchemaProperty("nugget", TypeAdapter(float), default_factory=lambda: 0.0)
    """The variance between two samples separated by near-zero lag distance."""

    data_variance: float | None = SchemaProperty("data_variance", TypeAdapter(float | None))
    """The variance of the data, used for normalising or rescaling the variogram."""

    modelling_space: Literal["data", "normalscore"] | None = SchemaProperty(
        "modelling_space", TypeAdapter(Literal["data", "normalscore"] | None)
    )
    """The modelling space the variogram model was fitted in."""

    domain: str | None = SchemaProperty("domain", TypeAdapter(str | None))
    """The domain the variogram is modelled for."""

    attribute: str | None = SchemaProperty("attribute", TypeAdapter(str | None))
    """The attribute the variogram is modelled for."""

    def get_ellipsoid(self, structure_index: int | None = None) -> "Ellipsoid":
        """Get an Ellipsoid from a variogram structure for visualization or search.

        Returns an Ellipsoid from evo.objects.typed that can be used for:
        - 3D visualization with surface_points() or wireframe_points()
        - Creating search ellipsoids via scaled()
        - Kriging search neighborhoods

        Args:
            structure_index: Index of the structure to use. If None (default), selects
                            the structure with the largest volume (major × semi_major × minor).

        Returns:
            Ellipsoid configured with the structure's anisotropy ranges and rotation.

        Example:
            >>> # Get ellipsoid from structure with largest range (default)
            >>> var_ell = variogram.get_ellipsoid()
            >>>
            >>> # Or explicitly select a structure by index
            >>> var_ell = variogram.get_ellipsoid(structure_index=0)
            >>>
            >>> # Create search ellipsoid scaled by 2x
            >>> search_ell = var_ell.scaled(2.0)
            >>>
            >>> # Visualize with Plotly
            >>> x, y, z = var_ell.surface_points(center=(100, 200, 50))
            >>> mesh = go.Mesh3d(x=x, y=y, z=z, alphahull=0, opacity=0.3)
        """
        from .ellipsoid import Ellipsoid, EllipsoidRanges as EllipsoidRangesObj, Rotation

        if not self.structures:
            raise ValueError("Variogram has no structures")

        # If no index specified, find structure with largest volume
        if structure_index is None:
            max_volume = -1.0
            structure_index = 0
            for i, struct in enumerate(self.structures):
                anisotropy = struct.get("anisotropy", {})
                ranges = anisotropy.get("ellipsoid_ranges", {})
                volume = (
                    ranges.get("major", 1.0) *
                    ranges.get("semi_major", 1.0) *
                    ranges.get("minor", 1.0)
                )
                if volume > max_volume:
                    max_volume = volume
                    structure_index = i

        if structure_index >= len(self.structures):
            raise ValueError(f"structure_index {structure_index} out of range (max {len(self.structures) - 1})")

        struct = self.structures[structure_index]
        anisotropy = struct.get("anisotropy", {})
        ranges_dict = anisotropy.get("ellipsoid_ranges", {})
        rotation_dict = anisotropy.get("rotation", {})

        return Ellipsoid(
            ranges=EllipsoidRangesObj(
                major=ranges_dict.get("major", 1.0),
                semi_major=ranges_dict.get("semi_major", 1.0),
                minor=ranges_dict.get("minor", 1.0),
            ),
            rotation=Rotation(
                dip_azimuth=rotation_dict.get("dip_azimuth", 0.0),
                dip=rotation_dict.get("dip", 0.0),
                pitch=rotation_dict.get("pitch", 0.0),
            ),
        )

    def get_principal_directions(
        self,
        max_distance: float | None = None,
        n_points: int = 200,
    ) -> tuple["VariogramCurveData", "VariogramCurveData", "VariogramCurveData"]:
        """Generate variogram curve data for the three principal directions.

        Calculates the variogram model along the major, semi-major, and minor
        axis directions. Each direction uses the corresponding range from the
        anisotropy ellipsoid.

        Args:
            max_distance: Maximum distance for the curves. If None, uses 1.2x the maximum range.
            n_points: Number of points for smooth curves.

        Returns:
            Tuple of (major_curve, semi_major_curve, minor_curve) as VariogramCurveData.

        Example with Plotly:
            >>> major, semi_maj, minor = variogram.get_principal_directions()
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
        """
        max_ranges = {"major": 0.0, "semi_major": 0.0, "minor": 0.0}
        for struct in self.structures:
            anisotropy = struct.get("anisotropy", {})
            ranges = anisotropy.get("ellipsoid_ranges", {})
            for direction in max_ranges:
                max_ranges[direction] = max(max_ranges[direction], ranges.get(direction, 0))

        if max_distance is None:
            max_distance = max(max_ranges.values()) * 1.2 if max(max_ranges.values()) > 0 else 100.0

        h = np.linspace(0, max_distance, n_points)

        results = []
        for direction in ["major", "semi_major", "minor"]:
            gamma = np.full_like(h, self.nugget, dtype=float)

            for struct in self.structures:
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
                    sill=self.sill,
                )
            )

        return results[0], results[1], results[2]

    def get_direction(
        self,
        azimuth: float,
        dip: float,
        max_distance: float | None = None,
        n_points: int = 200,
    ) -> tuple["NDArray[np.floating[Any]]", "NDArray[np.floating[Any]]"]:
        """Calculate variogram model curve in an arbitrary direction.

        Computes the variogram semivariance along a specified direction defined by
        azimuth and dip angles. The effective range in that direction is calculated
        using the anisotropic transform.

        Args:
            azimuth: Azimuth angle in degrees (0-360), measured clockwise from north.
            dip: Dip angle in degrees (-90 to 90), positive downward.
            max_distance: Maximum distance for the curve. If None, uses 1.3x the
                         effective range in the specified direction.
            n_points: Number of points for smooth curve.

        Returns:
            Tuple of (distance, semivariance) as numpy arrays, suitable for plotting.

        Example with Plotly:
            >>> distance, semivariance = variogram.get_direction(azimuth=45, dip=30)
            >>> import plotly.graph_objects as go
            >>> fig = go.Figure()
            >>> fig.add_trace(go.Scatter(x=distance, y=semivariance, name='Az=45°, Dip=30°'))
            >>> fig.update_layout(xaxis_title='Distance', yaxis_title='Semivariance')
            >>> fig.show()

        Example with Matplotlib:
            >>> distance, semivariance = variogram.get_direction(azimuth=0, dip=0)
            >>> import matplotlib.pyplot as plt
            >>> plt.plot(distance, semivariance)
            >>> plt.xlabel('Distance')
            >>> plt.ylabel('Semivariance')
            >>> plt.show()
        """
        # Convert azimuth/dip to a unit direction vector
        # Azimuth is clockwise from north (Y-axis), dip is angle below horizontal
        az_rad = np.radians(azimuth)
        dip_rad = np.radians(dip)

        # Direction vector in world coordinates
        # X = east, Y = north, Z = up
        direction = np.array([
            np.sin(az_rad) * np.cos(dip_rad),  # X (east)
            np.cos(az_rad) * np.cos(dip_rad),  # Y (north)
            -np.sin(dip_rad),                   # Z (down is positive dip)
        ])

        # Calculate effective range in this direction for each structure
        # Using anisotropic transform: range = 1 / ||A^(-1) * direction||
        # where A is the diagonal scaling matrix with ranges
        total_range = 0.0
        for struct in self.structures:
            anisotropy = struct.get("anisotropy", {})
            ranges = anisotropy.get("ellipsoid_ranges", {})
            rotation_dict = anisotropy.get("rotation", {})

            major = ranges.get("major", 1.0)
            semi_major = ranges.get("semi_major", 1.0)
            minor = ranges.get("minor", 1.0)

            dip_azimuth = rotation_dict.get("dip_azimuth", 0.0)
            struct_dip = rotation_dict.get("dip", 0.0)
            pitch = rotation_dict.get("pitch", 0.0)

            # Build rotation matrix (same as ellipsoid)
            rot_matrix = self._build_rotation_matrix(dip_azimuth, struct_dip, pitch)

            # Transform direction to local (ellipsoid-aligned) coordinates
            local_dir = rot_matrix.T @ direction  # Inverse rotation

            # Apply anisotropic scaling (divide by ranges)
            if major > 0 and semi_major > 0 and minor > 0:
                scaled_dir = np.array([
                    local_dir[0] / major,
                    local_dir[1] / semi_major,
                    local_dir[2] / minor,
                ])
                # Effective range = 1 / ||scaled_direction||
                norm = np.linalg.norm(scaled_dir)
                if norm > 0:
                    total_range += 1.0 / norm

        # Use 1.3x the total effective range as default max distance
        if max_distance is None:
            max_distance = total_range * 1.3 if total_range > 0 else 100.0

        h = np.linspace(0, max_distance, n_points)
        gamma = np.full_like(h, self.nugget, dtype=float)

        # Evaluate variogram model for each structure
        for struct in self.structures:
            vtype = struct.get("variogram_type", "unknown")
            contribution = struct.get("contribution", 0)
            alpha = struct.get("alpha")
            anisotropy = struct.get("anisotropy", {})
            ranges = anisotropy.get("ellipsoid_ranges", {})
            rotation_dict = anisotropy.get("rotation", {})

            major = ranges.get("major", 1.0)
            semi_major = ranges.get("semi_major", 1.0)
            minor = ranges.get("minor", 1.0)

            dip_azimuth = rotation_dict.get("dip_azimuth", 0.0)
            struct_dip = rotation_dict.get("dip", 0.0)
            pitch = rotation_dict.get("pitch", 0.0)

            # Calculate effective range in this direction for this structure
            rot_matrix = self._build_rotation_matrix(dip_azimuth, struct_dip, pitch)
            local_dir = rot_matrix.T @ direction

            if major > 0 and semi_major > 0 and minor > 0:
                scaled_dir = np.array([
                    local_dir[0] / major,
                    local_dir[1] / semi_major,
                    local_dir[2] / minor,
                ])
                norm = np.linalg.norm(scaled_dir)
                effective_range = 1.0 / norm if norm > 0 else major
            else:
                effective_range = major

            gamma += _evaluate_structure(vtype, h, contribution, effective_range, alpha)

        return h, gamma

    @staticmethod
    def _build_rotation_matrix(
        dip_azimuth: float, dip: float, pitch: float
    ) -> "NDArray[np.floating[Any]]":
        """Build rotation matrix using Leapfrog convention for column vectors.

        Leapfrog uses row vector post-multiplication (vR), but we use column
        vectors (Rv), so we apply the transpose which reverses the order
        and uses positive angles.
        """
        az = np.radians(dip_azimuth)
        d = np.radians(dip)
        p = np.radians(pitch)

        # Rz(azimuth)
        rz_az = np.array([
            [np.cos(az), -np.sin(az), 0],
            [np.sin(az), np.cos(az), 0],
            [0, 0, 1],
        ])

        # Rx(dip)
        rx_dip = np.array([
            [1, 0, 0],
            [0, np.cos(d), -np.sin(d)],
            [0, np.sin(d), np.cos(d)],
        ])

        # Rz(pitch)
        rz_pitch = np.array([
            [np.cos(p), -np.sin(p), 0],
            [np.sin(p), np.cos(p), 0],
            [0, 0, 1],
        ])

        return rz_az @ rx_dip @ rz_pitch

    def _repr_html_(self) -> str:
        """Return an HTML representation for Jupyter notebooks."""
        doc = self.as_dict()

        # Get basic info
        name = doc.get("name", "Unnamed")

        # Build title links for viewer and portal
        title_links = [("Portal", self.portal_url), ("Viewer", self.viewer_url)]

        # Build basic rows
        rows = [
            ("Sill:", f"{self.sill:.4g}"),
            ("Nugget:", f"{self.nugget:.4g}"),
            ("Rotation Fixed:", str(self.is_rotation_fixed)),
        ]

        # Add optional fields
        if self.attribute:
            rows.append(("Attribute:", self.attribute))
        if self.domain:
            rows.append(("Domain:", self.domain))
        if self.modelling_space:
            rows.append(("Modelling Space:", self.modelling_space))
        if self.data_variance is not None:
            rows.append(("Data Variance:", f"{self.data_variance:.4g}"))

        # Build structures section
        structures_html = ""
        if self.structures:
            struct_rows = []
            for i, struct in enumerate(self.structures):
                vtype = struct.get("variogram_type", "unknown")
                contribution = struct.get("contribution", 0)

                # Calculate standardized sill (% of variance)
                standardized_sill = np.round(contribution / self.sill, 2) if self.sill != 0 else 0.0

                # Extract anisotropy info
                anisotropy = struct.get("anisotropy", {})
                ranges = anisotropy.get("ellipsoid_ranges", {})
                rotation = anisotropy.get("rotation", {})

                range_str = f"({ranges.get('major', 0):.1f}, {ranges.get('semi_major', 0):.1f}, {ranges.get('minor', 0):.1f})"
                # Rotation order: dip, dip_az, pitch
                rot_str = f"({rotation.get('dip', 0):.1f}°, {rotation.get('dip_azimuth', 0):.1f}°, {rotation.get('pitch', 0):.1f}°)"

                struct_rows.append([
                    f"{i+1}",
                    vtype,
                    f"{contribution:.4g}",
                    f"{standardized_sill:.2f}",
                    range_str,
                    rot_str,
                ])

            structures_table = build_nested_table(
                ["#", "Type", "Contribution", "Std. Sill", "Ranges (maj, semi, min)", "Rotation (dip, dip_az, pitch)"],
                struct_rows
            )
            structures_html = f'<div style="margin-top: 8px;"><strong>Structures ({len(self.structures)}):</strong></div>{structures_table}'

        # Build the main table
        table_rows = [build_table_row(label, value) for label, value in rows]
        main_table = f'<table>{"".join(table_rows)}</table>'

        return f'{STYLESHEET}<div class="evo">{build_title(name, title_links)}{main_table}{structures_html}</div>'

