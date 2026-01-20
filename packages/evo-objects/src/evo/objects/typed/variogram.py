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

from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import TypeAdapter

from evo.objects import SchemaVersion

from ._property import SchemaProperty
from .base import BaseObjectData, ConstructableObject

__all__ = [
    "Anisotropy",
    "EllipsoidRanges",
    "SphericalStructure",
    "ExponentialStructure",
    "GaussianStructure",
    "CubicStructure",
    "Variogram",
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


@dataclass(kw_only=True, frozen=True)
class VariogramData(BaseObjectData):
    """Data for creating a Variogram.

    A variogram is a geostatistical model describing spatial correlation structure.
    The variogram model is defined by the nugget and multiple structures using the
    leapfrog-convention rotation.

    Example using typed structures (recommended):
        >>> data = VariogramData(
        ...     name="My Variogram",
        ...     sill=1.0,
        ...     nugget=0.1,
        ...     is_rotation_fixed=True,
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
    creation_schema_version = SchemaVersion(major=1, minor=2, patch=0)

    @classmethod
    async def _data_to_dict(cls, data: VariogramData, context: Any) -> dict[str, Any]:
        """Convert VariogramData to a dictionary for creating the Geoscience Object.

        Overrides the base implementation to handle typed structure conversion.
        """
        # Get base dict from parent class
        result = await super()._data_to_dict(data, context)

        # Override structures with converted version (handles typed classes)
        result["structures"] = data.get_structures_as_dicts()

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

    def _repr_html_(self) -> str:
        """Return an HTML representation for Jupyter notebooks."""
        from .._html_styles import STYLESHEET, build_nested_table, build_table_row, build_title

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

                # Extract anisotropy info
                anisotropy = struct.get("anisotropy", {})
                ranges = anisotropy.get("ellipsoid_ranges", {})
                rotation = anisotropy.get("rotation", {})

                range_str = f"({ranges.get('major', 0):.1f}, {ranges.get('semi_major', 0):.1f}, {ranges.get('minor', 0):.1f})"
                rot_str = f"({rotation.get('dip_azimuth', 0):.1f}°, {rotation.get('dip', 0):.1f}°, {rotation.get('pitch', 0):.1f}°)"

                struct_rows.append([
                    f"{i+1}",
                    vtype,
                    f"{contribution:.4g}",
                    range_str,
                    rot_str,
                ])

            structures_table = build_nested_table(
                ["#", "Type", "Contribution", "Ranges (maj, semi, min)", "Rotation (az, dip, pitch)"],
                struct_rows
            )
            structures_html = f'<div style="margin-top: 8px;"><strong>Structures ({len(self.structures)}):</strong></div>{structures_table}'

        # Build the main table
        table_rows = [build_table_row(label, value) for label, value in rows]
        main_table = f'<table>{"".join(table_rows)}</table>'

        return f'{STYLESHEET}<div class="evo-object">{build_title(name, title_links)}{main_table}{structures_html}</div>'

