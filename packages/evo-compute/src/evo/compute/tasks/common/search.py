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

"""Search neighborhood parameters for geostatistical operations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any, Literal

from evo.objects.typed.types import Ellipsoid, EllipsoidRanges, Rotation
from pydantic import (
    AliasChoices,
    BaseModel,
    Field,
    PlainSerializer,
    PlainValidator,
    SerializerFunctionWrapHandler,
    WithJsonSchema,
    model_serializer,
    model_validator,
)

__all__ = [
    "SearchNeighborhood",
]

_SECTOR_AND_DRILLHOLE_LIMITS = (
    "max_empty_octants",
    "max_samples_per_octant",
    "max_empty_quadrants",
    "max_samples_per_quadrant",
    "max_samples_per_drillhole",
    "max_drillholes",
)

_ELLIPSOID_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "title": "Ellipsoid",
    "additionalProperties": False,
    "required": ["ellipsoid_ranges", "rotation"],
    "properties": {
        "ellipsoid_ranges": {
            "type": "object",
            "title": "Ellipsoid Ranges",
            "additionalProperties": False,
            "required": ["major", "semi_major", "minor"],
            "properties": {
                "major": {"type": "number", "exclusiveMinimum": 0, "title": "Major"},
                "semi_major": {"type": "number", "exclusiveMinimum": 0, "title": "Semi Major"},
                "minor": {"type": "number", "exclusiveMinimum": 0, "title": "Minor"},
            },
        },
        "rotation": {
            "type": "object",
            "title": "Rotation",
            "additionalProperties": False,
            "properties": {
                "dip_azimuth": {"type": "number", "minimum": 0, "maximum": 360, "default": 0, "title": "Dip Azimuth"},
                "dip": {"type": "number", "minimum": 0, "maximum": 180, "default": 0, "title": "Dip"},
                "pitch": {"type": "number", "minimum": 0, "maximum": 360, "default": 0, "title": "Pitch"},
            },
        },
    },
}


def _parse_ellipsoid_ranges(raw: Any) -> EllipsoidRanges:
    if isinstance(raw, EllipsoidRanges):
        return raw
    if isinstance(raw, Mapping):
        unknown = set(raw) - {"major", "semi_major", "minor"}
        if unknown:
            raise ValueError(f"Unexpected ellipsoid_ranges field(s): {', '.join(sorted(map(str, unknown)))}.")
        missing = {"major", "semi_major", "minor"} - set(raw)
        if missing:
            raise ValueError(f"Missing ellipsoid_ranges field(s): {', '.join(sorted(missing))}.")
        try:
            return EllipsoidRanges(major=raw["major"], semi_major=raw["semi_major"], minor=raw["minor"])
        except TypeError as exc:
            raise ValueError(str(exc)) from exc
    try:
        return EllipsoidRanges.from_array_like(raw)
    except TypeError as exc:
        raise ValueError(str(exc)) from exc


def _parse_rotation(raw: Any) -> Rotation | None:
    if raw is None or isinstance(raw, Rotation):
        return raw
    if isinstance(raw, Mapping):
        unknown = set(raw) - {"dip_azimuth", "dip", "pitch"}
        if unknown:
            raise ValueError(f"Unexpected rotation field(s): {', '.join(sorted(map(str, unknown)))}.")
        try:
            return Rotation(
                dip_azimuth=raw["dip_azimuth"] if "dip_azimuth" in raw else 0.0,
                dip=raw["dip"] if "dip" in raw else 0.0,
                pitch=raw["pitch"] if "pitch" in raw else 0.0,
            )
        except TypeError as exc:
            raise ValueError(str(exc)) from exc
    try:
        return Rotation.from_array_like(raw)
    except TypeError as exc:
        raise ValueError(str(exc)) from exc


def _validate_ellipsoid(value: Any) -> Ellipsoid:
    if isinstance(value, Ellipsoid):
        return value
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected an Ellipsoid or mapping, got {type(value).__name__}.")
    unknown = set(value) - {"ellipsoid_ranges", "ranges", "rotation"}
    if unknown:
        raise ValueError(f"Unexpected ellipsoid field(s): {', '.join(sorted(map(str, unknown)))}.")
    if "ellipsoid_ranges" not in value and "ranges" not in value:
        raise ValueError("Ellipsoid mapping requires 'ellipsoid_ranges' (or 'ranges').")
    if "ellipsoid_ranges" in value and "ranges" in value:
        wire_ranges = _parse_ellipsoid_ranges(value["ellipsoid_ranges"])
        attr_ranges = _parse_ellipsoid_ranges(value["ranges"])
        if wire_ranges.to_dict() != attr_ranges.to_dict():
            raise ValueError("Conflicting 'ellipsoid_ranges' and 'ranges' values on ellipsoid.")
        ranges = wire_ranges
    elif "ellipsoid_ranges" in value:
        ranges = _parse_ellipsoid_ranges(value["ellipsoid_ranges"])
    else:
        ranges = _parse_ellipsoid_ranges(value["ranges"])
    return Ellipsoid(ranges=ranges, rotation=_parse_rotation(value.get("rotation")))


def _serialize_ellipsoid(ellipsoid: Ellipsoid) -> dict[str, Any]:
    return ellipsoid.to_dict()


class SearchNeighborhood(BaseModel):
    """Search neighborhood parameters for geostatistical operations.

    Defines how to find nearby samples when performing spatial interpolation
    or estimation. Used by kriging, simulation, and other geostatistical tasks.

    The search neighborhood is defined by an ellipsoid (spatial extent and
    orientation) and constraints on the number of samples to use.

    Kriging, IDW, KNN, cross-validation and declustering can further limit the
    samples by octant, quadrant and drillhole (``max_samples_per_drillhole`` and
    ``max_drillholes``). Kriging additionally supports ``min_drillholes`` and
    non-global ``drillhole_limit_scope``. The other tasks do not support these
    limits and refuse a neighborhood that sets them.

    Example::

        >>> search = SearchNeighborhood(
        ...     ellipsoid=Ellipsoid(
        ...         ranges=EllipsoidRanges(major=200.0, semi_major=150.0, minor=100.0),
        ...         rotation=Rotation(dip_azimuth=45.0, dip=0.0, pitch=0.0),
        ...     ),
        ...     max_samples=20,
        ... )
        >>>
        >>> # Octant search, using at most 3 samples from each drillhole across at most 6 drillholes:
        >>> search = SearchNeighborhood(
        ...     ellipsoid=Ellipsoid(ranges=EllipsoidRanges(major=200.0, semi_major=150.0, minor=100.0)),
        ...     max_samples=24,
        ...     max_samples_per_octant=3,
        ...     max_empty_octants=4,
        ...     max_samples_per_drillhole=3,
        ...     max_drillholes=6,
        ... )
    """

    ellipsoid: Annotated[
        Ellipsoid,
        PlainValidator(_validate_ellipsoid),
        PlainSerializer(_serialize_ellipsoid),
        WithJsonSchema(_ELLIPSOID_JSON_SCHEMA),
    ]
    """The ellipsoid defining the spatial extent to search for samples."""

    max_samples: int = Field(ge=1)
    """The maximum number of samples to use for each evaluation point."""

    min_samples: int | None = Field(None, ge=1)
    """The minimum number of samples required, between 1 and ``max_samples``.

    If fewer are found, the point may be skipped.
    """

    max_empty_octants: int | None = Field(None, ge=0, le=8)
    """The maximum number of empty octants (sectors) allowed when searching for samples.

    Omit, or use 8, to disable the octant check.
    """

    max_samples_per_octant: int | None = Field(None, ge=1)
    """The maximum number of samples to use from each octant."""

    max_empty_quadrants: int | None = Field(None, ge=0, le=4)
    """The maximum number of empty quadrants (2D sectors, ignoring Z) allowed when searching for samples.

    Omit, or use 4, to disable the quadrant check.
    """

    max_samples_per_quadrant: int | None = Field(None, ge=1)
    """The maximum number of samples to use from each quadrant (2D sectors, ignoring Z)."""

    max_samples_per_drillhole: int | None = Field(None, ge=1)
    """The maximum number of samples to use from each drillhole. Requires a downhole intervals source object."""

    max_drillholes: int | None = Field(
        None,
        ge=1,
        validation_alias=AliasChoices("max_drillholes", "max_drillholes_per_estimate"),
    )
    """The maximum number of drillholes used in each estimate. Requires a downhole intervals source object."""

    min_drillholes: int | None = Field(
        None,
        ge=0,
        validation_alias=AliasChoices("min_drillholes", "min_drillholes_per_estimate"),
    )
    """The minimum number of distinct drillholes required per estimate.

    Omitted (disabled) when ``None`` or ``0``. Currently supported only by kriging
    and requires a downhole intervals source object.
    """

    drillhole_limit_scope: Literal["global", "quadrant", "octant"] | None = None
    """Scope at which ``max_samples_per_drillhole`` is applied.

    ``"global"`` (or ``None``) applies ``max_samples_per_drillhole`` across the whole
    neighborhood and is omitted on the wire. ``"quadrant"`` and ``"octant"`` apply
    ``max_samples_per_drillhole`` within each sector, require ``max_samples_per_drillhole``
    to be set, and are currently supported only by kriging.
    """

    @property
    def max_drillholes_per_estimate(self) -> int | None:
        """Backwards-compatible alias for :attr:`max_drillholes`."""
        return self.max_drillholes

    @model_validator(mode="before")
    @classmethod
    def _reject_conflicting_aliases(cls, data: Any) -> Any:
        if isinstance(data, Mapping):
            for canonical, legacy in (
                ("max_drillholes", "max_drillholes_per_estimate"),
                ("min_drillholes", "min_drillholes_per_estimate"),
            ):
                if canonical in data and legacy in data and data[canonical] != data[legacy]:
                    raise ValueError(
                        f"Conflicting values for {canonical!r} ({data[canonical]!r}) and {legacy!r} ({data[legacy]!r})."
                    )
        return data

    @model_validator(mode="after")
    def _check_sample_limits(self) -> SearchNeighborhood:
        if self.drillhole_limit_scope in ("quadrant", "octant") and self.max_samples_per_drillhole is None:
            raise ValueError(
                f"drillhole_limit_scope={self.drillhole_limit_scope!r} requires max_samples_per_drillhole to be set."
            )

        if self.min_drillholes is not None and self.max_drillholes is not None:
            if self.min_drillholes > self.max_drillholes:
                raise ValueError(
                    f"min_drillholes ({self.min_drillholes}) cannot exceed max_drillholes ({self.max_drillholes})."
                )

        scope_sectors = {"quadrant": 4, "octant": 8}.get(self.drillhole_limit_scope or "global", 1)
        sample_ceilings: list[tuple[int, str]] = [(self.max_samples, f"max_samples ({self.max_samples})")]
        if self.max_samples_per_octant is not None:
            octant_ceiling = 8 * self.max_samples_per_octant
            sample_ceilings.append((octant_ceiling, f"8 * max_samples_per_octant ({octant_ceiling})"))
        if self.max_samples_per_quadrant is not None:
            quadrant_ceiling = 4 * self.max_samples_per_quadrant
            sample_ceilings.append((quadrant_ceiling, f"4 * max_samples_per_quadrant ({quadrant_ceiling})"))
        if self.max_drillholes is not None and self.max_samples_per_drillhole is not None:
            drillhole_ceiling = self.max_drillholes * self.max_samples_per_drillhole * scope_sectors
            if scope_sectors == 1:
                label = f"max_drillholes * max_samples_per_drillhole ({drillhole_ceiling})"
            else:
                label = (
                    f"max_drillholes * max_samples_per_drillhole * {scope_sectors} "
                    f"({self.drillhole_limit_scope} sectors = {drillhole_ceiling})"
                )
            sample_ceilings.append((drillhole_ceiling, label))

        if self.min_samples is not None:
            for ceiling, label in sample_ceilings:
                if self.min_samples > ceiling:
                    raise ValueError(f"min_samples ({self.min_samples}) cannot exceed {label}.")

        if self.min_drillholes is not None and self.min_drillholes > 0:
            for ceiling, label in sample_ceilings:
                if self.min_drillholes > ceiling:
                    raise ValueError(f"min_drillholes ({self.min_drillholes}) cannot exceed {label}.")

        if self.max_empty_octants is not None:
            required_octants = 8 - self.max_empty_octants
            for ceiling, label in sample_ceilings:
                if required_octants > ceiling:
                    raise ValueError(
                        f"max_empty_octants={self.max_empty_octants} requires at least {required_octants} "
                        f"occupied octants, which exceeds {label}."
                    )

        if self.max_empty_quadrants is not None:
            required_quadrants = 4 - self.max_empty_quadrants
            for ceiling, label in sample_ceilings:
                if required_quadrants > ceiling:
                    raise ValueError(
                        f"max_empty_quadrants={self.max_empty_quadrants} requires at least {required_quadrants} "
                        f"occupied quadrants, which exceeds {label}."
                    )

        return self

    @model_serializer(mode="wrap")
    def _serialize(self, handler: SerializerFunctionWrapHandler):
        return {
            key: value
            for key, value in handler(self).items()
            if value is not None
            and not (key == "min_drillholes" and value == 0)
            and not (key == "drillhole_limit_scope" and value == "global")
        }


def _active_kriging_only_drillhole_limits(neighborhood: SearchNeighborhood) -> list[str]:
    limits: list[str] = []
    if neighborhood.min_drillholes is not None and neighborhood.min_drillholes > 0:
        limits.append("min_drillholes")
    if neighborhood.drillhole_limit_scope not in (None, "global"):
        limits.append("drillhole_limit_scope")
    return limits


def _reject_sector_and_drillhole_limits(neighborhood: SearchNeighborhood) -> SearchNeighborhood:
    """Refuse octant, quadrant and drillhole limits, for tasks whose service does not accept them."""
    limits = [name for name in _SECTOR_AND_DRILLHOLE_LIMITS if getattr(neighborhood, name) is not None]
    limits.extend(_active_kriging_only_drillhole_limits(neighborhood))
    if limits:
        raise ValueError(
            f"This task does not support {', '.join(limits)}. "
            "Octant, quadrant and drillhole limits are only available for kriging, IDW, KNN, cross-validation "
            "and declustering."
        )
    return neighborhood


def _reject_kriging_only_drillhole_limits(neighborhood: SearchNeighborhood) -> SearchNeighborhood:
    """Refuse drillhole limits that are only supported by the kriging task."""
    if limits := _active_kriging_only_drillhole_limits(neighborhood):
        raise ValueError(
            f"This task does not support {', '.join(limits)}. "
            "min_drillholes and non-global drillhole_limit_scope are only supported by Kriging."
        )
    return neighborhood
