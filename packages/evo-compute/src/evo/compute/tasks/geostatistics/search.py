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

"""Search neighborhood models for geostatistics compute tasks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, Self

from evo.objects.typed.types import Ellipsoid
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    ValidatorFunctionWrapHandler,
    model_serializer,
    model_validator,
)

__all__ = [
    "SearchNeighborhood",
    "SimpleSearchNeighborhood",
]


class SimpleSearchNeighborhood(BaseModel):
    """Basic search neighborhood with spatial ellipsoid and sample count bounds.

    Used by geostatistics tasks whose search accepts only an ellipsoid and
    overall ``min_samples`` / ``max_samples`` bounds (such as simulation and
    spatial tie-breaking tasks).

    Example::

        >>> neighborhood = SimpleSearchNeighborhood(
        ...     ellipsoid=Ellipsoid(
        ...         ranges=EllipsoidRanges(major=200.0, semi_major=150.0, minor=100.0),
        ...         rotation=Rotation(dip_azimuth=45.0, dip=0.0, pitch=0.0),
        ...     ),
        ...     max_samples=20,
        ... )
    """

    model_config = ConfigDict(extra="forbid")

    ellipsoid: Ellipsoid
    """The ellipsoid defining the spatial extent to search for samples."""

    max_samples: int = Field(ge=1)
    """The maximum number of samples to use for each evaluation point."""

    min_samples: int | None = Field(None, ge=1)
    """The minimum number of samples required, between 1 and ``max_samples``.

    If fewer are found, the point may be skipped.
    """

    @model_validator(mode="wrap")
    @classmethod
    def _enforce_exact_tier(cls, data: Any, handler: ValidatorFunctionWrapHandler) -> Self:
        if isinstance(data, SimpleSearchNeighborhood) and type(data) is not cls:
            return handler(data.model_dump())
        return handler(data)

    @model_validator(mode="after")
    def _check_sample_limits(self) -> Self:
        if self.min_samples is not None and self.min_samples > self.max_samples:
            raise ValueError(f"min_samples ({self.min_samples}) cannot exceed max_samples ({self.max_samples}).")
        return self

    @model_serializer(mode="wrap")
    def _serialize(self, handler: SerializerFunctionWrapHandler):
        return {key: value for key, value in handler(self).items() if value is not None}


class SearchNeighborhood(SimpleSearchNeighborhood):
    """Extended search neighborhood with octant, quadrant, and drillhole controls.

    Extends :class:`SimpleSearchNeighborhood` with sector (octant / quadrant)
    and drillhole quotas for estimation and declustering tasks (Kriging, IDW,
    KNN, and Declustering).

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

    ellipsoid: Ellipsoid
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

    Omitted (disabled) when ``None`` or ``0``. Requires a downhole intervals source object.
    """

    drillhole_limit_scope: Literal["global", "quadrant", "octant"] | None = None
    """Scope at which ``max_samples_per_drillhole`` is applied.

    ``"global"`` (or ``None``) applies ``max_samples_per_drillhole`` across the whole
    neighborhood and is omitted on the wire. ``"quadrant"`` and ``"octant"`` apply
    ``max_samples_per_drillhole`` within each sector and require ``max_samples_per_drillhole``
    to be set.
    """

    @property
    def max_drillholes_per_estimate(self) -> int | None:
        """Backwards-compatible alias for :attr:`max_drillholes`."""
        return self.max_drillholes

    @model_validator(mode="before")
    @classmethod
    def _normalize_legacy_aliases(cls, data: Any) -> Any:
        if isinstance(data, Mapping):
            normalized: dict[str, Any] | None = None
            for canonical, legacy in (
                ("max_drillholes", "max_drillholes_per_estimate"),
                ("min_drillholes", "min_drillholes_per_estimate"),
            ):
                if canonical in data and legacy in data:
                    if data[canonical] != data[legacy]:
                        raise ValueError(
                            f"Conflicting values for {canonical!r} ({data[canonical]!r}) "
                            f"and {legacy!r} ({data[legacy]!r})."
                        )
                    if normalized is None:
                        normalized = dict(data)
                    normalized.pop(legacy, None)
            if normalized is not None:
                return normalized
        return data

    @model_validator(mode="after")
    def _check_extended_limits(self) -> Self:
        if self.drillhole_limit_scope in ("quadrant", "octant") and self.max_samples_per_drillhole is None:
            raise ValueError(
                f"drillhole_limit_scope={self.drillhole_limit_scope!r} requires max_samples_per_drillhole to be set."
            )

        if self.min_drillholes is not None and self.max_drillholes is not None:
            if self.min_drillholes > self.max_drillholes:
                raise ValueError(
                    f"min_drillholes ({self.min_drillholes}) cannot exceed max_drillholes ({self.max_drillholes})."
                )

        match self.drillhole_limit_scope:
            case "quadrant":
                scope_sectors = 4
            case "octant":
                scope_sectors = 8
            case _:
                scope_sectors = 1

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
