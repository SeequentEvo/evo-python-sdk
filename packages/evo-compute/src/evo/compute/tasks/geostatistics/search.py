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

from typing import Any, Literal, Self

from evo.objects.typed.types import Ellipsoid
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "SearchNeighborhood",
    "SimpleSearchNeighborhood",
]

_EXTENDED_OPTIONAL_LIMIT_FIELDS: tuple[str, ...] = (
    "max_empty_octants",
    "max_samples_per_octant",
    "max_empty_quadrants",
    "max_samples_per_quadrant",
    "max_samples_per_drillhole",
    "max_drillholes",
)


class _SearchNeighborhoodBase(BaseModel):
    """Shared spatial ellipsoid and sample count bounds for geostatistics neighborhoods."""

    model_config = ConfigDict(extra="forbid")

    ellipsoid: Ellipsoid
    """The ellipsoid defining the spatial extent to search for samples."""

    max_samples: int = Field(ge=1)
    """The maximum number of samples to use for each evaluation point."""

    min_samples: int | None = Field(None, ge=1)
    """The minimum number of samples required, between 1 and `max_samples`.

    If fewer are found, the point may be skipped.
    """

    @model_validator(mode="after")
    def _check_sample_limits(self) -> Self:
        if self.min_samples is not None and self.min_samples > self.max_samples:
            raise ValueError(f"min_samples ({self.min_samples}) cannot exceed max_samples ({self.max_samples}).")
        return self


class SimpleSearchNeighborhood(_SearchNeighborhoodBase):
    """Basic search neighborhood with spatial ellipsoid and sample count bounds.

    Used by geostatistics tasks whose search accepts only an ellipsoid and
    overall `min_samples` / `max_samples` bounds (such as simulation and
    spatial tie-breaking tasks):

    - `ellipsoid`: `Ellipsoid` defining the spatial search extent and orientation.
    - `max_samples`: Maximum number of samples (`>= 1`) to use per evaluation point.
    - `min_samples`: Optional minimum number of samples (`1 <= min_samples <= max_samples`).

    ```python
    neighborhood = SimpleSearchNeighborhood(
        ellipsoid=Ellipsoid(
            ranges=EllipsoidRanges(major=200.0, semi_major=150.0, minor=100.0),
            rotation=Rotation(dip_azimuth=45.0, dip=0.0, pitch=0.0),
        ),
        max_samples=20,
    )
    ```
    """

    @model_validator(mode="before")
    @classmethod
    def _coerce_compatible_search_neighborhood(cls, data: Any) -> Any:
        if isinstance(data, SearchNeighborhood):
            active_limits = [name for name in _EXTENDED_OPTIONAL_LIMIT_FIELDS if getattr(data, name) is not None]
            if data.min_drillholes not in (None, 0):
                active_limits.append("min_drillholes")
            if data.drillhole_limit_scope not in (None, "global"):
                active_limits.append("drillhole_limit_scope")
            if active_limits:
                raise ValueError(
                    f"This task's search neighborhood does not support {', '.join(active_limits)}; "
                    "use SimpleSearchNeighborhood with only ellipsoid, max_samples, and min_samples."
                )
            return {
                "ellipsoid": data.ellipsoid,
                "max_samples": data.max_samples,
                "min_samples": data.min_samples,
            }
        return data


class SearchNeighborhood(_SearchNeighborhoodBase):
    """Extended search neighborhood adding octant, quadrant, and drillhole controls to `ellipsoid`, `max_samples`, and `min_samples`.

    Used by estimation and declustering tasks (Kriging, IDW, KNN, and
    Declustering) that support sector (octant / quadrant) and drillhole quotas
    in addition to the base `ellipsoid`, `max_samples`, and `min_samples` fields.

    ```python
    search = SearchNeighborhood(
        ellipsoid=Ellipsoid(
            ranges=EllipsoidRanges(major=200.0, semi_major=150.0, minor=100.0),
            rotation=Rotation(dip_azimuth=45.0, dip=0.0, pitch=0.0),
        ),
        max_samples=20,
    )

    # Octant search, using at most 3 samples from each drillhole across at most 6 drillholes:
    search = SearchNeighborhood(
        ellipsoid=Ellipsoid(ranges=EllipsoidRanges(major=200.0, semi_major=150.0, minor=100.0)),
        max_samples=24,
        max_samples_per_octant=3,
        max_empty_octants=4,
        max_samples_per_drillhole=3,
        max_drillholes=6,
    )
    ```
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

    `None` or `0` disables the check. Requires a downhole intervals source object.
    """

    drillhole_limit_scope: Literal["global", "quadrant", "octant"] | None = None
    """Scope at which `max_samples_per_drillhole` is applied.

    `"global"` (the task default when `None`) applies `max_samples_per_drillhole` across the whole
    neighborhood. `"quadrant"` and `"octant"` apply `max_samples_per_drillhole` within each sector
    and require `max_samples_per_drillhole` to be set.
    """

    @property
    def max_drillholes_per_estimate(self) -> int | None:
        """Backwards-compatible alias for `max_drillholes`."""
        return self.max_drillholes

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
