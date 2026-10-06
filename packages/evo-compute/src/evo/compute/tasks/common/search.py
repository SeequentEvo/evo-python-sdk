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

from typing import Any

from evo.objects.typed.types import Ellipsoid
from pydantic import BaseModel, Field, model_serializer

__all__ = [
    "SearchNeighborhood",
]

_SECTOR_AND_DRILLHOLE_LIMITS = (
    "max_empty_octants",
    "max_samples_per_octant",
    "max_empty_quadrants",
    "max_samples_per_quadrant",
    "max_samples_per_drillhole",
    "max_drillholes_per_estimate",
)


class SearchNeighborhood(BaseModel):
    """Search neighborhood parameters for geostatistical operations.

    Defines how to find nearby samples when performing spatial interpolation
    or estimation. Used by kriging, simulation, and other geostatistical tasks.

    The search neighborhood is defined by an ellipsoid (spatial extent and
    orientation) and constraints on the number of samples to use.

    Kriging, IDW, KNN and declustering can further limit the samples by octant,
    quadrant and drillhole. The other tasks do not support these limits and
    refuse a neighborhood that sets them.

    Example::

        >>> search = SearchNeighborhood(
        ...     ellipsoid=Ellipsoid(
        ...         ranges=EllipsoidRanges(major=200.0, semi_major=150.0, minor=100.0),
        ...         rotation=Rotation(dip_azimuth=45.0),
        ...     ),
        ...     max_samples=20,
        ... )
        >>>
        >>> # Octant search, using at most 3 samples from each drillhole:
        >>> search = SearchNeighborhood(
        ...     ellipsoid=Ellipsoid(ranges=EllipsoidRanges(major=200.0, semi_major=150.0, minor=100.0)),
        ...     max_samples=24,
        ...     max_samples_per_octant=3,
        ...     max_empty_octants=4,
        ...     max_samples_per_drillhole=3,
        ... )
    """

    model_config = {"arbitrary_types_allowed": True}

    ellipsoid: Ellipsoid
    """The ellipsoid defining the spatial extent to search for samples."""

    max_samples: int
    """The maximum number of samples to use for each evaluation point."""

    min_samples: int | None = None
    """The minimum number of samples required. If fewer are found, the point may be skipped."""

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

    max_drillholes_per_estimate: int | None = Field(None, ge=1)
    """The maximum number of drillholes used in each estimate. Requires a downhole intervals source object."""

    @model_serializer
    def _serialize(self) -> dict[str, Any]:
        result = {
            "ellipsoid": self.ellipsoid.to_dict(),
            "max_samples": self.max_samples,
        }
        for name in ("min_samples", *_SECTOR_AND_DRILLHOLE_LIMITS):
            if (value := getattr(self, name)) is not None:
                result[name] = value
        return result


def _reject_sector_and_drillhole_limits(neighborhood: SearchNeighborhood) -> SearchNeighborhood:
    """Refuse octant, quadrant and drillhole limits, for tasks whose service does not accept them."""
    if limits := [name for name in _SECTOR_AND_DRILLHOLE_LIMITS if getattr(neighborhood, name) is not None]:
        raise ValueError(
            f"This task does not support {', '.join(limits)}. "
            "Octant, quadrant and drillhole limits are only available for kriging, IDW, KNN and declustering."
        )
    return neighborhood
