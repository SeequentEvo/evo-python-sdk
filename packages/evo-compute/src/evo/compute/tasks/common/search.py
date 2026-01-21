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

"""Search neighbourhood parameters for geostatistical operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .ellipsoid import Ellipsoid

__all__ = [
    "SearchNeighbourhood",
]


@dataclass
class SearchNeighbourhood:
    """Search neighbourhood parameters for geostatistical operations.

    Defines how to find nearby samples when performing spatial interpolation
    or estimation. Used by kriging, simulation, and other geostatistical tasks.

    The search neighbourhood is defined by an ellipsoid (spatial extent and
    orientation) and constraints on the number of samples to use.

    Example:
        >>> search = SearchNeighbourhood(
        ...     ellipsoid=Ellipsoid(
        ...         ranges=EllipsoidRanges(major=200.0, semi_major=150.0, minor=100.0),
        ...         rotation=Rotation(dip_azimuth=45.0),
        ...     ),
        ...     max_samples=20,
        ... )
    """

    ellipsoid: Ellipsoid
    """The ellipsoid defining the spatial extent to search for samples."""

    max_samples: int
    """The maximum number of samples to use for each evaluation point."""

    min_samples: int | None = None
    """The minimum number of samples required. If fewer are found, the point may be skipped."""

    def __init__(
        self,
        ellipsoid: Ellipsoid,
        max_samples: int,
        min_samples: int | None = None,
    ):
        self.ellipsoid = ellipsoid
        self.max_samples = max_samples
        self.min_samples = min_samples

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        result = {
            "ellipsoid": self.ellipsoid.to_dict(),
            "max_samples": self.max_samples,
        }
        if self.min_samples is not None:
            result["min_samples"] = self.min_samples
        return result

