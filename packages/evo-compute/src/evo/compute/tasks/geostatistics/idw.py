#  Copyright © 2026 Bentley Systems, Incorporated
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Inverse distance weighting (IDW) estimation compute task client.

The IDW task estimates values at target locations using a weighted average of
accepted neighbours, where each weight is inversely proportional to the distance
raised to ``power``.  A higher power concentrates influence on the closest
samples; ``power=2`` is the classic IDW setting.

Example::

    >>> from evo.compute.tasks import run, SearchNeighborhood, Ellipsoid, EllipsoidRanges
    >>> from evo.compute.tasks.common import Source, Target
    >>> from evo.compute.tasks.geostatistics.idw import IDWParameters
    >>>
    >>> params = IDWParameters(
    ...     source=Source(object=pointset, attribute="locations.attributes[?name=='grade']"),
    ...     target=Target.new_attribute(grid, "idw_grade"),
    ...     neighborhood=SearchNeighborhood(
    ...         ellipsoid=Ellipsoid(ranges=EllipsoidRanges(200, 150, 100)),
    ...         max_samples=20,
    ...     ),
    ...     power=2.0,
    ... )
    >>> result = await run(manager, params)
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar

from evo.common import IContext
from evo.objects.typed import BaseSpatialObject
from pydantic import BaseModel, Field, field_validator

from ..common import (
    AnySourceAttribute,
    AnyTargetAttribute,
    Filter,
    SearchNeighborhood,
    SingleTargetTaskResult,
    SourceTargetTaskParameters,
)
from ..common.results import TaskTarget
from ..common.runner import TaskRunner
from ..common.search import _reject_kriging_only_drillhole_limits

__all__ = [
    "IDWParameters",
    "IDWResult",
    "IDWResultModel",
    "IDWRunner",
]


# =============================================================================
# Parameters
# =============================================================================


class IDWParameters(SourceTargetTaskParameters):
    """Parameters for the IDW (inverse distance weighting) estimation task.

    Estimates values at target locations as a distance-weighted average of
    nearby samples.  The weighting exponent is controlled by ``power``.

    Example::

        >>> from evo.compute.tasks.common import Filter, FilterCondition, Source, Target
        >>> from evo.compute.tasks.geostatistics.idw import IDWParameters
        >>>
        >>> params = IDWParameters(
        ...     source=Source(object=pointset, attribute="locations.attributes[?name=='grade']"),
        ...     target=Target.new_attribute(grid, "idw_grade"),
        ...     neighborhood=SearchNeighborhood(
        ...         ellipsoid=Ellipsoid(ranges=EllipsoidRanges(200, 150, 100)),
        ...         max_samples=20,
        ...     ),
        ...     power=2.0,
        ... )
        >>>
        >>> # With a target filter to restrict estimation to specific categories on the target:
        >>> params_filtered = IDWParameters(
        ...     source=Source(object=pointset, attribute="locations.attributes[?name=='grade']"),
        ...     target=Target.new_attribute(grid, "idw_grade"),
        ...     neighborhood=SearchNeighborhood(...),
        ...     power=2.0,
        ...     target_filter=Filter(
        ...         where=FilterCondition(
        ...             attribute=grid.attributes["domain"],
        ...             operator="in",
        ...             values=["LMS1", "LMS2"],
        ...         ),
        ...     ),
        ... )
    """

    _object_roles: ClassVar[Mapping[str, tuple[type, ...]]] = {
        "source": (BaseSpatialObject,),
        "target": (BaseSpatialObject,),
    }
    _drillhole_source_field: ClassVar[str | None] = "source"

    source: AnySourceAttribute
    """Source object and attribute containing the known values."""

    target: AnyTargetAttribute
    """Target object and attribute to create or update with IDW estimates."""

    neighborhood: SearchNeighborhood
    """Search neighbourhood defining which nearby samples to include."""

    power: float = Field(gt=0.0)
    """Distance-weighting exponent.

    Weights are computed as ``1 / distance ** power`` in the anisotropic space
    defined by the search ellipsoid.  Common values: ``1.0`` (linear inverse
    distance), ``2.0`` (classic IDW).  Must be strictly positive.
    """

    source_filter: Filter | None = Field(None, exclude=True)
    """Optional filter to restrict estimation to a subset of the source data.

    Excluded from top-level serialisation (``exclude=True``) and injected as
    ``source["filter"]`` by the custom serialiser.
    """

    target_filter: Filter | None = Field(None, exclude=True)
    """Optional filter to restrict estimation to a subset of the target object.

    Excluded from top-level serialisation (``exclude=True``) and injected as
    ``target["filter"]`` by the custom serialiser.
    """

    @field_validator("neighborhood")
    @classmethod
    def _check_neighborhood(cls, neighborhood: SearchNeighborhood) -> SearchNeighborhood:
        return _reject_kriging_only_drillhole_limits(neighborhood)


# =============================================================================
# Result Types
# =============================================================================


class IDWResultModel(BaseModel):
    """Pydantic model for the raw IDW task result payload."""

    message: str
    """A message describing what happened in the task."""

    target: TaskTarget
    """Target information from the task result."""


class IDWResult(SingleTargetTaskResult[TaskTarget]):
    """Rich result object for the IDW estimation task."""

    TASK_DISPLAY_NAME: ClassVar[str] = "IDW Estimation"

    def __init__(self, context: IContext, model: IDWResultModel) -> None:
        super().__init__(context, model)

    def __str__(self) -> str:
        return f"✓ {self.TASK_DISPLAY_NAME} Result\n  Target:    {self.target_name}\n  Attribute: {self.attribute_name}"


# =============================================================================
# Task Runner
# =============================================================================


class IDWRunner(
    TaskRunner[IDWParameters, IDWResultModel, IDWResult],
    topic="geostatistics",
    task="idw",
):
    async def _get_result(self, raw_result: IDWResultModel) -> IDWResult:
        return IDWResult(self._context, raw_result)
