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

"""
Break Ties compute task client.

This module provides typed dataclass models and convenience functions for running
the Break Ties task (geostatistics/break-ties).

Example::

    >>> from evo.compute.tasks import run, SearchNeighborhood, Ellipsoid, EllipsoidRanges
    >>> from evo.compute.tasks.geostatistics.break_ties import BreakTiesParameters
    >>>
    >>> params = BreakTiesParameters(
    ...     source=pointset.attributes["grade"],
    ...     target=Target.new_attribute(pointset, "grade_tiebroken"),
    ...     neighborhood=SearchNeighborhood(
    ...         ellipsoid=Ellipsoid(ranges=EllipsoidRanges(200, 150, 100)),
    ...         max_samples=20,
    ...     ),
    ... )
    >>> result = await run(manager, params)
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar

from evo.common import IContext
from evo.objects.typed import BaseSpatialObject
from pydantic import BaseModel, field_validator

from ..common import (
    AnySourceAttribute,
    AnyTargetAttribute,
    SearchNeighborhood,
    SingleTargetTaskResult,
    TaskParameters,
)
from ..common.results import TaskTarget
from ..common.runner import TaskRunner
from ..common.search import _reject_sector_and_drillhole_limits

__all__ = [
    "BreakTiesParameters",
    "BreakTiesResult",
    "BreakTiesResultModel",
    "BreakTiesRunner",
]


# =============================================================================
# Break Ties Parameters
# =============================================================================


class BreakTiesParameters(TaskParameters):
    """Parameters for the break-ties task.

    Defines all inputs needed to run a break-ties spatial tie-breaking task.

    Example::

        >>> from evo.compute.tasks import run, SearchNeighborhood, Ellipsoid, EllipsoidRanges, Target
        >>> from evo.compute.tasks.geostatistics.break_ties import BreakTiesParameters
        >>>
        >>> params = BreakTiesParameters(
        ...     source=pointset.attributes["grade"],
        ...     target=Target.new_attribute(pointset, "grade_tiebroken"),
        ...     neighborhood=SearchNeighborhood(
        ...         ellipsoid=Ellipsoid(ranges=EllipsoidRanges(200, 150, 100)),
        ...         max_samples=20,
        ...     ),
        ... )
        >>> result = await run(manager, params)
    """

    _object_roles: ClassVar[Mapping[str, tuple[type, ...]]] = {
        "source": (BaseSpatialObject,),
        "target": (BaseSpatialObject,),
    }

    source: AnySourceAttribute
    """The source object and attribute containing values in which ties will be broken."""

    target: AnyTargetAttribute
    """The target object and attribute to create or update with tie-broken results."""

    neighborhood: SearchNeighborhood
    """Search neighborhood parameters.

    In this break-ties algorithm, the final value of each tied point is influenced
    by the mean of nearby points. The search parameters determine which nearby points to use.
    """

    seed: int = 38239342
    """Seed for the random number generator."""

    @field_validator("neighborhood")
    @classmethod
    def _check_neighborhood(cls, neighborhood: SearchNeighborhood) -> SearchNeighborhood:
        return _reject_sector_and_drillhole_limits(neighborhood)


# =============================================================================
# Break Ties Result Types
# =============================================================================


class BreakTiesResultModel(BaseModel):
    """Pydantic model for the raw break-ties task result."""

    message: str
    """A message describing what happened in the task."""

    target: TaskTarget
    """Target information from the task result."""


class BreakTiesResult(SingleTargetTaskResult[TaskTarget]):
    TASK_DISPLAY_NAME: ClassVar[str] = "Break Ties"

    def __init__(self, context: IContext, model: BreakTiesResultModel) -> None:
        super().__init__(context, model)

    def __str__(self) -> str:
        """String representation."""
        lines = [
            f"✓ {self.TASK_DISPLAY_NAME} Result",
            f"  Message:   {self.message}",
            f"  Target:    {self.target_name}",
            f"  Attribute: {self.attribute_name}",
        ]
        return "\n".join(lines)


# =============================================================================
# Task Runner
# =============================================================================


class BreakTiesRunner(
    TaskRunner[BreakTiesParameters, BreakTiesResultModel, BreakTiesResult],
    topic="geostatistics",
    task="break-ties",
):
    """Runner for break-ties compute tasks.

    Automatically registered — used by ``run()`` for dispatch, or directly::

        result = await BreakTiesRunner(context, params)
    """

    async def _get_result(self, raw_result: BreakTiesResultModel) -> BreakTiesResult:
        return BreakTiesResult(self._context, raw_result)
