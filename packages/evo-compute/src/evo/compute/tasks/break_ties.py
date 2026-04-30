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

Example:
    >>> from evo.compute.tasks import run, SearchNeighborhood, Ellipsoid, EllipsoidRanges
    >>> from evo.compute.tasks.break_ties import BreakTiesParameters
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

from typing import ClassVar, Protocol, runtime_checkable

import pandas as pd
from evo.common import IContext, IFeedback
from evo.objects import ObjectSchema
from evo.objects.typed import BaseObject, object_from_reference
from pydantic import BaseModel

from .common import (
    AnySourceAttribute,
    AnyTargetAttribute,
    SearchNeighborhood,
)
from .common.results import TaskTarget
from .common.runner import TaskRunner

__all__ = [
    "BreakTiesParameters",
    "BreakTiesResult",
    "BreakTiesResultModel",
    "BreakTiesRunner",
]


# =============================================================================
# Break Ties Parameters
# =============================================================================


class BreakTiesParameters(BaseModel):
    """Parameters for the break-ties task.

    Defines all inputs needed to run a break-ties spatial tie-breaking task.

    Example:
        >>> from evo.compute.tasks import run, SearchNeighborhood, Ellipsoid, EllipsoidRanges, Target
        >>> from evo.compute.tasks.break_ties import BreakTiesParameters
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


# =============================================================================
# Break Ties Result Types
# =============================================================================


@runtime_checkable
class _ObjToDataframeProtocol(Protocol):
    """Protocol for objects that can convert themselves to a DataFrame."""

    async def to_dataframe(self, *keys: str, fb: IFeedback = ...) -> pd.DataFrame: ...


class BreakTiesResultModel(BaseModel):
    """Pydantic model for the raw break-ties task result."""

    message: str
    """A message describing what happened in the task."""

    target: TaskTarget
    """Target information from the task result."""


class BreakTiesResult:
    TASK_DISPLAY_NAME: ClassVar[str] = "Break Ties"

    def __init__(self, context: IContext, model: BreakTiesResultModel) -> None:
        self._target = model.target
        self._message = model.message
        self._context = context

    @property
    def message(self) -> str:
        """A message describing what happened in the task."""
        return self._message

    @property
    def target_name(self) -> str:
        """The name of the target object."""
        return self._target.name

    @property
    def target_reference(self) -> str:
        """Reference URL to the target object."""
        return self._target.reference

    @property
    def attribute_name(self) -> str:
        """The name of the attribute that was created/updated."""
        return self._target.attribute.name

    @property
    def schema(self) -> ObjectSchema:
        """The schema type of the target object (e.g., 'pointset').

        Uses ``ObjectSchema.from_id`` to parse the schema ID.
        """
        return ObjectSchema.from_id(self._target.schema_id)

    async def get_target_object(self) -> BaseObject:
        """Load and return the target geoscience object.

        Returns:
            The typed geoscience object (e.g., PointSet, Regular3DGrid)

        Example:
            >>> result = await run(manager, params)
            >>> target = await result.get_target_object()
        """
        return await object_from_reference(self._context, self._target.reference)

    async def to_dataframe(self, *columns: str) -> pd.DataFrame:
        """Get the task results as a DataFrame.

        Args:
            columns: Optional column names to include. If omitted, returns all columns.

        Returns:
            A pandas DataFrame containing the tie-broken attribute values.

        Example:
            >>> result = await run(manager, params)
            >>> df = await result.to_dataframe()
            >>> df.head()
        """
        target_obj = await self.get_target_object()

        if isinstance(target_obj, _ObjToDataframeProtocol):
            return await target_obj.to_dataframe(*columns)
        else:
            raise TypeError(
                f"Don't know how to get DataFrame from {type(target_obj).__name__}. "
                "Use get_target_object() and access the data manually."
            )

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
