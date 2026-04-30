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
Normal-score transform compute task client.

This module provides typed dataclass models and convenience functions for running
the Normal Score task (geostatistics/normal-score).

Two transform directions are available:

- **Forward**: transforms data values to standard Gaussian (normal-score) space.
- **Backward**: transforms normal-score values back to the original data space.

Both transforms require a pre-existing continuous distribution object.
The ``method`` field selects the transform direction.

Example:
    >>> from evo.compute.tasks import run, Source, Target, CreateAttribute
    >>> from evo.compute.tasks.normal_score import NormalScoreParameters
    >>>
    >>> params = NormalScoreParameters(
    ...     method="forward",
    ...     source=Source(object=pointset_url, attribute="locations.attributes[0]"),
    ...     target=Target(
    ...         object=pointset_url,
    ...         attribute=CreateAttribute(name="grade_ns"),
    ...     ),
    ...     distribution=distribution_url,
    ... )
    >>> result = await run(manager, params)
"""

from __future__ import annotations

from typing import ClassVar, Literal, Protocol, runtime_checkable

import pandas as pd
from evo.common import IContext, IFeedback
from evo.objects import ObjectSchema
from evo.objects.typed import BaseObject, object_from_reference
from pydantic import BaseModel

from .common import (
    AnySourceAttribute,
    AnyTargetAttribute,
    GeoscienceObjectReference,
)
from .common.results import TaskTarget
from .common.runner import TaskRunner

__all__ = [
    "NormalScoreParameters",
    "NormalScoreResult",
    "NormalScoreResultModel",
    "NormalScoreRunner",
]


# =============================================================================
# Normal Score Parameters
# =============================================================================


class NormalScoreParameters(BaseModel):
    """Parameters for the normal-score compute task.

    Set ``method="forward"`` to transform from data space to Gaussian,
    or ``method="backward"`` to transform from Gaussian back to data space.

    Example:
        >>> params = NormalScoreParameters(
        ...     method="forward",
        ...     source=Source(object=pointset_url, attribute="locations.attributes[0]"),
        ...     target=Target(
        ...         object=pointset_url,
        ...         attribute=CreateAttribute(name="grade_ns"),
        ...     ),
        ...     distribution=distribution_url,
        ... )
    """

    method: Literal["forward", "backward"]
    """Transform direction: ``"forward"`` (data → Gaussian) or ``"backward"`` (Gaussian → data)."""

    source: AnySourceAttribute
    """The source object and attribute containing values to transform."""

    target: AnyTargetAttribute
    """The target object and attribute to create or update with transformed values."""

    distribution: GeoscienceObjectReference
    """Reference URL to the continuous distribution object used for the transform."""


# =============================================================================
# Normal Score Result Types
# =============================================================================


@runtime_checkable
class _ObjToDataframeProtocol(Protocol):
    """Protocol for objects that can convert themselves to a DataFrame."""

    async def to_dataframe(self, *keys: str, fb: IFeedback = ...) -> pd.DataFrame: ...


class NormalScoreResultModel(BaseModel):
    """Pydantic model for the raw normal-score task result."""

    message: str
    """A message describing what happened in the task."""

    target: TaskTarget
    """Target information from the task result."""


class NormalScoreResult:
    """Result of a normal-score transform task.

    Provides access to the target object and the created/updated attribute.
    """

    TASK_DISPLAY_NAME: ClassVar[str] = "Normal Score"

    def __init__(self, context: IContext, model: NormalScoreResultModel) -> None:
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
        """The reference URL to the target object."""
        return self._target.reference

    @property
    def attribute_name(self) -> str:
        """The name of the attribute that was created/updated."""
        return self._target.attribute.name

    @property
    def schema(self) -> ObjectSchema:
        """The schema type of the target object."""
        return ObjectSchema.from_id(self._target.schema_id)

    async def get_target_object(self) -> BaseObject:
        """Load and return the target geoscience object.

        Returns:
            The typed geoscience object (e.g., PointSet)
        """
        return await object_from_reference(self._context, self._target.reference)

    async def to_dataframe(self, *columns: str) -> pd.DataFrame:
        """Get the task results as a DataFrame.

        Args:
            columns: Optional column names to include. If omitted, returns all.

        Returns:
            A pandas DataFrame containing the transformed attribute values.
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


class NormalScoreRunner(
    TaskRunner[NormalScoreParameters, NormalScoreResultModel, NormalScoreResult],
    topic="geostatistics",
    task="normal-score",
):
    """Runner for normal-score compute tasks.

    Automatically registered — used by ``run()`` for dispatch, or directly::

        result = await NormalScoreRunner(context, params)
    """

    async def _get_result(self, raw_result: NormalScoreResultModel) -> NormalScoreResult:
        return NormalScoreResult(self._context, raw_result)
