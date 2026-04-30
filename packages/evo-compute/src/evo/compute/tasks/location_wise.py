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
Location-wise compute task client.

This module provides typed dataclass models and convenience functions for running
the Location-Wise task (geostatistics/location-wise).

Computes per-location statistics across ensemble simulation realizations.
All operations are optional but at least one must be specified:

- **summary**: min, max, mean, variance (4 attributes)
- **quantiles**: specified percentiles (N attributes)
- **probability_above_cutoff**: P(X > cutoff) for each cutoff (N attributes)
- **mean_above_cutoff**: E[X | X > cutoff] for each cutoff (N attributes)

Example:
    >>> from evo.compute.tasks import run, Source
    >>> from evo.compute.tasks.location_wise import (
    ...     LocationWiseParameters,
    ...     LocationWiseTarget,
    ...     ProbabilityAboveCutoff,
    ... )
    >>>
    >>> params = LocationWiseParameters(
    ...     source=Source(object=grid_url, attribute="cell_attributes[0]"),
    ...     target=LocationWiseTarget(object=grid_url),
    ...     summary=True,
    ...     quantiles=[0.1, 0.5, 0.9],
    ...     probability_above_cutoff=ProbabilityAboveCutoff(cutoffs=[0.5, 1.0]),
    ... )
    >>> result = await run(manager, params)
"""

from __future__ import annotations

from typing import ClassVar, Protocol, runtime_checkable

import pandas as pd
from evo.common import IContext, IFeedback
from evo.objects import ObjectSchema
from evo.objects.typed import BaseObject, object_from_reference
from pydantic import BaseModel, Field

from .common import (
    AnySourceAttribute,
    GeoscienceObjectReference,
)
from .common.results import TaskAttribute
from .common.runner import TaskRunner

__all__ = [
    "LocationWiseParameters",
    "LocationWiseResult",
    "LocationWiseResultModel",
    "LocationWiseRunner",
    "LocationWiseTarget",
    "LocationWiseTargetResult",
    "MeanAboveCutoff",
    "ProbabilityAboveCutoff",
]


# =============================================================================
# Location-Wise Target & Option Types
# =============================================================================


class LocationWiseTarget(BaseModel):
    """Target object for location-wise tasks.

    Only the object reference is needed — the task creates output attributes
    automatically based on the requested operations.
    """

    object: GeoscienceObjectReference
    """Reference URL to the target geoscience object."""


class ProbabilityAboveCutoff(BaseModel):
    """Configuration for probability-above-cutoff operation.

    For each cutoff value, the task computes P(X > cutoff) at every location
    across all realizations.
    """

    cutoffs: list[float] = Field(min_length=1)
    """List of cutoff values. Must contain at least one value."""


class MeanAboveCutoff(BaseModel):
    """Configuration for mean-above-cutoff operation.

    For each cutoff value, the task computes E[X | X > cutoff] at every
    location across all realizations.
    """

    cutoffs: list[float] = Field(min_length=1)
    """List of cutoff values. Must contain at least one value."""


# =============================================================================
# Location-Wise Parameters
# =============================================================================


class LocationWiseParameters(BaseModel):
    """Parameters for the unified location-wise task.

    At least one operation must be specified (summary, quantiles,
    probability_above_cutoff, or mean_above_cutoff).

    Example:
        >>> params = LocationWiseParameters(
        ...     source=Source(object=grid_url, attribute="cell_attributes[0]"),
        ...     target=LocationWiseTarget(object=grid_url),
        ...     summary=True,
        ...     quantiles=[0.1, 0.5, 0.9],
        ... )
    """

    source: AnySourceAttribute
    """The source object and attribute containing simulation realizations."""

    target: LocationWiseTarget
    """The target object where computed statistics will be stored."""

    summary: bool | None = None
    """If True, compute min, max, mean, and variance at each location."""

    quantiles: list[float] | None = None
    """List of quantile values (0–1) to compute at each location."""

    probability_above_cutoff: ProbabilityAboveCutoff | None = None
    """Configuration for probability-above-cutoff computation."""

    mean_above_cutoff: MeanAboveCutoff | None = None
    """Configuration for mean-above-cutoff computation."""


# =============================================================================
# Location-Wise Result Types
# =============================================================================


@runtime_checkable
class _ObjToDataframeProtocol(Protocol):
    """Protocol for objects that can convert themselves to a DataFrame."""

    async def to_dataframe(self, *keys: str, fb: IFeedback = ...) -> pd.DataFrame: ...


class LocationWiseTargetResult(BaseModel):
    """Target information from a location-wise task result.

    Unlike single-attribute tasks, location-wise produces multiple attributes
    (one per requested statistic/quantile/cutoff).
    """

    reference: str
    """Reference URL to the target object."""

    name: str
    """Name of the target object."""

    description: str | None = None
    """Optional description of the target object."""

    schema_id: str
    """Schema identifier for the target object type."""

    attributes: list[TaskAttribute]
    """List of attributes created by the task."""


class LocationWiseResultModel(BaseModel):
    """Pydantic model for the raw location-wise task result."""

    message: str
    """A message describing what happened in the task."""

    target: LocationWiseTargetResult
    """Target information with the list of created attributes."""


class LocationWiseResult:
    """Result of a location-wise task.

    Provides access to the multiple computed statistics attributes.
    """

    TASK_DISPLAY_NAME: ClassVar[str] = "Location-Wise"

    def __init__(self, context: IContext, model: LocationWiseResultModel) -> None:
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
    def attribute_names(self) -> list[str]:
        """Names of all attributes created by the task."""
        return [a.name for a in self._target.attributes]

    @property
    def schema(self) -> ObjectSchema:
        """The schema type of the target object."""
        return ObjectSchema.from_id(self._target.schema_id)

    async def get_target_object(self) -> BaseObject:
        """Load and return the target geoscience object.

        Returns:
            The typed geoscience object (e.g., Regular3DGrid)
        """
        return await object_from_reference(self._context, self._target.reference)

    async def to_dataframe(self, *columns: str) -> pd.DataFrame:
        """Get the task results as a DataFrame.

        Args:
            columns: Optional column names to include. If omitted, returns all.

        Returns:
            A pandas DataFrame containing the computed statistics.
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
        attrs = ", ".join(self.attribute_names)
        lines = [
            f"✓ {self.TASK_DISPLAY_NAME} Result",
            f"  Message:    {self.message}",
            f"  Target:     {self.target_name}",
            f"  Attributes: {attrs}",
        ]
        return "\n".join(lines)


# =============================================================================
# Task Runner
# =============================================================================


class LocationWiseRunner(
    TaskRunner[LocationWiseParameters, LocationWiseResultModel, LocationWiseResult],
    topic="geostatistics",
    task="location-wise",
):
    """Runner for location-wise compute tasks.

    Automatically registered — used by ``run()`` for dispatch, or directly::

        result = await LocationWiseRunner(context, params)
    """

    async def _get_result(self, raw_result: LocationWiseResultModel) -> LocationWiseResult:
        return LocationWiseResult(self._context, raw_result)
