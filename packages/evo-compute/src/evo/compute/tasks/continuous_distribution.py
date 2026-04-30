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

"""Continuous distribution compute task client.

Creates a non-parametric continuous cumulative distribution from a set of
source values, with optional weights and tail extrapolation.

Example:
    >>> from evo.compute.tasks import run
    >>> from evo.compute.tasks.continuous_distribution import (
    ...     ContinuousDistributionParameters,
    ...     DistributionTarget,
    ... )
    >>>
    >>> params = ContinuousDistributionParameters(
    ...     source=pointset.attributes["grade"],
    ...     target=DistributionTarget(reference=dist_url),
    ... )
    >>> result = await run(manager, params)
"""

from __future__ import annotations

from typing import ClassVar

from evo.common import IContext
from evo.objects import ObjectSchema
from evo.objects.typed import BaseObject, object_from_reference
from pydantic import BaseModel, Field

from .common import AnySourceAttribute, GeoscienceObjectReference
from .common.runner import TaskRunner

__all__ = [
    "ContinuousDistributionParameters",
    "ContinuousDistributionResult",
    "ContinuousDistributionResultModel",
    "ContinuousDistributionRunner",
    "DistributionTarget",
    "TailExtrapolation",
    "UpperTail",
    "LowerTail",
]


# =============================================================================
# Tail Extrapolation
# =============================================================================


class UpperTail(BaseModel):
    """Power model for extrapolating the upper tail of the distribution."""

    power: float = Field(gt=0.0, le=1.0)
    """Denominator of the exponent for the power model (0 < power <= 1)."""

    max: float
    """Maximum extent of tail, must be greater than the maximum value in the data."""


class LowerTail(BaseModel):
    """Power model for extrapolating the lower tail of the distribution."""

    power: float = Field(gt=0.0, le=1.0)
    """Denominator of the exponent for the power model (0 < power <= 1)."""

    min: float
    """Minimum extent of tail, must be less than the minimum value in the data."""


class TailExtrapolation(BaseModel):
    """Parameters for extending the distribution beyond the data range."""

    upper: UpperTail
    """Upper tail extrapolation parameters."""

    lower: LowerTail | None = None
    """Optional lower tail extrapolation parameters."""


# =============================================================================
# Target and Source
# =============================================================================


class DistributionTarget(BaseModel):
    """Target specifying where to create the continuous distribution object."""

    reference: GeoscienceObjectReference
    """Reference to the target distribution object."""

    overwrite: bool = False
    """Whether to overwrite an existing object."""

    description: str | None = None
    """Description to put into the target object."""

    tags: dict[str, str] = Field(default_factory=dict)
    """Tags to put into the target object."""


# =============================================================================
# Parameters
# =============================================================================


class ContinuousDistributionParameters(BaseModel):
    """Parameters for the continuous-distribution task."""

    source: AnySourceAttribute
    """The source object and attribute containing values to create a distribution from."""

    weights: AnySourceAttribute | None = None
    """Optional weights for creating a weighted (declustered) distribution."""

    tail_extrapolation: TailExtrapolation | None = None
    """Optional tail extrapolation parameters."""

    target: DistributionTarget
    """Target specifying where to create the distribution object."""


# =============================================================================
# Result Types
# =============================================================================


class DistributionOutput(BaseModel):
    """Reference to the created distribution object."""

    reference: str
    name: str
    description: str | None = None
    schema_id: str


class ContinuousDistributionResultModel(BaseModel):
    """Pydantic model for the raw continuous-distribution task result."""

    message: str
    distribution: DistributionOutput


class ContinuousDistributionResult:
    TASK_DISPLAY_NAME: ClassVar[str] = "Continuous Distribution"

    def __init__(self, context: IContext, model: ContinuousDistributionResultModel) -> None:
        self._distribution = model.distribution
        self._message = model.message
        self._context = context

    @property
    def message(self) -> str:
        return self._message

    @property
    def distribution_name(self) -> str:
        """The name of the created distribution object."""
        return self._distribution.name

    @property
    def distribution_reference(self) -> str:
        """Reference URL to the distribution object."""
        return self._distribution.reference

    @property
    def schema(self) -> ObjectSchema:
        return ObjectSchema.from_id(self._distribution.schema_id)

    async def get_distribution_object(self) -> BaseObject:
        """Load and return the created distribution object."""
        return await object_from_reference(self._context, self._distribution.reference)

    def __str__(self) -> str:
        lines = [
            f"✓ {self.TASK_DISPLAY_NAME} Result",
            f"  Message:      {self.message}",
            f"  Distribution: {self.distribution_name}",
        ]
        return "\n".join(lines)


# =============================================================================
# Task Runner
# =============================================================================


class ContinuousDistributionRunner(
    TaskRunner[ContinuousDistributionParameters, ContinuousDistributionResultModel, ContinuousDistributionResult],
    topic="geostatistics",
    task="continuous-distribution",
):
    async def _get_result(self, raw_result: ContinuousDistributionResultModel) -> ContinuousDistributionResult:
        return ContinuousDistributionResult(self._context, raw_result)
