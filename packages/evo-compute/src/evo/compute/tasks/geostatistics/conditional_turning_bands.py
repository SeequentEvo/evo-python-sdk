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

"""Conditional turning-band simulation (conditional-tbsim) compute task client.

The conditional turning-bands task performs a conditional simulation of a
numeric attribute onto a 3-D grid using a pre-computed continuous distribution.
Unlike the full conditioned-simulator workflow, this task takes an existing
distribution object produced by the ``continuous-distribution`` task and skips
the internal normal-score transform; this makes it faster when the distribution
is already available.

Example:
    >>> from evo.compute.tasks import run, SearchNeighborhood, Ellipsoid, EllipsoidRanges
    >>> from evo.compute.tasks.geostatistics.conditioned_simulator import BlockDiscretization
    >>> from evo.compute.tasks.geostatistics.conditional_turning_bands import (
    ...     ConditionalTurningBandsParameters,
    ... )
    >>>
    >>> params = ConditionalTurningBandsParameters(
    ...     source=pointset,
    ...     source_attribute="locations.attributes[?name=='grade']",
    ...     target=grid,
    ...     distribution=distribution_object,
    ...     variogram_model=variogram,
    ...     neighborhood=SearchNeighborhood(
    ...         ellipsoid=Ellipsoid(ranges=EllipsoidRanges(70, 70, 5)),
    ...         max_samples=40,
    ...     ),
    ...     block_discretization=BlockDiscretization(nx=5, ny=5, nz=5),
    ...     realizations=10,
    ... )
    >>> result = await run(manager, params)
"""

from __future__ import annotations

from typing import ClassVar, Literal, Protocol, runtime_checkable

import pandas as pd
from evo.common import IContext, IFeedback
from evo.objects import ObjectSchema
from evo.objects.typed import BaseObject, object_from_reference
from pydantic import BaseModel, Field

from ..common import (
    Filter,
    GeoscienceObjectReference,
    SearchNeighborhood,
)
from ..common.results import TaskAttribute
from ..common.runner import TaskRunner
from .conditioned_simulator import BlockDiscretization

__all__ = [
    "BlockDiscretization",
    "ConditionalTurningBandsParameters",
    "ConditionalTurningBandsResult",
    "ConditionalTurningBandsResultModel",
    "ConditionalTurningBandsRunner",
    "ConditionalTurningBandsTargetResult",
    "Filter",
]


# =============================================================================
# Parameters
# =============================================================================


class ConditionalTurningBandsParameters(BaseModel):
    """Parameters for the conditional turning-band simulation task.

    Performs a conditional block turning-bands simulation using a pre-existing
    continuous distribution object.  The task outputs an ensemble attribute on
    the target grid, containing one column per realization.

    Example:
        >>> params = ConditionalTurningBandsParameters(
        ...     source=pointset,
        ...     source_attribute="locations.attributes[?name=='grade']",
        ...     target=grid,
        ...     distribution=distribution_object,
        ...     variogram_model=variogram,
        ...     neighborhood=SearchNeighborhood(
        ...         ellipsoid=Ellipsoid(ranges=EllipsoidRanges(70, 70, 5)),
        ...         max_samples=40,
        ...     ),
        ...     block_discretization=BlockDiscretization(nx=5, ny=5, nz=5),
        ...     realizations=10,
        ... )
    """

    source: GeoscienceObjectReference
    """Reference to the pointset containing the source conditioning points."""

    source_attribute: str
    """Attribute reference for the source values (e.g. ``\"locations.attributes[?name=='grade']\"``)."""

    target: GeoscienceObjectReference
    """Reference to the target 3-D grid or masked grid to simulate onto."""

    filter: Filter | None = None
    """Optional filter restricting simulation to a subset of target-grid locations."""

    source_filter: Filter | None = None
    """Optional filter restricting conditioning to a subset of the source data."""

    neighborhood: SearchNeighborhood
    """Search neighbourhood used both for simulation and for the conditioning kriging step."""

    distribution: GeoscienceObjectReference
    """Reference to a non-parametric continuous cumulative distribution object.

    Typically created by the ``continuous-distribution`` task.  The distribution
    is used for the normal-score back-transformation of simulation results.
    """

    variogram_model: GeoscienceObjectReference
    """Reference to the variogram model used to model spatial covariance."""

    kriging_method: Literal["simple", "ordinary"] = "simple"
    """The kriging method for the conditioning step.

    - ``"simple"`` (default) — assumes a known constant mean.
    - ``"ordinary"`` — estimates the local mean from nearby samples.
    """

    block_discretization: BlockDiscretization = Field(default_factory=BlockDiscretization)
    """Sub-block discretisation for support correction.

    Each grid cell is subdivided into ``nx * ny * nz`` sub-cells, simulated
    individually, and then averaged to the block scale.  Defaults to
    ``BlockDiscretization(nx=1, ny=1, nz=1)`` (point simulation at cell centres).
    """

    number_of_lines: int = Field(500, ge=1, le=1000)
    """Number of turning-band lines.

    Higher values produce more accurate results at the cost of runtime.
    Must be between 1 and 1000.  Defaults to 500.
    """

    realizations: int = Field(1, ge=1, le=100)
    """Number of simulation realizations to produce.

    All realizations are saved to the ensemble attribute on the target object.
    Must be between 1 and 100.  Defaults to 1.
    """

    random_seed: int = 38239342
    """Random seed for reproducible simulations."""


# =============================================================================
# Result Types
# =============================================================================


class ConditionalTurningBandsTargetResult(BaseModel):
    """Target grid result from the conditional turning-bands task."""

    reference: str
    """Object reference URL for the target grid."""

    name: str
    """Name of the target grid object."""

    schema_id: str
    """Schema identifier for the target object type."""

    simulations: TaskAttribute | None = None
    """Ensemble attribute containing the block-scale simulation realizations."""


class ConditionalTurningBandsResultModel(BaseModel):
    """Pydantic model for the raw conditional turning-bands task result payload."""

    target: ConditionalTurningBandsTargetResult
    """Target grid with simulation outputs."""


@runtime_checkable
class _ObjToDataframeProtocol(Protocol):
    async def to_dataframe(self, *keys: str, fb: IFeedback = ...) -> pd.DataFrame: ...


class ConditionalTurningBandsResult:
    """Rich result object for the conditional turning-bands simulation task."""

    TASK_DISPLAY_NAME: ClassVar[str] = "Conditional Turning-Band Simulation"

    def __init__(self, context: IContext, model: ConditionalTurningBandsResultModel) -> None:
        self._target = model.target
        self._context = context

    @property
    def target_name(self) -> str:
        """The name of the target grid object."""
        return self._target.name

    @property
    def target_reference(self) -> str:
        """Reference URL to the target grid object."""
        return self._target.reference

    @property
    def schema(self) -> ObjectSchema:
        """The schema type of the target object (e.g., ``'regular-3d-grid'``)."""
        return ObjectSchema.from_id(self._target.schema_id)

    @property
    def simulations_attribute(self) -> TaskAttribute | None:
        """Ensemble attribute containing the saved block-scale simulation realizations."""
        return self._target.simulations

    async def get_target_object(self) -> BaseObject:
        """Load and return the target geoscience object."""
        return await object_from_reference(self._context, self._target.reference)

    async def to_dataframe(self, *columns: str) -> pd.DataFrame:
        """Load the target object and return its data as a DataFrame.

        Args:
            *columns: Optional column names to select.

        Raises:
            TypeError: If the target object type does not support DataFrame export.
        """
        target_obj = await self.get_target_object()
        if isinstance(target_obj, _ObjToDataframeProtocol):
            return await target_obj.to_dataframe(*columns)
        raise TypeError(
            f"Don't know how to get DataFrame from {type(target_obj).__name__}. "
            "Use get_target_object() and access the data manually."
        )

    def __str__(self) -> str:
        lines = [
            f"✓ {self.TASK_DISPLAY_NAME} Result",
            f"  Target: {self.target_name}",
        ]
        if self._target.simulations:
            lines.append(f"  Simulations: {self._target.simulations.name}")
        return "\n".join(lines)


# =============================================================================
# Task Runner
# =============================================================================


class ConditionalTurningBandsRunner(
    TaskRunner[
        ConditionalTurningBandsParameters,
        ConditionalTurningBandsResultModel,
        ConditionalTurningBandsResult,
    ],
    topic="geostatistics",
    task="conditional-turning-bands",
):
    async def _get_result(self, raw_result: ConditionalTurningBandsResultModel) -> ConditionalTurningBandsResult:
        return ConditionalTurningBandsResult(self._context, raw_result)
