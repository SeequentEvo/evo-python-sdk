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

"""Conditional simulation (consim) compute task client.

Full conditional turning-band simulation workflow: transforms source data to
normal-score space, performs kriging + turning-band simulation on a target grid,
and back-transforms results.  Supports optional loss calculation,
location-wise quantile/summary statistics, and validation reporting.

Example:
    >>> from evo.compute.tasks import run
    >>> from evo.compute.tasks.conditioned_simulator import (
    ...     ConSimParameters,
    ...     BlockDiscretization,
    ... )
    >>>
    >>> params = ConSimParameters(
    ...     source_object=pointset,
    ...     source_attribute="locations.attributes[?name=='grade']",
    ...     target_object=grid,
    ...     variogram_model=variogram,
    ...     neighborhood=SearchNeighborhood(
    ...         ellipsoid=Ellipsoid(ranges=EllipsoidRanges(300, 200, 100)),
    ...         max_samples=24,
    ...     ),
    ...     block_discretization=BlockDiscretization(nx=3, ny=3, nz=2),
    ...     number_of_simulations=10,
    ... )
    >>> result = await run(manager, params)
"""

from __future__ import annotations

from typing import Any, ClassVar, Literal, Protocol, runtime_checkable

import pandas as pd
from evo.common import IContext, IFeedback
from evo.objects import ObjectSchema
from evo.objects.typed import BaseObject, object_from_reference
from pydantic import BaseModel, Field, SerializerFunctionWrapHandler, model_serializer

from .common import (
    AnySourceAttribute,
    AnyTargetAttribute,
    CreateAttribute,
    GeoscienceObjectReference,
    SearchNeighborhood,
    UpdateAttribute,
)
from .common.results import TaskAttribute
from .common.runner import TaskRunner

__all__ = [
    "BlockDiscretization",
    "ConSimLossCalculation",
    "ConSimMaterialCategory",
    "ConSimParameters",
    "ConSimResult",
    "ConSimResultModel",
    "ConSimRunner",
    "DistributionParams",
    "ReportContext",
    "ReportMeanThresholds",
    "ValidationReportContextItem",
]


# =============================================================================
# Supporting Types
# =============================================================================


class UpperTailParams(BaseModel):
    """Upper tail extrapolation for the distribution."""

    power: float = Field(gt=0.0, le=1.0)
    max: float


class LowerTailParams(BaseModel):
    """Lower tail extrapolation for the distribution."""

    power: float = Field(gt=0.0, le=1.0)
    min: float


class TailExtrapolationParams(BaseModel):
    """Tail extrapolation parameters."""

    upper: UpperTailParams
    lower: LowerTailParams | None = None


class DistributionParams(BaseModel):
    """Parameters for the continuous distribution used for normal-score transformation."""

    weights: str | None = None
    """Optional weights attribute reference for weighted distribution."""

    tail_extrapolation: TailExtrapolationParams | None = None
    """Optional tail extrapolation parameters."""


class BlockDiscretization(BaseModel):
    """Sub-block discretization for support correction."""

    nx: int = Field(1, ge=1, le=9)
    ny: int = Field(1, ge=1, le=9)
    nz: int = Field(1, ge=1, le=9)


class ConSimMaterialCategory(BaseModel):
    """Material category for loss calculation within simulation."""

    cutoff_grade: float = Field(ge=0.0)
    metal_price: float = Field(ge=0.0)
    label: str


class ConSimLossCalculation(BaseModel):
    """Settings for loss calculation within the simulation."""

    material_categories: list[ConSimMaterialCategory]
    processing_cost: float = Field(ge=0.0)
    mining_waste_cost: float = Field(ge=0.0)
    mining_ore_cost: float = Field(ge=0.0)
    metal_recovery_fraction: float = Field(ge=0.0, le=1.0)
    target_attribute: CreateAttribute | UpdateAttribute


class ValidationReportContextItem(BaseModel):
    """A single item of contextual information for the validation report."""

    label: str
    value: str


class ReportContext(BaseModel):
    """Context for the validation report."""

    title: str
    details: list[ValidationReportContextItem] = Field(min_length=1, max_length=10)


class ReportMeanThresholds(BaseModel):
    """Thresholds for mean comparison in the validation report."""

    acceptable: float = Field(ge=0.0)
    marginal: float = Field(ge=0.0)


# =============================================================================
# Parameters
# =============================================================================


class ConSimParameters(BaseModel):
    """Parameters for the conditional simulation (consim) task.

    This is the most comprehensive geostatistics compute task, supporting
    the full conditional turning-band simulation workflow.
    """

    source_object: GeoscienceObjectReference
    """Reference to the pointset containing source conditioning points."""

    source_attribute: str
    """Attribute reference for the source values."""

    target_object: GeoscienceObjectReference
    """Reference to the target 3D grid."""

    neighborhood: SearchNeighborhood
    """Search neighborhood for conditioning."""

    variogram_model: GeoscienceObjectReference
    """Reference to the variogram model."""

    block_discretization: BlockDiscretization = Field(default_factory=BlockDiscretization)
    """Sub-block discretization for support correction."""

    distribution: DistributionParams | None = None
    """Optional distribution parameters for normal-score transformation."""

    kriging_method: Literal["simple", "ordinary"] = "simple"
    """The kriging method for conditioning (default: simple)."""

    number_of_lines: int = Field(500, ge=1, le=1000)
    """Number of lines for turning-band simulation."""

    number_of_simulations: int = Field(1, ge=1, le=100)
    """Number of simulations to run."""

    random_seed: int = 38239342
    """Random seed for simulation and tie-breaking."""

    number_of_simulations_to_save: int = Field(5, ge=0, le=100)
    """Number of simulations to publish to the output object."""

    loss_calculation: ConSimLossCalculation | None = None
    """Optional loss calculation settings."""

    location_wise_quantiles: list[float] | None = None
    """Optional list of quantiles to compute for each location."""

    perform_validation: bool = False
    """Whether to run validation and generate a validation report."""

    report_context: ReportContext | None = None
    """Optional context for the validation report."""

    report_mean_thresholds: ReportMeanThresholds | None = None
    """Optional thresholds for mean comparison in the validation report."""


# =============================================================================
# Result Types
# =============================================================================


class ConSimSummaryAttributes(BaseModel):
    """Summary statistic attributes from the simulation."""

    mean: TaskAttribute
    variance: TaskAttribute
    min: TaskAttribute
    max: TaskAttribute


class ConSimQuantileAttribute(BaseModel):
    """A quantile attribute from the simulation results."""

    reference: str
    name: str
    quantile: float


class ConSimValidationSummary(BaseModel):
    """Summary of validation results."""

    reference_mean: float
    mean: float


class ConSimValidationReport(BaseModel):
    """Reference to the validation report file."""

    reference: str
    name: str


class ConSimLinks(BaseModel):
    """Links related to the simulation task."""

    dashboard: str | None = None


class ConSimTargetResult(BaseModel):
    """Target grid result with all simulation outputs."""

    reference: str
    name: str
    description: str | None = None
    schema_id: str
    summary_attributes: ConSimSummaryAttributes
    quantile_attributes: list[ConSimQuantileAttribute] | None = None
    simulations: TaskAttribute | None = None
    simulations_normal_score: TaskAttribute | None = None
    point_simulations: TaskAttribute | None = None
    point_simulations_normal_score: TaskAttribute | None = None
    loss_calculation_attribute: TaskAttribute | None = None


class ConSimResultModel(BaseModel):
    """Pydantic model for the raw consim task result."""

    target: ConSimTargetResult
    validation_summary: ConSimValidationSummary | None = None
    validation_report: ConSimValidationReport | None = None
    links: ConSimLinks


@runtime_checkable
class _ObjToDataframeProtocol(Protocol):
    async def to_dataframe(self, *keys: str, fb: IFeedback = ...) -> pd.DataFrame: ...


class ConSimResult:
    TASK_DISPLAY_NAME: ClassVar[str] = "Conditional Simulation"

    def __init__(self, context: IContext, model: ConSimResultModel) -> None:
        self._target = model.target
        self._validation_summary = model.validation_summary
        self._validation_report = model.validation_report
        self._links = model.links
        self._context = context

    @property
    def target_name(self) -> str:
        return self._target.name

    @property
    def target_reference(self) -> str:
        return self._target.reference

    @property
    def schema(self) -> ObjectSchema:
        return ObjectSchema.from_id(self._target.schema_id)

    @property
    def summary_attributes(self) -> ConSimSummaryAttributes:
        """Summary statistics (mean, variance, min, max) attribute references."""
        return self._target.summary_attributes

    @property
    def quantile_attributes(self) -> list[ConSimQuantileAttribute]:
        """Quantile attribute references, if computed."""
        return self._target.quantile_attributes or []

    @property
    def simulations_attribute(self) -> TaskAttribute | None:
        """Ensemble attribute containing saved simulations."""
        return self._target.simulations

    @property
    def loss_calculation_attribute(self) -> TaskAttribute | None:
        """Category attribute containing loss calculation results."""
        return self._target.loss_calculation_attribute

    @property
    def validation_summary(self) -> ConSimValidationSummary | None:
        return self._validation_summary

    @property
    def dashboard_url(self) -> str | None:
        return self._links.dashboard

    async def get_target_object(self) -> BaseObject:
        return await object_from_reference(self._context, self._target.reference)

    async def to_dataframe(self, *columns: str) -> pd.DataFrame:
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
            f"  Target:     {self.target_name}",
            f"  Mean attr:  {self.summary_attributes.mean.name}",
            f"  Var attr:   {self.summary_attributes.variance.name}",
        ]
        if self._validation_summary:
            lines.append(f"  Ref mean:   {self._validation_summary.reference_mean:.4f}")
            lines.append(f"  Sim mean:   {self._validation_summary.mean:.4f}")
        if self._links.dashboard:
            lines.append(f"  Dashboard:  {self._links.dashboard}")
        return "\n".join(lines)


# =============================================================================
# Task Runner
# =============================================================================


class ConSimRunner(
    TaskRunner[ConSimParameters, ConSimResultModel, ConSimResult],
    topic="geostatistics",
    task="consim",
):
    async def _get_result(self, raw_result: ConSimResultModel) -> ConSimResult:
        return ConSimResult(self._context, raw_result)
