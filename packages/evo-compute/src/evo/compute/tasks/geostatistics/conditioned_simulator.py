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

"""Conditional simulation (consim) compute task client.

Full conditional turning-band simulation workflow: transforms source data to
normal-score space, performs kriging + turning-band simulation on a target grid,
and back-transforms results.  Supports optional loss calculation,
location-wise quantile/summary statistics, and validation reporting.

Example::

    >>> from evo.compute.tasks import run
    >>> from evo.compute.tasks.geostatistics.conditioned_simulator import (
    ...     ConSimParameters,
    ...     BlockDiscretization,
    ... )
    >>>
    >>> params = ConSimParameters(
    ...     source_object=pointset,
    ...     source_attribute=pointset.attributes["grade"],
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

from collections.abc import Mapping
from typing import Any, ClassVar, Literal, Protocol, runtime_checkable

import pandas as pd
from evo.common import IContext, IFeedback
from evo.objects import ObjectSchema
from evo.objects.typed import BaseObject, PointSet, Regular3DGrid, RegularMasked3DGrid, Variogram, object_from_reference
from pydantic import (
    BaseModel,
    Field,
    PrivateAttr,
    ValidatorFunctionWrapHandler,
    field_validator,
    model_validator,
)

from ..common import (
    BlockDiscretization,
    CreateAttribute,
    ExistingAttributeExpression,
    Filter,
    FiniteFloat,
    GeoscienceObjectReference,
    QuantileFloat,
    SearchNeighborhood,
    TaskInputModel,
    TaskParameters,
    UpdateAttribute,
    object_identity,
    object_matches_reference,
)
from ..common.results import TaskAttribute
from ..common.runner import TaskRunner
from ..common.search import _reject_sector_and_drillhole_limits
from ..common.source_target import AnyTypedAttribute

__all__ = [
    "BlockDiscretization",
    "ConSimCutoffAttribute",
    "ConSimLinks",
    "ConSimLossCalculation",
    "ConSimMaterialCategory",
    "ConSimParameters",
    "ConSimQuantileAttribute",
    "ConSimResult",
    "ConSimResultModel",
    "ConSimRunner",
    "ConSimSummaryAttributes",
    "ConSimTargetResult",
    "ConSimValidationReport",
    "ConSimValidationSummary",
    "DistributionParams",
    "Filter",
    "LowerTailParams",
    "ReportContext",
    "ReportMeanThresholds",
    "TailExtrapolationParams",
    "UpperTailParams",
    "ValidationReportContextItem",
]


# =============================================================================
# Supporting Types
# =============================================================================


class UpperTailParams(TaskInputModel):
    """Upper tail extrapolation for the distribution."""

    power: float = Field(gt=0.0, le=1.0)
    max: float


class LowerTailParams(TaskInputModel):
    """Lower tail extrapolation for the distribution."""

    power: float = Field(gt=0.0, le=1.0)
    min: float


class TailExtrapolationParams(TaskInputModel):
    """Tail extrapolation parameters."""

    upper: UpperTailParams
    lower: LowerTailParams | None = None

    @model_validator(mode="after")
    def _validate_tail_bounds(self) -> TailExtrapolationParams:
        if self.lower is not None and self.lower.min >= self.upper.max:
            raise ValueError(f"lower.min ({self.lower.min}) must be strictly less than upper.max ({self.upper.max}).")
        return self


class DistributionParams(TaskInputModel):
    """Parameters for the continuous distribution used for normal-score transformation."""

    _weights_object_url: str | None = PrivateAttr(default=None)
    _weights_parent: Any = PrivateAttr(default=None)

    weights: ExistingAttributeExpression | None = None
    """Optional weights attribute for a weighted (declustered) distribution.

    Accepts a typed attribute or an attribute expression.
    """

    tail_extrapolation: TailExtrapolationParams | None = None
    """Optional tail extrapolation parameters."""

    @model_validator(mode="wrap")
    @classmethod
    def _remember_weights_object(cls, data: Any, handler: ValidatorFunctionWrapHandler) -> DistributionParams:
        weights_url: str | None = None
        weights_parent: Any = None
        if isinstance(data, DistributionParams):
            weights_url = data._weights_object_url
            weights_parent = data._weights_parent
        elif isinstance(data, Mapping):
            raw_weights = data.get("weights")
            if isinstance(raw_weights, AnyTypedAttribute) and raw_weights._obj is not None:
                weights_parent = raw_weights._obj
                try:
                    weights_url = str(raw_weights._obj.metadata.url)
                except Exception:
                    weights_url = None
        dist = handler(data)
        if weights_url is not None and dist._weights_object_url is None:
            dist._weights_object_url = weights_url
        if weights_parent is not None and dist._weights_parent is None:
            dist._weights_parent = weights_parent
        return dist


class ConSimMaterialCategory(TaskInputModel):
    """Material category for loss calculation within simulation."""

    cutoff_grade: float = Field(ge=0.0)
    metal_price: float = Field(ge=0.0)
    label: str = Field(min_length=1)


def _validate_material_category_sequence(categories: list[Any]) -> None:
    seen_labels: set[str] = set()
    for index, cat in enumerate(categories):
        if cat.label in seen_labels:
            raise ValueError(f"Duplicate material category label {cat.label!r}.")
        seen_labels.add(cat.label)
        if index > 0 and cat.cutoff_grade <= categories[index - 1].cutoff_grade:
            raise ValueError(
                "material_categories must be ordered by strictly increasing cutoff_grade "
                f"(got {categories[index - 1].cutoff_grade} followed by {cat.cutoff_grade})."
            )


class ConSimLossCalculation(TaskInputModel):
    """Settings for loss calculation within the simulation."""

    material_categories: list[ConSimMaterialCategory] = Field(min_length=1)
    processing_cost: float = Field(ge=0.0)
    mining_waste_cost: float = Field(ge=0.0)
    mining_ore_cost: float = Field(ge=0.0)
    metal_recovery_fraction: float = Field(ge=0.0, le=1.0)
    target_attribute: CreateAttribute | UpdateAttribute

    @model_validator(mode="after")
    def _check_categories(self) -> ConSimLossCalculation:
        _validate_material_category_sequence(self.material_categories)
        return self


class ValidationReportContextItem(TaskInputModel):
    """A single item of contextual information for the validation report."""

    label: str
    value: str


class ReportContext(TaskInputModel):
    """Context for the validation report."""

    title: str
    details: list[ValidationReportContextItem] = Field(min_length=1, max_length=10)


class ReportMeanThresholds(TaskInputModel):
    """Thresholds for mean comparison in the validation report."""

    acceptable: float = Field(ge=0.0)
    marginal: float = Field(ge=0.0)

    @model_validator(mode="after")
    def _validate_threshold_order(self) -> ReportMeanThresholds:
        if self.marginal < self.acceptable:
            raise ValueError(
                f"marginal threshold ({self.marginal}) must be greater than or equal to "
                f"acceptable threshold ({self.acceptable})."
            )
        return self


# =============================================================================
# Parameters
# =============================================================================


class ConSimParameters(TaskParameters):
    """Parameters for the conditional simulation (consim) task.

    This is the most comprehensive geostatistics compute task, supporting
    the full conditional turning-band simulation workflow.
    """

    _object_roles: ClassVar[Mapping[str, tuple[type, ...]]] = {
        "source_object": (PointSet,),
        "source_attribute": (PointSet,),
        "target_object": (Regular3DGrid, RegularMasked3DGrid),
        "variogram_model": (Variogram,),
    }
    _attribute_object_bindings: ClassVar[tuple[tuple[str, str], ...]] = (("source_attribute", "source_object"),)
    _filter_bindings: ClassVar[tuple[tuple[str, str], ...]] = (
        ("source_filter", "source_object"),
        ("filter", "target_object"),
    )

    source_object: GeoscienceObjectReference
    """Reference to the pointset containing source conditioning points."""

    source_attribute: ExistingAttributeExpression
    """The source values. Accepts a typed attribute such as ``pointset.attributes["grade"]``,
    or an attribute expression such as ``"locations.attributes[?name=='grade']"``."""

    target_object: GeoscienceObjectReference
    """Reference to the target 3D grid."""

    filter: Filter | None = None
    """Optional filter restricting the simulation to a subset of target-grid locations."""

    source_filter: Filter | None = None
    """Optional filter restricting conditioning to a subset of the source data."""

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

    location_wise_quantiles: list[QuantileFloat] | None = Field(None, min_length=1)
    """Optional list of quantiles to compute for each location."""

    probability_above_cutoff: list[FiniteFloat] | None = Field(None, min_length=1)
    """Optional list of cutoff values to compute the probability of being above, for each location."""

    mean_above_cutoff: list[FiniteFloat] | None = Field(None, min_length=1)
    """Optional list of cutoff values to compute the mean of values above, for each location."""

    perform_validation: bool = False
    """Whether to run validation and generate a validation report."""

    report_context: ReportContext | None = None
    """Optional context for the validation report."""

    report_mean_thresholds: ReportMeanThresholds | None = None
    """Optional thresholds for mean comparison in the validation report."""

    @field_validator("neighborhood")
    @classmethod
    def _check_neighborhood(cls, neighborhood: SearchNeighborhood) -> SearchNeighborhood:
        return _reject_sector_and_drillhole_limits(neighborhood)

    @model_validator(mode="after")
    def _validate_simulation_invariants(self) -> ConSimParameters:
        if (
            "number_of_simulations_to_save" not in self.model_fields_set
            and self.number_of_simulations_to_save > self.number_of_simulations
        ):
            self.number_of_simulations_to_save = self.number_of_simulations
        elif self.number_of_simulations_to_save > self.number_of_simulations:
            raise ValueError(
                f"number_of_simulations_to_save ({self.number_of_simulations_to_save}) "
                f"cannot exceed number_of_simulations ({self.number_of_simulations})."
            )

        if self.perform_validation and self.number_of_simulations_to_save < 1:
            raise ValueError("perform_validation=True requires number_of_simulations_to_save >= 1.")

        if not self.perform_validation:
            if self.report_context is not None:
                raise ValueError("report_context requires perform_validation=True.")
            if self.report_mean_thresholds is not None:
                raise ValueError("report_mean_thresholds requires perform_validation=True.")

        if self.distribution is not None and self.distribution._weights_parent is not None:
            source_obj = self._typed_objects.get(object_identity(self.source_object))
            if not object_matches_reference(
                self.distribution._weights_parent,
                self.source_object,
                supplied_obj=source_obj,
            ):
                actual = (
                    object_identity(self.distribution._weights_object_url)
                    if self.distribution._weights_object_url
                    else "unknown"
                )
                raise ValueError(
                    f"distribution.weights belongs to object {actual!r}, "
                    f"which does not match 'source_object' ({object_identity(self.source_object)!r})."
                )

        return self


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


class ConSimCutoffAttribute(BaseModel):
    """An attribute computed against a cutoff value (probability/mean above cutoff)."""

    reference: str
    name: str
    cutoff: float


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
    probability_above_cutoff_attributes: list[ConSimCutoffAttribute] | None = None
    mean_above_cutoff_attributes: list[ConSimCutoffAttribute] | None = None


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
    def target(self) -> ConSimTargetResult:
        """Target grid result metadata."""
        return self._target

    @property
    def target_name(self) -> str:
        return self._target.name

    @property
    def target_reference(self) -> str:
        return self._target.reference

    @property
    def target_description(self) -> str | None:
        return self._target.description

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
    def probability_above_cutoff_attributes(self) -> list[ConSimCutoffAttribute]:
        """Probability-above-cutoff attribute references, if computed."""
        return self._target.probability_above_cutoff_attributes or []

    @property
    def mean_above_cutoff_attributes(self) -> list[ConSimCutoffAttribute]:
        """Mean-above-cutoff attribute references, if computed."""
        return self._target.mean_above_cutoff_attributes or []

    @property
    def simulations_attribute(self) -> TaskAttribute | None:
        """Ensemble attribute containing saved block-scale simulations."""
        return self._target.simulations

    @property
    def simulations_normal_score_attribute(self) -> TaskAttribute | None:
        """Ensemble attribute containing saved block-scale normal-score simulations."""
        return self._target.simulations_normal_score

    @property
    def point_simulations_attribute(self) -> TaskAttribute | None:
        """Ensemble attribute containing saved point-scale simulations."""
        return self._target.point_simulations

    @property
    def point_simulations_normal_score_attribute(self) -> TaskAttribute | None:
        """Ensemble attribute containing saved point-scale normal-score simulations."""
        return self._target.point_simulations_normal_score

    @property
    def loss_calculation_attribute(self) -> TaskAttribute | None:
        """Category attribute containing loss calculation results."""
        return self._target.loss_calculation_attribute

    @property
    def validation_summary(self) -> ConSimValidationSummary | None:
        return self._validation_summary

    @property
    def validation_report(self) -> ConSimValidationReport | None:
        """Validation report metadata, if validation was performed."""
        return self._validation_report

    @property
    def report_reference(self) -> str | None:
        """URL to the generated validation report file, if validation was performed."""
        if self._validation_report is not None:
            return self._validation_report.reference
        return None

    @property
    def links(self) -> ConSimLinks:
        """Links related to the simulation task."""
        return self._links

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
