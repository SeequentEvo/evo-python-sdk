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
Simulation report compute task client.

This module provides typed dataclass models and convenience functions for running
the Simulation Report task (geostatistics/simulation-report).

Generates a validation report for a conditional turning-band simulation,
including variogram reproduction statistics, summary metrics, and an
optional interactive dashboard.

Example::

    >>> from evo.compute.tasks import run, SearchNeighborhood
    >>> from evo.compute.tasks.geostatistics.simulation_report import (
    ...     SimulationReportParameters,
    ...     SimulationReportBlockDiscretization,
    ... )
    >>>
    >>> params = SimulationReportParameters(
    ...     simulation_source=pointset,
    ...     source_attribute=pointset.attributes["grade"],
    ...     simulation_target=grid,
    ...     point_simulations=grid.attributes["point_sims"],
    ...     point_simulations_normal_score=grid.attributes["point_sims_ns"],
    ...     block_simulations=grid.attributes["block_sims"],
    ...     block_simulations_normal_score=grid.attributes["block_sims_ns"],
    ...     variogram_model=variogram_url,
    ...     neighborhood=SearchNeighborhood(...),
    ...     block_discretization=SimulationReportBlockDiscretization(nx=2, ny=2, nz=2),
    ...     number_of_simulations=50,
    ...     random_seed=38239342,
    ... )
    >>> result = await run(manager, params)
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar, Literal

from evo.common import IContext
from evo.objects.typed import BlockModel, PointSet, Regular3DGrid, RegularMasked3DGrid, Variogram
from pydantic import (
    BaseModel,
    Field,
    PrivateAttr,
    StrictInt,
    ValidatorFunctionWrapHandler,
    field_validator,
    model_validator,
)

from ..common import (
    ExistingAttributeExpression,
    GeoscienceObjectReference,
    OutputTarget,
    SearchNeighborhood,
    TaskInputModel,
    TaskParameters,
    object_identity,
    object_matches_reference,
)
from ..common.runner import TaskRunner
from ..common.search import _reject_sector_and_drillhole_limits
from ..common.source_target import AnyTypedAttribute
from .conditioned_simulator import TailExtrapolationParams

__all__ = [
    "SimReportContext",
    "SimReportContextItem",
    "SimReportLinks",
    "SimReportThresholds",
    "SimReportValidationReport",
    "SimReportValidationSummary",
    "SimulationReportBlockDiscretization",
    "SimulationReportDistribution",
    "SimulationReportParameters",
    "SimulationReportResult",
    "SimulationReportResultModel",
    "SimulationReportRunner",
]


# =============================================================================
# Supporting Types
# =============================================================================


class SimulationReportBlockDiscretization(TaskInputModel):
    """Sub-block discretization for support correction.

    Controls how each block is subdivided for point-to-block variance correction.
    Each dimension must be between 1 and 9 inclusive.
    """

    nx: StrictInt = Field(1, ge=1, le=9)
    """Number of sub-blocks along the X axis."""

    ny: StrictInt = Field(1, ge=1, le=9)
    """Number of sub-blocks along the Y axis."""

    nz: StrictInt = Field(1, ge=1, le=9)
    """Number of sub-blocks along the Z axis."""


class SimulationReportDistribution(TaskInputModel):
    """Optional distribution settings for the simulation report.

    When provided, the report includes distribution-related validation.
    """

    _weights_object_url: str | None = PrivateAttr(default=None)
    _weights_parent: Any = PrivateAttr(default=None)

    tail_extrapolation: TailExtrapolationParams | None = None
    """Optional tail extrapolation parameters."""

    weights: ExistingAttributeExpression | None = None
    """Optional sample weights. Accepts a typed attribute or an attribute expression."""

    @model_validator(mode="wrap")
    @classmethod
    def _remember_weights_object(cls, data: Any, handler: ValidatorFunctionWrapHandler) -> SimulationReportDistribution:
        weights_url: str | None = None
        weights_parent: Any = None
        if isinstance(data, SimulationReportDistribution):
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


class SimReportContextItem(TaskInputModel):
    """Single key-value detail for a simulation report."""

    label: str
    """Display label for this detail."""

    value: str
    """Display value for this detail."""


class SimReportContext(TaskInputModel):
    """Report context metadata.

    Adds a title and descriptive details to the generated validation report.
    """

    title: str
    """Title for the validation report."""

    details: list[SimReportContextItem] = Field(default_factory=list, max_length=10)
    """Key-value details to include in the report header."""


class SimReportThresholds(TaskInputModel):
    """Acceptable/marginal thresholds for mean validation.

    Used to classify the difference between reference mean and simulation mean.
    """

    acceptable: float = Field(ge=0.0)
    """Maximum difference considered acceptable."""

    marginal: float = Field(ge=0.0)
    """Maximum difference considered marginal (between acceptable and failing)."""

    @model_validator(mode="after")
    def _validate_threshold_order(self) -> SimReportThresholds:
        if self.marginal < self.acceptable:
            raise ValueError(
                f"marginal threshold ({self.marginal}) must be greater than or equal to "
                f"acceptable threshold ({self.acceptable})."
            )
        return self


# =============================================================================
# Simulation Report Parameters
# =============================================================================


class SimulationReportParameters(TaskParameters):
    """Parameters for the simulation-report compute task.

    Simulation data is referenced by flat object/attribute field pairs rather than
    nested Source/Target objects. Attribute fields accept a typed attribute such as
    ``pointset.attributes["grade"]`` or a raw attribute expression.

    Example::

        >>> params = SimulationReportParameters(
        ...     simulation_source=pointset,
        ...     source_attribute=pointset.attributes["grade"],
        ...     simulation_target=grid,
        ...     point_simulations=grid.attributes["point_sims"],
        ...     point_simulations_normal_score=grid.attributes["point_sims_ns"],
        ...     block_simulations=grid.attributes["block_sims"],
        ...     block_simulations_normal_score=grid.attributes["block_sims_ns"],
        ...     variogram_model=variogram_url,
        ...     neighborhood=SearchNeighborhood(...),
        ...     number_of_simulations=50,
        ...     random_seed=38239342,
        ... )
    """

    _object_roles: ClassVar[Mapping[str, tuple[type, ...]]] = {
        "simulation_source": (PointSet,),
        "source_attribute": (PointSet,),
        "simulation_target": (Regular3DGrid, RegularMasked3DGrid, BlockModel),
        "point_simulations": (Regular3DGrid, RegularMasked3DGrid, BlockModel),
        "point_simulations_normal_score": (Regular3DGrid, RegularMasked3DGrid, BlockModel),
        "block_simulations": (Regular3DGrid, RegularMasked3DGrid, BlockModel),
        "block_simulations_normal_score": (Regular3DGrid, RegularMasked3DGrid, BlockModel),
        "variogram_model": (Variogram,),
    }
    _attribute_object_bindings: ClassVar[tuple[tuple[str, str], ...]] = (
        ("source_attribute", "simulation_source"),
        ("point_simulations", "simulation_target"),
        ("point_simulations_normal_score", "simulation_target"),
        ("block_simulations", "simulation_target"),
        ("block_simulations_normal_score", "simulation_target"),
    )

    simulation_source: GeoscienceObjectReference
    """Reference URL to the source geoscience object (typically a pointset)."""

    source_attribute: ExistingAttributeExpression
    """The source values. Accepts a typed attribute such as ``pointset.attributes["grade"]``,
    or an attribute expression such as ``"locations.attributes[0]"``."""

    simulation_target: GeoscienceObjectReference
    """Reference URL to the target geoscience object (typically a block model)."""

    point_simulations: ExistingAttributeExpression
    """Point simulations on the target. Accepts a typed attribute or an attribute expression."""

    point_simulations_normal_score: ExistingAttributeExpression
    """Point simulation normal scores on the target. Accepts a typed attribute or an attribute expression."""

    block_simulations: ExistingAttributeExpression
    """Block simulations on the target. Accepts a typed attribute or an attribute expression."""

    block_simulations_normal_score: ExistingAttributeExpression
    """Block simulation normal scores on the target. Accepts a typed attribute or an attribute expression."""

    variogram_model: GeoscienceObjectReference
    """Reference URL to the variogram model object."""

    neighborhood: SearchNeighborhood
    """Search neighborhood parameters for kriging within the report."""

    block_discretization: SimulationReportBlockDiscretization = Field(
        default_factory=SimulationReportBlockDiscretization
    )
    """Sub-block discretization for support correction."""

    distribution: SimulationReportDistribution | None = None
    """Optional distribution settings."""

    number_of_simulations: int = Field(ge=1, le=100)
    """Number of simulation realizations to validate."""

    random_seed: int = 38239342
    """Seed for the random number generator."""

    kriging_method: Literal["simple", "ordinary"] = "simple"
    """Kriging method used in the simulation."""

    number_of_lines: int = Field(500, ge=1, le=1000)
    """Number of turning-band lines."""

    report_context: SimReportContext | None = None
    """Optional report context metadata."""

    report_mean_thresholds: SimReportThresholds | None = None
    """Optional thresholds for mean comparison validation."""

    @field_validator("neighborhood")
    @classmethod
    def _check_neighborhood(cls, neighborhood: SearchNeighborhood) -> SearchNeighborhood:
        return _reject_sector_and_drillhole_limits(neighborhood)

    @model_validator(mode="after")
    def _validate_distribution_weights_object(self) -> SimulationReportParameters:
        if self.distribution is not None and self.distribution._weights_parent is not None:
            source_obj = self._typed_objects.get(object_identity(self.simulation_source))
            if not object_matches_reference(
                self.distribution._weights_parent,
                self.simulation_source,
                supplied_obj=source_obj,
            ):
                actual = (
                    object_identity(self.distribution._weights_object_url)
                    if self.distribution._weights_object_url
                    else "unknown"
                )
                raise ValueError(
                    f"distribution.weights belongs to object {actual!r}, "
                    f"which does not match 'simulation_source' ({object_identity(self.simulation_source)!r})."
                )
        return self

    def output_targets(self) -> list[OutputTarget]:
        return []


# =============================================================================
# Simulation Report Result Types
# =============================================================================


class SimReportValidationSummary(BaseModel):
    """Validation summary from the simulation report."""

    reference_mean: float
    """Mean of the reference (source) data."""

    mean: float
    """Mean of the simulated data."""


class SimReportValidationReport(BaseModel):
    """Reference to the generated validation report file."""

    reference: str
    """URL to the generated report file."""


class SimReportLinks(BaseModel):
    """Links associated with the simulation report result."""

    dashboard: str | None = None
    """URL to an interactive dashboard (if generated)."""


class SimulationReportResultModel(BaseModel):
    """Pydantic model for the raw simulation-report task result."""

    validation_summary: SimReportValidationSummary | None = None
    """Summary comparing reference and simulated means."""

    validation_report: SimReportValidationReport | None = None
    """Reference to the generated validation report."""

    links: SimReportLinks | None = None
    """Links to additional resources like dashboards."""


class SimulationReportResult:
    """Result of a simulation-report task.

    Provides access to validation summary, report reference, and dashboard link.
    """

    TASK_DISPLAY_NAME: ClassVar[str] = "Simulation Report"

    def __init__(self, context: IContext, model: SimulationReportResultModel) -> None:
        self._validation_summary = model.validation_summary
        self._validation_report = model.validation_report
        self._links = model.links
        self._context = context

    @property
    def validation_summary(self) -> SimReportValidationSummary | None:
        """Validation summary comparing reference and simulated means."""
        return self._validation_summary

    @property
    def validation_report(self) -> SimReportValidationReport | None:
        """Validation report metadata, if available."""
        return self._validation_report

    @property
    def report_reference(self) -> str | None:
        """URL to the generated validation report file."""
        if self._validation_report:
            return self._validation_report.reference
        return None

    @property
    def links(self) -> SimReportLinks | None:
        """Links associated with the simulation report result."""
        return self._links

    @property
    def dashboard_url(self) -> str | None:
        """URL to the interactive dashboard (if available)."""
        if self._links:
            return self._links.dashboard
        return None

    def __str__(self) -> str:
        """String representation."""
        lines = [f"✓ {self.TASK_DISPLAY_NAME} Result"]
        if self._validation_summary:
            lines.append(f"  Ref mean:   {self._validation_summary.reference_mean:.4f}")
            lines.append(f"  Sim mean:   {self._validation_summary.mean:.4f}")
        if self._validation_report:
            lines.append(f"  Report:     {self._validation_report.reference}")
        if self._links and self._links.dashboard:
            lines.append(f"  Dashboard:  {self._links.dashboard}")
        return "\n".join(lines)


# =============================================================================
# Task Runner
# =============================================================================


class SimulationReportRunner(
    TaskRunner[SimulationReportParameters, SimulationReportResultModel, SimulationReportResult],
    topic="geostatistics",
    task="simulation-report",
):
    """Runner for simulation-report compute tasks.

    Automatically registered — used by ``run()`` for dispatch, or directly::

        result = await SimulationReportRunner(context, params)
    """

    async def _get_result(self, raw_result: SimulationReportResultModel) -> SimulationReportResult:
        return SimulationReportResult(self._context, raw_result)
