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

Example:
    >>> from evo.compute.tasks import run, SearchNeighborhood
    >>> from evo.compute.tasks.simulation_report import (
    ...     SimulationReportParameters,
    ...     SimulationReportBlockDiscretization,
    ... )
    >>>
    >>> params = SimulationReportParameters(
    ...     simulation_source=pointset_url,
    ...     source_attribute="locations.attributes[0]",
    ...     simulation_target=grid_url,
    ...     point_simulations="cell_attributes[0]",
    ...     point_simulations_normal_score="cell_attributes[1]",
    ...     block_simulations="cell_attributes[2]",
    ...     block_simulations_normal_score="cell_attributes[3]",
    ...     variogram_model=variogram_url,
    ...     neighborhood=SearchNeighborhood(...),
    ...     block_discretization=SimulationReportBlockDiscretization(nx=2, ny=2, nz=2),
    ...     number_of_simulations=50,
    ...     random_seed=38239342,
    ... )
    >>> result = await run(manager, params)
"""

from __future__ import annotations

from typing import ClassVar, Literal

from evo.common import IContext
from pydantic import BaseModel, Field

from .common import (
    GeoscienceObjectReference,
    SearchNeighborhood,
)
from .common.runner import TaskRunner

__all__ = [
    "SimReportContext",
    "SimReportContextItem",
    "SimReportThresholds",
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


class SimulationReportBlockDiscretization(BaseModel):
    """Sub-block discretization for support correction.

    Controls how each block is subdivided for point-to-block variance correction.
    Each dimension must be between 1 and 9 inclusive.
    """

    nx: int = Field(1, ge=1, le=9)
    """Number of sub-blocks along the X axis."""

    ny: int = Field(1, ge=1, le=9)
    """Number of sub-blocks along the Y axis."""

    nz: int = Field(1, ge=1, le=9)
    """Number of sub-blocks along the Z axis."""


class SimulationReportDistribution(BaseModel):
    """Optional distribution settings for the simulation report.

    When provided, the report includes distribution-related validation.
    """

    tail_extrapolation: None = None
    """Reserved for future tail-extrapolation options."""

    weights: str | None = None
    """Optional attribute expression for sample weights."""


class SimReportContextItem(BaseModel):
    """Single key-value detail for a simulation report."""

    label: str
    """Display label for this detail."""

    value: str
    """Display value for this detail."""


class SimReportContext(BaseModel):
    """Report context metadata.

    Adds a title and descriptive details to the generated validation report.
    """

    title: str
    """Title for the validation report."""

    details: list[SimReportContextItem] = Field(default_factory=list)
    """Key-value details to include in the report header."""


class SimReportThresholds(BaseModel):
    """Acceptable/marginal thresholds for mean validation.

    Used to classify the difference between reference mean and simulation mean.
    """

    acceptable: float = Field(ge=0.0)
    """Maximum difference considered acceptable."""

    marginal: float = Field(ge=0.0)
    """Maximum difference considered marginal (between acceptable and failing)."""


# =============================================================================
# Simulation Report Parameters
# =============================================================================


class SimulationReportParameters(BaseModel):
    """Parameters for the simulation-report compute task.

    All simulation data references use flat string fields (object URLs and
    attribute expressions) rather than nested Source/Target objects.

    Example:
        >>> params = SimulationReportParameters(
        ...     simulation_source=pointset_url,
        ...     source_attribute="locations.attributes[0]",
        ...     simulation_target=grid_url,
        ...     point_simulations="cell_attributes[0]",
        ...     point_simulations_normal_score="cell_attributes[1]",
        ...     block_simulations="cell_attributes[2]",
        ...     block_simulations_normal_score="cell_attributes[3]",
        ...     variogram_model=variogram_url,
        ...     neighborhood=SearchNeighborhood(...),
        ...     number_of_simulations=50,
        ...     random_seed=38239342,
        ... )
    """

    simulation_source: GeoscienceObjectReference
    """Reference URL to the source geoscience object (typically a pointset)."""

    source_attribute: str
    """Attribute expression on the source object (e.g. ``"locations.attributes[0]"``)."""

    simulation_target: GeoscienceObjectReference
    """Reference URL to the target geoscience object (typically a block model)."""

    point_simulations: str
    """Attribute expression for point simulations on the target."""

    point_simulations_normal_score: str
    """Attribute expression for point simulation normal scores on the target."""

    block_simulations: str
    """Attribute expression for block simulations on the target."""

    block_simulations_normal_score: str
    """Attribute expression for block simulation normal scores on the target."""

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

    number_of_simulations: int = Field(ge=1)
    """Number of simulation realizations to validate."""

    random_seed: int = 38239342
    """Seed for the random number generator."""

    kriging_method: Literal["simple", "ordinary"] = "simple"
    """Kriging method used in the simulation."""

    number_of_lines: int = Field(500, ge=1)
    """Number of turning-band lines."""

    report_context: SimReportContext | None = None
    """Optional report context metadata."""

    report_mean_thresholds: SimReportThresholds | None = None
    """Optional thresholds for mean comparison validation."""


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
    def report_reference(self) -> str | None:
        """URL to the generated validation report file."""
        if self._validation_report:
            return self._validation_report.reference
        return None

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
