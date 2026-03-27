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

"""
Conditional simulation compute task client.

This module provides typed dataclass models and convenience functions for running
the conditional simulation task (geostatistics/consim).

Example:
    >>> from evo.compute.tasks import run, SearchNeighborhood, Ellipsoid, EllipsoidRanges
    >>> from evo.compute.tasks.simulation import ConsimParameters, Distribution
    >>>
    >>> params = ConsimParameters(
    ...     source=pointset.attributes["grade"],
    ...     target_object=grid,
    ...     variogram=variogram,
    ...     search=SearchNeighborhood(
    ...         ellipsoid=Ellipsoid(ranges=EllipsoidRanges(200, 150, 100)),
    ...         max_samples=20,
    ...     ),
    ...     distribution=Distribution(
    ...         tail_extrapolation=TailExtrapolation(
    ...             upper=UpperTail(power=0.5, max=10.0),
    ...         ),
    ...     ),
    ...     number_of_simulations=100,
    ... )
    >>> result = await run(manager, params, preview=True)
"""

from __future__ import annotations

from typing import Any, ClassVar, Literal, Protocol, runtime_checkable

import pandas as pd
from evo.common import IContext, IFeedback
from evo.objects import ObjectSchema
from evo.objects.typed import BaseObject, object_from_reference
from pydantic import BaseModel, Field, SerializerFunctionWrapHandler, model_serializer

# Import shared components
from .common import (
    AnySourceAttribute,
    CreateAttribute,
    GeoscienceObjectReference,
    SearchNeighborhood,
    UpdateAttribute,
)
from .common.results import TaskAttribute
from .common.runner import TaskRunner

# Import BlockDiscretisation from kriging to reuse
# This avoids duplication while keeping the type available
from .kriging import BlockDiscretisation

__all__ = [
    # Simulation-specific exports
    "ConsimParameters",
    "ConsimResult",
    "ConsimResultModel",
    "ConsimRunner",
    "ConsimTarget",
    "Distribution",
    "LossCalculation",
    "LowerTail",
    "MaterialCategory",
    "QuantileAttribute",
    "ReportMeanThresholds",
    "SummaryAttributes",
    "TailExtrapolation",
    "UpperTail",
    "ValidationReportContext",
    "ValidationReportContextItem",
    "ValidationSummary",
]


# =============================================================================
# Distribution Types
# =============================================================================


class UpperTail(BaseModel):
    """Power model for extrapolating the upper tail of the distribution.

    Example:
        >>> upper = UpperTail(power=0.5, max=10.0)
    """

    power: float = Field(gt=0, le=1)
    """Denominator of the exponent for the power model (between 0 exclusive and 1 inclusive)."""

    max: float
    """Maximum extent of tail, must be greater than the maximum value in the data."""


class LowerTail(BaseModel):
    """Power model for extrapolating the lower tail of the distribution.

    Example:
        >>> lower = LowerTail(power=0.5, min=0.0)
    """

    power: float = Field(gt=0, le=1)
    """Denominator of the exponent for the power model (between 0 exclusive and 1 inclusive)."""

    min: float
    """Minimum extent of tail, must be less than the minimum value in the data."""


class TailExtrapolation(BaseModel):
    """Tail extrapolation configuration for the distribution.

    Example:
        >>> extrapolation = TailExtrapolation(
        ...     upper=UpperTail(power=0.5, max=10.0),
        ...     lower=LowerTail(power=0.5, min=0.0),
        ... )
    """

    upper: UpperTail
    """Power model for extrapolating the upper tail of the distribution."""

    lower: LowerTail | None = None
    """Power model for extrapolating the lower tail of the distribution. Optional."""


class Distribution(BaseModel):
    """Parameters for a continuous distribution used for normal-score transformation.

    Example:
        >>> dist = Distribution(
        ...     tail_extrapolation=TailExtrapolation(
        ...         upper=UpperTail(power=0.5, max=10.0),
        ...     ),
        ...     weights="weight_attribute",
        ... )
    """

    tail_extrapolation: TailExtrapolation
    """Tail extrapolation configuration."""

    weights: str | None = None
    """Optional weights attribute for weighted distribution."""


# =============================================================================
# Loss Calculation Types
# =============================================================================


class MaterialCategory(BaseModel):
    """Material category for loss calculations.

    Ordered list of material categories, from lowest to highest cutoff grade.

    Example:
        >>> category = MaterialCategory(
        ...     label="Ore",
        ...     cutoff_grade=0.5,
        ...     metal_price=1000.0,
        ...     processing_cost=10.0,
        ...     mining_waste_cost=2.0,
        ...     mining_ore_cost=5.0,
        ...     metal_recovery_fraction=0.9,
        ... )
    """

    cutoff_grade: float | None = None
    """Cutoff grade for the category."""

    metal_price: float | None = None
    """Price of the metal per unit."""

    label: str | None = None
    """Label for the material category."""

    processing_cost: float
    """Processing cost per unit of material."""

    mining_waste_cost: float
    """Mining cost per unit of waste material."""

    mining_ore_cost: float
    """Mining cost per unit of ore material."""

    metal_recovery_fraction: float
    """Fraction of metal expected to be recovered during processing."""


class LossCalculation(BaseModel):
    """Settings for loss calculations.

    Example:
        >>> loss = LossCalculation(
        ...     material_categories=[
        ...         MaterialCategory(
        ...             label="Waste",
        ...             processing_cost=0.0,
        ...             mining_waste_cost=2.0,
        ...             mining_ore_cost=0.0,
        ...             metal_recovery_fraction=0.0,
        ...         ),
        ...         MaterialCategory(
        ...             label="Ore",
        ...             cutoff_grade=0.5,
        ...             metal_price=1000.0,
        ...             processing_cost=10.0,
        ...             mining_waste_cost=0.0,
        ...             mining_ore_cost=5.0,
        ...             metal_recovery_fraction=0.9,
        ...         ),
        ...     ],
        ...     target_attribute=CreateAttribute(name="loss_results"),
        ... )
    """

    material_categories: list[MaterialCategory]
    """Ordered list of material categories, from lowest to highest cutoff grade."""

    target_attribute: CreateAttribute | UpdateAttribute
    """The target attribute that will be created or updated with loss calculation results."""


# =============================================================================
# Validation Report Types
# =============================================================================


class ValidationReportContextItem(BaseModel):
    """Key-value pair for validation report context."""

    label: str
    """Label of the context item, which will appear on the report."""

    value: str
    """Value of the context item."""


class ValidationReportContext(BaseModel):
    """Context for the validation report.

    Example:
        >>> context = ValidationReportContext(
        ...     title="Copper Grade Simulation Validation",
        ...     details=[
        ...         ValidationReportContextItem(label="Domain", value="LMS1"),
        ...         ValidationReportContextItem(label="Variable", value="CU_pct"),
        ...     ],
        ... )
    """

    title: str
    """Title that will appear on the report."""

    details: list[ValidationReportContextItem]
    """Key-value pairs of details to appear on the report."""


class ReportMeanThresholds(BaseModel):
    """Thresholds for the mean comparison in the validation report.

    Example:
        >>> thresholds = ReportMeanThresholds(acceptable=5.0, marginal=10.0)
    """

    acceptable: float
    """Threshold for the percentage difference between the reference and simulated means that is considered acceptable."""

    marginal: float
    """Threshold for the percentage difference between the reference and simulated means that is considered marginal."""


# =============================================================================
# Consim Parameters
# =============================================================================


class ConsimParameters(BaseModel):
    """Parameters for the conditional simulation task.

    Defines all inputs needed to run a conditional simulation (turning bands) task.

    Example:
        >>> from evo.compute.tasks import run, SearchNeighborhood, Ellipsoid, EllipsoidRanges
        >>> from evo.compute.tasks.simulation import ConsimParameters, Distribution, TailExtrapolation, UpperTail
        >>>
        >>> params = ConsimParameters(
        ...     source=pointset.attributes["grade"],  # Source attribute
        ...     target_object=grid,  # Target grid object
        ...     variogram=variogram,  # Variogram model
        ...     search=SearchNeighborhood(
        ...         ellipsoid=Ellipsoid(ranges=EllipsoidRanges(200, 150, 100)),
        ...         max_samples=20,
        ...     ),
        ...     distribution=Distribution(
        ...         tail_extrapolation=TailExtrapolation(
        ...             upper=UpperTail(power=0.5, max=10.0),
        ...         ),
        ...     ),
        ...     number_of_simulations=100,
        ...     location_wise_quantiles=[0.1, 0.5, 0.9],
        ... )
        >>>
        >>> result = await run(manager, params, preview=True)
    """

    model_config = {"populate_by_name": True}

    # Source specification
    source: AnySourceAttribute
    """The source object and attribute containing known values for conditioning."""

    # Target specification
    target_object: GeoscienceObjectReference
    """Reference to the target grid object. New attributes will be created on this object.
    Supported schemas: regular-3d-grid, regular-masked-3d-grid."""

    # Variogram
    variogram: GeoscienceObjectReference = Field(alias="variogram_model")
    """Reference to the variogram model used to model the covariance within the domain.
    Supported schemas: variogram/[>=1.1,<2]."""

    # Search neighborhood
    search: SearchNeighborhood = Field(alias="neighborhood")
    """Search neighborhood for conditioning."""

    # Distribution (required for normal-score transformation)
    distribution: Distribution | None = None
    """Parameters for the continuous distribution used for normal-score transformation."""

    # Kriging method for conditioning
    kriging_method: Literal["simple", "ordinary"] = "simple"
    """The kriging method to use for the conditioning step. Defaults to 'simple'."""

    # Block discretization (required by API)
    block_discretisation: "BlockDiscretisation" = Field(
        default_factory=lambda: BlockDiscretisation(nx=3, ny=3, nz=3),
        alias="block_discretization",
    )
    """Sub-block discretization used for support correction within each grid block.
    Defaults to BlockDiscretisation(nx=3, ny=3, nz=3) for block kriging."""

    # Simulation parameters
    number_of_lines: int = 500
    """Number of lines to use for the turning-band simulation. Defaults to 500."""

    number_of_simulations: int = 1
    """Number of simulations to run. Defaults to 1."""

    random_seed: int = 38239342
    """Random seed for simulation and tie-breaking. Defaults to 38239342."""

    number_of_simulations_to_save: int = 5
    """Number of simulations to publish to the output object.
    If perform_validation is true, the minimum value must be at least 1. Defaults to 5."""

    # Output options
    location_wise_quantiles: list[float] | None = None
    """List of quantiles to compute for each location (e.g., [0.1, 0.5, 0.9])."""

    # Validation
    perform_validation: bool = False
    """Whether to run validation and generate a validation report. Defaults to False."""

    report_context: ValidationReportContext | None = None
    """Context for the validation report."""

    report_mean_thresholds: ReportMeanThresholds | None = None
    """Thresholds for the mean comparison in the validation report."""

    # Loss calculation
    loss_calculation: LossCalculation | None = None
    """Settings for loss calculations. Optional."""

    @model_serializer(mode="wrap")
    def _serialize(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        result = handler(self)

        # The API expects source_object and source_attribute as separate fields
        # Convert from our unified source representation
        if "source" in result:
            source = result.pop("source")
            result["source_object"] = source["object"]
            result["source_attribute"] = source["attribute"]

        return result


# =============================================================================
# Consim Result Types
# =============================================================================


class SummaryAttributes(BaseModel):
    """Summary statistics attributes from simulation results."""

    mean: TaskAttribute
    """Attribute containing the mean of the simulations."""

    variance: TaskAttribute
    """Attribute containing the variance of the simulations."""

    min: TaskAttribute
    """Attribute containing the minimum of the simulations."""

    max: TaskAttribute
    """Attribute containing the maximum of the simulations."""


class QuantileAttribute(BaseModel):
    """Quantile attribute from simulation results."""

    reference: str
    """Reference to the attribute in the geoscience object."""

    name: str
    """The name of the output attribute."""

    quantile: float
    """Quantile of the simulations that this attribute represents."""


class ValidationSummary(BaseModel):
    """Summary of validation results."""

    reference_mean: float
    """Mean of the input values."""

    mean: float
    """Mean of the simulated values."""


class FileReference(BaseModel):
    """Reference to a file."""

    reference: str
    """Reference to the file."""

    name: str
    """The name of the file, typically the terminal segment of the path."""


class ConsimLinks(BaseModel):
    """Links related to the simulation task."""

    dashboard: str | None = None
    """Link to the validation dashboard."""


class ConsimTarget(BaseModel):
    """Target information from a conditional simulation result.

    Contains references to all output attributes created by the simulation.
    """

    reference: str
    """Reference to the target geoscience object."""

    name: str
    """The name of the geoscience object."""

    description: str | None = None
    """The description of the geoscience object."""

    schema_id: str
    """The ID of the Geoscience Object schema."""

    summary_attributes: SummaryAttributes
    """Attributes containing summary statistics (mean, variance, min, max)."""

    quantile_attributes: list[QuantileAttribute]
    """List of attributes containing quantiles of the simulation results."""

    simulations: TaskAttribute | None = None
    """Attribute containing the simulation results."""

    simulations_normal_score: TaskAttribute | None = None
    """Attribute containing the simulation results in normal score space."""

    point_simulations: TaskAttribute | None = None
    """Attribute containing the point simulation results."""

    point_simulations_normal_score: TaskAttribute | None = None
    """Attribute containing the point simulation results in normal score space."""

    loss_calculation_attribute: TaskAttribute | None = None
    """Attribute containing the loss calculation results."""

    validation_summary: ValidationSummary | None = None
    """Summary of the validation results."""

    validation_report: FileReference | None = None
    """Link to the validation report file."""


class ConsimResultModel(BaseModel):
    """Raw API response model for conditional simulation.

    This model matches the exact structure returned by the API.
    """

    target: ConsimTarget
    """The target grid with simulation results."""

    links: ConsimLinks
    """Links related to the simulation task."""


# =============================================================================
# Result Wrapper
# =============================================================================


@runtime_checkable
class _ObjToDataframeProtocol(Protocol):
    """Protocol for objects that can convert themselves to a DataFrame."""

    async def to_dataframe(self, *keys: str, fb: IFeedback = ...) -> pd.DataFrame: ...


class ConsimResult:
    """Result wrapper for conditional simulation tasks.

    Provides convenient access to simulation outputs including summary statistics,
    quantiles, individual simulations, and validation results.

    Example:
        >>> result = await run(manager, params, preview=True)
        >>> print(result.message)
        >>> print(result.mean_attribute_name)
        >>> df = await result.to_dataframe("mean", "variance")
        >>> target = await result.get_target_object()
    """

    TASK_DISPLAY_NAME: ClassVar[str] = "Conditional Simulation"

    def __init__(self, context: IContext, model: ConsimResultModel) -> None:
        self._target = model.target
        self._links = model.links
        self._context = context

    @property
    def target_name(self) -> str:
        """The name of the target object."""
        return self._target.name

    @property
    def target_reference(self) -> str:
        """Reference URL to the target object."""
        return self._target.reference

    @property
    def schema(self) -> ObjectSchema:
        """The schema type of the target object."""
        return ObjectSchema.from_id(self._target.schema_id)

    # Summary attribute accessors
    @property
    def mean_attribute(self) -> TaskAttribute:
        """Attribute containing the mean of simulations."""
        return self._target.summary_attributes.mean

    @property
    def variance_attribute(self) -> TaskAttribute:
        """Attribute containing the variance of simulations."""
        return self._target.summary_attributes.variance

    @property
    def min_attribute(self) -> TaskAttribute:
        """Attribute containing the minimum of simulations."""
        return self._target.summary_attributes.min

    @property
    def max_attribute(self) -> TaskAttribute:
        """Attribute containing the maximum of simulations."""
        return self._target.summary_attributes.max

    @property
    def quantile_attributes(self) -> list[QuantileAttribute]:
        """List of quantile attributes with their quantile values."""
        return self._target.quantile_attributes

    @property
    def simulations_attribute(self) -> TaskAttribute | None:
        """Attribute containing simulation realizations, if saved."""
        return self._target.simulations

    @property
    def loss_calculation_attribute(self) -> TaskAttribute | None:
        """Attribute containing loss calculation results, if computed."""
        return self._target.loss_calculation_attribute

    @property
    def validation_summary(self) -> ValidationSummary | None:
        """Summary of validation results, if validation was performed."""
        return self._target.validation_summary

    @property
    def validation_report(self) -> FileReference | None:
        """Reference to the validation report file, if generated."""
        return self._target.validation_report

    @property
    def dashboard_link(self) -> str | None:
        """Link to the validation dashboard, if available."""
        return self._links.dashboard

    async def get_target_object(self) -> BaseObject:
        """Load and return the target geoscience object.

        Returns:
            The typed geoscience object (e.g., Regular3DGrid, RegularMasked3DGrid)

        Example:
            >>> result = await run(manager, params)
            >>> target = await result.get_target_object()
        """
        return await object_from_reference(self._context, self._target.reference)

    async def to_dataframe(self, *columns: str) -> pd.DataFrame:
        """Get the task results as a DataFrame.

        Args:
            columns: Optional list of column names to include. If empty, includes
                    all columns. Common columns include 'mean', 'variance', 'min', 'max',
                    and any quantile attributes.

        Returns:
            A pandas DataFrame containing the simulation results.

        Example:
            >>> result = await run(manager, params)
            >>> df = await result.to_dataframe("mean", "variance")
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
            f"  Target:    {self.target_name}",
            f"  Mean:      {self.mean_attribute.name}",
            f"  Variance:  {self.variance_attribute.name}",
            f"  Quantiles: {len(self.quantile_attributes)} attributes",
        ]
        if self.simulations_attribute:
            lines.append(f"  Simulations: {self.simulations_attribute.name}")
        if self.validation_summary:
            lines.append(
                f"  Validation: ref_mean={self.validation_summary.reference_mean:.4f}, sim_mean={self.validation_summary.mean:.4f}"
            )
        if self.dashboard_link:
            lines.append(f"  Dashboard: {self.dashboard_link}")
        return "\n".join(lines)


# =============================================================================
# Task Runner
# =============================================================================


class ConsimRunner(
    TaskRunner[ConsimParameters, ConsimResultModel, ConsimResult],
    topic="geostatistics",
    task="consim",
):
    """Runner for conditional simulation compute tasks.

    Automatically registered — used by ``run()`` for dispatch, or directly::

        result = await ConsimRunner(context, params, preview=True)
    """

    async def _get_result(self, raw_result: ConsimResultModel) -> ConsimResult:
        return ConsimResult(self._context, raw_result)
