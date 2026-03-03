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
Kriging compute task client.

This module provides typed dataclass models and convenience functions for running
the Kriging task (geostatistics/kriging).

Example:
    >>> from evo.compute.tasks import run, SearchNeighborhood, Ellipsoid, EllipsoidRanges
    >>> from evo.compute.tasks.kriging import KrigingParameters
    >>>
    >>> params = KrigingParameters(
    ...     source=pointset.attributes["grade"],
    ...     target=Target.new_attribute(block_model, "kriged_grade"),
    ...     variogram=variogram,
    ...     search=SearchNeighborhood(
    ...         ellipsoid=Ellipsoid(ranges=EllipsoidRanges(200, 150, 100)),
    ...         max_samples=20,
    ...     ),
    ... )
    >>> result = await run(manager, params, preview=True)
"""

from __future__ import annotations

from typing import Any, Literal

from evo.objects.typed.attributes import (
    Attribute,
    BlockModelAttribute,
    BlockModelPendingAttribute,
    PendingAttribute,
)
from pydantic import BaseModel, field_validator, model_serializer

# Import shared components
from .common import (
    GeoscienceObjectReference,
    SearchNeighborhood,
    Source,
    Target,
    get_attribute_expression,
    source_from_attribute,
    target_from_attribute,
)
from .common.results import TaskAttribute, TaskResult, TaskResults, TaskTarget
from .common.runner import TaskRunner

__all__ = [
    # Kriging-specific (users import from evo.compute.tasks.kriging)
    "BlockDiscretisation",
    "KrigingMethod",
    "KrigingParameters",
    "KrigingResult",
    "KrigingRunner",
    "OrdinaryKriging",
    "RegionFilter",
    "SimpleKriging",
    # Re-exported from common for backwards compatibility
    "TaskResult",
    "TaskResults",
]


# Backwards-compatible aliases for the renamed internal dataclasses.
_KrigingAttribute = TaskAttribute
_KrigingTarget = TaskTarget


# =============================================================================
# Kriging Method Types
# =============================================================================


class SimpleKriging(BaseModel):
    """Simple kriging method with a known constant mean.

    Use when the mean of the variable is known and constant across the domain.

    Example:
        >>> method = SimpleKriging(mean=100.0)
    """

    type: Literal["simple"] = "simple"
    """The method type discriminator."""

    mean: float
    """The mean value, assumed to be constant across the domain."""


class OrdinaryKriging(BaseModel):
    """Ordinary kriging method with unknown local mean.

    The most common kriging method. Estimates the local mean from nearby samples.
    This is the default kriging method if none is specified.
    """

    type: Literal["ordinary"] = "ordinary"
    """The method type discriminator."""


class KrigingMethod:
    """Factory for kriging methods.

    Provides convenient access to kriging method types.

    Example:
        >>> # Use ordinary kriging (most common)
        >>> method = KrigingMethod.ORDINARY
        >>>
        >>> # Use simple kriging with known mean
        >>> method = KrigingMethod.simple(mean=100.0)
    """

    ORDINARY: OrdinaryKriging = OrdinaryKriging()
    """Ordinary kriging - estimates local mean from nearby samples."""

    @staticmethod
    def simple(mean: float) -> SimpleKriging:
        """Create a simple kriging method with the given mean.

        Args:
            mean: The known constant mean value across the domain.

        Returns:
            SimpleKriging instance configured with the given mean.
        """
        return SimpleKriging(mean=mean)


# =============================================================================
# Block Discretisation
# =============================================================================


class BlockDiscretisation(BaseModel):
    """Sub-block discretisation for block kriging.

    When provided, each target block is subdivided into ``nx * ny * nz``
    sub-cells and the kriged value is averaged across these sub-cells.
    When omitted (``None``), point kriging is performed.

    Only applicable when the target is a 3D grid or block model.

    Each dimension must be an integer between 1 and 9 (inclusive).
    The default value of 1 in every direction is equivalent to point kriging.

    Example:
        >>> discretisation = BlockDiscretisation(nx=3, ny=3, nz=2)
    """

    nx: int = 1
    """Number of subdivisions in the x direction (1–9)."""

    ny: int = 1
    """Number of subdivisions in the y direction (1–9)."""

    nz: int = 1
    """Number of subdivisions in the z direction (1–9)."""

    @field_validator("nx", "ny", "nz")
    @classmethod
    def _validate_range(cls, v: int, info) -> int:
        if not isinstance(v, int):
            raise TypeError(f"{info.field_name} must be an integer, got {type(v).__name__}")
        if v < 1 or v > 9:
            raise ValueError(f"{info.field_name} must be between 1 and 9, got {v}")
        return v


# =============================================================================
# Region Filter
# =============================================================================


class RegionFilter(BaseModel):
    """Region filter for restricting kriging to specific categories on the target.

    Use either `names` OR `values`, not both:
    - `names`: Category names (strings) - used for CategoryAttribute with string lookup
    - `values`: Integer values - used for integer-indexed categories or BlockModel integer columns

    Example:
        >>> # Filter by category names (string lookup)
        >>> filter_by_name = RegionFilter(
        ...     attribute=block_model.attributes["domain"],
        ...     names=["LMS1", "LMS2"],
        ... )
        >>>
        >>> # Filter by integer values (direct index matching)
        >>> filter_by_value = RegionFilter(
        ...     attribute=block_model.attributes["domain"],
        ...     values=[1, 2, 3],
        ... )
    """

    model_config = {"arbitrary_types_allowed": True}

    attribute: str | Attribute | BlockModelAttribute
    """The category attribute to filter on (from target object)."""

    names: list[str] | None = None
    """Category names to include (mutually exclusive with values)."""

    values: list[int] | None = None
    """Integer category keys to include (mutually exclusive with names)."""

    def model_post_init(self, __context: Any) -> None:
        if self.names is not None and self.values is not None:
            raise ValueError("Only one of 'names' or 'values' may be provided, not both.")
        if self.names is None and self.values is None:
            raise ValueError("One of 'names' or 'values' must be provided.")

    @model_serializer
    def _serialize(self) -> dict[str, Any]:
        """Serialize to dictionary for the compute task API."""
        if isinstance(self.attribute, (Attribute, BlockModelAttribute)):
            attribute_expr = get_attribute_expression(self.attribute)
        elif isinstance(self.attribute, str):
            attribute_expr = self.attribute
        else:
            raise TypeError(f"Cannot serialize region filter attribute of type {type(self.attribute)}")

        result: dict[str, Any] = {"attribute": attribute_expr}

        if self.names is not None:
            result["names"] = self.names
        if self.values is not None:
            result["values"] = self.values

        return result


# =============================================================================
# Kriging Parameters
# =============================================================================


class KrigingParameters(BaseModel):
    """Parameters for the kriging task.

    Defines all inputs needed to run a kriging interpolation task.

    Example:
        >>> from evo.compute.tasks import run, SearchNeighborhood, Ellipsoid, EllipsoidRanges
        >>> from evo.compute.tasks.kriging import KrigingParameters, RegionFilter
        >>>
        >>> params = KrigingParameters(
        ...     source=pointset.attributes["grade"],  # Source attribute
        ...     target=block_model.attributes["kriged_grade"],  # Target attribute (creates if doesn't exist)
        ...     variogram=variogram,  # Variogram model
        ...     search=SearchNeighborhood(
        ...         ellipsoid=Ellipsoid(ranges=EllipsoidRanges(200, 150, 100)),
        ...         max_samples=20,
        ...     ),
        ...     # method defaults to ordinary kriging
        ... )
        >>>
        >>> # With region filter to restrict kriging to specific categories on target:
        >>> params_filtered = KrigingParameters(
        ...     source=pointset.attributes["grade"],
        ...     target=block_model.attributes["kriged_grade"],
        ...     variogram=variogram,
        ...     search=SearchNeighborhood(...),
        ...     target_region_filter=RegionFilter(
        ...         attribute=block_model.attributes["domain"],
        ...         names=["LMS1", "LMS2"],
        ...     ),
        ... )
    """

    source: Source
    """The source object and attribute containing known values."""

    target: Target
    """The target object and attribute to create or update with kriging results."""

    variogram: GeoscienceObjectReference
    """Model of the covariance within the domain (Variogram object or reference)."""

    search: SearchNeighborhood
    """Search neighborhood parameters."""

    method: SimpleKriging | OrdinaryKriging = OrdinaryKriging()
    """The kriging method to use. Defaults to ordinary kriging if not specified."""

    target_region_filter: RegionFilter | None = None
    """Optional region filter to restrict kriging to specific categories on the target object."""

    block_discretisation: BlockDiscretisation | None = None
    """Optional sub-block discretisation for block kriging.

    When provided, each target block is subdivided into nx × ny × nz sub-cells
    and the kriged value is averaged across these sub-cells. When omitted,
    point kriging is performed. Only applicable when the target is a 3D grid
    or block model.
    """

    @field_validator("source", mode="before")
    @classmethod
    def _convert_source(cls, v: Any) -> Source:
        if isinstance(v, (Attribute, BlockModelAttribute)):
            return source_from_attribute(v)
        return v

    @field_validator("target", mode="before")
    @classmethod
    def _convert_target(cls, v: Any) -> Target:
        if isinstance(v, (Attribute, PendingAttribute, BlockModelAttribute, BlockModelPendingAttribute)):
            return target_from_attribute(v)
        return v

    @model_serializer
    def _serialize(self) -> dict[str, Any]:
        target_dict = self.target.model_dump()

        # Embed region filter inside the target dict for the API
        if self.target_region_filter is not None:
            target_dict["region_filter"] = self.target_region_filter.model_dump()

        result: dict[str, Any] = {
            "source": self.source.model_dump(),
            "target": target_dict,
            "variogram": self.variogram,
            "neighborhood": self.search.model_dump(),
            "kriging_method": self.method.model_dump(),
        }

        if self.block_discretisation is not None:
            result["block_discretisation"] = self.block_discretisation.model_dump()

        return result


# =============================================================================
# Kriging Result Types
# =============================================================================


class KrigingResult(TaskResult):
    """Result of a kriging task.

    Contains information about the completed kriging operation and provides
    convenient methods to access the target object and its data.

    Example:
        >>> result = await run(manager, params)
        >>> result  # Pretty-prints the result
        >>>
        >>> # Get data directly as DataFrame (simplest approach)
        >>> df = await result.to_dataframe()
        >>>
        >>> # Or load the target object for more control
        >>> target = await result.get_target_object()
    """

    def _get_result_type_name(self) -> str:
        """Get the display name for this result type."""
        return "Kriging"


# =============================================================================
# Task Runner
# =============================================================================


class KrigingRunner(
    TaskRunner[KrigingParameters, KrigingResult],
    topic="geostatistics",
    task="kriging",
):
    """Runner for kriging compute tasks.

    Automatically registered — used by ``run()`` for dispatch, or directly::

        result = await KrigingRunner(context, params, preview=True)
    """
