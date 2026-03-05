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

from typing import Any, ClassVar, Literal, Protocol, runtime_checkable
from uuid import UUID

import pandas as pd
from evo.common import IContext, IFeedback
from evo.objects import ObjectSchema
from evo.objects.typed import BaseObject, BlockModelPendingAttribute, PendingAttribute, object_from_reference
from pydantic import BaseModel, Field, SerializerFunctionWrapHandler, field_validator, model_serializer

# Import shared components
from .common import (
    AnySourceAttribute,
    AnyTargetAttribute,
    AttributeExpression,
    GeoscienceObjectReference,
    SearchNeighborhood,
)
from .common.results import TaskTarget
from .common.runner import TaskRunner

__all__ = [
    # Kriging-specific (users import from evo.compute.tasks.kriging)
    "BlockDiscretisation",
    "KrigingMethod",
    "KrigingParameters",
    "KrigingResult",
    "KrigingResultModel",
    "KrigingRunner",
    "OrdinaryKriging",
    "RegionFilter",
    "SimpleKriging",
]


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

    nx: int = Field(1, ge=1, le=9)
    """Number of subdivisions in the x direction (1-9)."""

    ny: int = Field(1, ge=1, le=9)
    """Number of subdivisions in the y direction (1-9)."""

    nz: int = Field(1, ge=1, le=9)
    """Number of subdivisions in the z direction (1-9)."""


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

    attribute: AttributeExpression
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

    @field_validator("attribute", mode="before")
    @classmethod
    def _validate_attribute(cls, v: Any) -> AttributeExpression:
        if isinstance(v, (PendingAttribute, BlockModelPendingAttribute)):
            raise ValueError("RegionFilter attribute cannot be a PendingAttribute. Provide a valid existing attribute.")
        return v


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

    model_config = {"populate_by_name": True}

    source: AnySourceAttribute
    """The source object and attribute containing known values."""

    target: AnyTargetAttribute
    """The target object and attribute to create or update with kriging results."""

    variogram: GeoscienceObjectReference
    """Model of the covariance within the domain (Variogram object or reference)."""

    search: SearchNeighborhood = Field(alias="neighborhood")
    """Search neighborhood parameters."""

    method: SimpleKriging | OrdinaryKriging = Field(default_factory=OrdinaryKriging, alias="kriging_method")
    """The kriging method to use. Defaults to ordinary kriging if not specified."""

    target_region_filter: RegionFilter | None = Field(None, exclude=True)
    """Optional region filter to restrict kriging to specific categories on the target object."""

    block_discretisation: BlockDiscretisation | None = None
    """Optional sub-block discretisation for block kriging.

    When provided, each target block is subdivided into nx × ny × nz sub-cells
    and the kriged value is averaged across these sub-cells. When omitted,
    point kriging is performed. Only applicable when the target is a 3D grid
    or block model.
    """

    @model_serializer(mode="wrap")
    def _serialize(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        result = handler(self)
        if self.target_region_filter is not None:
            result["target"]["region_filter"] = self.target_region_filter.model_dump()
        return result


# =============================================================================
# Kriging Result Types
# =============================================================================

# TODO: tidy up `to_dataframe` implementations for better consistency. in _theory_ and spatial object does implement
# `to_dataframe()` (and should!!), but `BaseSPatialObject` does not declare this, and the `BlockModel` object has a different
# signature.


@runtime_checkable
class _ObjToDataframeProtocol(Protocol):
    """Protocol for objects that can convert themselves to a DataFrame."""

    async def to_dataframe(self, *keys: str, fb: IFeedback = ...) -> pd.DataFrame: ...


@runtime_checkable
class _BlockModelToDataFrameProtocol(Protocol):
    """Protocol for block models that can convert themselves to a DataFrame."""

    async def to_dataframe(
        self,
        columns: list[str] | None = None,
        version_uuid: UUID | Literal["latest"] | None = None,
        fb: IFeedback = ...,
    ) -> pd.DataFrame: ...


class KrigingResultModel(BaseModel):
    """Base class for compute task results.

    Provides common functionality for all task results including:
    - Pretty-printing in Jupyter notebooks
    - Portal URL extraction
    - Access to target object and data
    """

    message: str
    """A message describing what happened in the task."""

    target: TaskTarget
    """Target information from the task result."""


class KrigingResult:
    TASK_DISPLAY_NAME: ClassVar[str] = "Kriging"

    def __init__(self, context: IContext, model: KrigingResultModel) -> None:
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
        """Reference URL to the target object."""
        return self._target.reference

    @property
    def attribute_name(self) -> str:
        """The name of the attribute that was created/updated."""
        return self._target.attribute.name

    @property
    def schema(self) -> ObjectSchema:
        """The schema type of the target object (e.g., 'regular-masked-3d-grid').

        Uses ``ObjectSchema.from_id`` to parse the schema ID. Falls back to the
        raw ``schema_id`` string when it cannot be parsed.
        """
        return ObjectSchema.from_id(self._target.schema_id)

    async def get_target_object(self) -> BaseObject:
        """Load and return the target geoscience object.

        Args:
            context: Optional context to use. If not provided, uses the context
                    from when the task was run.

        Returns:
            The typed geoscience object (e.g., Regular3DGrid, RegularMasked3DGrid, BlockModel)

        Example:
            >>> result = await run(manager, params)
            >>> target = await result.get_target_object()
            >>> target  # Pretty-prints with Portal/Viewer links
        """
        return await object_from_reference(self._context, self._target.reference)

    async def to_dataframe(self, *columns: str) -> pd.DataFrame:
        """Get the task results as a DataFrame.

        This is the simplest way to access the task output data. It loads
        the target object and returns its data as a pandas DataFrame.

        Args:
            context: Optional context to use. If not provided, uses the context
                    from when the task was run.
            columns: Optional list of column names to include. If None, includes
                    all columns. Use ["*"] to explicitly request all columns.

        Returns:
            A pandas DataFrame containing the task results.

        Example:
            >>> result = await run(manager, params)
            >>> df = await result.to_dataframe()
            >>> df.head()
        """
        target_obj = await self.get_target_object()

        if isinstance(target_obj, _ObjToDataframeProtocol):
            return await target_obj.to_dataframe(*columns)
        elif isinstance(target_obj, _BlockModelToDataFrameProtocol):
            return target_obj.to_dataframe(columns=list(columns) if columns else None)
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


class KrigingRunner(
    TaskRunner[KrigingParameters, KrigingResultModel, KrigingResult],
    topic="geostatistics",
    task="kriging",
):
    """Runner for kriging compute tasks.

    Automatically registered — used by ``run()`` for dispatch, or directly::

        result = await KrigingRunner(context, params, preview=True)
    """

    async def _get_result(self, raw_result: KrigingResultModel) -> KrigingResult:
        return KrigingResult(self._context, raw_result)
