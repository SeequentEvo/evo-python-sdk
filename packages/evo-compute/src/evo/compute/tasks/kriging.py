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

from dataclasses import dataclass
from typing import Any, TypeVar

from evo.common import IContext
from evo.common.interfaces import IFeedback
from evo.common.utils import NoFeedback, Retry
from evo.objects.typed.attributes import Attribute

from ..client import JobClient

# Import shared components
from .common import (
    GeoscienceObjectReference,
    SearchNeighborhood,
    Source,
    Target,
    get_attribute_expression,
    is_typed_attribute,
    serialize_object_reference,
    source_from_attribute,
    target_from_attribute,
)
from .common.runner import register_task_runner

__all__ = [
    # Kriging-specific (users import from evo.compute.tasks.kriging)
    "BlockDiscretisation",
    "KrigingMethod",
    "KrigingParameters",
    "OrdinaryKriging",
    "RegionFilter",
    "SimpleKriging",
]


# Type variable for generic result type
TResult = TypeVar("TResult", bound="TaskResult")


# =============================================================================
# Kriging Method Types
# =============================================================================


@dataclass
class SimpleKriging:
    """Simple kriging method with a known constant mean.

    Use when the mean of the variable is known and constant across the domain.

    Example:
        >>> method = SimpleKriging(mean=100.0)
    """

    mean: float
    """The mean value, assumed to be constant across the domain."""

    def __init__(self, mean: float):
        self.mean = mean

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "type": "simple",
            "mean": self.mean,
        }


@dataclass
class OrdinaryKriging:
    """Ordinary kriging method with unknown local mean.

    The most common kriging method. Estimates the local mean from nearby samples.
    This is the default kriging method if none is specified.
    """

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "type": "ordinary",
        }


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
        return SimpleKriging(mean)


# =============================================================================
# Block Discretisation
# =============================================================================


@dataclass
class BlockDiscretisation:
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

    nx: int
    """Number of subdivisions in the x direction (1–9)."""

    ny: int
    """Number of subdivisions in the y direction (1–9)."""

    nz: int
    """Number of subdivisions in the z direction (1–9)."""

    def __init__(self, nx: int = 1, ny: int = 1, nz: int = 1):
        for name, value in [("nx", nx), ("ny", ny), ("nz", nz)]:
            if not isinstance(value, int):
                raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
            if value < 1 or value > 9:
                raise ValueError(f"{name} must be between 1 and 9, got {value}")
        self.nx = nx
        self.ny = ny
        self.nz = nz

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "nx": self.nx,
            "ny": self.ny,
            "nz": self.nz,
        }


# =============================================================================
# Region Filter
# =============================================================================


@dataclass
class RegionFilter:
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

    attribute: Any
    """The category attribute to filter on (from target object)."""

    names: list[str] | None = None
    """Category names to include (mutually exclusive with values)."""

    values: list[int] | None = None
    """Integer category keys to include (mutually exclusive with names)."""

    def __init__(
        self,
        attribute: Any,
        names: list[str] | None = None,
        values: list[int] | None = None,
    ):
        if names is not None and values is not None:
            raise ValueError("Only one of 'names' or 'values' may be provided, not both.")
        if names is None and values is None:
            raise ValueError("One of 'names' or 'values' must be provided.")

        self.attribute = attribute
        self.names = names
        self.values = values

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for the compute task API."""
        if is_typed_attribute(self.attribute):
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


@dataclass
class KrigingParameters:
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

    method: SimpleKriging | OrdinaryKriging | None = None
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

    def __init__(
        self,
        source: Source | Any,  # Also accepts Attribute from evo.objects.typed
        target: Target | Any,  # Also accepts Attribute/PendingAttribute from evo.objects.typed
        variogram: GeoscienceObjectReference,
        search: SearchNeighborhood,
        method: SimpleKriging | OrdinaryKriging | None = None,
        target_region_filter: RegionFilter | None = None,
        block_discretisation: BlockDiscretisation | None = None,
    ):
        # Handle Attribute types from evo.objects.typed.attributes
        if isinstance(source, Attribute):
            source = source_from_attribute(source)

        # Handle target attribute types (Attribute, PendingAttribute, BlockModelAttribute, BlockModelPendingAttribute)
        if is_typed_attribute(target):
            target = target_from_attribute(target)

        self.source = source
        self.target = target
        self.variogram = variogram
        self.search = search
        self.method = method or OrdinaryKriging()
        self.target_region_filter = target_region_filter
        self.block_discretisation = block_discretisation

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        target_dict = self.target.to_dict()

        # Add region filter to target if provided
        if self.target_region_filter is not None:
            target_dict["region_filter"] = self.target_region_filter.to_dict()

        result = {
            "source": self.source.to_dict(),
            "target": target_dict,
            "variogram": serialize_object_reference(self.variogram),
            "neighborhood": self.search.to_dict(),
            "kriging_method": self.method.to_dict(),
        }

        # Add block discretisation if provided (omit for point kriging)
        if self.block_discretisation is not None:
            result["block_discretisation"] = self.block_discretisation.to_dict()

        return result


# =============================================================================
# Kriging Result Types
# =============================================================================


@dataclass
class _KrigingAttribute:
    """Attribute containing the kriging result (internal)."""

    reference: str
    name: str


@dataclass
class _KrigingTarget:
    """The target that was created or updated (internal)."""

    reference: str
    name: str
    description: Any
    schema_id: str
    attribute: _KrigingAttribute


# =============================================================================
# Base Task Result Classes
# =============================================================================


class TaskResult:
    """Base class for compute task results.

    Provides common functionality for all task results including:
    - Pretty-printing in Jupyter notebooks
    - Portal URL extraction
    - Access to target object and data
    """

    message: str
    """A message describing what happened in the task."""

    _target: _KrigingTarget
    """Internal target information."""

    _context: IContext | None = None
    """The context used to run the task (for convenience methods)."""

    def __init__(self, message: str, target: _KrigingTarget):
        self.message = message
        self._target = target
        self._context = None

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
    def schema_type(self) -> str:
        """The schema type of the target object (e.g., 'regular-masked-3d-grid')."""
        schema = self._target.schema_id
        if "/" in schema:
            parts = schema.split("/")
            for part in parts:
                if part and not part.startswith("objects") and "." not in part and part[0].isalpha():
                    return part
        return schema

    async def get_target_object(self, context: IContext | None = None):
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
        from evo.objects.typed import object_from_reference

        ctx = context or self._context
        if ctx is None:
            raise ValueError(
                "No context available. Either pass a context to get_target_object() "
                "or ensure the result was returned from run()."
            )
        return await object_from_reference(ctx, self._target.reference)

    async def to_dataframe(self, context: IContext | None = None, columns: list[str] | None = None):
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
        target_obj = await self.get_target_object(context)

        # Try different methods to get the dataframe based on object type
        if hasattr(target_obj, "to_dataframe"):
            # BlockModel, PointSet, and similar objects with to_dataframe
            if columns is not None:
                return await target_obj.to_dataframe(columns=columns)
            return await target_obj.to_dataframe()
        elif hasattr(target_obj, "cells") and hasattr(target_obj.cells, "to_dataframe"):
            # Grid objects (Regular3DGrid, RegularMasked3DGrid, etc.)
            return await target_obj.cells.to_dataframe()
        else:
            raise TypeError(
                f"Don't know how to get DataFrame from {type(target_obj).__name__}. "
                "Use get_target_object() and access the data manually."
            )

    def _get_result_type_name(self) -> str:
        """Get the display name for this result type."""
        return "Task"

    def __repr__(self) -> str:
        """String representation."""
        lines = [
            f"✓ {self._get_result_type_name()} Result",
            f"  Message:   {self.message}",
            f"  Target:    {self.target_name}",
            f"  Attribute: {self.attribute_name}",
        ]
        return "\n".join(lines)


class TaskResults:
    """Container for multiple task results with pretty-printing support.

    Provides iteration and indexing support for accessing individual results.

    Example:
        >>> results = await run(manager, [params1, params2, params3])
        >>> results  # Pretty-prints all results
        >>> results[0]  # Access first result
        >>> for result in results:
        ...     print(result.attribute_name)
    """

    def __init__(self, results: list[TaskResult]):
        self._results = results

    @property
    def results(self) -> list[TaskResult]:
        """The list of task results."""
        return self._results

    def __len__(self) -> int:
        return len(self._results)

    def __iter__(self):
        return iter(self._results)

    def __getitem__(self, index: int) -> TaskResult:
        return self._results[index]

    def __repr__(self) -> str:
        """String representation."""
        if not self._results:
            return "TaskResults([])"
        result_type = self._results[0]._get_result_type_name()
        lines = [f"✓ {len(self._results)} {result_type} Results:"]
        for i, result in enumerate(self._results):
            lines.append(f"  [{i}] {result.target_name} → {result.attribute_name}")
        return "\n".join(lines)


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

    def __init__(self, message: str, target: _KrigingTarget):
        """Initialize a KrigingResult.

        Args:
            message: A message describing what happened in the task.
            target: The target information from the kriging result.
        """
        super().__init__(message=message, target=target)

    def _get_result_type_name(self) -> str:
        """Get the display name for this result type."""
        return "Kriging"


# =============================================================================
# Run Functions
# =============================================================================


def _parse_kriging_result(data: dict[str, Any]) -> KrigingResult:
    """Parse the kriging result from the API response."""
    target_data = data["target"]
    attr_data = target_data["attribute"]

    attribute = _KrigingAttribute(
        reference=attr_data["reference"],
        name=attr_data["name"],
    )
    target = _KrigingTarget(
        reference=target_data["reference"],
        name=target_data["name"],
        description=target_data.get("description"),
        schema_id=target_data["schema_id"],
        attribute=attribute,
    )
    return KrigingResult(message=data["message"], target=target)


async def _run_single_kriging(
    context: IContext,
    parameters: KrigingParameters,
    *,
    preview: bool = False,
    polling_interval_seconds: float = 0.5,
    retry: Retry | None = None,
    fb: IFeedback = NoFeedback,
) -> KrigingResult:
    """Internal function to run a single kriging task."""
    connector = context.get_connector()
    org_id = context.get_org_id()

    params_dict = parameters.to_dict()

    # Submit the job
    job = await JobClient.submit(
        connector=connector,
        org_id=org_id,
        topic="geostatistics",
        task="kriging",
        parameters=params_dict,
        result_type=dict,  # Get raw dict, we'll parse it ourselves
        preview=preview,
    )

    # Wait for results
    raw_result = await job.wait_for_results(
        polling_interval_seconds=polling_interval_seconds,
        retry=retry,
        fb=fb,
    )

    # Parse and return
    result = _parse_kriging_result(raw_result)
    result._context = context
    return result


async def _run_kriging_for_registry(
    context: IContext,
    parameters: KrigingParameters,
    *,
    preview: bool = False,
) -> KrigingResult:
    """Simplified runner function for task registry (no extra options).

    This is the function registered with the TaskRegistry. For more control
    over polling and retry behavior, use the full `run()` function.
    """
    return await _run_single_kriging(context, parameters, preview=preview)


# Register kriging task runner with the task registry

register_task_runner(KrigingParameters, _run_kriging_for_registry)
