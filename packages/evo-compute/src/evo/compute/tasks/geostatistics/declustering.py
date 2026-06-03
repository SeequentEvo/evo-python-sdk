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
Declustering compute task client.

This module provides typed models for running the declustering task
(geostatistics/declustering).  The task computes grid-based declustering
weights by measuring each sample's influence on evaluation locations.

Two estimation modes are supported via factory functions:

- :func:`idw` — inverse-distance weighting (default ``power=2.0``)
- :func:`knn` — nearest-neighbour (equal weights, typically ``max_samples=1``)

Example — IDW declustering:
    >>> from evo.compute.tasks import run, SearchNeighborhood, Ellipsoid, EllipsoidRanges, Target
    >>> from evo.compute.tasks.geostatistics.declustering import idw
    >>>
    >>> params = idw(
    ...     source=pointset,
    ...     grid=regular_grid,
    ...     target=Target.new_attribute(pointset, "declustering_weights"),
    ...     neighborhood=SearchNeighborhood(
    ...         ellipsoid=Ellipsoid(ranges=EllipsoidRanges(200, 150, 100)),
    ...         max_samples=20,
    ...     ),
    ... )
    >>> result = await run(context, params)

Example — KNN declustering:
    >>> from evo.compute.tasks.geostatistics.declustering import knn
    >>>
    >>> params = knn(
    ...     source=pointset,
    ...     grid=regular_grid,
    ...     target=Target.new_attribute(pointset, "declustering_weights"),
    ...     neighborhood=SearchNeighborhood(
    ...         ellipsoid=Ellipsoid(ranges=EllipsoidRanges(200, 150, 100)),
    ...         max_samples=1,
    ...     ),
    ... )
    >>> result = await run(context, params)
"""

from __future__ import annotations

from typing import Annotated, Any, ClassVar, Protocol, runtime_checkable

import pandas as pd
from evo.common import IContext, IFeedback
from evo.objects import ObjectSchema
from evo.objects.typed import BaseObject, object_from_reference
from pydantic import BaseModel, BeforeValidator, Field, SerializerFunctionWrapHandler, model_serializer

from ..common import (
    AnyTargetAttribute,
    GeoscienceObjectReference,
    SearchNeighborhood,
)
from ..common.results import TaskTarget
from ..common.runner import TaskRunner
from ..common.source_target import _convert_object_reference

__all__ = [
    "DeclusteringParameters",
    "DeclusteringResult",
    "DeclusteringResultModel",
    "DeclusteringRunner",
    "DeclusteringSource",
    "idw",
    "knn",
]


# =============================================================================
# Shared input models
# =============================================================================


class DeclusteringSource(BaseModel):
    """Source sample locations for declustering weight computation."""

    object: GeoscienceObjectReference
    """Reference to the spatial object containing sample locations."""


class DeclusteringGrid(BaseModel):
    """Evaluation grid used for computing sample influence."""

    object: GeoscienceObjectReference
    """Reference to a regular grid or set of evaluation locations."""


def _wrap_source(v: Any) -> DeclusteringSource:
    """Accept a raw object reference and wrap it in DeclusteringSource."""
    if isinstance(v, DeclusteringSource):
        return v
    return DeclusteringSource(object=_convert_object_reference(v))


def _wrap_grid(v: Any) -> DeclusteringGrid:
    """Accept a raw object reference and wrap it in DeclusteringGrid."""
    if isinstance(v, DeclusteringGrid):
        return v
    return DeclusteringGrid(object=_convert_object_reference(v))


AnyDeclusteringSource = Annotated[DeclusteringSource, BeforeValidator(_wrap_source)]
AnyDeclusteringGrid = Annotated[DeclusteringGrid, BeforeValidator(_wrap_grid)]


# =============================================================================
# Declustering Parameters
# =============================================================================


class DeclusteringParameters(BaseModel):
    """Parameters for the declustering task.

    Computes grid-based declustering weights by measuring each sample's
    influence on evaluation locations.

    The ``power`` parameter controls the estimation mode:

    - ``power=2.0`` (default) — inverse-distance weighting (IDW)
    - ``power=None`` — arithmetic-mean KNN (equal neighbour weights)

    Example:
        >>> params = DeclusteringParameters(
        ...     source=pointset,
        ...     grid=regular_grid,
        ...     target=Target.new_attribute(pointset, "weights"),
        ...     neighborhood=SearchNeighborhood(
        ...         ellipsoid=Ellipsoid(ranges=EllipsoidRanges(200, 150, 100)),
        ...         max_samples=20,
        ...     ),
        ... )
    """

    source: AnyDeclusteringSource
    """Source sample locations (accepts a geoscience object directly)."""

    grid: AnyDeclusteringGrid
    """Evaluation grid for measuring sample influence (accepts a geoscience object directly)."""

    target: AnyTargetAttribute
    """Target object and attribute to write declustering weights to."""

    neighborhood: SearchNeighborhood
    """Search neighborhood parameters."""

    power: float | None = Field(default=2.0, gt=0.0)
    """Inverse-distance weighting power.

    Controls how strongly distance affects influence: higher values give
    nearer samples more weight.  Set to ``None`` to use KNN mode
    (arithmetic-mean, equal neighbour votes).

    Default is ``2.0`` (standard IDW).  Must be positive when provided.
    """

    @model_serializer(mode="wrap")
    def _serialize(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        result = handler(self)
        # Explicitly include power even when None so exclude_none=True in
        # TaskRunner.__call__ doesn't drop it — the server interprets a
        # missing power as IDW with the default, while null means KNN.
        if self.power is None:
            result["power"] = None
        return result


# =============================================================================
# Factory functions
# =============================================================================


def idw(
    source: Any,
    grid: Any,
    target: AnyTargetAttribute,
    neighborhood: SearchNeighborhood,
    *,
    power: float = 2.0,
) -> DeclusteringParameters:
    """Create IDW (inverse-distance weighting) declustering parameters.

    Higher ``power`` values give nearer samples more weight.

    Args:
        source: Source sample locations (geoscience object URL or DeclusteringSource).
        grid: Evaluation grid (geoscience object URL or DeclusteringGrid).
        target: Target object and attribute to write declustering weights to.
        neighborhood: Search neighborhood parameters.
        power: IDW power exponent.  Default ``2.0``.  Must be positive.

    Returns:
        Configured DeclusteringParameters ready for ``run()``.
    """
    return DeclusteringParameters(
        source=source,
        grid=grid,
        target=target,
        neighborhood=neighborhood,
        power=power,
    )


def knn(
    source: Any,
    grid: Any,
    target: AnyTargetAttribute,
    neighborhood: SearchNeighborhood,
) -> DeclusteringParameters:
    """Create KNN (nearest-neighbour) declustering parameters.

    Uses equal neighbour weights (arithmetic mean).  For true
    nearest-neighbour behaviour, set ``max_samples=1`` on the
    :class:`~evo.compute.tasks.common.SearchNeighborhood`.

    Args:
        source: Source sample locations (geoscience object URL or DeclusteringSource).
        grid: Evaluation grid (geoscience object URL or DeclusteringGrid).
        target: Target object and attribute to write declustering weights to.
        neighborhood: Search neighborhood parameters.

    Returns:
        Configured DeclusteringParameters ready for ``run()``.
    """
    return DeclusteringParameters(
        source=source,
        grid=grid,
        target=target,
        neighborhood=neighborhood,
        power=None,
    )


# =============================================================================
# Declustering Result Types
# =============================================================================


@runtime_checkable
class _ObjToDataframeProtocol(Protocol):
    """Protocol for objects that can convert themselves to a DataFrame."""

    async def to_dataframe(self, *keys: str, fb: IFeedback = ...) -> pd.DataFrame: ...


class DeclusteringResultModel(BaseModel):
    """Pydantic model for the raw declustering task result."""

    message: str
    """A message describing what happened in the task."""

    target: TaskTarget
    """Target information from the task result."""


class DeclusteringResult:
    """Rich wrapper around the declustering task result.

    Provides convenience properties and methods for accessing the
    computed declustering weights.
    """

    TASK_DISPLAY_NAME: ClassVar[str] = "Declustering"

    def __init__(self, context: IContext, model: DeclusteringResultModel) -> None:
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
        """The name of the attribute containing declustering weights."""
        return self._target.attribute.name

    @property
    def schema(self) -> ObjectSchema:
        """The schema type of the target object."""
        return ObjectSchema.from_id(self._target.schema_id)

    async def get_target_object(self) -> BaseObject:
        """Load and return the target geoscience object.

        Returns:
            The typed geoscience object containing the declustering weights.
        """
        return await object_from_reference(self._context, self._target.reference)

    async def to_dataframe(self, *columns: str) -> pd.DataFrame:
        """Get the declustering weights as a DataFrame.

        Args:
            columns: Optional column names to include. If omitted, returns all columns.

        Returns:
            A pandas DataFrame containing the declustering weight values.
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
            f"  Message:   {self.message}",
            f"  Target:    {self.target_name}",
            f"  Attribute: {self.attribute_name}",
        ]
        return "\n".join(lines)


# =============================================================================
# Task Runner
# =============================================================================


class DeclusteringRunner(
    TaskRunner[DeclusteringParameters, DeclusteringResultModel, DeclusteringResult],
    topic="geostatistics",
    task="declustering",
):
    """Runner for declustering compute tasks.

    Automatically registered — used by ``run()`` for dispatch, or directly::

        result = await DeclusteringRunner(context, params)
    """

    async def _get_result(self, raw_result: DeclusteringResultModel) -> DeclusteringResult:
        return DeclusteringResult(self._context, raw_result)
