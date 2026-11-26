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

Example — IDW declustering::

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

Example — KNN declustering::

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

from collections.abc import Mapping
from typing import Annotated, Any, ClassVar

from evo.common import IContext
from evo.objects.typed import BaseObject, BaseSpatialObject
from pydantic import (
    BaseModel,
    BeforeValidator,
    Field,
    PrivateAttr,
    SerializerFunctionWrapHandler,
    ValidatorFunctionWrapHandler,
    field_validator,
    model_serializer,
    model_validator,
)

from ..common import (
    AnyTargetAttribute,
    GeoscienceObjectReference,
    SearchNeighborhood,
    SingleTargetTaskResult,
    TaskInputModel,
    TaskParameters,
    extract_typed_object,
)
from ..common.results import TaskTarget
from ..common.runner import TaskRunner
from ..common.search import _reject_kriging_only_drillhole_limits

__all__ = [
    "DeclusteringGrid",
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


class DeclusteringSource(TaskInputModel):
    """Source sample locations for declustering weight computation."""

    _typed_object: BaseObject | None = PrivateAttr(default=None)

    object: GeoscienceObjectReference
    """Reference to the spatial object containing sample locations."""

    @model_validator(mode="wrap")
    @classmethod
    def _remember_typed_object(cls, data: Any, handler: ValidatorFunctionWrapHandler) -> DeclusteringSource:
        typed_obj: BaseObject | None = None
        if isinstance(data, DeclusteringSource):
            typed_obj = data._typed_object
        elif isinstance(data, Mapping):
            typed_obj = extract_typed_object(data.get("object"))
        source = handler(data)
        if typed_obj is not None and source._typed_object is None:
            source._typed_object = typed_obj
        return source


class DeclusteringGrid(TaskInputModel):
    """Evaluation grid used for computing sample influence."""

    _typed_object: BaseObject | None = PrivateAttr(default=None)

    object: GeoscienceObjectReference
    """Reference to a regular grid or set of evaluation locations."""

    @model_validator(mode="wrap")
    @classmethod
    def _remember_typed_object(cls, data: Any, handler: ValidatorFunctionWrapHandler) -> DeclusteringGrid:
        typed_obj: BaseObject | None = None
        if isinstance(data, DeclusteringGrid):
            typed_obj = data._typed_object
        elif isinstance(data, Mapping):
            typed_obj = extract_typed_object(data.get("object"))
        grid = handler(data)
        if typed_obj is not None and grid._typed_object is None:
            grid._typed_object = typed_obj
        return grid


def _wrap_source(v: Any) -> Any:
    """Accept a raw object reference and wrap it in DeclusteringSource."""
    if isinstance(v, (DeclusteringSource, Mapping)):
        return v
    return DeclusteringSource(object=v)


def _wrap_grid(v: Any) -> Any:
    """Accept a raw object reference and wrap it in DeclusteringGrid."""
    if isinstance(v, (DeclusteringGrid, Mapping)):
        return v
    return DeclusteringGrid(object=v)


AnyDeclusteringSource = Annotated[DeclusteringSource, BeforeValidator(_wrap_source)]
AnyDeclusteringGrid = Annotated[DeclusteringGrid, BeforeValidator(_wrap_grid)]


# =============================================================================
# Declustering Parameters
# =============================================================================


class DeclusteringParameters(TaskParameters):
    """Parameters for the declustering task.

    Computes grid-based declustering weights by measuring each sample's
    influence on evaluation locations.

    The ``power`` parameter controls the estimation mode:

    - ``power=2.0`` (default) — inverse-distance weighting (IDW)
    - ``power=None`` — arithmetic-mean KNN (equal neighbour weights)

    Example::

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

    _object_roles: ClassVar[Mapping[str, tuple[type, ...]]] = {
        "source": (BaseSpatialObject,),
        "grid": (BaseSpatialObject,),
        "target": (BaseSpatialObject,),
    }
    _drillhole_source_field: ClassVar[str | None] = "source"

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

    @field_validator("neighborhood")
    @classmethod
    def _check_neighborhood(cls, neighborhood: SearchNeighborhood) -> SearchNeighborhood:
        return _reject_kriging_only_drillhole_limits(neighborhood)

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


class DeclusteringResultModel(BaseModel):
    """Pydantic model for the raw declustering task result."""

    message: str
    """A message describing what happened in the task."""

    target: TaskTarget
    """Target information from the task result."""


class DeclusteringResult(SingleTargetTaskResult[TaskTarget]):
    """Rich wrapper around the declustering task result.

    Provides convenience properties and methods for accessing the
    computed declustering weights.
    """

    TASK_DISPLAY_NAME: ClassVar[str] = "Declustering"

    def __init__(self, context: IContext, model: DeclusteringResultModel) -> None:
        super().__init__(context, model)

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
