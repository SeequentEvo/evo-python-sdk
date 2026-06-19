#  Copyright © 2026 Bentley Systems, Incorporated
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""K-nearest neighbour (KNN) estimation compute task client.

The KNN task estimates values at target locations by computing the arithmetic
mean of the accepted neighbours found within the search ellipsoid.  No
variogram model is required — it is purely neighbourhood-based.

Example:
    >>> from evo.compute.tasks import run, SearchNeighborhood, Ellipsoid, EllipsoidRanges
    >>> from evo.compute.tasks.common import Source, Target
    >>> from evo.compute.tasks.geostatistics.knn import KNNParameters
    >>>
    >>> params = KNNParameters(
    ...     source=Source(object=pointset, attribute="locations.attributes[?name=='grade']"),
    ...     target=Target.new_attribute(grid, "knn_grade"),
    ...     neighborhood=SearchNeighborhood(
    ...         ellipsoid=Ellipsoid(ranges=EllipsoidRanges(200, 150, 100)),
    ...         max_samples=20,
    ...     ),
    ... )
    >>> result = await run(manager, params)
"""

from __future__ import annotations

from typing import Any, ClassVar, Protocol, runtime_checkable

import pandas as pd
from evo.common import IContext, IFeedback
from evo.objects import ObjectSchema
from evo.objects.typed import BaseObject, object_from_reference
from pydantic import BaseModel, Field, SerializerFunctionWrapHandler, model_serializer

from ..common import (
    AnySourceAttribute,
    AnyTargetAttribute,
    Filter,
    SearchNeighborhood,
)
from ..common.results import TaskTarget
from ..common.runner import TaskRunner

__all__ = [
    "KNNParameters",
    "KNNResult",
    "KNNResultModel",
    "KNNRunner",
]


# =============================================================================
# Parameters
# =============================================================================


class KNNParameters(BaseModel):
    """Parameters for the KNN (K-nearest neighbour) estimation task.

    Estimates values at target locations by computing the arithmetic mean of
    accepted neighbours found within the search ellipsoid.

    Example:
        >>> from evo.compute.tasks.common import Filter, FilterCondition, Source, Target
        >>> from evo.compute.tasks.geostatistics.knn import KNNParameters
        >>>
        >>> params = KNNParameters(
        ...     source=Source(object=pointset, attribute="locations.attributes[?name=='grade']"),
        ...     target=Target.new_attribute(grid, "knn_grade"),
        ...     neighborhood=SearchNeighborhood(
        ...         ellipsoid=Ellipsoid(ranges=EllipsoidRanges(200, 150, 100)),
        ...         max_samples=20,
        ...     ),
        ... )
        >>>
        >>> # With a target filter to restrict estimation to specific categories on the target:
        >>> params_filtered = KNNParameters(
        ...     source=Source(object=pointset, attribute="locations.attributes[?name=='grade']"),
        ...     target=Target.new_attribute(grid, "knn_grade"),
        ...     neighborhood=SearchNeighborhood(...),
        ...     target_filter=Filter(
        ...         where=FilterCondition(
        ...             attribute=grid.attributes["domain"],
        ...             operator="in",
        ...             values=["LMS1", "LMS2"],
        ...         ),
        ...     ),
        ... )
    """

    source: AnySourceAttribute
    """Source object and attribute containing the known values."""

    target: AnyTargetAttribute
    """Target object and attribute to create or update with KNN estimates."""

    neighborhood: SearchNeighborhood
    """Search neighbourhood defining which nearby samples to include."""

    source_filter: Filter | None = Field(None, exclude=True)
    """Optional filter to restrict estimation to a subset of the source data.

    Excluded from top-level serialisation (``exclude=True``) and injected as
    ``source["filter"]`` by the custom serialiser.
    """

    target_filter: Filter | None = Field(None, exclude=True)
    """Optional filter to restrict estimation to a subset of the target object.

    Excluded from top-level serialisation (``exclude=True``) and injected as
    ``target["filter"]`` by the custom serialiser.
    """

    @model_serializer(mode="wrap")
    def _serialize(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        result = handler(self)
        if self.source_filter is not None:
            result["source"]["filter"] = self.source_filter.model_dump()
        if self.target_filter is not None:
            result["target"]["filter"] = self.target_filter.model_dump()
        return result


# =============================================================================
# Result Types
# =============================================================================


@runtime_checkable
class _ObjToDataframeProtocol(Protocol):
    async def to_dataframe(self, *keys: str, fb: IFeedback = ...) -> pd.DataFrame: ...


class KNNResultModel(BaseModel):
    """Pydantic model for the raw KNN task result payload."""

    message: str
    """A message describing what happened in the task."""

    target: TaskTarget
    """Target information from the task result."""


class KNNResult:
    """Rich result object for the KNN estimation task."""

    TASK_DISPLAY_NAME: ClassVar[str] = "KNN Estimation"

    def __init__(self, context: IContext, model: KNNResultModel) -> None:
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
        """The name of the attribute that was created or updated."""
        return self._target.attribute.name

    @property
    def schema(self) -> ObjectSchema:
        """The schema type of the target object (e.g., ``'pointset'``)."""
        return ObjectSchema.from_id(self._target.schema_id)

    async def get_target_object(self) -> BaseObject:
        """Load and return the target geoscience object."""
        return await object_from_reference(self._context, self._target.reference)

    async def to_dataframe(self, *columns: str) -> pd.DataFrame:
        """Load the target object and return its data as a DataFrame.

        Args:
            *columns: Optional column names to select.

        Raises:
            TypeError: If the target object type does not support DataFrame export.
        """
        target_obj = await self.get_target_object()
        if isinstance(target_obj, _ObjToDataframeProtocol):
            return await target_obj.to_dataframe(*columns)
        raise TypeError(
            f"Don't know how to get DataFrame from {type(target_obj).__name__}. "
            "Use get_target_object() and access the data manually."
        )

    def __str__(self) -> str:
        return f"✓ {self.TASK_DISPLAY_NAME} Result\n  Target:    {self.target_name}\n  Attribute: {self.attribute_name}"


# =============================================================================
# Task Runner
# =============================================================================


class KNNRunner(
    TaskRunner[KNNParameters, KNNResultModel, KNNResult],
    topic="geostatistics",
    task="knn",
):
    async def _get_result(self, raw_result: KNNResultModel) -> KNNResult:
        return KNNResult(self._context, raw_result)
