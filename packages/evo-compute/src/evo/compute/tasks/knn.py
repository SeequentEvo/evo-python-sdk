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

"""K-Nearest Neighbors (KNN) estimation compute task client.

Estimates values at target locations using the arithmetic mean of the
*k*-nearest source samples within a search neighborhood.

.. note::

   This task uses the older ``geostat`` API format with
   ``object_reference`` / ``object_element`` parameter encoding.

Example:
    >>> from evo.compute.tasks import run
    >>> from evo.compute.tasks.knn import KNNParameters, KNNSource, KNNTarget
    >>>
    >>> params = KNNParameters(
    ...     source=KNNSource(
    ...         object_reference=pointset_url,
    ...         object_element=[GORefElement(path="/locations/attributes/@name=grade")],
    ...     ),
    ...     target=KNNTarget(
    ...         object_reference=grid_url,
    ...         object_element=[GORefElement(path="/cell_attributes/@name=knn_grade")],
    ...     ),
    ...     neighborhood=SearchNeighborhood(
    ...         ellipsoid=Ellipsoid(ranges=EllipsoidRanges(200, 150, 100)),
    ...         max_samples=20,
    ...     ),
    ... )
    >>> result = await run(manager, params)
"""

from __future__ import annotations

from typing import ClassVar, Literal

from evo.common import IContext
from pydantic import BaseModel, Field

from .common import GeoscienceObjectReference, SearchNeighborhood
from .common.runner import TaskRunner

__all__ = [
    "GORefElement",
    "KNNParameters",
    "KNNResult",
    "KNNResultModel",
    "KNNRunner",
    "KNNSource",
    "KNNTarget",
]


# =============================================================================
# Old-format reference types
# =============================================================================


class GORefElement(BaseModel):
    """A geoscience object reference element (old API format)."""

    type: Literal["element"] = "element"
    """Element type discriminator."""

    path: str
    """Attribute path, e.g. ``/locations/attributes/@name=grade``."""


class KNNSource(BaseModel):
    """Source for the KNN task (old-format geoscience-object-reference)."""

    type: Literal["geoscience-object-reference"] = "geoscience-object-reference"
    object_reference: GeoscienceObjectReference
    object_element: list[GORefElement]


class KNNTarget(BaseModel):
    """Target for the KNN task (old-format geoscience-object-reference)."""

    type: Literal["geoscience-object-reference"] = "geoscience-object-reference"
    object_reference: GeoscienceObjectReference
    object_element: list[GORefElement]


# =============================================================================
# Parameters
# =============================================================================


class KNNParameters(BaseModel):
    """Parameters for the KNN estimation task."""

    source: KNNSource
    """The source object and attribute containing known values."""

    target: KNNTarget
    """The target object and attribute to create or update with KNN results."""

    neighborhood: SearchNeighborhood
    """Search neighborhood parameters."""


# =============================================================================
# Result Types
# =============================================================================


class KNNModifiedResult(BaseModel):
    """Modified geoscience object reference from a KNN result."""

    object_reference: str
    object_element: list[GORefElement] | None = None


class KNNResultModel(BaseModel):
    """Pydantic model for the raw KNN task result."""

    message: str
    object_modified: KNNModifiedResult


class KNNResult:
    TASK_DISPLAY_NAME: ClassVar[str] = "KNN Estimation"

    def __init__(self, context: IContext, model: KNNResultModel) -> None:
        self._result = model.object_modified
        self._message = model.message
        self._context = context

    @property
    def message(self) -> str:
        return self._message

    @property
    def object_reference(self) -> str:
        """Reference URL to the modified object."""
        return self._result.object_reference

    def __str__(self) -> str:
        lines = [
            f"✓ {self.TASK_DISPLAY_NAME} Result",
            f"  Message: {self.message}",
            f"  Object:  {self.object_reference}",
        ]
        return "\n".join(lines)


# =============================================================================
# Task Runner
# =============================================================================


class KNNRunner(
    TaskRunner[KNNParameters, KNNResultModel, KNNResult],
    topic="geostat",
    task="knn",
):
    async def _get_result(self, raw_result: KNNResultModel) -> KNNResult:
        return KNNResult(self._context, raw_result)
