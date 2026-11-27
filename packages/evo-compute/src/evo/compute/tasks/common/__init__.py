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

"""Common task runner, source/target, and filter primitives for compute tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from evo.objects.typed.types import Ellipsoid, EllipsoidRanges, Rotation

from .discretization import BlockDiscretization
from .filters import AllOfFilter, AnyOfFilter, Filter, FilterCondition, FilterExpression, FilterOperator
from .results import TaskAttribute, TaskResultList, TaskTarget
from .runner import TaskRegistry, TaskRunner, TParams, TResult, TResultModel, run_tasks
from .source_target import (
    AnySourceAttribute,
    AnyTargetAttribute,
    AttributeExpression,
    CreateAttribute,
    GeoscienceObjectReference,
    Source,
    Target,
    UpdateAttribute,
    attribute_spec,
)

if TYPE_CHECKING:
    from ..geostatistics.search import SearchNeighborhood  # noqa: F401


def __getattr__(name: str) -> Any:
    if name == "SearchNeighborhood":
        from ..geostatistics.search import SearchNeighborhood

        return SearchNeighborhood
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AllOfFilter",
    "AnyOfFilter",
    "AnySourceAttribute",
    "AnyTargetAttribute",
    "AttributeExpression",
    "BlockDiscretization",
    "CreateAttribute",
    "Ellipsoid",
    "EllipsoidRanges",
    "Filter",
    "FilterCondition",
    "FilterExpression",
    "FilterOperator",
    "GeoscienceObjectReference",
    "Rotation",
    "Source",
    "TParams",
    "TResult",
    "TResultModel",
    "Target",
    "TaskAttribute",
    "TaskRegistry",
    "TaskResultList",
    "TaskRunner",
    "TaskTarget",
    "UpdateAttribute",
    "attribute_spec",
    "run_tasks",
]
