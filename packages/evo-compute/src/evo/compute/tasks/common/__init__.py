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

"""Common primitives shared across geostatistics compute tasks."""

from evo.objects.typed.types import Ellipsoid, EllipsoidRanges, Rotation

from .discretization import BlockDiscretization
from .filters import AllOfFilter, AnyOfFilter, Filter, FilterCondition, FilterExpression, FilterOperator
from .models import (
    FiniteFloat,
    OutputTarget,
    QuantileFloat,
    SourceTargetTaskParameters,
    TaskInputModel,
    TaskParameters,
    canonicalize_attribute_selector,
    check_batch_output_collisions,
    deduplicate_aliases,
    extract_typed_object,
    object_identity,
    resolve_output_target,
    validate_distribution_object,
    validate_object_role,
)
from .results import SingleTargetTaskResult, TaskAttribute, TaskResultList, TaskTarget
from .runner import TaskRegistry, TaskRunner, TParams, TResult, TResultModel, run_tasks
from .search import SearchNeighborhood
from .source_target import (
    AnySourceAttribute,
    AnyTargetAttribute,
    AttributeExpression,
    CreateAttribute,
    ExistingAttributeExpression,
    GeoscienceObjectReference,
    Source,
    Target,
    UpdateAttribute,
    attribute_spec,
)

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
    "ExistingAttributeExpression",
    "Filter",
    "FilterCondition",
    "FilterExpression",
    "FilterOperator",
    "FiniteFloat",
    "GeoscienceObjectReference",
    "OutputTarget",
    "QuantileFloat",
    "Rotation",
    "SearchNeighborhood",
    "SingleTargetTaskResult",
    "Source",
    "SourceTargetTaskParameters",
    "TParams",
    "TResult",
    "TResultModel",
    "Target",
    "TaskAttribute",
    "TaskInputModel",
    "TaskParameters",
    "TaskRegistry",
    "TaskResultList",
    "TaskRunner",
    "TaskTarget",
    "UpdateAttribute",
    "attribute_spec",
    "canonicalize_attribute_selector",
    "check_batch_output_collisions",
    "deduplicate_aliases",
    "extract_typed_object",
    "object_identity",
    "resolve_output_target",
    "run_tasks",
    "validate_distribution_object",
    "validate_object_role",
]
