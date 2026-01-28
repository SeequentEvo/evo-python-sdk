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

"""Task-specific clients for Evo Compute.

This module provides a unified interface for running compute tasks. Tasks are
dispatched based on their parameter types using a registry system.

Example:
    >>> from evo.compute.tasks import run, SearchNeighborhood, Target
    >>> from evo.compute.tasks.kriging import KrigingParameters
    >>>
    >>> # Run a single task
    >>> result = await run(manager, KrigingParameters(...))
    >>>
    >>> # Run multiple tasks (same or different types)
    >>> results = await run(manager, [
    ...     KrigingParameters(...),
    ...     KrigingParameters(...),
    ... ])
"""

from __future__ import annotations

from typing import Any, overload

from evo.common import IContext
from evo.common.interfaces import IFeedback
from evo.common.utils import NoFeedback, split_feedback

# Shared components from common module
from .common import (
    CreateAttribute,
    Ellipsoid,
    EllipsoidRanges,
    Rotation,
    SearchNeighborhood,
    Source,
    Target,
    UpdateAttribute,
    run_tasks,
)

# Import kriging module to trigger registration
from . import kriging as _kriging_module  # noqa: F401

# Result types from kriging (these are general enough for other tasks too)
from .kriging import (
    KrigingResult,
    TaskResult,
    TaskResults,
)


class _DefaultFeedback:
    """Marker class to indicate default feedback should be used."""
    pass


DEFAULT_FEEDBACK = _DefaultFeedback()


@overload
async def run(
    context: IContext,
    parameters: Any,
    *,
    fb: IFeedback | _DefaultFeedback = ...,
) -> TaskResult: ...


@overload
async def run(
    context: IContext,
    parameters: list[Any],
    *,
    fb: IFeedback | _DefaultFeedback = ...,
) -> TaskResults: ...


async def run(
    context: IContext,
    parameters: Any | list[Any],
    *,
    fb: IFeedback | _DefaultFeedback = DEFAULT_FEEDBACK,
) -> TaskResult | TaskResults:
    """
    Run one or more compute tasks.

    Tasks are dispatched to the appropriate runner based on the parameter type.
    This allows running different task types together in a single call.

    Args:
        context: The context providing connector and org_id
        parameters: A single parameter object or list of parameters (can be mixed types)
        fb: Feedback interface for progress updates. If not provided, uses default
            feedback showing "Running x/y..."

    Returns:
        TaskResult for a single task, or TaskResults for multiple tasks

    Example (single task):
        >>> from evo.compute.tasks import run, SearchNeighborhood, Target
        >>> from evo.compute.tasks.kriging import KrigingParameters
        >>>
        >>> params = KrigingParameters(
        ...     source=pointset.attributes["grade"],
        ...     target=Target.new_attribute(block_model, "kriged_grade"),
        ...     variogram=variogram,
        ...     search=SearchNeighborhood(
        ...         ellipsoid=var_ell.scaled(2.0),
        ...         max_samples=20,
        ...     ),
        ... )
        >>> result = await run(manager, params)

    Example (multiple tasks):
        >>> results = await run(manager, [
        ...     KrigingParameters(...),
        ...     KrigingParameters(...),
        ... ])
        >>> results[0]  # Access first result
    """
    # Convert single parameter to list for uniform handling
    is_single = not isinstance(parameters, list)
    param_list = [parameters] if is_single else parameters

    if len(param_list) == 0:
        return TaskResults([])

    total = len(param_list)

    # Create default feedback
    if isinstance(fb, _DefaultFeedback):
        try:
            from evo.notebooks import FeedbackWidget
            actual_fb: IFeedback = FeedbackWidget(label=f"Running 0/{total}...")
        except ImportError:
            actual_fb = NoFeedback
    else:
        actual_fb = fb

    # Use the task registry to run tasks
    results = await run_tasks(context, param_list, fb=actual_fb)

    # Update feedback on completion
    if isinstance(fb, _DefaultFeedback) and hasattr(actual_fb, "description"):
        actual_fb.description = f"Running {total}/{total}..."

    # Return single result or wrapped results
    if is_single:
        return results[0]
    return TaskResults(results)


__all__ = [
    # Shared components
    "CreateAttribute",
    "Ellipsoid",
    "EllipsoidRanges",
    "Rotation",
    "SearchNeighborhood",
    "Source",
    "Target",
    "UpdateAttribute",
    # Run function and results
    "run",
    "TaskResult",
    "TaskResults",
    "KrigingResult",
]

