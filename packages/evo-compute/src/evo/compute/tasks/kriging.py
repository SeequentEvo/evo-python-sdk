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
    >>> result = await run(manager, params)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, TypeVar, overload

from evo.common import IContext
from evo.common.interfaces import IFeedback
from evo.common.utils import NoFeedback, Retry, split_feedback

from ..client import JobClient

# Import shared components
from .common import (
    CreateAttribute,
    Ellipsoid,
    EllipsoidRanges,
    Rotation,
    SearchNeighborhood,
    Source,
    Target,
    UpdateAttribute,
)
from .common.source_target import GeoscienceObjectReference, _serialize_object_reference

__all__ = [
    # Kriging-specific (users import from evo.compute.tasks.kriging)
    "KrigingMethod",
    "KrigingParameters",
    "OrdinaryKriging",
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
# Kriging Parameters
# =============================================================================


@dataclass
class KrigingParameters:
    """Parameters for the kriging task.

    Defines all inputs needed to run a kriging interpolation task.

    Example:
        >>> from evo.compute.tasks import run, SearchNeighborhood, Ellipsoid, EllipsoidRanges
        >>> from evo.compute.tasks.kriging import KrigingParameters
        >>>
        >>> params = KrigingParameters(
        ...     source=pointset.attributes["grade"],  # Source attribute
        ...     target=Target.new_attribute(block_model, "kriged_grade"),
        ...     variogram=variogram,  # Variogram model
        ...     search=SearchNeighborhood(
        ...         ellipsoid=Ellipsoid(ranges=EllipsoidRanges(200, 150, 100)),
        ...         max_samples=20,
        ...     ),
        ...     # method defaults to ordinary kriging
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

    def __init__(
        self,
        source: Source | Any,  # Also accepts Attribute from evo.objects.typed
        target: Target,
        variogram: GeoscienceObjectReference,
        search: SearchNeighborhood,
        method: SimpleKriging | OrdinaryKriging | None = None,
    ):
        # Handle Attribute type from evo.objects.typed.dataset
        if hasattr(source, "_obj") and hasattr(source, "expression"):
            # source is an Attribute, construct a Source object
            source = Source(object=source._obj.metadata.url, attribute=source.expression)

        self.source = source
        self.target = target
        self.variogram = variogram
        self.search = search
        self.method = method or OrdinaryKriging()

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "source": self.source.to_dict(),
            "target": self.target.to_dict(),
            "variogram": _serialize_object_reference(self.variogram),
            "neighborhood": self.search.to_dict(),
            "kriging_method": self.method.to_dict(),
        }


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

    def _get_portal_url(self) -> str | None:
        """Extract Portal URL from the target reference."""
        ref = self._target.reference
        if not ref:
            return None
        try:
            from urllib.parse import urlparse

            parsed = urlparse(ref)
            if parsed.scheme != "evo":
                return None
            parts = parsed.path.split("/")
            if "orgs" in parts and "workspaces" in parts and "objects" in parts:
                org_idx = parts.index("orgs") + 1
                ws_idx = parts.index("workspaces") + 1
                obj_idx = parts.index("objects") + 1
                org_id = parts[org_idx]
                workspace_id = parts[ws_idx]
                object_id = parts[obj_idx]
                hub = parsed.netloc
                if "int" in hub or "integration" in hub or "qa" in hub:
                    portal_base = "https://evo.integration.seequent.com"
                else:
                    portal_base = "https://evo.seequent.com"
                return f"{portal_base}/{org_id}/workspaces/workspace/{workspace_id}/overview?id={object_id}"
        except Exception:
            pass
        return None

    def _get_result_type_name(self) -> str:
        """Get the display name for this result type."""
        return "Task"

    def _repr_html_(self) -> str:
        """Generate HTML representation for Jupyter notebooks."""
        portal_url = self._get_portal_url()

        links_html = ""
        if portal_url:
            links_html = f'<a href="{portal_url}" target="_blank">Portal</a>'

        html = """
<style>
    .task-result {
        border: 1px solid #ccc;
        border-radius: 3px;
        padding: 16px;
        margin: 8px 0;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        font-size: 13px;
        display: inline-block;
        max-width: 800px;
        background-color: var(--jp-layout-color1, #fff);
    }
    .task-result .title {
        font-size: 15px;
        font-weight: 600;
        margin-bottom: 12px;
        color: var(--jp-ui-font-color1, #111);
        display: flex;
        align-items: baseline;
        gap: 8px;
    }
    .task-result .title-links {
        font-size: 12px;
        font-weight: normal;
    }
    .task-result .title-links a {
        color: #666;
        text-decoration: none;
    }
    .task-result .title-links a:hover {
        color: #0066cc;
        text-decoration: underline;
    }
    .task-result table {
        border-collapse: collapse;
        width: 100%;
    }
    .task-result td.label {
        padding: 3px 8px 3px 0;
        font-weight: 600;
        white-space: nowrap;
        vertical-align: top;
        color: var(--jp-ui-font-color1, #333);
    }
    .task-result td.value {
        padding: 3px 0;
        color: var(--jp-ui-font-color1, #111);
    }
    .task-result .attr-highlight {
        background: #e3f2fd;
        padding: 2px 8px;
        border-radius: 3px;
        font-family: monospace;
        font-weight: 600;
        color: #1565c0;
    }
    .task-result .message {
        background: #e8f5e9;
        padding: 6px 10px;
        border-radius: 3px;
        color: #2e7d32;
        margin-bottom: 12px;
        font-size: 12px;
    }
</style>
<div class="task-result">
"""
        title = f"✓ {self._get_result_type_name()} Result"
        if links_html:
            html += f'<div class="title"><span>{title}</span><span class="title-links">{links_html}</span></div>'
        else:
            html += f'<div class="title">{title}</div>'

        html += f'<div class="message">{self.message}</div>'

        html += '<table>'
        html += f'<tr><td class="label">Target:</td><td class="value">{self.target_name}</td></tr>'
        html += f'<tr><td class="label">Schema:</td><td class="value">{self.schema_type}</td></tr>'
        html += f'<tr><td class="label">Attribute:</td><td class="value"><span class="attr-highlight">{self.attribute_name}</span></td></tr>'
        html += '</table>'

        html += '</div>'
        return html

    def __repr__(self) -> str:
        """String representation."""
        portal_url = self._get_portal_url()
        lines = [
            f"✓ {self._get_result_type_name()} Result",
            f"  Message:   {self.message}",
            f"  Target:    {self.target_name}",
            f"  Attribute: {self.attribute_name}",
        ]
        if portal_url:
            lines.append(f"  Portal:    {portal_url}")
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

    def _repr_html_(self) -> str:
        """Generate HTML representation for Jupyter notebooks."""
        if not self._results:
            return "<div>No results</div>"

        result_type = self._results[0]._get_result_type_name() if self._results else "Task"
        html = f"""
<style>
    .task-results {{
        border: 1px solid #ccc;
        border-radius: 3px;
        padding: 16px;
        margin: 8px 0;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        font-size: 13px;
        max-width: 800px;
        background-color: var(--jp-layout-color1, #fff);
    }}
    .task-results .title {{
        font-size: 15px;
        font-weight: 600;
        margin-bottom: 12px;
        color: var(--jp-ui-font-color1, #111);
    }}
    .task-results table {{
        border-collapse: collapse;
        width: 100%;
    }}
    .task-results th {{
        text-align: left;
        padding: 6px 8px;
        border-bottom: 2px solid #ccc;
        color: var(--jp-ui-font-color1, #333);
    }}
    .task-results td {{
        padding: 6px 8px;
        border-bottom: 1px solid #eee;
        color: var(--jp-ui-font-color1, #111);
    }}
    .task-results .attr-highlight {{
        background: #e3f2fd;
        padding: 2px 6px;
        border-radius: 3px;
        font-family: monospace;
        font-size: 12px;
        color: #1565c0;
    }}
    .task-results .success {{
        color: #2e7d32;
    }}
</style>
<div class="task-results">
    <div class="title">✓ {len(self._results)} {result_type} Results</div>
    <table>
        <tr>
            <th>#</th>
            <th>Target</th>
            <th>Attribute</th>
            <th>Schema</th>
        </tr>
"""
        for i, result in enumerate(self._results):
            html += f"""
        <tr>
            <td>{i + 1}</td>
            <td>{result.target_name}</td>
            <td><span class="attr-highlight">{result.attribute_name}</span></td>
            <td>{result.schema_type}</td>
        </tr>
"""
        html += """
    </table>
</div>
"""
        return html

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
    polling_interval_seconds: float = 0.5,
    retry: Retry | None = None,
    fb: IFeedback = NoFeedback,
) -> KrigingResult:
    """Internal function to run a single kriging task."""
    connector = context.get_connector()
    org_id = context.get_org_id()

    # Add API-Preview header for preview API
    if connector._additional_headers is None:
        connector._additional_headers = {}
    connector._additional_headers["API-Preview"] = "opt-in"

    params_dict = parameters.to_dict()

    # Submit the job
    job = await JobClient.submit(
        connector=connector,
        org_id=org_id,
        topic="geostatistics",
        task="kriging",
        parameters=params_dict,
        result_type=dict,  # Get raw dict, we'll parse it ourselves
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


async def _run_kriging_for_registry(context: IContext, parameters: KrigingParameters) -> KrigingResult:
    """Simplified runner function for task registry (no extra options).

    This is the function registered with the TaskRegistry. For more control
    over polling and retry behavior, use the full `run()` function.
    """
    return await _run_single_kriging(context, parameters)


# Register kriging task runner with the task registry
from .common.runner import register_task_runner
register_task_runner(KrigingParameters, _run_kriging_for_registry)
