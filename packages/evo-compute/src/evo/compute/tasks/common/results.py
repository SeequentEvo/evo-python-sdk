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

"""Common result types for compute tasks.

This module provides base result classes that all compute task types can inherit
from. These were originally defined in the kriging module but are generic enough
for any task type.
"""

from __future__ import annotations

from typing import Any

from evo.common import IContext
from evo.objects.data import ObjectSchema
from evo.objects.exceptions import SchemaIDFormatError
from evo.objects.typed import object_from_reference
from evo.objects.typed.spatial import BaseSpatialObject
from pydantic import BaseModel, PrivateAttr

__all__ = [
    "TaskAttribute",
    "TaskResult",
    "TaskResults",
    "TaskTarget",
]


class TaskAttribute(BaseModel):
    """Attribute information from a task result."""

    reference: str
    name: str


class TaskTarget(BaseModel):
    """Target information from a task result."""

    reference: str
    name: str
    description: Any = None
    schema_id: str
    attribute: TaskAttribute


class TaskResult(BaseModel):
    """Base class for compute task results.

    Provides common functionality for all task results including:
    - Pretty-printing in Jupyter notebooks
    - Portal URL extraction
    - Access to target object and data
    """

    model_config = {"arbitrary_types_allowed": True}

    message: str
    """A message describing what happened in the task."""

    target: TaskTarget
    """Target information from the task result."""

    _context: IContext | None = PrivateAttr(default=None)
    """The context used to run the task (for convenience methods)."""

    @property
    def target_name(self) -> str:
        """The name of the target object."""
        return self.target.name

    @property
    def target_reference(self) -> str:
        """Reference URL to the target object."""
        return self.target.reference

    @property
    def attribute_name(self) -> str:
        """The name of the attribute that was created/updated."""
        return self.target.attribute.name

    @property
    def schema_type(self) -> str:
        """The schema type of the target object (e.g., 'regular-masked-3d-grid').

        Uses ``ObjectSchema.from_id`` to parse the schema ID. Falls back to the
        raw ``schema_id`` string when it cannot be parsed.
        """
        schema = self.target.schema_id
        try:
            parsed = ObjectSchema.from_id(schema)
            return parsed.sub_classification
        except SchemaIDFormatError:
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
        ctx = context or self._context
        if ctx is None:
            raise ValueError(
                "No context available. Either pass a context to get_target_object() "
                "or ensure the result was returned from run()."
            )
        return await object_from_reference(ctx, self.target.reference)

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

        if not isinstance(target_obj, BaseSpatialObject):
            raise TypeError(
                f"Don't know how to get DataFrame from {type(target_obj).__name__}. "
                "Use get_target_object() and access the data manually."
            )

        if columns is not None:
            return await target_obj.to_dataframe(columns=columns)
        return await target_obj.to_dataframe()

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
