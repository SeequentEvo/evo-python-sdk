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

from typing import ClassVar, Generic, Iterator, Protocol, TypeVar, overload, runtime_checkable

import pandas as pd
from evo.common import IContext, IFeedback
from evo.objects import ObjectSchema
from evo.objects.typed import BaseObject, BlockModel, object_from_reference
from pydantic import BaseModel

T = TypeVar("T")
TTarget_co = TypeVar("TTarget_co", bound="TaskTarget", covariant=True)

__all__ = [
    "SingleTargetTaskResult",
    "TaskAttribute",
    "TaskResultList",
    "TaskTarget",
]


@runtime_checkable
class _ObjToDataframeProtocol(Protocol):
    async def to_dataframe(self, *keys: str, fb: IFeedback = ...) -> pd.DataFrame: ...


class _HasTargetAndMessage(Protocol[TTarget_co]):
    @property
    def message(self) -> str: ...

    @property
    def target(self) -> TTarget_co: ...


class TaskAttribute(BaseModel):
    """Attribute information from a task result."""

    reference: str
    name: str


class TaskTarget(BaseModel):
    """Target information from a task result."""

    reference: str
    name: str
    description: str | None = None
    schema_id: str
    attribute: TaskAttribute


class SingleTargetTaskResult(Generic[TTarget_co]):
    """Shared result wrapper for tasks that write a primary attribute onto a target object."""

    TASK_DISPLAY_NAME: ClassVar[str] = "Task"

    def __init__(self, context: IContext, model: _HasTargetAndMessage[TTarget_co]) -> None:
        self._target: TTarget_co = model.target
        self._message: str = model.message
        self._context: IContext = context

    @property
    def message(self) -> str:
        """A message describing what happened in the task."""
        return self._message

    @property
    def target(self) -> TTarget_co:
        """Target metadata returned by the task."""
        return self._target

    @property
    def target_name(self) -> str:
        """The name of the target object."""
        return self._target.name

    @property
    def target_reference(self) -> str:
        """Reference URL to the target object."""
        return self._target.reference

    @property
    def target_description(self) -> str | None:
        """Optional description of the target object."""
        return self._target.description

    @property
    def attribute(self) -> TaskAttribute:
        """The attribute metadata that was created or updated."""
        return self._target.attribute

    @property
    def attribute_name(self) -> str:
        """The name of the attribute that was created or updated."""
        return self._target.attribute.name

    @property
    def attribute_reference(self) -> str:
        """JMESPath reference to the attribute that was created or updated."""
        return self._target.attribute.reference

    @property
    def schema(self) -> ObjectSchema:
        """The schema type of the target object."""
        return ObjectSchema.from_id(self._target.schema_id)

    async def get_target_object(self) -> BaseObject:
        """Load and return the target geoscience object."""
        return await object_from_reference(self._context, self._target.reference)

    async def to_dataframe(self, *columns: str) -> pd.DataFrame:
        """Load the target object and return its data as a DataFrame."""
        target_obj = await self.get_target_object()
        if isinstance(target_obj, BlockModel) and hasattr(target_obj, "to_dataframe"):
            return await target_obj.to_dataframe(columns=list(columns) if columns else None)
        if isinstance(target_obj, _ObjToDataframeProtocol):
            return await target_obj.to_dataframe(*columns)
        raise TypeError(
            f"Don't know how to get DataFrame from {type(target_obj).__name__}. "
            "Use get_target_object() and access the data manually."
        )


class TaskResultList(Generic[T]):
    """A list-like container for task results with pretty-printing support.

    Wraps a plain list of result objects so that Jupyter can render them
    with rich HTML via ``_repr_html_``.  When ``evo.widgets`` is loaded its
    ``for_type_by_name`` registration overrides the built-in HTML with the
    fully-styled version.

    Supports indexing, iteration, ``len()``, and ``bool()`` just like a
    regular list.

    Example::

        >>> results = await run(manager, [params1, params2], preview=True)
        >>> results        # pretty-printed table in Jupyter
        >>> results[0]     # access individual result
        >>> len(results)   # number of results
    """

    def __init__(self, results: list[T]) -> None:
        self._results = results

    @overload
    def __getitem__(self, index: int) -> T: ...

    @overload
    def __getitem__(self, index: slice) -> list[T]: ...

    def __getitem__(self, index: int | slice) -> T | list[T]:
        return self._results[index]

    def __len__(self) -> int:
        return len(self._results)

    def __iter__(self) -> Iterator[T]:
        return iter(self._results)

    def __bool__(self) -> bool:
        return len(self._results) > 0

    def __repr__(self) -> str:
        if not self._results:
            return "TaskResultList([])"
        result_type = getattr(self._results[0], "TASK_DISPLAY_NAME", "Task")
        return f"TaskResultList([{len(self._results)} {result_type} result(s)])"

    def __str__(self) -> str:
        if not self._results:
            return "No results"
        result_type = getattr(self._results[0], "TASK_DISPLAY_NAME", "Task")
        lines = [f"✓ {len(self._results)} {result_type} Result(s)"]
        for i, result in enumerate(self._results):
            target = getattr(result, "target_name", None)
            attr = getattr(result, "attribute_name", None)
            if target is not None and attr is not None:
                lines.append(f"  {i + 1}. {target} → {attr}")
            elif target is not None and (attr_names := getattr(result, "attribute_names", None)):
                lines.append(f"  {i + 1}. {target} → {', '.join(attr_names)}")
            elif target is not None and (summary := getattr(result, "summary_attributes", None)) is not None:
                lines.append(f"  {i + 1}. {target} → {summary.mean.name}")
            elif target is not None and (sims := getattr(result, "simulations_attribute", None)) is not None:
                lines.append(f"  {i + 1}. {target} → {sims.name}")
            elif (dist_name := getattr(result, "distribution_name", None)) is not None:
                lines.append(f"  {i + 1}. {dist_name}")
            elif (report_ref := getattr(result, "report_reference", None)) is not None:
                lines.append(f"  {i + 1}. {report_ref}")
            else:
                lines.append(f"  {i + 1}. {target or '?'} → {attr or '?'}")
        return "\n".join(lines)
