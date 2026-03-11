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

from typing import Generic, Iterator, TypeVar, overload

from pydantic import BaseModel

T = TypeVar("T")

__all__ = [
    "TaskAttribute",
    "TaskResultList",
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
    description: str | None = None
    schema_id: str
    attribute: TaskAttribute


class TaskResultList(Generic[T]):
    """A list-like container for task results with pretty-printing support.

    Wraps a plain list of result objects so that Jupyter can render them
    with rich HTML via ``_repr_html_``.  When ``evo.widgets`` is loaded its
    ``for_type_by_name`` registration overrides the built-in HTML with the
    fully-styled version.

    Supports indexing, iteration, ``len()``, and ``bool()`` just like a
    regular list.

    Example:
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
            target = getattr(result, "target_name", "?")
            attr = getattr(result, "attribute_name", "?")
            lines.append(f"  {i + 1}. {target} → {attr}")
        return "\n".join(lines)
