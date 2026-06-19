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

"""Generic attribute filter expressions shared across geostatistics compute tasks.

Several tasks (e.g. ``kriging`` and ``conditioned_simulator``) allow restricting a
computation to a subset of locations by filtering an attribute on the source or
target object.  A filter is a tree of expressions: a single
:class:`FilterCondition`, or a composite :class:`AllOfFilter` (AND) / :class:`AnyOfFilter`
(OR) combining nested expressions.

Example:
    >>> from evo.compute.tasks.common import Filter, FilterCondition, AllOfFilter
    >>>
    >>> # Single condition: include only locations where 'domain' is LMS1 or LMS2
    >>> f = Filter(where=FilterCondition(attribute=grid.attributes["domain"], operator="in", values=["LMS1", "LMS2"]))
    >>>
    >>> # Compound condition: 'domain' in {1, 2} AND 'grade' >= 0.5
    >>> f = Filter(
    ...     where=AllOfFilter(
    ...         filters=[
    ...             FilterCondition(attribute=grid.attributes["domain"], operator="in", values=[1, 2]),
    ...             FilterCondition(attribute=grid.attributes["grade"], operator="greater_than_or_equal_to", threshold=0.5),
    ...         ]
    ...     )
    ... )
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator

from .source_target import AttributeExpression

__all__ = [
    "AllOfFilter",
    "AnyOfFilter",
    "Filter",
    "FilterCondition",
    "FilterExpression",
    "FilterOperator",
]

FilterOperator = Literal[
    "in",
    "not_in",
    "equal",
    "not_equal",
    "greater_than",
    "greater_than_or_equal_to",
    "less_than",
    "less_than_or_equal_to",
]
"""Supported filter operators.

Membership operators (``"in"``, ``"not_in"``) require ``values``; all other
(numeric comparison) operators require ``threshold``.
"""

_MEMBERSHIP_OPERATORS: frozenset[str] = frozenset(("in", "not_in"))


class FilterCondition(BaseModel):
    """A leaf filter condition applied to a single attribute of the filtered object.

    Provide exactly one operator and the matching payload:

    - ``"in"`` / ``"not_in"``: membership tests — pair with ``values`` (a list of
      string category names or integer category keys).
    - ``"equal"``, ``"not_equal"``, ``"greater_than"``, ``"greater_than_or_equal_to"``,
      ``"less_than"``, ``"less_than_or_equal_to"``: numeric comparisons — pair with
      ``threshold`` (a single float).

    Example:
        >>> FilterCondition(attribute=grid.attributes["domain"], operator="in", values=["LMS1"])
        >>> FilterCondition(attribute=grid.attributes["grade"], operator="greater_than", threshold=0.5)
    """

    type: Literal["condition"] = "condition"
    """Discriminator identifying this as a leaf filter condition."""

    attribute: AttributeExpression
    """The attribute on the filtered object to evaluate the condition against."""

    operator: FilterOperator
    """The filter operator. Use ``in``/``not_in`` with ``values``; numeric operators with ``threshold``."""

    values: list[str | int] | None = None
    """Category values to match. Required for ``in``/``not_in``. Strings for category names, ints for category keys."""

    threshold: float | None = None
    """Numeric threshold. Required for the numeric comparison operators."""

    @model_validator(mode="after")
    def _validate_operator_payload(self) -> FilterCondition:
        if self.operator in _MEMBERSHIP_OPERATORS:
            if self.values is None:
                raise ValueError(
                    f"Operator '{self.operator}' requires 'values' (list of category names or integer keys)."
                )
            if self.threshold is not None:
                raise ValueError(f"Operator '{self.operator}' does not accept 'threshold'.")
        else:
            if self.threshold is None:
                raise ValueError(f"Operator '{self.operator}' requires 'threshold' (a numeric value).")
            if self.values is not None:
                raise ValueError(f"Operator '{self.operator}' does not accept 'values'.")
        return self


class AllOfFilter(BaseModel):
    """A composite filter that passes locations satisfying ALL child filters (AND logic)."""

    type: Literal["all_of"] = "all_of"
    """Discriminator identifying this as an AND composite filter."""

    filters: list[FilterExpression]
    """The child expressions that must all be satisfied."""


class AnyOfFilter(BaseModel):
    """A composite filter that passes locations satisfying ANY child filter (OR logic)."""

    type: Literal["any_of"] = "any_of"
    """Discriminator identifying this as an OR composite filter."""

    filters: list[FilterExpression]
    """The child expressions, at least one of which must be satisfied."""


FilterExpression = Annotated[
    FilterCondition | AllOfFilter | AnyOfFilter,
    Field(discriminator="type"),
]
"""A filter condition or composite filter combining multiple conditions."""


class Filter(BaseModel):
    """A filter expression defining which locations to include in a computation."""

    where: FilterExpression
    """The root filter expression. A condition, or an ``all_of`` / ``any_of`` composite."""


# ``FilterExpression`` is recursive: it includes ``AllOfFilter``/``AnyOfFilter``, which
# in turn hold ``filters: list[FilterExpression]``. This cycle forces a forward reference —
# the composite classes are built before ``FilterExpression`` is defined below, so Pydantic
# initially leaves their schemas incomplete with an unresolved reference. ``model_rebuild()``
# re-resolves it now that ``FilterExpression`` exists, finalizing the schema so nested
# expressions are validated and parsed as ``FilterExpression``.
AllOfFilter.model_rebuild()
AnyOfFilter.model_rebuild()
