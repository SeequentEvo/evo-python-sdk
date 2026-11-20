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

Example::

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

from collections.abc import Mapping, Sequence
from typing import Annotated, Any, Literal

from evo.objects.typed.attributes import BlockModelPendingAttribute, PendingAttribute
from evo.objects.typed.base import BaseObject
from pydantic import Field, PrivateAttr, ValidatorFunctionWrapHandler, field_validator, model_validator

from .models import TaskInputModel, object_identity
from .source_target import AnyTypedAttribute, ExistingAttributeExpression, _get_existing_attribute_expression

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


class FilterCondition(TaskInputModel):
    """A leaf filter condition applied to a single attribute of the filtered object.

    Provide exactly one operator and the matching payload:

    - ``"in"`` / ``"not_in"``: membership tests — pair with ``values`` (a list of
      string category names or integer category keys).
    - ``"equal"``, ``"not_equal"``, ``"greater_than"``, ``"greater_than_or_equal_to"``,
      ``"less_than"``, ``"less_than_or_equal_to"``: numeric comparisons — pair with
      ``threshold`` (a single float).

    Example::

        >>> FilterCondition(attribute=grid.attributes["domain"], operator="in", values=["LMS1"])
        >>> FilterCondition(attribute=grid.attributes["grade"], operator="greater_than", threshold=0.5)
    """

    _attribute_object_url: str | None = PrivateAttr(default=None)

    type: Literal["condition"] = "condition"
    """Discriminator identifying this as a leaf filter condition."""

    attribute: ExistingAttributeExpression
    """The attribute on the filtered object to evaluate the condition against."""

    operator: FilterOperator
    """The filter operator. Use ``in``/``not_in`` with ``values``; numeric operators with ``threshold``."""

    values: list[str | int] | None = None
    """Category values to match. Required for ``in``/``not_in``. Strings for category names, ints for category keys."""

    threshold: float | None = None
    """Numeric threshold. Required for the numeric comparison operators."""

    @field_validator("values", mode="before")
    @classmethod
    def _reject_bool_values(cls, raw: Any) -> Any:
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            if any(isinstance(item, bool) for item in raw):
                raise ValueError("Filter 'values' must contain category names (str) or keys (int), not booleans.")
        return raw

    @model_validator(mode="wrap")
    @classmethod
    def _remember_attribute_object(cls, data: Any, handler: ValidatorFunctionWrapHandler) -> FilterCondition:
        attr_url: str | None = None
        if isinstance(data, FilterCondition):
            attr_url = data._attribute_object_url
        elif isinstance(data, Mapping):
            raw_attr = data.get("attribute")
            if isinstance(raw_attr, AnyTypedAttribute) and raw_attr._obj is not None:
                try:
                    attr_url = str(raw_attr._obj.metadata.url)
                except Exception:
                    attr_url = None
        condition = handler(data)
        if attr_url is not None and condition._attribute_object_url is None:
            condition._attribute_object_url = attr_url
        return condition

    @model_validator(mode="after")
    def _validate_operator_payload(self) -> FilterCondition:
        if self.operator in _MEMBERSHIP_OPERATORS:
            if self.values is None or len(self.values) == 0:
                raise ValueError(
                    f"Operator '{self.operator}' requires a non-empty 'values' list (category names or integer keys)."
                )
            if self.threshold is not None:
                raise ValueError(f"Operator '{self.operator}' does not accept 'threshold'.")
        else:
            if self.threshold is None:
                raise ValueError(f"Operator '{self.operator}' requires 'threshold' (a numeric value).")
            if self.values is not None:
                raise ValueError(f"Operator '{self.operator}' does not accept 'values'.")
        return self


class AllOfFilter(TaskInputModel):
    """A composite filter that passes locations satisfying ALL child filters (AND logic)."""

    type: Literal["all_of"] = "all_of"
    """Discriminator identifying this as an AND composite filter."""

    filters: list[FilterExpression] = Field(min_length=1)
    """The child expressions that must all be satisfied."""


class AnyOfFilter(TaskInputModel):
    """A composite filter that passes locations satisfying ANY child filter (OR logic)."""

    type: Literal["any_of"] = "any_of"
    """Discriminator identifying this as an OR composite filter."""

    filters: list[FilterExpression] = Field(min_length=1)
    """The child expressions, at least one of which must be satisfied."""


FilterExpression = Annotated[
    FilterCondition | AllOfFilter | AnyOfFilter,
    Field(discriminator="type"),
]
"""A filter condition or composite filter combining multiple conditions."""


def _check_expression_object_url(expr: FilterExpression, expected_url: str) -> None:
    if isinstance(expr, (AllOfFilter, AnyOfFilter)):
        for child in expr.filters:
            _check_expression_object_url(child, expected_url)
        return
    if expr._attribute_object_url is not None and object_identity(expr._attribute_object_url) != object_identity(
        expected_url
    ):
        raise ValueError(
            f"Filter condition references an attribute from object {object_identity(expr._attribute_object_url)!r}, "
            f"which does not match filtered object {object_identity(expected_url)!r}."
        )


def _resolve_expression(expr: FilterExpression, obj: BaseObject) -> FilterExpression:
    if isinstance(expr, AllOfFilter):
        return AllOfFilter(filters=[_resolve_expression(child, obj) for child in expr.filters])
    if isinstance(expr, AnyOfFilter):
        return AnyOfFilter(filters=[_resolve_expression(child, obj) for child in expr.filters])

    if expr._attribute_object_url is not None:
        try:
            obj_url = str(obj.metadata.url)
        except Exception:
            obj_url = None
        if obj_url is not None and object_identity(expr._attribute_object_url) != object_identity(obj_url):
            raise ValueError(
                f"Filter condition references an attribute from object "
                f"{object_identity(expr._attribute_object_url)!r}, which does not match "
                f"filtered object {object_identity(obj_url)!r}."
            )

    if "[" in expr.attribute:
        resolved = FilterCondition(
            type=expr.type,
            attribute=expr.attribute,
            operator=expr.operator,
            values=expr.values,
            threshold=expr.threshold,
        )
        resolved._attribute_object_url = expr._attribute_object_url
        return resolved

    attributes = getattr(obj, "attributes", None)
    if attributes is None:
        raise ValueError(
            f"Cannot resolve filter attribute {expr.attribute!r} on {type(obj).__name__}: object has no attributes."
        )

    attr = attributes[expr.attribute]
    if isinstance(attr, (PendingAttribute, BlockModelPendingAttribute)) or not getattr(attr, "exists", False):
        available = [name for a in attributes if isinstance(name := getattr(a, "name", None), str)]
        hint = f" Available attributes: {available}." if available else ""
        raise ValueError(f"Filter attribute {expr.attribute!r} does not exist on {type(obj).__name__}.{hint}")

    resolved = FilterCondition(
        type=expr.type,
        attribute=_get_existing_attribute_expression(attr),
        operator=expr.operator,
        values=expr.values,
        threshold=expr.threshold,
    )
    try:
        resolved._attribute_object_url = str(obj.metadata.url)
    except Exception:
        pass
    return resolved


class Filter(TaskInputModel):
    """A filter expression defining which locations to include in a computation."""

    where: FilterExpression
    """The root filter expression. A condition, or an ``all_of`` / ``any_of`` composite."""

    def resolve(self, obj: BaseObject) -> Filter:
        """Resolve bare attribute names in this filter against ``obj.attributes``.

        Already-constructed JMESPath expressions (containing ``"["``) are preserved unchanged.

        Args:
            obj: The typed geoscience object whose attributes should be searched.

        Returns:
            A new :class:`Filter` with bare attribute names resolved to JMESPath expressions.

        Raises:
            TypeError: If *obj* is not a :class:`BaseObject`.
            ValueError: If any bare attribute name does not exist on *obj*, or if a typed
                attribute in the filter belongs to a different object.
        """
        if not isinstance(obj, BaseObject):
            raise TypeError(f"Expected a BaseObject instance to resolve filter attributes, got {type(obj).__name__}.")
        return Filter(where=_resolve_expression(self.where, obj))

    def _check_object_url(self, expected_url: str) -> None:
        _check_expression_object_url(self.where, expected_url)


# ``FilterExpression`` is recursive: it includes ``AllOfFilter``/``AnyOfFilter``, which
# in turn hold ``filters: list[FilterExpression]``. This cycle forces a forward reference —
# the composite classes are built before ``FilterExpression`` is defined below, so Pydantic
# initially leaves their schemas incomplete with an unresolved reference. ``model_rebuild()``
# re-resolves it now that ``FilterExpression`` exists, finalizing the schema so nested
# expressions are validated and parsed as ``FilterExpression``.
AllOfFilter.model_rebuild()
AnyOfFilter.model_rebuild()
