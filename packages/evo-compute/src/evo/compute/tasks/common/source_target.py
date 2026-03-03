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

"""Source and target specifications for compute tasks."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from evo.objects import DownloadedObject, ObjectMetadata, ObjectReference
from evo.objects.typed.attributes import (
    Attribute,
    BlockModelAttribute,
    BlockModelPendingAttribute,
    PendingAttribute,
)
from evo.objects.typed.base import BaseObject
from pydantic import BaseModel, BeforeValidator

__all__ = [
    "CreateAttribute",
    "GeoscienceObjectReference",
    "Source",
    "Target",
    "UpdateAttribute",
    "get_attribute_expression",
    "is_typed_attribute",
    "source_from_attribute",
    "target_from_attribute",
]

# All typed attribute types that compute tasks can work with.
TYPED_ATTRIBUTE_TYPES = (Attribute, PendingAttribute, BlockModelAttribute, BlockModelPendingAttribute)


def _convert_object_reference(value: Any) -> str:
    """Convert an object reference to a validated URL string.

    Used as a Pydantic ``BeforeValidator`` on :data:`GeoscienceObjectReference`
    fields so that any accepted input type is normalised to a validated URL
    string at model construction time.  The value is validated through
    :class:`ObjectReference` (which enforces strict URL structure) but stored
    as a plain ``str`` so pydantic can serialise it natively.

    Supports:
    - ``str`` / ``ObjectReference``: validated via ``ObjectReference(value)``
    - ``BaseObject`` (typed objects like PointSet, Regular3DGrid): ``value.metadata.url``
    - ``DownloadedObject``: ``value.metadata.url``
    - ``ObjectMetadata``: ``value.url``

    Args:
        value: The value to convert.

    Returns:
        A validated URL string.

    Raises:
        TypeError: If the value type is not supported.
    """
    if isinstance(value, str):
        # Validate the URL structure via ObjectReference, return as str
        return str(ObjectReference(value))

    if isinstance(value, (BaseObject, DownloadedObject)):
        return str(value.metadata.url)

    if isinstance(value, ObjectMetadata):
        return str(value.url)

    raise TypeError(f"Cannot convert object reference from type {type(value).__name__}")


# Annotated type: accepts str, BaseObject, DownloadedObject or ObjectMetadata
# and normalises to a validated URL string at validation time.
GeoscienceObjectReference = Annotated[str, BeforeValidator(_convert_object_reference)]


def is_typed_attribute(value: Any) -> bool:
    """Check if a value is a typed attribute object from evo.objects.typed."""
    return isinstance(value, TYPED_ATTRIBUTE_TYPES)


def get_attribute_expression(
    attr: Attribute | PendingAttribute | BlockModelAttribute | BlockModelPendingAttribute,
) -> str:
    """Get the JMESPath expression to access an attribute from its parent object.

    For ``Attribute`` (existing, from a DownloadedObject): uses the schema path context
    and key-based lookup, e.g. ``"locations.attributes[?key=='abc']"``.

    For ``PendingAttribute``, ``BlockModelAttribute``, or ``BlockModelPendingAttribute``:
    uses name-based lookup, e.g. ``"attributes[?name=='grade']"``.

    Args:
        attr: A typed attribute object.

    Returns:
        A JMESPath expression string.

    Raises:
        TypeError: If the attribute type is not recognised.
    """
    if isinstance(attr, Attribute):
        base_path = attr._context.schema_path or "attributes"
        return f"{base_path}[?key=='{attr.key}']"
    elif isinstance(attr, (PendingAttribute, BlockModelAttribute, BlockModelPendingAttribute)):
        return f"attributes[?name=='{attr.name}']"
    else:
        raise TypeError(f"Cannot get expression for attribute type {type(attr).__name__}")


class Source(BaseModel):
    """The source object and attribute containing known values.

    Used to specify where input data comes from for geostatistical operations.
    Can be initialized directly, or more commonly from a typed object's attribute.

    Example:
        >>> # From a typed object attribute (preferred):
        >>> source = pointset.attributes["grade"]
        >>>
        >>> # Or explicitly:
        >>> source = Source(object=pointset, attribute="grade")
    """

    object: GeoscienceObjectReference
    """Reference to the source geoscience object."""

    attribute: str
    """Name of the attribute on the source object."""


class CreateAttribute(BaseModel):
    """Specification for creating a new attribute on a target object."""

    operation: Literal["create"] = "create"
    """The operation type (always 'create')."""

    name: str
    """The name of the attribute to create."""


class UpdateAttribute(BaseModel):
    """Specification for updating an existing attribute on a target object."""

    operation: Literal["update"] = "update"
    """The operation type (always 'update')."""

    reference: str
    """Reference to an existing attribute to update."""


class Target(BaseModel):
    """The target object and attribute to create or update with results.

    Used to specify where output data should be written for geostatistical operations.

    Example:
        >>> # Create a new attribute on a target object:
        >>> target = Target.new_attribute(block_model, "kriged_grade")
        >>>
        >>> # Or update an existing attribute:
        >>> target = Target(object=grid, attribute=UpdateAttribute(reference="existing_ref"))
    """

    object: GeoscienceObjectReference
    """Object to write results onto."""

    attribute: CreateAttribute | UpdateAttribute
    """Attribute specification (create new or update existing)."""

    @classmethod
    def new_attribute(cls, object: GeoscienceObjectReference, attribute_name: str) -> Target:
        """Create a Target that will create a new attribute on the target object.

        Args:
            object: The target object to write results onto.
            attribute_name: The name of the new attribute to create.

        Returns:
            A Target instance configured to create a new attribute.

        Example:
            >>> target = Target.new_attribute(block_model, "kriged_grade")
        """
        return cls(object=object, attribute=CreateAttribute(name=attribute_name))


# =============================================================================
# Typed attribute → Source / Target conversion
# =============================================================================


def source_from_attribute(attr: Attribute | BlockModelAttribute) -> Source:
    """Convert a typed ``Attribute`` or ``BlockModelAttribute`` to a :class:`Source`.

    Only existing attributes can be used as a source, since source data must already exist.

    Args:
        attr: An existing ``Attribute`` from a DownloadedObject, or a ``BlockModelAttribute``.

    Returns:
        A :class:`Source` referencing the parent object and attribute expression.

    Raises:
        TypeError: If *attr* is not a supported attribute type, or if it has
            no ``_obj`` reference to its parent object.
    """
    if not isinstance(attr, (Attribute, BlockModelAttribute)):
        raise TypeError(f"Only Attribute or BlockModelAttribute can be used as a source, got {type(attr).__name__}")

    if attr._obj is None:
        raise TypeError(
            f"Cannot determine source object from attribute type {type(attr).__name__}. "
            "Attribute must have an _obj reference to its parent object."
        )

    return Source(
        object=str(attr._obj.metadata.url),
        attribute=get_attribute_expression(attr),
    )


def target_from_attribute(
    attr: Attribute | PendingAttribute | BlockModelAttribute | BlockModelPendingAttribute,
) -> Target:
    """Convert a typed attribute object to a :class:`Target`.

    Handles ``Attribute``, ``PendingAttribute``, ``BlockModelAttribute``, and
    ``BlockModelPendingAttribute`` from ``evo.objects.typed.attributes``.

    For existing attributes, returns an update operation referencing the attribute.
    For pending attributes, returns a create operation with the attribute name.

    Args:
        attr: A typed attribute object. Must have a non-``None`` ``_obj``
            reference to its parent object.

    Returns:
        A :class:`Target` configured based on the attribute.

    Raises:
        TypeError: If *attr* is not a recognised typed attribute, or if it has
            no ``_obj`` reference to its parent object.
    """
    if not is_typed_attribute(attr):
        raise TypeError(
            f"Cannot convert {type(attr).__name__} to a Target. "
            "Expected Attribute, PendingAttribute, BlockModelAttribute, or BlockModelPendingAttribute."
        )

    if attr._obj is None:
        raise TypeError(
            f"Cannot determine target object from attribute type {type(attr).__name__}. "
            "Attribute must have an _obj reference to its parent object."
        )

    # Serialize object reference to URL string at conversion time
    # (same pattern as source_from_attribute)
    obj_url = str(attr._obj.metadata.url)

    if attr.exists:
        attr_spec: CreateAttribute | UpdateAttribute = UpdateAttribute(
            reference=get_attribute_expression(attr),
        )
    else:
        attr_spec = CreateAttribute(name=attr.name)

    return Target(object=obj_url, attribute=attr_spec)
