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

from dataclasses import dataclass
from typing import Any

__all__ = [
    "CreateAttribute",
    "Source",
    "Target",
    "UpdateAttribute",
]


# Type alias for any object that can be serialized to a geoscience object reference URL
type GeoscienceObjectReference = str | Any  # str, ObjectReference, BaseObject, DownloadedObject, ObjectMetadata


def _serialize_object_reference(value: GeoscienceObjectReference) -> str:
    """
    Serialize an object reference to a string URL.

    Supports:
    - str: returned as-is
    - ObjectReference: str(value)
    - BaseObject (typed objects like PointSet): value.metadata.url
    - DownloadedObject: value.metadata.url
    - ObjectMetadata: value.url

    Args:
        value: The value to serialize

    Returns:
        String URL of the object reference

    Raises:
        TypeError: If the value type is not supported
    """
    if isinstance(value, str):
        return value

    # Check for ObjectReference (has __str__ that returns the URL)
    type_name = type(value).__name__
    if type_name == "ObjectReference":
        return str(value)

    # Check for typed objects (BaseObject subclasses like PointSet, Regular3DGrid)
    if hasattr(value, "metadata") and hasattr(value.metadata, "url"):
        return value.metadata.url

    # Check for ObjectMetadata
    if hasattr(value, "url") and isinstance(value.url, str):
        return value.url

    raise TypeError(f"Cannot serialize object reference of type {type(value)}")


@dataclass
class Source:
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

    def __init__(self, object: GeoscienceObjectReference, attribute: str):
        self.object = object
        self.attribute = attribute

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "object": _serialize_object_reference(self.object),
            "attribute": self.attribute,
        }


@dataclass
class CreateAttribute:
    """Specification for creating a new attribute on a target object."""

    name: str
    """The name of the attribute to create."""

    def __init__(self, name: str):
        self.name = name

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "operation": "create",
            "name": self.name,
        }


@dataclass
class UpdateAttribute:
    """Specification for updating an existing attribute on a target object."""

    reference: str
    """Reference to an existing attribute to update."""

    def __init__(self, reference: str):
        self.reference = reference

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "operation": "update",
            "reference": self.reference,
        }


@dataclass
class Target:
    """The target object and attribute to create or update with results.

    Used to specify where output data should be written for geostatistical operations.

    Example:
        >>> # Create a new attribute on a target object:
        >>> target = Target.new_attribute(block_model, "kriged_grade")
        >>>
        >>> # Or update an existing attribute:
        >>> target = Target(object=grid, attribute=UpdateAttribute("existing_ref"))
    """

    object: GeoscienceObjectReference
    """Object to write results onto."""

    attribute: CreateAttribute | UpdateAttribute
    """Attribute specification (create new or update existing)."""

    def __init__(self, object: GeoscienceObjectReference, attribute: CreateAttribute | UpdateAttribute):
        self.object = object
        self.attribute = attribute

    @classmethod
    def new_attribute(cls, object: GeoscienceObjectReference, attribute_name: str) -> Target:
        """
        Create a Target that will create a new attribute on the target object.

        Args:
            object: The target object to write results onto.
            attribute_name: The name of the new attribute to create.

        Returns:
            A Target instance configured to create a new attribute.

        Example:
            >>> target = Target.new_attribute(block_model, "kriged_grade")
        """
        return cls(object=object, attribute=CreateAttribute(name=attribute_name))

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        if hasattr(self.attribute, "to_dict"):
            attribute_value = self.attribute.to_dict()
        elif isinstance(self.attribute, dict):
            attribute_value = self.attribute
        else:
            attribute_value = self.attribute

        return {
            "object": _serialize_object_reference(self.object),
            "attribute": attribute_value,
        }

