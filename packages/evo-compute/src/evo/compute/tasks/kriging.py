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
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from evo.common import IContext
from evo.common.interfaces import IFeedback
from evo.common.utils import NoFeedback, Retry, split_feedback

from ..client import JobClient

__all__ = [
    "CreateAttribute",
    "Ellipsoid",
    "EllipsoidRanges",
    "KrigingAttribute",
    "KrigingParameters",
    "KrigingResults",
    "KrigingSearch",
    "KrigingTarget",
    "OrdinaryKriging",
    "Rotation",
    "run_kriging",
    "run_kriging_multiple",
    "SimpleKriging",
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
    """The source object and attribute containing known values."""

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
    """Specification for creating a new attribute."""

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
    """Specification for updating an existing attribute."""

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
    """The target object and attribute to create or update with kriging results."""

    object: GeoscienceObjectReference
    """Object to evaluate onto."""

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
            object: The target object to evaluate onto.
            attribute_name: The name of the new attribute to create.

        Returns:
            A Target instance configured to create a new attribute.
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


@dataclass
class SimpleKriging:
    """Kriging method with a constant mean value."""

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
    """Kriging method with a variable mean value."""

    def __init__(self):
        pass

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "type": "ordinary",
        }


@dataclass
class EllipsoidRanges:
    """The ranges of the ellipsoid."""

    major: float
    """The major axis length of the ellipsoid."""

    semi_major: float
    """The semi major axis length of the ellipsoid."""

    minor: float
    """The minor axis length of the ellipsoid."""

    def __init__(self, major: float, semi_major: float, minor: float):
        self.major = major
        self.semi_major = semi_major
        self.minor = minor

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "major": self.major,
            "semi_major": self.semi_major,
            "minor": self.minor,
        }


@dataclass
class Rotation:
    """The rotation of the ellipsoid."""

    dip_azimuth: float = 0.0
    """First rotation, about the z-axis, in degrees."""

    dip: float = 0.0
    """Second rotation, about the x-axis, in degrees."""

    pitch: float = 0.0
    """Third rotation, about the z-axis, in degrees."""

    def __init__(self, dip_azimuth: float = 0.0, dip: float = 0.0, pitch: float = 0.0):
        self.dip_azimuth = dip_azimuth
        self.dip = dip
        self.pitch = pitch

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "dip_azimuth": self.dip_azimuth,
            "dip": self.dip,
            "pitch": self.pitch,
        }


@dataclass
class Ellipsoid:
    """The ellipsoid, to search for points within."""

    ellipsoid_ranges: EllipsoidRanges
    """The ranges of the ellipsoid."""

    rotation: Rotation
    """The rotation of the ellipsoid."""

    def __init__(self, ellipsoid_ranges: EllipsoidRanges, rotation: Rotation):
        self.ellipsoid_ranges = ellipsoid_ranges
        self.rotation = rotation

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "ellipsoid_ranges": self.ellipsoid_ranges.to_dict(),
            "rotation": self.rotation.to_dict(),
        }


@dataclass
class KrigingSearch:
    """
    Search parameters.

    In Kriging, the value of each evaluation point is determined by a set of nearby points with known values.
    The search parameters determines which nearby points to use.
    """

    ellipsoid: Ellipsoid
    """The ellipsoid, to search for points within."""

    max_samples: int
    """The maximum number of samples to use for each evaluation point."""

    def __init__(self, ellipsoid: Ellipsoid, max_samples: int):
        self.ellipsoid = ellipsoid
        self.max_samples = max_samples

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "ellipsoid": self.ellipsoid.to_dict(),
            "max_samples": self.max_samples,
        }


@dataclass
class KrigingParameters:
    """Parameters for the kriging task."""

    source: Source
    """The source object and attribute containing known values."""

    target: Target
    """The target object and attribute to create or update with kriging results."""

    kriging_method: SimpleKriging | OrdinaryKriging
    """The kriging method to use. Either simple or ordinary kriging."""

    variogram: GeoscienceObjectReference
    """Model of the covariance within the domain."""

    neighborhood: KrigingSearch
    """Search parameters."""

    def __init__(
        self,
        source: Source | Any,  # Also accepts Attribute from evo.objects.typed
        target: Target,
        kriging_method: SimpleKriging | OrdinaryKriging,
        variogram: GeoscienceObjectReference,
        neighborhood: KrigingSearch,
    ):
        # Handle Attribute type from evo.objects.typed.dataset
        if hasattr(source, "_obj") and hasattr(source, "expression"):
            # source is an Attribute, construct a Source object
            source = Source(object=source._obj.metadata.url, attribute=source.expression)
        self.source = source
        self.target = target
        self.kriging_method = kriging_method
        self.variogram = variogram
        self.neighborhood = neighborhood

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "source": self.source.to_dict(),
            "target": self.target.to_dict(),
            "kriging_method": self.kriging_method.to_dict(),
            "variogram": _serialize_object_reference(self.variogram),
            "neighborhood": self.neighborhood.to_dict(),
        }


@dataclass
class KrigingAttribute:
    """Attribute containing the kriging result."""

    reference: str
    """Reference to the attribute in the geoscience object."""

    name: str
    """The name of the output attribute."""

    def __init__(self, reference: str, name: str):
        self.reference = reference
        self.name = name

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "reference": self.reference,
            "name": self.name,
        }


@dataclass
class KrigingTarget:
    """The target that was created or updated."""

    reference: str
    """Reference to a geoscience object."""

    name: str
    """The name of the geoscience object."""

    description: Any
    """The description of the geoscience object."""

    schema_id: str
    """The ID of the Geoscience Object schema."""

    attribute: KrigingAttribute
    """Attribute containing the kriging result."""

    def __init__(
        self,
        reference: str,
        name: str,
        description: Any,
        schema_id: str,
        attribute: KrigingAttribute,
    ):
        self.reference = reference
        self.name = name
        self.description = description
        self.schema_id = schema_id
        self.attribute = attribute

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "reference": _serialize_object_reference(self.reference),
            "name": self.name,
            "description": self.description,
            "schema_id": self.schema_id,
            "attribute": self.attribute.to_dict(),
        }


@dataclass
class KrigingResults:
    """Result of the kriging task."""

    message: str
    """A message that says what happened in the task."""

    target: KrigingTarget
    """The target that was created or updated."""

    def __init__(self, message: str, target: KrigingTarget):
        self.message = message
        self.target = target

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "message": self.message,
            "target": self.target.to_dict(),
        }


async def run_kriging(
    context: IContext,
    parameters: KrigingParameters,
    *,
    polling_interval_seconds: float = 0.5,
    retry: Retry | None = None,
    fb: IFeedback = NoFeedback,
) -> KrigingResults:
    """
    Run a kriging compute task.

    For more information, please read the kriging guide at:
    https://developer.seequent.com/docs/guides/geostatistics-tasks/tasks/kriging

    Args:
        context: The context providing connector and org_id
        parameters: The kriging task parameters
        polling_interval_seconds: Interval between status checks when waiting
        retry: Retry strategy for waiting (if None, uses default)
        fb: Feedback interface for progress updates

    Returns:
        The kriging task results

    Example:
        ```python
        from evo.compute.tasks import (
            run_kriging,
            KrigingParameters,
            Source,
            Target,
            OrdinaryKriging,
            KrigingSearch,
            Ellipsoid,
            EllipsoidRanges,
            Rotation,
        )

        params = KrigingParameters(
            source=Source(object=pointset, attribute="grade"),
            target=Target.new_attribute(object=grid, attribute_name="kriged_grade"),
            kriging_method=OrdinaryKriging(),
            variogram=variogram,
            neighborhood=KrigingSearch(
                ellipsoid=Ellipsoid(
                    ellipsoid_ranges=EllipsoidRanges(major=200, semi_major=150, minor=100),
                    rotation=Rotation(0, 0, 0),
                ),
                max_samples=20,
            ),
        )
        result = await run_kriging(manager, params)
        ```
    """
    connector = context.get_connector()
    org_id = context.get_org_id()

    # Add API-Preview header for opt-in features
    # Must set on connector so it's included in ALL requests (submit, get_status, get_results)
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
        result_type=KrigingResults,
    )

    # Wait for results and return them directly
    results = await job.wait_for_results(
        polling_interval_seconds=polling_interval_seconds,
        retry=retry,
        fb=fb,
    )

    return results


async def run_kriging_multiple(
    context: IContext,
    parameters: list[KrigingParameters],
    *,
    polling_interval_seconds: float = 0.5,
    retry: Retry | None = None,
    fb: IFeedback = NoFeedback,
) -> list[KrigingResults]:
    """
    Run multiple kriging compute tasks concurrently.

    This function submits multiple kriging tasks and waits for all to complete.
    Progress is aggregated across all tasks.

    For more information, please read the kriging guide at:
    https://developer.seequent.com/docs/guides/geostatistics-tasks/tasks/kriging

    Args:
        context: The context providing connector and org_id
        parameters: List of kriging task parameters
        polling_interval_seconds: Interval between status checks when waiting
        retry: Retry strategy for waiting (if None, uses default)
        fb: Feedback interface for progress updates

    Returns:
        List of kriging task results in the same order as the input parameters

    Example:
        ```python
        from evo.compute.tasks import run_kriging_multiple, KrigingParameters

        # Create parameter sets for different scenarios
        param_sets = [
            KrigingParameters(..., neighborhood=KrigingSearch(..., max_samples=10)),
            KrigingParameters(..., neighborhood=KrigingSearch(..., max_samples=20)),
            KrigingParameters(..., neighborhood=KrigingSearch(..., max_samples=30)),
        ]

        results = await run_kriging_multiple(manager, param_sets)
        ```
    """
    if len(parameters) == 0:
        return []

    total = len(parameters)

    # Split feedback across tasks to aggregate total progress linearly
    per_task_fb = split_feedback(fb, [1.0] * total)

    # Wrapper that returns (index, result) for robust mapping
    async def _run_one(i: int, param: KrigingParameters) -> tuple[int, KrigingResults]:
        res = await run_kriging(
            context,
            param,
            polling_interval_seconds=polling_interval_seconds,
            retry=retry,
            fb=per_task_fb[i],
        )
        return i, res

    tasks = [asyncio.create_task(_run_one(i, param)) for i, param in enumerate(parameters)]

    results: list[KrigingResults | None] = [None] * total

    done_count = 0
    for fut in asyncio.as_completed(tasks):
        try:
            i, res = await fut
            results[i] = res
            done_count += 1
            percent = done_count / total
            msg = f"{done_count}/{total} scenarios completed ({int(percent * 100)}%)"
            # Update message via a child feedback (progress already aggregated)
            per_task_fb[i].progress(1.0, msg)
        except Exception:
            # Find index if the wrapper raised before returning (i, res)
            done_count += 1
            percent = done_count / total
            msg = f"{done_count}/{total} scenarios completed ({int(percent * 100)}%) with errors"
            # We cannot pinpoint i reliably here; update parent
            fb.progress(percent, msg)
            # Cancel remaining to fail fast
            for t in tasks:
                t.cancel()
            # Re-raise the first error to caller
            raise

    # Type assertion: all results should be populated
    return [r for r in results if r is not None]

