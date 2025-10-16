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

from __future__ import annotations

import enum
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol
from urllib.parse import parse_qs, urlparse
from uuid import UUID

from evo.common import Environment, ResourceMetadata
from evo.workspaces import ServiceUser

from .exceptions import SchemaIDFormatError

__all__ = [
    "ObjectMetadata",
    "ObjectOrderByEnum",
    "ObjectReference",
    "ObjectSchema",
    "ObjectVersion",
    "SchemaVersion",
    "Stage",
]


class ObjectOrderByEnum(str, enum.Enum):
    author = "author"
    created_at = "created_at"
    created_by = "created_by"
    deleted_at = "deleted_at"
    modified_at = "modified_at"
    modified_by = "modified_by"
    object_name = "object_name"


class ObjectReference(str):
    """A structured URL reference to a geoscience object, optionally including a version ID.

    Geoscience Object URL references are the fully qualified HTTPS URLs used to access objects in the
    Geoscience Object API. The URL may follow the path or UUID format, and may optionally include a version ID.

    In most cases, UUID-based references are preferred, as they are immutable and unambiguous. However, path-based references
    can be useful in scenarios where the object ID is not known, such as when creating new objects or when working with
    objects in a more human-readable way.
    """

    _RE_PATH = re.compile(
        r"""
        ^/geoscience-object
        /orgs/(?P<org_id>[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})
        /workspaces/(?P<workspace_id>[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})
        /objects
        (?:
            /path/(?P<object_path>[^?]+) | /(?P<object_id>[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})
        )$
        """,
        re.IGNORECASE | re.VERBOSE,
    )

    hub_url: str
    """The base URL of the Evo Hub."""

    org_id: UUID
    """The ID of the Evo Organization the object belongs to."""

    workspace_id: UUID
    """The ID of the Evo Workspace the object belongs to."""

    object_id: UUID | None
    """The UUID of the object, if specified in the URL."""

    object_path: str | None
    """The path of the object, if specified in the URL."""

    version_id: str | None
    """The version ID of the object, if specified in the URL."""

    def __new__(cls, value: str) -> ObjectReference:
        inst = str.__new__(cls, value)

        parsed = urlparse(value)
        if parsed.scheme != "https":
            raise ValueError("Reference must be a valid HTTPS URL")

        inst.hub_url = f"{parsed.scheme}://{parsed.netloc}/"

        if match := cls._RE_PATH.fullmatch(parsed.path):
            inst.org_id = UUID(match.group("org_id"))
            inst.workspace_id = UUID(match.group("workspace_id"))

            if match.group("object_id"):
                inst.object_id = UUID(match.group("object_id"))
                inst.object_path = None
            else:
                inst.object_id = None
                inst.object_path = match.group("object_path").lstrip("/")
        else:
            raise ValueError("Reference path is not valid")

        query_params = parse_qs(parsed.query)
        inst.version_id = query_params.get("version", [None])[0]
        return inst

    @property
    def environment(self) -> Environment:
        return Environment(hub_url=self.hub_url, org_id=self.org_id, workspace_id=self.workspace_id)

    @staticmethod
    def new(
        environment: Environment,
        object_id: UUID | None = None,
        object_path: str | None = None,
        version_id: str | None = None,
    ) -> ObjectReference:
        """Create a new ObjectReference from its components.

        Either object_id or object_path must be provided, but not both.

        :param environment: The Evo environment the object belongs to.
        :param object_id: The UUID of the object, if known.
        :param object_path: The path of the object, if known.
        :param version_id: The version ID of the object, if known.

        :returns: A new ObjectReference instance.

        :raises ValueError: If neither or both of object_id and object_path are provided.
        """
        if object_id is None and object_path is None:
            raise ValueError("Either object_id or object_path must be provided")
        if object_id is not None and object_path is not None:
            raise ValueError("Only one of object_id or object_path can be provided")

        if object_id is not None:
            path = (
                f"geoscience-object/orgs/{environment.org_id}/workspaces/{environment.workspace_id}/objects/{object_id}"
            )
        else:
            path = f"geoscience-object/orgs/{environment.org_id}/workspaces/{environment.workspace_id}/objects/path/{object_path.lstrip('/')}"

        if version_id is not None:
            path += f"?version={version_id}"

        return ObjectReference(f"{environment.hub_url.rstrip('/')}/{path}")


@dataclass(frozen=True, kw_only=True)
class ObjectMetadata(ResourceMetadata):
    """Metadata about a geoscience object."""

    parent: str
    """The parent path of the object."""

    schema_id: ObjectSchema
    """The geoscience object schema."""

    version_id: str
    """An arbitrary identifier for the object version."""

    modified_at: datetime
    """The date and time when the object was last modified."""

    modified_by: ServiceUser | None
    """The user who last modified the object."""

    stage: Stage | None
    """The stage of the object, if available."""

    @property
    def path(self) -> str:
        """The full path of the object, formed by joining the parent and name, separated by a slash ('/')."""
        return f"{self.parent}/{self.name}"

    @property
    def url(self) -> ObjectReference:
        """The url of the object."""
        return ObjectReference.new(
            environment=self.environment,
            object_id=self.id,
            version_id=self.version_id,
        )


@dataclass(frozen=True, kw_only=True)
class OrgObjectMetadata(ResourceMetadata):
    """Metadata about a geoscience object in an organization listing."""

    workspace_id: UUID
    """The ID of the workspace that contains the object."""

    workspace_name: str | None
    """The name of the workspace that contains the object."""

    schema_id: ObjectSchema
    """The geoscience object schema."""

    modified_at: datetime
    """The date and time when the object was last modified."""

    modified_by: ServiceUser | None
    """The user who last modified the object."""

    stage: Stage | None
    """The stage of the object, if available."""

    @property
    def url(self) -> str:
        """The url of the object."""
        return ObjectReference.new(
            environment=self.environment,
            object_id=self.id,
        )


@dataclass(frozen=True, kw_only=True)
class ObjectVersion:
    """Represents a version of an object."""

    version_id: str
    """Used by the service to identify a unique resource version."""

    created_at: datetime
    """A UTC timestamp representing when the version was uploaded to the service."""

    created_by: ServiceUser | None
    """The user that uploaded the version."""

    stage: Stage | None
    """The stage associated with this version, if applicable."""


class _StageModel(Protocol):
    stage_id: UUID
    name: str


@dataclass(frozen=True, kw_only=True)
class Stage:
    """Metadata about a stage"""

    id: UUID
    """The stage UUID."""

    name: str
    """The stage name."""

    @classmethod
    def from_model(cls, model: _StageModel) -> Stage:
        """Create a new instance from an instance of a generated model."""
        return cls(id=model.stage_id, name=model.name)


@dataclass(frozen=True, order=True)
class SchemaVersion:
    major: int
    minor: int
    patch: int

    @classmethod
    def from_str(cls, string: str) -> SchemaVersion:
        return cls(*[int(i) for i in string.split(".")])

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"


@dataclass(frozen=True)
class ObjectSchema:
    root_classification: str
    sub_classification: str
    version: SchemaVersion

    @property
    def classification(self) -> str:
        return f"{self.root_classification}/{self.sub_classification}"

    @classmethod
    def from_id(cls, schema_id: str) -> ObjectSchema:
        schema_components = re.match(
            r"/(?P<root>[-\w]+)/(?P<sub>[-\w]+)/(?P<version>\d+\.\d+\.\d+)/(?P=sub)\.schema\.json",
            schema_id,
        )
        if schema_components is None:
            raise SchemaIDFormatError(f"Could not parse schema id: '{schema_id}'")

        return cls(
            root_classification=schema_components.group("root"),
            sub_classification=schema_components.group("sub"),
            version=SchemaVersion.from_str(schema_components.group("version")),
        )

    def __str__(self) -> str:
        return f"/{self.classification}/{self.version}/{self.sub_classification}.schema.json"
