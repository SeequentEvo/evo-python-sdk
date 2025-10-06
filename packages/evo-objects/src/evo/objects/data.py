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
from uuid import UUID

from evo.common import ResourceMetadata
from evo.workspaces import ServiceUser

from .exceptions import SchemaIDFormatError

__all__ = [
    "ObjectMetadata",
    "ObjectOrderByEnum",
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
    def url(self) -> str:
        """The url of the object."""
        return "{hub_url}/geoscience-object/orgs/{org_id}/workspaces/{workspace_id}/objects/{object_id}?version={version_id}".format(
            hub_url=self.environment.hub_url.rstrip("/"),
            org_id=self.environment.org_id,
            workspace_id=self.environment.workspace_id,
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
        return "{hub_url}/geoscience-object/orgs/{org_id}/workspaces/{workspace_id}/objects/{object_id}".format(
            hub_url=self.environment.hub_url.rstrip("/"),
            org_id=self.environment.org_id,
            workspace_id=self.workspace_id,
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
