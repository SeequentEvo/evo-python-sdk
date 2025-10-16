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

import dataclasses
from pathlib import PurePosixPath
from typing import overload

from evo.common import Environment, Page, ServiceUser

from ..data import ObjectMetadata, ObjectSchema, ObjectVersion, OrgObjectMetadata, Stage
from ..endpoints import models

__all__ = [
    "object_metadata",
    "org_object_metadata",
    "page_of_metadata",
    "schema",
    "stage",
    "stage_or_none",
    "user_or_none",
    "version",
    "versions",
]


def user_or_none(model: models.User | None) -> ServiceUser | None:
    """Parse a ServiceUser or None value from the generated model.

    :param model: The model returned by the generated code, or None.

    :return: The parsed ServiceUser, or None if the input model is None.
    """
    return None if model is None else ServiceUser.from_model(model)


def stage(model: models.StageResponse) -> Stage:
    """Parse a Stage from the generated model.

    :param model: The model returned by the generated code.

    :return: A Stage instance.
    """
    return Stage.from_model(model)


def stage_or_none(model: models.StageResponse | None) -> Stage | None:
    """Parse a Stage or None value from the generated model.

    :param model: The model returned by the generated code, or None.

    :return: The parsed Stage, or None if the input model is None.
    """
    return None if model is None else stage(model)


def version(model: models.GeoscienceObjectVersion) -> ObjectVersion:
    """Parse an ObjectVersion from the generated model.

    :param model: The model returned by the generated code.

    :return: An ObjectVersion instance.
    """
    return ObjectVersion(
        version_id=model.version_id,
        created_at=model.created_at,
        created_by=user_or_none(model.created_by),
        stage=stage_or_none(model.stage),
    )


def versions(model: models.GetObjectResponse) -> list[ObjectVersion]:
    """Parse a list of ObjectVersion from the generated model.

    :param model: The model returned by the generated code.

    :return: A list of ObjectVersion instances, sorted by created_at in descending order.
    """
    object_versions = [version(model) for model in model.versions]
    return sorted(object_versions, key=lambda v: v.created_at, reverse=True)


def schema(schema_id: str) -> ObjectSchema:
    """Parse an ObjectSchema from the schema ID.

    :param schema_id: The schema ID string.

    :return: An ObjectSchema instance.
    """
    return ObjectSchema.from_id(schema_id)


def object_metadata(
    model: models.ListedObject | models.GetObjectResponse | models.PostObjectResponse, environment: Environment
) -> ObjectMetadata:
    """Parse an ObjectMetadata from the generated model.

    :param model: The model returned by the generated code.
    :param environment: The Evo environment associated with the object.

    :return: An ObjectMetadata instance.
    """
    # There appears to be a schema defect where object_id may possibly be None, even though it shouldn't be.
    assert model.object_id is not None

    # Parse name, parent, and schema_id from the appropriate fields depending on the model type.
    if isinstance(model, models.ListedObject):
        name = model.name
        parent = model.path.rstrip("/")
        schema_id = model.schema_
    elif model.object_path is not None:
        path = PurePosixPath(model.object_path)
        name = path.name
        parent = str(path.parent)
        schema_id = model.object.schema_
    else:
        # There appears to be _another_ schema defect where object_path may be None in
        # GetObjectResponse or PostObjectResponse, even though this never happens in practice.
        raise ValueError("Model must be a ListedObject or have an object_path")

    return ObjectMetadata(
        environment=environment,
        id=model.object_id,
        name=name,
        created_at=model.created_at,
        created_by=user_or_none(model.created_by),
        modified_at=model.modified_at,
        modified_by=user_or_none(model.modified_by),
        parent=parent,
        schema_id=schema(schema_id),
        version_id=model.version_id,
        stage=stage_or_none(model.stage),
    )


def org_object_metadata(model: models.OrgListedObject, environment: Environment) -> OrgObjectMetadata:
    """Parse an OrgObjectMetadata from the generated model.

    :param model: The model returned by the generated code.
    :param environment: The Evo environment associated with the object.

    :return: An ObjectMetadata instance.
    """
    return OrgObjectMetadata(
        environment=dataclasses.replace(environment, workspace_id=model.workspace_id),
        workspace_id=model.workspace_id,
        workspace_name=model.workspace_name,
        id=model.object_id,
        name=model.name,
        created_at=model.created_at,
        created_by=user_or_none(model.created_by),
        modified_at=model.modified_at,
        modified_by=user_or_none(model.modified_by),
        schema_id=schema(model.schema_),
        stage=stage_or_none(model.stage),
    )


@overload
def page_of_metadata(model: models.ListObjectsResponse, environment: Environment) -> Page[ObjectMetadata]: ...


@overload
def page_of_metadata(model: models.ListOrgObjectsResponse, environment: Environment) -> Page[OrgObjectMetadata]: ...


def page_of_metadata(
    model: models.ListObjectsResponse | models.ListOrgObjectsResponse, environment: Environment
) -> Page[ObjectMetadata] | Page[OrgObjectMetadata]:
    """Parse a Page of ObjectMetadata or OrgObjectMetadata from the generated model.

    :param model: The model returned by the generated code.
    :param environment: The Evo environment associated with the objects.

    :return: A Page of ObjectMetadata or OrgObjectMetadata instances.

    :raises TypeError: If the model type is unsupported.
    """
    match model:
        case models.ListObjectsResponse():
            parse_metadata = object_metadata
        case models.ListOrgObjectsResponse():
            parse_metadata = org_object_metadata
        case _:
            raise TypeError(f"Unsupported model type: {type(model)}")

    return Page(
        offset=model.offset,
        limit=model.limit,
        total=model.total,
        items=[parse_metadata(item, environment) for item in model.objects],
    )
