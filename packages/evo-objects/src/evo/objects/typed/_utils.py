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

from logging import getLogger
from typing import Any

from evo import jmespath
from evo.common import Environment, EvoContext, ICache
from evo.objects import DownloadedObject, ObjectReference
from evo.objects.client import parse
from evo.objects.endpoints import ObjectsApi, models
from evo.objects.utils import ObjectDataClient

logger = getLogger(__name__)


def _extract_field_name(node: Any) -> str | None:
    if node["type"] == "field":
        return node["value"]
    return None


def assign_jmespath_value(document: dict[str, Any], path: jmespath.ParsedResult | str, value: Any) -> None:
    """Assign a value to a location in a document specified by a JMESPath expression.

    This is very limited at the moment and only supports expressions like: `a.b.c`
    """
    if isinstance(path, str):
        path = jmespath.compile(path)
    node = path.parsed
    field_name = _extract_field_name(node)
    if field_name:
        document[field_name] = value
        return

    if node["type"] != "subexpression":
        raise ValueError("Only subexpression paths are supported for assignment.")
    children = node["children"]
    for child in children[:-1]:
        field_name = _extract_field_name(child)
        if not field_name:
            raise ValueError("Unsupported JMESPath node type for assignment.")
        document = document.setdefault(field_name, {})

    last_field_name = _extract_field_name(children[-1])
    if not last_field_name:
        raise ValueError("Unsupported JMESPath node type for assignment.")
    document[last_field_name] = value


def delete_jmespath_value(document: dict[str, Any], path: jmespath.ParsedResult | str) -> None:
    """Delete a value from a location in a document specified by a JMESPath expression.

    This is very limited at the moment and only supports expressions like: `a.b.c`
    """
    if isinstance(path, str):
        path = jmespath.compile(path)
    node = path.parsed
    field_name = _extract_field_name(node)
    if field_name:
        document.pop(field_name, None)
        return

    if node["type"] != "subexpression":
        raise ValueError("Only subexpression paths are supported for assignment.")
    children = node["children"]
    for child in children[:-1]:
        field_name = _extract_field_name(child)
        if not field_name:
            raise ValueError("Unsupported JMESPath node type for assignment.")
        document = document.get(field_name, {})

    last_field_name = _extract_field_name(children[-1])
    if not last_field_name:
        raise ValueError("Unsupported JMESPath node type for assignment.")
    document.pop(last_field_name, None)


def get_data_client(context: EvoContext) -> ObjectDataClient:
    """Get an ObjectDataClient for the current context."""
    connector = context.get_connector()
    environment = context.get_environment()
    return ObjectDataClient(connector=connector, environment=environment, cache=context.cache)


def _response_to_downloaded_object(
    response: models.PostObjectResponse, environment: Environment, connector, cache: ICache | None
) -> DownloadedObject:
    metadata = parse.object_metadata(response, environment)
    urls_by_name = {getattr(link, "name", link.id): link.download_url for link in response.links.data}
    return DownloadedObject(
        object_=response.object,
        metadata=metadata,
        urls_by_name=urls_by_name,
        connector=connector,
        cache=cache,
    )


async def create_geoscience_object(context: EvoContext, object_dict: dict[str, Any], parent: str) -> DownloadedObject:
    connector = context.get_connector()
    environment = context.get_environment()

    objects_api = ObjectsApi(connector=connector)

    name = object_dict["name"]

    # TODO Smarter path handling, i.e. URL encode the name to handle arbitrary characters
    path = parent + name + ".json" if parent else name + ".json"
    object_for_upload = models.GeoscienceObject.model_validate(object_dict)
    response = await objects_api.post_objects(
        org_id=str(environment.org_id),
        workspace_id=str(environment.workspace_id),
        objects_path=path,
        geoscience_object=object_for_upload,
    )
    return _response_to_downloaded_object(response, environment, connector, context.cache)


async def replace_geoscience_object(
    context: EvoContext, reference: ObjectReference, object_dict: dict[str, Any]
) -> DownloadedObject:
    if reference.object_id is not None:
        object_dict["uuid"] = reference.object_id
    else:
        # Need to perform a GET request to get the existing object's UUID
        existing_obj = await download_geoscience_object(context, reference)
        object_dict["uuid"] = existing_obj.metadata.id

    connector = context.get_connector()
    environment = context.get_environment()
    objects_api = ObjectsApi(connector=connector)
    object_for_upload = models.UpdateGeoscienceObject.model_validate(object_dict)
    response = await objects_api.update_objects_by_id(
        object_id=str(object_dict["uuid"]),
        org_id=str(environment.org_id),
        workspace_id=str(environment.workspace_id),
        update_geoscience_object=object_for_upload,
    )
    return _response_to_downloaded_object(response, environment, connector, context.cache)


async def download_geoscience_object(context: EvoContext, reference: ObjectReference) -> DownloadedObject:
    return await DownloadedObject.from_reference(
        connector=context.get_connector(),
        reference=reference,
        cache=context.cache,
    )
