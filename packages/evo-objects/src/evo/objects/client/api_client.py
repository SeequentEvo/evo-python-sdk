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

from collections.abc import AsyncIterator, Sequence
from uuid import UUID

from evo import logging
from evo.common import APIConnector, BaseAPIClient, HealthCheckType, ICache, Page, ServiceHealth
from evo.common.data import EmptyResponse, Environment, OrderByOperatorEnum
from evo.common.utils import get_service_health, parse_order_by

from ..data import ObjectMetadata, ObjectOrderByEnum, ObjectReference, ObjectVersion, OrgObjectMetadata, Stage
from ..endpoints import MetadataApi, ObjectsApi, StagesApi
from ..endpoints.models import GeoscienceObject, MetadataUpdateBody, UpdateGeoscienceObject
from ..exceptions import ObjectUUIDError
from ..io import ObjectDataDownload, ObjectDataUpload
from . import parse
from .object_client import DownloadedObject

try:
    from ..utils import ObjectDataClient
except ImportError:
    _DATA_CLIENT_AVAILABLE = False
else:
    _DATA_CLIENT_AVAILABLE = True

logger = logging.getLogger("object.client")

__all__ = ["ObjectAPIClient"]


class ObjectAPIClient(BaseAPIClient):
    def __init__(self, environment: Environment, connector: APIConnector, cache: ICache | None = None) -> None:
        """
        :param environment: The target Evo environment, providing org and workspace IDs.
        :param connector: The API connector to use for making API calls.
        :param cache: An optional cache to use for data downloads.
        """
        super().__init__(environment, connector)
        self._stages_api = StagesApi(connector=connector)
        self._objects_api = ObjectsApi(connector=connector)
        self._metadata_api = MetadataApi(connector=connector)
        self._cache = cache

    async def get_service_health(self, check_type: HealthCheckType = HealthCheckType.FULL) -> ServiceHealth:
        """Get the health of the geoscience object service.

        :param check_type: The type of health check to perform.

        :return: A ServiceHealth object.

        :raises EvoAPIException: If the API returns an unexpected status code.
        :raises ClientValueError: If the response is not a valid service health check response.
        """
        return await get_service_health(self._connector, "geoscience-object", check_type=check_type)

    async def list_objects(
        self,
        offset: int = 0,
        limit: int = 5000,
        order_by: dict[ObjectOrderByEnum | str, OrderByOperatorEnum] | None = None,
        schema_id: list[str] | None = None,
        deleted: bool | None = None,
        request_timeout: int | float | tuple[int | float, int | float] | None = None,
    ) -> Page[ObjectMetadata]:
        """List up to `limit` geoscience objects, starting at `offset`.

        The geoscience objects will be the latest version of the object.
        If there are no objects to list, the page will be empty.

        :param offset: The number of objects to skip before listing.
        :param limit: Max number of objects to list.
        :param order_by: A dictionary of fields to order the results by, with the field name as the key and the direction of ordering as the value.
        :param schema_id: A list of schema IDs to filter the objects by. If None, objects of all schema types are returned.
        :param deleted: When true, only objects that have been deleted will be returned. If false or None, only non-deleted objects will be returned.
        :param request_timeout: Timeout setting for this request. If one number is provided, it will be the
            total request timeout. It can also be a pair (tuple) of (connection, read) timeouts.

        :return: A page of all objects from the query.
        """
        assert limit > 0
        assert offset >= 0
        parsed_order_by = parse_order_by(order_by)
        response = await self._objects_api.list_objects(
            org_id=str(self._environment.org_id),
            workspace_id=str(self._environment.workspace_id),
            limit=limit,
            offset=offset,
            order_by=parsed_order_by,
            schema_id=schema_id,
            request_timeout=request_timeout,
            deleted=deleted,
        )
        return parse.page_of_metadata(response, self._environment)

    async def list_all_objects(
        self,
        limit_per_request: int = 5000,
        order_by: dict[ObjectOrderByEnum | str, OrderByOperatorEnum] | None = None,
        schema_id: list[str] | None = None,
        deleted: bool | None = None,
        request_timeout: int | float | tuple[int | float, int | float] | None = None,
    ) -> list[ObjectMetadata]:
        """List all geoscience objects in the workspace.

        This method makes multiple calls to the `list_objects` endpoint until all objects have been listed.

        :param limit_per_request: The maximum number of objects to list in one request.
        :param order_by: A dictionary of fields to order the results by, with the field name as the key and the direction of ordering as the value.
        :param schema_id: A list of schema IDs to filter the objects by. If None, objects of all schema types are returned.
        :param deleted: When true, only objects that have been deleted will be returned. If false or None, only non-deleted objects will be returned.
        :param request_timeout: Timeout setting for this request. If one number is provided, it will be the
            total request timeout. It can also be a pair (tuple) of (connection, read) timeouts.

        :return: A list of all objects in the workspace.
        """
        items = []
        offset = 0
        while True:
            page = await self.list_objects(
                offset=offset,
                limit=limit_per_request,
                order_by=order_by,
                schema_id=schema_id,
                request_timeout=request_timeout,
                deleted=deleted,
            )
            items += page.items()
            if page.is_last:
                break
            offset = page.next_offset
        return items

    async def list_objects_for_instance(
        self,
        offset: int = 0,
        limit: int = 5000,
        order_by: dict[ObjectOrderByEnum | str, OrderByOperatorEnum] | None = None,
        schema_id: list[str] | None = None,
        deleted: bool | None = None,
        request_timeout: int | float | tuple[int | float, int | float] | None = None,
    ) -> Page[OrgObjectMetadata]:
        """List up to `limit` geoscience objects for all accessible workspaces in the instance, starting at `offset`.

        The geoscience objects will be the latest version of the object.
        If there are no objects to list, the page will be empty.

        :param offset: The number of objects to skip before listing.
        :param limit: Max number of objects to list.
        :param order_by: A dictionary of fields to order the results by, with the field name as the key and the direction of ordering as the value.
        :param schema_id: A list of schema IDs to filter the objects by. If None, objects of all schema types are returned.
        :param deleted: When true, only objects that have been deleted will be returned. If false or None, only non-deleted objects will be returned.
        :param request_timeout: Timeout setting for this request. If one number is provided, it will be the
            total request timeout. It can also be a pair (tuple) of (connection, read) timeouts.

        :return: A page of all objects from the query.
        """
        assert limit > 0
        assert offset >= 0
        parsed_order_by = parse_order_by(order_by)
        response = await self._objects_api.list_objects_by_org(
            org_id=str(self._environment.org_id),
            limit=limit,
            offset=offset,
            order_by=parsed_order_by,
            schema_id=schema_id,
            request_timeout=request_timeout,
            permitted_workspaces_only=True,
            deleted=deleted,
        )
        return parse.page_of_metadata(response, self._environment)

    async def list_versions_by_path(
        self, path: str, request_timeout: int | float | tuple[int | float, int | float] | None = None
    ) -> list[ObjectVersion]:
        """List all version for the given object.

        :param path: The path to the geoscience object.
        :param request_timeout: Timeout setting for this request. If one number is provided, it will be the
            total request timeout. It can also be a pair (tuple) of (connection, read) timeouts.

        :return: A sorted list of object versions. The latest version is the first element of the list.
        """
        response = await self._objects_api.get_object(
            org_id=str(self._environment.org_id),
            workspace_id=str(self._environment.workspace_id),
            objects_path=path,
            include_versions=True,
            request_timeout=request_timeout,
        )
        return parse.versions(response)

    async def list_versions_by_id(
        self, object_id: UUID, request_timeout: int | float | tuple[int | float, int | float] | None = None
    ) -> list[ObjectVersion]:
        """List all version for the given object.

        :param object_id: The UUID of the geoscience object.
        :param request_timeout: Timeout setting for this request. If one number is provided, it will be the
            total request timeout. It can also be a pair (tuple) of (connection, read) timeouts.

        :return: A sorted list of object versions. The latest version is the first element of the list.
        """
        response = await self._objects_api.get_object_by_id(
            org_id=str(self._environment.org_id),
            workspace_id=str(self._environment.workspace_id),
            object_id=str(object_id),
            include_versions=True,
            request_timeout=request_timeout,
        )
        return parse.versions(response)

    async def prepare_data_upload(self, data_identifiers: Sequence[str | UUID]) -> AsyncIterator[ObjectDataUpload]:
        """Prepare to upload multiple data files to the geoscience object service.

        Referenced data that already exists in the workspace will be skipped.

        :param data_identifiers: A list of sha256 digests or UUIDs for the data to be uploaded.

        :return: An async iterator of data upload contexts that can be used to upload the data. Data identifiers
            that already exist in the workspace will be skipped.
        """
        async for ctx in ObjectDataUpload._create_multiple(
            connector=self._connector,
            environment=self._environment,
            names=data_identifiers,
        ):
            yield ctx

    async def prepare_data_download(
        self,
        object_id: UUID,
        version_id: str,
        data_identifiers: Sequence[str | UUID],
        request_timeout: int | float | tuple[int | float, int | float] | None = None,
    ) -> AsyncIterator[ObjectDataDownload]:
        """Prepare to download multiple data files from the geoscience object service.

        Any data IDs that are not associated with the requested object will raise a DataNotFoundError.

        :param object_id: The ID of the object to download data from.
        :param version_id: The version ID of the object to download data from.
        :param data_identifiers: A list of sha256 digests or UUIDs for the data to be downloaded.
        :param request_timeout: Timeout setting for this request. If one number is provided, it will be the
            total request timeout. It can also be a pair (tuple) of (connection, read) timeouts.

        :return: An async iterator of data download contexts that can be used to download the data.

        :raises DataNotFoundError: If any requested data ID is not associated with the referenced object.
        """
        downloaded_object = await self.download_object_by_id(
            object_id, version=version_id, request_timeout=request_timeout
        )
        for ctx in downloaded_object.prepare_data_download(data_identifiers):
            yield ctx

    if _DATA_CLIENT_AVAILABLE:
        # Optional data client functionality, enabled if the data client dependencies are installed.

        def get_data_client(self, cache: ICache) -> ObjectDataClient:
            """Get a data client for the geoscience object service.

            The data client provides a high-level interface for uploading and downloading data that is referenced in
            geoscience objects, and caching the data locally. It depends on the optional dependency `pyarrow`, which is
            not installed by default. This dependency can be installed with `pip install evo-objects[utils]`.

            :param cache: The cache to use for data downloads.

            :return: An ObjectDataClient instance.

            :raises RuntimeError: If the `pyarrow` package is not installed.
            """
            return ObjectDataClient(environment=self._environment, connector=self._connector, cache=cache)

    async def create_geoscience_object(
        self, path: str, object_dict: dict, request_timeout: int | float | tuple[int | float, int | float] | None = None
    ) -> ObjectMetadata:
        """Upload a new geoscience object to the geoscience object service.

        New geoscience objects must not have a UUID, so that one can be assigned by the Geoscience Object Service.
        The `object_instance` that is passed in will be updated with the service-assigned UUID after it has been
        uploaded, and the metadata of the created object will be returned.

        :param path: The path to upload the object to.
        :param object_dict: The geoscience object to be uploaded.
        :param request_timeout: Timeout setting for this request. If one number is provided, it will be the
            total request timeout. It can also be a pair (tuple) of (connection, read) timeouts.

        :return: The metadata of the uploaded object.

        :raises ObjectUUIDError: If the provided object has a UUID.
        """
        if object_dict.get("uuid") is not None:
            raise ObjectUUIDError("Object has a UUID but new objects should have None")
        object_for_upload = GeoscienceObject.model_validate(object_dict)

        result = await self._objects_api.post_objects(
            org_id=str(self._environment.org_id),
            workspace_id=str(self._environment.workspace_id),
            objects_path=path,
            geoscience_object=object_for_upload,
            request_timeout=request_timeout,
        )
        object_dict["uuid"] = result.object_id
        return parse.object_metadata(result, self._environment)

    async def move_geoscience_object(
        self, path: str, object_dict: dict, request_timeout: int | float | tuple[int | float, int | float] | None = None
    ) -> ObjectMetadata:
        """Move an existing geoscience object to a new path in the geoscience object service.

        :param path: The new path to move the object to.
        :param object_dict: The geoscience object to be moved.
        :param request_timeout: Timeout setting for this request. If one number is provided, it will be the
            total request timeout. It can also be a pair (tuple) of (connection, read) timeouts.

        :return: The metadata of the moved object.

        :raises ObjectUUIDError: If the provided object does not have a UUID.
        """
        if object_dict.get("uuid") is None:
            raise ObjectUUIDError("Object does not have a UUID")
        object_for_upload = GeoscienceObject.model_validate(object_dict)

        result = await self._objects_api.post_objects(
            org_id=str(self._environment.org_id),
            workspace_id=str(self._environment.workspace_id),
            objects_path=path,
            geoscience_object=object_for_upload,
            request_timeout=request_timeout,
        )
        return parse.object_metadata(result, self._environment)

    async def update_geoscience_object(
        self, object_dict: dict, request_timeout: int | float | tuple[int | float, int | float] | None = None
    ) -> ObjectMetadata:
        """Update an existing geoscience object in the geoscience object service.

        :param object_dict: The geoscience object to be updated.
        :param request_timeout: Timeout setting for this request. If one number is provided, it will be the
            total request timeout. It can also be a pair (tuple) of (connection, read) timeouts.

        :return: The metadata of the updated object.

        :raises ObjectUUIDError: If the provided object does not have a UUID.
        """
        if object_dict.get("uuid") is None:
            raise ObjectUUIDError("Object does not have a UUID")
        object_for_upload = UpdateGeoscienceObject.model_validate(object_dict)

        result = await self._objects_api.update_objects_by_id(
            object_id=str(object_for_upload.uuid),
            org_id=str(self._environment.org_id),
            workspace_id=str(self._environment.workspace_id),
            update_geoscience_object=object_for_upload,
            request_timeout=request_timeout,
        )
        return parse.object_metadata(result, self._environment)

    async def download_object_by_path(
        self,
        path: str,
        version: str | None = None,
        request_timeout: int | float | tuple[int | float, int | float] | None = None,
    ) -> DownloadedObject:
        """Download a geoscience object definition (by path).

        :param path: The path to the geoscience object.
        :param version: The version of the geoscience object to download. This will download the latest version by
            default.
        :param request_timeout: Timeout setting for this request. If one number is provided, it will be the
            total request timeout. It can also be a pair (tuple) of (connection, read) timeouts.

        :return: A tuple containing the object metadata and a data model of the requested geoscience object.
        """
        reference = ObjectReference.new(environment=self._environment, object_path=path, version_id=version)
        return await DownloadedObject.from_reference(
            connector=self._connector,
            reference=reference,
            cache=self._cache,
            request_timeout=request_timeout,
        )

    async def download_object_by_id(
        self,
        object_id: UUID,
        version: str | None = None,
        request_timeout: int | float | tuple[int | float, int | float] | None = None,
    ) -> DownloadedObject:
        """Download a geoscience object definition (by UUID).

        :param object_id: The uuid of the geoscience object.
        :param version: The version of the geoscience object to download. This will download the latest version by
            default.
        :param request_timeout: Timeout setting for this request. If one number is provided, it will be the
            total request timeout. It can also be a pair (tuple) of (connection, read) timeouts.

        :return: A tuple containing the object metadata and a data model of the requested geoscience object.
        """
        reference = ObjectReference.new(environment=self._environment, object_id=object_id, version_id=version)
        return await DownloadedObject.from_reference(
            connector=self._connector,
            reference=reference,
            cache=self._cache,
            request_timeout=request_timeout,
        )

    async def get_latest_object_versions(
        self,
        object_ids: list[UUID],
        batch_size: int = 500,
        request_timeout: int | float | tuple[int | float, int | float] | None = None,
    ) -> dict[UUID, str]:
        """Get the latest version of each object by uuid.

        :param object_ids: A list of object uuids.
        :param batch_size: The maximum number of objects to check in one API call (max 500).
        :param request_timeout: Timeout setting for this request. If one number is provided, it will be the
            total request timeout. It can also be a pair (tuple) of (connection, read) timeouts.

        :return: A mapping of uuids to the latest version id.
        """
        offset = 0
        n_objects = len(object_ids)
        latest_ids: dict[UUID, str] = {}
        while batch_object_ids := object_ids[offset : min(offset + batch_size, n_objects)]:
            offset += batch_size
            response = await self._objects_api.list_object_version_ids(
                org_id=str(self._environment.org_id),
                workspace_id=str(self._environment.workspace_id),
                request_body=[str(object_id) for object_id in batch_object_ids],
                request_timeout=request_timeout,
            )
            latest_ids.update({UUID(latest.object_id): latest.version_id for latest in response})
        return latest_ids

    async def delete_object_by_path(
        self, path: str, request_timeout: int | float | tuple[int | float, int | float] | None = None
    ) -> None:
        """Soft-delete a geoscience object (by path).

        :param path: The path to the geoscience object.
        :param request_timeout: Timeout setting for this request. If one number is provided, it will be the
            total request timeout. It can also be a pair (tuple) of (connection, read) timeouts.

        :return: None
        """
        await self._objects_api.delete_object_by_path(
            org_id=str(self._environment.org_id),
            workspace_id=str(self._environment.workspace_id),
            objects_path=path,
            additional_headers={"Accept-Encoding": "gzip"},
            request_timeout=request_timeout,
        )

    async def delete_object_by_id(
        self, object_id: UUID, request_timeout: int | float | tuple[int | float, int | float] | None = None
    ) -> None:
        """Soft-delete a geoscience object (by UUID).

        :param object_id: The uuid of the geoscience object
        :param request_timeout: Timeout setting for this request. If one number is provided, it will be the
            total request timeout. It can also be a pair (tuple) of (connection, read) timeouts.

        :return: None
        """
        await self._objects_api.delete_objects_by_id(
            org_id=str(self._environment.org_id),
            workspace_id=str(self._environment.workspace_id),
            object_id=str(object_id),
            additional_headers={"Accept-Encoding": "gzip"},
            request_timeout=request_timeout,
        )

    async def restore_geoscience_object(
        self, object_id: UUID, request_timeout: int | float | tuple[int | float, int | float] | None = None
    ) -> None | ObjectMetadata:
        """Restore a soft-deleted geoscience object in the geoscience object service.

        :param object_id: The uuid of the geoscience object
        :param request_timeout: Timeout setting for this request. If one number is provided, it will be the
            total request timeout. It can also be a pair (tuple) of (connection, read) timeouts.

        :return: The metadata of the restored object, if a rename occurred. Otherwise, None.
        """
        result = await self._objects_api.update_objects_by_id(
            object_id=str(object_id),
            org_id=str(self._environment.org_id),
            workspace_id=str(self._environment.workspace_id),
            deleted=False,
            request_timeout=request_timeout,
        )
        # If the restore happened without a rename, the response will be empty
        # If the restore happened with a rename, the response will be the metadata of the restored object
        if isinstance(result, EmptyResponse):
            return None
        return parse.object_metadata(result, self._environment)

    async def list_stages(self) -> list[Stage]:
        """List all available stages in the organisation.

        :return: A list of all available stages."""
        response = await self._stages_api.list_stages(org_id=str(self._environment.org_id))
        return [parse.stage(model) for model in response.stages]

    async def set_stage(self, object_id: UUID, version_id: int, stage_id: UUID) -> None:
        """Set the stage of a specific version of a geoscience object.

        :param object_id: The UUID of the geoscience object.
        :param version_id: The version ID of the geoscience object.
        :param stage_id: The UUID of the stage to set.

        :return: None
        """

        await self._metadata_api.update_metadata(
            object_id=str(object_id),
            org_id=str(self._environment.org_id),
            workspace_id=str(self._environment.workspace_id),
            metadata_update_body=MetadataUpdateBody(stage_id=stage_id),
            version_id=version_id,
        )
