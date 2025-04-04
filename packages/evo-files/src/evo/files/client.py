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

from pathlib import PurePosixPath
from uuid import UUID

from evo import logging
from evo.common import APIConnector, BaseAPIClient, Environment, HealthCheckType, Page, ServiceHealth, ServiceUser
from evo.common.utils import get_service_health

from .data import FileMetadata, FileVersion
from .endpoints import FileV2Api
from .endpoints.models import DownloadFileResponse, FileVersionResponse, ListFile, UserInfo
from .io import FileAPIDownload, FileAPIUpload

logger = logging.getLogger("file.client")

__all__ = ["FileAPIClient"]


def _user_from_model(model: UserInfo | None) -> ServiceUser | None:
    """Create a ServiceUser instance from a generated UserInfo model.

    :param model: The model to create the ServiceUser instance from, or None.

    :return: A ServiceUser instance, or None if the model is None.
    """
    return None if model is None else ServiceUser(id=model.id, name=model.name, email=model.email)


def _versions_from_listed_versions(models: list[FileVersionResponse]) -> list[FileVersion]:
    """Create a list of FileVersion instances from a list of generated FileVersionResponse models.

    :param models: The models to create the FileVersion instances from.

    :return: A sorted list of FileVersion instances.
    """
    versions = (
        FileVersion(
            version_id=model.version_id,
            created_at=model.created_at,
            created_by=_user_from_model(model.created_by),
        )
        for model in models
    )
    return sorted(versions, key=lambda v: v.created_at, reverse=True)


class FileAPIClient(BaseAPIClient):
    def __init__(self, environment: Environment, connector: APIConnector) -> None:
        """
        :param environment: The environment object
        :param connector: The connector object.
        """
        super().__init__(environment, connector)
        self._api = FileV2Api(connector=connector)

    async def get_service_health(self, check_type: HealthCheckType = HealthCheckType.FULL) -> ServiceHealth:
        """Get the health of the file service.

        :param check_type: The type of health check to perform.

        :return: A ServiceHealth object.

        :raises EvoAPIException: If the API returns an unexpected status code.
        :raises ClientValueError: If the response is not a valid service health check response.
        """
        return await get_service_health(self._connector, "file", check_type=check_type)

    def _metadata_from_listed_file(self, model: ListFile) -> FileMetadata:
        """Create a FileMetadata instance from a generated ListFile model.

        :param model: The model to create the FileMetadata instance from.

        :return: A FileMetadata instance.
        """
        return FileMetadata(
            environment=self._environment,
            id=model.file_id,
            name=model.name,
            created_at=model.created_at,
            created_by=_user_from_model(model.created_by),
            modified_at=model.modified_at,
            modified_by=_user_from_model(model.modified_by),
            parent=model.path,
            version_id=model.version_id,
            size=model.size,
        )

    def _metadata_from_endpoint_model(self, model: DownloadFileResponse) -> FileMetadata:
        """Create a FileMetadata instance from a generated DownloadFileResponse model.

        :param model: The model to create the FileMetadata instance from.

        :return: A FileMetadata instance.
        """
        file_path = PurePosixPath(model.path)
        return FileMetadata(
            environment=self._environment,
            id=model.file_id,
            name=model.name,
            created_at=model.created_at,
            created_by=_user_from_model(model.created_by),
            modified_at=model.modified_at,
            modified_by=_user_from_model(model.modified_by),
            parent=str(file_path.parent),
            version_id=model.version_id,
            size=model.size,
        )

    async def list_files(
        self,
        offset: int = 0,
        limit: int = 5000,
        name: str | None = None,
    ) -> Page[FileMetadata]:
        """List up to `limit` files in the workspace, starting at `offset`.

        The files will be the latest version of the file.
        If there are no files starting at `offset`, the page will be empty.

        :param offset: The number of files to skip before listing.
        :param limit: Max number of files to list.
        :param name: Filter files by name.

        :return: A page of all files from the query.
        """
        assert limit > 0, "Limit must be a positive integer"
        assert offset >= 0, "Offset must be a non-negative integer"
        response = await self._api.list_files(
            organisation_id=str(self._environment.org_id),
            workspace_id=str(self._environment.workspace_id),
            limit=limit,
            offset=offset,
            file_name=name,
        )
        return Page(
            offset=offset,
            limit=limit,
            total=response.total,
            items=[self._metadata_from_listed_file(file) for file in response.files],
        )

    async def list_all_files(self, limit_per_request: int = 5000, name: str | None = None) -> list[FileMetadata]:
        """List all files in the workspace.

        This method makes multiple calls to the `list_files` endpoint until all files have been listed.

        :param limit_per_request: The maximum number of files to list in one request.
        :param name: Filter files by name.

        :return: A list of all files in the workspace.
        """
        items = []
        offset = 0
        while True:
            page = await self.list_files(offset=offset, limit=limit_per_request, name=name)
            items += page.items()
            if page.is_last:
                break
            offset = page.next_offset
        return items

    async def get_file_by_path(self, path: str, version_id: str | None = None) -> FileMetadata:
        """Get a file by its path.

        :param path: The path to the file.
        :param version_id: ID of the desired file version. By default, the response will return the latest version.
        :return: A FileMetadata representation of the file on the service.
        """
        file_response = await self._api.get_file_by_path(
            organisation_id=str(self._environment.org_id),
            workspace_id=str(self._environment.workspace_id),
            file_path=path,
            version_id=version_id,
        )
        return self._metadata_from_endpoint_model(file_response)

    async def get_file_by_id(self, file_id: UUID, version_id: str | None = None) -> FileMetadata:
        """Get a file by its ID.

        :param file_id: UUID of a file
        :param version_id: ID of the desired file version. By default, the response will return the latest version.
        :return: A FileMetadata representation of the file on the service
        """
        file_response = await self._api.get_file_by_id(
            organisation_id=str(self._environment.org_id),
            workspace_id=str(self._environment.workspace_id),
            file_id=str(file_id),
            version_id=version_id,
        )
        return self._metadata_from_endpoint_model(file_response)

    async def list_versions_by_path(self, path: str) -> list[FileVersion]:
        """List the versions of a file by path.

        :param path: The path to the file.
        :return: A sorted list of file versions. The latest version is the first element of the list.
        """
        file_response = await self._api.get_file_by_path(
            organisation_id=str(self._environment.org_id),
            workspace_id=str(self._environment.workspace_id),
            file_path=path,
            include_versions=True,
        )
        return _versions_from_listed_versions(file_response.versions)

    async def list_versions_by_id(self, file_id: UUID) -> list[FileVersion]:
        """List the versions of a file by ID

        :param file_id: UUID of the file.
        :return: A sorted list of file versions. The latest version is the first element of the list.
        """
        file_response = await self._api.get_file_by_id(
            organisation_id=str(self._environment.org_id),
            workspace_id=str(self._environment.workspace_id),
            file_id=str(file_id),
            include_versions=True,
        )
        return _versions_from_listed_versions(file_response.versions)

    async def prepare_download_by_path(self, path: str, version_id: str | None = None) -> FileAPIDownload:
        """Prepares a file for download by path.

        :param path: Path to the file.
        :param version_id: Versions of the file.

        :return: A FileAPIDownload object.
        """
        response = await self._api.get_file_by_path(
            organisation_id=str(self._environment.org_id),
            workspace_id=str(self._environment.workspace_id),
            file_path=path,
            version_id=version_id,
        )
        metadata = self._metadata_from_endpoint_model(response)
        return FileAPIDownload(connector=self._connector, metadata=metadata, initial_url=response.download)

    async def prepare_download_by_id(self, file_id: UUID, version_id: str | None = None) -> FileAPIDownload:
        """Prepares a file for download by ID.

        :param file_id: UUID of the file.
        :param version_id: Version of the file.

        :return: A FileAPIDownload object.
        """
        response = await self._api.get_file_by_id(
            organisation_id=str(self._environment.org_id),
            workspace_id=str(self._environment.workspace_id),
            file_id=str(file_id),
            version_id=version_id,
        )
        metadata = self._metadata_from_endpoint_model(response)
        return FileAPIDownload(connector=self._connector, metadata=metadata, initial_url=response.download)

    async def prepare_upload_by_path(self, path: str) -> FileAPIUpload:
        """Prepares a file for upload by path. If the file already exists, a new version will be created.

        :param path: Path the file is being uploaded to.

        :return: A FileAPIUpload object.
        """
        response = await self._api.upsert_file_by_path(
            organisation_id=str(self._environment.org_id),
            workspace_id=str(self._environment.workspace_id),
            file_path=path,
        )
        return FileAPIUpload(
            connector=self._connector,
            environment=self._environment,
            file_id=response.file_id,
            version_id=response.version_id,
            initial_url=response.upload,
        )

    async def prepare_upload_by_id(self, file_id: UUID) -> FileAPIUpload:
        """Prepares a file for upload by ID. The file_id must be the ID of an existing file, for which a new version
        will be created.

        :param file_id: UUID of the file.

        :return: A FileAPIUpload object.
        """
        response = await self._api.update_file_by_id(
            organisation_id=str(self._environment.org_id),
            workspace_id=str(self._environment.workspace_id),
            file_id=str(file_id),
        )
        return FileAPIUpload(
            connector=self._connector,
            environment=self._environment,
            file_id=file_id,
            version_id=response.version_id,
            initial_url=response.upload,
        )

    async def delete_file_by_path(self, path: str) -> None:
        """Deletes a file by path.

        :param path: Path of the file to delete.
        """
        await self._api.delete_file_by_path(
            organisation_id=str(self._environment.org_id),
            workspace_id=str(self._environment.workspace_id),
            file_path=path,
        )

    async def delete_file_by_id(self, file_id: UUID) -> None:
        """Deletes a file by ID.

        :param file_id: UUID of the file to delete.
        """
        await self._api.delete_file_by_id(
            organisation_id=str(self._environment.org_id),
            workspace_id=str(self._environment.workspace_id),
            file_id=str(file_id),
        )
