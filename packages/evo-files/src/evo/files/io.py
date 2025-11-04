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

import asyncio
from pathlib import Path
from uuid import UUID

from evo.common import APIConnector, Environment, ICache
from evo.common.io import Download, Upload

from .data import FileMetadata
from .endpoints import FileV2Api

__all__ = [
    "FileAPIDownload",
    "FileAPIUpload",
]

_CACHE_SCOPE = "filev2"
_INVALID_CHARS = {"<", ">", ":", '"', "/", "\\", "|", "?", "*"} | {chr(i) for i in range(32)}
_INVALID_TRAILING_CHARS = {" ", "."}
_INVALID_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    "COM1",
    "COM2",
    "COM3",
    "COM4",
    "COM5",
    "COM6",
    "COM7",
    "COM8",
    "COM9",
    "LPT1",
    "LPT2",
    "LPT3",
    "LPT4",
    "LPT5",
    "LPT6",
    "LPT7",
    "LPT8",
    "LPT9",
}


def _quote_char(c: str) -> str:
    return "%{:02X}".format(ord(c))


def _make_name_safe(name: str) -> str:
    # Replace invalid characters with URL encoded form.
    new_name = "".join(_quote_char(c) if c in _INVALID_CHARS else c for c in name)
    # Replace invalid trailing characters with URL encoded form.
    if new_name and new_name[-1] in _INVALID_TRAILING_CHARS:
        new_name = new_name[:-1] + _quote_char(new_name[-1])
    # Fix reserved names by replacing 'aux' with 'aux_' and 'aux.txt' with 'aux_.txt'.
    parts = new_name.split(".", 1)
    if parts[0].upper() in _INVALID_NAMES:
        if len(parts) == 1:
            new_name += "_"
        else:
            new_name = f"{parts[0]}_.{parts[1]}"
    return new_name


class _FileIOMixin:
    def __init__(self, connector: APIConnector, initial_url: str) -> None:
        self._api = FileV2Api(connector)
        self._mutex = asyncio.Lock()
        self._initial_url = initial_url

    async def _get_initial_url(self) -> str:
        async with self._mutex:
            if (url := self._initial_url) is not None:
                self._initial_url = None
                return url


class FileAPIUpload(Upload, _FileIOMixin):
    """A context for uploading a file to the File API.

    Do not use this class directly. Instead, use the `FileAPIClient.prepare_upload_by_*` methods.
    """

    def __init__(
        self, connector: APIConnector, environment: Environment, file_id: UUID, version_id: str, initial_url: str
    ) -> None:
        """
        :param connector: The connector to use for the API calls.
        :param environment: The environment the file is being uploaded to.
        :param file_id: The ID of the file that will be uploaded.
        :param version_id: The version ID of the file that will be uploaded.
        :param initial_url: The initial URL to use for the upload.
        """
        super().__init__(connector, initial_url)
        self._environment = environment
        self._id = file_id
        self._version_id = version_id

    @property
    def file_id(self) -> UUID:
        """The ID of the file that is being uploaded."""
        return self._id

    @property
    def version_id(self) -> str:
        """The ID of the version that will be created after the upload completes."""
        return self._version_id

    @property
    def label(self) -> str:
        """The file and version ID of the file that is being uploaded."""
        return f"{self._id}?version_id={self._version_id}"

    async def get_upload_url(self) -> str:
        """Generate a URL that will be used to upload the resource.

        This method may be called multiple times to generate a new URL if the last URL expires.

        :returns: The upload URL.
        """
        # Use the initial URL first.
        if (url := await self._get_initial_url()) is not None:
            return url

        # Otherwise, generate a new URL.
        response = await self._api.update_file_by_id(
            organisation_id=str(self._environment.org_id),
            workspace_id=str(self._environment.workspace_id),
            file_id=str(self._id),
            version_id=self._version_id,
        )
        return response.upload


class FileAPIDownload(Download[FileMetadata], _FileIOMixin):
    """A context for downloading a file from the File API.

    Do not use this class directly. Instead, use the `FileAPIClient.prepare_download_by_*` methods.
    """

    def __init__(self, connector: APIConnector, metadata: FileMetadata, initial_url: str) -> None:
        """
        :param connector: The connector to use for the API calls.
        :param metadata: The metadata of the file that will be downloaded.
        :param initial_url: The initial URL to use for the download.
        """
        super().__init__(connector, initial_url)
        self._metadata = metadata

    @property
    def label(self) -> str:
        """The file and version ID for the file that is being downloaded."""
        return f"{self.metadata.id}?version_id={self.metadata.version_id}"

    @property
    def metadata(self) -> FileMetadata:
        """The metadata of the file that is being downloaded."""
        return self._metadata

    def _get_cache_location(self, cache: ICache) -> Path:
        download_dir = (
            cache.get_location(self.metadata.environment, _CACHE_SCOPE)
            / str(self.metadata.id)
            / self.metadata.version_id
        )
        download_dir.mkdir(parents=True, exist_ok=True)
        return download_dir / _make_name_safe(self.metadata.name)

    async def get_download_url(self) -> str:
        """Generate a URL that will be used to download the resource.

        This method may be called multiple times to generate a new URL if the last URL expires.

        :returns: The download URL.
        """
        # Use the initial URL first.
        if (url := await self._get_initial_url()) is not None:
            return url

        # Otherwise, generate a new URL.
        response = await self._api.get_file_by_id(
            organisation_id=str(self.metadata.environment.org_id),
            workspace_id=str(self.metadata.environment.workspace_id),
            file_id=str(self.metadata.id),
            version_id=self.metadata.version_id,
        )
        return response.download
