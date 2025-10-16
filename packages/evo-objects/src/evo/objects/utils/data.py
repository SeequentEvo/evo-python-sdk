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

from collections.abc import Iterator
from pathlib import Path
from typing import Any
from uuid import UUID

from evo import logging
from evo.common import APIConnector, Environment, ICache, IFeedback
from evo.common.exceptions import StorageFileNotFoundError
from evo.common.io.exceptions import DataExistsError
from evo.common.utils import NoFeedback, PartialFeedback

from ..io import _CACHE_SCOPE, ObjectDataUpload

try:
    import pyarrow as pa
except ImportError:
    raise ImportError("ObjectDataClient requires the `pyarrow` package to be installed")

try:
    import pandas as pd
except ImportError:
    _PD_AVAILABLE = False
else:
    _PD_AVAILABLE = True

__all__ = ["ObjectDataClient"]

logger = logging.getLogger("object.data")

_DATA_ID_KEY = "data"  # The key used to identify data references in geoscience objects.


def _iter_refs(target: Any, _key: str | None = None) -> Iterator[str]:
    """Iterate over all data references in a geoscience object.

    :param target: Any part of a geoscience object.
    :param _key: The key of the current target, if any.

    :return: An iterator over all data references in the geoscience object.
    """
    match target:
        case dict() as nested:
            for key, value in nested.items():
                yield from _iter_refs(value, key)
        case list() as items:
            for item in items:
                yield from _iter_refs(item)
        case str() | UUID() as value if _key == _DATA_ID_KEY:
            yield str(value)


class ObjectDataClient:
    """An optional wrapper around data upload and download functionality for geoscience objects.

    This class provides a high-level interface for uploading and downloading data that is referenced in geoscience
    objects, and caching the data locally. It depends on the optional dependency `pyarrow`, which is not installed
    by default. This dependency can be installed with `pip install evo-objects[utils]`.
    """

    def __init__(self, environment: Environment, connector: APIConnector, cache: ICache) -> None:
        """
        :param environment: The environment to upload and download data from.
        :param connector: The API connector to use for uploading and downloading data.
        :param cache: The cache to use for storing data locally.
        """
        self._environment = environment
        self._connector = connector
        self._cache = cache

    @property
    def cache_location(self) -> Path:
        """The location of the cache for this client."""
        return self._cache.get_location(environment=self._environment, scope=_CACHE_SCOPE)

    def clear_cache(self) -> None:
        """Clear the cache used by this client."""
        self._cache.clear_cache(environment=self._environment, scope=_CACHE_SCOPE)

    async def upload_referenced_data(self, object_model: dict, fb: IFeedback = NoFeedback) -> None:
        """Upload all data referenced by a geoscience object.

        All referenced data that has not already been uploaded must be available in the cache. The best way of ensuring
        this is to use the `save_table` method to save the data to the cache before calling this method.

        :param object_model: The geoscience object to upload data from.
        :param fb: The feedback object to report progress to.

        :raises StorageFileNotFoundError: If the data to be uploaded is not found in the cache.
        """
        names = list(_iter_refs(object_model))
        n_names = len(names)
        logger.debug(f"Found {n_names} data reference{'' if n_names == 1 else 's'}")
        n_uploaded = 0
        async for ctx in ObjectDataUpload._create_multiple(
            connector=self._connector,
            environment=self._environment,
            names=names,
        ):
            if fb is not NoFeedback:
                fb_part = PartialFeedback(parent=fb, start=n_uploaded / n_names, end=(n_uploaded + 1) / n_names)
            else:
                fb_part = NoFeedback

            data_file = self.cache_location / ctx.name
            logger.debug(f"Uploading data: {ctx.name}")
            if not data_file.exists():
                raise StorageFileNotFoundError(f"Object data file not found in cache: {data_file}")
            await ctx.upload_from_path(filename=data_file, transport=self._connector.transport, fb=fb_part)
            n_uploaded += 1

        logger.debug(
            f"Uploaded {n_uploaded} data reference{'' if n_uploaded == 1 else 's'}, skipped {n_names - n_uploaded}"
        )
        fb.progress(1)

    def save_table(self, table: pa.Table) -> dict:
        """Save a pyarrow table to a file, returning the table info as a dictionary.

        :param table: The pyarrow table to save.

        :return: Information about the saved table.

        :raises TableFormatError: If the provided table does not match this format.
        :raises StorageFileNotFoundError: If the destination does not exist or is not a directory.
        """
        from .table_formats import get_known_format

        known_format = get_known_format(table)
        table_info = known_format.save_table(table=table, destination=self.cache_location)
        return table_info

    async def upload_table(self, table: pa.Table, fb: IFeedback = NoFeedback) -> dict:
        """Upload pyarrow table to the geoscience object service, returning a GO model of the uploaded data.

        :param table: The table to be uploaded.
        :param fb: A feedback object for tracking upload progress.

        :return: A description of the uploaded data.

        :raises TableFormatError: If the table does not match a known format.
        """
        table_info = self.save_table(table)
        upload = ObjectDataUpload(connector=self._connector, environment=self._environment, name=table_info["data"])
        try:
            await upload.upload_from_cache(cache=self._cache, transport=self._connector.transport, fb=fb)
        except DataExistsError:
            logger.debug(f"Data not uploaded because data already exists (label: {table_info['data']})")
            fb.progress(1)
        return table_info

    async def download_table(
        self, object_id: UUID, version_id: str, table_info: dict, fb: IFeedback = NoFeedback
    ) -> pa.Table:
        """Download pyarrow table from the geoscience object service.

        The parquet metadata will be used to make sure the file contents matches the expected format before the table
        is read into memory.

        :param object_id: The object ID to download the data from.
        :param version_id: The version ID to download the data from.
        :param table_info: The table info that defines the expected format. The model's `data` will be downloaded from
            the service.
        :param fb: A feedback object for tracking download progress.

        :return: A pyarrow table loaded directly from the parquet file.

        :raises DataNotFoundError: If the data does not exist or is not associated with this object version.
        :raises TableFormatError: If the data does not match the expected format.
        :raises SchemaValidationError: If the data has a different number of rows than expected.
        """
        # Import here to avoid circular import.
        from ..client import ObjectAPIClient
        from ..parquet import ParquetDownloader

        client = ObjectAPIClient(self._environment, self._connector)
        (download,) = [d async for d in client.prepare_data_download(object_id, version_id, [table_info["data"]])]

        # Defer downloading the table to the new ParquetLoader class.
        async with ParquetDownloader(
            download=download, transport=self._connector.transport, cache=self._cache
        ).with_feedback(fb) as loader:
            loader.validate_with_table_info(table_info)
            return loader.load_as_table()

    if _PD_AVAILABLE:
        # Optional support for pandas dataframes. Depends on both pyarrow and pandas.

        def save_dataframe(self, dataframe: pd.DataFrame) -> dict:
            """Save a pandas dataframe to a file, returning the table info as a dictionary.

            :param dataframe: The pandas dataframe to save.

            :return: Information about the saved table.

            :raises TableFormatError: If the provided table does not match this format.
            :raises StorageFileNotFoundError: If the destination does not exist or is not a directory.
            """
            return self.save_table(pa.Table.from_pandas(dataframe))

        async def upload_dataframe(self, dataframe: pd.DataFrame, fb: IFeedback = NoFeedback) -> dict:
            """Upload pandas dataframe to the geoscience object service, returning a GO model of the uploaded data.

            :param dataframe: The pandas dataframe to be uploaded.
            :param fb: A feedback object for tracking upload progress.

            :return: A description of the uploaded data.

            :raises TableFormatError: If the table does not match a known format.
            """
            table_info = await self.upload_table(pa.Table.from_pandas(dataframe), fb=fb)
            return table_info

        async def download_dataframe(
            self, object_id: UUID, version_id: str, table_info: dict, fb: IFeedback = NoFeedback
        ) -> pd.DataFrame:
            """Download pandas dataframe data from the geoscience object service.

            The parquet metadata will be used to make sure the file contents matches the expected format before the table
            is read into memory.

            :param object_id: The object ID to download the data from.
            :param version_id: The version ID to download the data from.
            :param table_info: The table info that defines the expected format. The model's `data` will be downloaded from
                the service.
            :param fb: A feedback object for tracking download progress.

            :return: A pandas dataframe loaded directly from the parquet file.

            :raises DataNotFoundError: If the data does not exist or is not associated with this object version.
            :raises TableFormatError: If the data does not match the expected format.
            :raises SchemaValidationError: If the data has a different number of rows than expected.
            """
            table = await self.download_table(object_id, version_id, table_info, fb)
            try:
                return table.to_pandas()
            except ModuleNotFoundError:
                raise RuntimeError("Unable to download dataframe because the `pandas` package is not installed")
