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
import contextlib
from collections.abc import AsyncGenerator, Iterator, Sequence
from typing import Any, TypeVar
from uuid import UUID

from pydantic import ConfigDict, TypeAdapter

from evo import jmespath, logging
from evo.common import APIConnector, ICache, IContext, IFeedback
from evo.common.io.exceptions import DataNotFoundError
from evo.common.utils import NoFeedback, split_feedback

from ..data import ObjectMetadata, ObjectReference, ObjectSchema
from ..endpoints import ObjectsApi, models
from ..io import ObjectDataDownload
from . import parse

try:
    import pyarrow as pa
    import pyarrow.compute as pc

    from ..parquet import (
        AttributeInfo,
        CategoryInfo,
        ParquetDownloader,
        ParquetLoader,
        TableInfo,
    )
except ImportError as e:
    print(e)
    _LOADER_AVAILABLE = False
else:
    _LOADER_AVAILABLE = True

    _TABLE_INFO_VALIDATOR: TypeAdapter[TableInfo] = TypeAdapter(TableInfo, config=ConfigDict(extra="ignore"))
    _CATEGORY_INFO_VALIDATOR: TypeAdapter[CategoryInfo] = TypeAdapter(CategoryInfo)
    _NAN_VALIDATOR: TypeAdapter[list[int] | list[float]] = TypeAdapter(
        list[int] | list[float], config=ConfigDict(extra="ignore")
    )
    _ATTRIBUTE_VALIDATOR: TypeAdapter[AttributeInfo] = TypeAdapter(AttributeInfo)

try:
    import pandas as pd
except ImportError:
    _PD_AVAILABLE = False
else:
    _PD_AVAILABLE = True

try:
    import numpy as np
except ImportError:
    _NP_AVAILABLE = False
else:
    _NP_AVAILABLE = True

__all__ = ["DownloadedObject"]

logger = logging.getLogger("object.client")

_T = TypeVar("_T")


class DownloadedObject:
    """A downloaded geoscience object."""

    def __init__(
        self,
        object_: models.GeoscienceObject,
        metadata: ObjectMetadata,
        urls_by_name: dict[str, str],
        connector: APIConnector,
        cache: ICache | None = None,
    ) -> None:
        """
        :param object_: The raw geoscience object model.
        :param metadata: The parsed metadata for the object.
        :param urls_by_name: A mapping of data names to their initial download URLs.
        :param connector: The API connector to use for downloading data.
        :param cache: An optional cache to use for data downloads.
        """
        self._object = object_
        self._metadata = metadata
        self._urls_by_name = urls_by_name
        self._connector = connector
        self._cache = cache

    @staticmethod
    async def from_reference(
        connector: APIConnector,
        reference: ObjectReference | str,
        cache: ICache | None = None,
        request_timeout: int | float | tuple[int | float, int | float] | None = None,
    ) -> DownloadedObject:
        """Download a geoscience object from the service, given an object reference.

        :param connector: The API connector to use for downloading data.
        :param reference: The reference to the object to download, or a URL as a string that can be parsed into
            a reference.
        :param cache: An optional cache to use for data downloads.
        :param request_timeout: An optional timeout to use for API requests. See evo.common.APIConnector for details.

        :raises ValueError: If the reference is invalid, or if the connector base URL does not match the reference hub URL.
        """
        ref = ObjectReference(reference)  # Parse the reference if it's a string

        if connector.base_url != ref.hub_url:
            raise ValueError(
                f"The connector base URL '{connector.base_url}' does not match the reference hub URL '{ref.hub_url}'"
            )

        api = ObjectsApi(connector)

        request_kwargs = dict(
            org_id=str(ref.org_id),
            workspace_id=str(ref.workspace_id),
            version=ref.version_id,
            additional_headers={"Accept-Encoding": "gzip"},
            request_timeout=request_timeout,
        )

        if ref.object_id is not None and ref.object_path is not None:
            raise ValueError("Only one of object_id or object_path should be provided")

        if ref.object_id is not None:
            response = await api.get_object_by_id(object_id=ref.object_id, **request_kwargs)
        elif ref.object_path is not None:
            response = await api.get_object(objects_path=ref.object_path, **request_kwargs)
        else:
            raise ValueError("Either object_id or object_path must be provided")

        metadata = parse.object_metadata(response, ref.environment)
        urls_by_name = {getattr(link, "name", link.id): link.download_url for link in response.links.data}
        return DownloadedObject(
            object_=response.object,
            metadata=metadata,
            urls_by_name=urls_by_name,
            connector=connector,
            cache=cache,
        )

    @staticmethod
    async def from_context(
        context: IContext,
        reference: ObjectReference | str,
    ) -> DownloadedObject:
        """Download a geoscience object from the service using a context.

        :param context: The context providing the connector and cache.
        :param reference: The reference to the object to download, or a URL as a string that can be parsed into
            a reference.

        :raises ValueError: If the reference is invalid, or if the connector base URL does not match the reference hub URL.
        """
        return await DownloadedObject.from_reference(
            connector=context.get_connector(),
            reference=reference,
            cache=context.get_cache(),
        )

    @property
    def schema(self) -> ObjectSchema:
        """The schema of the object."""
        return self._metadata.schema_id

    @property
    def metadata(self) -> ObjectMetadata:
        """The metadata of the object."""
        return self._metadata

    def as_dict(self) -> dict:
        """Get this object as a dictionary."""
        return self._object.model_dump(mode="python", by_alias=True)

    def search(self, expression: str) -> Any:
        """Search the object metadata using a JMESPath expression.

        :param expression: The JMESPath expression to use for the search.

        :return: The result of the search.
        """
        return jmespath.search(expression, self.as_dict())

    def prepare_data_download(self, data_identifiers: Sequence[str | UUID]) -> Iterator[ObjectDataDownload]:
        """Prepare to download multiple data files from the geoscience object service, for this object.

        Any data IDs that are not associated with the requested object will raise a DataNotFoundError.

        :param data_identifiers: A list of sha256 digests or UUIDs for the data to be downloaded.

        :return: An iterator of data download contexts that can be used to download the data.

        :raises DataNotFoundError: If any requested data ID is not associated with this object.
        """
        try:
            filtered_urls_by_name = {str(name): self._urls_by_name[str(name)] for name in data_identifiers}
        except KeyError as exc:
            raise DataNotFoundError(f"Unable to find the requested data: {exc.args[0]}") from exc
        for ctx in ObjectDataDownload._create_multiple(
            connector=self._connector, metadata=self._metadata, urls_by_name=filtered_urls_by_name
        ):
            yield ctx

    async def update(
        self,
        object_dict: dict,
        check_for_conflict: bool = True,
        request_timeout: int | float | tuple[int | float, int | float] | None = None,
    ) -> DownloadedObject:
        """Update the geoscience object on the geoscience object service. Returning a new DownloadedObject representing
        the new version of the object.

        This will create a new version of the object, that fully replaces the existing properties of the object with
        those provided in `object_dict`.

        Note, this will not update the "DownloadedObject" instance in-place - it will still represent the original
        version of the object. You will need to download the updated version separately if you wish to work with it.

        :param object_dict: The new properties of the object as a dictionary.
        :param check_for_conflict: If True, and if a newer version of the object exists on the geoscience object
            service, the update will fail with a ObjectModifiedError exception. If False, it will not check whether
            there is a newer version, so will perform the update regardless.
        :param request_timeout: An optional timeout to use for API requests. See evo.common.APIConnector for details.

        :returns: The new version of the object as a DownloadedObject.
        """

        api = ObjectsApi(self._connector)

        if "uuid" not in object_dict:
            object_dict["uuid"] = self.metadata.id

        model = models.UpdateGeoscienceObject.model_validate(object_dict)
        if model.uuid != self.metadata.id:
            raise ValueError("The object ID in the new object does not match the current object ID")

        response = await api.update_objects_by_id(
            object_id=str(self.metadata.id),
            org_id=str(self.metadata.environment.org_id),
            workspace_id=str(self.metadata.environment.workspace_id),
            update_geoscience_object=model,
            request_timeout=request_timeout,
            if_match=self.metadata.version_id if check_for_conflict else None,
        )
        metadata = parse.object_metadata(response, self.metadata.environment)
        urls_by_name = {getattr(link, "name", link.id): link.download_url for link in response.links.data}
        return DownloadedObject(
            object_=response.object,
            metadata=metadata,
            urls_by_name=urls_by_name,
            connector=self._connector,
            cache=self._cache,
        )

    if _LOADER_AVAILABLE:
        # Optional support for loading Parquet data using PyArrow.

        def _validate_typed_dict(self, value: _T | str, validator: TypeAdapter[_T]) -> _T:
            if isinstance(value, str):
                resolved = self.search(value)
                # Implicitly unwrap single-element arrays for convenience
                # This allows, using predicates like: attributes[?name=='my_attribute']
                if isinstance(resolved, jmespath.JMESPathArrayProxy) and len(resolved) == 1:
                    resolved = resolved[0]
                if isinstance(resolved, jmespath.JMESPathObjectProxy):
                    value = resolved.raw
                else:
                    raise ValueError(f"Expected object, got {type(resolved)}")
            return validator.validate_python(value)

        def _validate_nan_values(self, nan_values: list[int] | list[float] | str | None) -> list[int] | list[float]:
            if nan_values is None:
                return []
            if isinstance(nan_values, str):
                resolved = self.search(nan_values)
                if isinstance(resolved, jmespath.JMESPathArrayProxy) and len(resolved) == 1:
                    # Consider single-element arrays for unwrapping
                    # This allows, using predicates like: attributes[?name=='my_attribute']
                    child = resolved[0]
                    if not isinstance(child, (int, float)):
                        resolved = child
                if isinstance(resolved, jmespath.JMESPathArrayProxy):
                    nan_values = resolved.raw
                # Support passing nan_description structure too
                elif isinstance(resolved, jmespath.JMESPathObjectProxy) and "values" in resolved.raw:
                    nan_values = resolved.raw["values"]
                else:
                    raise ValueError(f"Expected list, got {type(resolved)}")
            return _NAN_VALIDATOR.validate_python(nan_values)

        @contextlib.asynccontextmanager
        async def _with_parquet_loader(
            self, table_info: TableInfo | str, fb: IFeedback
        ) -> AsyncGenerator[ParquetLoader, None]:
            """Download parquet data and get a ParquetLoader for the data referenced by the given
            table info or data reference string.

            :param table_info: The table info dict, JMESPath to table info within the object.
            :param fb: An optional feedback instance to report download progress to.

            :returns: A ParquetLoader that can be used to read the referenced data.
            """
            table_info = self._validate_typed_dict(table_info, _TABLE_INFO_VALIDATOR)
            (download,) = self.prepare_data_download([table_info["data"]])
            async with ParquetDownloader(download, self._connector.transport, self._cache).with_feedback(fb) as loader:
                loader.validate_with_table_info(table_info)
                yield loader

        async def download_table(
            self,
            table_info: TableInfo | str,
            fb: IFeedback = NoFeedback,
            *,
            nan_values: list[int] | list[float] | str | None = None,
            column_names: Sequence[str] | None = None,
        ) -> pa.Table:
            """Download the data referenced by the given table info as a PyArrow Table.

            :param table_info: The table info dict, ot JMESPath to table info within the object.
            :param fb: An optional feedback instance to report download progress to.
            :param nan_values: An optional list of values to treat as null. Can also be a JMESPath expression to the
               list of nan values, or the nan_description structure.
            :param column_names: An optional list of column names for the table, instead of those in the Parquet file.

            :returns: A PyArrow Table containing the downloaded data.
            """
            async with self._with_parquet_loader(table_info, fb) as loader:
                table = loader.load_as_table()

            if column_names is None:
                column_names = table.column_names

            nan_values = self._validate_nan_values(nan_values)
            if len(nan_values) == 0:
                return table.rename_columns(column_names)

            # Replace specified nan_values with nulls
            arrays = []
            for array in table.columns:
                if isinstance(array, pa.ChunkedArray):
                    array = array.combine_chunks()
                null_scalar = pa.scalar(None, type=array.type)
                nan_value_array = pa.array(nan_values, type=array.type)
                arrays.append(pc.replace_with_mask(array, pc.is_in(array, nan_value_array), null_scalar))
            return pa.Table.from_arrays(arrays, names=column_names)

        async def download_category_table(
            self,
            category_info: CategoryInfo | str,
            *,
            nan_values: list[int] | list[float] | str | None = None,
            column_names: Sequence[str] | None = None,
            fb: IFeedback = NoFeedback,
        ) -> pa.Table:
            """Download the data referenced by the given category info as a PyArrow Table.

            The arrays into the table will be DictionaryArrays constructed from the values and lookup tables.

            :param category_info: The category info dict, or JMESPath to the category info within the object.
            :param nan_values: An optional list of values to treat as null. Can also be a JMESPath expression to
                nan_description structure.
            :param column_names: An optional list of column names for the table, instead of those in the Parquet file.
            :param fb: An optional feedback instance to report download progress to.

            :returns: A PyArrow Table containing the downloaded data.
            """
            category_info = self._validate_typed_dict(category_info, _CATEGORY_INFO_VALIDATOR)

            v_size = (
                category_info["values"]["length"] * category_info["values"]["width"]
            )  # Total number of cells in values
            t_size = category_info["table"]["length"] * 2  # Lookup tables always have 2 columns
            values_fb, table_fb = split_feedback(fb, [v_size, t_size])

            # Download both tables concurrently
            values_table_coro = self.download_table(
                category_info["values"],
                nan_values=nan_values,
                column_names=column_names,
                fb=values_fb,
            )
            lookup_table_coro = self.download_table(category_info["table"], fb=table_fb)
            values_table, lookup_table = await asyncio.gather(values_table_coro, lookup_table_coro)

            arrays = []
            for array in values_table.columns:
                indices = pc.index_in(array, lookup_table[0])
                arrays.append(pa.DictionaryArray.from_arrays(indices, lookup_table[1]))
            return pa.Table.from_arrays(arrays, names=values_table.column_names)

        async def download_attribute_table(
            self,
            attribute: AttributeInfo | str,
            fb: IFeedback = NoFeedback,
        ) -> pa.Table:
            """Download the data referenced by the given attribute as a PyArrow Table.

            :param attribute: The attribute info dict, or JMESPath to the attribute info within the object.
            :param fb: An optional feedback instance to report download progress to.

            :returns: A PyArrow Table containing the downloaded data.
            """
            attribute = self._validate_typed_dict(attribute, _ATTRIBUTE_VALIDATOR)

            if "table" in attribute:
                table = await self.download_category_table(
                    attribute,
                    nan_values=attribute["nan_description"]["values"] if "nan_description" in attribute else None,
                    fb=fb,
                )
            else:
                table = await self.download_table(
                    attribute["values"],
                    nan_values=attribute["nan_description"]["values"] if "nan_description" in attribute else None,
                    fb=fb,
                )
            if len(table.column_names) == 1:
                table = table.rename_columns([attribute["name"]])
            else:
                table = table.rename_columns([f"{attribute['name']}_{i}" for i in range(len(table.column_names))])
            return table

        if _PD_AVAILABLE:
            # Optional support for loading data as Pandas DataFrames. Requires parquet support via PyArrow as well.

            async def download_dataframe(
                self,
                table_info: TableInfo | str,
                fb: IFeedback = NoFeedback,
                *,
                nan_values: list[int] | list[float] | str | None = None,
                column_names: Sequence[str] | None = None,
            ) -> pd.DataFrame:
                """Download the data referenced by the given table info as a Pandas DataFrame.

                :param table_info: The table info dict, JMESPath to table info within the object.
                :param fb: An optional feedback instance to report download progress to.
                :param nan_values: An optional list of values to treat as null. Can also be a JMESPath expression to
                    nan_description structure.
                :param column_names: An optional list of column names for the table, instead of those from the Parquet file.

                :returns: A Pandas DataFrame containing the downloaded data.
                """
                table = await self.download_table(table_info, fb=fb, nan_values=nan_values, column_names=column_names)
                return table.to_pandas()

            async def download_category_dataframe(
                self,
                category_info: CategoryInfo | str,
                fb: IFeedback = NoFeedback,
                *,
                nan_values: list[int] | list[float] | str | None = None,
                column_names: Sequence[str] | None = None,
            ) -> pd.DataFrame:
                """Download the data referenced by the given category info as a Pandas DataFrame.

                :param category_info: The category info dict, or JMESPath to the category info within the object.
                :param nan_values: An optional list of values to treat as null. Can also be a JMESPath expression to
                    nan_description structure.
                :param column_names: An optional list of column names for the table, instead of those from the Parquet file.
                :param fb: An optional feedback instance to report download progress to.

                :returns: A Pandas DataFrame containing the downloaded data.
                """
                table = await self.download_category_table(
                    category_info, fb=fb, nan_values=nan_values, column_names=column_names
                )
                return table.to_pandas()

            async def download_attribute_dataframe(
                self,
                attribute: AttributeInfo | str,
                fb: IFeedback = NoFeedback,
            ) -> pd.DataFrame:
                """Download the data referenced by the given attribute as a Pandas DataFrame.

                :param attribute: The attribute info dict, or JMESPath to the attribute within the object.
                :param fb: An optional feedback instance to report download progress to.

                :returns: A Pandas DataFrame containing the downloaded data.
                """
                attribute = self._validate_typed_dict(attribute, _ATTRIBUTE_VALIDATOR)

                if "table" in attribute:
                    df = await self.download_category_dataframe(
                        attribute,
                        nan_values=attribute["nan_description"]["values"] if "nan_description" in attribute else None,
                        fb=fb,
                    )
                else:
                    df = await self.download_dataframe(
                        attribute["values"],
                        nan_values=attribute["nan_description"]["values"] if "nan_description" in attribute else None,
                        fb=fb,
                    )
                if len(df.columns) == 1:
                    df.columns = [attribute["name"]]
                else:
                    df.columns = [f"{attribute['name']}_{i}" for i in range(len(df.columns))]
                return df

        if _NP_AVAILABLE:
            # Optional support for loading data as NumPy arrays. Requires parquet support via PyArrow as well.

            async def download_array(self, table_info: TableInfo | str, fb: IFeedback = NoFeedback) -> np.ndarray:
                """Download the data referenced by the given table info as a NumPy array.

                :param table_info: The table info dict, JMESPath to table info within the object.
                :param fb: An optional feedback instance to report download progress to.

                :returns: A NumPy array containing the downloaded data.
                """
                async with self._with_parquet_loader(table_info, fb) as loader:
                    return loader.load_as_array()
