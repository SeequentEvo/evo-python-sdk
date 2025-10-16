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

import contextlib
from collections.abc import AsyncGenerator
from io import BytesIO
from logging import getLogger
from types import TracebackType
from typing import cast

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import TypeAdapter

from evo.common import ICache, IFeedback, ITransport
from evo.common.io import BytesDestination, ChunkedIOManager, Download, HTTPSource
from evo.common.utils import NoFeedback

from ..exceptions import SchemaValidationError, TableFormatError
from ..utils import ArrowTableFormat, KnownTableFormat
from .types import TableInfo

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

__all__ = [
    "ParquetDownloader",
    "ParquetLoader",
]

logger = getLogger(__name__)

_TABLE_INFO_ADAPTER: TypeAdapter[TableInfo] = TypeAdapter(TableInfo)


class ParquetLoader:
    """A loader for Parquet data from a pyarrow.parquet.ParquetFile.

    This class adds standardised support for validating Geoscience Object table info
    against the loaded Parquet schema, as well as convenience methods for loading
    the data as a PyArrow Table, Pandas DataFrame, or NumPy array.
    """

    def __init__(self, pa_file: pa.NativeFile) -> None:
        """
        :param pa_file: A PyArrow NativeFile containing the Parquet data.
        """
        self._pa_file = pa_file
        self._parquet_file: pq.ParquetFile | None = None

    def __enter__(self) -> ParquetLoader:
        if self._parquet_file is not None:
            raise RuntimeError("ParquetLoader is already in use")
        self._parquet_file = pq.ParquetFile(self._pa_file.__enter__())
        return self

    async def __aenter__(self) -> ParquetLoader:
        # Delegate to the synchronous context manager.
        # This implementation is just to support async with
        # syntax for combination with ParquetDownloader below.
        return self.__enter__()

    def __exit__(
        self,
        exc_type: type[Exception] | None,
        exc_val: Exception | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self._parquet_file = None
        return self._pa_file.__exit__(exc_type, exc_val, exc_tb)

    async def __aexit__(
        self,
        exc_type: type[Exception] | None,
        exc_val: Exception | None,
        exc_tb: TracebackType | None,
    ) -> None:
        # Delegate to the synchronous context manager.
        # This implementation is just to support async with
        # syntax for combination with ParquetDownloader below.
        return self.__exit__(exc_type, exc_val, exc_tb)

    def validate_with_table_info(self, table_info: TableInfo) -> None:
        """Validate the provided TableInfo against the loaded Parquet schema.

        :param table_info: The TableInfo to validate against the loaded Parquet schema.

        :raises SchemaValidationError: If the loaded Parquet schema does not match the expected schema.
        """
        if (pa_file := self._parquet_file) is None:
            raise RuntimeError("ParquetLoader context is not active")

        logger.debug("Checking parquet data format")

        validated_table_info = _TABLE_INFO_ADAPTER.validate_python(table_info)
        expected_format = KnownTableFormat.from_table_info(validated_table_info)
        actual_format = ArrowTableFormat.from_schema(pa_file.schema_arrow)
        try:
            expected_format._check_format(actual_format)
        except TableFormatError as e:
            raise SchemaValidationError(str(e)) from None

        logger.debug("Checking parquet data length")
        actual_length = pa_file.metadata.num_rows
        if table_info["length"] != actual_length:
            raise SchemaValidationError(
                f"Row count ({actual_length}) does not match expectation ({table_info['length']})"
            )

        logger.debug("Parquet metadata checks succeeded")

    def load_as_table(self) -> pa.Table:
        """Load the Parquet data as a PyArrow Table."""
        if self._parquet_file is None:
            raise RuntimeError("ParquetLoader context is not active")
        else:
            return self._parquet_file.read()

    if _PD_AVAILABLE:
        # Optional support for pandas dataframes

        def load_as_dataframe(self) -> pd.DataFrame:
            """Load the Parquet data as a Pandas DataFrame."""
            table = self.load_as_table()
            return table.to_pandas()

    if _NP_AVAILABLE:
        # Optional support for numpy arrays

        def load_as_array(self) -> np.ndarray:
            """Load the Parquet data as a NumPy array.

            The array will have a shape of (N,) for single-column data or (N, M) for multi-column data,
            where N is the number of rows and M is the number of columns. The target data _must_ have a uniform dtype.

            :return: A NumPy array containing the data.
            """
            table = self.load_as_table()
            columns = cast(
                list[np.ndarray], [col.combine_chunks().to_numpy(zero_copy_only=False) for col in table.itercolumns()]
            )
            if len(columns) == 1:
                return columns[0]
            else:
                return np.column_stack(columns)


class ParquetDownloader:
    """A downloader for Parquet data that provides a ParquetLoader for reading the data.

    This class supports downloading the data to a cache or to memory, and provides
    a ParquetLoader for reading the downloaded data.
    """

    def __init__(self, download: Download, transport: ITransport, cache: ICache | None = None) -> None:
        """
        :param download: The download information for the Parquet data.
        :param transport: The transport to use for data downloads.
        :param cache: An optional cache to use for data downloads.
        """
        self._evo_download = download
        self._transport = transport
        self._cache = cache

    async def _download_to_cache(self, fb: IFeedback) -> pa.OSFile:
        cached = await self._evo_download.download_to_cache(self._cache, self._transport, fb=fb)
        return pa.OSFile(str(cached), "r")

    async def _download_to_memory(self, fb: IFeedback) -> pa.BufferReader:
        # Initialize a buffer to store the downloaded data in memory
        memory = BytesIO()

        # Use ChunkedIOManager to download the data into the memory buffer
        manager = ChunkedIOManager()
        async with HTTPSource(self._evo_download.get_download_url, self._transport) as source:
            destination = BytesDestination(memory)
            await manager.run(source, destination, fb=fb)

        # Reset the buffer's position to the beginning
        memory.seek(0)
        return pa.BufferReader(memory.getbuffer())

    async def download(self, fb: IFeedback = NoFeedback) -> ParquetLoader:
        """Download the Parquet data and return a ParquetLoader for reading it.

        :param fb: An optional feedback instance to report download progress to.

        :return: A ParquetLoader that can be used to read the downloaded data.
        """
        if self._cache is not None:
            file = await self._download_to_cache(fb)
        else:
            file = await self._download_to_memory(fb)

        return ParquetLoader(file)

    @contextlib.asynccontextmanager
    async def __aenter__(self) -> AsyncGenerator[ParquetLoader, None]:
        # Delegate to the download method to get a ParquetLoader.
        async with await self.download() as loader:
            yield loader

    @contextlib.asynccontextmanager
    async def with_feedback(self, fb: IFeedback) -> AsyncGenerator[ParquetLoader, None]:
        """Async context manager to download the Parquet data with feedback and provide a ParquetLoader for reading it.

        :param fb: A feedback instance to report download progress to.

        :yields: A ParquetLoader that can be used to read the downloaded data.
        """
        async with await self.download(fb=fb) as loader:
            yield loader
