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

import contextlib
import json
from collections.abc import Generator
from typing import cast
from unittest import mock
from urllib.parse import quote
from uuid import UUID

import pyarrow as pa
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal
from parameterized import parameterized

from data import load_test_data
from evo.common import RequestMethod
from evo.common.test_tools import (
    BASE_URL,
    ORG,
    WORKSPACE_ID,
    DownloadRequestHandler,
    TestWithConnector,
    TestWithStorage,
)
from evo.common.utils import NoFeedback
from evo.jmespath import JMESPathObjectProxy
from evo.objects import DownloadedObject, ObjectReference
from evo.objects.endpoints import models
from evo.objects.io import _CACHE_SCOPE
from evo.objects.parquet import TableInfo
from evo.objects.utils import KnownTableFormat
from helpers import NoImport, UnloadModule, get_sample_table_and_bytes

_OBJECTS_URL = f"{BASE_URL.rstrip('/')}/geoscience-object/orgs/{ORG.id}/workspaces/{WORKSPACE_ID}/objects"

_TABLE_INFO_VARIANTS: list[tuple[str, TableInfo | str]] = [
    (
        "with TableInfo dict",
        {
            "data": "0000000000000000000000000000000000000000000000000000000000000000",
            "length": 123,
            "width": 3,
            "data_type": "float64",
        },
    ),
    ("with JMESPath reference", "locations.coordinates"),
]


class TestDownloadedObject(TestWithConnector, TestWithStorage):
    def setUp(self) -> None:
        from evo.objects.client import parse

        TestWithConnector.setUp(self)
        TestWithStorage.setUp(self)

        raw = models.GetObjectResponse.model_validate(load_test_data("get_object_detailed.json"))
        self.object = DownloadedObject(
            object_=raw.object,
            metadata=parse.object_metadata(raw, self.environment),
            urls_by_name={link.name: link.download_url for link in raw.links.data},
            connector=self.connector,
            cache=self.cache,
        )

    def tearDown(self) -> None:
        # Clear cache between tests to avoid cached files interfering with subsequent tests.
        self.cache.clear_cache()

    @parameterized.expand(
        [
            ("by id as string", f"{_OBJECTS_URL}/00000000-0000-0000-0000-000000000002"),
            (
                "by id as ObjectReference",
                ObjectReference(f"{_OBJECTS_URL}/00000000-0000-0000-0000-000000000002"),
            ),
            (
                "by id with version id",
                ObjectReference(
                    f"{_OBJECTS_URL}/00000000-0000-0000-0000-000000000002?version=2023-08-03T05:47:18.3402289Z"
                ),
            ),
            ("by path as string", f"{_OBJECTS_URL}/path/A/m.json"),
            ("by path as ObjectReference", ObjectReference(f"{_OBJECTS_URL}/path/A/m.json")),
            (
                "by path with version id",
                ObjectReference(f"{_OBJECTS_URL}/path/A/m.json?version=2023-08-03T05:47:18.3402289Z"),
            ),
        ]
    )
    async def test_from_reference(self, _label: str, reference: str) -> None:
        """Test downloading a geoscience object by reference."""
        get_object_response = load_test_data("get_object.json")
        expected_uuid = UUID(int=2)
        expected_object_dict = {
            "schema": "/objects/pointset/1.0.1/pointset.schema.json",
            "uuid": UUID("00000000-0000-0000-0000-000000000002"),
            "name": "Sample pointset",
            "description": "A sample pointset object",
            "bounding_box": {"min_x": 0.0, "max_x": 0.0, "min_y": 0.0, "max_y": 0.0, "min_z": 0.0, "max_z": 0.0},
            "coordinate_reference_system": {"epsg_code": 2048},
            "locations": {
                "coordinates": {
                    "data": "0000000000000000000000000000000000000000000000000000000000000001",
                    "length": 1,
                    "width": 3,
                    "data_type": "float64",
                }
            },
        }
        expected_path = "A/m.json"
        expected_version = "2023-08-03T05:47:18.3402289Z"
        with self.transport.set_http_response(status_code=200, content=json.dumps(get_object_response)):
            actual_object = await DownloadedObject.from_reference(self.connector, reference, self.cache)

        ref = ObjectReference(reference)
        if ref.object_id is not None:
            expected_request_path = f"{_OBJECTS_URL}/{ref.object_id}"
        else:
            expected_request_path = f"{_OBJECTS_URL}/path/{ref.object_path}"

        if ref.version_id is not None:
            expected_request_path += f"?version={quote(ref.version_id)}"

        self.assert_request_made(
            method=RequestMethod.GET,
            path=expected_request_path,
            headers={"Accept": "application/json", "Accept-Encoding": "gzip"},
        )
        # Check metadata.
        actual_metadata = actual_object.metadata
        self.assertEqual(expected_path, actual_metadata.path)
        self.assertEqual("A", actual_metadata.parent)
        self.assertEqual("m.json", actual_metadata.name)
        self.assertEqual(expected_uuid, actual_metadata.id)
        self.assertEqual(expected_version, actual_metadata.version_id)

        # Check geoscience object.
        self.assertEqual(expected_object_dict, actual_object.as_dict())

    def test_search(self) -> None:
        """Test the JMESPath search implementation."""
        expected_result = JMESPathObjectProxy(
            {
                "x": [0.0, 1.0],
                "y": [2.0, 3.0],
                "z": [4.0, 5.0],
            }
        )
        actual_result = self.object.search("bounding_box | {x: [min_x, max_x], y: [min_y, max_y], z: [min_z, max_z]}")
        self.assertEqual(expected_result, actual_result)

    def _assert_optional_method(self, method_name: str, *, unload: list[str], no_import: list[str]) -> None:
        # Verify the method exists before unloading any modules.
        from evo.objects.client import DownloadedObject

        self.assertTrue(
            all(
                [
                    hasattr(DownloadedObject, method_name),
                    hasattr(self.object, method_name),
                ]
            ),
            f"DownloadedObject.{method_name} should be available for this test to be valid",
        )

        with UnloadModule("evo.objects.client.object_client", *unload), NoImport(*no_import):
            # Re-import the class to ensure the module is re-evaluated without the optional dependency.
            from evo.objects.client import DownloadedObject

            # Re-create the object to ensure the class is re-evaluated without the optional dependency.
            client = DownloadedObject(
                object_=self.object._object,
                metadata=self.object.metadata,
                urls_by_name=self.object._urls_by_name,
                connector=self.object._connector,
                cache=self.object._cache,
            )
            self.assertFalse(
                all(
                    [
                        hasattr(DownloadedObject, method_name),
                        hasattr(client, method_name),
                    ]
                ),
                f"DownloadedObject.{method_name} should not be available if "
                f"{', '.join(no_import)} {'is' if len(no_import) == 1 else 'are'} not available",
            )

    @contextlib.contextmanager
    def _patch_downloading_table(self, table_info: TableInfo | str) -> Generator[pa.Table, None, None]:
        mock_table_info = table_info
        if isinstance(mock_table_info, str):
            mock_table_info = cast(TableInfo, self.object.search(mock_table_info))

        mock_data_id = mock_table_info["data"]
        expected_filename = self.cache.get_location(self.environment, _CACHE_SCOPE) / mock_data_id
        sample_table, payload_bytes = get_sample_table_and_bytes(
            KnownTableFormat.from_table_info(mock_table_info), mock_table_info["length"]
        )
        with mock.patch("evo.common.io.download.HTTPSource", autospec=True) as mock_source:

            async def _mock_download_file_side_effect(*args, **kwargs):
                expected_download_url = self.object._urls_by_name[mock_data_id]
                actual_download_url = await kwargs["url_generator"]()
                self.assertEqual(expected_filename, kwargs["filename"])
                self.assertEqual(expected_download_url, actual_download_url)
                self.assertIs(self.transport, kwargs["transport"])
                self.assertIs(NoFeedback, kwargs["fb"])
                expected_filename.write_bytes(payload_bytes)

            mock_source.download_file.side_effect = _mock_download_file_side_effect
            yield sample_table

        mock_source.download_file.assert_called_once()
        self.transport.assert_no_requests()

    @parameterized.expand(_TABLE_INFO_VARIANTS)
    async def test_download_table(self, _label: str, table_info: TableInfo | str) -> None:
        """Test downloading parquet data as a pyarrow.Table."""
        with self._patch_downloading_table(table_info) as sample_table:
            actual_table = await self.object.download_table(table_info)

            # Should use cache second time.
            # The _patch_downloading_table context manager verifies this by checking the data is only downloaded once.
            cached_table = await self.object.download_table(table_info)

        self.assertEqual(sample_table, actual_table)
        self.assertEqual(sample_table, cached_table)

    @contextlib.contextmanager
    def _patch_downloading_table_in_memory(self, table_info: TableInfo | str) -> Generator[pa.Table, None, None]:
        self.object._cache = None  # Disable the cache for this test.

        mock_table_info = table_info
        if isinstance(mock_table_info, str):
            mock_table_info = cast(TableInfo, self.object.search(mock_table_info))

        sample_table, payload_bytes = get_sample_table_and_bytes(
            KnownTableFormat.from_table_info(mock_table_info), mock_table_info["length"]
        )

        # Use the DownloadRequestHandler from evo.common.test_tools.io to mock the binary download.
        download_handler = DownloadRequestHandler(data=payload_bytes)
        self.transport.set_request_handler(download_handler)
        yield sample_table

    @parameterized.expand(_TABLE_INFO_VARIANTS)
    async def test_download_table_without_cache(self, _label: str, table_info: TableInfo | str) -> None:
        """Test downloading parquet data in memory as a pyarrow.Table."""
        with self._patch_downloading_table_in_memory(table_info) as sample_table:
            actual_table = await self.object.download_table(table_info)

        self.assertEqual(sample_table, actual_table)

    def test_download_table_is_optional(self) -> None:
        """Test that the download_table method is not available when pyarrow is not installed."""
        self._assert_optional_method("download_table", unload=["evo.objects.parquet.loader"], no_import=["pyarrow"])

    @parameterized.expand(_TABLE_INFO_VARIANTS)
    async def test_download_dataframe(self, _label: str, table_info: TableInfo | str) -> None:
        """Test downloading parquet data as a pandas.DataFrame."""
        with self._patch_downloading_table(table_info) as sample_table:
            actual_dataframe = await self.object.download_dataframe(table_info)

            # Should use cache second time.
            # The _patch_downloading_table context manager verifies this by checking the data is only downloaded once.
            cached_dataframe = await self.object.download_dataframe(table_info)

        expected_dataframe = sample_table.to_pandas()
        assert_frame_equal(expected_dataframe, actual_dataframe)
        assert_frame_equal(expected_dataframe, cached_dataframe)

    @parameterized.expand(_TABLE_INFO_VARIANTS)
    async def test_download_dataframe_without_cache(self, _label: str, table_info: TableInfo | str) -> None:
        """Test downloading parquet data in memory as a pandas.DataFrame."""
        with self._patch_downloading_table_in_memory(table_info) as sample_table:
            actual_dataframe = await self.object.download_dataframe(table_info)

        expected_dataframe = sample_table.to_pandas()
        assert_frame_equal(expected_dataframe, actual_dataframe)

    @parameterized.expand(
        [
            ("pyarrow",),
            ("pandas",),
        ]
    )
    def test_download_dataframe_is_optional(self, missing: str) -> None:
        """Test that the download_dataframe method is not available when pandas or pyarrow is not installed."""
        self._assert_optional_method("download_dataframe", unload=["evo.objects.parquet.loader"], no_import=[missing])

    @parameterized.expand(_TABLE_INFO_VARIANTS)
    async def test_download_array(self, _label: str, table_info: TableInfo | str) -> None:
        """Test downloading parquet data as a numpy.ndarray."""
        with self._patch_downloading_table(table_info) as sample_table:
            actual_array = await self.object.download_array(table_info)

            # Should use cache second time.
            # The _patch_downloading_table context manager verifies this by checking the data is only downloaded once.
            cached_array = await self.object.download_array(table_info)

        expected_array = sample_table.to_pandas().to_numpy()
        assert_array_equal(expected_array, actual_array, strict=True)
        assert_array_equal(expected_array, cached_array, strict=True)

    @parameterized.expand(_TABLE_INFO_VARIANTS)
    async def test_download_array_without_cache(self, _label: str, table_info: TableInfo | str) -> None:
        """Test downloading parquet data in memory as a numpy.ndarray."""
        with self._patch_downloading_table_in_memory(table_info) as sample_table:
            actual_array = await self.object.download_array(table_info)

        expected_array = sample_table.to_pandas().to_numpy()
        assert_array_equal(expected_array, actual_array, strict=True)

    @parameterized.expand(
        [
            ("pyarrow",),
            ("numpy",),
        ]
    )
    def test_download_array_is_optional(self, missing: str) -> None:
        """Test that the download_array method is not available when numpy or pyarrow is not installed."""
        self._assert_optional_method("download_array", unload=["evo.objects.parquet.loader"], no_import=[missing])
