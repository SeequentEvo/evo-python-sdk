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
import copy
import json
from collections.abc import Generator
from typing import Any, cast
from unittest import mock
from urllib.parse import quote
from uuid import UUID

import numpy as np
import pandas as pd
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
    MultiDownloadRequestHandler,
    TestWithConnector,
    TestWithStorage,
)
from evo.common.utils import NoFeedback, get_header_metadata
from evo.jmespath import JMESPathObjectProxy
from evo.objects import DownloadedObject, ObjectReference
from evo.objects.endpoints import models
from evo.objects.exceptions import ObjectModifiedError
from evo.objects.io import _CACHE_SCOPE
from evo.objects.parquet import TableInfo
from evo.objects.utils import KnownTableFormat
from helpers import NoImport, UnloadModule, assign_property, get_sample_table_and_bytes, write_table_to_bytes

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


_category_dtype = pd.CategoricalDtype(categories=["NULL", "A", "B", "C"], ordered=False)


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
        self.setup_universal_headers(get_header_metadata(DownloadedObject.__module__))

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

    @parameterized.expand(
        [
            ("pass UUID", True, False),
            ("omit UUID", False, False),
            ("pass UUID with conflict check", True, True),
        ]
    )
    async def test_update(self, _label: str, pass_uuid: bool, check_for_conflict: bool) -> None:
        """Test updating a geoscience object succeeds."""
        post_object_response = load_test_data("get_object.json")
        post_object_response["version_id"] = "2"

        updated_pointset = post_object_response["object"]

        updated_pointset_parameter = copy.deepcopy(updated_pointset)
        if not pass_uuid:
            del updated_pointset_parameter["uuid"]

        with self.transport.set_http_response(status_code=201, content=json.dumps(post_object_response)):
            new_object = await self.object.update(updated_pointset_parameter, check_for_conflict=check_for_conflict)

        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if check_for_conflict:
            headers["If-Match"] = self.object.metadata.version_id
        self.assert_request_made(
            method=RequestMethod.POST,
            path=f"{_OBJECTS_URL}/{self.object.metadata.id}",
            headers=headers,
            body=updated_pointset,
        )
        # Check metadata.
        actual_metadata = new_object.metadata
        self.assertEqual(self.object.metadata.id, actual_metadata.id)
        self.assertEqual("2", actual_metadata.version_id)

        self.maxDiff = None

        # Check geoscience object.
        actual_pointset = new_object.as_dict()
        # Convert UUID in actual_pointset to a string, to make it consistent with updated_pointset
        actual_pointset["uuid"] = str(actual_pointset["uuid"])
        self.assertEqual(updated_pointset, actual_pointset)

    async def test_update_wrong_uuid(self):
        """Test updating a geoscience object fails when the object ID in the new object does not match the current object ID."""
        updated_pointset = load_test_data("get_object.json")["object"]
        updated_pointset["uuid"] = "00000000-0000-0000-0000-000000000003"
        with self.assertRaises(ValueError, msg="The object ID in the new object does not match the current object ID"):
            await self.object.update(updated_pointset)

    async def test_update_with_conflict(self):
        """Test updating a geoscience object fails when there is a new version on the server."""
        updated_pointset = load_test_data("get_object.json")["object"]
        response = load_test_data("object_modified_error.json")

        with self.transport.set_http_response(status_code=412, content=json.dumps(response)):
            with self.assertRaises(ObjectModifiedError):
                await self.object.update(updated_pointset, check_for_conflict=True)

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

    def _set_property(self, expression: str, value: Any) -> None:
        obj = self.object.as_dict()
        assign_property(obj, expression, value)
        self.object._object = models.GeoscienceObject.model_validate(obj)

    def _setup_table(self, expression: str, table: pa.Table):
        self.object._cache = None  # Disable the cache for this test.

        # Change the number of rows in the object to match the data
        self._set_property(f"{expression}.length", table.num_rows)
        url = self.object._urls_by_name[self.object.search(f"{expression}.data")]
        payload_bytes = write_table_to_bytes(table)
        download_handler = DownloadRequestHandler(data=payload_bytes)
        self.transport.set_request_handler(download_handler)
        return url, payload_bytes

    def _setup_category(self, attribute: str):
        table = pa.Table.from_pydict(
            {
                "key": pa.array([0, 1, 2, 100], type=pa.int32()),
                "category": pa.array(["NULL", "A", "B", "C"], type=pa.string()),
            }
        )
        values = pa.Table.from_pydict(
            {
                "value": pa.array([0, 1, 2, 1, 100, 101, None], type=pa.int32()),
            }
        )

        table_url, table_bytes = self._setup_table(f"{attribute}.table", table)
        values_url, values_bytes = self._setup_table(f"{attribute}.values", values)
        self.transport.set_request_handler(
            MultiDownloadRequestHandler(
                {
                    table_url: table_bytes,
                    values_url: values_bytes,
                }
            )
        )

    async def test_download_table_nan_values(self):
        values_path = "locations.attributes[1].values"
        values = pa.Table.from_pydict(
            {
                "value": pa.array([0.2, None, 3.4, 5.3, 2.0, None], type=pa.float64()),
            }
        )
        self._setup_table(values_path, values)

        actual_table = await self.object.download_table(values_path, nan_values=[3.4, 2.0])

        expected_table = pa.Table.from_pydict(
            {
                "value": pa.array([0.2, None, None, 5.3, None, None], type=pa.float64()),
            }
        )
        self.assertEqual(actual_table, expected_table)

    async def test_download_table_column_names(self):
        with self._patch_downloading_table_in_memory("locations.coordinates") as sample_table:
            actual_table = await self.object.download_table("locations.coordinates", column_names=["XC", "YC", "ZC"])

        expected_table = sample_table.rename_columns(["XC", "YC", "ZC"])
        self.assertEqual(expected_table, actual_table)

    async def test_download_attribute_table(self):
        with self._patch_downloading_table_in_memory("locations.attributes[1].values") as sample_table:
            actual_table = await self.object.download_attribute_table("locations.attributes[1]")

        expected_table = sample_table.rename_columns(["InvRes"])
        self.assertEqual(expected_table, actual_table)

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

    async def test_download_dataframe_nan_values(self):
        values_path = "locations.attributes[1].values"
        values = pa.Table.from_pydict(
            {
                "value": pa.array([0.2, None, 3.4, 5.3, 2.0, None], type=pa.float64()),
            }
        )
        self._setup_table(values_path, values)

        actual_dataframe = await self.object.download_dataframe(values_path, nan_values=[0.2, 0.1, 2.0])

        expected_dataframe = pd.DataFrame({"value": [np.nan, np.nan, 3.4, 5.3, np.nan, np.nan]})
        assert_frame_equal(expected_dataframe, actual_dataframe)

    async def test_download_dataframe_column_names(self):
        with self._patch_downloading_table_in_memory("locations.coordinates") as sample_table:
            actual_dataframe = await self.object.download_dataframe(
                "locations.coordinates", column_names=["XC", "YC", "ZC"]
            )

        expected_dataframe = sample_table.to_pandas()
        expected_dataframe.columns = ["XC", "YC", "ZC"]
        assert_frame_equal(expected_dataframe, actual_dataframe)

    async def test_download_attribute_dataframe(self):
        with self._patch_downloading_table_in_memory("locations.attributes[1].values") as sample_table:
            actual_dataframe = await self.object.download_attribute_dataframe("locations.attributes[1]")

        expected_dataframe = sample_table.to_pandas()
        expected_dataframe.columns = ["InvRes"]
        assert_frame_equal(expected_dataframe, actual_dataframe)

    @parameterized.expand(
        [
            (
                "no nan values",
                None,
                None,
                pd.DataFrame({"value": ["NULL", "A", "B", "A", "C", pd.NA, pd.NA]}, dtype=_category_dtype),
            ),
            (
                "list of nan values",
                [0, 2],
                None,
                pd.DataFrame({"value": [pd.NA, "A", pd.NA, "A", "C", pd.NA, pd.NA]}, dtype=_category_dtype),
            ),
            (
                "JMESPath nan description",
                "locations.attributes[0].nan_description",
                None,
                pd.DataFrame({"value": [pd.NA, "A", "B", "A", "C", pd.NA, pd.NA]}, dtype=_category_dtype),
            ),
            (
                "JMESPath nan description predicate",
                "locations.attributes[?name == 'Stn'].nan_description",
                None,
                pd.DataFrame({"value": [pd.NA, "A", "B", "A", "C", pd.NA, pd.NA]}, dtype=_category_dtype),
            ),
            (
                "JMESPath nan value list",
                "locations.attributes[0].nan_description.values",
                None,
                pd.DataFrame({"value": [pd.NA, "A", "B", "A", "C", pd.NA, pd.NA]}, dtype=_category_dtype),
            ),
            (
                "JMESPath nan value list predicate",
                "locations.attributes[?name == 'Stn'].nan_description.values",
                None,
                pd.DataFrame({"value": [pd.NA, "A", "B", "A", "C", pd.NA, pd.NA]}, dtype=_category_dtype),
            ),
            (
                "JMESPath attribute info",
                [0, 2],
                "locations.attributes[0]",
                pd.DataFrame({"value": [pd.NA, "A", pd.NA, "A", "C", pd.NA, pd.NA]}, dtype=_category_dtype),
            ),
            (
                "JMESPath attribute info predicate",
                [0, 2],
                "locations.attributes[?name == 'Stn']",
                pd.DataFrame({"value": [pd.NA, "A", pd.NA, "A", "C", pd.NA, pd.NA]}, dtype=_category_dtype),
            ),
        ]
    )
    async def test_download_category(
        self, _label: str, nan_values: str | list[int] | None, attribute_info: str | None, expected: pd.DataFrame
    ) -> None:
        attribute_path = "locations.attributes[0]"
        self._setup_category(attribute_path)

        # None, means pass it as a dictionary from the object.
        if attribute_info is None:
            attribute_info_parameter = self.object.search(attribute_path).raw
        else:
            attribute_info_parameter = attribute_info
        actual_dataframe = await self.object.download_category_dataframe(
            attribute_info_parameter, nan_values=nan_values
        )
        assert_frame_equal(expected, actual_dataframe)
        actual_table = await self.object.download_category_table(attribute_info_parameter, nan_values=nan_values)

        # Ensure the indices are int32, to be consistent with the parquet data.
        expected_table = pa.Table.from_pandas(
            expected, schema=pa.schema({"value": pa.dictionary(pa.int32(), pa.string())})
        )
        self.assertEqual(expected_table, actual_table)

        # Test loading the table through download_attribute_dataframe as well.
        if isinstance(nan_values, list):
            self._set_property(f"{attribute_path}.nan_description.values", nan_values)
        elif nan_values is None:
            self._set_property(f"{attribute_path}.nan_description.values", [])

        # None, means pass it as a dictionary from the object.
        if attribute_info is None:
            attribute_info_parameter = self.object.search(attribute_path).raw
        else:
            attribute_info_parameter = attribute_info

        expected.columns = ["Stn"]
        expected_table = expected_table.rename_columns(["Stn"])

        attribute_dataframe = await self.object.download_attribute_dataframe(attribute_info_parameter)
        assert_frame_equal(expected, attribute_dataframe)
        actual_table = await self.object.download_attribute_table(attribute_info_parameter)
        self.assertEqual(expected_table, actual_table)

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
