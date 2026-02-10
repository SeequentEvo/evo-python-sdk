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

import copy
import json
from unittest import mock
from uuid import UUID

import pandas as pd
import pyarrow as pa
from pandas.testing import assert_frame_equal
from parameterized import parameterized

from data import load_test_data
from evo.common import IFeedback, RequestMethod
from evo.common.io.exceptions import DataExistsError
from evo.common.test_tools import TestWithConnector, TestWithStorage
from evo.common.utils import NoFeedback, PartialFeedback, get_header_metadata
from evo.objects.exceptions import TableFormatError
from evo.objects.utils import KnownTableFormat, ObjectDataClient, table_formats
from helpers import NoImport, UnloadModule, get_sample_table_and_bytes


class TestObjectDataClient(TestWithConnector, TestWithStorage):
    def setUp(self) -> None:
        TestWithConnector.setUp(self)
        TestWithStorage.setUp(self)
        self.data_client = ObjectDataClient(environment=self.environment, connector=self.connector, cache=self.cache)
        self.setup_universal_headers(get_header_metadata(ObjectDataClient.__module__))

    def tearDown(self) -> None:
        # Clear cache between tests to avoid cached files interfering with subsequent tests.
        self.cache.clear_cache()

    @property
    def base_path(self) -> str:
        return f"geoscience-object/orgs/{self.environment.org_id}/workspaces/{self.environment.workspace_id}"

    @parameterized.expand(
        [
            ("with_explicit_table_format", True),
            ("with_implicit_table_format", False),
        ]
    )
    def test_save_table(self, _name: str, pass_table_format: bool) -> None:
        """Test saving tabular data using pyarrow."""
        with (
            mock.patch("evo.objects.utils.table_formats.get_known_format") as mock_get_known_format,
            mock.patch("evo.common.io.upload.StorageDestination") as mock_destination,
        ):
            mock_table = mock.Mock()
            mock_get_known_format.return_value = mock_known_format = mock.Mock(spec=KnownTableFormat)
            mock_known_format.save_table.return_value = mock_table_info = {}

            if pass_table_format:
                actual_table_info = self.data_client.save_table(mock_table, table_format=mock_known_format)
            else:
                actual_table_info = self.data_client.save_table(mock_table)

        if pass_table_format:
            mock_get_known_format.assert_not_called()
        else:
            mock_get_known_format.assert_called_once_with(mock_table, table_formats=None)

        # Verify cast_table was NOT called(allow_casting=False by default)
        mock_known_format.cast_table.assert_not_called()

        mock_known_format.save_table.assert_called_once_with(
            table=mock_table, destination=self.data_client.cache_location
        )
        mock_destination.upload_file.assert_not_called()
        self.transport.assert_no_requests()
        self.assertIs(mock_table_info, actual_table_info)

    def save_table_with_table_format_list(self) -> None:
        """Test saving tabular data using pyarrow with specified table format."""
        with (
            mock.patch("evo.objects.utils.table_formats.get_known_format") as mock_get_known_format,
            mock.patch("evo.common.io.upload.StorageDestination") as mock_destination,
        ):
            sample_table, _ = get_sample_table_and_bytes(table_formats.FLOAT_ARRAY_2, 1)

            mock_known_format = mock.Mock(wraps=table_formats.FLOAT_ARRAY_2)
            mock_known_format.save_table.return_value = mock_table_info = {}
            mock_get_known_format.return_value = mock_known_format
            actual_table_info = self.data_client.save_table(
                sample_table, table_format=[table_formats.FLOAT_ARRAY_2, mock_known_format]
            )

            mock_get_known_format.assert_called_once_with(
                table_formats=[table_formats.FLOAT_ARRAY_2, mock_known_format]
            )

        mock_known_format.save_table.assert_called_once_with(
            table=sample_table, destination=self.data_client.cache_location
        )
        mock_destination.upload_file.assert_not_called()
        self.transport.assert_no_requests()
        self.assertIs(mock_table_info, actual_table_info)

    def test_save_table_with_allow_casting_true(self) -> None:
        """Test saving a table with allow_casting=True casts the table before saving."""
        with mock.patch("evo.common.io.upload.StorageDestination"):
            sample_table, _ = get_sample_table_and_bytes(table_formats.INTEGER_ARRAY_1_INT32, 1)
            schema = pa.schema([pa.field(sample_table.schema.names[0], pa.int64())])
            casted_table = sample_table.cast(schema)

            # To allow isinstance checks to work, replace the method on a copy of the known format.
            mock_known_format = copy.copy(table_formats.INTEGER_ARRAY_1_INT64)
            mock_known_format.save_table = mock.Mock(return_value={})

            self.data_client.save_table(sample_table, table_format=mock_known_format, allow_casting=True)

        # Verify save_table was called with the casted table
        mock_known_format.save_table.assert_called_once_with(
            table=casted_table, destination=self.data_client.cache_location
        )

    def test_save_table_with_allow_casting_without_single_format_raises(self) -> None:
        """Test that allow_casting=True raises ValueError when not providing a single table format."""
        mock_table = mock.Mock()

        # Test with no table format
        with self.assertRaises(ValueError) as ctx:
            self.data_client.save_table(mock_table, table_format=None, allow_casting=True)
        self.assertIn("allow_casting can only be used when a single table_format is provided", str(ctx.exception))

        # Test with multiple table formats
        with self.assertRaises(ValueError) as ctx:
            self.data_client.save_table(
                mock_table, table_format=[table_formats.FLOAT_ARRAY_2, table_formats.FLOAT_ARRAY_3], allow_casting=True
            )
        self.assertIn("allow_casting can only be used when a single table_format is provided", str(ctx.exception))

    @parameterized.expand(
        [
            ("with_explicit_table_format", True),
            ("with_implicit_table_format", False),
        ]
    )
    def test_save_dataframe(self, _name: str, pass_table_format: bool) -> None:
        """Test saving tabular data using pandas."""
        with (
            mock.patch("evo.objects.utils.table_formats.get_known_format") as mock_get_known_format,
            mock.patch("evo.common.io.upload.StorageDestination") as mock_destination,
            mock.patch("pyarrow.Table") as mock_pyarrow_table,
        ):
            mock_pyarrow_table.from_pandas.return_value = mock_table = mock.Mock()
            mock_get_known_format.return_value = mock_known_format = mock.Mock(spec=KnownTableFormat)
            mock_known_format.save_table.return_value = mock_table_info = {}

            mock_dataframe = mock.Mock()
            if pass_table_format:
                actual_table_info = self.data_client.save_dataframe(mock_dataframe, table_format=mock_known_format)
            else:
                actual_table_info = self.data_client.save_dataframe(mock_dataframe)

        if pass_table_format:
            mock_get_known_format.assert_not_called()
        else:
            mock_get_known_format.assert_called_once_with(mock_table, table_formats=None)
        mock_known_format.save_table.assert_called_once_with(
            table=mock_table, destination=self.data_client.cache_location
        )
        mock_destination.upload_file.assert_not_called()
        self.transport.assert_no_requests()
        self.assertIs(mock_table_info, actual_table_info)

    async def test_upload_referenced_data(self) -> None:
        put_data_response = load_test_data("put_data_batch.json")[:5]
        test_pointset = {
            "name": "Test Pointset",
            "uuid": None,
            "bounding_box": {"min_x": 0.0, "max_x": 0.0, "min_y": 0.0, "max_y": 0.0, "min_z": 0.0, "max_z": 0.0},
            "coordinate_reference_system": "unspecified",
            "locations": {
                "attributes": [
                    {
                        "name": f"Test Attribute {n}",
                        "nan_description": {"values": []},
                        "values": {
                            "data": data["name"],
                            "length": 0,
                            "width": 1,
                            "data_type": "float64",
                        },
                        "attribute_type": "scalar",
                    }
                    for n, data in enumerate(put_data_response[1:])
                ],
                "coordinates": {
                    "data": put_data_response[0]["name"],
                    "length": 0,
                    "width": 3,
                    "data_type": "float64",
                },
            },
            "schema": "/objects/pointset/1.1.0/pointset.schema.json",
        }

        data_by_name = {}
        for data in put_data_response:
            data_by_name[data["name"]] = data
            if data["exists"] is False:
                # Create an empty file to simulate the data being present in the cache.
                (self.data_client.cache_location / data["name"]).touch()

        with (
            self.transport.set_http_response(status_code=200, content=json.dumps(put_data_response)),
            mock.patch("evo.common.io.upload.StorageDestination", autospec=True) as mock_destination,
        ):

            async def _mock_upload_file_side_effect(*args, **kwargs):
                filename = kwargs["filename"]
                data = data_by_name.pop(filename.name)
                actual_upload_url = await kwargs["url_generator"]()
                self.assertEqual(data["upload_url"], actual_upload_url)
                self.assertIs(self.transport, kwargs["transport"])
                self.assertIsInstance(kwargs["fb"], PartialFeedback)

            mock_destination.upload_file.side_effect = _mock_upload_file_side_effect
            await self.data_client.upload_referenced_data(test_pointset, mock.Mock(spec=IFeedback))

        self.assert_request_made(
            method=RequestMethod.PUT,
            path=f"{self.base_path}/data",
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            body=[{"name": data["name"]} for data in put_data_response[1:]] + [{"name": put_data_response[0]["name"]}],
        )

    @parameterized.expand(
        [
            ("with_explicit_table_format_no_cast", True, False),
            ("with_implicit_table_format_no_cast", False, False),
            ("with_explicit_table_format", True, True),
        ]
    )
    async def test_upload_table(self, _name: str, pass_table_format: bool, cast: bool) -> None:
        """Test uploading tabular data using pyarrow or pandas."""
        put_data_response = load_test_data("put_data.json")
        with (
            self.transport.set_http_response(status_code=200, content=json.dumps(put_data_response)),
            mock.patch("evo.objects.utils.table_formats.get_known_format") as mock_get_known_format,
            mock.patch("evo.common.io.upload.StorageDestination", autospec=True) as mock_destination,
        ):
            mock_table = mock.Mock()
            mock_get_known_format.return_value = mock_known_format = mock.Mock(spec=KnownTableFormat)
            mock_casted_table = mock.Mock()
            mock_known_format.cast_table.return_value = mock_casted_table
            mock_known_format.save_table.return_value = mock_table_info = {}
            mock_table_info["data"] = mock_data_id = put_data_response[0]["name"]

            async def _mock_upload_file_side_effect(*args, **kwargs):
                expected_filename = self.data_client.cache_location / mock_data_id
                expected_upload_url = put_data_response[0].get("upload_url")
                actual_upload_url = await kwargs["url_generator"]()
                self.assertEqual(expected_filename, kwargs["filename"])
                self.assertEqual(expected_upload_url, actual_upload_url)
                self.assertIs(self.transport, kwargs["transport"])
                self.assertIs(NoFeedback, kwargs["fb"])

            mock_destination.upload_file.side_effect = _mock_upload_file_side_effect

            if pass_table_format:
                actual_table_info = await self.data_client.upload_table(
                    mock_table, table_format=mock_known_format, allow_casting=cast
                )
            else:
                actual_table_info = await self.data_client.upload_table(mock_table)

        if pass_table_format:
            mock_get_known_format.assert_not_called()
        else:
            mock_get_known_format.assert_called_once_with(mock_table, table_formats=None)
        mock_known_format.save_table.assert_called_once_with(
            table=mock_casted_table if cast else mock_table, destination=self.data_client.cache_location
        )
        mock_destination.upload_file.assert_called_once()
        self.assert_request_made(
            method=RequestMethod.PUT,
            path=f"{self.base_path}/data",
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            body=[{"name": mock_data_id}],
        )
        self.assertIs(mock_table_info, actual_table_info)

    @parameterized.expand(
        [
            ("with_explicit_table_format_no_cast", True, False),
            ("with_implicit_table_format_no_cast", False, False),
            ("with_explicit_table_format", True, True),
        ]
    )
    async def test_upload_dataframe(self, _name: str, pass_table_format: bool, cast: bool) -> None:
        """Test uploading tabular data using pyarrow or pandas."""
        put_data_response = load_test_data("put_data.json")
        with (
            self.transport.set_http_response(status_code=200, content=json.dumps(put_data_response)),
            mock.patch("evo.objects.utils.table_formats.get_known_format") as mock_get_known_format,
            mock.patch("evo.common.io.upload.StorageDestination", autospec=True) as mock_destination,
            mock.patch("pyarrow.Table") as mock_pyarrow_table,
        ):
            mock_pyarrow_table.from_pandas.return_value = mock_table = mock.Mock()
            mock_get_known_format.return_value = mock_known_format = mock.Mock(spec=KnownTableFormat)
            mock_casted_table = mock.Mock()
            mock_known_format.cast_table.return_value = mock_casted_table
            mock_known_format.save_table.return_value = mock_table_info = {}
            mock_table_info["data"] = mock_data_id = put_data_response[0]["name"]

            async def _mock_upload_file_side_effect(*args, **kwargs):
                expected_filename = self.data_client.cache_location / mock_data_id
                expected_upload_url = put_data_response[0].get("upload_url")
                actual_upload_url = await kwargs["url_generator"]()
                self.assertEqual(expected_filename, kwargs["filename"])
                self.assertEqual(expected_upload_url, actual_upload_url)
                self.assertIs(self.transport, kwargs["transport"])
                self.assertIs(NoFeedback, kwargs["fb"])

            mock_destination.upload_file.side_effect = _mock_upload_file_side_effect

            mock_dataframe = mock.Mock()
            if pass_table_format:
                actual_table_info = await self.data_client.upload_dataframe(
                    mock_dataframe, table_format=mock_known_format, allow_casting=cast
                )
            else:
                actual_table_info = await self.data_client.upload_dataframe(mock_dataframe)

        if pass_table_format:
            mock_get_known_format.assert_not_called()
        else:
            mock_get_known_format.assert_called_once_with(mock_table, table_formats=None)
        mock_known_format.save_table.assert_called_once_with(
            table=mock_casted_table if cast else mock_table, destination=self.data_client.cache_location
        )
        mock_destination.upload_file.assert_called_once()
        self.assert_request_made(
            method=RequestMethod.PUT,
            path=f"{self.base_path}/data",
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            body=[{"name": mock_data_id}],
        )
        self.assertIs(mock_table_info, actual_table_info)

    async def test_upload_table_exists(self) -> None:
        """Test uploading tabular data using pyarrow when the table exists."""
        put_data_response = load_test_data("put_data_exists.json")
        with (
            self.transport.set_http_response(status_code=200, content=json.dumps(put_data_response)),
            mock.patch("evo.objects.utils.table_formats.get_known_format") as mock_get_known_format,
            mock.patch("evo.common.io.upload.StorageDestination", autospec=True) as mock_destination,
        ):
            mock_table = mock.Mock()
            mock_get_known_format.return_value = mock_known_format = mock.Mock(spec=KnownTableFormat)
            mock_known_format.save_table.return_value = mock_table_info = {}
            mock_table_info["data"] = mock_data_id = put_data_response[0]["name"]

            async def _mock_upload_file_side_effect(*args, **kwargs):
                expected_filename = self.data_client.cache_location / mock_data_id
                self.assertEqual(expected_filename, kwargs["filename"])
                self.assertIs(self.transport, kwargs["transport"])
                self.assertIs(NoFeedback, kwargs["fb"])

                with self.assertRaises(DataExistsError) as cm:
                    await kwargs["url_generator"]()
                raise cm.exception  # upload_table() should catch this exception.

            mock_destination.upload_file.side_effect = _mock_upload_file_side_effect
            actual_table_info = await self.data_client.upload_table(mock_table)

        mock_get_known_format.assert_called_once_with(mock_table, table_formats=None)
        mock_known_format.save_table.assert_called_once_with(
            table=mock_table, destination=self.data_client.cache_location
        )
        mock_destination.upload_file.assert_called_once()
        self.assert_request_made(
            method=RequestMethod.PUT,
            path=f"{self.base_path}/data",
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            body=[{"name": mock_data_id}],
        )
        self.assertIs(mock_table_info, actual_table_info)

    async def test_upload_dataframe_exists(self) -> None:
        """Test uploading tabular data using pandas when the table exists."""
        put_data_response = load_test_data("put_data_exists.json")
        with (
            self.transport.set_http_response(status_code=200, content=json.dumps(put_data_response)),
            mock.patch("evo.objects.utils.table_formats.get_known_format") as mock_get_known_format,
            mock.patch("evo.common.io.upload.StorageDestination", autospec=True) as mock_destination,
            mock.patch("pyarrow.Table") as mock_pyarrow_table,
        ):
            mock_pyarrow_table.from_pandas.return_value = mock_table = mock.Mock()
            mock_get_known_format.return_value = mock_known_format = mock.Mock(spec=KnownTableFormat)
            mock_known_format.save_table.return_value = mock_table_info = {}
            mock_table_info["data"] = mock_data_id = put_data_response[0]["name"]

            async def _mock_upload_file_side_effect(*args, **kwargs):
                expected_filename = self.data_client.cache_location / mock_data_id
                self.assertEqual(expected_filename, kwargs["filename"])
                self.assertIs(self.transport, kwargs["transport"])
                self.assertIs(NoFeedback, kwargs["fb"])

                with self.assertRaises(DataExistsError) as cm:
                    await kwargs["url_generator"]()
                raise cm.exception  # upload_table() should catch this exception.

            mock_destination.upload_file.side_effect = _mock_upload_file_side_effect

            mock_dataframe = mock.Mock()
            actual_table_info = await self.data_client.upload_dataframe(mock_dataframe)

        mock_get_known_format.assert_called_once_with(mock_table, table_formats=None)
        mock_known_format.save_table.assert_called_once_with(
            table=mock_table, destination=self.data_client.cache_location
        )
        mock_destination.upload_file.assert_called_once()
        self.assert_request_made(
            method=RequestMethod.PUT,
            path=f"{self.base_path}/data",
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            body=[{"name": mock_data_id}],
        )
        self.assertIs(mock_table_info, actual_table_info)

    async def test_upload_table_with_allow_casting_without_single_format_raises(self) -> None:
        """Test that allow_casting=True raises ValueError when not providing a single table format."""
        mock_table = mock.Mock()

        # Test with no table format
        with self.assertRaises(ValueError) as ctx:
            await self.data_client.upload_table(mock_table, table_format=None, allow_casting=True)
        self.assertIn("allow_casting can only be used when a single table_format is provided", str(ctx.exception))

        # Test with multiple table formats
        with self.assertRaises(ValueError) as ctx:
            await self.data_client.upload_table(
                mock_table, table_format=[table_formats.FLOAT_ARRAY_2, table_formats.FLOAT_ARRAY_3], allow_casting=True
            )
        self.assertIn("allow_casting can only be used when a single table_format is provided", str(ctx.exception))

    async def test_upload_dataframe_with_allow_casting_without_single_format_raises(self) -> None:
        """Test that allow_casting=True raises ValueError when not providing a single table format."""
        with mock.patch("pyarrow.Table") as mock_pyarrow_table:
            mock_dataframe = mock.Mock()
            mock_pyarrow_table.from_pandas.return_value = mock.Mock()

            # Test with no table format
            with self.assertRaises(ValueError) as ctx:
                await self.data_client.upload_dataframe(mock_dataframe, table_format=None, allow_casting=True)
            self.assertIn("allow_casting can only be used when a single table_format is provided", str(ctx.exception))

            # Test with multiple table formats
            with self.assertRaises(ValueError) as ctx:
                await self.data_client.upload_dataframe(
                    mock_dataframe,
                    table_format=[table_formats.FLOAT_ARRAY_2, table_formats.FLOAT_ARRAY_3],
                    allow_casting=True,
                )
            self.assertIn("allow_casting can only be used when a single table_format is provided", str(ctx.exception))

    @parameterized.expand(
        [
            (
                "dictionary",
                pa.table({"Category": pa.array(["A", "B", "A", "C"], type=pa.dictionary(pa.int32(), pa.string()))}),
                pa.table(
                    {"key": pa.array([0, 1, 2], type=pa.int32()), "value": pa.array(["A", "B", "C"], type=pa.string())}
                ),
                pa.table({"Category": pa.array([0, 1, 0, 2], type=pa.int32())}),
                table_formats.INTEGER_ARRAY_1_INT32,
            ),
            (
                "string",
                pa.table({"Category": pa.array(["A", "B", "A", "C"], type=pa.string())}),
                pa.table(
                    {"key": pa.array([0, 1, 2], type=pa.int32()), "value": pa.array(["A", "B", "C"], type=pa.string())}
                ),
                pa.table({"Category": pa.array([0, 1, 0, 2], type=pa.int32())}),
                table_formats.INTEGER_ARRAY_1_INT32,
            ),
            (
                "multiple_columns",
                pa.table(
                    {
                        "Category": pa.array(["A", "B", "A", "C"], type=pa.string()),
                        "Other": pa.array(["B", "C", "B", "D"], type=pa.string()),
                    }
                ),
                pa.table(
                    {
                        "key": pa.array([0, 1, 2, 3], type=pa.int32()),
                        "value": pa.array(["A", "B", "C", "D"], type=pa.string()),
                    }
                ),
                pa.table(
                    {
                        "Category": pa.array([0, 1, 0, 2], type=pa.int32()),
                        "Other": pa.array([1, 2, 1, 3], type=pa.int32()),
                    }
                ),
                table_formats.INTEGER_ARRAY_MD_INT32,
            ),
            (
                "multiple_columns_chunked",
                pa.table(
                    {
                        "Category": pa.chunked_array(
                            [["A", "B"], ["A", "C"]], type=pa.dictionary(pa.int8(), pa.string())
                        ),
                        "Other": pa.array(["B", "C", "B", "D"], type=pa.string()),
                        "Extra": pa.chunked_array([["A"], ["B", "C"], ["A"]], type=pa.string()),
                    }
                ),
                pa.table(
                    {
                        "key": pa.array([0, 1, 2, 3], type=pa.int32()),
                        "value": pa.array(["A", "B", "C", "D"], type=pa.string()),
                    }
                ),
                pa.table(
                    {
                        "Category": pa.array([0, 1, 0, 2], type=pa.int32()),
                        "Other": pa.array([1, 2, 1, 3], type=pa.int32()),
                        "Extra": pa.array([0, 1, 2, 0], type=pa.int32()),
                    }
                ),
                table_formats.INTEGER_ARRAY_MD_INT32,
            ),
        ]
    )
    async def test_upload_category_table(
        self,
        _name: str,
        table: pa.Table,
        lookup_table: pa.Table,
        values_table: pa.Table,
        values_table_format: KnownTableFormat,
    ) -> None:
        """Test uploading categorical data using pandas."""
        with (
            mock.patch("evo.objects.utils.ObjectDataClient.upload_table") as mock_upload_table,
        ):

            def side_effect(table_to_upload, table_format=None, fb=None):
                return {"table": table_to_upload, "table_format": table_format}

            mock_upload_table.side_effect = side_effect
            category_info = await self.data_client.upload_category_table(table)

        self.assertEqual(mock_upload_table.call_count, 2)
        self.assertEqual(category_info["table"]["table"], lookup_table)
        self.assertEqual(category_info["table"]["table_format"], table_formats.LOOKUP_TABLE_INT32)
        self.assertEqual(category_info["values"]["table"], values_table)
        self.assertEqual(category_info["values"]["table_format"], values_table_format)

    async def test_upload_category_table_error(self) -> None:
        """Test uploading categorical data raises error for unsupported types."""
        table = pa.table({"Category": pa.array([1.0, 2.0, 1.0, 3.0], type=pa.float64())})
        with self.assertRaises(TableFormatError) as cm:
            await self.data_client.upload_category_table(table)
        assert "Category columns must be of type string" in str(cm.exception)

    @parameterized.expand(
        [
            (
                "category",
                pd.DataFrame({"Category": pd.Categorical(["A", "B", "A", "C"])}),
                pa.table(
                    {"key": pa.array([0, 1, 2], type=pa.int32()), "value": pa.array(["A", "B", "C"], type=pa.string())}
                ),
                pa.table({"Category": pa.array([0, 1, 0, 2], type=pa.int32())}),
                table_formats.INTEGER_ARRAY_1_INT32,
            ),
            (
                "string",
                pd.DataFrame({"Category": ["A", "B", "A", "C"]}),
                pa.table(
                    {"key": pa.array([0, 1, 2], type=pa.int32()), "value": pa.array(["A", "B", "C"], type=pa.string())}
                ),
                pa.table({"Category": pa.array([0, 1, 0, 2], type=pa.int32())}),
                table_formats.INTEGER_ARRAY_1_INT32,
            ),
            (
                "multiple_columns",
                pd.DataFrame({"Category": ["A", "B", "A", "C"], "Other": ["B", "C", "B", "D"]}),
                pa.table(
                    {
                        "key": pa.array([0, 1, 2, 3], type=pa.int32()),
                        "value": pa.array(["A", "B", "C", "D"], type=pa.string()),
                    }
                ),
                pa.table(
                    {
                        "Category": pa.array([0, 1, 0, 2], type=pa.int32()),
                        "Other": pa.array([1, 2, 1, 3], type=pa.int32()),
                    }
                ),
                table_formats.INTEGER_ARRAY_MD_INT32,
            ),
        ]
    )
    async def test_upload_category_dataframe(
        self,
        _name: str,
        dataframe: pd.DataFrame,
        lookup_table: pa.Table,
        values_table: pa.Table,
        values_table_format: KnownTableFormat,
    ) -> None:
        """Test uploading categorical data using pandas."""
        with (
            mock.patch("evo.objects.utils.ObjectDataClient.upload_table") as mock_upload_table,
        ):

            def side_effect(table, table_format=None, fb=None):
                return {"table": table, "table_format": table_format}

            mock_upload_table.side_effect = side_effect
            category_info = await self.data_client.upload_category_dataframe(dataframe)

        self.assertEqual(mock_upload_table.call_count, 2)
        self.assertEqual(category_info["table"]["table"], lookup_table)
        self.assertEqual(category_info["table"]["table_format"], table_formats.LOOKUP_TABLE_INT32)
        self.assertEqual(category_info["values"]["table"], values_table)
        self.assertEqual(category_info["values"]["table_format"], values_table_format)

    async def test_download_table(self) -> None:
        """Test downloading tabular data using pyarrow."""
        get_object_response = load_test_data("get_object.json")
        object_id = UUID(int=2)
        with (
            self.transport.set_http_response(status_code=200, content=json.dumps(get_object_response)),
            mock.patch("evo.common.io.download.HTTPSource", autospec=True) as mock_source,
        ):
            mock_table_info = {
                "data": "0000000000000000000000000000000000000000000000000000000000000001",
                "length": 1,
                "width": 3,
                "data_type": "float64",
            }
            mock_data_id: str = mock_table_info["data"]
            expected_filename = self.data_client.cache_location / mock_data_id
            sample_table, payload_bytes = get_sample_table_and_bytes(
                KnownTableFormat.from_table_info(mock_table_info), 1
            )

            async def _mock_download_file_side_effect(*args, **kwargs):
                expected_download_url = get_object_response["links"]["data"][1]["download_url"]
                actual_download_url = await kwargs["url_generator"]()
                self.assertEqual(expected_filename, kwargs["filename"])
                self.assertEqual(expected_download_url, actual_download_url)
                self.assertIs(self.transport, kwargs["transport"])
                self.assertIs(NoFeedback, kwargs["fb"])
                expected_filename.write_bytes(payload_bytes)

            mock_source.download_file.side_effect = _mock_download_file_side_effect
            actual_table = await self.data_client.download_table(object_id, None, mock_table_info)

        mock_source.download_file.assert_called_once()
        self.assert_request_made(
            method=RequestMethod.GET,
            path=f"{self.base_path}/objects/{object_id}",
            headers={"Accept": "application/json", "Accept-Encoding": "gzip"},
        )
        self.assertEqual(sample_table, actual_table)

    async def test_download_dataframe(self) -> None:
        """Test downloading tabular data using pandas."""
        get_object_response = load_test_data("get_object.json")
        object_id = UUID(int=2)
        with (
            self.transport.set_http_response(status_code=200, content=json.dumps(get_object_response)),
            mock.patch("evo.common.io.download.HTTPSource", autospec=True) as mock_source,
        ):
            mock_table_info = {
                "data": "0000000000000000000000000000000000000000000000000000000000000001",
                "length": 1,
                "width": 3,
                "data_type": "float64",
            }
            mock_data_id: str = mock_table_info["data"]
            expected_filename = self.data_client.cache_location / mock_data_id
            sample_table, payload_bytes = get_sample_table_and_bytes(
                KnownTableFormat.from_table_info(mock_table_info), 1
            )

            async def _mock_download_file_side_effect(*args, **kwargs):
                expected_download_url = get_object_response["links"]["data"][1]["download_url"]
                actual_download_url = await kwargs["url_generator"]()
                self.assertEqual(expected_filename, kwargs["filename"])
                self.assertEqual(expected_download_url, actual_download_url)
                self.assertIs(self.transport, kwargs["transport"])
                self.assertIs(NoFeedback, kwargs["fb"])
                expected_filename.write_bytes(payload_bytes)

            mock_source.download_file.side_effect = _mock_download_file_side_effect
            actual_dataframe = await self.data_client.download_dataframe(object_id, None, mock_table_info)

        mock_source.download_file.assert_called_once()
        self.assert_request_made(
            method=RequestMethod.GET,
            path=f"{self.base_path}/objects/{object_id}",
            headers={"Accept": "application/json", "Accept-Encoding": "gzip"},
        )
        assert_frame_equal(sample_table.to_pandas(), actual_dataframe)

    async def test_download_dataframe_optional(self) -> None:
        """Test download dataframe is not available if pandas is not installed."""
        with UnloadModule("evo.objects.utils.data"), NoImport("pandas"):
            from evo.objects.utils.data import ObjectDataClient

            client = ObjectDataClient(environment=self.environment, connector=self.connector, cache=self.cache)
            self.assertFalse(
                any(
                    (
                        hasattr(ObjectDataClient, "download_dataframe"),
                        hasattr(client, "download_dataframe"),
                    )
                ),
                "download_dataframe should not be available if pandas is missing",
            )

    async def test_download_table_already_downloaded(self) -> None:
        """Test downloading tabular data using pyarrow or pandas when the table is already downloaded."""
        get_object_response = load_test_data("get_object.json")
        object_id = UUID(int=2)
        with (
            self.transport.set_http_response(status_code=200, content=json.dumps(get_object_response)),
            mock.patch("evo.common.io.download.HTTPSource", autospec=True) as mock_source,
        ):
            mock_table_info = {
                "data": "0000000000000000000000000000000000000000000000000000000000000001",
                "length": 1,
                "width": 3,
                "data_type": "float64",
            }
            mock_data_id: str = mock_table_info["data"]
            expected_filename = self.data_client.cache_location / mock_data_id
            sample_table, payload_bytes = get_sample_table_and_bytes(
                KnownTableFormat.from_table_info(mock_table_info), 1
            )

            async def _mock_download_file_side_effect(*args, **kwargs):
                expected_download_url = get_object_response["links"]["data"][1]["download_url"]
                actual_download_url = await kwargs["url_generator"]()
                self.assertEqual(expected_filename, kwargs["filename"])
                self.assertEqual(expected_download_url, actual_download_url)
                self.assertIs(self.transport, kwargs["transport"])
                self.assertIs(NoFeedback, kwargs["fb"])

            mock_source.download_file.side_effect = _mock_download_file_side_effect

            expected_filename.write_bytes(payload_bytes)
            actual_table = await self.data_client.download_table(object_id, None, mock_table_info)

        mock_source.download_file.assert_not_called()
        # the object metadata is still requested to get the initial download URL and check permissions.
        self.assert_request_made(
            method=RequestMethod.GET,
            path=f"{self.base_path}/objects/{object_id}",
            headers={"Accept": "application/json", "Accept-Encoding": "gzip"},
        )
        self.transport.request.assert_called_once()  # Ensure no other requests were made.
        self.assertEqual(sample_table, actual_table)

    async def test_download_dataframe_already_downloaded(self) -> None:
        """Test downloading tabular data using pandas when the table is already downloaded."""
        get_object_response = load_test_data("get_object.json")
        object_id = UUID(int=2)
        with (
            self.transport.set_http_response(status_code=200, content=json.dumps(get_object_response)),
            mock.patch("evo.common.io.download.HTTPSource", autospec=True) as mock_source,
        ):
            mock_table_info = {
                "data": "0000000000000000000000000000000000000000000000000000000000000001",
                "length": 1,
                "width": 3,
                "data_type": "float64",
            }
            mock_data_id: str = mock_table_info["data"]
            expected_filename = self.data_client.cache_location / mock_data_id
            sample_table, payload_bytes = get_sample_table_and_bytes(
                KnownTableFormat.from_table_info(mock_table_info), 1
            )

            async def _mock_download_file_side_effect(*args, **kwargs):
                expected_download_url = get_object_response["links"]["data"][1]["download_url"]
                actual_download_url = await kwargs["url_generator"]()
                self.assertEqual(expected_filename, kwargs["filename"])
                self.assertEqual(expected_download_url, actual_download_url)
                self.assertIs(self.transport, kwargs["transport"])
                self.assertIs(NoFeedback, kwargs["fb"])

            mock_source.download_file.side_effect = _mock_download_file_side_effect

            expected_filename.write_bytes(payload_bytes)
            actual_dataframe = await self.data_client.download_dataframe(object_id, None, mock_table_info)

        mock_source.download_file.assert_not_called()
        # the object metadata is still requested to get the initial download URL and check permissions.
        self.assert_request_made(
            method=RequestMethod.GET,
            path=f"{self.base_path}/objects/{object_id}",
            headers={"Accept": "application/json", "Accept-Encoding": "gzip"},
        )
        self.transport.request.assert_called_once()  # Ensure no other requests were made.
        assert_frame_equal(sample_table.to_pandas(), actual_dataframe)
