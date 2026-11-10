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

import asyncio
import json
import logging
import sys
import threading
from contextlib import nullcontext
from dataclasses import replace
from unittest import mock
from uuid import UUID, uuid4

import pandas as pd
import pyarrow as pa
from pandas.testing import assert_frame_equal
from parameterized import parameterized

from data import load_test_data
from evo.common import IFeedback, RequestMethod
from evo.common.exceptions import BadRequestException, EvoAPIException, RetryError, TransportError
from evo.common.io.exceptions import ChunkedIOError, DataExistsError
from evo.common.test_tools import TestWithConnector, TestWithStorage
from evo.common.utils import BackoffLinear, NoFeedback, PartialFeedback, Retry, get_header_metadata
from evo.objects.endpoints.models import DataUploadResponseBody
from evo.objects.exceptions import TableFormatError
from evo.objects.utils import KnownTableFormat, ObjectDataClient, table_formats, upload_concurrency
from evo.objects.utils._concurrency import gather_uploads
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
            mock.patch("evo.objects.utils.data._get_schema_from_dataframe", return_value=None),
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
        [("referenced", ConnectionError), ("table", RuntimeError), ("table_transient", ConnectionError)]
    )
    async def test_upload_failure_preserves_exception_and_identifies_data(
        self, upload_type: str, error_type: type[Exception]
    ) -> None:
        response = load_test_data("put_data_batch.json")[:1]
        name = response[0]["name"]
        (self.data_client.cache_location / name).touch()
        error = error_type("Storage unavailable")
        with (
            self.transport.set_http_response(status_code=200, content=json.dumps(response)),
            mock.patch("evo.common.io.upload.StorageDestination", autospec=True) as destination,
            mock.patch("evo.objects.utils.data.logger") as logger,
            self.assertRaises(error_type) as cm,
        ):
            destination.upload_file.side_effect = error
            if upload_type == "referenced":
                await self.data_client.upload_referenced_data({"coordinates": {"data": name}})
            else:
                table, _ = get_sample_table_and_bytes(table_formats.FLOAT_ARRAY_2, 1)
                with mock.patch.object(self.data_client, "save_table", return_value={"data": name}):
                    await self.data_client.upload_table(
                        table, retry=Retry(logger=logger, backoff_method=BackoffLinear(0))
                    )
        if upload_type == "table_transient":
            self.assertIsInstance(cm.exception.__context__, RetryError)
            self.assertEqual((error, error, error), cm.exception.__context__.exceptions)
        self.assertIs(error, cm.exception)
        context = (
            "\n".join(getattr(cm.exception, "__notes__", []))
            if sys.version_info >= (3, 11)
            else "\n".join(call.args[0] for call in logger.error.call_args_list)
        )
        self.assertIn("While uploading", context)
        self.assertIn(name, context)

    async def test_upload_url_failure_reports_upload_stage(self) -> None:
        with (
            self.transport.set_http_response(status_code=400, content='{"title": "Bad request"}'),
            mock.patch("evo.objects.utils.data.logger") as logger,
            self.assertRaises(BadRequestException) as cm,
        ):
            await self.data_client.upload_referenced_data({"coordinates": {"data": "0" * 64}})
        context = (
            "\n".join(getattr(cm.exception, "__notes__", []))
            if sys.version_info >= (3, 11)
            else logger.error.call_args.args[0]
        )
        self.assertIn("While uploading referenced data", context)

    def test_incompatible_table_is_reported_as_preparation_not_upload(self) -> None:
        with (
            mock.patch("evo.objects.utils.data.logger") as logger,
            self.assertRaises(TableFormatError) as cm,
        ):
            self.data_client.save_table(pa.table({"value": ["text"]}), table_format=table_formats.FLOAT_ARRAY_2)
        context = (
            "\n".join(getattr(cm.exception, "__notes__", []))
            if sys.version_info >= (3, 11)
            else logger.error.call_args.args[0]
        )
        self.assertIn("While preparing an Arrow table for Evo Parquet storage", context)
        self.assertNotIn("While uploading", context)
        self.transport.assert_no_requests()

    @parameterized.expand(
        [
            ("with_explicit_table_format", True),
            ("with_implicit_table_format", False),
        ]
    )
    async def test_upload_table(self, _name: str, pass_table_format: bool) -> None:
        """Test uploading tabular data using pyarrow or pandas."""
        put_data_response = load_test_data("put_data.json")
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
                expected_upload_url = put_data_response[0].get("upload_url")
                actual_upload_url = await kwargs["url_generator"]()
                self.assertEqual(expected_filename, kwargs["filename"])
                self.assertEqual(expected_upload_url, actual_upload_url)
                self.assertIs(self.transport, kwargs["transport"])
                self.assertIs(NoFeedback, kwargs["fb"])

            mock_destination.upload_file.side_effect = _mock_upload_file_side_effect

            if pass_table_format:
                actual_table_info = await self.data_client.upload_table(mock_table, table_format=mock_known_format)
            else:
                actual_table_info = await self.data_client.upload_table(mock_table)

        if pass_table_format:
            mock_get_known_format.assert_not_called()
        else:
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

    @parameterized.expand(
        [
            ("with_explicit_table_format", True),
            ("with_implicit_table_format", False),
        ]
    )
    async def test_upload_dataframe(self, _name: str, pass_table_format: bool) -> None:
        """Test uploading tabular data using pyarrow or pandas."""
        put_data_response = load_test_data("put_data.json")
        with (
            self.transport.set_http_response(status_code=200, content=json.dumps(put_data_response)),
            mock.patch("evo.objects.utils.table_formats.get_known_format") as mock_get_known_format,
            mock.patch("evo.common.io.upload.StorageDestination", autospec=True) as mock_destination,
            mock.patch("pyarrow.Table") as mock_pyarrow_table,
            mock.patch("evo.objects.utils.data._get_schema_from_dataframe", return_value=None),
        ):
            mock_pyarrow_table.from_pandas.return_value = mock_table = mock.Mock()
            mock_get_known_format.return_value = mock_known_format = mock.Mock(spec=KnownTableFormat)
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
                    mock_dataframe, table_format=mock_known_format
                )
            else:
                actual_table_info = await self.data_client.upload_dataframe(mock_dataframe)

        if pass_table_format:
            mock_get_known_format.assert_not_called()
        else:
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

    @parameterized.expand([("direct", False), ("nested_group", True)])
    async def test_cancelled_upload_waits_for_parquet_writer(self, _name: str, nested_group: bool) -> None:
        started = asyncio.Event()
        release = threading.Event()
        finished = threading.Event()
        loop = asyncio.get_running_loop()

        def save_table(*_args):
            loop.call_soon_threadsafe(started.set)
            try:
                if not release.wait(timeout=5):
                    raise TimeoutError("The test did not release the parquet writer")
                return {"data": "saved-table"}
            finally:
                finished.set()

        with (
            mock.patch.object(self.data_client, "save_table", side_effect=save_table),
            mock.patch("evo.objects.io.ObjectDataUpload.upload_from_cache") as upload,
        ):

            async def run():
                operation = self.data_client.upload_table(pa.table({"value": [1.0]}))
                if nested_group:
                    return await gather_uploads(operation)
                return await operation

            task = asyncio.create_task(run())
            cancelled = False
            try:
                await asyncio.wait_for(started.wait(), timeout=2)
                task.cancel()
                cancelled = True
                await asyncio.sleep(0)
                self.assertFalse(task.done(), "Cancellation returned while the parquet writer was still running")
            finally:
                release.set()
                if not cancelled:
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)

        self.assertTrue(task.cancelled())
        self.assertTrue(finished.is_set())
        upload.assert_not_awaited()

    async def test_parquet_serialization_runs_off_the_event_loop(self) -> None:
        threads = []
        main_thread = threading.get_ident()

        def save_table(*_args):
            threads.append(threading.get_ident())
            return {"data": "saved-table"}

        with (
            mock.patch.object(self.data_client, "save_table", side_effect=save_table),
            mock.patch("evo.objects.io.ObjectDataUpload.upload_from_cache") as upload,
        ):
            result = await self.data_client.upload_table(pa.table({"value": [1.0]}))

        self.assertEqual({"data": "saved-table"}, result)
        self.assertEqual(1, len(threads))
        self.assertNotEqual(main_thread, threads[0])
        upload.assert_awaited_once()

    async def test_parquet_writer_failure_is_propagated_without_uploading(self) -> None:
        with (
            mock.patch.object(self.data_client, "save_table", side_effect=RuntimeError("parquet write failed")),
            mock.patch("evo.objects.io.ObjectDataUpload.upload_from_cache") as upload,
        ):
            with self.assertRaisesRegex(RuntimeError, "parquet write failed"):
                await self.data_client.upload_table(pa.table({"value": [1.0]}))

        upload.assert_not_awaited()

    @parameterized.expand(
        [
            ("default", None, False),
            ("serial", 1, False),
            ("lower", 3, False),
            ("higher", 12, False),
            ("multiple_clients_default", None, True),
            ("multiple_clients_scope", 3, True),
        ]
    )
    async def test_direct_concurrent_uploads_share_loop_limit(self, _name, configured_limit, multiple_clients) -> None:
        limit = configured_limit if configured_limit is not None else 8
        clients = [
            ObjectDataClient(environment=self.environment, connector=self.connector, cache=self.cache)
            if multiple_clients
            else self.data_client
            for _ in range(24)
        ]
        release = asyncio.Event()
        limit_reached = asyncio.Event()
        active = 0
        peak = 0

        async def upload(**_kwargs):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            if active == limit:
                limit_reached.set()
            try:
                await release.wait()
            finally:
                active -= 1

        with (
            mock.patch.object(ObjectDataClient, "save_table", side_effect=lambda table, _: {"data": table}) as save,
            mock.patch("evo.objects.utils.data.asyncio.to_thread", side_effect=lambda fn, *args: fn(*args)),
            mock.patch("evo.objects.io.ObjectDataUpload.upload_from_cache", side_effect=upload),
            upload_concurrency(configured_limit) if configured_limit is not None else nullcontext(),
        ):
            task = asyncio.gather(*(client.upload_table(str(i)) for i, client in enumerate(clients)))
            try:
                await asyncio.wait_for(limit_reached.wait(), timeout=2)
                await asyncio.sleep(0)
                self.assertEqual(limit, active)
                release.set()
                await asyncio.wait_for(task, timeout=2)
            finally:
                release.set()
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        self.assertEqual(limit, peak)
        self.assertEqual(0, active)
        self.assertEqual(24, save.call_count)

    async def test_serialization_and_upload_attempts_share_the_same_limit(self):
        active = 0
        peak = 0

        async def work(function, *args):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            try:
                await asyncio.sleep(0)
                return function(*args)
            finally:
                active -= 1

        async def upload(**_kwargs):
            await work(lambda: None)

        with (
            mock.patch.object(self.data_client, "save_table", side_effect=lambda table, _: {"data": table}),
            mock.patch("evo.objects.utils.data.asyncio.to_thread", side_effect=work) as serialize,
            mock.patch("evo.objects.io.ObjectDataUpload.upload_from_cache", side_effect=upload) as transfer,
            upload_concurrency(3),
        ):
            await asyncio.gather(*(self.data_client.upload_table(str(i)) for i in range(24)))
        self.assertEqual(3, peak)
        self.assertEqual(0, active)
        self.assertEqual(24, serialize.await_count)
        self.assertEqual(24, transfer.await_count)

    @parameterized.expand([("default", None), ("scope", 1)])
    def test_client_can_be_reused_across_contended_event_loops(self, _name, limit):
        async def upload(**_kwargs):
            await asyncio.sleep(0)

        async def run():
            return await asyncio.gather(*(self.data_client.upload_table(str(i)) for i in range(16)))

        with (
            mock.patch.object(self.data_client, "save_table", side_effect=lambda table, _: {"data": table}),
            mock.patch("evo.objects.io.ObjectDataUpload.upload_from_cache", side_effect=upload),
            upload_concurrency(limit) if limit is not None else nullcontext(),
        ):
            for _ in range(2):
                result = asyncio.run(run())
                self.assertEqual([str(i) for i in range(16)], [info["data"] for info in result])

    @parameterized.expand([("default", None), ("scope", 3)])
    async def test_concurrent_category_operations_share_limit(self, _name, configured_limit):
        limit = configured_limit if configured_limit is not None else 8
        release = asyncio.Event()
        limit_reached = asyncio.Event()
        active = 0
        peak = 0

        async def upload(**_kwargs):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            if active == limit:
                limit_reached.set()
            try:
                await release.wait()
            finally:
                active -= 1

        with (
            mock.patch.object(self.data_client, "save_table", side_effect=lambda *_: {"data": str(uuid4())}),
            mock.patch("evo.objects.io.ObjectDataUpload.upload_from_cache", side_effect=upload),
            upload_concurrency(configured_limit) if configured_limit is not None else nullcontext(),
        ):
            task = asyncio.gather(
                *(self.data_client.upload_category_table(pa.table({"value": [str(i)]})) for i in range(10))
            )
            try:
                await asyncio.wait_for(limit_reached.wait(), timeout=2)
                await asyncio.sleep(0)
                release.set()
                await asyncio.wait_for(task, timeout=2)
            finally:
                release.set()
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        self.assertEqual(limit, peak)
        self.assertEqual(0, active)

    @parameterized.expand(
        [(str(status), EvoAPIException(status, "transient", None, None)) for status in (408, 429, 500, 502, 503, 504)]
        + [
            ("timeout", TimeoutError()),
            ("transport", TransportError("connection lost")),
            ("connection", ConnectionError()),
        ]
    )
    async def test_transient_upload_retry_preserves_order_and_cached_tables(self, _name, error):
        attempts = {}
        completed = []
        sibling_finished = asyncio.Event()

        async def upload(upload_context, **_kwargs):
            name = upload_context.name
            attempts[name] = attempts.get(name, 0) + 1
            if name == "first" and attempts[name] <= 2:
                await sibling_finished.wait()
                raise error
            completed.append(name)
            sibling_finished.set()

        retry = Retry(logger=mock.Mock(), backoff_method=BackoffLinear(0))
        with (
            mock.patch.object(self.data_client, "save_table", side_effect=lambda table, _: {"data": table}) as save,
            mock.patch("evo.objects.io.ObjectDataUpload.upload_from_cache", new=upload),
        ):
            results = await asyncio.wait_for(
                gather_uploads(
                    self.data_client.upload_table("first", retry=retry),
                    self.data_client.upload_table("second", retry=retry),
                ),
                timeout=2,
            )
        self.assertEqual(["first", "second"], [result["data"] for result in results])
        self.assertEqual(["second", "first"], completed)
        self.assertEqual({"first": 3, "second": 1}, attempts)
        self.assertEqual(2, save.call_count)

    async def test_retry_covers_actual_upload_url_acquisition(self):
        response = load_test_data("put_data.json")
        with (
            self.transport.set_http_response(status_code=201, content=""),
            mock.patch.object(self.data_client, "save_table", wraps=self.data_client.save_table) as save,
            mock.patch(
                "evo.objects.io.DataApi.put_data_in_workspace",
                side_effect=[TimeoutError(), [DataUploadResponseBody(**response[0])]],
            ) as request_url,
        ):
            result = await self.data_client.upload_table(
                pa.table({"value": [1.0]}),
                retry=Retry(logger=mock.Mock(), backoff_method=BackoffLinear(0)),
            )
        self.assertTrue((self.data_client.cache_location / result["data"]).is_file())
        self.assertEqual(2, request_url.await_count)
        save.assert_called_once()

    @parameterized.expand([("chunk", "comp=block&"), ("commit", "comp=blocklist")])
    async def test_retry_covers_wrapped_storage_transport_failure(self, _name, query):
        response = load_test_data("put_data.json")
        original_request = self.transport.request
        failed = False

        async def request(**kwargs):
            nonlocal failed
            if query in kwargs["url"] and not failed:
                failed = True
                raise TransportError("connection lost")
            return await original_request(**kwargs)

        with (
            self.transport.set_http_response(status_code=201, content=""),
            mock.patch.object(self.transport, "request", side_effect=request),
            mock.patch.object(self.data_client, "save_table", wraps=self.data_client.save_table) as save,
            mock.patch(
                "evo.objects.io.DataApi.put_data_in_workspace",
                return_value=[DataUploadResponseBody(**response[0])],
            ) as request_url,
        ):
            result = await self.data_client.upload_table(
                pa.table({"value": [1.0]}),
                retry=Retry(logger=mock.Mock(), backoff_method=BackoffLinear(0)),
            )
        self.assertTrue(failed)
        self.assertTrue((self.data_client.cache_location / result["data"]).is_file())
        self.assertEqual(2, request_url.await_count)
        save.assert_called_once()

    @parameterized.expand(
        [
            ("url", EvoAPIException),
            ("storage", ChunkedIOError),
            ("connection", ConnectionError),
            ("transport", TransportError),
        ]
    )
    async def test_retry_exhaustion_is_bounded_and_preserves_errors(self, _name, error_type):
        errors = []
        for i in range(3):
            cause = EvoAPIException(503, f"unavailable-{i}", None, None)
            error = cause if error_type is EvoAPIException else error_type(f"unavailable-{i}")
            if error_type is ChunkedIOError:
                error.__cause__ = cause
            errors.append(error)
        original_cause = errors[-1].__cause__
        with (
            mock.patch.object(self.data_client, "save_table", return_value={"data": "saved"}) as save,
            mock.patch("evo.objects.io.ObjectDataUpload.upload_from_cache", side_effect=errors) as upload,
        ):
            with self.assertRaises(error_type) as caught:
                await self.data_client.upload_table(
                    mock.Mock(), retry=Retry(logger=mock.Mock(), backoff_method=BackoffLinear(0))
                )
        self.assertIs(errors[-1], caught.exception)
        self.assertIs(original_cause, caught.exception.__cause__)
        self.assertIsInstance(caught.exception.__context__, RetryError)
        self.assertEqual(tuple(errors), caught.exception.__context__.exceptions)
        self.assertEqual(3, upload.await_count)
        save.assert_called_once()

    async def test_recovered_upload_logs_warning_not_error(self):
        with (
            mock.patch.object(self.data_client, "save_table", return_value={"data": "saved"}),
            mock.patch(
                "evo.objects.io.ObjectDataUpload.upload_from_cache", side_effect=[ConnectionError("retry"), None]
            ),
            self.assertLogs("evo.object.data", level=logging.WARNING) as logs,
        ):
            await self.data_client.upload_table(
                mock.Mock(), retry=Retry(logger=logging.getLogger("evo.object.data"), backoff_method=BackoffLinear(0))
            )
        self.assertEqual([logging.WARNING], [record.levelno for record in logs.records])

    async def test_other_uploads_proceed_during_backoff(self):
        backing_off = asyncio.Event()
        release = asyncio.Event()
        attempts = {}
        sleep = asyncio.sleep

        async def backoff(delay):
            if delay == 1:
                backing_off.set()
                await release.wait()
            else:
                await sleep(delay)

        async def upload(upload, **_kwargs):
            attempts[upload.name] = attempts.get(upload.name, 0) + 1
            if upload.name == "first" and attempts[upload.name] == 1:
                raise ConnectionError("retry")

        with (
            mock.patch.object(self.data_client, "save_table", side_effect=lambda table, _: {"data": table}),
            mock.patch("evo.objects.io.ObjectDataUpload.upload_from_cache", new=upload),
            mock.patch("evo.common.utils.retry.asyncio.sleep", side_effect=backoff),
            upload_concurrency(1),
        ):
            task = asyncio.create_task(
                self.data_client.upload_table("first", retry=Retry(logger=mock.Mock(), backoff_method=BackoffLinear(1)))
            )
            try:
                await asyncio.wait_for(backing_off.wait(), timeout=2)
                self.assertEqual({"data": "second"}, await asyncio.wait_for(self.data_client.upload_table("second"), 2))
                self.assertFalse(task.done())
                release.set()
                await asyncio.wait_for(task, timeout=2)
            finally:
                release.set()
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        self.assertEqual({"first": 2, "second": 1}, attempts)

    @parameterized.expand([("success", False), ("failure", True)])
    async def test_identical_tables_share_inflight_upload_across_clients(self, _name, fail):
        started = asyncio.Event()
        joined = asyncio.Event()
        release = asyncio.Event()
        error = RuntimeError("upload failed")
        shield = asyncio.shield
        feedback = mock.Mock(spec=IFeedback)
        other_client = ObjectDataClient(environment=self.environment, connector=self.connector, cache=self.cache)
        table = pa.table({"value": [1.0]})

        def watch_join(awaitable):
            if not isinstance(awaitable, asyncio.Task):
                joined.set()
            return shield(awaitable)

        async def upload(**_kwargs):
            started.set()
            await release.wait()
            if fail:
                raise error

        with (
            mock.patch("evo.objects.io.ObjectDataUpload.upload_from_cache", side_effect=upload) as upload_mock,
            mock.patch("evo.objects.utils.data.asyncio.shield", side_effect=watch_join),
        ):
            owner = asyncio.create_task(self.data_client.upload_table(table))
            tasks = [owner]
            try:
                await asyncio.wait_for(started.wait(), timeout=2)
                tasks.append(asyncio.create_task(other_client.upload_table(table, fb=feedback)))
                await asyncio.wait_for(joined.wait(), timeout=2)
                upload_mock.assert_awaited_once()
                self.assertFalse(tasks[1].done())
                release.set()
                results = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=2)
                if fail:
                    self.assertTrue(all(result is error for result in results))
                    feedback.progress.assert_not_called()
                else:
                    self.assertEqual(results[0], results[1])
                    self.assertIsNot(results[0], results[1])
                    feedback.progress.assert_called_with(1)

                # Completion (including failure) must remove the in-flight entry.
                upload_mock.side_effect = None
                await self.data_client.upload_table(table)
                self.assertEqual(2, upload_mock.await_count)
            finally:
                release.set()
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)

    @parameterized.expand([("waiter", False), ("owner", True)])
    async def test_duplicate_upload_cancellation_cleans_up_without_orphaned_work(self, _name, cancel_owner):
        started = asyncio.Event()
        joined = asyncio.Event()
        release = asyncio.Event()
        shield = asyncio.shield

        def watch_join(awaitable):
            if not isinstance(awaitable, asyncio.Task):
                joined.set()
            return shield(awaitable)

        async def upload(**_kwargs):
            started.set()
            await release.wait()

        with (
            mock.patch.object(self.data_client, "save_table", return_value={"data": "saved"}),
            mock.patch("evo.objects.io.ObjectDataUpload.upload_from_cache", side_effect=upload) as upload_mock,
            mock.patch("evo.objects.utils.data.asyncio.shield", side_effect=watch_join),
        ):
            owner = asyncio.create_task(self.data_client.upload_table(mock.Mock()))
            tasks = [owner]
            try:
                await asyncio.wait_for(started.wait(), timeout=2)
                waiter = asyncio.create_task(self.data_client.upload_table(mock.Mock()))
                tasks.append(waiter)
                await asyncio.wait_for(joined.wait(), timeout=2)
                cancelled = owner if cancel_owner else waiter
                cancelled.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await cancelled
                if cancel_owner:
                    with self.assertRaises(asyncio.CancelledError):
                        await asyncio.wait_for(waiter, timeout=2)
                else:
                    self.assertFalse(owner.done())
                    release.set()
                    self.assertEqual({"data": "saved"}, await asyncio.wait_for(owner, timeout=2))
                upload_mock.assert_awaited_once()
                release.set()
                await self.data_client.upload_table(mock.Mock())
                self.assertEqual(2, upload_mock.await_count)
            finally:
                release.set()
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)

    @parameterized.expand([("hub_url",), ("org_id",), ("workspace_id",), ("connector",)])
    async def test_inflight_uploads_do_not_share_across_destinations_or_connectors(self, field):
        environment = self.environment
        connector = self.connector
        if field == "connector":
            connector = mock.Mock(wraps=self.connector)
        else:
            environment = replace(environment, **{field: "https://other.example" if field == "hub_url" else uuid4()})
        other_client = ObjectDataClient(environment=environment, connector=connector, cache=self.cache)
        started = asyncio.Event()
        both_started = asyncio.Event()
        release = asyncio.Event()
        count = 0

        async def upload(**_kwargs):
            nonlocal count
            count += 1
            started.set()
            if count == 2:
                both_started.set()
            await release.wait()

        with (
            mock.patch.object(ObjectDataClient, "save_table", return_value={"data": "saved"}),
            mock.patch("evo.objects.io.ObjectDataUpload.upload_from_cache", side_effect=upload),
        ):
            tasks = [asyncio.create_task(self.data_client.upload_table(mock.Mock()))]
            try:
                await asyncio.wait_for(started.wait(), timeout=2)
                tasks.append(asyncio.create_task(other_client.upload_table(mock.Mock())))
                await asyncio.wait_for(both_started.wait(), timeout=2)
                release.set()
                await asyncio.wait_for(asyncio.gather(*tasks), timeout=2)
            finally:
                release.set()
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
        self.assertEqual(2, count)

    @parameterized.expand(
        [
            ("bad_request", EvoAPIException(400, "bad request", None, None)),
            ("unauthorized", EvoAPIException(401, "unauthorized", None, None)),
            ("forbidden", EvoAPIException(403, "forbidden", None, None)),
            ("not_found", EvoAPIException(404, "not found", None, None)),
            ("unclassified_chunk_error", ChunkedIOError("unknown cause")),
            ("invalid_data", ValueError("invalid")),
            ("cancelled", asyncio.CancelledError()),
        ]
    )
    async def test_permanent_errors_and_cancellation_are_not_retried(self, _name, error):
        with (
            mock.patch.object(self.data_client, "save_table", return_value={"data": "saved"}),
            mock.patch("evo.objects.io.ObjectDataUpload.upload_from_cache", side_effect=error) as upload,
        ):
            with self.assertRaises(type(error)) as caught:
                await self.data_client.upload_table(mock.Mock())
        self.assertIs(error, caught.exception)
        upload.assert_awaited_once()

    async def test_already_existing_data_after_retry_is_success(self):
        feedback = mock.Mock(spec=IFeedback)
        with (
            mock.patch.object(self.data_client, "save_table", return_value={"data": "saved"}) as save,
            mock.patch(
                "evo.objects.io.ObjectDataUpload.upload_from_cache", side_effect=[TimeoutError(), DataExistsError()]
            ) as upload,
        ):
            result = await self.data_client.upload_table(
                mock.Mock(), fb=feedback, retry=Retry(logger=mock.Mock(), backoff_method=BackoffLinear(0))
            )
        self.assertEqual({"data": "saved"}, result)
        self.assertEqual(2, upload.await_count)
        save.assert_called_once()
        feedback.progress.assert_called_with(1)

    async def test_category_values_and_lookup_upload_concurrently(self) -> None:
        started = []
        all_started = asyncio.Event()
        release = asyncio.Event()

        async def upload(table, **_kwargs):
            started.append(table.column_names)
            if len(started) == 2:
                all_started.set()
            await release.wait()
            return {"data": table.column_names[0], "length": table.num_rows}

        with mock.patch.object(self.data_client, "upload_table", side_effect=upload):
            task = asyncio.create_task(self.data_client.upload_category_table(pa.table({"category": ["sand", "clay"]})))
            try:
                await asyncio.wait_for(all_started.wait(), timeout=2)
                release.set()
                result = await asyncio.wait_for(task, timeout=2)
            finally:
                release.set()
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

        self.assertEqual("category", result["values"]["data"])
        self.assertEqual("key", result["table"]["data"])

    async def test_upload_dataframe_exists(self) -> None:
        """Test uploading tabular data using pandas when the table exists."""
        put_data_response = load_test_data("put_data_exists.json")
        with (
            self.transport.set_http_response(status_code=200, content=json.dumps(put_data_response)),
            mock.patch("evo.objects.utils.table_formats.get_known_format") as mock_get_known_format,
            mock.patch("evo.common.io.upload.StorageDestination", autospec=True) as mock_destination,
            mock.patch("pyarrow.Table") as mock_pyarrow_table,
            mock.patch("evo.objects.utils.data._get_schema_from_dataframe", return_value=None),
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

    async def test_upload_dataframe_converts_large_string_via_schema(self) -> None:
        """Test that upload_dataframe converts large_string columns (from pandas string[pyarrow]) to string via schema."""
        dataframe = pd.DataFrame({"col": pd.Series(["A", "B", "C"], dtype="string[pyarrow]")})
        with mock.patch.object(self.data_client, "upload_table", new_callable=mock.AsyncMock) as mock_upload:
            mock_upload.return_value = {}
            await self.data_client.upload_dataframe(dataframe)

        (table,), _ = mock_upload.await_args
        self.assertTrue(pa.types.is_string(table.schema.field("col").type))

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

    async def test_download_tables_prepares_data_once(self) -> None:
        """Test that downloading multiple tables prepares all data with one metadata request."""
        object_id = UUID(int=2)
        data_id_1 = "0000000000000000000000000000000000000000000000000000000000000001"
        data_id_2 = "0000000000000000000000000000000000000000000000000000000000000002"
        table_info_1 = {"data": data_id_1, "length": 1, "width": 1, "data_type": "float64"}
        table_info_2 = {"data": data_id_2, "length": 1, "width": 1, "data_type": "float64"}
        download_1 = mock.Mock()
        download_1.name = data_id_1
        download_2 = mock.Mock()
        download_2.name = data_id_2
        table_1 = pa.table({"value": [1.0]})
        table_2 = pa.table({"value": [2.0]})

        async def download_side_effect(_download, table_info, _fb):
            return table_1 if table_info is table_info_1 else table_2

        with (
            mock.patch.object(self.data_client, "_prepare_data_downloads", new_callable=mock.AsyncMock) as prepare,
            mock.patch.object(
                self.data_client, "_download_prepared_table", new_callable=mock.AsyncMock
            ) as download_table,
        ):
            prepare.return_value = {data_id_1: download_1, data_id_2: download_2}
            download_table.side_effect = download_side_effect
            actual_tables = await self.data_client.download_tables(
                object_id, None, [table_info_1, table_info_1, table_info_2]
            )

        prepare.assert_awaited_once_with(object_id, None, [data_id_1, data_id_2])
        self.assertEqual({data_id_1: table_1, data_id_2: table_2}, actual_tables)
        self.assertEqual(2, download_table.await_count)

    async def test_download_tables_with_no_table_infos(self) -> None:
        """Test that downloading no tables does not prepare any downloads."""
        with mock.patch.object(self.data_client, "_prepare_data_downloads", new_callable=mock.AsyncMock) as prepare:
            actual_tables = await self.data_client.download_tables(UUID(int=2), None, [])

        prepare.assert_not_awaited()
        self.assertEqual({}, actual_tables)

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
