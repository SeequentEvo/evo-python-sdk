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

"""Tests for base model class and related functionality."""

from __future__ import annotations

import asyncio
import contextlib
from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, ClassVar
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, Mock, patch

import pandas as pd
from parameterized import parameterized

from evo.common import Environment, IFeedback, StaticContext
from evo.common.test_tools import BASE_URL, ORG, WORKSPACE_ID, TestWithConnector
from evo.objects.typed._data import DataTable, DataTableAndAttributes
from evo.objects.typed._model import ModelContext, SchemaBuilder, SchemaList, SchemaLocation, SchemaModel
from evo.objects.utils.table_formats import FLOAT_ARRAY_3, KnownTableFormat

from .helpers import MockClient


class TestSchemaConstants(TestWithConnector):
    """Tests for ClassVar constant auto-detection in build_from_data."""

    def setUp(self) -> None:
        TestWithConnector.setUp(self)
        self.environment = Environment(hub_url=BASE_URL, org_id=ORG.id, workspace_id=WORKSPACE_ID)
        self.context = StaticContext.from_environment(
            environment=self.environment,
            connector=self.connector,
        )

    async def test_constant_not_overridden_when_data_missing(self):
        """Test that constant persists when the data object doesn't have the attribute."""

        class ConstantModel(SchemaModel):
            table_format: ClassVar[str] = "float_array_3"
            version: ClassVar[Annotated[str, SchemaLocation("format_version")]] = "1.0.0"
            name: Annotated[str, SchemaLocation("name")]

        @dataclass
        class FakeData:
            version: str = "2.0.0"
            name: str = "test"

        builder = SchemaBuilder(ConstantModel, self.context)
        result = await builder.build_from_data(FakeData())

        self.assertEqual(result["format_version"], "1.0.0")
        self.assertEqual(result["name"], "test")

    def test_union_schema_list_requires_polymorphic_overrides(self):
        class First(SchemaModel):
            pass

        class Second(SchemaModel):
            pass

        with self.assertRaisesRegex(TypeError, r"must override _resolve_item_type\(\) and _data_to_schema\(\)"):

            class InvalidUnionList(SchemaList[First | Second]):
                pass

    async def test_schema_list_uploads_items_concurrently_and_preserves_order(self):
        class Models(SchemaList[SchemaModel]):
            pass

        started = set()
        all_started = asyncio.Event()
        release = asyncio.Event()
        second_finished = asyncio.Event()

        async def convert(value, context):
            started.add(value)
            if len(started) == 2:
                all_started.set()
            await release.wait()
            if value == 0:
                await second_finished.wait()
            else:
                second_finished.set()
            return {"value": value}

        with patch.object(SchemaModel, "_data_to_schema", side_effect=convert):
            task = asyncio.create_task(Models._data_to_schema([0, 1], self.context))
            try:
                await asyncio.wait_for(all_started.wait(), timeout=2)
                release.set()
                result = await asyncio.wait_for(task, timeout=2)
            finally:
                release.set()
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

        self.assertEqual([{"value": 0}, {"value": 1}], result)

    async def test_schema_list_accepts_missing_and_empty_data(self):
        for data in (None, [], ()):
            with self.subTest(data=data):
                self.assertEqual([], await SchemaList._data_to_schema(data, self.context))

    async def test_schema_list_rejects_non_sequences(self):
        with self.assertRaisesRegex(TypeError, "Expected a sequence"):
            await SchemaList._data_to_schema(42, self.context)


class CoordinateTable(DataTable):
    table_format: ClassVar[KnownTableFormat] = FLOAT_ARRAY_3
    data_columns: ClassVar[list[str]] = ["x", "y", "z"]


class ExtendedLocations(DataTableAndAttributes):
    _table: Annotated[CoordinateTable, SchemaLocation("coordinates")]
    point_count: Annotated[int, SchemaLocation("point_count")]


class TestDataTableAndAttributesMutation(IsolatedAsyncioTestCase):
    def setUp(self):
        self.document = {
            "coordinates": {"data": "old-coordinates", "length": 1},
            "attributes": [
                {
                    "name": "first",
                    "key": "stable",
                    "attribute_type": "scalar",
                    "values": {"data": "old-first", "length": 1},
                }
            ],
        }
        self.context = ModelContext(obj=Mock(), root_model=None, data_modified={"previous-change"})
        self.model = ExtendedLocations(self.context, self.document)
        self.frame = pd.DataFrame({"x": [1.0], "y": [2.0], "z": [3.0], "first": [4.0]})

    @parameterized.expand([("success", None), ("table_fails", "x"), ("attribute_fails", "first")])
    async def test_concurrent_mutations_commit_only_after_all_uploads_succeed(self, _name, failing_column):
        started = set()
        all_started = asyncio.Event()
        successful_finished = asyncio.Event()
        original = deepcopy(self.document)
        original_table = self.model._table
        original_attributes = self.model.attributes
        feedback = Mock(spec=IFeedback)

        async def upload(frame, *, fb, **_kwargs):
            column = frame.columns[0]
            started.add(column)
            if len(started) == 2:
                all_started.set()
            await all_started.wait()
            if column == failing_column:
                await successful_finished.wait()
                raise RuntimeError("upload failed")
            fb.progress(1)
            successful_finished.set()
            return {"data": f"new-{column}", "length": 1}

        client = Mock(upload_dataframe=AsyncMock(side_effect=upload))
        with (
            patch("evo.objects.typed.attributes.get_data_client", return_value=client),
            patch("evo.objects.typed._data.get_data_client", return_value=client) as get_client,
        ):
            if failing_column:
                with self.assertRaisesRegex(RuntimeError, "upload failed"):
                    await asyncio.wait_for(self.model.from_dataframe(self.frame, fb=feedback), timeout=2)
            else:
                await asyncio.wait_for(self.model.from_dataframe(self.frame, fb=feedback), timeout=2)

        if failing_column:
            self.assertEqual(original, self.document)
            self.assertEqual({"previous-change"}, self.context.data_modified)
        else:
            self.assertEqual("new-x", self.document["coordinates"]["data"])
            self.assertEqual("new-first", self.document["attributes"][0]["values"]["data"])
            self.assertEqual("stable", self.document["attributes"][0]["key"])
            self.assertEqual({"previous-change", "new-x", "new-first"}, self.context.data_modified)
            self.assertEqual([0.5, 1], [call.args[0] for call in feedback.progress.call_args_list])
        self.assertIs(original_table, self.model._table)
        self.assertIs(original_attributes, self.model.attributes)
        get_client.assert_called_once_with(self.context.obj)

    async def test_cancellation_keeps_table_and_attributes_unchanged(self):
        original = deepcopy(self.document)
        started = asyncio.Event()
        release = asyncio.Event()

        async def upload(frame, **_kwargs):
            if frame.columns[0] == "first":
                started.set()
                await release.wait()
            return {"data": "new", "length": 1}

        client = Mock(upload_dataframe=AsyncMock(side_effect=upload))
        with (
            patch("evo.objects.typed.attributes.get_data_client", return_value=client),
            patch("evo.objects.typed._data.get_data_client", return_value=client),
        ):
            task = asyncio.create_task(self.model.from_dataframe(self.frame))
            try:
                await asyncio.wait_for(started.wait(), timeout=2)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
            finally:
                release.set()
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        self.assertEqual(original, self.document)
        self.assertEqual({"previous-change"}, self.context.data_modified)

    async def test_invalid_attribute_metadata_prevents_table_upload(self):
        self.frame.attrs["attribute_descriptions"] = {"first": "invalid"}
        client = Mock(upload_dataframe=AsyncMock())
        with (
            patch("evo.objects.typed.attributes.get_data_client", return_value=client),
            patch("evo.objects.typed._data.get_data_client", return_value=client),
        ):
            with self.assertRaisesRegex(TypeError, "attribute description"):
                await self.model.from_dataframe(self.frame)
        client.upload_dataframe.assert_not_awaited()

    @parameterized.expand([("success", False), ("failure", True)])
    async def test_table_only_update_clears_attributes_only_on_success(self, _name, fail):
        original = deepcopy(self.document)
        upload = (
            AsyncMock(side_effect=RuntimeError("failed"))
            if fail
            else AsyncMock(return_value={"data": "new-table", "length": 1})
        )
        with patch("evo.objects.typed._data.get_data_client", return_value=Mock(upload_dataframe=upload)) as get_client:
            if fail:
                with self.assertRaisesRegex(RuntimeError, "failed"):
                    await self.model.from_dataframe(self.frame[["x", "y", "z"]])
                self.assertEqual(original, self.document)
            else:
                await self.model.from_dataframe(self.frame[["x", "y", "z"]])
                self.assertEqual([], self.document["attributes"])
                self.assertEqual("new-table", self.document["coordinates"]["data"])
        get_client.assert_called_once_with(self.context.obj)


class TestDataTableAndAttributesBuildFromData(TestWithConnector):
    """Tests for DataTableAndAttributes._data_to_schema using build_from_data with skip_sub_models."""

    def setUp(self) -> None:
        TestWithConnector.setUp(self)
        self.environment = Environment(hub_url=BASE_URL, org_id=ORG.id, workspace_id=WORKSPACE_ID)
        self.context = StaticContext.from_environment(
            environment=self.environment,
            connector=self.connector,
        )

    @contextlib.contextmanager
    def _mock_geoscience_objects(self):
        mock_client = MockClient(self.environment)
        with (
            patch("evo.objects.typed.attributes.get_data_client", lambda _: mock_client),
            patch("evo.objects.typed._data.get_data_client", lambda _: mock_client),
        ):
            yield mock_client

    async def test_data_to_schema_extra_properties(self):
        """Test that build_from_data processes additional properties defined on a subclass."""

        # The DataFrame has only coordinate columns (no attributes)
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0], "z": [7.0, 8.0, 9.0]})

        with self._mock_geoscience_objects():
            result = await ExtendedLocations._data_to_schema(df, self.context)

        # Coordinates and attributes are handled before the remaining properties.
        self.assertIn("coordinates", result)
        self.assertEqual(result["attributes"], [])

        # The extra property is not set because DataFrame doesn't have 'point_count' attribute
        # (getattr(df, "point_count", None) returns None, which deletes the key)
        self.assertNotIn("point_count", result)
