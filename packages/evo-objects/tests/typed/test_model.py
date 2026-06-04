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

import contextlib
from dataclasses import dataclass
from typing import Annotated, ClassVar
from unittest.mock import patch

import pandas as pd

from evo.common import Environment, StaticContext
from evo.common.test_tools import BASE_URL, ORG, WORKSPACE_ID, TestWithConnector
from evo.objects.typed._data import DataTable, DataTableAndAttributes
from evo.objects.typed._model import SchemaBuilder, SchemaLocation, SchemaModel
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


class TestTable(DataTable):
    table_format: ClassVar[KnownTableFormat] = FLOAT_ARRAY_3
    data_columns: ClassVar[list[str]] = ["x", "y", "z"]


class ExtendedLocations(DataTableAndAttributes):
    _table: Annotated[TestTable, SchemaLocation("coordinates")]
    point_count: Annotated[int, SchemaLocation("point_count")]


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

        # Coordinates and attributes should be handled by the manual set_sub_model_value calls
        self.assertIn("coordinates", result)
        self.assertEqual(result["attributes"], [])

        # The extra property is not set because DataFrame doesn't have 'point_count' attribute
        # (getattr(df, "point_count", None) returns None, which deletes the key)
        self.assertNotIn("point_count", result)
