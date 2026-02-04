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

"""Tests for base class extensions (refresh, to_dataframe) on typed objects."""

from __future__ import annotations

import contextlib
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import pandas as pd

from evo.common import Environment, StaticContext
from evo.common.test_tools import BASE_URL, ORG, WORKSPACE_ID, TestWithConnector
from evo.objects.typed import PointSet, PointSetData, Regular3DGrid, Regular3DGridData
from evo.objects.typed.base import _BaseObject
from evo.objects.typed.types import Point3, Size3d, Size3i

from .helpers import MockClient


class TestRefreshMethodExists(TestCase):
    """Test that refresh() method exists on all typed objects."""

    def test_base_object_has_refresh(self):
        """Test that _BaseObject has refresh method."""
        self.assertTrue(hasattr(_BaseObject, "refresh"))

    def test_pointset_has_refresh(self):
        """Test that PointSet has refresh method."""
        self.assertTrue(hasattr(PointSet, "refresh"))

    def test_regular3dgrid_has_refresh(self):
        """Test that Regular3DGrid has refresh method."""
        self.assertTrue(hasattr(Regular3DGrid, "refresh"))


class TestRefreshOnPointSet(TestWithConnector):
    """Integration tests for refresh() on PointSet."""

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
            patch("evo.objects.typed.base.create_geoscience_object", mock_client.create_geoscience_object),
            patch("evo.objects.typed.base.replace_geoscience_object", mock_client.replace_geoscience_object),
            patch("evo.objects.typed.base.download_geoscience_object", mock_client.from_reference),
        ):
            yield mock_client

    async def test_refresh_returns_same_type(self):
        """Test that refresh() returns the same type as the original object."""
        data = PointSetData(
            name="Test PointSet",
            locations=pd.DataFrame({
                "x": [1.0, 2.0, 3.0],
                "y": [4.0, 5.0, 6.0],
                "z": [7.0, 8.0, 9.0],
            }),
        )

        with self._mock_geoscience_objects():
            original = await PointSet.create(context=self.context, data=data)
            refreshed = await original.refresh()

        self.assertIsInstance(refreshed, PointSet)
        self.assertEqual(type(refreshed), type(original))

    async def test_refresh_preserves_name(self):
        """Test that refresh() preserves object name."""
        data = PointSetData(
            name="Test PointSet",
            locations=pd.DataFrame({
                "x": [1.0, 2.0],
                "y": [3.0, 4.0],
                "z": [5.0, 6.0],
            }),
        )

        with self._mock_geoscience_objects():
            original = await PointSet.create(context=self.context, data=data)
            refreshed = await original.refresh()

        self.assertEqual(refreshed.name, original.name)

    async def test_refresh_preserves_data(self):
        """Test that refresh() preserves object data."""
        locations_df = pd.DataFrame({
            "x": np.random.rand(10),
            "y": np.random.rand(10),
            "z": np.random.rand(10),
            "value": np.random.rand(10),
        })
        data = PointSetData(name="Test PointSet", locations=locations_df)

        with self._mock_geoscience_objects():
            original = await PointSet.create(context=self.context, data=data)
            refreshed = await original.refresh()

        original_df = await original.locations.to_dataframe()
        refreshed_df = await refreshed.locations.to_dataframe()
        pd.testing.assert_frame_equal(original_df, refreshed_df)


class TestRefreshOnRegular3DGrid(TestWithConnector):
    """Integration tests for refresh() on Regular3DGrid."""

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
            patch("evo.objects.typed.base.create_geoscience_object", mock_client.create_geoscience_object),
            patch("evo.objects.typed.base.replace_geoscience_object", mock_client.replace_geoscience_object),
            patch("evo.objects.typed.base.download_geoscience_object", mock_client.from_reference),
        ):
            yield mock_client

    async def test_refresh_on_grid(self):
        """Test that refresh() works on Regular3DGrid."""
        data = Regular3DGridData(
            name="Test Grid",
            origin=Point3(0.0, 0.0, 0.0),
            cell_size=Size3d(1.0, 1.0, 1.0),
            size=Size3i(5, 5, 5),
            cell_data=pd.DataFrame({
                "value": np.random.rand(125),  # 5x5x5 = 125 cells
            }),
        )

        with self._mock_geoscience_objects():
            original = await Regular3DGrid.create(context=self.context, data=data)
            refreshed = await original.refresh()

        self.assertIsInstance(refreshed, Regular3DGrid)
        self.assertEqual(refreshed.name, original.name)

