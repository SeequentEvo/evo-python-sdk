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
from unittest.mock import patch

import pandas as pd

from evo.common import Environment, StaticContext
from evo.common.test_tools import BASE_URL, ORG, WORKSPACE_ID, TestWithConnector
from evo.objects.typed import (
    PointSet,
    PointSetData,
    Regular3DGrid,
    Regular3DGridData,
    object_from_reference,
)
from evo.objects.typed.base import BaseSpatialObject
from evo.objects.typed.types import Point3, Size3d, Size3i

from .helpers import MockClient


class TestObjectFactory(TestWithConnector):
    """Test the ObjectFactory for creating typed objects from references."""

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
            patch("evo.objects.typed.dataset.get_data_client", lambda _: mock_client),
            patch("evo.objects.typed.base.create_geoscience_object", mock_client.create_geoscience_object),
            patch("evo.objects.typed.base.replace_geoscience_object", mock_client.replace_geoscience_object),
            patch("evo.objects.typed.base.download_geoscience_object", mock_client.from_reference),
        ):
            yield mock_client

    async def test_factory_from_reference_pointset(self):
        """Test that object_from_reference correctly creates a PointSet from a reference."""
        with self._mock_geoscience_objects():
            # Create a PointSet
            locations_df = pd.DataFrame(
                {
                    "x": [0.0, 1.0, 2.0],
                    "y": [0.0, 1.0, 2.0],
                    "z": [0.0, 1.0, 2.0],
                }
            )
            data = PointSetData(
                name="Test PointSet",
                locations=locations_df,
            )
            original = await PointSet.create(context=self.context, data=data)

            # Use object_from_reference to load it back
            result = await object_from_reference(context=self.context, reference=original.metadata.url)

            # Verify it's the correct type (inferred automatically)
            self.assertIsInstance(result, PointSet)
            self.assertEqual(result.name, "Test PointSet")

    async def test_factory_from_reference_regular_grid(self):
        """Test that object_from_reference correctly creates a Regular3DGrid from a reference."""
        with self._mock_geoscience_objects():
            # Create a Regular3DGrid
            data = Regular3DGridData(
                name="Test Grid",
                origin=Point3(x=0.0, y=0.0, z=0.0),
                size=Size3i(nx=10, ny=10, nz=10),
                cell_size=Size3d(dx=1.0, dy=1.0, dz=1.0),
            )
            original = await Regular3DGrid.create(context=self.context, data=data)

            # Use object_from_reference to load it back
            result = await object_from_reference(context=self.context, reference=original.metadata.url)

            # Verify it's the correct type (inferred automatically)
            self.assertIsInstance(result, Regular3DGrid)
            self.assertEqual(result.name, "Test Grid")

    async def test_factory_infers_correct_types(self):
        """Test that object_from_reference automatically infers the correct type based on sub-classification."""
        with self._mock_geoscience_objects():
            # Create a PointSet
            locations_df = pd.DataFrame(
                {
                    "x": [0.0, 1.0, 2.0],
                    "y": [0.0, 1.0, 2.0],
                    "z": [0.0, 1.0, 2.0],
                }
            )
            data = PointSetData(
                name="Test PointSet",
                locations=locations_df,
            )
            original = await PointSet.create(context=self.context, data=data)

            # Use object_from_reference without specifying type
            result = await object_from_reference(context=self.context, reference=original.metadata.url)

            # Verify it's automatically the correct type and is also a BaseSpatialObject
            self.assertIsInstance(result, PointSet)
            self.assertIsInstance(result, BaseSpatialObject)
            self.assertEqual(result.name, "Test PointSet")

    async def test_factory_automatically_selects_type(self):
        """Test that object_from_reference automatically selects the correct type based on sub-classification."""
        with self._mock_geoscience_objects():
            # Create different types of objects
            pointset_data = PointSetData(
                name="PointSet Object",
                locations=pd.DataFrame(
                    {
                        "x": [0.0, 1.0],
                        "y": [0.0, 1.0],
                        "z": [0.0, 1.0],
                    }
                ),
            )
            grid_data = Regular3DGridData(
                name="Grid Object",
                origin=Point3(x=0.0, y=0.0, z=0.0),
                size=Size3i(nx=5, ny=5, nz=5),
                cell_size=Size3d(dx=1.0, dy=1.0, dz=1.0),
            )

            pointset = await PointSet.create(context=self.context, data=pointset_data)
            grid = await Regular3DGrid.create(context=self.context, data=grid_data)

            # Load them back using object_from_reference
            loaded_pointset = await object_from_reference(context=self.context, reference=pointset.metadata.url)
            loaded_grid = await object_from_reference(context=self.context, reference=grid.metadata.url)

            # Verify the types were automatically selected correctly
            self.assertIsInstance(loaded_pointset, PointSet)
            self.assertNotIsInstance(loaded_pointset, Regular3DGrid)
            self.assertEqual(loaded_pointset.name, "PointSet Object")

            self.assertIsInstance(loaded_grid, Regular3DGrid)
            self.assertNotIsInstance(loaded_grid, PointSet)
            self.assertEqual(loaded_grid.name, "Grid Object")
