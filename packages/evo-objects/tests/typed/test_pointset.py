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
import dataclasses
import uuid
from unittest.mock import patch

import numpy as np
import pandas as pd
from parameterized import parameterized

from evo.common import Environment, StaticContext
from evo.common.test_tools import BASE_URL, ORG, WORKSPACE_ID, TestWithConnector
from evo.objects import ObjectReference
from evo.objects.typed import PointSet, PointSetData
from evo.objects.typed.base import BaseObject
from evo.objects.typed.dataset import DataLoaderError

from .helpers import MockClient


class TestPointSet(TestWithConnector):
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

    example_pointset = PointSetData(
        name="Test PointSet",
        locations=pd.DataFrame(
            {
                "x": np.random.rand(100),
                "y": np.random.rand(100),
                "z": np.random.rand(100),
                "value": np.random.rand(100),
                "category": pd.Categorical(np.random.choice(range(3), size=100), ["a", "b", "c"]),
            }
        ),
    )

    @parameterized.expand([BaseObject, PointSet])
    async def test_create(self, class_to_call):
        with self._mock_geoscience_objects():
            result = await class_to_call.create(context=self.context, data=self.example_pointset)
        self.assertIsInstance(result, PointSet)
        self.assertEqual(result.name, "Test PointSet")

        locations_df = await result.locations.as_dataframe()
        pd.testing.assert_frame_equal(locations_df, self.example_pointset.locations)

    @parameterized.expand([BaseObject, PointSet])
    async def test_replace(self, class_to_call):
        data = dataclasses.replace(
            self.example_pointset,
            locations=pd.DataFrame(
                {
                    "x": [1.0, 2.0, 3.0],
                    "y": [4.0, 5.0, 6.0],
                    "z": [7.0, 8.0, 9.0],
                }
            ),
        )
        with self._mock_geoscience_objects():
            result = await class_to_call.replace(
                context=self.context,
                reference=ObjectReference.new(
                    environment=self.context.get_environment(),
                    object_id=uuid.uuid4(),
                ),
                data=data,
            )
        self.assertIsInstance(result, PointSet)
        self.assertEqual(result.name, "Test PointSet")

        locations_df = await result.locations.as_dataframe()
        pd.testing.assert_frame_equal(locations_df, data.locations)

    @parameterized.expand([BaseObject, PointSet])
    async def test_create_or_replace(self, class_to_call):
        with self._mock_geoscience_objects():
            result = await class_to_call.create_or_replace(
                context=self.context,
                reference=ObjectReference.new(
                    environment=self.context.get_environment(),
                    object_id=uuid.uuid4(),
                ),
                data=self.example_pointset,
            )
        self.assertIsInstance(result, PointSet)
        self.assertEqual(result.name, "Test PointSet")

        locations_df = await result.locations.as_dataframe()
        pd.testing.assert_frame_equal(locations_df, self.example_pointset.locations)

    async def test_from_reference(self):
        with self._mock_geoscience_objects():
            original = await PointSet.create(context=self.context, data=self.example_pointset)

            result = await PointSet.from_reference(context=self.context, reference=original.metadata.url)
            self.assertEqual(result.name, "Test PointSet")

            locations_df = await result.locations.as_dataframe()
            pd.testing.assert_frame_equal(locations_df, self.example_pointset.locations)

    async def test_update(self):
        with self._mock_geoscience_objects():
            obj = await PointSet.create(context=self.context, data=self.example_pointset)

            self.assertEqual(obj.metadata.version_id, "1")
            obj.name = "Updated PointSet"
            await obj.locations.set_dataframe(
                pd.DataFrame(
                    {
                        "x": [10.0, 20.0],
                        "y": [30.0, 40.0],
                        "z": [50.0, 60.0],
                        "attribute": [1.0, 2.0],
                    }
                )
            )

            with self.assertRaises(DataLoaderError):
                await obj.locations.as_dataframe()

            await obj.update()

            self.assertEqual(obj.name, "Updated PointSet")
            self.assertEqual(obj.metadata.version_id, "2")

            locations_df = await obj.locations.as_dataframe()
            pd.testing.assert_frame_equal(
                locations_df,
                pd.DataFrame(
                    {
                        "x": [10.0, 20.0],
                        "y": [30.0, 40.0],
                        "z": [50.0, 60.0],
                        "attribute": [1.0, 2.0],
                    }
                ),
            )

    async def test_create_empty_pointset(self):
        """Test creating a pointset with no locations."""
        data = PointSetData(
            name="Empty PointSet",
            locations=pd.DataFrame(columns=["x", "y", "z"]),
        )
        with self._mock_geoscience_objects():
            result = await PointSet.create(context=self.context, data=data)
        self.assertIsInstance(result, PointSet)
        self.assertEqual(result.name, "Empty PointSet")

        locations_df = await result.locations.as_dataframe()
        self.assertEqual(len(locations_df), 0)

    async def test_bounding_box_computation(self):
        """Test that bounding box is computed correctly from point data."""
        data = PointSetData(
            name="Test Bounds",
            locations=pd.DataFrame(
                {
                    "x": [1.0, 5.0, 3.0],
                    "y": [2.0, 8.0, 4.0],
                    "z": [0.0, 10.0, 5.0],
                }
            ),
        )

        bbox = data.compute_bounding_box()
        self.assertEqual(bbox.min_x, 1.0)
        self.assertEqual(bbox.max_x, 5.0)
        self.assertEqual(bbox.min_y, 2.0)
        self.assertEqual(bbox.max_y, 8.0)
        self.assertEqual(bbox.min_z, 0.0)
        self.assertEqual(bbox.max_z, 10.0)

    async def test_bounding_box_empty(self):
        """Test bounding box computation for empty pointset."""
        data = PointSetData(
            name="Empty",
            locations=pd.DataFrame(columns=["x", "y", "z"]),
        )

        bbox = data.compute_bounding_box()
        self.assertEqual(bbox.min_x, 0.0)
        self.assertEqual(bbox.max_x, 0.0)
        self.assertEqual(bbox.min_y, 0.0)
        self.assertEqual(bbox.max_y, 0.0)
        self.assertEqual(bbox.min_z, 0.0)
        self.assertEqual(bbox.max_z, 0.0)

    async def test_bounding_box_none(self):
        """Test bounding box computation when locations is None."""
        data = PointSetData(
            name="No Locations",
            locations=None,
        )

        bbox = data.compute_bounding_box()
        self.assertEqual(bbox.min_x, 0.0)
        self.assertEqual(bbox.max_x, 0.0)
        self.assertEqual(bbox.min_y, 0.0)
        self.assertEqual(bbox.max_y, 0.0)
        self.assertEqual(bbox.min_z, 0.0)
        self.assertEqual(bbox.max_z, 0.0)

    async def test_bounding_box_in_created_object(self):
        """Test that bounding box is stored correctly in created object."""
        data = PointSetData(
            name="Test Bounds Storage",
            locations=pd.DataFrame(
                {
                    "x": [0.0, 10.0],
                    "y": [5.0, 15.0],
                    "z": [2.0, 8.0],
                    "value": [1.0, 2.0],
                }
            ),
        )

        with self._mock_geoscience_objects() as mock_client:
            obj = await PointSet.create(context=self.context, data=data)

            bbox = obj.bounding_box
            self.assertEqual(bbox.min_x, 0.0)
            self.assertEqual(bbox.max_x, 10.0)
            self.assertEqual(bbox.min_y, 5.0)
            self.assertEqual(bbox.max_y, 15.0)
            self.assertEqual(bbox.min_z, 2.0)
            self.assertEqual(bbox.max_z, 8.0)

            # Check that it's stored in the mock object
            bbox_dict = mock_client.objects[str(obj.metadata.url.object_id)]["bounding_box"]
            self.assertEqual(bbox_dict["min_x"], 0.0)
            self.assertEqual(bbox_dict["max_x"], 10.0)
            self.assertEqual(bbox_dict["min_y"], 5.0)
            self.assertEqual(bbox_dict["max_y"], 15.0)
            self.assertEqual(bbox_dict["min_z"], 2.0)
            self.assertEqual(bbox_dict["max_z"], 8.0)

    async def test_pointset_with_only_coordinates(self):
        """Test creating a pointset with only x, y, z coordinates and no additional attributes."""
        data = PointSetData(
            name="Coordinates Only",
            locations=pd.DataFrame(
                {
                    "x": [1.0, 2.0, 3.0, 4.0, 5.0],
                    "y": [5.0, 4.0, 3.0, 2.0, 1.0],
                    "z": [0.0, 1.0, 2.0, 1.0, 0.0],
                }
            ),
        )

        with self._mock_geoscience_objects():
            result = await PointSet.create(context=self.context, data=data)
        self.assertIsInstance(result, PointSet)

        locations_df = await result.locations.as_dataframe()
        pd.testing.assert_frame_equal(locations_df, data.locations)

    async def test_pointset_with_many_attributes(self):
        """Test creating a pointset with many different attribute types."""
        data = PointSetData(
            name="Many Attributes",
            locations=pd.DataFrame(
                {
                    "x": np.random.rand(50),
                    "y": np.random.rand(50),
                    "z": np.random.rand(50),
                    "temperature": np.random.rand(50) * 100,
                    "pressure": np.random.rand(50) * 1000,
                    "density": np.random.rand(50) * 2.5,
                    "rock_type": pd.Categorical(
                        np.random.choice(["granite", "basalt", "sandstone", "limestone"], size=50)
                    ),
                    "quality": pd.Categorical(np.random.choice(["good", "fair", "poor"], size=50)),
                }
            ),
        )

        with self._mock_geoscience_objects():
            result = await PointSet.create(context=self.context, data=data)
        self.assertIsInstance(result, PointSet)

        locations_df = await result.locations.as_dataframe()
        pd.testing.assert_frame_equal(locations_df, data.locations)
