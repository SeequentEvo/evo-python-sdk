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
import uuid
from unittest.mock import patch

import pandas as pd
from parameterized import parameterized

from evo.common import Environment, StaticContext
from evo.common.test_tools import BASE_URL, ORG, WORKSPACE_ID, TestWithConnector
from evo.objects import ObjectReference
from evo.objects.typed import BoundingBox, PointSet, PointSetData
from evo.objects.typed.base import BaseObject
from evo.objects.typed.exceptions import ObjectValidationError

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
            patch("evo.objects.typed.attributes.get_data_client", lambda _: mock_client),
            patch("evo.objects.typed._data.get_data_client", lambda _: mock_client),
            patch("evo.objects.typed.base.create_geoscience_object", mock_client.create_geoscience_object),
            patch("evo.objects.typed.base.replace_geoscience_object", mock_client.replace_geoscience_object),
            patch("evo.objects.typed.base.download_geoscience_object", mock_client.from_reference),
            patch("evo.objects.DownloadedObject.from_context", mock_client.from_reference),
        ):
            yield mock_client

    example_pointset = PointSetData(
        name="Test PointSet",
        locations=pd.DataFrame(
            {
                "x": [0.0, 1.0, 0.0, 1.0, 0.5],
                "y": [0.0, 0.0, 1.0, 1.0, 0.5],
                "z": [0.0, 0.0, 0.0, 0.0, 1.0],
                "value": [1.0, 2.0, 3.0, 4.0, 5.0],
                "category": pd.Categorical(["a", "b", "a", "b", "c"]),
            }
        ),
    )

    def _assert_bounding_box_equal(
        self, bbox: BoundingBox, min_x: float, max_x: float, min_y: float, max_y: float, min_z: float, max_z: float
    ):
        self.assertAlmostEqual(bbox.min_x, min_x)
        self.assertAlmostEqual(bbox.max_x, max_x)
        self.assertAlmostEqual(bbox.min_y, min_y)
        self.assertAlmostEqual(bbox.max_y, max_y)
        self.assertAlmostEqual(bbox.min_z, min_z)
        self.assertAlmostEqual(bbox.max_z, max_z)

    @parameterized.expand([BaseObject, PointSet])
    async def test_create(self, class_to_call):
        with self._mock_geoscience_objects():
            result = await class_to_call.create(context=self.context, data=self.example_pointset)
        self.assertIsInstance(result, PointSet)
        self.assertEqual(result.name, "Test PointSet")
        self.assertEqual(result.num_points, 5)

        attr_df = await result.locations.get_dataframe()
        pd.testing.assert_frame_equal(attr_df, self.example_pointset.locations)

    @parameterized.expand([BaseObject, PointSet])
    async def test_replace(self, class_to_call):
        # Create a pointset with only coordinates (no attributes)
        df = pd.DataFrame(
            {
                "x": [0.0, 1.0, 0.0, 1.0, 0.5],
                "y": [0.0, 0.0, 1.0, 1.0, 0.5],
                "z": [0.0, 0.0, 0.0, 0.0, 1.0],
            }
        )
        data = PointSetData(
            name="Test PointSet",
            locations=df,
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
        self.assertEqual(result.num_points, 5)

        actual_df = await result.locations.get_dataframe()
        pd.testing.assert_frame_equal(actual_df, df)

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
        self.assertEqual(result.num_points, 5)

        actual_df = await result.locations.get_dataframe()
        pd.testing.assert_frame_equal(actual_df, self.example_pointset.locations)

    @parameterized.expand([BaseObject, PointSet])
    async def test_from_reference(self, class_to_call):
        with self._mock_geoscience_objects():
            original = await PointSet.create(context=self.context, data=self.example_pointset)

            result = await class_to_call.from_reference(context=self.context, reference=original.metadata.url)
            self.assertEqual(result.name, "Test PointSet")
            self.assertEqual(result.num_points, 5)

            actual_df = await result.locations.get_dataframe()
            pd.testing.assert_frame_equal(actual_df, self.example_pointset.locations)

    def test_bounding_box_from_data(self):
        """Test that the bounding box is computed correctly from the data."""
        bbox = self.example_pointset.compute_bounding_box()
        self._assert_bounding_box_equal(bbox, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0)

    async def test_bounding_box_from_object(self):
        """Test that the bounding box is stored correctly on the created object."""
        with self._mock_geoscience_objects() as mock_client:
            obj = await PointSet.create(context=self.context, data=self.example_pointset)

            bbox = obj.bounding_box
            self._assert_bounding_box_equal(bbox, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0)

            # Verify it was saved to the document
            bbox_dict = mock_client.objects[str(obj.metadata.url.object_id)]["bounding_box"]
            self.assertAlmostEqual(bbox_dict["min_x"], 0.0)
            self.assertAlmostEqual(bbox_dict["max_x"], 1.0)

    def test_coordinates_validation(self):
        """Test that coordinates validation works correctly."""
        # Missing x column
        with self.assertRaises(ObjectValidationError):
            PointSetData(
                name="Bad PointSet",
                locations=pd.DataFrame({"y": [0.0], "z": [0.0]}),
            )

        # Missing y and z columns
        with self.assertRaises(ObjectValidationError):
            PointSetData(
                name="Bad PointSet",
                locations=pd.DataFrame({"x": [0.0, 1.0]}),
            )

    async def test_create_with_coordinates_only(self):
        """Test creating a pointset with only coordinates (no attributes)."""
        data = PointSetData(
            name="Coordinates Only PointSet",
            locations=pd.DataFrame(
                {
                    "x": [0.0, 1.0, 2.0],
                    "y": [0.0, 1.0, 2.0],
                    "z": [0.0, 1.0, 2.0],
                }
            ),
        )
        with self._mock_geoscience_objects():
            result = await PointSet.create(context=self.context, data=data)
        self.assertEqual(result.num_points, 3)

    async def test_description_and_tags(self):
        """Test setting and getting description and tags."""
        data = PointSetData(
            name="Test PointSet",
            locations=self.example_pointset.locations,
            description="A test pointset for testing",
            tags={"category": "test", "priority": "high"},
        )
        with self._mock_geoscience_objects():
            result = await PointSet.create(context=self.context, data=data)

        self.assertEqual(result.description, "A test pointset for testing")
        self.assertEqual(result.tags, {"category": "test", "priority": "high"})

    async def test_update_dataframe_after_creation(self):
        """Test updating the locations DataFrame after pointset creation."""
        with self._mock_geoscience_objects():
            obj = await PointSet.create(context=self.context, data=self.example_pointset)

            # Set new data with different attribute
            new_df = pd.DataFrame(
                {
                    "x": [0.0, 1.0, 0.0, 1.0, 0.5],
                    "y": [0.0, 0.0, 1.0, 1.0, -0.5],
                    "z": [0.0, 0.0, 0.0, 0.0, 3.0],
                    "new_value": [10.0, 20.0, 30.0, 40.0, 50.0],
                }
            )
            await obj.locations.set_dataframe(new_df)

            # Verify the data was updated
            self.assertEqual(obj.num_points, 5)
            self._assert_bounding_box_equal(obj.bounding_box, 0.0, 1.0, -0.5, 1.0, 0.0, 3.0)

            await obj.update()
            df = await obj.locations.get_dataframe()
            pd.testing.assert_frame_equal(df, new_df)

    async def test_validate_attribute_length(self):
        """Test that validation fails when an attribute has incorrect length."""
        with self._mock_geoscience_objects():
            obj = await PointSet.create(context=self.context, data=self.example_pointset)

            # Manually modify the attribute length in the document to simulate incorrect data
            obj._document["locations"]["attributes"][0]["values"]["length"] = 100  # Wrong length
            obj._rebuild_models()

            with self.assertRaises(ObjectValidationError) as cm:
                obj.validate()
            self.assertIn("does not match expected length", str(cm.exception))

    async def test_json(self):
        """Test the JSON structure of the created object."""
        with self._mock_geoscience_objects() as mock_client:
            obj = await PointSet.create(context=self.context, data=self.example_pointset)

            # Get the JSON that was stored (would be sent to the API)
            object_json = mock_client.objects[str(obj.metadata.url.object_id)]

            # Verify schema
            self.assertEqual(object_json["schema"], "/objects/pointset/1.2.0/pointset.schema.json")

            # Verify base properties
            self.assertEqual(object_json["name"], "Test PointSet")
            self.assertIn("uuid", object_json)
            self.assertIn("bounding_box", object_json)
            self.assertEqual(object_json["coordinate_reference_system"], "unspecified")

            # Verify locations structure
            self.assertIn("locations", object_json)
            self.assertIn("coordinates", object_json["locations"])
            self.assertIn("data", object_json["locations"]["coordinates"])
            self.assertEqual(object_json["locations"]["coordinates"]["length"], 5)

            # Verify attributes structure
            self.assertEqual(len(object_json["locations"]["attributes"]), 2)
            self.assertEqual(object_json["locations"]["attributes"][0]["name"], "value")
            self.assertEqual(object_json["locations"]["attributes"][0]["attribute_type"], "scalar")
            self.assertEqual(object_json["locations"]["attributes"][1]["name"], "category")
            self.assertEqual(object_json["locations"]["attributes"][1]["attribute_type"], "category")

    async def test_repr_html_includes_attributes(self):
        """Test that _repr_html_ includes the locations attributes table."""
        with self._mock_geoscience_objects():
            obj = await PointSet.create(context=self.context, data=self.example_pointset)

            html = obj._repr_html_()

            # Verify basic metadata is present
            self.assertIn("Test PointSet", html)
            self.assertIn("Object ID:", html)
            self.assertIn("Schema:", html)

            # Verify bounding box section
            self.assertIn("Bounding box:", html)

            # Verify CRS section
            self.assertIn("CRS:", html)

            # Verify locations attributes table
            self.assertIn("locations:", html)
            self.assertIn("Attribute", html)
            self.assertIn("Type", html)
            self.assertIn("value", html)
            self.assertIn("scalar", html)
            self.assertIn("category", html)

    async def test_repr_html_no_attributes(self):
        """Test that _repr_html_ works correctly when there are no attributes."""
        df = pd.DataFrame(
            {
                "x": [0.0, 1.0, 0.0],
                "y": [0.0, 0.0, 1.0],
                "z": [0.0, 0.0, 0.0],
            }
        )
        data = PointSetData(name="No Attributes PointSet", locations=df)

        with self._mock_geoscience_objects():
            obj = await PointSet.create(context=self.context, data=data)

            html = obj._repr_html_()

            # Verify basic metadata is present
            self.assertIn("No Attributes PointSet", html)
            self.assertIn("Object ID:", html)

            # Verify no locations section when there are no attributes
            self.assertNotIn("locations:", html)
