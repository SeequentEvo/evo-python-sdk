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
from evo.objects.typed import Point3, Regular3DGrid, Regular3DGridData, Rotation, Size3d, Size3i
from evo.objects.typed.attributes import DataLoaderError
from evo.objects.typed.base import BaseObject
from evo.objects.typed.exceptions import ObjectValidationError

from .helpers import MockClient


class TestRegularGrid(TestWithConnector):
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
            patch("evo.objects.typed.base.create_geoscience_object", mock_client.create_geoscience_object),
            patch("evo.objects.typed.base.replace_geoscience_object", mock_client.replace_geoscience_object),
            patch("evo.objects.typed.base.download_geoscience_object", mock_client.from_reference),
        ):
            yield mock_client

    example_grid = Regular3DGridData(
        name="Test Grid",
        origin=Point3(0, 0, 0),
        size=Size3i(10, 10, 5),
        cell_size=Size3d(2.5, 5, 5),
        cell_data=pd.DataFrame(
            {
                "value": np.random.rand(10 * 10 * 5),
                "cat": pd.Categorical(np.random.choice(range(4), size=10 * 10 * 5), ["a", "b", "c", "d"]),
            }
        ),
        vertex_data=pd.DataFrame(
            {
                "elevation": np.random.rand(11 * 11 * 6),
            }
        ),
        rotation=Rotation(90, 0, 0),
    )

    @parameterized.expand([BaseObject, Regular3DGrid])
    async def test_create(self, class_to_call):
        with self._mock_geoscience_objects():
            result = await class_to_call.create(context=self.context, data=self.example_grid)
        self.assertIsInstance(result, Regular3DGrid)
        self.assertEqual(result.name, "Test Grid")
        self.assertEqual(result.origin, Point3(0, 0, 0))
        self.assertEqual(result.size, Size3i(10, 10, 5))
        self.assertEqual(result.cell_size, Size3d(2.5, 5, 5))
        self.assertEqual(result.rotation, Rotation(90, 0, 0))

        cell_df = await result.cells.as_dataframe()
        pd.testing.assert_frame_equal(cell_df, self.example_grid.cell_data)
        vertices_df = await result.vertices.as_dataframe()
        pd.testing.assert_frame_equal(vertices_df, self.example_grid.vertex_data)

    @parameterized.expand([BaseObject, Regular3DGrid])
    async def test_replace(self, class_to_call):
        data = dataclasses.replace(self.example_grid, vertex_data=None)
        with self._mock_geoscience_objects():
            result = await class_to_call.replace(
                context=self.context,
                reference=ObjectReference.new(
                    environment=self.context.get_environment(),
                    object_id=uuid.uuid4(),
                ),
                data=data,
            )
        self.assertIsInstance(result, Regular3DGrid)
        self.assertEqual(result.name, "Test Grid")
        self.assertEqual(result.origin, Point3(0, 0, 0))
        self.assertEqual(result.size, Size3i(10, 10, 5))
        self.assertEqual(result.cell_size, Size3d(2.5, 5, 5))
        self.assertEqual(result.rotation, Rotation(90, 0, 0))

        cell_df = await result.cells.as_dataframe()
        pd.testing.assert_frame_equal(cell_df, data.cell_data)
        vertices_df = await result.vertices.as_dataframe()
        self.assertEqual(vertices_df.shape[0], 0)  # No vertex data provided

    @parameterized.expand([BaseObject, Regular3DGrid])
    async def test_create_or_replace(self, class_to_call):
        with self._mock_geoscience_objects():
            result = await class_to_call.create_or_replace(
                context=self.context,
                reference=ObjectReference.new(
                    environment=self.context.get_environment(),
                    object_id=uuid.uuid4(),
                ),
                data=self.example_grid,
            )
        self.assertIsInstance(result, Regular3DGrid)
        self.assertEqual(result.name, "Test Grid")
        self.assertEqual(result.origin, Point3(0, 0, 0))
        self.assertEqual(result.size, Size3i(10, 10, 5))
        self.assertEqual(result.cell_size, Size3d(2.5, 5, 5))
        self.assertEqual(result.rotation, Rotation(90, 0, 0))

        cell_df = await result.cells.as_dataframe()
        pd.testing.assert_frame_equal(cell_df, self.example_grid.cell_data)
        vertices_df = await result.vertices.as_dataframe()
        pd.testing.assert_frame_equal(vertices_df, self.example_grid.vertex_data)

    async def test_from_reference(self):
        with self._mock_geoscience_objects():
            original = await Regular3DGrid.create(context=self.context, data=self.example_grid)

            result = await Regular3DGrid.from_reference(context=self.context, reference=original.metadata.url)
            self.assertEqual(result.name, "Test Grid")
            self.assertEqual(result.origin, Point3(0, 0, 0))
            self.assertEqual(result.size, Size3i(10, 10, 5))
            self.assertEqual(result.cell_size, Size3d(2.5, 5, 5))
            self.assertEqual(result.rotation, Rotation(90, 0, 0))

            cell_df = await result.cells.as_dataframe()
            pd.testing.assert_frame_equal(cell_df, self.example_grid.cell_data)
            vertices_df = await result.vertices.as_dataframe()
            pd.testing.assert_frame_equal(vertices_df, self.example_grid.vertex_data)

    async def test_update(self):
        with self._mock_geoscience_objects():
            obj = await Regular3DGrid.create(context=self.context, data=self.example_grid)

            self.assertEqual(obj.metadata.version_id, "1")
            obj.name = "Updated Grid"
            obj.origin = Point3(1, 1, 1)
            await obj.cells.set_dataframe(
                pd.DataFrame(
                    {
                        "value": np.ones(10 * 10 * 5),
                    }
                )
            )

            with self.assertRaises(DataLoaderError):
                await obj.cells.as_dataframe()

            await obj.update()

            self.assertEqual(obj.name, "Updated Grid")
            self.assertEqual(obj.origin, Point3(1, 1, 1))
            self.assertEqual(obj.metadata.version_id, "2")

            cell_df = await obj.cells.as_dataframe()
            pd.testing.assert_frame_equal(
                cell_df,
                pd.DataFrame(
                    {
                        "value": np.ones(10 * 10 * 5),
                    }
                ),
            )

    async def test_size_check(self):
        with self.assertRaises(ValueError):
            dataclasses.replace(self.example_grid, size=Size3i(15, 10, 6))

        with self._mock_geoscience_objects():
            obj = await Regular3DGrid.create(context=self.context, data=self.example_grid)
            with self.assertRaises(ObjectValidationError):
                await obj.cells.set_dataframe(
                    pd.DataFrame(
                        {
                            "value": np.random.rand(11 * 10 * 5),
                        }
                    )
                )

            obj.size = Size3i(5, 10, 6)
            with self.assertRaises(ObjectValidationError):
                obj.validate()

    async def test_size_check_data(self):
        with self.assertRaises(ObjectValidationError):
            Regular3DGridData(
                name="Test Grid",
                origin=Point3(0, 0, 0),
                size=Size3i(10, 10, 5),
                cell_size=Size3d(2.5, 5, 5),
                cell_data=pd.DataFrame(
                    {
                        "value": np.random.rand(12 * 10 * 5),
                    }
                ),
                rotation=Rotation(90, 0, 0),
            )

    async def test_bounding_box(self):
        with self._mock_geoscience_objects() as mock_client:
            obj = await Regular3DGrid.create(context=self.context, data=self.example_grid)

            bbox = obj.bounding_box
            self.assertAlmostEqual(bbox.min_x, 0.0)
            self.assertAlmostEqual(bbox.min_y, -25.0)
            self.assertAlmostEqual(bbox.min_z, 0.0)
            self.assertAlmostEqual(bbox.max_x, 50.0)
            self.assertAlmostEqual(bbox.max_y, 0.0)
            self.assertAlmostEqual(bbox.max_z, 25.0)

            bbox = mock_client.objects[str(obj.metadata.url.object_id)]["bounding_box"]
            self.assertAlmostEqual(bbox["min_x"], 0.0)
            self.assertAlmostEqual(bbox["min_y"], -25.0)
            self.assertAlmostEqual(bbox["min_z"], 0.0)
            self.assertAlmostEqual(bbox["max_x"], 50.0)
            self.assertAlmostEqual(bbox["max_y"], 0.0)
            self.assertAlmostEqual(bbox["max_z"], 25.0)

            obj.origin = Point3(1, 1, 1)
            bbox = obj.bounding_box
            self.assertAlmostEqual(bbox.min_x, 1.0)
            self.assertAlmostEqual(bbox.min_y, -24.0)
            self.assertAlmostEqual(bbox.min_z, 1.0)
            self.assertAlmostEqual(bbox.max_x, 51.0)
            self.assertAlmostEqual(bbox.max_y, 1.0)
            self.assertAlmostEqual(bbox.max_z, 26.0)

    async def test_validate_cell_attribute_length(self):
        """Test that validation fails when a cell attribute has incorrect length."""
        with self._mock_geoscience_objects():
            obj = await Regular3DGrid.create(context=self.context, data=self.example_grid)

            # Manually modify the attribute length in the document to simulate incorrect data
            obj._document["cell_attributes"][0]["values"]["length"] = 100  # Wrong length
            obj._rebuild_models()

            with self.assertRaises(ObjectValidationError) as cm:
                obj.validate()
            self.assertIn("does not match expected length", str(cm.exception))

    async def test_validate_vertex_attribute_length(self):
        """Test that validation fails when a vertex attribute has incorrect length."""
        with self._mock_geoscience_objects():
            obj = await Regular3DGrid.create(context=self.context, data=self.example_grid)

            # Manually modify the attribute length in the document to simulate incorrect data
            obj._document["vertex_attributes"][0]["values"]["length"] = 100  # Wrong length
            obj._rebuild_models()

            with self.assertRaises(ObjectValidationError) as cm:
                obj.validate()
            self.assertIn("does not match expected length", str(cm.exception))

    async def test_validate_multiple_attributes_same_length(self):
        """Test that validation passes when all cell attributes have correct length."""
        with self._mock_geoscience_objects():
            obj = await Regular3DGrid.create(context=self.context, data=self.example_grid)

            # Verify validation passes with correct data
            obj.validate()  # Should not raise

    async def test_from_reference_with_base_object(self):
        """Test that from_reference works when called on BaseObject."""
        with self._mock_geoscience_objects():
            original = await Regular3DGrid.create(context=self.context, data=self.example_grid)

            # Download using BaseObject.from_reference
            result = await BaseObject.from_reference(context=self.context, reference=original.metadata.url)

            # Should return a Regular3DGrid instance
            self.assertIsInstance(result, Regular3DGrid)
            self.assertEqual(result.name, "Test Grid")

    async def test_description_and_tags(self):
        """Test setting and getting description and tags."""
        data = dataclasses.replace(
            self.example_grid,
            description="A test grid for testing",
            tags={"category": "test", "priority": "high"},
        )
        with self._mock_geoscience_objects():
            result = await Regular3DGrid.create(context=self.context, data=data)

        self.assertEqual(result.description, "A test grid for testing")
        self.assertEqual(result.tags, {"category": "test", "priority": "high"})

    async def test_append_attribute_after_creation(self):
        """Test appending an attribute after grid creation."""
        with self._mock_geoscience_objects():
            obj = await Regular3DGrid.create(context=self.context, data=self.example_grid)

            # Append a new attribute
            new_attr = pd.DataFrame({"new_value": np.random.rand(10 * 10 * 5)})
            await obj.cells.attributes.append_attribute(new_attr)

            # Verify the attribute was added
            self.assertEqual(len(obj.cells.attributes), 3)  # Original 2 + 1 new
            self.assertEqual(obj.cells.attributes[-1].name, "new_value")

    async def test_set_cell_size(self):
        """Test modifying cell_size property."""
        # Use a grid without rotation for simpler bounding box calculation
        data = dataclasses.replace(self.example_grid, rotation=None)
        with self._mock_geoscience_objects():
            obj = await Regular3DGrid.create(context=self.context, data=data)

            obj.cell_size = Size3d(5.0, 10.0, 10.0)
            self.assertEqual(obj.cell_size, Size3d(5.0, 10.0, 10.0))

            # Bounding box should reflect new cell size (no rotation)
            bbox = obj.bounding_box
            self.assertAlmostEqual(bbox.max_x - bbox.min_x, 50.0)  # 10 * 5.0
            self.assertAlmostEqual(bbox.max_y - bbox.min_y, 100.0)  # 10 * 10.0
            self.assertAlmostEqual(bbox.max_z - bbox.min_z, 50.0)  # 5 * 10.0
