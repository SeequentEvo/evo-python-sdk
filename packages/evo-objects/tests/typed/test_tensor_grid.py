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

from evo.common import Environment, StaticContext
from evo.common.test_tools import BASE_URL, ORG, WORKSPACE_ID, TestWithConnector
from evo.objects import ObjectReference
from evo.objects.typed import Point3, Rotation, Size3i
from evo.objects.typed.exceptions import ObjectValidationError
from evo.objects.typed.tensor_grid import Tensor3DGrid, Tensor3DGridData

from .helpers import MockClient


class TestTensorGrid(TestWithConnector):
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
            patch("evo.objects.DownloadedObject.from_context", mock_client.from_reference),
        ):
            yield mock_client

    example_grid = Tensor3DGridData(
        name="Test Tensor Grid",
        origin=Point3(0, 0, 0),
        size=Size3i(10, 10, 5),
        cell_sizes_x=np.array([1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 4.0, 3.0, 2.0, 1.0]),
        cell_sizes_y=np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]),
        cell_sizes_z=np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
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

    async def test_create(self):
        with self._mock_geoscience_objects():
            result = await Tensor3DGrid.create(context=self.context, data=self.example_grid)

        self.assertEqual(result.name, "Test Tensor Grid")
        self.assertEqual(result.origin, Point3(0, 0, 0))
        self.assertEqual(result.size, Size3i(10, 10, 5))
        self.assertEqual(result.rotation, Rotation(90, 0, 0))
        np.testing.assert_array_equal(result.cell_sizes_x, self.example_grid.cell_sizes_x)
        np.testing.assert_array_equal(result.cell_sizes_y, self.example_grid.cell_sizes_y)
        np.testing.assert_array_equal(result.cell_sizes_z, self.example_grid.cell_sizes_z)

        cell_df = await result.cells.as_dataframe()
        pd.testing.assert_frame_equal(cell_df, self.example_grid.cell_data)
        vertices_df = await result.vertices.as_dataframe()
        pd.testing.assert_frame_equal(vertices_df, self.example_grid.vertex_data)

    async def test_create_with_no_data(self):
        data = dataclasses.replace(self.example_grid, cell_data=None, vertex_data=None)
        with self._mock_geoscience_objects():
            result = await Tensor3DGrid.create(context=self.context, data=data)

        self.assertEqual(result.name, "Test Tensor Grid")
        np.testing.assert_array_equal(result.cell_sizes_x, self.example_grid.cell_sizes_x)

        cell_df = await result.cells.as_dataframe()
        self.assertEqual(cell_df.shape[0], 0)

    async def test_replace(self):
        data = dataclasses.replace(self.example_grid, vertex_data=None)
        with self._mock_geoscience_objects():
            result = await Tensor3DGrid.replace(
                context=self.context,
                reference=ObjectReference.new(
                    environment=self.context.get_environment(),
                    object_id=uuid.uuid4(),
                ),
                data=data,
            )

        self.assertEqual(result.name, "Test Tensor Grid")
        self.assertEqual(result.origin, Point3(0, 0, 0))
        self.assertEqual(result.size, Size3i(10, 10, 5))
        np.testing.assert_array_equal(result.cell_sizes_x, self.example_grid.cell_sizes_x)

        cell_df = await result.cells.as_dataframe()
        pd.testing.assert_frame_equal(cell_df, data.cell_data)

    async def test_from_reference(self):
        with self._mock_geoscience_objects():
            original = await Tensor3DGrid.create(context=self.context, data=self.example_grid)

            result = await Tensor3DGrid.from_reference(context=self.context, reference=original.metadata.url)

            self.assertEqual(result.name, "Test Tensor Grid")
            self.assertEqual(result.origin, Point3(0, 0, 0))
            self.assertEqual(result.size, Size3i(10, 10, 5))
            self.assertEqual(result.rotation, Rotation(90, 0, 0))
            np.testing.assert_array_equal(result.cell_sizes_x, self.example_grid.cell_sizes_x)
            np.testing.assert_array_equal(result.cell_sizes_y, self.example_grid.cell_sizes_y)
            np.testing.assert_array_equal(result.cell_sizes_z, self.example_grid.cell_sizes_z)

            cell_df = await result.cells.as_dataframe()
            pd.testing.assert_frame_equal(cell_df, self.example_grid.cell_data)

    async def test_cell_sizes_x_validation(self):
        # Wrong number of x cell sizes
        with self.assertRaises(ObjectValidationError):
            Tensor3DGridData(
                name="Bad Grid",
                origin=Point3(0, 0, 0),
                size=Size3i(10, 10, 5),
                cell_sizes_x=np.array([1.0, 2.0, 3.0]),  # Only 3 instead of 10
                cell_sizes_y=np.array([2.0] * 10),
                cell_sizes_z=np.array([1.0] * 5),
            )

    async def test_cell_sizes_y_validation(self):
        # Wrong number of y cell sizes
        with self.assertRaises(ObjectValidationError):
            Tensor3DGridData(
                name="Bad Grid",
                origin=Point3(0, 0, 0),
                size=Size3i(10, 10, 5),
                cell_sizes_x=np.array([1.0] * 10),
                cell_sizes_y=np.array([2.0] * 5),  # Only 5 instead of 10
                cell_sizes_z=np.array([1.0] * 5),
            )

    async def test_cell_sizes_z_validation(self):
        # Wrong number of z cell sizes
        with self.assertRaises(ObjectValidationError):
            Tensor3DGridData(
                name="Bad Grid",
                origin=Point3(0, 0, 0),
                size=Size3i(10, 10, 5),
                cell_sizes_x=np.array([1.0] * 10),
                cell_sizes_y=np.array([2.0] * 10),
                cell_sizes_z=np.array([1.0] * 10),  # 10 instead of 5
            )

    async def test_positive_cell_sizes_validation(self):
        # Negative x cell size
        with self.assertRaises(ObjectValidationError):
            Tensor3DGridData(
                name="Bad Grid",
                origin=Point3(0, 0, 0),
                size=Size3i(10, 10, 5),
                cell_sizes_x=np.array([1.0, -2.0, 3.0, 4.0, 5.0, 5.0, 4.0, 3.0, 2.0, 1.0]),  # Negative value
                cell_sizes_y=np.array([2.0] * 10),
                cell_sizes_z=np.array([1.0] * 5),
            )

        # Zero y cell size
        with self.assertRaises(ObjectValidationError):
            Tensor3DGridData(
                name="Bad Grid",
                origin=Point3(0, 0, 0),
                size=Size3i(10, 10, 5),
                cell_sizes_x=np.array([1.0] * 10),
                cell_sizes_y=np.array([2.0, 0.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]),  # Zero value
                cell_sizes_z=np.array([1.0] * 5),
            )

    async def test_cell_data_size_validation(self):
        # Cell data wrong size
        with self.assertRaises(ObjectValidationError):
            Tensor3DGridData(
                name="Bad Grid",
                origin=Point3(0, 0, 0),
                size=Size3i(10, 10, 5),
                cell_sizes_x=np.array([1.0] * 10),
                cell_sizes_y=np.array([2.0] * 10),
                cell_sizes_z=np.array([1.0] * 5),
                cell_data=pd.DataFrame({"value": np.random.rand(100)}),  # Should be 500
            )

    async def test_vertex_data_size_validation(self):
        # Vertex data wrong size
        with self.assertRaises(ObjectValidationError):
            Tensor3DGridData(
                name="Bad Grid",
                origin=Point3(0, 0, 0),
                size=Size3i(10, 10, 5),
                cell_sizes_x=np.array([1.0] * 10),
                cell_sizes_y=np.array([2.0] * 10),
                cell_sizes_z=np.array([1.0] * 5),
                vertex_data=pd.DataFrame({"elevation": np.random.rand(100)}),  # Should be 11*11*6
            )

    async def test_bounding_box(self):
        with self._mock_geoscience_objects() as mock_client:
            obj = await Tensor3DGrid.create(context=self.context, data=self.example_grid)

            bbox = obj.bounding_box
            # Sum of x cell sizes = 30, y = 20, z = 5, rotated 90 degrees around z axis
            # After rotation: x_extent=20, y_extent=-30 (becomes min_y), z_extent=5
            self.assertAlmostEqual(bbox.min_x, 0.0)
            self.assertAlmostEqual(bbox.min_y, -30.0)
            self.assertAlmostEqual(bbox.min_z, 0.0)
            self.assertAlmostEqual(bbox.max_x, 20.0)
            self.assertAlmostEqual(bbox.max_y, 0.0)
            self.assertAlmostEqual(bbox.max_z, 5.0)

            bbox = mock_client.objects[str(obj.metadata.url.object_id)]["bounding_box"]
            self.assertAlmostEqual(bbox["min_x"], 0.0)
            self.assertAlmostEqual(bbox["min_y"], -30.0)
            self.assertAlmostEqual(bbox["min_z"], 0.0)
            self.assertAlmostEqual(bbox["max_x"], 20.0)
            self.assertAlmostEqual(bbox["max_y"], 0.0)
            self.assertAlmostEqual(bbox["max_z"], 5.0)

    async def test_bounding_box_no_rotation(self):
        data = dataclasses.replace(self.example_grid, rotation=None)
        with self._mock_geoscience_objects():
            obj = await Tensor3DGrid.create(context=self.context, data=data)

            bbox = obj.bounding_box
            # Sum of x cell sizes = 30, y = 20, z = 5, no rotation
            self.assertAlmostEqual(bbox.min_x, 0.0)
            self.assertAlmostEqual(bbox.min_y, 0.0)
            self.assertAlmostEqual(bbox.min_z, 0.0)
            self.assertAlmostEqual(bbox.max_x, 30.0)
            self.assertAlmostEqual(bbox.max_y, 20.0)
            self.assertAlmostEqual(bbox.max_z, 5.0)

    async def test_update(self):
        with self._mock_geoscience_objects():
            obj = await Tensor3DGrid.create(context=self.context, data=self.example_grid)

            # Update cell data
            new_cell_data = pd.DataFrame({"value": np.ones(500)})
            await obj.cells.set_dataframe(new_cell_data)

            await obj.update()

            cell_df = await obj.cells.as_dataframe()
            pd.testing.assert_frame_equal(cell_df, new_cell_data)

    async def test_uniform_cell_sizes(self):
        """Test tensor grid with uniform cell sizes (equivalent to regular grid)."""
        data = Tensor3DGridData(
            name="Uniform Tensor Grid",
            origin=Point3(0, 0, 0),
            size=Size3i(5, 5, 5),
            cell_sizes_x=np.array([2.0] * 5),
            cell_sizes_y=np.array([2.0] * 5),
            cell_sizes_z=np.array([2.0] * 5),
        )

        with self._mock_geoscience_objects():
            result = await Tensor3DGrid.create(context=self.context, data=data)

        self.assertEqual(result.name, "Uniform Tensor Grid")
        np.testing.assert_array_equal(result.cell_sizes_x, [2.0] * 5)
        np.testing.assert_array_equal(result.cell_sizes_y, [2.0] * 5)
        np.testing.assert_array_equal(result.cell_sizes_z, [2.0] * 5)

        bbox = result.bounding_box
        self.assertAlmostEqual(bbox.max_x - bbox.min_x, 10.0)
        self.assertAlmostEqual(bbox.max_y - bbox.min_y, 10.0)
        self.assertAlmostEqual(bbox.max_z - bbox.min_z, 10.0)

    async def test_varying_cell_sizes(self):
        """Test tensor grid with varying cell sizes."""
        data = Tensor3DGridData(
            name="Varying Tensor Grid",
            origin=Point3(10, 20, 30),
            size=Size3i(3, 3, 3),
            cell_sizes_x=np.array([1.0, 2.0, 4.0]),  # Total: 7
            cell_sizes_y=np.array([0.5, 1.0, 1.5]),  # Total: 3
            cell_sizes_z=np.array([10.0, 20.0, 30.0]),  # Total: 60
        )

        with self._mock_geoscience_objects():
            result = await Tensor3DGrid.create(context=self.context, data=data)

        self.assertEqual(result.origin, Point3(10, 20, 30))
        bbox = result.bounding_box
        self.assertAlmostEqual(bbox.min_x, 10.0)
        self.assertAlmostEqual(bbox.max_x, 17.0)
        self.assertAlmostEqual(bbox.min_y, 20.0)
        self.assertAlmostEqual(bbox.max_y, 23.0)
        self.assertAlmostEqual(bbox.min_z, 30.0)
        self.assertAlmostEqual(bbox.max_z, 90.0)
