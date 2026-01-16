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
import numpy.testing as npt
import pandas as pd

from evo.common import Environment, StaticContext
from evo.common.test_tools import BASE_URL, ORG, WORKSPACE_ID, TestWithConnector
from evo.objects import ObjectReference
from evo.objects.typed import Point3, Rotation, Size3i, Tensor3DGrid, Tensor3DGridData
from evo.objects.typed.exceptions import ObjectValidationError

from .helpers import MockClient


class TestTensor3DGrid(TestWithConnector):
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

    example_grid = Tensor3DGridData(
        name="Test Tensor Grid",
        origin=Point3(0, 0, 0),
        size=Size3i(3, 3, 3),
        cell_sizes_x=np.array([1.0, 2.0, 1.5]),
        cell_sizes_y=np.array([2.0, 2.5, 1.0]),
        cell_sizes_z=np.array([0.5, 1.0, 1.5]),
        cell_data=pd.DataFrame(
            {
                "value": np.random.rand(27),
                "cat": pd.Categorical(np.random.choice(range(4), size=27), ["a", "b", "c", "d"]),
            }
        ),
        vertex_data=pd.DataFrame(
            {
                "elevation": np.random.rand(4 * 4 * 4),
            }
        ),
        rotation=Rotation(45, 0, 0),
    )

    async def test_create(self):
        with self._mock_geoscience_objects():
            result = await Tensor3DGrid.create(context=self.context, data=self.example_grid)

        self.assertEqual(result.name, "Test Tensor Grid")
        self.assertEqual(result.origin, Point3(0, 0, 0))
        self.assertEqual(result.size, Size3i(3, 3, 3))
        self.assertEqual(result.rotation, Rotation(45, 0, 0))

        # Check cell sizes
        npt.assert_array_equal(result.cell_sizes_x, np.array([1.0, 2.0, 1.5]))
        npt.assert_array_equal(result.cell_sizes_y, np.array([2.0, 2.5, 1.0]))
        npt.assert_array_equal(result.cell_sizes_z, np.array([0.5, 1.0, 1.5]))

        cell_df = await result.cells.to_dataframe()
        pd.testing.assert_frame_equal(cell_df, self.example_grid.cell_data)
        vertices_df = await result.vertices.to_dataframe()
        pd.testing.assert_frame_equal(vertices_df, self.example_grid.vertex_data)

    async def test_create_without_cell_data(self):
        data = dataclasses.replace(self.example_grid, cell_data=None, vertex_data=None)
        with self._mock_geoscience_objects():
            result = await Tensor3DGrid.create(context=self.context, data=data)

        self.assertEqual(result.name, "Test Tensor Grid")
        cell_df = await result.cells.to_dataframe()
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
        self.assertEqual(result.size, Size3i(3, 3, 3))

        cell_df = await result.cells.to_dataframe()
        pd.testing.assert_frame_equal(cell_df, data.cell_data)

    async def test_from_reference(self):
        with self._mock_geoscience_objects():
            original = await Tensor3DGrid.create(context=self.context, data=self.example_grid)

            result = await Tensor3DGrid.from_reference(context=self.context, reference=original.metadata.url)

            self.assertEqual(result.name, "Test Tensor Grid")
            self.assertEqual(result.origin, Point3(0, 0, 0))
            self.assertEqual(result.size, Size3i(3, 3, 3))
            self.assertEqual(result.rotation, Rotation(45, 0, 0))

            npt.assert_array_equal(result.cell_sizes_x, np.array([1.0, 2.0, 1.5]))
            npt.assert_array_equal(result.cell_sizes_y, np.array([2.0, 2.5, 1.0]))
            npt.assert_array_equal(result.cell_sizes_z, np.array([0.5, 1.0, 1.5]))

            cell_df = await result.cells.to_dataframe()
            pd.testing.assert_frame_equal(cell_df, self.example_grid.cell_data)

    async def test_update(self):
        with self._mock_geoscience_objects():
            obj = await Tensor3DGrid.create(context=self.context, data=self.example_grid)

            obj.name = "Updated Tensor Grid"
            await obj.cells.set_dataframe(
                pd.DataFrame(
                    {
                        "value": np.ones(27),
                    }
                )
            )

            await obj.update()

            self.assertEqual(obj.name, "Updated Tensor Grid")

            cell_df = await obj.cells.to_dataframe()
            pd.testing.assert_frame_equal(
                cell_df,
                pd.DataFrame(
                    {
                        "value": np.ones(27),
                    }
                ),
            )

    async def test_cell_sizes_validation(self):
        # Test wrong number of x cell sizes
        with self.assertRaises(ObjectValidationError):
            Tensor3DGridData(
                name="Bad Grid",
                origin=Point3(0, 0, 0),
                size=Size3i(3, 3, 3),
                cell_sizes_x=np.array([1.0, 2.0]),  # Wrong size - should be 3
                cell_sizes_y=np.array([2.0, 2.5, 1.0]),
                cell_sizes_z=np.array([0.5, 1.0, 1.5]),
            )

        # Test wrong number of y cell sizes
        with self.assertRaises(ObjectValidationError):
            Tensor3DGridData(
                name="Bad Grid",
                origin=Point3(0, 0, 0),
                size=Size3i(3, 3, 3),
                cell_sizes_x=np.array([1.0, 2.0, 1.5]),
                cell_sizes_y=np.array([2.0, 2.5, 1.0, 3.0]),  # Wrong size - should be 3
                cell_sizes_z=np.array([0.5, 1.0, 1.5]),
            )

        # Test wrong number of z cell sizes
        with self.assertRaises(ObjectValidationError):
            Tensor3DGridData(
                name="Bad Grid",
                origin=Point3(0, 0, 0),
                size=Size3i(3, 3, 3),
                cell_sizes_x=np.array([1.0, 2.0, 1.5]),
                cell_sizes_y=np.array([2.0, 2.5, 1.0]),
                cell_sizes_z=np.array([0.5]),  # Wrong size - should be 3
            )

    async def test_positive_cell_sizes_validation(self):
        # Test negative x cell size
        with self.assertRaises(ObjectValidationError):
            Tensor3DGridData(
                name="Bad Grid",
                origin=Point3(0, 0, 0),
                size=Size3i(3, 3, 3),
                cell_sizes_x=np.array([1.0, -2.0, 1.5]),
                cell_sizes_y=np.array([2.0, 2.5, 1.0]),
                cell_sizes_z=np.array([0.5, 1.0, 1.5]),
            )

        # Test zero y cell size
        with self.assertRaises(ObjectValidationError):
            Tensor3DGridData(
                name="Bad Grid",
                origin=Point3(0, 0, 0),
                size=Size3i(3, 3, 3),
                cell_sizes_x=np.array([1.0, 2.0, 1.5]),
                cell_sizes_y=np.array([2.0, 0.0, 1.0]),
                cell_sizes_z=np.array([0.5, 1.0, 1.5]),
            )

    async def test_cell_data_size_validation(self):
        # Test cell data with wrong number of rows
        with self.assertRaises(ObjectValidationError):
            Tensor3DGridData(
                name="Bad Grid",
                origin=Point3(0, 0, 0),
                size=Size3i(3, 3, 3),
                cell_sizes_x=np.array([1.0, 2.0, 1.5]),
                cell_sizes_y=np.array([2.0, 2.5, 1.0]),
                cell_sizes_z=np.array([0.5, 1.0, 1.5]),
                cell_data=pd.DataFrame({"value": np.random.rand(20)}),  # Should be 27
            )

    async def test_vertex_data_size_validation(self):
        # Test vertex data with wrong number of rows
        with self.assertRaises(ObjectValidationError):
            Tensor3DGridData(
                name="Bad Grid",
                origin=Point3(0, 0, 0),
                size=Size3i(3, 3, 3),
                cell_sizes_x=np.array([1.0, 2.0, 1.5]),
                cell_sizes_y=np.array([2.0, 2.5, 1.0]),
                cell_sizes_z=np.array([0.5, 1.0, 1.5]),
                vertex_data=pd.DataFrame({"elevation": np.random.rand(50)}),  # Should be 64
            )

    async def test_bounding_box(self):
        with self._mock_geoscience_objects() as mock_client:
            obj = await Tensor3DGrid.create(context=self.context, data=self.example_grid)

            bbox = obj.bounding_box
            # The bbox will be rotated, so we just check it exists and has reasonable values
            self.assertIsNotNone(bbox)
            self.assertTrue(bbox.max_x > bbox.min_x)
            self.assertTrue(bbox.max_y > bbox.min_y)
            self.assertTrue(bbox.max_z > bbox.min_z)

            # Check that bounding box is stored in the object
            bbox_from_mock = mock_client.objects[str(obj.metadata.url.object_id)]["bounding_box"]
            self.assertIsNotNone(bbox_from_mock)

    async def test_bounding_box_no_rotation(self):
        # Test without rotation for easier verification
        data = Tensor3DGridData(
            name="Simple Tensor Grid",
            origin=Point3(10, 20, 30),
            size=Size3i(2, 2, 2),
            cell_sizes_x=np.array([5.0, 10.0]),
            cell_sizes_y=np.array([3.0, 7.0]),
            cell_sizes_z=np.array([2.0, 4.0]),
        )

        with self._mock_geoscience_objects():
            obj = await Tensor3DGrid.create(context=self.context, data=data)

            bbox = obj.bounding_box
            # Origin at (10, 20, 30)
            # Extents: x=15, y=10, z=6
            self.assertAlmostEqual(bbox.min_x, 10.0)
            self.assertAlmostEqual(bbox.max_x, 25.0)  # 10 + 15
            self.assertAlmostEqual(bbox.min_y, 20.0)
            self.assertAlmostEqual(bbox.max_y, 30.0)  # 20 + 10
            self.assertAlmostEqual(bbox.min_z, 30.0)
            self.assertAlmostEqual(bbox.max_z, 36.0)  # 30 + 6
