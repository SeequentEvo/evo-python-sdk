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

from evo.common import Environment, EvoContext
from evo.common.test_tools import BASE_URL, ORG, WORKSPACE_ID, TestWithConnector
from evo.objects import ObjectReference
from evo.objects.typed import Point3, RegularMasked3DGrid, RegularMasked3DGridData, Rotation, Size3d, Size3i
from evo.objects.typed.exceptions import ObjectValidationError

from .helpers import MockClient


class TestRegularMaskedGrid(TestWithConnector):
    def setUp(self) -> None:
        TestWithConnector.setUp(self)
        self.environment = Environment(hub_url=BASE_URL, org_id=ORG.id, workspace_id=WORKSPACE_ID)
        self.context = EvoContext.from_environment(
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
            patch("evo.objects.typed.regular_masked_grid.get_data_client", lambda _: mock_client),
        ):
            yield mock_client

    example_mask = np.array([True, False, True, False, True] * 100, dtype=bool)
    example_grid = RegularMasked3DGridData(
        name="Test Masked Grid",
        origin=Point3(0, 0, 0),
        size=Size3i(10, 10, 5),
        cell_size=Size3d(2.5, 5, 5),
        mask=example_mask,
        cell_data=pd.DataFrame(
            {
                "value": np.random.rand(np.sum(example_mask)),
                "cat": pd.Categorical(
                    np.random.choice(range(4), size=np.sum(example_mask)), categories=["a", "b", "c", "d"]
                ),
            }
        ),
        rotation=Rotation(90, 0, 0),
    )

    async def test_create(self):
        with self._mock_geoscience_objects():
            result = await RegularMasked3DGrid.create(evo_context=self.context, data=self.example_grid)

        self.assertEqual(result.name, "Test Masked Grid")
        self.assertEqual(result.origin, Point3(0, 0, 0))
        self.assertEqual(result.size, Size3i(10, 10, 5))
        self.assertEqual(result.cell_size, Size3d(2.5, 5, 5))
        self.assertEqual(result.rotation, Rotation(90, 0, 0))
        self.assertEqual(result.cells.number_active, np.sum(self.example_mask))

        cell_df = await result.cells.as_dataframe()
        pd.testing.assert_frame_equal(cell_df, self.example_grid.cell_data)

    async def test_create_with_no_cell_data(self):
        data = dataclasses.replace(self.example_grid, cell_data=None)
        with self._mock_geoscience_objects():
            result = await RegularMasked3DGrid.create(evo_context=self.context, data=data)

        self.assertEqual(result.name, "Test Masked Grid")
        self.assertEqual(result.cells.number_active, np.sum(self.example_mask))

        cell_df = await result.cells.as_dataframe()
        self.assertEqual(cell_df.shape[0], 0)  # No cell data

    async def test_replace(self):
        data = dataclasses.replace(self.example_grid)
        with self._mock_geoscience_objects():
            result = await RegularMasked3DGrid.replace(
                evo_context=self.context,
                reference=ObjectReference.new(
                    environment=self.context.get_environment(),
                    object_id=uuid.uuid4(),
                ),
                data=data,
            )

        self.assertEqual(result.name, "Test Masked Grid")
        self.assertEqual(result.origin, Point3(0, 0, 0))
        self.assertEqual(result.size, Size3i(10, 10, 5))
        self.assertEqual(result.cell_size, Size3d(2.5, 5, 5))
        self.assertEqual(result.cells.number_active, np.sum(self.example_mask))

        cell_df = await result.cells.as_dataframe()
        pd.testing.assert_frame_equal(cell_df, data.cell_data)

    async def test_from_reference(self):
        with self._mock_geoscience_objects():
            original = await RegularMasked3DGrid.create(evo_context=self.context, data=self.example_grid)

            result = await RegularMasked3DGrid.from_reference(evo_context=self.context, reference=original.metadata.url)

            self.assertEqual(result.name, "Test Masked Grid")
            self.assertEqual(result.origin, Point3(0, 0, 0))
            self.assertEqual(result.size, Size3i(10, 10, 5))
            self.assertEqual(result.cell_size, Size3d(2.5, 5, 5))
            self.assertEqual(result.rotation, Rotation(90, 0, 0))
            self.assertEqual(result.cells.number_active, np.sum(self.example_mask))

            cell_df = await result.cells.as_dataframe()
            pd.testing.assert_frame_equal(cell_df, self.example_grid.cell_data)

    async def test_update_with_new_mask(self):
        with self._mock_geoscience_objects():
            obj = await RegularMasked3DGrid.create(evo_context=self.context, data=self.example_grid)

            # Create a new mask with different number of active cells
            new_mask = np.array([True, True, False, False, True] * 100, dtype=bool)
            new_active_count = np.sum(new_mask)

            obj.name = "Updated Masked Grid"
            await obj.cells.set_dataframe(
                pd.DataFrame(
                    {
                        "value": np.ones(new_active_count),
                    }
                ),
                mask=new_mask,
            )

            await obj.update()

            self.assertEqual(obj.name, "Updated Masked Grid")
            self.assertEqual(obj.cells.number_active, new_active_count)

            cell_df = await obj.cells.as_dataframe()
            pd.testing.assert_frame_equal(
                cell_df,
                pd.DataFrame(
                    {
                        "value": np.ones(new_active_count),
                    }
                ),
            )

    async def test_update_without_new_mask(self):
        with self._mock_geoscience_objects():
            obj = await RegularMasked3DGrid.create(evo_context=self.context, data=self.example_grid)

            original_active_count = obj.cells.number_active

            # Update data without changing mask
            await obj.cells.set_dataframe(
                pd.DataFrame(
                    {
                        "value": np.ones(original_active_count),
                    }
                )
            )

            await obj.update()

            self.assertEqual(obj.cells.number_active, original_active_count)

            cell_df = await obj.cells.as_dataframe()
            pd.testing.assert_frame_equal(
                cell_df,
                pd.DataFrame(
                    {
                        "value": np.ones(original_active_count),
                    }
                ),
            )

    async def test_mask_size_validation(self):
        # Mask too small
        with self.assertRaises(ObjectValidationError):
            RegularMasked3DGridData(
                name="Bad Grid",
                origin=Point3(0, 0, 0),
                size=Size3i(10, 10, 5),
                cell_size=Size3d(2.5, 5, 5),
                mask=np.array([True, False] * 100, dtype=bool),  # Only 200 elements
            )

        # Mask too large
        with self.assertRaises(ObjectValidationError):
            RegularMasked3DGridData(
                name="Bad Grid",
                origin=Point3(0, 0, 0),
                size=Size3i(10, 10, 5),
                cell_size=Size3d(2.5, 5, 5),
                mask=np.array([True, False] * 300, dtype=bool),  # 600 elements
            )

    async def test_cell_data_size_validation(self):
        mask = np.array([True, False, True] * 167, dtype=bool)  # 501 elements, 334 True

        # Cell data doesn't match active cells
        with self.assertRaises(ObjectValidationError):
            RegularMasked3DGridData(
                name="Bad Grid",
                origin=Point3(0, 0, 0),
                size=Size3i(10, 10, 5),  # 500 cells
                cell_size=Size3d(2.5, 5, 5),
                mask=mask[:500],  # First 500 elements
                cell_data=pd.DataFrame(
                    {
                        "value": np.random.rand(300),  # Wrong size
                    }
                ),
            )

    async def test_set_dataframe_wrong_size(self):
        with self._mock_geoscience_objects():
            obj = await RegularMasked3DGrid.create(evo_context=self.context, data=self.example_grid)

            # Try to set dataframe with wrong size (no new mask)
            with self.assertRaises(ObjectValidationError):
                await obj.cells.set_dataframe(
                    pd.DataFrame(
                        {
                            "value": np.random.rand(100),  # Wrong size
                        }
                    )
                )

            # Try to set dataframe with new mask but wrong data size
            new_mask = np.array([True] * 250 + [False] * 250, dtype=bool)
            with self.assertRaises(ObjectValidationError):
                await obj.cells.set_dataframe(
                    pd.DataFrame(
                        {
                            "value": np.random.rand(100),  # Should be 250
                        }
                    ),
                    mask=new_mask,
                )

    async def test_set_dataframe_wrong_mask_size(self):
        with self._mock_geoscience_objects():
            obj = await RegularMasked3DGrid.create(evo_context=self.context, data=self.example_grid)

            # Try to set new mask with wrong size
            bad_mask = np.array([True, False] * 100, dtype=bool)  # 200 elements instead of 500
            with self.assertRaises(ObjectValidationError):
                await obj.cells.set_dataframe(
                    pd.DataFrame(
                        {
                            "value": np.ones(100),
                        }
                    ),
                    mask=bad_mask,
                )

    async def test_bounding_box(self):
        with self._mock_geoscience_objects() as mock_client:
            obj = await RegularMasked3DGrid.create(evo_context=self.context, data=self.example_grid)

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

    async def test_all_active_mask(self):
        """Test grid where all cells are active."""
        all_active_mask = np.ones(500, dtype=bool)
        data = RegularMasked3DGridData(
            name="All Active Grid",
            origin=Point3(0, 0, 0),
            size=Size3i(10, 10, 5),
            cell_size=Size3d(2.5, 5, 5),
            mask=all_active_mask,
            cell_data=pd.DataFrame(
                {
                    "value": np.random.rand(500),
                }
            ),
        )

        with self._mock_geoscience_objects():
            result = await RegularMasked3DGrid.create(evo_context=self.context, data=data)

        self.assertEqual(result.cells.number_active, 500)
        cell_df = await result.cells.as_dataframe()
        self.assertEqual(cell_df.shape[0], 500)

    async def test_all_inactive_mask(self):
        """Test grid where all cells are inactive."""
        all_inactive_mask = np.zeros(500, dtype=bool)
        data = RegularMasked3DGridData(
            name="All Inactive Grid",
            origin=Point3(0, 0, 0),
            size=Size3i(10, 10, 5),
            cell_size=Size3d(2.5, 5, 5),
            mask=all_inactive_mask,
            cell_data=None,
        )

        with self._mock_geoscience_objects():
            result = await RegularMasked3DGrid.create(evo_context=self.context, data=data)

        self.assertEqual(result.cells.number_active, 0)
        cell_df = await result.cells.as_dataframe()
        self.assertEqual(cell_df.shape[0], 0)
