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
import copy
import dataclasses
import uuid
from unittest import TestCase
from unittest.mock import Mock, patch

import numpy as np
import numpy.testing as npt
import pandas as pd
from parameterized import parameterized
from pydantic import TypeAdapter

from evo.common import Environment, EvoContext
from evo.common.test_tools import BASE_URL, ORG, WORKSPACE_ID, TestWithConnector
from evo.objects import ObjectReference, ObjectSchema
from evo.objects.typed import (
    BoundingBox,
    CoordinateReferenceSystem,
    EpsgCode,
    Point3,
    Regular3DGrid,
    Regular3DGridData,
    Rotation,
    Size3d,
    Size3i,
)
from evo.objects.typed.grid import SizeChangeError

test_environment = Environment(hub_url=BASE_URL, org_id=ORG.id, workspace_id=WORKSPACE_ID)


class TestTypes(TestCase):
    @parameterized.expand(
        [
            (0, 0, 0, [2, 5, 2.5]),
            (90, 0, 0, [5, -2, 2.5]),
            (0, 90, 0, [2, 2.5, -5]),
            (0, 0, 90, [5, -2, 2.5]),
            (124, 63.5, 22.1, [1.2020506, -5.31502659, -2.35702497]),
        ]
    )
    def test_rotation_matrix(self, dip_azimuth, dip, pitch, expected):
        rotation = Rotation(dip_azimuth, dip, pitch)
        matrix = rotation.as_rotation_matrix()
        npt.assert_array_almost_equal(matrix @ np.array([2, 5, 2.5]), expected)

    def test_bounding_box(self):
        box = BoundingBox.from_points(np.array([[0, -1, 5], [1, 2, 4]]))
        self.assertEqual(box.min, Point3(0, -1, 4))
        self.assertEqual(box.max, Point3(1, 2, 5))
        self.assertEqual(box.min_x, 0)
        self.assertEqual(box.max_x, 1)
        self.assertEqual(box.min_y, -1)
        self.assertEqual(box.max_y, 2)
        self.assertEqual(box.min_z, 4)
        self.assertEqual(box.max_z, 5)

        box = BoundingBox.from_points([0, 1], [-1, 2], [4, 5])
        self.assertEqual(box.min, Point3(0, -1, 4))
        self.assertEqual(box.max, Point3(1, 2, 5))

    def test_crs(self):
        type_adapter = TypeAdapter(CoordinateReferenceSystem)
        crs1 = type_adapter.validate_python({"epsg_code": 4326})
        self.assertEqual(crs1, EpsgCode(4326))

        crs2 = type_adapter.validate_python({"ogc_wkt": "WKT_STRING"})
        self.assertEqual(crs2, "WKT_STRING")

        crs3 = type_adapter.validate_python("unspecified")
        self.assertIsNone(crs3)

        self.assertEqual(type_adapter.dump_python(crs1), {"epsg_code": 4326})
        self.assertEqual(type_adapter.dump_python(crs2), {"ogc_wkt": "WKT_STRING"})
        self.assertEqual(type_adapter.dump_python(crs3), "unspecified")


class MockDownloadedObject:
    def __init__(self, mock_client: MockClient, object_dict: dict, version_id: str = "1"):
        self.mock_client = mock_client
        self.object_dict = object_dict
        self.metadata = Mock()
        self.metadata.schema_id = ObjectSchema.from_id(object_dict["schema"])
        self.metadata.url = ObjectReference.new(
            environment=mock_client.environment,
            object_id=uuid.UUID(object_dict["uuid"]),
        )
        self.metadata.version_id = version_id

    def as_dict(self):
        return self.object_dict

    async def download_attribute_dataframe(self, data: dict, fb) -> pd.DataFrame:
        return self.mock_client.get_dataframe(data)

    async def update(self, object_dict):
        new_version_id = str(int(self.metadata.version_id) + 1)
        return MockDownloadedObject(self.mock_client, object_dict, new_version_id)


class MockClient:
    def __init__(self, environment: Environment):
        self.environment = environment
        self.data = {}
        self.objects = {}

    def get_dataframe(self, data: dict) -> pd.DataFrame:
        return self.data[data["data_id"]]

    async def upload_dataframe(self, df: pd.DataFrame, *args, **kwargs) -> dict:
        data_id = str(uuid.uuid4())
        self.data[data_id] = df
        return {"data_id": data_id}

    upload_category_dataframe = upload_dataframe

    async def create_geoscience_object(self, evo_context: EvoContext, object_dict: dict, parent: str):
        object_dict = object_dict.copy()
        object_dict["uuid"] = str(uuid.uuid4())
        self.objects[object_dict["uuid"]] = copy.deepcopy(object_dict)
        return MockDownloadedObject(self, object_dict)

    async def replace_geoscience_object(self, evo_context: EvoContext, reference: ObjectReference, object_dict: dict):
        object_dict = object_dict.copy()
        assert reference.object_id is not None, "Reference must have an object ID"
        object_dict["uuid"] = str(reference.object_id)
        self.objects[object_dict["uuid"]] = copy.deepcopy(object_dict)
        return MockDownloadedObject(self, object_dict)

    async def from_reference(self, evo_context: EvoContext, reference: ObjectReference):
        assert reference.object_id is not None, "Reference must have an object ID"
        object_dict = copy.deepcopy(self.objects[str(reference.object_id)])
        return MockDownloadedObject(self, object_dict)


class TestRegularGrid(TestWithConnector):
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
            patch("evo.objects.typed._store.get_data_client", lambda _: mock_client),
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

    async def test_create(self):
        with self._mock_geoscience_objects():
            result = await Regular3DGrid.create(evo_context=self.context, data=self.example_grid)
        self.assertEqual(result.name, "Test Grid")
        self.assertEqual(result.origin, Point3(0, 0, 0))
        self.assertEqual(result.size, Size3i(10, 10, 5))
        self.assertEqual(result.cell_size, Size3d(2.5, 5, 5))
        self.assertEqual(result.rotation, Rotation(90, 0, 0))

        cell_df = await result.cells.as_dataframe()
        pd.testing.assert_frame_equal(cell_df, self.example_grid.cell_data)
        vertices_df = await result.vertices.as_dataframe()
        pd.testing.assert_frame_equal(vertices_df, self.example_grid.vertex_data)

    async def test_replace(self):
        data = dataclasses.replace(self.example_grid, vertex_data=None)
        with self._mock_geoscience_objects():
            result = await Regular3DGrid.replace(
                evo_context=self.context,
                reference=ObjectReference.new(
                    environment=self.context.get_environment(),
                    object_id=uuid.uuid4(),
                ),
                data=data,
            )
        self.assertEqual(result.name, "Test Grid")
        self.assertEqual(result.origin, Point3(0, 0, 0))
        self.assertEqual(result.size, Size3i(10, 10, 5))
        self.assertEqual(result.cell_size, Size3d(2.5, 5, 5))
        self.assertEqual(result.rotation, Rotation(90, 0, 0))

        cell_df = await result.cells.as_dataframe()
        pd.testing.assert_frame_equal(cell_df, data.cell_data)
        vertices_df = await result.vertices.as_dataframe()
        self.assertEqual(vertices_df.shape[0], 0)  # No vertex data provided

    async def test_from_reference(self):
        with self._mock_geoscience_objects():
            original = await Regular3DGrid.create(evo_context=self.context, data=self.example_grid)

            result = await Regular3DGrid.from_reference(evo_context=self.context, reference=original.metadata.url)
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
            obj = await Regular3DGrid.create(evo_context=self.context, data=self.example_grid)

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

            with self.assertRaises(ValueError):
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
            obj = await Regular3DGrid.create(evo_context=self.context, data=self.example_grid)
            with self.assertRaises(SizeChangeError):
                await obj.cells.set_dataframe(
                    pd.DataFrame(
                        {
                            "value": np.random.rand(11 * 10 * 5),
                        }
                    )
                )

            with self.assertRaises(SizeChangeError):
                obj.size = Size3i(5, 10, 6)

            # Have to clear the attributes to change the size
            obj.cells.clear_attributes()
            obj.vertices.clear_attributes()

            obj.size = Size3i(5, 10, 6)
            self.assertEqual(obj.size, Size3i(5, 10, 6))

    async def test_bounding_box(self):
        with self._mock_geoscience_objects() as mock_client:
            obj = await Regular3DGrid.create(evo_context=self.context, data=self.example_grid)

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
