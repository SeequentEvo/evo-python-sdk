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
import json
import uuid
from unittest import TestCase
from unittest.mock import Mock, patch

import numpy as np
import numpy.testing as npt
import pandas as pd
from parameterized import parameterized
from pydantic import TypeAdapter

from data import load_test_data
from evo.common import Environment, EvoContext
from evo.common.data import RequestMethod
from evo.common.test_tools import BASE_URL, ORG, WORKSPACE_ID, TestWithConnector
from evo.common.utils.version import get_header_metadata
from evo.objects import ObjectReference, ObjectSchema
from evo.objects.client.api_client import ObjectAPIClient
from evo.objects.typed import (
    BoundingBox,
    CoordinateReferenceSystem,
    EpsgCode,
    Point3,
    Regular3DGrid,
    Regular3DGridData,
    RegularMasked3DGrid,
    RegularMasked3DGridData,
    Rotation,
    Size3d,
    Size3i,
)
from evo.objects.typed._utils import create_geoscience_object
from evo.objects.typed.dataset import DataLoaderError
from evo.objects.typed.exceptions import ObjectValidationError

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


class TestCreateGeoscienceObject(TestWithConnector):
    def setUp(self) -> None:
        TestWithConnector.setUp(self)
        self.environment = Environment(hub_url=BASE_URL, org_id=ORG.id, workspace_id=WORKSPACE_ID)
        self.context = EvoContext.from_environment(
            environment=self.environment,
            connector=self.connector,
        )
        self.setup_universal_headers(get_header_metadata(ObjectAPIClient.__module__))

    @property
    def instance_base_path(self) -> str:
        return f"geoscience-object/orgs/{self.environment.org_id}"

    @property
    def base_path(self) -> str:
        return f"{self.instance_base_path}/workspaces/{self.environment.workspace_id}"

    @parameterized.expand(
        [
            (None, "Sample%20pointset.json"),
            ("path/to/parent", "path/to/parent/Sample%20pointset.json"),
            ("path/to/parent/", "path/to/parent/Sample%20pointset.json"),
        ]
    )
    async def test_create_geoscience_object(self, parent: str | None, expected_object_path: str):
        get_object_response = load_test_data("get_object.json")
        new_pointset = {
            "name": "Sample pointset",
            "uuid": None,
            "description": "A sample pointset object",
            "bounding_box": {"min_x": 0.0, "max_x": 0.0, "min_y": 0.0, "max_y": 0.0, "min_z": 0.0, "max_z": 0.0},
            "coordinate_reference_system": {"epsg_code": 2048},
            "locations": {
                "coordinates": {
                    "data": "0000000000000000000000000000000000000000000000000000000000000001",
                    "length": 1,
                    "width": 3,
                    "data_type": "float64",
                }
            },
            "schema": "/objects/pointset/1.0.1/pointset.schema.json",
        }
        new_pointset_without_uuid = new_pointset.copy()
        with self.transport.set_http_response(status_code=201, content=json.dumps(get_object_response)):
            await create_geoscience_object(self.context, new_pointset, parent=parent)

        self.assert_request_made(
            method=RequestMethod.POST,
            path=f"{self.base_path}/objects/path/{expected_object_path}",
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            body=new_pointset_without_uuid,
        )


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
        return self.mock_client.get_dataframe(data["values"])

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
        return {"data_id": data_id, "length": df.shape[0]}

    async def upload_category_dataframe(self, df: pd.DataFrame, *args, **kwargs) -> dict:
        return {
            "values": await self.upload_dataframe(df),
            "category_data": True,
        }

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
            patch("evo.objects.typed.dataset.get_data_client", lambda _: mock_client),
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
            obj = await Regular3DGrid.create(evo_context=self.context, data=self.example_grid)
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
            patch("evo.objects.typed.grid.get_data_client", lambda _: mock_client),
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
