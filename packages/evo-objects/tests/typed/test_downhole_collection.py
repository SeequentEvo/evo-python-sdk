#  Copyright © 2026 Bentley Systems, Incorporated
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
import math
import uuid
from datetime import date
from unittest.mock import patch

import numpy as np
import numpy.testing as npt
import pandas as pd
from parameterized import parameterized

from evo.common import Environment, StaticContext
from evo.common.test_tools import BASE_URL, ORG, WORKSPACE_ID, TestWithConnector
from evo.objects import ObjectReference
from evo.objects.typed import BoundingBox
from evo.objects.typed.base import BaseObject
from evo.objects.typed.downhole_collection import (
    DistanceCollection,
    DownholeCollection,
    DownholeCollectionData,
)
from evo.objects.typed.exceptions import ObjectValidationError

from .helpers import MockClient


def _make_example_data(
    name: str = "Test DHC",
    description: str | None = None,
    tags: dict[str, str] | None = None,
    attributes: pd.DataFrame | None = None,
    collections: list[DistanceCollection] | None = None,
) -> DownholeCollectionData:
    """Helper to build a simple two-hole DownholeCollectionData."""
    # Concatenated path table: hole 0 has 4 rows, hole 1 has 3 rows
    path = pd.DataFrame(
        {
            "distance": [0.0, 10.0, 20.0, 30.0, 0.0, 15.0, 30.0],
            "azimuth": [0.0, 0.0, 0.0, 0.0, 90.0, 90.0, 90.0],
            "dip": [90.0, 90.0, 90.0, 90.0, 45.0, 45.0, 45.0],
        }
    )

    collection1 = DistanceCollection(
        name="collection1",
        collection_type="distance",
        holes=pd.DataFrame(
            {
                "hole_index": [0],
                "offset": [0],
                "count": [4],
            }
        ),
        distance_table=pd.DataFrame(
            {
                "distance": [0.0, 10.0, 20.0, 30.0],
                "attr_str": ["a", "b", "a", "c"],
                "attr_dt": [date(2000, 1, 1), date(2000, 1, 2), date(2000, 1, 3), date(2000, 1, 4)],
                "attr_num": [1.1, 2.2, 3.3, 4.4],
            }
        ),
    )

    holes = pd.DataFrame(
        {
            "hole_index": [0, 1],
            "offset": [0, 4],
            "count": [4, 3],
        }
    )

    properties = pd.DataFrame(
        {
            "hole_id": ["H001", "H002"],
            "x": [100.0, 200.0],
            "y": [150.0, 300.0],
            "z": [0.0, 50.0],
            "final": [30.0, 30.0],
            "target": [25.0, 25.0],
            "current": [30.0, 28.0],
        }
    )

    if collections is None:
        collections = [collection1]

    return DownholeCollectionData(
        name=name,
        path=path,
        holes=holes,
        properties=properties,
        attributes=attributes,
        collections=collections,
        distance_unit="m",
        desurvey="trench",
        description=description,
        tags=tags,
    )


class TestDownholeCollection(TestWithConnector):
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
            patch("evo.objects.typed._utils.get_data_client", lambda _: mock_client),
            patch("evo.objects.typed.base.create_geoscience_object", mock_client.create_geoscience_object),
            patch("evo.objects.typed.base.replace_geoscience_object", mock_client.replace_geoscience_object),
            patch("evo.objects.DownloadedObject.from_context", mock_client.from_reference),
        ):
            yield mock_client

    def _assert_bounding_box_equal(
        self, bbox: BoundingBox, min_x: float, max_x: float, min_y: float, max_y: float, min_z: float, max_z: float
    ):
        self.assertAlmostEqual(bbox.min_x, min_x, places=3)
        self.assertAlmostEqual(bbox.max_x, max_x, places=3)
        self.assertAlmostEqual(bbox.min_y, min_y, places=3)
        self.assertAlmostEqual(bbox.max_y, max_y, places=3)
        self.assertAlmostEqual(bbox.min_z, min_z, places=3)
        self.assertAlmostEqual(bbox.max_z, max_z, places=3)

    async def _check_locations(self, expected: DownholeCollectionData, result: DownholeCollection):
        loc = result.location
        xyz = ["x", "y", "z"]
        distances = ["final", "target", "current"]

        npt.assert_array_equal(expected.properties[xyz], await loc.coordinates.to_dataframe())
        npt.assert_array_equal(expected.properties[distances], await loc.distances.to_dataframe())
        npt.assert_array_equal(expected.properties[["hole_id"]], await loc.hole_id.to_dataframe())
        npt.assert_array_equal(expected.holes, await loc.holes.to_dataframe())
        if expected.attributes:
            npt.assert_array_equal(expected.attributes, await result.location.hole_id.to_dataframe())

    async def _check_path(self, expected: DownholeCollectionData, result: DownholeCollection):
        loc = result.location
        path_columns = ["distance", "azimuth", "dip"]
        attr_columns = [col for col in expected.path.columns if col not in path_columns]

        npt.assert_array_equal(expected.path[path_columns], await loc.path.to_dataframe())
        if attr_columns:
            npt.assert_array_equal(expected.path[attr_columns], await loc.attributes.to_dataframe())

    async def _check_collections(self, expected: DownholeCollectionData, result: DownholeCollection):
        for expected_collection, result_collection in zip(expected.collections, result.collections, strict=True):
            expected_distance_unit = expected_collection.distance_table.attrs.get("attribute_descriptions", {}).get(
                "distance"
            )
            self.assertEqual(expected_distance_unit, result_collection.distance.unit)

            expected_table = expected_collection.distance_table
            result_table = await result_collection.distance.to_dataframe()

            for col in result_table.columns:
                if pd.api.types.is_datetime64_any_dtype(result_table[col]):
                    for x, y in zip(expected_table[col], result_table[col]):
                        self.assertEqual(x.year, y.year)
                        self.assertEqual(x.month, y.month)
                        self.assertEqual(x.day, y.day)
                else:
                    npt.assert_array_equal(expected_table[col], result_table[col])

    async def _check_dhc(self, expected: DownholeCollectionData, result: DownholeCollection):
        self.assertIsInstance(result, DownholeCollection)
        self.assertEqual(expected.name, result.name)
        self.assertEqual(expected.distance_unit, result.distance_unit)
        self.assertEqual(expected.desurvey, result.desurvey)

        await self._check_locations(expected, result)
        await self._check_path(expected, result)
        await self._check_collections(expected, result)

    @parameterized.expand([BaseObject, DownholeCollection])
    async def test_create(self, class_to_call):
        """Includes collections and attributes"""
        data = _make_example_data()
        with self._mock_geoscience_objects():
            result = await class_to_call.create(context=self.context, data=data)
        await self._check_dhc(data, result)

    async def test_create_with_empty_collections(self):
        data = _make_example_data(collections=[])
        with self._mock_geoscience_objects():
            result = await DownholeCollection.create(context=self.context, data=data)
        self.assertIsInstance(result, DownholeCollection)

    @parameterized.expand([BaseObject, DownholeCollection])
    async def test_replace(self, class_to_call):
        data = _make_example_data()
        with self._mock_geoscience_objects():
            result = await class_to_call.replace(
                context=self.context,
                reference=ObjectReference.new(
                    environment=self.context.get_environment(),
                    object_id=uuid.uuid4(),
                ),
                data=data,
            )
        await self._check_dhc(data, result)

    @parameterized.expand([BaseObject, DownholeCollection])
    async def test_create_or_replace(self, class_to_call):
        data = _make_example_data()
        with self._mock_geoscience_objects():
            result = await class_to_call.create_or_replace(
                context=self.context,
                reference=ObjectReference.new(
                    environment=self.context.get_environment(),
                    object_id=uuid.uuid4(),
                ),
                data=data,
            )
        await self._check_dhc(data, result)

    @parameterized.expand([BaseObject, DownholeCollection])
    async def test_from_reference(self, class_to_call):
        data = _make_example_data()
        with self._mock_geoscience_objects():
            original = await DownholeCollection.create(context=self.context, data=data)
            result = await class_to_call.from_reference(context=self.context, reference=original.metadata.url)
        await self._check_dhc(data, result)

    def test_bounding_box(self):
        """Two vertical holes (dip=90deg) go straight down: bbox should reflect collar + depth. Azimuth doesn't matter"""

        path = pd.DataFrame(
            {
                "distance": [0.0, 10.0, 20.0, 30.0, 0.0, 15.0, 30.0],
                "azimuth": [0.0, 45.0, 20.0, 0.0, 10.0, 90.0, 90.0],
                "dip": [90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0],
            }
        )

        data = _make_example_data()
        data = dataclasses.replace(data, path=path)
        bbox = data.compute_bounding_box()
        self._assert_bounding_box_equal(bbox, 100.0, 200.0, 150.0, 300.0, -30.0, 50.0)

    def test_bounding_box_from_spiral(self):
        # First hole spirals, second hole zig-zags
        path = pd.DataFrame(
            {
                "distance": [0.0, 10.0, 20.0, 50.0, 0.0, 20.0, 40.0],
                "azimuth": [0.0, 90.0, 180.0, 270.0, 0.0, 315.0, 90.0],
                "dip": [60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0],
            }
        )

        data = _make_example_data()
        data = dataclasses.replace(data, path=path)
        bbox = data.compute_bounding_box()

        # Expected geometry, based on having spiraled and zig-zagged with 30/60/90 and 45/45/90 dips/azimuths
        xmin = 100.0 - 10
        xmax = 200.0 - 10 / math.sqrt(2) + 10
        ymin = 150.0 - 5
        ymax = 300.0 + 10 / math.sqrt(2)
        zmin = (-50.0 / 2) * math.sqrt(3)
        zmax = 50.0

        self._assert_bounding_box_equal(bbox, xmin, xmax, ymin, ymax, zmin, zmax)

    def test_bounding_box_with_nans(self):
        """Azimuth nans -> 0.0, dip nans -> 90.0"""
        path = pd.DataFrame(
            {
                "distance": [0.0, 10.0, 20.0, 50.0, 0.0, 20.0, 40.0],
                "azimuth": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                "dip": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            }
        )

        data = _make_example_data()
        data = dataclasses.replace(data, path=path)
        bbox = data.compute_bounding_box()

        self._assert_bounding_box_equal(bbox, 100.0, 200.0, 150.0, 300.0, -50.0, 50.0)

    def test_compute_bounding_box_np_unsorted_depths_raises(self):
        with self.assertRaises(ObjectValidationError):
            DownholeCollectionData._compute_bounding_box_np(
                depths=np.array([10.0, 5.0, 20.0]),
                dips=np.array([90.0, 90.0, 90.0]),
                azimuths=np.array([0.0, 0.0, 0.0]),
            )

    def test_compute_bounding_box_np_length_mismatch_raises(self):
        with self.assertRaises(ObjectValidationError):
            DownholeCollectionData._compute_bounding_box_np(
                depths=np.array([0.0, 10.0]),
                dips=np.array([90.0]),
                azimuths=np.array([0.0, 0.0]),
            )

    async def test_description_and_tags(self):
        data = _make_example_data(
            description="A test downhole collection",
            tags={"site": "alpha", "status": "active"},
        )
        with self._mock_geoscience_objects():
            result = await DownholeCollection.create(context=self.context, data=data)
        self.assertEqual(result.description, "A test downhole collection")
        self.assertEqual(result.tags, {"site": "alpha", "status": "active"})

    def test_attributes_length_raises(self):
        """attributes length must match holes length."""
        path = pd.DataFrame({"distance": [0.0, 10.0], "azimuth": [0.0, 0.0], "dip": [90.0, 90.0]})
        holes = pd.DataFrame({"hole_index": [0, 1], "offset": [0, 1], "count": [1, 1]})
        properties = pd.DataFrame(
            {
                "hole_id": ["H1", "H2"],
                "x": [0.0, 1.0],
                "y": [0.0, 1.0],
                "z": [0.0, 0.0],
                "final": [10.0, 10.0],
                "target": [10.0, 10.0],
                "current": [10.0, 10.0],
            }
        )
        # attributes has 3 rows, but holes has 2 - should assert
        bad_attributes = pd.DataFrame({"a": [1, 2, 3]})
        with self.assertRaises(ObjectValidationError):
            DownholeCollectionData(
                name="Bad",
                path=path,
                holes=holes,
                properties=properties,
                attributes=bad_attributes,
                collections=[],
                distance_unit=None,
                desurvey=None,
            )

    async def test_update_dataframe_after_creation(self):
        """Test updating the path DataFrame after downhole collection creation."""
        with self._mock_geoscience_objects():
            data = _make_example_data()
            obj = await DownholeCollection.create(context=self.context, data=data)

            new_path = pd.DataFrame(
                {
                    "distance": [0.0, 10.0, 20.0, 50.0, 0.0, 20.0, 40.0],
                    "azimuth": [0.0, 90.0, 180.0, 270.0, 0.0, 315.0, 90.0],
                    "dip": [60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0],
                }
            )
            await obj.location.path.from_dataframe(new_path)

            # Verify the data was updated
            await obj.update()
            expected = dataclasses.replace(data, path=new_path)
            await self._check_dhc(expected, obj)

    async def test_json(self):
        data = _make_example_data()
        with self._mock_geoscience_objects() as mock_client:
            obj = await DownholeCollection.create(context=self.context, data=data)
            object_json = mock_client.objects[str(obj.metadata.url.object_id)]

            # Verify schema
            self.assertIn("/objects/downhole-collection/", object_json["schema"])

            # Verify base properties
            self.assertEqual(object_json["name"], "Test DHC")
            self.assertIn("uuid", object_json)
            self.assertIn("bounding_box", object_json)
            self.assertEqual(object_json["coordinate_reference_system"], "unspecified")

            # Verify DHC top level properties
            self.assertEqual(object_json["type"], "downhole")
            self.assertIn("distance_unit", object_json)
            self.assertIn("desurvey", object_json)

            # Verify location structure
            self.assertIn("location", object_json)
            location = object_json["location"]
            self.assertIn("path", location)
            self.assertIn("holes", location)
            self.assertIn("coordinates", location)
            self.assertIn("distances", location)
            self.assertIn("hole_id", location)
            self.assertIn("collections", object_json)
            collection = object_json["collections"][0]
            self.assertIn("name", collection)
            self.assertIn("collection_type", collection)
            self.assertEqual(collection["collection_type"], "distance")
            self.assertIn("holes", collection)
            self.assertIn("distance", collection)
