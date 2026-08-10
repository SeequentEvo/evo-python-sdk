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
from evo.objects.typed.attributes import AttributeDescription
from evo.objects.typed.base import BaseObject
from evo.objects.typed.downhole_collection import (
    DistanceCollection,
    DownholeCollection,
    DownholeCollectionData,
    IntervalCollection,
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
        table=pd.DataFrame(
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
            expected_distance_unit = expected_collection.table.attrs.get("attribute_descriptions", {}).get("distance")
            self.assertEqual(expected_distance_unit, result_collection.distance.unit)

            expected_table = expected_collection.table
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

    async def test_mixed_collections_round_trip_and_mutation(self):
        interval = IntervalCollection(
            name="intervals",
            holes=pd.DataFrame({"hole_index": [0], "offset": [0], "count": [2]}),
            table=pd.DataFrame({"from": [0.0, 1.0], "to": [1.0, 2.0], "lithology": ["a", "b"]}),
            unit="m",
        )
        data = _make_example_data(collections=[_make_example_data().collections[0], interval])
        with self._mock_geoscience_objects() as mock_client:
            result = await DownholeCollection.create(context=self.context, data=data)
            self.assertEqual(result.collections.names(), ["collection1", "intervals"])
            self.assertEqual(result.collections.get("intervals").from_to.unit, "m")
            self.assertListEqual(
                (await result.collections.get("intervals").to_dataframe()).columns.tolist(), ["from", "to", "lithology"]
            )
            self.assertEqual(result.collections.remove("missing", "collection1"), 1)
            await result.collections.add(data.collections[0])
            self.assertEqual(result.collections.names(), ["intervals", "collection1"])
            object_json = mock_client.objects[str(result.metadata.url.object_id)]
            self.assertIn("start_and_end", object_json["collections"][1]["from_to"]["intervals"])

    async def test_interval_only_collection_uses_attribute_unit_when_not_explicit(self):
        table = pd.DataFrame({"from": [0.0, 1.0], "to": [1.0, 2.0], "grade": [1.0, 2.0]})
        table.attrs["attribute_descriptions"] = {"from": AttributeDescription(unit="ft")}
        interval = IntervalCollection(
            name="intervals",
            holes=pd.DataFrame({"hole_index": [0], "offset": [0], "count": [2]}),
            table=table,
        )
        data = _make_example_data(collections=[interval])
        with self._mock_geoscience_objects():
            result = await DownholeCollection.create(context=self.context, data=data)
        collection = result.collections.get("intervals")
        self.assertIsNotNone(collection)
        self.assertEqual(collection.from_to.unit, "ft")
        self.assertListEqual((await collection.to_dataframe()).columns.tolist(), ["from", "to", "grade"])

    @parameterized.expand(
        [
            ("explicit", "m", "ft", "m"),
            ("metadata", None, "ft", "ft"),
            ("omitted", None, None, None),
        ]
    )
    async def test_interval_collection_unit_precedence(self, _name, explicit_unit, metadata_unit, expected_unit):
        table = pd.DataFrame({"from": [0.0], "to": [1.0]})
        if metadata_unit is not None:
            table.attrs["attribute_descriptions"] = {"from": AttributeDescription(unit=metadata_unit)}
        collection = IntervalCollection(
            name="intervals",
            holes=pd.DataFrame({"hole_index": [0], "offset": [0], "count": [1]}),
            table=table,
            unit=explicit_unit,
        )
        with self._mock_geoscience_objects() as mock_client:
            result = await DownholeCollection.create(
                context=self.context, data=_make_example_data(collections=[collection])
            )
            document = mock_client.objects[str(result.metadata.url.object_id)]
        from_to = document["collections"][0]["from_to"]
        self.assertEqual(result.collections.get("intervals").from_to.unit, expected_unit)
        if expected_unit is None:
            self.assertNotIn("unit", from_to)
        else:
            self.assertEqual(from_to["unit"], expected_unit)

    async def test_interval_collection_round_trips_all_attribute_types(self):
        table = pd.DataFrame(
            {
                "from": [0.0, 1.0],
                "to": [1.0, 2.0],
                "scalar": pd.Series([1.5, 2.5], dtype="float64"),
                "integer": pd.Series([1, 2], dtype="int64"),
                "boolean": pd.Series([True, False], dtype="bool"),
                "string": pd.Series(["a", "b"], dtype="string"),
                "category": pd.Series(pd.Categorical(["ore", "waste"])),
                "date_time": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            }
        )
        collection = IntervalCollection(
            name="intervals",
            holes=pd.DataFrame({"hole_index": [0], "offset": [0], "count": [2]}),
            table=table,
        )
        expected_types = {
            "scalar": "scalar",
            "integer": "integer",
            "boolean": "bool",
            "string": "string",
            "category": "category",
            "date_time": "date_time",
        }
        with self._mock_geoscience_objects() as mock_client:
            result = await DownholeCollection.create(
                context=self.context, data=_make_example_data(collections=[collection])
            )
            document = mock_client.objects[str(result.metadata.url.object_id)]
        interval = result.collections.get("intervals")
        self.assertIsNotNone(interval)
        self.assertEqual(
            {
                attribute["name"]: attribute["attribute_type"]
                for attribute in document["collections"][0]["from_to"]["attributes"]
            },
            expected_types,
        )
        self.assertListEqual((await interval.to_dataframe()).columns.tolist(), list(table.columns))

    @parameterized.expand([("missing_from", {"to": [1.0]}), ("missing_to", {"from": [0.0]})])
    async def test_interval_collection_requires_from_and_to_columns(self, _name, table_data):
        collection = IntervalCollection(
            name="invalid",
            holes=pd.DataFrame({"hole_index": [0], "offset": [0], "count": [1]}),
            table=pd.DataFrame(table_data),
        )
        with self._mock_geoscience_objects(), self.assertRaises(ObjectValidationError):
            await DownholeCollection.create(context=self.context, data=_make_example_data(collections=[collection]))

    async def test_explicit_collection_unit_overrides_dataframe_metadata(self):
        table = pd.DataFrame({"distance": [0.0], "grade": [1.0]})
        table.attrs["attribute_descriptions"] = {"distance": AttributeDescription(unit="ft")}
        collection = DistanceCollection(
            name="distances",
            holes=pd.DataFrame({"hole_index": [0], "offset": [0], "count": [1]}),
            table=table,
            unit="m",
        )
        with self._mock_geoscience_objects():
            result = await DownholeCollection.create(
                context=self.context, data=_make_example_data(collections=[collection])
            )
        self.assertEqual(result.collections.get("distances").distance.unit, "m")

    async def test_attribute_descriptions_round_trip_to_dataframe_metadata(self):
        table = pd.DataFrame({"distance": [0.0], "grade": [1.0]})
        description = AttributeDescription(
            discipline="geology",
            type="grade",
            unit="ppm",
            scale="linear",
            tags={"source": "assay"},
        )
        table.attrs["attribute_descriptions"] = {"grade": description}
        collection = DistanceCollection(
            name="grades",
            holes=pd.DataFrame({"hole_index": [0], "offset": [0], "count": [1]}),
            table=table,
        )
        with self._mock_geoscience_objects():
            result = await DownholeCollection.create(
                context=self.context, data=_make_example_data(collections=[collection])
            )
        round_tripped = await result.collections.get("grades").to_dataframe()
        self.assertEqual(round_tripped.attrs["attribute_descriptions"]["grade"], description)

    async def test_attribute_without_description_reads_as_none(self):
        with self._mock_geoscience_objects():
            result = await DownholeCollection.create(context=self.context, data=_make_example_data())
        self.assertIsNone(result.collections.get("collection1").distance.attributes["attr_str"].attribute_description)

    async def test_none_optional_fields_are_omitted(self):
        data = _make_example_data()
        data = dataclasses.replace(data, distance_unit=None, desurvey=None)
        with self._mock_geoscience_objects() as mock_client:
            result = await DownholeCollection.create(context=self.context, data=data)
            document = mock_client.objects[str(result.metadata.url.object_id)]
        self.assertNotIn("distance_unit", document)
        self.assertNotIn("desurvey", document)

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

    def test_bounding_box_uses_hole_index_not_property_position(self):
        data = _make_example_data()
        expected = data.compute_bounding_box()
        properties = data.properties.iloc[[1, 0]].copy()
        properties.index = [10, 20]
        data = dataclasses.replace(data, properties=properties)
        bbox = data.compute_bounding_box()
        self._assert_bounding_box_equal(
            bbox, expected.min_x, expected.max_x, expected.min_y, expected.max_y, expected.min_z, expected.max_z
        )

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

    def test_location_chunks_require_each_creation_code_and_exact_path_coverage(self):
        base = _make_example_data(collections=[])
        for holes in (
            pd.DataFrame({"hole_index": [0], "offset": [0], "count": [7]}),
            pd.DataFrame({"hole_index": [0, 0], "offset": [0, 4], "count": [4, 3]}),
            pd.DataFrame({"hole_index": [0, 1], "offset": [0, 5], "count": [4, 2]}),
            pd.DataFrame({"hole_index": [0, 1], "offset": [0, 3], "count": [4, 3]}),
        ):
            with self.subTest(holes=holes.to_dict("records")), self.assertRaises(ObjectValidationError):
                dataclasses.replace(base, holes=holes)

    def test_collection_chunk_ranges_are_not_required_to_partition_the_table(self):
        base = _make_example_data(collections=[])
        table = pd.DataFrame({"distance": [0.0, 1.0, 2.0]})
        collection = DistanceCollection(
            name="repeated",
            holes=pd.DataFrame({"hole_index": [0, 0], "offset": [0, 2], "count": [1, 1]}),
            table=table,
        )
        dataclasses.replace(base, collections=[collection])

        with self.assertRaises(ObjectValidationError):
            dataclasses.replace(
                base,
                collections=[
                    DistanceCollection(
                        name="invalid",
                        holes=pd.DataFrame({"hole_index": [2], "offset": [0], "count": [2]}),
                        table=table,
                    )
                ],
            )

    async def test_collection_allows_zero_row_hole(self):
        base = _make_example_data(collections=[])
        collection = DistanceCollection(
            name="sparse",
            holes=pd.DataFrame({"hole_index": [0, 1], "offset": [0, 0], "count": [2, 0]}),
            table=pd.DataFrame({"distance": [0.0, 1.0]}),
        )
        data = dataclasses.replace(base, collections=[collection])
        with self._mock_geoscience_objects():
            result = await DownholeCollection.create(context=self.context, data=data)
        by_hole = await result.collections.get("sparse").to_dataframe_by_hole()
        self.assertEqual({hole_id: len(table) for hole_id, table in by_hole.items()}, {"H001": 2, "H002": 0})

    async def test_collection_add_allows_repeated_hole_chunks_and_replaces_in_place(self):
        with self._mock_geoscience_objects():
            result = await DownholeCollection.create(context=self.context, data=_make_example_data(collections=[]))
            collection = DistanceCollection(
                name="measurements",
                holes=pd.DataFrame({"hole_index": [0], "offset": [0], "count": [1]}),
                table=pd.DataFrame({"distance": [0.0]}),
            )
            await result.collections.add(collection)
            replacement = dataclasses.replace(collection, table=pd.DataFrame({"distance": [1.0]}))
            with self.assertRaises(ValueError):
                await result.collections.add(replacement)
            await result.collections.add(replacement, replace=True)
            self.assertEqual(result.collections.names(), ["measurements"])
            self.assertEqual((await result.collections.get("measurements").to_dataframe()).iloc[0, 0], 1.0)
            repeated = dataclasses.replace(
                collection,
                holes=pd.DataFrame({"hole_index": [0, 0], "offset": [0, 1], "count": [1, 0]}),
            )
            await result.collections.add(repeated, replace=True)

    async def test_collection_replacement_preserves_position_and_persists_after_update(self):
        first = DistanceCollection(
            name="first",
            holes=pd.DataFrame({"hole_index": [0], "offset": [0], "count": [1]}),
            table=pd.DataFrame({"distance": [0.0]}),
        )
        middle = DistanceCollection(
            name="middle",
            holes=pd.DataFrame({"hole_index": [1], "offset": [0], "count": [1]}),
            table=pd.DataFrame({"distance": [1.0]}),
        )
        last = DistanceCollection(
            name="last",
            holes=pd.DataFrame({"hole_index": [0], "offset": [0], "count": [1]}),
            table=pd.DataFrame({"distance": [2.0]}),
        )
        replacement = dataclasses.replace(middle, table=pd.DataFrame({"distance": [10.0]}))
        with self._mock_geoscience_objects():
            result = await DownholeCollection.create(context=self.context, data=_make_example_data(collections=[]))
            await result.collections.add(first)
            await result.collections.add(middle)
            await result.collections.add(last)
            await result.collections.add(replacement, replace=True)
            self.assertEqual(result.collections.names(), ["first", "middle", "last"])
            await result.update()
            persisted = await DownholeCollection.from_reference(context=self.context, reference=result.metadata.url)
        self.assertEqual(persisted.collections.names(), ["first", "middle", "last"])
        self.assertEqual((await persisted.collections.get("middle").to_dataframe()).iloc[0, 0], 10.0)

    async def test_location_path_and_collection_reads(self):
        with self._mock_geoscience_objects():
            result = await DownholeCollection.create(context=self.context, data=_make_example_data())
        collars = await result.location.to_dataframe()
        self.assertListEqual(collars.columns.tolist(), ["hole_id", "x", "y", "z", "final", "target", "current"])
        self.assertListEqual(
            (await result.location.path_to_dataframe()).columns.tolist(), ["distance", "azimuth", "dip"]
        )
        by_hole = await result.collections.get("collection1").to_dataframe_by_hole()
        self.assertListEqual(list(by_hole), ["H001"])
        self.assertEqual(len(by_hole["H001"]), 4)

    async def test_collection_read_uses_persisted_non_contiguous_lookup_keys(self):
        with self._mock_geoscience_objects() as mock_client:
            result = await DownholeCollection.create(context=self.context, data=_make_example_data())
            lookup_info = result.location.hole_id.as_dict()["table"]
            mock_client.data[lookup_info["data"]]["key"] = [10, 20]
            collection = result.collections.get("collection1")
            holes_info = collection.holes.as_dict()
            mock_client.data[holes_info["data"]]["hole_index"] = [10]

            by_hole = await collection.to_dataframe_by_hole()

        self.assertListEqual(list(by_hole), ["H001"])
        self.assertEqual(len(by_hole["H001"]), 4)

    async def test_repeated_collection_chunks_are_concatenated_in_table_order(self):
        collection = DistanceCollection(
            name="repeated",
            holes=pd.DataFrame({"hole_index": [0, 0], "offset": [2, 0], "count": [1, 1]}),
            table=pd.DataFrame({"distance": [0.0, 1.0, 2.0]}),
        )
        with self._mock_geoscience_objects():
            result = await DownholeCollection.create(
                context=self.context,
                data=_make_example_data(collections=[collection]),
            )
            by_hole = await result.collections.get("repeated").to_dataframe_by_hole()

        self.assertListEqual(by_hole["H001"]["distance"].tolist(), [0.0, 2.0])

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
