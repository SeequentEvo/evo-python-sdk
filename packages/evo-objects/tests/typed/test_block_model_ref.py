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

import uuid
from unittest import TestCase

from evo.objects.typed import (
    BlockModelAttribute,
    BlockModelData,
    BlockModelGeometry,
    Point3,
    Size3d,
    Size3i,
)
from evo.objects.typed.block_model_ref import (
    _parse_geometry,
    _serialize_geometry,
    _parse_attributes,
    _serialize_attributes,
)


class TestBlockModelGeometry(TestCase):
    def test_create_geometry(self):
        """Test creating a BlockModelGeometry."""
        geom = BlockModelGeometry(
            model_type="regular",
            origin=Point3(100.0, 200.0, 300.0),
            n_blocks=Size3i(10, 20, 30),
            block_size=Size3d(1.0, 2.0, 3.0),
        )

        self.assertEqual(geom.model_type, "regular")
        self.assertEqual(geom.origin, Point3(100.0, 200.0, 300.0))
        self.assertEqual(geom.n_blocks, Size3i(10, 20, 30))
        self.assertEqual(geom.block_size, Size3d(1.0, 2.0, 3.0))
        self.assertIsNone(geom.rotation)

    def test_create_geometry_with_rotation(self):
        """Test creating a BlockModelGeometry with rotation."""
        geom = BlockModelGeometry(
            model_type="regular",
            origin=Point3(0, 0, 0),
            n_blocks=Size3i(10, 10, 10),
            block_size=Size3d(1.0, 1.0, 1.0),
            rotation=(45.0, 30.0, 15.0),
        )

        self.assertEqual(geom.rotation, (45.0, 30.0, 15.0))


class TestBlockModelAttribute(TestCase):
    def test_create_attribute(self):
        """Test creating a BlockModelAttribute."""
        attr = BlockModelAttribute(
            name="grade",
            attribute_type="Float64",
            block_model_column_uuid=uuid.uuid4(),
            unit="g/t",
        )

        self.assertEqual(attr.name, "grade")
        self.assertEqual(attr.attribute_type, "Float64")
        self.assertIsNotNone(attr.block_model_column_uuid)
        self.assertEqual(attr.unit, "g/t")

    def test_create_attribute_minimal(self):
        """Test creating a BlockModelAttribute with minimal parameters."""
        attr = BlockModelAttribute(
            name="density",
            attribute_type="Float32",
        )

        self.assertEqual(attr.name, "density")
        self.assertEqual(attr.attribute_type, "Float32")
        self.assertIsNone(attr.block_model_column_uuid)
        self.assertIsNone(attr.unit)


class TestBlockModelData(TestCase):
    def test_create_data(self):
        """Test creating BlockModelData."""
        geom = BlockModelGeometry(
            model_type="regular",
            origin=Point3(0, 0, 0),
            n_blocks=Size3i(10, 10, 10),
            block_size=Size3d(1.0, 1.0, 1.0),
        )
        bm_uuid = uuid.uuid4()

        data = BlockModelData(
            name="Test Block Model",
            block_model_uuid=bm_uuid,
            geometry=geom,
        )

        self.assertEqual(data.name, "Test Block Model")
        self.assertEqual(data.block_model_uuid, bm_uuid)
        self.assertEqual(data.geometry, geom)
        self.assertIsNone(data.block_model_version_uuid)
        self.assertEqual(data.attributes, [])

    def test_compute_bounding_box(self):
        """Test computing bounding box from geometry."""
        geom = BlockModelGeometry(
            model_type="regular",
            origin=Point3(100.0, 200.0, 300.0),
            n_blocks=Size3i(10, 20, 30),
            block_size=Size3d(1.0, 2.0, 3.0),
        )

        data = BlockModelData(
            name="Test",
            block_model_uuid=uuid.uuid4(),
            geometry=geom,
        )

        bbox = data.compute_bounding_box()

        self.assertEqual(bbox.min_x, 100.0)
        self.assertEqual(bbox.max_x, 110.0)  # 100 + 10 * 1
        self.assertEqual(bbox.min_y, 200.0)
        self.assertEqual(bbox.max_y, 240.0)  # 200 + 20 * 2
        self.assertEqual(bbox.min_z, 300.0)
        self.assertEqual(bbox.max_z, 390.0)  # 300 + 30 * 3


class TestGeometrySerialization(TestCase):
    def test_parse_geometry(self):
        """Test parsing geometry from dictionary."""
        geometry_dict = {
            "model_type": "regular",
            "origin": [1.0, 2.0, 3.0],
            "n_blocks": [10, 20, 30],
            "block_size": [1.5, 2.5, 3.5],
        }

        geom = _parse_geometry(geometry_dict)

        self.assertEqual(geom.model_type, "regular")
        self.assertEqual(geom.origin, Point3(1.0, 2.0, 3.0))
        self.assertEqual(geom.n_blocks, Size3i(10, 20, 30))
        self.assertEqual(geom.block_size, Size3d(1.5, 2.5, 3.5))
        self.assertIsNone(geom.rotation)

    def test_parse_geometry_with_rotation(self):
        """Test parsing geometry with rotation."""
        geometry_dict = {
            "model_type": "regular",
            "origin": [0, 0, 0],
            "n_blocks": [10, 10, 10],
            "block_size": [1, 1, 1],
            "rotation": {
                "dip_azimuth": 45.0,
                "dip": 30.0,
                "pitch": 15.0,
            },
        }

        geom = _parse_geometry(geometry_dict)

        self.assertEqual(geom.rotation, (45.0, 30.0, 15.0))

    def test_serialize_geometry(self):
        """Test serializing geometry to dictionary."""
        geom = BlockModelGeometry(
            model_type="regular",
            origin=Point3(1.0, 2.0, 3.0),
            n_blocks=Size3i(10, 20, 30),
            block_size=Size3d(1.5, 2.5, 3.5),
        )

        result = _serialize_geometry(geom)

        self.assertEqual(result["model_type"], "regular")
        self.assertEqual(result["origin"], [1.0, 2.0, 3.0])
        self.assertEqual(result["n_blocks"], [10, 20, 30])
        self.assertEqual(result["block_size"], [1.5, 2.5, 3.5])
        self.assertNotIn("rotation", result)

    def test_serialize_geometry_with_rotation(self):
        """Test serializing geometry with rotation."""
        geom = BlockModelGeometry(
            model_type="regular",
            origin=Point3(0, 0, 0),
            n_blocks=Size3i(10, 10, 10),
            block_size=Size3d(1, 1, 1),
            rotation=(45.0, 30.0, 15.0),
        )

        result = _serialize_geometry(geom)

        self.assertIn("rotation", result)
        self.assertEqual(result["rotation"]["dip_azimuth"], 45.0)
        self.assertEqual(result["rotation"]["dip"], 30.0)
        self.assertEqual(result["rotation"]["pitch"], 15.0)

    def test_round_trip_geometry(self):
        """Test round-trip serialization of geometry."""
        original = BlockModelGeometry(
            model_type="regular",
            origin=Point3(100.0, 200.0, 300.0),
            n_blocks=Size3i(10, 20, 30),
            block_size=Size3d(1.5, 2.5, 3.5),
            rotation=(45.0, 30.0, 15.0),
        )

        serialized = _serialize_geometry(original)
        parsed = _parse_geometry(serialized)

        self.assertEqual(original.model_type, parsed.model_type)
        self.assertEqual(original.origin, parsed.origin)
        self.assertEqual(original.n_blocks, parsed.n_blocks)
        self.assertEqual(original.block_size, parsed.block_size)
        self.assertEqual(original.rotation, parsed.rotation)


class TestAttributeSerialization(TestCase):
    def test_parse_attributes(self):
        """Test parsing attributes from list of dictionaries."""
        attrs_list = [
            {
                "name": "grade",
                "attribute_type": "Float64",
                "block_model_column_uuid": "12345678-1234-5678-1234-567812345678",
                "unit": "g/t",
            },
            {
                "name": "density",
                "attribute_type": "Float32",
            },
        ]

        attrs = _parse_attributes(attrs_list)

        self.assertEqual(len(attrs), 2)
        self.assertEqual(attrs[0].name, "grade")
        self.assertEqual(attrs[0].attribute_type, "Float64")
        self.assertEqual(attrs[0].unit, "g/t")
        self.assertIsNotNone(attrs[0].block_model_column_uuid)
        self.assertEqual(attrs[1].name, "density")
        self.assertIsNone(attrs[1].block_model_column_uuid)

    def test_parse_attributes_with_invalid_uuid(self):
        """Test parsing attributes handles invalid UUID strings gracefully."""
        attrs_list = [
            {
                "name": "grade",
                "attribute_type": "Float64",
                "block_model_column_uuid": "i",  # Invalid - geometry column ID
            },
            {
                "name": "x_coord",
                "attribute_type": "Float64",
                "block_model_column_uuid": "x",  # Invalid - geometry column ID
            },
            {
                "name": "valid_attr",
                "attribute_type": "Float64",
                "block_model_column_uuid": "12345678-1234-5678-1234-567812345678",  # Valid UUID
            },
        ]

        attrs = _parse_attributes(attrs_list)

        self.assertEqual(len(attrs), 3)
        # Invalid UUIDs should result in None
        self.assertIsNone(attrs[0].block_model_column_uuid)
        self.assertIsNone(attrs[1].block_model_column_uuid)
        # Valid UUID should be parsed
        self.assertIsNotNone(attrs[2].block_model_column_uuid)
        self.assertEqual(str(attrs[2].block_model_column_uuid), "12345678-1234-5678-1234-567812345678")

    def test_parse_attributes_with_none_uuid(self):
        """Test parsing attributes with None UUID."""
        attrs_list = [
            {
                "name": "test",
                "attribute_type": "Float64",
                "block_model_column_uuid": None,
            },
        ]

        attrs = _parse_attributes(attrs_list)

        self.assertEqual(len(attrs), 1)
        self.assertIsNone(attrs[0].block_model_column_uuid)

    def test_parse_attributes_missing_uuid(self):
        """Test parsing attributes without UUID field."""
        attrs_list = [
            {
                "name": "test",
                "attribute_type": "Float64",
            },
        ]

        attrs = _parse_attributes(attrs_list)

        self.assertEqual(len(attrs), 1)
        self.assertIsNone(attrs[0].block_model_column_uuid)

    def test_serialize_attributes(self):
        """Test serializing attributes to list of dictionaries."""
        col_uuid = uuid.uuid4()
        attrs = [
            BlockModelAttribute(
                name="grade",
                attribute_type="Float64",
                block_model_column_uuid=col_uuid,
                unit="g/t",
            ),
            BlockModelAttribute(
                name="density",
                attribute_type="Float32",
            ),
        ]

        result = _serialize_attributes(attrs)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["name"], "grade")
        self.assertEqual(result[0]["attribute_type"], "Float64")
        self.assertEqual(result[0]["unit"], "g/t")
        self.assertEqual(result[0]["block_model_column_uuid"], str(col_uuid))
        self.assertEqual(result[1]["name"], "density")
        self.assertNotIn("unit", result[1])
        self.assertNotIn("block_model_column_uuid", result[1])

