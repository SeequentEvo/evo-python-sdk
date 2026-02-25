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
    RegularBlockModelData,
    Size3d,
    Size3i,
)
from evo.objects.typed.block_model_ref import (
    BlockModelAttributes,
    BlockModelPendingAttribute,
    _parse_attributes,
    _parse_geometry,
    _serialize_attributes,
    _serialize_geometry,
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


class TestBlockModelAttributeTarget(TestCase):
    """Tests for BlockModelAttribute and BlockModelPendingAttribute target functionality."""

    def test_existing_attribute_exists_property(self):
        """Test that existing attributes have exists=True."""
        attr = BlockModelAttribute(
            name="grade",
            attribute_type="Float64",
        )
        self.assertTrue(attr.exists)

    def test_pending_attribute_exists_property(self):
        """Test that pending attributes have exists=False."""
        pending = BlockModelPendingAttribute(None, "new_attribute")

        self.assertFalse(pending.exists)

    def test_pending_attribute_repr(self):
        """Test the string representation of BlockModelPendingAttribute."""
        from evo.objects.typed.block_model_ref import BlockModelPendingAttribute

        pending = BlockModelPendingAttribute(None, "new_attribute")
        repr_str = repr(pending)

        self.assertIn("BlockModelPendingAttribute", repr_str)
        self.assertIn("new_attribute", repr_str)
        self.assertIn("exists=False", repr_str)

    def test_attributes_getitem_returns_pending_for_missing(self):
        """Test that accessing a non-existent attribute returns PendingAttribute."""
        existing_attrs = [
            BlockModelAttribute(name="grade", attribute_type="Float64"),
        ]
        attrs = BlockModelAttributes(existing_attrs, block_model=None)

        # Accessing existing attribute returns BlockModelAttribute
        existing = attrs["grade"]
        self.assertIsInstance(existing, BlockModelAttribute)
        self.assertTrue(existing.exists)

        # Accessing non-existent attribute returns BlockModelPendingAttribute
        pending = attrs["new_attribute"]
        self.assertIsInstance(pending, BlockModelPendingAttribute)
        self.assertFalse(pending.exists)

    def test_attributes_getitem_by_index(self):
        """Test that accessing attributes by index works correctly."""
        from evo.objects.typed.block_model_ref import BlockModelAttributes

        existing_attrs = [
            BlockModelAttribute(name="grade", attribute_type="Float64"),
            BlockModelAttribute(name="density", attribute_type="Float32"),
        ]
        attrs = BlockModelAttributes(existing_attrs, block_model=None)

        self.assertEqual(attrs[0].name, "grade")
        self.assertEqual(attrs[1].name, "density")

    def test_attribute_has_obj_reference(self):
        """Test that attributes have _obj reference to the parent BlockModel."""
        from evo.objects.typed.block_model_ref import BlockModelAttributes

        # Create a mock block model (using None for simplicity in unit tests)
        mock_block_model = "mock_block_model"  # In real use, this would be a BlockModel instance

        existing_attrs = [
            BlockModelAttribute(name="grade", attribute_type="Float64"),
        ]
        attrs = BlockModelAttributes(existing_attrs, block_model=mock_block_model)

        # The attribute should have _obj reference to the block model
        attr = attrs["grade"]
        self.assertEqual(attr._obj, mock_block_model)

    def test_pending_attribute_has_obj_reference(self):
        """Test that pending attributes have _obj reference to the parent BlockModel."""
        from evo.objects.typed.block_model_ref import BlockModelAttributes, BlockModelPendingAttribute

        # Create a mock block model
        mock_block_model = "mock_block_model"

        attrs = BlockModelAttributes([], block_model=mock_block_model)

        # Accessing non-existent attribute returns BlockModelPendingAttribute with _obj set
        pending = attrs["new_attribute"]
        self.assertIsInstance(pending, BlockModelPendingAttribute)
        self.assertEqual(pending._obj, mock_block_model)


class TestBlockModelOptionalDependency(TestCase):
    """Tests verifying that BlockModel metadata-only operations work regardless of
    evo-blockmodels availability, and that data operations correctly use evo-blockmodels."""

    def test_geometry_always_available(self):
        """Test that geometry parsing works without data operations."""
        geometry_dict = {
            "model_type": "regular",
            "origin": [100.0, 200.0, 300.0],
            "n_blocks": [10, 20, 30],
            "block_size": [1.0, 2.0, 3.0],
        }

        geom = _parse_geometry(geometry_dict)
        self.assertEqual(geom.model_type, "regular")
        self.assertEqual(geom.origin.x, 100.0)

    def test_attributes_always_available(self):
        """Test that attribute parsing works without data operations."""
        attrs_list = [
            {"name": "grade", "attribute_type": "Float64"},
        ]

        attrs = _parse_attributes(attrs_list)
        self.assertEqual(len(attrs), 1)
        self.assertEqual(attrs[0].name, "grade")

    def test_block_model_data_compute_bounding_box(self):
        """Test that bounding box computation works without blockmodels."""
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
        self.assertEqual(bbox.max_x, 110.0)

    def test_blockmodels_available(self):
        """Test that _BLOCKMODELS_AVAILABLE is True when blockmodels IS installed."""
        from evo.objects.typed.block_model_ref import _BLOCKMODELS_AVAILABLE

        # In test environment, blockmodels should be available
        self.assertTrue(_BLOCKMODELS_AVAILABLE)

    def test_regular_block_model_data_importable(self):
        """Test that RegularBlockModelData is importable from evo.objects.typed."""
        data = RegularBlockModelData(
            name="Test",
            origin=Point3(0, 0, 0),
            n_blocks=Size3i(10, 10, 10),
            block_size=Size3d(1.0, 1.0, 1.0),
        )
        self.assertEqual(data.name, "Test")

    def test_report_types_importable_when_blockmodels_available(self):
        """Test that report types are importable from evo.objects.typed when blockmodels installed."""
        from evo.objects.typed.block_model_ref import _BLOCKMODELS_AVAILABLE

        if _BLOCKMODELS_AVAILABLE:
            from evo.objects.typed import Report, ReportSpecificationData

            self.assertIsNotNone(Report)
            self.assertIsNotNone(ReportSpecificationData)

    def test_common_types_from_evo_common(self):
        """Test that Point3, Size3d, Size3i are available from evo.common.typed."""
        from evo.common.typed import BoundingBox as CommonBBox
        from evo.common.typed import Point3 as CommonPoint3
        from evo.common.typed import Size3d as CommonSize3d
        from evo.common.typed import Size3i as CommonSize3i

        p = CommonPoint3(1.0, 2.0, 3.0)
        self.assertEqual(p.x, 1.0)

        s = CommonSize3i(10, 20, 30)
        self.assertEqual(s.total_size, 6000)

        sd = CommonSize3d(1.0, 2.0, 3.0)
        self.assertEqual(sd.dx, 1.0)

        bbox = CommonBBox.from_origin_and_size(p, s, sd)
        self.assertEqual(bbox.x_min, 1.0)
        self.assertEqual(bbox.x_max, 11.0)
