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

"""Tests for evo.widgets.formatters module."""

import unittest
from datetime import datetime
from unittest.mock import MagicMock
from uuid import UUID

from evo.widgets.formatters import (
    _format_bounding_box,
    _format_crs,
    _get_base_metadata,
    _get_task_result_portal_url,
    format_attributes_collection,
    format_base_object,
    format_block_model,
    format_block_model_attributes,
    format_block_model_version,
    format_report,
    format_report_result,
    format_task_result,
    format_task_results,
    format_variogram,
)


class TestHelperFunctions(unittest.TestCase):
    """Tests for the helper functions."""

    def test_format_bounding_box(self):
        """Test formatting a bounding box as HTML table."""
        bbox = {
            "min_x": 0.0,
            "max_x": 100.5,
            "min_y": 10.0,
            "max_y": 200.75,
            "min_z": -50.0,
            "max_z": 50.25,
        }

        html = _format_bounding_box(bbox)

        self.assertIn("Min", html)
        self.assertIn("Max", html)
        self.assertIn("0.00", html)
        self.assertIn("100.50", html)
        self.assertIn("10.00", html)
        self.assertIn("200.75", html)
        self.assertIn("-50.00", html)
        self.assertIn("50.25", html)

    def test_format_crs_with_epsg_code(self):
        """Test formatting CRS with EPSG code."""
        crs = {"epsg_code": 4326}
        result = _format_crs(crs)
        self.assertEqual(result, "EPSG:4326")

    def test_format_crs_with_string(self):
        """Test formatting CRS as string."""
        crs = "WGS84"
        result = _format_crs(crs)
        self.assertEqual(result, "WGS84")

    def test_format_crs_with_dict_no_epsg(self):
        """Test formatting CRS dict without EPSG code."""
        crs = {"ogc_wkt": "some wkt string"}
        result = _format_crs(crs)
        self.assertIn("ogc_wkt", result)

    def test_get_base_metadata_basic(self):
        """Test extracting base metadata from an object."""
        obj = MagicMock()
        obj.as_dict.return_value = {
            "name": "Test Object",
            "schema": "test-schema",
            "uuid": "12345-abcd",
        }
        obj.metadata = MagicMock()
        obj.metadata.environment = MagicMock()
        obj.metadata.environment.org_id = "org-123"
        obj.metadata.environment.workspace_id = "ws-456"
        obj.metadata.environment.hub_url = "https://test.api.seequent.com"
        obj.metadata.id = "12345-abcd"

        name, title_links, rows = _get_base_metadata(obj)

        self.assertEqual(name, "Test Object")
        self.assertIsNotNone(title_links)
        self.assertEqual(len(title_links), 2)  # Portal and Viewer
        self.assertEqual(rows[0], ("Object ID:", "12345-abcd"))
        self.assertEqual(rows[1], ("Schema:", "test-schema"))

    def test_get_base_metadata_with_extra_links(self):
        """Test extracting base metadata with extra links."""
        obj = MagicMock()
        obj.as_dict.return_value = {
            "name": "Test Object",
            "schema": "test-schema",
            "uuid": "12345-abcd",
        }
        obj.metadata = MagicMock()
        obj.metadata.environment = MagicMock()
        obj.metadata.environment.org_id = "org-123"
        obj.metadata.environment.workspace_id = "ws-456"
        obj.metadata.environment.hub_url = "https://test.api.seequent.com"
        obj.metadata.id = "12345-abcd"

        extra_links = [("BlockSync", "https://blocksync.seequent.com/test")]
        name, title_links, rows = _get_base_metadata(obj, extra_links=extra_links)

        self.assertEqual(len(title_links), 3)  # Portal, Viewer, and BlockSync
        self.assertEqual(title_links[2], ("BlockSync", "https://blocksync.seequent.com/test"))

    def test_get_base_metadata_with_tags(self):
        """Test extracting base metadata with tags."""
        obj = MagicMock()
        obj.as_dict.return_value = {
            "name": "Test Object",
            "schema": "test-schema",
            "uuid": "12345-abcd",
            "tags": {"key1": "value1", "key2": "value2"},
        }
        obj.metadata = MagicMock()
        obj.metadata.environment = MagicMock()
        obj.metadata.environment.org_id = "org-123"
        obj.metadata.environment.workspace_id = "ws-456"
        obj.metadata.environment.hub_url = "https://test.api.seequent.com"
        obj.metadata.id = "12345-abcd"

        name, title_links, rows = _get_base_metadata(obj)

        # Should have 3 rows: Object ID, Schema, Tags
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[2][0], "Tags:")
        self.assertIn("key1", rows[2][1])
        self.assertIn("value1", rows[2][1])


class TestFormatBaseObject(unittest.TestCase):
    """Tests for the format_base_object function."""

    def test_formats_object_with_basic_metadata(self):
        """Test formatting an object with basic metadata."""
        obj = MagicMock()
        obj.as_dict.return_value = {
            "name": "Test Object",
            "schema": "test-schema",
            "uuid": "12345-abcd",
        }
        obj._sub_models = []
        obj.metadata = MagicMock()
        obj.metadata.environment = MagicMock()
        obj.metadata.environment.org_id = "org-123"
        obj.metadata.environment.workspace_id = "ws-456"
        obj.metadata.environment.hub_url = "https://test.api.seequent.com"
        obj.metadata.id = "12345-abcd"

        html = format_base_object(obj)

        self.assertIn("Test Object", html)
        self.assertIn("test-schema", html)
        self.assertIn("12345-abcd", html)
        self.assertIn("Object ID:", html)
        self.assertIn("Schema:", html)

    def test_formats_object_with_bounding_box(self):
        """Test formatting an object that has a bounding box."""
        obj = MagicMock()
        obj.as_dict.return_value = {
            "name": "Test Object",
            "schema": "test-schema",
            "uuid": "12345",
            "bounding_box": {
                "min_x": 0,
                "max_x": 100,
                "min_y": 0,
                "max_y": 200,
                "min_z": 0,
                "max_z": 50,
            },
        }
        obj._sub_models = []
        obj.metadata = MagicMock()
        obj.metadata.environment = MagicMock()
        obj.metadata.environment.org_id = "org-123"
        obj.metadata.environment.workspace_id = "ws-456"
        obj.metadata.environment.hub_url = "https://test.api.seequent.com"
        obj.metadata.id = "12345"

        html = format_base_object(obj)

        self.assertIn("Bounding box:", html)
        self.assertIn("Min", html)
        self.assertIn("Max", html)

    def test_formats_object_with_crs(self):
        """Test formatting an object that has a coordinate reference system."""
        obj = MagicMock()
        obj.as_dict.return_value = {
            "name": "Test Object",
            "schema": "test-schema",
            "uuid": "12345",
            "coordinate_reference_system": {"epsg_code": 4326},
        }
        obj._sub_models = []
        obj.metadata = MagicMock()
        obj.metadata.environment = MagicMock()
        obj.metadata.environment.org_id = "org-123"
        obj.metadata.environment.workspace_id = "ws-456"
        obj.metadata.environment.hub_url = "https://test.api.seequent.com"
        obj.metadata.id = "12345"

        html = format_base_object(obj)

        self.assertIn("CRS:", html)
        self.assertIn("EPSG:4326", html)

    def test_formats_object_with_tags(self):
        """Test formatting an object that has tags."""
        obj = MagicMock()
        obj.as_dict.return_value = {
            "name": "Test Object",
            "schema": "test-schema",
            "uuid": "12345",
            "tags": {"key1": "value1", "key2": "value2"},
        }
        obj._sub_models = []
        obj.metadata = MagicMock()
        obj.metadata.environment = MagicMock()
        obj.metadata.environment.org_id = "org-123"
        obj.metadata.environment.workspace_id = "ws-456"
        obj.metadata.environment.hub_url = "https://test.api.seequent.com"
        obj.metadata.id = "12345"

        html = format_base_object(obj)

        self.assertIn("Tags:", html)
        self.assertIn("key1", html)
        self.assertIn("value1", html)


class TestFormatAttributes(unittest.TestCase):
    """Tests for the format_attributes_collection function (formats Attributes class)."""

    def test_formats_empty_collection(self):
        """Test formatting an empty attributes collection."""
        obj = MagicMock()
        obj.__len__ = MagicMock(return_value=0)

        html = format_attributes_collection(obj)

        self.assertIn("No attributes available", html)

    def test_formats_collection_with_attributes(self):
        """Test formatting a collection with attributes."""
        attr1 = MagicMock()
        attr1.as_dict.return_value = {
            "name": "grade",
            "attribute_type": "scalar",
            "values": {"data_type": "float64"},
        }
        attr2 = MagicMock()
        attr2.as_dict.return_value = {
            "name": "rock_type",
            "attribute_type": "category",
        }

        obj = MagicMock()
        obj.__len__ = MagicMock(return_value=2)
        obj.__iter__ = MagicMock(return_value=iter([attr1, attr2]))

        html = format_attributes_collection(obj)

        self.assertIn("Name", html)
        self.assertIn("Type", html)
        self.assertIn("grade", html)
        self.assertIn("scalar", html)
        self.assertIn("float64", html)
        self.assertIn("rock_type", html)
        self.assertIn("category", html)


class TestFormatVariogram(unittest.TestCase):
    """Tests for the format_variogram function."""

    def _create_mock_variogram(self, **kwargs):
        """Create a mock variogram object with the given properties."""
        obj = MagicMock()

        # Default values
        defaults = {
            "name": "Test Variogram",
            "schema": "objects/variogram/v1.1.0",
            "uuid": "12345-abcd",
            "sill": 1.5,
            "nugget": 0.2,
            "is_rotation_fixed": True,
            "attribute": None,
            "domain": None,
            "modelling_space": None,
            "data_variance": None,
            "structures": [
                {
                    "variogram_type": "spherical",
                    "contribution": 0.8,
                    "anisotropy": {
                        "ellipsoid_ranges": {"major": 100.0, "semi_major": 50.0, "minor": 25.0},
                        "rotation": {"dip": 0.0, "dip_azimuth": 0.0, "pitch": 0.0},
                    },
                }
            ],
            "tags": None,
        }
        defaults.update(kwargs)

        # Set up as_dict return value
        obj.as_dict.return_value = {
            "name": defaults["name"],
            "schema": defaults["schema"],
            "uuid": defaults["uuid"],
            "sill": defaults["sill"],
            "nugget": defaults["nugget"],
            "is_rotation_fixed": defaults["is_rotation_fixed"],
            "structures": defaults["structures"],
            "tags": defaults["tags"],
        }
        if defaults["attribute"]:
            obj.as_dict.return_value["attribute"] = defaults["attribute"]
        if defaults["domain"]:
            obj.as_dict.return_value["domain"] = defaults["domain"]
        if defaults["modelling_space"]:
            obj.as_dict.return_value["modelling_space"] = defaults["modelling_space"]
        if defaults["data_variance"] is not None:
            obj.as_dict.return_value["data_variance"] = defaults["data_variance"]

        # Set up direct attributes
        obj.sill = defaults["sill"]
        obj.nugget = defaults["nugget"]
        obj.is_rotation_fixed = defaults["is_rotation_fixed"]
        obj.attribute = defaults["attribute"]
        obj.domain = defaults["domain"]
        obj.modelling_space = defaults["modelling_space"]
        obj.data_variance = defaults["data_variance"]
        obj.structures = defaults["structures"]

        # Set up metadata for URL generation
        obj.metadata = MagicMock()
        obj.metadata.environment = MagicMock()
        obj.metadata.environment.org_id = "org-123"
        obj.metadata.environment.workspace_id = "ws-456"
        obj.metadata.environment.hub_url = "https://test.api.seequent.com"
        obj.metadata.id = defaults["uuid"]

        return obj

    def test_formats_variogram_with_basic_properties(self):
        """Test formatting a variogram with basic properties."""
        obj = self._create_mock_variogram()

        html = format_variogram(obj)

        self.assertIn("Test Variogram", html)
        self.assertIn("objects/variogram/v1.1.0", html)
        self.assertIn("12345-abcd", html)
        self.assertIn("Sill:", html)
        self.assertIn("1.5", html)
        self.assertIn("Nugget:", html)
        self.assertIn("0.2", html)
        self.assertIn("Rotation Fixed:", html)
        self.assertIn("True", html)

    def test_formats_variogram_with_optional_fields(self):
        """Test formatting a variogram with optional fields."""
        obj = self._create_mock_variogram(
            attribute="gold_grade",
            domain="ore_zone",
            modelling_space="data",
            data_variance=1.5,
        )

        html = format_variogram(obj)

        self.assertIn("Attribute:", html)
        self.assertIn("gold_grade", html)
        self.assertIn("Domain:", html)
        self.assertIn("ore_zone", html)
        self.assertIn("Modelling Space:", html)
        self.assertIn("data", html)
        self.assertIn("Data Variance:", html)

    def test_formats_variogram_structures(self):
        """Test that variogram structures are rendered."""
        obj = self._create_mock_variogram(
            structures=[
                {
                    "variogram_type": "spherical",
                    "contribution": 0.8,
                    "anisotropy": {
                        "ellipsoid_ranges": {"major": 100.0, "semi_major": 50.0, "minor": 25.0},
                        "rotation": {"dip": 0.0, "dip_azimuth": 0.0, "pitch": 0.0},
                    },
                },
                {
                    "variogram_type": "exponential",
                    "contribution": 0.5,
                    "anisotropy": {
                        "ellipsoid_ranges": {"major": 200.0, "semi_major": 100.0, "minor": 50.0},
                        "rotation": {"dip": 10.0, "dip_azimuth": 45.0, "pitch": 5.0},
                    },
                },
            ]
        )

        html = format_variogram(obj)

        self.assertIn("Structures (2):", html)
        self.assertIn("spherical", html)
        self.assertIn("exponential", html)
        self.assertIn("0.8", html)
        self.assertIn("0.5", html)

    def test_formats_variogram_without_optional_fields(self):
        """Test that optional fields are not shown when not present."""
        obj = self._create_mock_variogram()

        html = format_variogram(obj)

        self.assertNotIn("Attribute:", html)
        self.assertNotIn("Domain:", html)
        self.assertNotIn("Modelling Space:", html)
        self.assertNotIn("Data Variance:", html)

    def test_formats_variogram_with_tags(self):
        """Test formatting a variogram with tags."""
        obj = self._create_mock_variogram(tags={"project": "mining", "stage": "exploration"})

        html = format_variogram(obj)

        self.assertIn("Tags:", html)
        self.assertIn("project", html)
        self.assertIn("mining", html)

    def test_formats_variogram_structure_ranges(self):
        """Test that structure ranges are properly formatted."""
        obj = self._create_mock_variogram(
            structures=[
                {
                    "variogram_type": "spherical",
                    "contribution": 1.0,
                    "anisotropy": {
                        "ellipsoid_ranges": {"major": 150.5, "semi_major": 75.2, "minor": 30.8},
                        "rotation": {"dip": 15.0, "dip_azimuth": 90.0, "pitch": 0.0},
                    },
                },
            ]
        )

        html = format_variogram(obj)

        # Check ranges are formatted
        self.assertIn("150.5", html)
        self.assertIn("75.2", html)
        self.assertIn("30.8", html)
        # Check rotation values are included
        self.assertIn("15.0", html)
        self.assertIn("90.0", html)


class TestFormatBlockModelVersion(unittest.TestCase):
    """Tests for the format_block_model_version function."""

    def _create_mock_version(self, **kwargs):
        """Create a mock Version object."""
        defaults = {
            "version_id": 1,
            "version_uuid": UUID("12345678-1234-1234-1234-123456789abc"),
            "bm_uuid": UUID("abcd1234-1234-1234-1234-123456789abc"),
            "parent_version_id": None,
            "base_version_id": None,
            "created_at": datetime(2025, 1, 15, 10, 30, 0),
            "comment": "Initial version",
            "columns": [],
        }
        defaults.update(kwargs)

        obj = MagicMock()
        obj.version_id = defaults["version_id"]
        obj.version_uuid = defaults["version_uuid"]
        obj.bm_uuid = defaults["bm_uuid"]
        obj.parent_version_id = defaults["parent_version_id"]
        obj.base_version_id = defaults["base_version_id"]
        obj.created_at = defaults["created_at"]
        obj.comment = defaults["comment"]
        obj.columns = defaults["columns"]
        obj.bbox = None

        # Create mock user
        created_by = MagicMock()
        created_by.name = "Test User"
        created_by.email = "test@example.com"
        created_by.id = "user-123"
        obj.created_by = created_by

        return obj

    def test_formats_version_basic_info(self):
        """Test formatting a version with basic information."""
        obj = self._create_mock_version()

        html = format_block_model_version(obj)

        self.assertIn("Version ID", html)
        self.assertIn("1", html)
        self.assertIn("Version UUID", html)
        self.assertIn("Block Model UUID", html)
        self.assertIn("Created At", html)
        self.assertIn("2025-01-15", html)
        self.assertIn("Created By", html)
        self.assertIn("Test User", html)
        self.assertIn("Comment", html)
        self.assertIn("Initial version", html)

    def test_formats_version_with_columns(self):
        """Test formatting a version with columns."""
        col1 = MagicMock()
        col1.title = "Au"
        col1.data_type = MagicMock()
        col1.data_type.value = "Float64"
        col1.unit_id = "g/t"

        col2 = MagicMock()
        col2.title = "density"
        col2.data_type = MagicMock()
        col2.data_type.value = "Float64"
        col2.unit_id = "t/m3"

        obj = self._create_mock_version(columns=[col1, col2])

        html = format_block_model_version(obj)

        self.assertIn("Columns", html)
        self.assertIn("Au", html)
        self.assertIn("Float64", html)
        self.assertIn("g/t", html)
        self.assertIn("density", html)
        self.assertIn("t/m3", html)

    def test_formats_version_with_bbox(self):
        """Test formatting a version with bounding box."""
        bbox = MagicMock()
        bbox.i_minmax = MagicMock()
        bbox.i_minmax.min = 0
        bbox.i_minmax.max = 10
        bbox.j_minmax = MagicMock()
        bbox.j_minmax.min = 0
        bbox.j_minmax.max = 20
        bbox.k_minmax = MagicMock()
        bbox.k_minmax.min = 0
        bbox.k_minmax.max = 5

        obj = self._create_mock_version()
        obj.bbox = bbox

        html = format_block_model_version(obj)

        self.assertIn("Bounding Box", html)
        self.assertIn("10", html)
        self.assertIn("20", html)


class TestFormatBlockModel(unittest.TestCase):
    """Tests for the format_block_model function."""

    def _create_mock_block_model(self, **kwargs):
        """Create a mock BlockModel object."""
        defaults = {
            "name": "Test Block Model",
            "schema": "objects/block-model/v1.0.0",
            "uuid": "12345-abcd",
            "block_model_uuid": UUID("abcd1234-1234-1234-1234-123456789abc"),
            "bounding_box": None,
            "coordinate_reference_system": None,
            "tags": None,
        }
        defaults.update(kwargs)

        obj = MagicMock()
        obj.as_dict.return_value = {
            "name": defaults["name"],
            "schema": defaults["schema"],
            "uuid": defaults["uuid"],
        }
        if defaults["bounding_box"]:
            obj.as_dict.return_value["bounding_box"] = defaults["bounding_box"]
        if defaults["coordinate_reference_system"]:
            obj.as_dict.return_value["coordinate_reference_system"] = defaults["coordinate_reference_system"]
        if defaults["tags"]:
            obj.as_dict.return_value["tags"] = defaults["tags"]

        obj.block_model_uuid = defaults["block_model_uuid"]

        # Set up geometry
        geometry = MagicMock()
        origin = MagicMock()
        origin.x = 0.0
        origin.y = 0.0
        origin.z = 0.0
        geometry.origin = origin

        n_blocks = MagicMock()
        n_blocks.nx = 10
        n_blocks.ny = 20
        n_blocks.nz = 5
        geometry.n_blocks = n_blocks

        block_size = MagicMock()
        block_size.dx = 2.5
        block_size.dy = 5.0
        block_size.dz = 5.0
        geometry.block_size = block_size

        geometry.rotation = None
        obj.geometry = geometry

        # Set up attributes
        obj.attributes = []

        # Set up inner _obj for metadata
        inner_obj = MagicMock()
        inner_obj.metadata = MagicMock()
        inner_obj.metadata.environment = MagicMock()
        inner_obj.metadata.environment.org_id = "org-123"
        inner_obj.metadata.environment.workspace_id = "ws-456"
        inner_obj.metadata.environment.hub_url = "https://test.api.seequent.com"
        inner_obj.metadata.id = defaults["uuid"]
        obj._obj = inner_obj

        # Set up metadata on obj for URL generation
        obj.metadata = inner_obj.metadata

        return obj

    def test_formats_block_model_basic_info(self):
        """Test formatting a block model with basic information."""
        obj = self._create_mock_block_model()

        html = format_block_model(obj)

        self.assertIn("Test Block Model", html)
        self.assertIn("objects/block-model/v1.0.0", html)
        self.assertIn("12345-abcd", html)
        self.assertIn("Block Model UUID:", html)

    def test_formats_block_model_geometry(self):
        """Test formatting a block model with geometry information."""
        obj = self._create_mock_block_model()

        html = format_block_model(obj)

        self.assertIn("Geometry:", html)
        self.assertIn("Origin:", html)
        self.assertIn("N Blocks:", html)
        self.assertIn("Block Size:", html)
        self.assertIn("(10, 20, 5)", html)  # n_blocks
        self.assertIn("2.50", html)  # block_size.dx

    def test_formats_block_model_with_rotation(self):
        """Test formatting a block model with rotation."""
        obj = self._create_mock_block_model()
        rotation = MagicMock()
        rotation.dip_azimuth = 45.0
        rotation.dip = 30.0
        rotation.pitch = 15.0
        obj.geometry.rotation = rotation

        html = format_block_model(obj)

        self.assertIn("Rotation:", html)
        self.assertIn("45.00", html)
        self.assertIn("30.00", html)
        self.assertIn("15.00", html)

    def test_formats_block_model_with_bounding_box(self):
        """Test formatting a block model with bounding box."""
        obj = self._create_mock_block_model(
            bounding_box={
                "min_x": 0.0,
                "max_x": 25.0,
                "min_y": 0.0,
                "max_y": 100.0,
                "min_z": 0.0,
                "max_z": 25.0,
            }
        )

        html = format_block_model(obj)

        self.assertIn("Bounding Box:", html)
        self.assertIn("25.00", html)
        self.assertIn("100.00", html)

    def test_formats_block_model_with_crs(self):
        """Test formatting a block model with CRS."""
        obj = self._create_mock_block_model(coordinate_reference_system={"epsg_code": 28354})

        html = format_block_model(obj)

        self.assertIn("CRS:", html)
        self.assertIn("EPSG:28354", html)

    def test_formats_block_model_with_attributes(self):
        """Test formatting a block model with attributes."""
        attr1 = MagicMock()
        attr1.name = "Au"
        attr1.attribute_type = "Float64"
        attr1.unit = "g/t"

        attr2 = MagicMock()
        attr2.name = "density"
        attr2.attribute_type = "Float64"
        attr2.unit = "t/m3"

        obj = self._create_mock_block_model()
        obj.attributes = [attr1, attr2]

        html = format_block_model(obj)

        self.assertIn("Attributes (2):", html)
        self.assertIn("Au", html)
        self.assertIn("Float64", html)
        self.assertIn("g/t", html)
        self.assertIn("density", html)
        self.assertIn("t/m3", html)


class TestFormatBlockModelAttributes(unittest.TestCase):
    """Tests for the format_block_model_attributes function."""

    def test_formats_empty_attributes(self):
        """Test formatting empty block model attributes collection."""
        obj = MagicMock()
        obj.__len__ = MagicMock(return_value=0)

        html = format_block_model_attributes(obj)

        self.assertIn("No attributes available", html)

    def test_formats_attributes_collection(self):
        """Test formatting a collection of block model attributes."""
        attr1 = MagicMock()
        attr1.name = "Au"
        attr1.attribute_type = "Float64"
        attr1.unit = "g/t"

        attr2 = MagicMock()
        attr2.name = "density"
        attr2.attribute_type = "Float64"
        attr2.unit = None

        obj = MagicMock()
        obj.__len__ = MagicMock(return_value=2)
        obj.__iter__ = MagicMock(return_value=iter([attr1, attr2]))

        html = format_block_model_attributes(obj)

        self.assertIn("Name", html)
        self.assertIn("Type", html)
        self.assertIn("Unit", html)
        self.assertIn("Au", html)
        self.assertIn("Float64", html)
        self.assertIn("g/t", html)
        self.assertIn("density", html)


class TestFormatReport(unittest.TestCase):
    """Tests for the format_report function."""

    def _create_mock_report(self, **kwargs):
        """Create a mock Report object."""
        defaults = {
            "id": UUID("12345678-1234-1234-1234-123456789abc"),
            "name": "Test Report",
            "revision": 1,
            "block_model_uuid": UUID("abcd1234-1234-1234-1234-123456789abc"),
            "block_model_name": "Test Block Model",
            "columns": [],
            "categories": [],
            "last_result_created_at": None,
        }
        defaults.update(kwargs)

        obj = MagicMock()
        obj.id = defaults["id"]
        obj.name = defaults["name"]
        obj.revision = defaults["revision"]
        obj._block_model_uuid = defaults["block_model_uuid"]
        obj._block_model_name = defaults["block_model_name"]

        # Set up specification
        spec = MagicMock()
        spec.columns = defaults["columns"]
        spec.categories = defaults["categories"]
        spec.last_result_created_at = defaults["last_result_created_at"]
        obj._specification = spec

        # Set up context for URL generation
        context = MagicMock()
        env = MagicMock()
        env.org_id = "org-123"
        env.workspace_id = "ws-456"
        env.hub_url = "https://test.api.seequent.com"
        context.get_environment.return_value = env
        obj._context = context

        return obj

    def test_formats_report_basic_info(self):
        """Test formatting a report with basic information."""
        obj = self._create_mock_report()

        html = format_report(obj)

        self.assertIn("Test Report", html)
        self.assertIn("Report ID:", html)
        self.assertIn("Block Model:", html)
        self.assertIn("Test Block Model", html)
        self.assertIn("Revision:", html)
        self.assertIn("1", html)

    def test_formats_report_with_columns(self):
        """Test formatting a report with column specifications."""
        col1 = MagicMock()
        col1.label = "Au Grade"
        col1.aggregation = "MASS_AVERAGE"
        col1.output_unit_id = "g/t"

        col2 = MagicMock()
        col2.label = "Tonnage"
        col2.aggregation = "SUM"
        col2.output_unit_id = "t"

        obj = self._create_mock_report(columns=[col1, col2])

        html = format_report(obj)

        self.assertIn("Columns:", html)
        self.assertIn("Au Grade", html)
        self.assertIn("MASS_AVERAGE", html)
        self.assertIn("g/t", html)
        self.assertIn("Tonnage", html)
        self.assertIn("SUM", html)

    def test_formats_report_with_categories(self):
        """Test formatting a report with category specifications."""
        cat1 = MagicMock()
        cat1.label = "Domain"
        cat1.values = ["ore", "waste"]

        cat2 = MagicMock()
        cat2.label = "Rock Type"
        cat2.values = None

        obj = self._create_mock_report(categories=[cat1, cat2])

        html = format_report(obj)

        self.assertIn("Categories:", html)
        self.assertIn("Domain", html)
        self.assertIn("ore", html)
        self.assertIn("waste", html)
        self.assertIn("Rock Type", html)
        self.assertIn("(all)", html)

    def test_formats_report_with_last_run(self):
        """Test formatting a report with last run timestamp."""
        obj = self._create_mock_report(last_result_created_at=datetime(2025, 1, 15, 10, 30, 0))

        html = format_report(obj)

        self.assertIn("Last run:", html)
        self.assertIn("2025-01-15", html)


class TestFormatReportResult(unittest.TestCase):
    """Tests for the format_report_result function."""

    def _create_mock_report_result(self, **kwargs):
        """Create a mock ReportResult object."""
        import pandas as pd

        defaults = {
            "version_id": 1,
            "created_at": datetime(2025, 1, 15, 10, 30, 0),
            "dataframe": pd.DataFrame(
                {"cutoff": [0.0, 0.5], "Domain": ["ore", "waste"], "Tonnage": [1000.0, 500.0], "Au Grade": [2.5, 1.2]}
            ),
        }
        defaults.update(kwargs)

        obj = MagicMock()
        obj.version_id = defaults["version_id"]
        obj.created_at = defaults["created_at"]
        obj.to_dataframe.return_value = defaults["dataframe"]

        return obj

    def test_formats_report_result_basic_info(self):
        """Test formatting a report result with basic information."""
        obj = self._create_mock_report_result()

        html = format_report_result(obj)

        self.assertIn("Report Result", html)
        self.assertIn("Version 1", html)
        self.assertIn("Created:", html)
        self.assertIn("2025-01-15", html)
        self.assertIn("Rows: 2", html)

    def test_formats_report_result_table(self):
        """Test formatting a report result with data table."""
        obj = self._create_mock_report_result()

        html = format_report_result(obj)

        self.assertIn("cutoff", html)
        self.assertIn("Domain", html)
        self.assertIn("Tonnage", html)
        self.assertIn("Au Grade", html)
        self.assertIn("ore", html)
        self.assertIn("waste", html)
        self.assertIn("1000", html)
        self.assertIn("2.5", html)


class TestFormatTaskResult(unittest.TestCase):
    """Tests for the format_task_result function."""

    def _create_mock_task_result(self, **kwargs):
        """Create a mock TaskResult object."""
        defaults = {
            "message": "Task completed successfully",
            "target_name": "Test Grid",
            "schema_type": "objects/regular-3d-grid/v1.0.0",
            "attribute_name": "kriged_grade",
            "target_reference": (
                "https://350mt.api.seequent.com/geoscience-object"
                "/orgs/12345678-1234-1234-1234-123456789abc"
                "/workspaces/87654321-4321-4321-4321-abcdef123456"
                "/objects/aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
            ),
        }
        defaults.update(kwargs)

        obj = MagicMock()
        obj.message = defaults["message"]
        obj.target_name = defaults["target_name"]
        obj.schema_type = defaults["schema_type"]
        obj.attribute_name = defaults["attribute_name"]

        # Mock _target with reference for portal URL
        obj._target = MagicMock()
        obj._target.reference = defaults["target_reference"]

        # Mock _get_result_type_name
        obj._get_result_type_name = MagicMock(return_value="Kriging")

        return obj

    def test_formats_task_result_basic_info(self):
        """Test formatting a task result with basic information."""
        obj = self._create_mock_task_result()

        html = format_task_result(obj)

        self.assertIn("Kriging Result", html)
        self.assertIn("Test Grid", html)
        self.assertIn("objects/regular-3d-grid/v1.0.0", html)
        self.assertIn("kriged_grade", html)
        self.assertIn("Task completed successfully", html)
        self.assertIn("attr-highlight", html)  # Attribute should be highlighted

    def test_formats_task_result_with_portal_link(self):
        """Test formatting a task result includes portal link."""
        obj = self._create_mock_task_result()

        html = format_task_result(obj)

        self.assertIn("Portal", html)
        self.assertIn("href=", html)

    def test_formats_task_result_without_portal_link(self):
        """Test formatting a task result without reference doesn't fail."""
        obj = self._create_mock_task_result(target_reference=None)

        html = format_task_result(obj)

        # Should still render without crashing
        self.assertIn("Kriging Result", html)
        self.assertIn("Test Grid", html)

    def test_formats_task_result_checkmark(self):
        """Test formatting a task result shows checkmark for success."""
        obj = self._create_mock_task_result()

        html = format_task_result(obj)

        self.assertIn("✓", html)

    def test_formats_task_result_target_row(self):
        """Test formatting includes Target row."""
        obj = self._create_mock_task_result()

        html = format_task_result(obj)

        self.assertIn("Target:", html)
        self.assertIn("Test Grid", html)

    def test_formats_task_result_schema_row(self):
        """Test formatting includes Schema row."""
        obj = self._create_mock_task_result()

        html = format_task_result(obj)

        self.assertIn("Schema:", html)

    def test_formats_task_result_attribute_row(self):
        """Test formatting includes Attribute row."""
        obj = self._create_mock_task_result()

        html = format_task_result(obj)

        self.assertIn("Attribute:", html)

    def test_formats_task_result_without_get_result_type_name(self):
        """Test formatting a task result that doesn't have _get_result_type_name."""
        obj = self._create_mock_task_result()
        del obj._get_result_type_name

        html = format_task_result(obj)

        # Should fall back to "Task"
        self.assertIn("Task Result", html)


class TestFormatTaskResults(unittest.TestCase):
    """Tests for the format_task_results function."""

    def _create_mock_task_result(self, **kwargs):
        """Create a mock TaskResult object."""
        defaults = {
            "message": "Task completed successfully",
            "target_name": "Test Grid",
            "schema_type": "objects/regular-3d-grid/v1.0.0",
            "attribute_name": "kriged_grade",
            "target_reference": (
                "https://350mt.api.seequent.com/geoscience-object"
                "/orgs/12345678-1234-1234-1234-123456789abc"
                "/workspaces/87654321-4321-4321-4321-abcdef123456"
                "/objects/aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
            ),
            "result_type": "Kriging",
        }
        defaults.update(kwargs)

        obj = MagicMock()
        obj.message = defaults["message"]
        obj.target_name = defaults["target_name"]
        obj.schema_type = defaults["schema_type"]
        obj.attribute_name = defaults["attribute_name"]
        obj._target = MagicMock()
        obj._target.reference = defaults["target_reference"]
        obj._get_result_type_name = MagicMock(return_value=defaults["result_type"])

        return obj

    def test_formats_empty_results(self):
        """Test formatting an empty results collection."""
        obj = MagicMock()
        obj._results = []

        html = format_task_results(obj)

        self.assertIn("No results", html)

    def test_formats_single_result(self):
        """Test formatting a collection with one result."""
        result1 = self._create_mock_task_result(target_name="Grid 1", attribute_name="attr_1")

        obj = MagicMock()
        obj._results = [result1]

        html = format_task_results(obj)

        self.assertIn("1 Kriging Results", html)
        self.assertIn("Grid 1", html)
        self.assertIn("attr_1", html)
        self.assertIn("✓", html)

    def test_formats_multiple_results(self):
        """Test formatting a collection with multiple results."""
        result1 = self._create_mock_task_result(target_name="Grid 1", attribute_name="attr_1")
        result2 = self._create_mock_task_result(target_name="Grid 2", attribute_name="attr_2")
        result3 = self._create_mock_task_result(target_name="Grid 3", attribute_name="attr_3")

        obj = MagicMock()
        obj._results = [result1, result2, result3]

        html = format_task_results(obj)

        self.assertIn("3 Kriging Results", html)
        self.assertIn("Grid 1", html)
        self.assertIn("Grid 2", html)
        self.assertIn("Grid 3", html)
        self.assertIn("attr_1", html)
        self.assertIn("attr_2", html)
        self.assertIn("attr_3", html)

    def test_formats_results_with_table_headers(self):
        """Test formatting includes proper table headers."""
        result1 = self._create_mock_task_result()

        obj = MagicMock()
        obj._results = [result1]

        html = format_task_results(obj)

        self.assertIn("#", html)
        self.assertIn("Target", html)
        self.assertIn("Attribute", html)
        self.assertIn("Schema", html)
        self.assertIn("Link", html)

    def test_formats_results_with_portal_links(self):
        """Test formatting includes portal links for each result."""
        result1 = self._create_mock_task_result(target_name="Grid 1")
        result2 = self._create_mock_task_result(target_name="Grid 2")

        obj = MagicMock()
        obj._results = [result1, result2]

        html = format_task_results(obj)

        # Should have portal links
        self.assertIn("Portal", html)
        self.assertIn("href=", html)

    def test_formats_results_without_portal_link(self):
        """Test formatting handles results without references."""
        result1 = self._create_mock_task_result(target_reference=None)

        obj = MagicMock()
        obj._results = [result1]

        html = format_task_results(obj)

        self.assertIn("N/A", html)

    def test_formats_results_row_numbers(self):
        """Test formatting includes sequential row numbers."""
        result1 = self._create_mock_task_result(target_name="Grid 1")
        result2 = self._create_mock_task_result(target_name="Grid 2")

        obj = MagicMock()
        obj._results = [result1, result2]

        html = format_task_results(obj)

        # Row numbers
        self.assertIn(">1<", html)
        self.assertIn(">2<", html)


class TestGetTaskResultPortalUrl(unittest.TestCase):
    """Tests for the _get_task_result_portal_url helper function."""

    def test_extracts_portal_url_from_valid_reference(self):
        """Test extracting portal URL from a valid object reference."""
        result = MagicMock()
        result._target = MagicMock()
        result._target.reference = (
            "https://350mt.api.seequent.com/geoscience-object"
            "/orgs/12345678-1234-1234-1234-123456789abc"
            "/workspaces/87654321-4321-4321-4321-abcdef123456"
            "/objects/aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        )

        url = _get_task_result_portal_url(result)

        self.assertIsNotNone(url)
        self.assertIn("evo.seequent.com", url)

    def test_returns_none_for_no_reference(self):
        """Test returns None when target has no reference."""
        result = MagicMock()
        result._target = MagicMock()
        result._target.reference = None

        url = _get_task_result_portal_url(result)

        self.assertIsNone(url)

    def test_returns_none_for_invalid_reference(self):
        """Test returns None for invalid reference URL."""
        result = MagicMock()
        result._target = MagicMock()
        result._target.reference = "not-a-valid-url"

        url = _get_task_result_portal_url(result)

        self.assertIsNone(url)

    def test_returns_none_when_no_target(self):
        """Test returns None when result has no _target attribute."""
        result = MagicMock(spec=[])  # Empty spec means no attributes

        url = _get_task_result_portal_url(result)

        self.assertIsNone(url)

    def test_returns_none_for_non_string_reference(self):
        """Test returns None when reference is not a string."""
        result = MagicMock()
        result._target = MagicMock()
        result._target.reference = 12345  # Not a string

        url = _get_task_result_portal_url(result)

        self.assertIsNone(url)

    def test_returns_none_for_empty_string_reference(self):
        """Test returns None when reference is an empty string."""
        result = MagicMock()
        result._target = MagicMock()
        result._target.reference = ""

        url = _get_task_result_portal_url(result)

        self.assertIsNone(url)


if __name__ == "__main__":
    unittest.main()
