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
from unittest.mock import MagicMock

from evo.widgets.formatters import format_attributes_collection, format_base_object, format_variogram


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


if __name__ == "__main__":
    unittest.main()

