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

from evo.widgets.formatters import format_attributes_collection, format_base_object


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


if __name__ == "__main__":
    unittest.main()
