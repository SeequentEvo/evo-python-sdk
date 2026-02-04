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

"""Tests for URL generation utilities."""

import unittest

from evo.common.urls import (
    get_evo_base_url,
    get_hub_code,
    get_portal_url,
    get_portal_url_from_reference,
    get_viewer_url,
    get_viewer_url_from_reference,
    parse_object_reference_url,
    serialize_object_reference,
)


class TestGetEvoBaseUrl(unittest.TestCase):
    """Tests for get_evo_base_url function."""

    def test_integration_environment(self):
        """Integration environment hub URL returns integration portal URL."""
        hub_url = "https://350mt.api.integration.seequent.com"
        result = get_evo_base_url(hub_url)
        self.assertEqual(result, "https://evo.integration.seequent.com")

    def test_test_environment(self):
        """Test environment hub URL returns test portal URL."""
        hub_url = "https://350mt.api.test.seequent.com"
        result = get_evo_base_url(hub_url)
        self.assertEqual(result, "https://evo.test.seequent.com")

    def test_production_environment(self):
        """Production environment hub URL returns production portal URL."""
        hub_url = "https://350mt.api.seequent.com"
        result = get_evo_base_url(hub_url)
        self.assertEqual(result, "https://evo.seequent.com")

    def test_production_environment_with_different_hub(self):
        """Production with different hub code still returns production portal URL."""
        hub_url = "https://mining.api.seequent.com"
        result = get_evo_base_url(hub_url)
        self.assertEqual(result, "https://evo.seequent.com")


class TestGetHubCode(unittest.TestCase):
    """Tests for get_hub_code function."""

    def test_extracts_hub_code_from_integration_url(self):
        """Hub code is extracted from integration URL."""
        hub_url = "https://350mt.api.integration.seequent.com"
        result = get_hub_code(hub_url)
        self.assertEqual(result, "350mt")

    def test_extracts_hub_code_from_production_url(self):
        """Hub code is extracted from production URL."""
        hub_url = "https://mining.api.seequent.com"
        result = get_hub_code(hub_url)
        self.assertEqual(result, "mining")

    def test_raises_on_invalid_url(self):
        """Raises ValueError for invalid URL."""
        with self.assertRaises(ValueError):
            get_hub_code("")


class TestGetPortalUrl(unittest.TestCase):
    """Tests for get_portal_url function."""

    def test_generates_new_format_portal_url(self):
        """Portal URL uses new format: /{org}/data/{workspace}/objects/{object}."""
        result = get_portal_url(
            org_id="org-123",
            workspace_id="ws-456",
            object_id="obj-789",
            hub_url="https://350mt.api.integration.seequent.com",
        )
        self.assertEqual(
            result,
            "https://evo.integration.seequent.com/org-123/data/ws-456/objects/obj-789",
        )

    def test_production_portal_url(self):
        """Production portal URL is generated correctly."""
        result = get_portal_url(
            org_id="org-123",
            workspace_id="ws-456",
            object_id="obj-789",
            hub_url="https://350mt.api.seequent.com",
        )
        self.assertEqual(
            result,
            "https://evo.seequent.com/org-123/data/ws-456/objects/obj-789",
        )


class TestGetViewerUrl(unittest.TestCase):
    """Tests for get_viewer_url function."""

    def test_generates_single_object_viewer_url(self):
        """Viewer URL with single object uses query parameter."""
        result = get_viewer_url(
            org_id="org-123",
            workspace_id="ws-456",
            object_ids="obj-789",
            hub_url="https://350mt.api.integration.seequent.com",
        )
        self.assertEqual(
            result,
            "https://evo.integration.seequent.com/org-123/workspaces/350mt/ws-456/viewer?id=obj-789",
        )

    def test_generates_multi_object_viewer_url(self):
        """Viewer URL with multiple objects uses comma-separated IDs."""
        result = get_viewer_url(
            org_id="org-123",
            workspace_id="ws-456",
            object_ids=["obj-1", "obj-2", "obj-3"],
            hub_url="https://350mt.api.integration.seequent.com",
        )
        self.assertEqual(
            result,
            "https://evo.integration.seequent.com/org-123/workspaces/350mt/ws-456/viewer?id=obj-1,obj-2,obj-3",
        )

    def test_production_viewer_url(self):
        """Production viewer URL is generated correctly."""
        result = get_viewer_url(
            org_id="org-123",
            workspace_id="ws-456",
            object_ids="obj-789",
            hub_url="https://mining.api.seequent.com",
        )
        self.assertEqual(
            result,
            "https://evo.seequent.com/org-123/workspaces/mining/ws-456/viewer?id=obj-789",
        )


class TestParseObjectReferenceUrl(unittest.TestCase):
    """Tests for parse_object_reference_url function."""

    def test_parses_integration_reference_url(self):
        """Object reference URL is parsed correctly for integration."""
        ref_url = "https://350mt.api.integration.seequent.com/geoscience-object/orgs/org-123/workspaces/ws-456/objects/obj-789"
        org_id, workspace_id, object_id, hub_url = parse_object_reference_url(ref_url)

        self.assertEqual(org_id, "org-123")
        self.assertEqual(workspace_id, "ws-456")
        self.assertEqual(object_id, "obj-789")
        self.assertEqual(hub_url, "https://350mt.api.integration.seequent.com")

    def test_parses_production_reference_url(self):
        """Object reference URL is parsed correctly for production."""
        ref_url = "https://mining.api.seequent.com/geoscience-object/orgs/my-org/workspaces/my-ws/objects/my-obj"
        org_id, workspace_id, object_id, hub_url = parse_object_reference_url(ref_url)

        self.assertEqual(org_id, "my-org")
        self.assertEqual(workspace_id, "my-ws")
        self.assertEqual(object_id, "my-obj")
        self.assertEqual(hub_url, "https://mining.api.seequent.com")

    def test_raises_on_invalid_path(self):
        """Raises ValueError for URLs with invalid path format."""
        invalid_url = "https://350mt.api.integration.seequent.com/invalid/path"
        with self.assertRaises(ValueError) as ctx:
            parse_object_reference_url(invalid_url)
        self.assertIn("does not match expected format", str(ctx.exception))


class TestGetPortalUrlFromReference(unittest.TestCase):
    """Tests for get_portal_url_from_reference function."""

    def test_generates_portal_url_from_reference(self):
        """Portal URL is generated from object reference URL."""
        ref_url = "https://350mt.api.integration.seequent.com/geoscience-object/orgs/org-123/workspaces/ws-456/objects/obj-789"
        result = get_portal_url_from_reference(ref_url)

        self.assertEqual(
            result,
            "https://evo.integration.seequent.com/org-123/data/ws-456/objects/obj-789",
        )


class TestGetViewerUrlFromReference(unittest.TestCase):
    """Tests for get_viewer_url_from_reference function."""

    def test_generates_viewer_url_from_reference(self):
        """Viewer URL is generated from object reference URL."""
        ref_url = "https://350mt.api.integration.seequent.com/geoscience-object/orgs/org-123/workspaces/ws-456/objects/obj-789"
        result = get_viewer_url_from_reference(ref_url)

        self.assertEqual(
            result,
            "https://evo.integration.seequent.com/org-123/workspaces/350mt/ws-456/viewer?id=obj-789",
        )


class TestSerializeObjectReference(unittest.TestCase):
    """Tests for serialize_object_reference function."""

    def test_returns_string_as_is(self):
        """String input is returned unchanged."""
        url = "https://example.com/ref"
        result = serialize_object_reference(url)
        self.assertEqual(result, url)

    def test_raises_on_unsupported_type(self):
        """Raises TypeError for unsupported input types."""
        with self.assertRaises(TypeError) as ctx:
            serialize_object_reference(12345)
        self.assertIn("Cannot serialize object reference", str(ctx.exception))

    def test_serializes_object_with_hub_url_and_org_id(self):
        """Objects with hub_url and org_id attributes are serialized."""

        class MockObjectReference:
            hub_url = "https://test.com"
            org_id = "org-123"

            def __str__(self):
                return "mock://reference"

        result = serialize_object_reference(MockObjectReference())
        self.assertEqual(result, "mock://reference")

    def test_serializes_object_with_metadata_url(self):
        """Objects with metadata.url attribute are serialized."""

        class MockUrl:
            def __str__(self):
                return "mock://metadata-url"

        class MockMetadata:
            url = MockUrl()

        class MockTypedObject:
            metadata = MockMetadata()

        result = serialize_object_reference(MockTypedObject())
        self.assertEqual(result, "mock://metadata-url")

    def test_serializes_object_with_url_attribute(self):
        """Objects with url attribute are serialized."""

        class MockObjectWithUrl:
            url = "mock://direct-url"

        result = serialize_object_reference(MockObjectWithUrl())
        self.assertEqual(result, "mock://direct-url")


if __name__ == "__main__":
    unittest.main()
