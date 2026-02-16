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

"""Tests for evo.widgets.urls module."""

import unittest
from unittest.mock import MagicMock

from evo.widgets.urls import (
    get_evo_base_url,
    get_hub_code,
    get_portal_url,
    get_portal_url_for_object,
    get_portal_url_from_reference,
    get_viewer_url,
    get_viewer_url_for_object,
    get_viewer_url_for_objects,
    get_viewer_url_from_reference,
    parse_object_reference_url,
    serialize_object_reference,
)


class TestGetEvoBaseUrl(unittest.TestCase):
    """Tests for get_evo_base_url function."""

    def test_production_environment(self):
        """Production environment hub URL returns production portal URL."""
        result = get_evo_base_url("https://350mt.api.seequent.com")
        self.assertEqual(result, "https://evo.seequent.com")

    def test_production_environment_with_different_hub(self):
        """Production with different hub code still returns production portal URL."""
        result = get_evo_base_url("https://mining.api.seequent.com")
        self.assertEqual(result, "https://evo.seequent.com")


class TestGetHubCode(unittest.TestCase):
    """Tests for get_hub_code function."""

    def test_extracts_hub_code_from_url(self):
        """Hub code is extracted from URL."""
        result = get_hub_code("https://350mt.api.seequent.com")
        self.assertEqual(result, "350mt")

    def test_extracts_hub_code_from_production_url(self):
        """Hub code is extracted from production URL."""
        result = get_hub_code("https://mining.api.seequent.com")
        self.assertEqual(result, "mining")

    def test_raises_on_invalid_url(self):
        """Raises ValueError for invalid URL."""
        with self.assertRaises(ValueError):
            get_hub_code("")


class TestGetPortalUrl(unittest.TestCase):
    """Tests for get_portal_url function."""

    def test_generates_portal_url(self):
        """Portal URL uses format: /{org}/data/{workspace}/objects/{object}."""
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
            hub_url="https://350mt.api.seequent.com",
        )
        self.assertEqual(
            result,
            "https://evo.seequent.com/org-123/workspaces/350mt/ws-456/viewer?id=obj-789",
        )

    def test_generates_multi_object_viewer_url(self):
        """Viewer URL with multiple objects uses comma-separated IDs."""
        result = get_viewer_url(
            org_id="org-123",
            workspace_id="ws-456",
            object_ids=["obj-1", "obj-2", "obj-3"],
            hub_url="https://350mt.api.seequent.com",
        )
        self.assertEqual(
            result,
            "https://evo.seequent.com/org-123/workspaces/350mt/ws-456/viewer?id=obj-1,obj-2,obj-3",
        )


class TestParseObjectReferenceUrl(unittest.TestCase):
    """Tests for parse_object_reference_url function."""

    def test_parses_reference_url(self):
        """Object reference URL is parsed correctly."""
        ref_url = "https://350mt.api.seequent.com/geoscience-object/orgs/org-123/workspaces/ws-456/objects/obj-789"
        org_id, workspace_id, object_id, hub_url = parse_object_reference_url(ref_url)

        self.assertEqual(org_id, "org-123")
        self.assertEqual(workspace_id, "ws-456")
        self.assertEqual(object_id, "obj-789")
        self.assertEqual(hub_url, "https://350mt.api.seequent.com")

    def test_raises_on_invalid_path(self):
        """Raises ValueError for URLs with invalid path format."""
        with self.assertRaises(ValueError) as ctx:
            parse_object_reference_url("https://350mt.api.seequent.com/invalid/path")
        self.assertIn("does not match expected format", str(ctx.exception))


class TestGetPortalUrlFromReference(unittest.TestCase):
    """Tests for get_portal_url_from_reference function."""

    def test_generates_portal_url_from_reference(self):
        """Portal URL is generated from object reference URL."""
        ref_url = "https://350mt.api.seequent.com/geoscience-object/orgs/org-123/workspaces/ws-456/objects/obj-789"
        result = get_portal_url_from_reference(ref_url)
        self.assertEqual(
            result,
            "https://evo.seequent.com/org-123/data/ws-456/objects/obj-789",
        )


class TestGetViewerUrlFromReference(unittest.TestCase):
    """Tests for get_viewer_url_from_reference function."""

    def test_generates_viewer_url_from_reference(self):
        """Viewer URL is generated from object reference URL."""
        ref_url = "https://350mt.api.seequent.com/geoscience-object/orgs/org-123/workspaces/ws-456/objects/obj-789"
        result = get_viewer_url_from_reference(ref_url)
        self.assertEqual(
            result,
            "https://evo.seequent.com/org-123/workspaces/350mt/ws-456/viewer?id=obj-789",
        )


class TestSerializeObjectReference(unittest.TestCase):
    """Tests for serialize_object_reference function."""

    def test_returns_string_as_is(self):
        """String input is returned unchanged."""
        result = serialize_object_reference("https://example.com/ref")
        self.assertEqual(result, "https://example.com/ref")

    def test_raises_on_unsupported_type(self):
        """Raises TypeError for unsupported input types."""
        with self.assertRaises(TypeError) as ctx:
            serialize_object_reference(12345)
        self.assertIn("Cannot serialize object reference", str(ctx.exception))

    def test_serializes_object_with_metadata_url(self):
        """Objects with metadata.url attribute are serialized."""
        obj = MagicMock()
        obj.metadata.url = "mock://metadata-url"
        result = serialize_object_reference(obj)
        self.assertEqual(result, "mock://metadata-url")

    def test_serializes_object_with_url_attribute(self):
        """Objects with url attribute are serialized."""
        obj = MagicMock(spec=["url"])
        obj.url = "mock://direct-url"
        result = serialize_object_reference(obj)
        self.assertEqual(result, "mock://direct-url")


class TestGetPortalUrlForObject(unittest.TestCase):
    """Tests for get_portal_url_for_object function."""

    def test_generates_portal_url_from_object(self):
        """Portal URL is generated from object metadata."""
        obj = MagicMock()
        obj.metadata.environment.org_id = "org-123"
        obj.metadata.environment.workspace_id = "ws-456"
        obj.metadata.environment.hub_url = "https://350mt.api.seequent.com"
        obj.metadata.id = "obj-789"

        result = get_portal_url_for_object(obj)
        self.assertEqual(
            result,
            "https://evo.seequent.com/org-123/data/ws-456/objects/obj-789",
        )


class TestGetViewerUrlForObject(unittest.TestCase):
    """Tests for get_viewer_url_for_object function."""

    def test_generates_viewer_url_from_object(self):
        """Viewer URL is generated from object metadata."""
        obj = MagicMock()
        obj.metadata.environment.org_id = "org-123"
        obj.metadata.environment.workspace_id = "ws-456"
        obj.metadata.environment.hub_url = "https://350mt.api.seequent.com"
        obj.metadata.id = "obj-789"

        result = get_viewer_url_for_object(obj)
        self.assertEqual(
            result,
            "https://evo.seequent.com/org-123/workspaces/350mt/ws-456/viewer?id=obj-789",
        )


class TestGetViewerUrlForObjects(unittest.TestCase):
    """Tests for get_viewer_url_for_objects function."""

    def _create_mock_context(self):
        """Create a mock context with environment."""
        context = MagicMock()
        env = MagicMock()
        env.org_id = "org-123"
        env.workspace_id = "ws-456"
        env.hub_url = "https://350mt.api.seequent.com"
        context.get_environment.return_value = env
        return context

    def _create_mock_object(self, obj_id: str):
        """Create a mock object with metadata."""
        obj = MagicMock()
        obj.metadata.id = obj_id
        return obj

    def test_generates_viewer_url_for_single_object(self):
        """Viewer URL is generated for single object."""
        context = self._create_mock_context()
        obj = self._create_mock_object("obj-789")

        result = get_viewer_url_for_objects(context, [obj])
        self.assertEqual(
            result,
            "https://evo.seequent.com/org-123/workspaces/350mt/ws-456/viewer?id=obj-789",
        )

    def test_generates_viewer_url_for_multiple_objects(self):
        """Viewer URL is generated with comma-separated IDs for multiple objects."""
        context = self._create_mock_context()
        objects = [self._create_mock_object(f"obj-{i}") for i in range(1, 4)]

        result = get_viewer_url_for_objects(context, objects)
        self.assertEqual(
            result,
            "https://evo.seequent.com/org-123/workspaces/350mt/ws-456/viewer?id=obj-1,obj-2,obj-3",
        )

    def test_accepts_string_object_ids(self):
        """String object IDs are accepted directly."""
        context = self._create_mock_context()
        result = get_viewer_url_for_objects(context, ["id-1", "id-2"])
        self.assertEqual(
            result,
            "https://evo.seequent.com/org-123/workspaces/350mt/ws-456/viewer?id=id-1,id-2",
        )

    def test_raises_on_empty_list(self):
        """Raises ValueError when objects list is empty."""
        context = self._create_mock_context()
        with self.assertRaises(ValueError) as ctx:
            get_viewer_url_for_objects(context, [])
        self.assertIn("At least one object is required", str(ctx.exception))

    def test_raises_on_unsupported_object_type(self):
        """Raises TypeError for objects without extractable ID."""
        context = self._create_mock_context()
        with self.assertRaises(TypeError) as ctx:
            get_viewer_url_for_objects(context, [object()])
        self.assertIn("Cannot extract object ID", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
