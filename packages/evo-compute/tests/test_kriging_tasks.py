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

"""Tests for kriging task parameter handling."""

from unittest import TestCase
from unittest.mock import MagicMock

from evo.compute.tasks import (
    KrigingParameters,
    SearchNeighbourhood,
    Source,
    Target,
    CreateAttribute,
    UpdateAttribute,
)
from evo.compute.tasks.common import Ellipsoid, EllipsoidRanges


class TestKrigingParametersWithAttributes(TestCase):
    """Tests for KrigingParameters handling of typed attribute objects."""

    def _create_mock_attribute(self, name: str, exists: bool, object_url: str) -> MagicMock:
        """Create a mock attribute that behaves like Attribute or PendingAttribute."""
        attr = MagicMock()
        attr.name = name
        attr.exists = exists
        attr.expression = f"locations.attributes[?name=='{name}']"

        # Mock the _obj for object URL access
        mock_obj = MagicMock()
        mock_obj.metadata.url = object_url
        attr._obj = mock_obj

        if exists:
            attr.to_target_dict.return_value = {
                "operation": "update",
                "reference": attr.expression,
            }
        else:
            attr.to_target_dict.return_value = {
                "operation": "create",
                "name": name,
            }

        return attr

    def _create_mock_block_model_attribute(self, name: str, exists: bool, object_url: str) -> MagicMock:
        """Create a mock BlockModelAttribute or BlockModelPendingAttribute."""
        attr = MagicMock()
        attr.name = name
        attr.exists = exists
        attr.expression = f"attributes[?name=='{name}']"

        # Mock the _obj for object URL access (unified interface with Attribute)
        mock_bm = MagicMock()
        mock_bm.metadata.url = object_url
        attr._obj = mock_bm

        if exists:
            attr.to_target_dict.return_value = {
                "operation": "update",
                "reference": attr.expression,
            }
        else:
            attr.to_target_dict.return_value = {
                "operation": "create",
                "name": name,
            }

        return attr

    def test_kriging_params_with_pending_attribute_target(self):
        """Test KrigingParameters accepts PendingAttribute as target."""
        source = Source(object="https://example.com/pointset", attribute="locations.attributes[?name=='grade']")
        target_attr = self._create_mock_attribute(
            name="kriged_grade",
            exists=False,
            object_url="https://example.com/grid",
        )
        variogram = "https://example.com/variogram"
        search = SearchNeighbourhood(
            ellipsoid=Ellipsoid(ranges=EllipsoidRanges(100, 100, 50)),
            max_samples=20,
        )

        params = KrigingParameters(
            source=source,
            target=target_attr,
            variogram=variogram,
            search=search,
        )

        # Verify the target was converted correctly
        params_dict = params.to_dict()
        self.assertEqual(params_dict["target"]["object"], "https://example.com/grid")
        self.assertEqual(params_dict["target"]["attribute"]["operation"], "create")
        self.assertEqual(params_dict["target"]["attribute"]["name"], "kriged_grade")

    def test_kriging_params_with_existing_attribute_target(self):
        """Test KrigingParameters accepts existing Attribute as target."""
        source = Source(object="https://example.com/pointset", attribute="locations.attributes[?name=='grade']")
        target_attr = self._create_mock_attribute(
            name="existing_attr",
            exists=True,
            object_url="https://example.com/grid",
        )
        variogram = "https://example.com/variogram"
        search = SearchNeighbourhood(
            ellipsoid=Ellipsoid(ranges=EllipsoidRanges(100, 100, 50)),
            max_samples=20,
        )

        params = KrigingParameters(
            source=source,
            target=target_attr,
            variogram=variogram,
            search=search,
        )

        # Verify the target was converted correctly
        params_dict = params.to_dict()
        self.assertEqual(params_dict["target"]["object"], "https://example.com/grid")
        self.assertEqual(params_dict["target"]["attribute"]["operation"], "update")
        self.assertIn("reference", params_dict["target"]["attribute"])

    def test_kriging_params_with_block_model_pending_attribute(self):
        """Test KrigingParameters accepts BlockModelPendingAttribute as target."""
        source = Source(object="https://example.com/pointset", attribute="locations.attributes[?name=='grade']")
        target_attr = self._create_mock_block_model_attribute(
            name="new_bm_attr",
            exists=False,
            object_url="https://example.com/blockmodel",
        )
        variogram = "https://example.com/variogram"
        search = SearchNeighbourhood(
            ellipsoid=Ellipsoid(ranges=EllipsoidRanges(100, 100, 50)),
            max_samples=20,
        )

        params = KrigingParameters(
            source=source,
            target=target_attr,
            variogram=variogram,
            search=search,
        )

        # Verify the target was converted correctly
        params_dict = params.to_dict()
        self.assertEqual(params_dict["target"]["object"], "https://example.com/blockmodel")
        self.assertEqual(params_dict["target"]["attribute"]["operation"], "create")
        self.assertEqual(params_dict["target"]["attribute"]["name"], "new_bm_attr")

    def test_kriging_params_with_block_model_existing_attribute(self):
        """Test KrigingParameters accepts existing BlockModelAttribute as target."""
        source = Source(object="https://example.com/pointset", attribute="locations.attributes[?name=='grade']")
        target_attr = self._create_mock_block_model_attribute(
            name="existing_bm_attr",
            exists=True,
            object_url="https://example.com/blockmodel",
        )
        variogram = "https://example.com/variogram"
        search = SearchNeighbourhood(
            ellipsoid=Ellipsoid(ranges=EllipsoidRanges(100, 100, 50)),
            max_samples=20,
        )

        params = KrigingParameters(
            source=source,
            target=target_attr,
            variogram=variogram,
            search=search,
        )

        # Verify the target was converted correctly
        params_dict = params.to_dict()
        self.assertEqual(params_dict["target"]["object"], "https://example.com/blockmodel")
        self.assertEqual(params_dict["target"]["attribute"]["operation"], "update")
        self.assertIn("reference", params_dict["target"]["attribute"])

    def test_kriging_params_with_explicit_target(self):
        """Test KrigingParameters still works with explicit Target object."""
        source = Source(object="https://example.com/pointset", attribute="locations.attributes[?name=='grade']")
        target = Target.new_attribute("https://example.com/grid", "kriged_grade")
        variogram = "https://example.com/variogram"
        search = SearchNeighbourhood(
            ellipsoid=Ellipsoid(ranges=EllipsoidRanges(100, 100, 50)),
            max_samples=20,
        )

        params = KrigingParameters(
            source=source,
            target=target,
            variogram=variogram,
            search=search,
        )

        # Verify the target works correctly
        params_dict = params.to_dict()
        self.assertEqual(params_dict["target"]["object"], "https://example.com/grid")
        self.assertEqual(params_dict["target"]["attribute"]["operation"], "create")
        self.assertEqual(params_dict["target"]["attribute"]["name"], "kriged_grade")

    def test_kriging_params_source_attribute_conversion(self):
        """Test KrigingParameters converts source Attribute correctly."""
        # Create mock source attribute
        source_attr = MagicMock()
        source_attr.expression = "locations.attributes[?key=='grade']"
        mock_obj = MagicMock()
        mock_obj.metadata.url = "https://example.com/pointset"
        source_attr._obj = mock_obj

        target = Target.new_attribute("https://example.com/grid", "kriged_grade")
        variogram = "https://example.com/variogram"
        search = SearchNeighbourhood(
            ellipsoid=Ellipsoid(ranges=EllipsoidRanges(100, 100, 50)),
            max_samples=20,
        )

        params = KrigingParameters(
            source=source_attr,
            target=target,
            variogram=variogram,
            search=search,
        )

        # Verify the source was converted correctly
        params_dict = params.to_dict()
        self.assertEqual(params_dict["source"]["object"], "https://example.com/pointset")
        self.assertEqual(params_dict["source"]["attribute"], "locations.attributes[?key=='grade']")


class TestTargetSerialization(TestCase):
    """Tests for Target serialization with different attribute types."""

    def test_target_with_create_attribute(self):
        """Test Target serializes CreateAttribute correctly."""
        target = Target(
            object="https://example.com/grid",
            attribute=CreateAttribute(name="new_attr"),
        )

        result = target.to_dict()

        self.assertEqual(result["object"], "https://example.com/grid")
        self.assertEqual(result["attribute"]["operation"], "create")
        self.assertEqual(result["attribute"]["name"], "new_attr")

    def test_target_with_update_attribute(self):
        """Test Target serializes UpdateAttribute correctly."""
        target = Target(
            object="https://example.com/grid",
            attribute=UpdateAttribute(reference="cell_attributes[?name=='existing']"),
        )

        result = target.to_dict()

        self.assertEqual(result["object"], "https://example.com/grid")
        self.assertEqual(result["attribute"]["operation"], "update")
        self.assertEqual(result["attribute"]["reference"], "cell_attributes[?name=='existing']")

    def test_target_with_dict_attribute(self):
        """Test Target serializes dict attribute correctly."""
        target = Target(
            object="https://example.com/grid",
            attribute={"operation": "create", "name": "dict_attr"},
        )

        result = target.to_dict()

        self.assertEqual(result["object"], "https://example.com/grid")
        self.assertEqual(result["attribute"]["operation"], "create")
        self.assertEqual(result["attribute"]["name"], "dict_attr")

    def test_target_new_attribute_factory(self):
        """Test Target.new_attribute factory method."""
        target = Target.new_attribute("https://example.com/grid", "new_attr")

        result = target.to_dict()

        self.assertEqual(result["object"], "https://example.com/grid")
        self.assertEqual(result["attribute"]["operation"], "create")
        self.assertEqual(result["attribute"]["name"], "new_attr")

