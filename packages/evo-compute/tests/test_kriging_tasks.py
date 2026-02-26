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

from evo.objects.typed.attributes import (
    Attribute,
    BlockModelAttribute,
    BlockModelPendingAttribute,
    PendingAttribute,
)

from evo.compute.tasks import (
    BlockDiscretisation,
    CreateAttribute,
    RegionFilter,
    SearchNeighborhood,
    Source,
    Target,
    UpdateAttribute,
)
from evo.compute.tasks.common import (
    Ellipsoid,
    EllipsoidRanges,
    get_attribute_expression,
    is_typed_attribute,
    source_from_attribute,
    target_from_attribute,
)
from evo.compute.tasks.kriging import KrigingParameters


def _create_mock_source_attribute(name: str, key: str, object_url: str, schema_path: str = "") -> MagicMock:
    """Create a mock Attribute (existing) that passes isinstance checks.

    Uses ``spec=Attribute`` so ``isinstance(mock, Attribute)`` returns True.
    Sets the underlying properties that the adapter functions inspect.
    """
    attr = MagicMock(spec=Attribute)
    attr.name = name
    attr.key = key
    attr.exists = True

    # ModelContext-like _context
    mock_context = MagicMock()
    mock_context.schema_path = schema_path
    attr._context = mock_context

    # Parent object
    mock_obj = MagicMock()
    mock_obj.metadata.url = object_url
    attr._obj = mock_obj

    return attr


def _create_pending_attribute(name: str, parent_obj: MagicMock | None = None) -> PendingAttribute:
    """Create a real PendingAttribute with an optional mock parent."""
    mock_parent = MagicMock()
    mock_parent._obj = parent_obj
    return PendingAttribute(mock_parent, name)


class TestAttributeAdapters(TestCase):
    """Tests for the attribute adapter functions in source_target."""

    # ---- is_typed_attribute ----

    def test_is_typed_attribute_with_attribute(self):
        attr = _create_mock_source_attribute("grade", "abc-key", "https://example.com/obj")
        self.assertTrue(is_typed_attribute(attr))

    def test_is_typed_attribute_with_pending_attribute(self):
        pending = _create_pending_attribute("new_attr")
        self.assertTrue(is_typed_attribute(pending))

    def test_is_typed_attribute_with_block_model_attribute(self):
        bm_attr = BlockModelAttribute(name="grade", attribute_type="Float64")
        self.assertTrue(is_typed_attribute(bm_attr))

    def test_is_typed_attribute_with_block_model_pending_attribute(self):
        bm_pending = BlockModelPendingAttribute(obj=None, name="new_col")
        self.assertTrue(is_typed_attribute(bm_pending))

    def test_is_typed_attribute_with_string(self):
        self.assertFalse(is_typed_attribute("some_string"))

    def test_is_typed_attribute_with_source(self):
        source = Source(object="https://example.com/obj", attribute="grade")
        self.assertFalse(is_typed_attribute(source))

    # ---- get_attribute_expression ----

    def test_expression_for_attribute_with_schema_path(self):
        attr = _create_mock_source_attribute(
            "grade", "abc-key", "https://example.com/obj", schema_path="locations.attributes"
        )
        result = get_attribute_expression(attr)
        self.assertEqual(result, "locations.attributes[?key=='abc-key']")

    def test_expression_for_attribute_without_schema_path(self):
        attr = _create_mock_source_attribute("grade", "abc-key", "https://example.com/obj", schema_path="")
        result = get_attribute_expression(attr)
        self.assertEqual(result, "attributes[?key=='abc-key']")

    def test_expression_for_pending_attribute(self):
        pending = _create_pending_attribute("my_attribute")
        result = get_attribute_expression(pending)
        self.assertEqual(result, "attributes[?name=='my_attribute']")

    def test_expression_for_block_model_attribute(self):
        bm_attr = BlockModelAttribute(name="grade", attribute_type="Float64")
        result = get_attribute_expression(bm_attr)
        self.assertEqual(result, "attributes[?name=='grade']")

    def test_expression_for_block_model_pending_attribute(self):
        bm_pending = BlockModelPendingAttribute(obj=None, name="new_col")
        result = get_attribute_expression(bm_pending)
        self.assertEqual(result, "attributes[?name=='new_col']")

    def test_expression_raises_for_invalid_type(self):
        with self.assertRaises(TypeError):
            get_attribute_expression("not_an_attribute")

    # ---- source_from_attribute ----

    def test_source_from_existing_attribute(self):
        attr = _create_mock_source_attribute(
            "grade", "abc-key", "https://example.com/pointset", schema_path="locations.attributes"
        )
        result = source_from_attribute(attr)
        self.assertIsInstance(result, Source)
        result_dict = result.to_dict()
        self.assertEqual(result_dict["object"], "https://example.com/pointset")
        self.assertEqual(result_dict["attribute"], "locations.attributes[?key=='abc-key']")

    def test_source_from_attribute_without_schema_path(self):
        attr = _create_mock_source_attribute("grade", "abc-key", "https://example.com/pointset", schema_path="")
        result = source_from_attribute(attr)
        result_dict = result.to_dict()
        self.assertEqual(result_dict["object"], "https://example.com/pointset")
        self.assertEqual(result_dict["attribute"], "attributes[?key=='abc-key']")

    def test_source_from_attribute_raises_for_pending(self):
        pending = _create_pending_attribute("new_attr")
        with self.assertRaises(TypeError):
            source_from_attribute(pending)

    def test_source_from_attribute_raises_for_block_model_attribute(self):
        bm_attr = BlockModelAttribute(name="grade", attribute_type="Float64")
        with self.assertRaises(TypeError):
            source_from_attribute(bm_attr)

    def test_source_from_attribute_raises_for_string(self):
        with self.assertRaises(TypeError):
            source_from_attribute("not_an_attribute")

    # ---- target_from_attribute ----

    def test_target_from_existing_attribute(self):
        attr = _create_mock_source_attribute(
            "grade", "abc-key", "https://example.com/obj", schema_path="locations.attributes"
        )
        result = target_from_attribute(attr)
        self.assertIsInstance(result, Target)
        result_dict = result.to_dict()
        self.assertEqual(result_dict["attribute"]["operation"], "update")
        self.assertEqual(result_dict["attribute"]["reference"], "locations.attributes[?key=='abc-key']")

    def test_target_from_pending_attribute(self):
        mock_obj = MagicMock()
        mock_obj.metadata.url = "https://example.com/grid"
        pending = _create_pending_attribute("new_column", parent_obj=mock_obj)
        result = target_from_attribute(pending)
        self.assertIsInstance(result, Target)
        result_dict = result.to_dict()
        self.assertEqual(result_dict["attribute"]["operation"], "create")
        self.assertEqual(result_dict["attribute"]["name"], "new_column")

    def test_target_from_block_model_existing_attribute(self):
        mock_bm = MagicMock()
        mock_bm.metadata.url = "https://example.com/blockmodel"
        bm_attr = BlockModelAttribute(name="grade", attribute_type="Float64", obj=mock_bm)
        result = target_from_attribute(bm_attr)
        self.assertIsInstance(result, Target)
        result_dict = result.to_dict()
        self.assertEqual(result_dict["attribute"]["operation"], "update")
        self.assertEqual(result_dict["attribute"]["reference"], "attributes[?name=='grade']")

    def test_target_from_block_model_pending_attribute(self):
        mock_bm = MagicMock()
        mock_bm.metadata.url = "https://example.com/blockmodel"
        bm_pending = BlockModelPendingAttribute(obj=mock_bm, name="new_col")
        result = target_from_attribute(bm_pending)
        self.assertIsInstance(result, Target)
        result_dict = result.to_dict()
        self.assertEqual(result_dict["attribute"]["operation"], "create")
        self.assertEqual(result_dict["attribute"]["name"], "new_col")

    def test_target_from_attribute_raises_for_invalid_type(self):
        with self.assertRaises(TypeError):
            target_from_attribute("not_an_attribute")

    def test_target_from_attribute_raises_for_none_obj(self):
        bm_pending = BlockModelPendingAttribute(obj=None, name="new_col")
        with self.assertRaises(TypeError):
            target_from_attribute(bm_pending)


class TestKrigingParametersWithAttributes(TestCase):
    """Tests for KrigingParameters handling of typed attribute objects."""

    def test_kriging_params_with_pending_attribute_target(self):
        """Test KrigingParameters accepts PendingAttribute as target."""
        source = Source(object="https://example.com/pointset", attribute="locations.attributes[?name=='grade']")

        mock_obj = MagicMock()
        mock_obj.metadata.url = "https://example.com/grid"
        target_attr = _create_pending_attribute("kriged_grade", parent_obj=mock_obj)

        variogram = "https://example.com/variogram"
        search = SearchNeighborhood(
            ellipsoid=Ellipsoid(ranges=EllipsoidRanges(100, 100, 50)),
            max_samples=20,
        )

        params = KrigingParameters(
            source=source,
            target=target_attr,
            variogram=variogram,
            search=search,
        )

        params_dict = params.to_dict()
        self.assertEqual(params_dict["target"]["object"], "https://example.com/grid")
        self.assertEqual(params_dict["target"]["attribute"]["operation"], "create")
        self.assertEqual(params_dict["target"]["attribute"]["name"], "kriged_grade")

    def test_kriging_params_with_existing_attribute_target(self):
        """Test KrigingParameters accepts existing Attribute as target."""
        source = Source(object="https://example.com/pointset", attribute="locations.attributes[?name=='grade']")
        target_attr = _create_mock_source_attribute(
            name="existing_attr",
            key="exist-key",
            object_url="https://example.com/grid",
            schema_path="locations.attributes",
        )

        variogram = "https://example.com/variogram"
        search = SearchNeighborhood(
            ellipsoid=Ellipsoid(ranges=EllipsoidRanges(100, 100, 50)),
            max_samples=20,
        )

        params = KrigingParameters(
            source=source,
            target=target_attr,
            variogram=variogram,
            search=search,
        )

        params_dict = params.to_dict()
        self.assertEqual(params_dict["target"]["object"], "https://example.com/grid")
        self.assertEqual(params_dict["target"]["attribute"]["operation"], "update")
        self.assertIn("reference", params_dict["target"]["attribute"])

    def test_kriging_params_with_block_model_pending_attribute(self):
        """Test KrigingParameters accepts BlockModelPendingAttribute as target."""
        source = Source(object="https://example.com/pointset", attribute="locations.attributes[?name=='grade']")

        mock_bm = MagicMock()
        mock_bm.metadata.url = "https://example.com/blockmodel"
        target_attr = BlockModelPendingAttribute(obj=mock_bm, name="new_bm_attr")

        variogram = "https://example.com/variogram"
        search = SearchNeighborhood(
            ellipsoid=Ellipsoid(ranges=EllipsoidRanges(100, 100, 50)),
            max_samples=20,
        )

        params = KrigingParameters(
            source=source,
            target=target_attr,
            variogram=variogram,
            search=search,
        )

        params_dict = params.to_dict()
        self.assertEqual(params_dict["target"]["object"], "https://example.com/blockmodel")
        self.assertEqual(params_dict["target"]["attribute"]["operation"], "create")
        self.assertEqual(params_dict["target"]["attribute"]["name"], "new_bm_attr")

    def test_kriging_params_with_block_model_existing_attribute(self):
        """Test KrigingParameters accepts existing BlockModelAttribute as target."""
        source = Source(object="https://example.com/pointset", attribute="locations.attributes[?name=='grade']")

        mock_bm = MagicMock()
        mock_bm.metadata.url = "https://example.com/blockmodel"
        target_attr = BlockModelAttribute(
            name="existing_bm_attr",
            attribute_type="Float64",
            obj=mock_bm,
        )

        variogram = "https://example.com/variogram"
        search = SearchNeighborhood(
            ellipsoid=Ellipsoid(ranges=EllipsoidRanges(100, 100, 50)),
            max_samples=20,
        )

        params = KrigingParameters(
            source=source,
            target=target_attr,
            variogram=variogram,
            search=search,
        )

        params_dict = params.to_dict()
        self.assertEqual(params_dict["target"]["object"], "https://example.com/blockmodel")
        self.assertEqual(params_dict["target"]["attribute"]["operation"], "update")
        self.assertIn("reference", params_dict["target"]["attribute"])

    def test_kriging_params_with_explicit_target(self):
        """Test KrigingParameters still works with explicit Target object."""
        source = Source(object="https://example.com/pointset", attribute="locations.attributes[?name=='grade']")
        target = Target.new_attribute("https://example.com/grid", "kriged_grade")
        variogram = "https://example.com/variogram"
        search = SearchNeighborhood(
            ellipsoid=Ellipsoid(ranges=EllipsoidRanges(100, 100, 50)),
            max_samples=20,
        )

        params = KrigingParameters(
            source=source,
            target=target,
            variogram=variogram,
            search=search,
        )

        params_dict = params.to_dict()
        self.assertEqual(params_dict["target"]["object"], "https://example.com/grid")
        self.assertEqual(params_dict["target"]["attribute"]["operation"], "create")
        self.assertEqual(params_dict["target"]["attribute"]["name"], "kriged_grade")

    def test_kriging_params_source_attribute_conversion(self):
        """Test KrigingParameters converts source Attribute correctly."""
        source_attr = _create_mock_source_attribute(
            name="grade",
            key="grade-key",
            object_url="https://example.com/pointset",
            schema_path="locations.attributes",
        )

        target = Target.new_attribute("https://example.com/grid", "kriged_grade")
        variogram = "https://example.com/variogram"
        search = SearchNeighborhood(
            ellipsoid=Ellipsoid(ranges=EllipsoidRanges(100, 100, 50)),
            max_samples=20,
        )

        params = KrigingParameters(
            source=source_attr,
            target=target,
            variogram=variogram,
            search=search,
        )

        params_dict = params.to_dict()
        self.assertEqual(params_dict["source"]["object"], "https://example.com/pointset")
        self.assertEqual(params_dict["source"]["attribute"], "locations.attributes[?key=='grade-key']")


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


class TestRegionFilter(TestCase):
    """Tests for RegionFilter class."""

    def test_region_filter_with_names(self):
        """Test RegionFilter with category names."""
        region_filter = RegionFilter(
            attribute="domain_attribute",
            names=["LMS1", "LMS2"],
        )

        result = region_filter.to_dict()

        self.assertEqual(result["attribute"], "domain_attribute")
        self.assertEqual(result["names"], ["LMS1", "LMS2"])
        self.assertNotIn("values", result)

    def test_region_filter_with_values(self):
        """Test RegionFilter with integer values."""
        region_filter = RegionFilter(
            attribute="domain_code_attribute",
            values=[1, 2, 3],
        )

        result = region_filter.to_dict()

        self.assertEqual(result["attribute"], "domain_code_attribute")
        self.assertEqual(result["values"], [1, 2, 3])
        self.assertNotIn("names", result)

    def test_region_filter_with_block_model_attribute(self):
        """Test RegionFilter with a real BlockModelAttribute."""
        bm_attr = BlockModelAttribute(name="domain", attribute_type="category")

        region_filter = RegionFilter(
            attribute=bm_attr,
            names=["Zone1"],
        )

        result = region_filter.to_dict()

        self.assertEqual(result["attribute"], "attributes[?name=='domain']")
        self.assertEqual(result["names"], ["Zone1"])

    def test_region_filter_with_pointset_attribute(self):
        """Test RegionFilter with a PointSet Attribute (mock with spec)."""
        mock_attr = _create_mock_source_attribute(
            name="domain",
            key="domain-key",
            object_url="https://example.com/pointset",
            schema_path="locations.attributes",
        )

        region_filter = RegionFilter(
            attribute=mock_attr,
            names=["Domain1"],
        )

        result = region_filter.to_dict()

        self.assertEqual(result["attribute"], "locations.attributes[?key=='domain-key']")
        self.assertEqual(result["names"], ["Domain1"])

    def test_region_filter_with_pending_attribute(self):
        """Test RegionFilter with a PendingAttribute."""
        pending = _create_pending_attribute("domain")

        region_filter = RegionFilter(
            attribute=pending,
            names=["Zone1"],
        )

        result = region_filter.to_dict()

        self.assertEqual(result["attribute"], "attributes[?name=='domain']")
        self.assertEqual(result["names"], ["Zone1"])

    def test_region_filter_cannot_have_both_names_and_values(self):
        """Test RegionFilter raises error when both names and values are provided."""
        with self.assertRaises(ValueError) as context:
            RegionFilter(
                attribute="domain_attribute",
                names=["LMS1"],
                values=[1],
            )

        self.assertIn("Only one of 'names' or 'values' may be provided", str(context.exception))

    def test_region_filter_must_have_names_or_values(self):
        """Test RegionFilter raises error when neither names nor values are provided."""
        with self.assertRaises(ValueError) as context:
            RegionFilter(
                attribute="domain_attribute",
            )

        self.assertIn("One of 'names' or 'values' must be provided", str(context.exception))

    def test_region_filter_raises_for_unsupported_type(self):
        """Test RegionFilter raises TypeError for unsupported attribute type."""
        with self.assertRaises(TypeError):
            region_filter = RegionFilter(attribute=12345, names=["Zone1"])
            region_filter.to_dict()


class TestKrigingParametersWithRegionFilter(TestCase):
    """Tests for KrigingParameters with target region filter support."""

    def test_kriging_params_with_target_region_filter_names(self):
        """Test KrigingParameters with target region filter using category names."""
        source = Source(object="https://example.com/pointset", attribute="grade")
        target = Target.new_attribute("https://example.com/grid", "kriged_grade")
        variogram = "https://example.com/variogram"
        search = SearchNeighborhood(
            ellipsoid=Ellipsoid(ranges=EllipsoidRanges(100, 100, 50)),
            max_samples=20,
        )
        region_filter = RegionFilter(
            attribute="domain_attribute",
            names=["LMS1", "LMS2"],
        )

        params = KrigingParameters(
            source=source,
            target=target,
            variogram=variogram,
            search=search,
            target_region_filter=region_filter,
        )

        params_dict = params.to_dict()

        # Verify region filter is in target
        self.assertIn("region_filter", params_dict["target"])
        self.assertEqual(params_dict["target"]["region_filter"]["attribute"], "domain_attribute")
        self.assertEqual(params_dict["target"]["region_filter"]["names"], ["LMS1", "LMS2"])

    def test_kriging_params_with_target_region_filter_values(self):
        """Test KrigingParameters with target region filter using integer values."""
        source = Source(object="https://example.com/pointset", attribute="grade")
        target = Target.new_attribute("https://example.com/grid", "kriged_grade")
        variogram = "https://example.com/variogram"
        search = SearchNeighborhood(
            ellipsoid=Ellipsoid(ranges=EllipsoidRanges(100, 100, 50)),
            max_samples=20,
        )
        region_filter = RegionFilter(
            attribute="domain_code",
            values=[1, 2, 3],
        )

        params = KrigingParameters(
            source=source,
            target=target,
            variogram=variogram,
            search=search,
            target_region_filter=region_filter,
        )

        params_dict = params.to_dict()

        # Verify region filter is in target
        self.assertIn("region_filter", params_dict["target"])
        self.assertEqual(params_dict["target"]["region_filter"]["attribute"], "domain_code")
        self.assertEqual(params_dict["target"]["region_filter"]["values"], [1, 2, 3])

    def test_kriging_params_without_target_region_filter(self):
        """Test KrigingParameters without target region filter (default behavior)."""
        source = Source(object="https://example.com/pointset", attribute="grade")
        target = Target.new_attribute("https://example.com/grid", "kriged_grade")
        variogram = "https://example.com/variogram"
        search = SearchNeighborhood(
            ellipsoid=Ellipsoid(ranges=EllipsoidRanges(100, 100, 50)),
            max_samples=20,
        )

        params = KrigingParameters(
            source=source,
            target=target,
            variogram=variogram,
            search=search,
        )

        params_dict = params.to_dict()

        # Verify region filter is not present
        self.assertNotIn("region_filter", params_dict["target"])


class TestBlockDiscretisation(TestCase):
    """Tests for BlockDiscretisation class."""

    def test_default_values(self):
        """Test BlockDiscretisation defaults to 1x1x1."""
        bd = BlockDiscretisation()

        self.assertEqual(bd.nx, 1)
        self.assertEqual(bd.ny, 1)
        self.assertEqual(bd.nz, 1)

    def test_custom_values(self):
        """Test BlockDiscretisation with custom values."""
        bd = BlockDiscretisation(nx=3, ny=4, nz=2)

        self.assertEqual(bd.nx, 3)
        self.assertEqual(bd.ny, 4)
        self.assertEqual(bd.nz, 2)

    def test_maximum_values(self):
        """Test BlockDiscretisation with maximum values (9)."""
        bd = BlockDiscretisation(nx=9, ny=9, nz=9)

        self.assertEqual(bd.nx, 9)
        self.assertEqual(bd.ny, 9)
        self.assertEqual(bd.nz, 9)

    def test_to_dict(self):
        """Test BlockDiscretisation serializes correctly."""
        bd = BlockDiscretisation(nx=3, ny=3, nz=2)

        result = bd.to_dict()

        self.assertEqual(result, {"nx": 3, "ny": 3, "nz": 2})

    def test_to_dict_defaults(self):
        """Test BlockDiscretisation serializes default values."""
        bd = BlockDiscretisation()

        result = bd.to_dict()

        self.assertEqual(result, {"nx": 1, "ny": 1, "nz": 1})

    def test_validation_nx_too_low(self):
        """Test BlockDiscretisation rejects nx < 1."""
        with self.assertRaises(ValueError) as ctx:
            BlockDiscretisation(nx=0)

        self.assertIn("nx", str(ctx.exception))
        self.assertIn("between 1 and 9", str(ctx.exception))

    def test_validation_ny_too_high(self):
        """Test BlockDiscretisation rejects ny > 9."""
        with self.assertRaises(ValueError) as ctx:
            BlockDiscretisation(ny=10)

        self.assertIn("ny", str(ctx.exception))
        self.assertIn("between 1 and 9", str(ctx.exception))

    def test_validation_nz_negative(self):
        """Test BlockDiscretisation rejects negative nz."""
        with self.assertRaises(ValueError) as ctx:
            BlockDiscretisation(nz=-1)

        self.assertIn("nz", str(ctx.exception))
        self.assertIn("between 1 and 9", str(ctx.exception))

    def test_validation_non_integer_type(self):
        """Test BlockDiscretisation rejects non-integer types."""
        with self.assertRaises(TypeError) as ctx:
            BlockDiscretisation(nx=2.5)

        self.assertIn("nx", str(ctx.exception))
        self.assertIn("integer", str(ctx.exception))


class TestKrigingParametersWithBlockDiscretisation(TestCase):
    """Tests for KrigingParameters with block_discretisation support."""

    def test_kriging_params_with_block_discretisation(self):
        """Test KrigingParameters includes block_discretisation in to_dict."""
        source = Source(object="https://example.com/pointset", attribute="grade")
        target = Target.new_attribute("https://example.com/grid", "kriged_grade")
        variogram = "https://example.com/variogram"
        search = SearchNeighborhood(
            ellipsoid=Ellipsoid(ranges=EllipsoidRanges(100, 100, 50)),
            max_samples=20,
        )
        bd = BlockDiscretisation(nx=3, ny=3, nz=2)

        params = KrigingParameters(
            source=source,
            target=target,
            variogram=variogram,
            search=search,
            block_discretisation=bd,
        )

        params_dict = params.to_dict()

        self.assertIn("block_discretisation", params_dict)
        self.assertEqual(params_dict["block_discretisation"], {"nx": 3, "ny": 3, "nz": 2})

    def test_kriging_params_without_block_discretisation(self):
        """Test KrigingParameters omits block_discretisation when None (default)."""
        source = Source(object="https://example.com/pointset", attribute="grade")
        target = Target.new_attribute("https://example.com/grid", "kriged_grade")
        variogram = "https://example.com/variogram"
        search = SearchNeighborhood(
            ellipsoid=Ellipsoid(ranges=EllipsoidRanges(100, 100, 50)),
            max_samples=20,
        )

        params = KrigingParameters(
            source=source,
            target=target,
            variogram=variogram,
            search=search,
        )

        params_dict = params.to_dict()

        self.assertNotIn("block_discretisation", params_dict)

    def test_kriging_params_block_discretisation_with_region_filter(self):
        """Test KrigingParameters with both block_discretisation and region filter."""
        source = Source(object="https://example.com/pointset", attribute="grade")
        target = Target.new_attribute("https://example.com/grid", "kriged_grade")
        variogram = "https://example.com/variogram"
        search = SearchNeighborhood(
            ellipsoid=Ellipsoid(ranges=EllipsoidRanges(100, 100, 50)),
            max_samples=20,
        )
        bd = BlockDiscretisation(nx=2, ny=2, nz=2)
        region_filter = RegionFilter(
            attribute="domain_attribute",
            names=["LMS1"],
        )

        params = KrigingParameters(
            source=source,
            target=target,
            variogram=variogram,
            search=search,
            block_discretisation=bd,
            target_region_filter=region_filter,
        )

        params_dict = params.to_dict()

        # Both should be present
        self.assertIn("block_discretisation", params_dict)
        self.assertEqual(params_dict["block_discretisation"], {"nx": 2, "ny": 2, "nz": 2})
        self.assertIn("region_filter", params_dict["target"])
        self.assertEqual(params_dict["target"]["region_filter"]["names"], ["LMS1"])
