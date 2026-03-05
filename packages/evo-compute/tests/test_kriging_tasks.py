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

from evo.objects import ObjectReference
from evo.objects.typed.attributes import (
    Attribute,
    BlockModelAttribute,
    BlockModelPendingAttribute,
    PendingAttribute,
)
from pydantic import TypeAdapter, ValidationError

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
    AnySourceAttribute,
    AnyTargetAttribute,
    AttributeExpression,
    Ellipsoid,
    EllipsoidRanges,
)
from evo.compute.tasks.kriging import KrigingParameters

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

_BASE = "https://hub.test.evo.bentley.com"
_ORG = "00000000-0000-0000-0000-000000000001"
_WS = "00000000-0000-0000-0000-000000000002"


def _obj_url(obj_id: str = "00000000-0000-0000-0000-000000000003") -> str:
    """Return a valid ObjectReference URL for testing."""
    return f"{_BASE}/geoscience-object/orgs/{_ORG}/workspaces/{_WS}/objects/{obj_id}"


POINTSET_URL = _obj_url("00000000-0000-0000-0000-000000000010")
GRID_URL = _obj_url("00000000-0000-0000-0000-000000000020")
VARIOGRAM_URL = _obj_url("00000000-0000-0000-0000-000000000030")
BLOCKMODEL_URL = _obj_url("00000000-0000-0000-0000-000000000040")


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

    # Parent object — metadata.url must be a real ObjectReference
    mock_obj = MagicMock()
    mock_obj.metadata.url = ObjectReference(object_url)
    attr._obj = mock_obj

    return attr


def _create_pending_attribute(name: str, parent_obj: MagicMock | None = None) -> PendingAttribute:
    """Create a real PendingAttribute with an optional mock parent."""
    mock_parent = MagicMock()
    mock_parent._obj = parent_obj
    return PendingAttribute(mock_parent, name)


class TestAttributeExpression(TestCase):
    def setUp(self):
        self.ta: TypeAdapter[AttributeExpression] = TypeAdapter(AttributeExpression)

    def test_expression_for_attribute_with_schema_path(self):
        attr = _create_mock_source_attribute("grade", "abc-key", POINTSET_URL, schema_path="locations.attributes")
        result = self.ta.validate_python(attr)
        self.assertEqual(result, "locations.attributes[?key=='abc-key']")

    def test_expression_for_attribute_without_schema_path(self):
        attr = _create_mock_source_attribute("grade", "abc-key", POINTSET_URL, schema_path="")
        result = self.ta.validate_python(attr)
        self.assertEqual(result, "attributes[?key=='abc-key']")

    def test_expression_for_pending_attribute(self):
        pending = _create_pending_attribute("my_attribute")
        result = self.ta.validate_python(pending)
        self.assertEqual(result, "attributes[?name=='my_attribute']")

    def test_expression_for_block_model_attribute(self):
        bm_attr = BlockModelAttribute(name="grade", attribute_type="Float64")
        result = self.ta.validate_python(bm_attr)
        self.assertEqual(result, "attributes[?name=='grade']")

    def test_expression_for_block_model_pending_attribute(self):
        bm_pending = BlockModelPendingAttribute(obj=None, name="new_col")
        result = self.ta.validate_python(bm_pending)
        self.assertEqual(result, "attributes[?name=='new_col']")

    def test_expression_raises_for_invalid_type(self):
        with self.assertRaises(ValidationError):
            self.ta.validate_python(object())


class TestAnySourceAttribute(TestCase):
    def setUp(self):
        self.ta: TypeAdapter[AnySourceAttribute] = TypeAdapter(AnySourceAttribute)

    def test_source_from_existing_attribute(self):
        attr = _create_mock_source_attribute("grade", "abc-key", POINTSET_URL, schema_path="locations.attributes")
        result = self.ta.validate_python(attr)
        self.assertIsInstance(result, Source)
        result_dict = result.model_dump()
        self.assertEqual(result_dict["object"], POINTSET_URL)
        self.assertEqual(result_dict["attribute"], "locations.attributes[?key=='abc-key']")

    def test_source_from_attribute_without_schema_path(self):
        attr = _create_mock_source_attribute("grade", "abc-key", POINTSET_URL, schema_path="")
        result = self.ta.validate_python(attr)
        result_dict = result.model_dump()
        self.assertEqual(result_dict["object"], POINTSET_URL)
        self.assertEqual(result_dict["attribute"], "attributes[?key=='abc-key']")

    def test_source_from_attribute_raises_for_pending(self):
        pending = _create_pending_attribute("new_attr")
        with self.assertRaises(ValidationError):
            self.ta.validate_python(pending)

    def test_source_from_attribute_raises_for_block_model_attribute(self):
        bm_attr = BlockModelAttribute(name="grade", attribute_type="Float64")
        with self.assertRaises(ValidationError):
            self.ta.validate_python(bm_attr)

    def test_source_from_attribute_raises_for_string(self):
        with self.assertRaises(ValidationError):
            self.ta.validate_python("not_an_attribute")


class TestAnyTargetAttribute(TestCase):
    def setUp(self):
        self.ta: TypeAdapter[AnyTargetAttribute] = TypeAdapter(AnyTargetAttribute)

    def test_target_from_existing_attribute(self):
        attr = _create_mock_source_attribute("grade", "abc-key", GRID_URL, schema_path="locations.attributes")
        result = self.ta.validate_python(attr)
        self.assertIsInstance(result, Target)
        result_dict = result.model_dump()
        self.assertEqual(result_dict["attribute"]["operation"], "update")
        self.assertEqual(result_dict["attribute"]["reference"], "locations.attributes[?key=='abc-key']")

    def test_target_from_pending_attribute(self):
        mock_obj = MagicMock()
        mock_obj.metadata.url = ObjectReference(GRID_URL)
        pending = _create_pending_attribute("new_column", parent_obj=mock_obj)
        result = self.ta.validate_python(pending)
        self.assertIsInstance(result, Target)
        result_dict = result.model_dump()
        self.assertEqual(result_dict["attribute"]["operation"], "create")
        self.assertEqual(result_dict["attribute"]["name"], "new_column")

    def test_target_from_block_model_existing_attribute(self):
        mock_bm = MagicMock()
        mock_bm.metadata.url = ObjectReference(BLOCKMODEL_URL)
        bm_attr = BlockModelAttribute(name="grade", attribute_type="Float64", obj=mock_bm)
        result = self.ta.validate_python(bm_attr)
        self.assertIsInstance(result, Target)
        result_dict = result.model_dump()
        self.assertEqual(result_dict["attribute"]["operation"], "update")
        self.assertEqual(result_dict["attribute"]["reference"], "attributes[?name=='grade']")

    def test_target_from_block_model_pending_attribute(self):
        mock_bm = MagicMock()
        mock_bm.metadata.url = ObjectReference(BLOCKMODEL_URL)
        bm_pending = BlockModelPendingAttribute(obj=mock_bm, name="new_col")
        result = self.ta.validate_python(bm_pending)
        self.assertIsInstance(result, Target)
        result_dict = result.model_dump()
        self.assertEqual(result_dict["attribute"]["operation"], "create")
        self.assertEqual(result_dict["attribute"]["name"], "new_col")

    def test_target_from_attribute_raises_for_invalid_type(self):
        with self.assertRaises(ValidationError):
            self.ta.validate_python("not_an_attribute")

    def test_target_from_attribute_raises_for_none_obj(self):
        bm_pending = BlockModelPendingAttribute(obj=None, name="new_col")
        with self.assertRaises(ValidationError):
            self.ta.validate_python(bm_pending)


class TestKrigingParametersWithAttributes(TestCase):
    """Tests for KrigingParameters handling of typed attribute objects."""

    def test_kriging_params_with_pending_attribute_target(self):
        """Test KrigingParameters accepts PendingAttribute as target."""
        source = Source(object=POINTSET_URL, attribute="locations.attributes[?name=='grade']")

        mock_obj = MagicMock()
        mock_obj.metadata.url = ObjectReference(GRID_URL)
        target_attr = _create_pending_attribute("kriged_grade", parent_obj=mock_obj)

        search = SearchNeighborhood(
            ellipsoid=Ellipsoid(ranges=EllipsoidRanges(major=100, semi_major=100, minor=50)),
            max_samples=20,
        )

        params = KrigingParameters(
            source=source,
            target=target_attr,
            variogram=VARIOGRAM_URL,
            search=search,
        )

        params_dict = params.model_dump(mode="json", by_alias=True, exclude_none=True)
        self.assertEqual(params_dict["target"]["object"], GRID_URL)
        self.assertEqual(params_dict["target"]["attribute"]["operation"], "create")
        self.assertEqual(params_dict["target"]["attribute"]["name"], "kriged_grade")

    def test_kriging_params_with_existing_attribute_target(self):
        """Test KrigingParameters accepts existing Attribute as target."""
        source = Source(object=POINTSET_URL, attribute="locations.attributes[?name=='grade']")
        target_attr = _create_mock_source_attribute(
            name="existing_attr",
            key="exist-key",
            object_url=GRID_URL,
            schema_path="locations.attributes",
        )

        search = SearchNeighborhood(
            ellipsoid=Ellipsoid(ranges=EllipsoidRanges(major=100, semi_major=100, minor=50)),
            max_samples=20,
        )

        params = KrigingParameters(
            source=source,
            target=target_attr,
            variogram=VARIOGRAM_URL,
            search=search,
        )

        params_dict = params.model_dump()
        self.assertEqual(params_dict["target"]["object"], GRID_URL)
        self.assertEqual(params_dict["target"]["attribute"]["operation"], "update")
        self.assertIn("reference", params_dict["target"]["attribute"])

    def test_kriging_params_with_block_model_pending_attribute(self):
        """Test KrigingParameters accepts BlockModelPendingAttribute as target."""
        source = Source(object=POINTSET_URL, attribute="locations.attributes[?name=='grade']")

        mock_bm = MagicMock()
        mock_bm.metadata.url = ObjectReference(BLOCKMODEL_URL)
        target_attr = BlockModelPendingAttribute(obj=mock_bm, name="new_bm_attr")

        search = SearchNeighborhood(
            ellipsoid=Ellipsoid(ranges=EllipsoidRanges(major=100, semi_major=100, minor=50)),
            max_samples=20,
        )

        params = KrigingParameters(
            source=source,
            target=target_attr,
            variogram=VARIOGRAM_URL,
            search=search,
        )

        params_dict = params.model_dump()
        self.assertEqual(params_dict["target"]["object"], BLOCKMODEL_URL)
        self.assertEqual(params_dict["target"]["attribute"]["operation"], "create")
        self.assertEqual(params_dict["target"]["attribute"]["name"], "new_bm_attr")

    def test_kriging_params_with_block_model_existing_attribute(self):
        """Test KrigingParameters accepts existing BlockModelAttribute as target."""
        source = Source(object=POINTSET_URL, attribute="locations.attributes[?name=='grade']")

        mock_bm = MagicMock()
        mock_bm.metadata.url = ObjectReference(BLOCKMODEL_URL)
        target_attr = BlockModelAttribute(
            name="existing_bm_attr",
            attribute_type="Float64",
            obj=mock_bm,
        )

        search = SearchNeighborhood(
            ellipsoid=Ellipsoid(ranges=EllipsoidRanges(major=100, semi_major=100, minor=50)),
            max_samples=20,
        )

        params = KrigingParameters(
            source=source,
            target=target_attr,
            variogram=VARIOGRAM_URL,
            search=search,
        )

        params_dict = params.model_dump()
        self.assertEqual(params_dict["target"]["object"], BLOCKMODEL_URL)
        self.assertEqual(params_dict["target"]["attribute"]["operation"], "update")
        self.assertIn("reference", params_dict["target"]["attribute"])

    def test_kriging_params_with_explicit_target(self):
        """Test KrigingParameters still works with explicit Target object."""
        source = Source(object=POINTSET_URL, attribute="locations.attributes[?name=='grade']")
        target = Target.new_attribute(GRID_URL, "kriged_grade")

        search = SearchNeighborhood(
            ellipsoid=Ellipsoid(ranges=EllipsoidRanges(major=100, semi_major=100, minor=50)),
            max_samples=20,
        )

        params = KrigingParameters(
            source=source,
            target=target,
            variogram=VARIOGRAM_URL,
            search=search,
        )

        params_dict = params.model_dump()
        self.assertEqual(params_dict["target"]["object"], GRID_URL)
        self.assertEqual(params_dict["target"]["attribute"]["operation"], "create")
        self.assertEqual(params_dict["target"]["attribute"]["name"], "kriged_grade")

    def test_kriging_params_source_attribute_conversion(self):
        """Test KrigingParameters converts source Attribute correctly."""
        source_attr = _create_mock_source_attribute(
            name="grade",
            key="grade-key",
            object_url=POINTSET_URL,
            schema_path="locations.attributes",
        )

        target = Target.new_attribute(GRID_URL, "kriged_grade")

        search = SearchNeighborhood(
            ellipsoid=Ellipsoid(ranges=EllipsoidRanges(major=100, semi_major=100, minor=50)),
            max_samples=20,
        )

        params = KrigingParameters(
            source=source_attr,
            target=target,
            variogram=VARIOGRAM_URL,
            search=search,
        )

        params_dict = params.model_dump()
        self.assertEqual(params_dict["source"]["object"], POINTSET_URL)
        self.assertEqual(params_dict["source"]["attribute"], "locations.attributes[?key=='grade-key']")


class TestTargetSerialization(TestCase):
    """Tests for Target serialization with different attribute types."""

    def test_target_with_create_attribute(self):
        """Test Target serializes CreateAttribute correctly."""
        target = Target(
            object=GRID_URL,
            attribute=CreateAttribute(name="new_attr"),
        )

        result = target.model_dump()

        self.assertEqual(result["object"], GRID_URL)
        self.assertEqual(result["attribute"]["operation"], "create")
        self.assertEqual(result["attribute"]["name"], "new_attr")

    def test_target_with_update_attribute(self):
        """Test Target serializes UpdateAttribute correctly."""
        target = Target(
            object=GRID_URL,
            attribute=UpdateAttribute(reference="cell_attributes[?name=='existing']"),
        )

        result = target.model_dump()

        self.assertEqual(result["object"], GRID_URL)
        self.assertEqual(result["attribute"]["operation"], "update")
        self.assertEqual(result["attribute"]["reference"], "cell_attributes[?name=='existing']")

    def test_target_new_attribute_factory(self):
        """Test Target.new_attribute factory method."""
        target = Target.new_attribute(GRID_URL, "new_attr")

        result = target.model_dump()

        self.assertEqual(result["object"], GRID_URL)
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

        result = region_filter.model_dump(mode="json", by_alias=True, exclude_none=True)

        self.assertEqual(result["attribute"], "domain_attribute")
        self.assertEqual(result["names"], ["LMS1", "LMS2"])
        self.assertNotIn("values", result)

    def test_region_filter_with_values(self):
        """Test RegionFilter with integer values."""
        region_filter = RegionFilter(
            attribute="domain_code_attribute",
            values=[1, 2, 3],
        )

        result = region_filter.model_dump(mode="json", by_alias=True, exclude_none=True)

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

        result = region_filter.model_dump()

        self.assertEqual(result["attribute"], "attributes[?name=='domain']")
        self.assertEqual(result["names"], ["Zone1"])

    def test_region_filter_with_pointset_attribute(self):
        """Test RegionFilter with a PointSet Attribute (mock with spec)."""
        mock_attr = _create_mock_source_attribute(
            name="domain",
            key="domain-key",
            object_url=POINTSET_URL,
            schema_path="locations.attributes",
        )

        region_filter = RegionFilter(
            attribute=mock_attr,
            names=["Domain1"],
        )

        result = region_filter.model_dump()

        self.assertEqual(result["attribute"], "locations.attributes[?key=='domain-key']")
        self.assertEqual(result["names"], ["Domain1"])

    def test_region_filter_with_pending_attribute(self):
        """Test RegionFilter rejects PendingAttribute (not a valid attribute type for filtering)."""
        pending = _create_pending_attribute("domain")

        with self.assertRaises(ValueError):
            RegionFilter(
                attribute=pending,
                names=["Zone1"],
            )

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
        """Test RegionFilter rejects unsupported attribute types at construction."""
        with self.assertRaises((TypeError, Exception)):
            RegionFilter(attribute=12345, names=["Zone1"])


class TestKrigingParametersWithRegionFilter(TestCase):
    """Tests for KrigingParameters with target region filter support."""

    def test_kriging_params_with_target_region_filter_names(self):
        """Test KrigingParameters with target region filter using category names."""
        source = Source(object=POINTSET_URL, attribute="grade")
        target = Target.new_attribute(GRID_URL, "kriged_grade")
        search = SearchNeighborhood(
            ellipsoid=Ellipsoid(ranges=EllipsoidRanges(major=100, semi_major=100, minor=50)),
            max_samples=20,
        )
        region_filter = RegionFilter(
            attribute="domain_attribute",
            names=["LMS1", "LMS2"],
        )

        params = KrigingParameters(
            source=source,
            target=target,
            variogram=VARIOGRAM_URL,
            search=search,
            target_region_filter=region_filter,
        )

        params_dict = params.model_dump()

        self.assertIn("region_filter", params_dict["target"])
        self.assertEqual(params_dict["target"]["region_filter"]["attribute"], "domain_attribute")
        self.assertEqual(params_dict["target"]["region_filter"]["names"], ["LMS1", "LMS2"])

    def test_kriging_params_with_target_region_filter_values(self):
        """Test KrigingParameters with target region filter using integer values."""
        source = Source(object=POINTSET_URL, attribute="grade")
        target = Target.new_attribute(GRID_URL, "kriged_grade")
        search = SearchNeighborhood(
            ellipsoid=Ellipsoid(ranges=EllipsoidRanges(major=100, semi_major=100, minor=50)),
            max_samples=20,
        )
        region_filter = RegionFilter(
            attribute="domain_code",
            values=[1, 2, 3],
        )

        params = KrigingParameters(
            source=source,
            target=target,
            variogram=VARIOGRAM_URL,
            search=search,
            target_region_filter=region_filter,
        )

        params_dict = params.model_dump()

        self.assertIn("region_filter", params_dict["target"])
        self.assertEqual(params_dict["target"]["region_filter"]["attribute"], "domain_code")
        self.assertEqual(params_dict["target"]["region_filter"]["values"], [1, 2, 3])

    def test_kriging_params_without_target_region_filter(self):
        """Test KrigingParameters without target region filter (default behavior)."""
        source = Source(object=POINTSET_URL, attribute="grade")
        target = Target.new_attribute(GRID_URL, "kriged_grade")
        search = SearchNeighborhood(
            ellipsoid=Ellipsoid(ranges=EllipsoidRanges(major=100, semi_major=100, minor=50)),
            max_samples=20,
        )

        params = KrigingParameters(
            source=source,
            target=target,
            variogram=VARIOGRAM_URL,
            search=search,
        )

        params_dict = params.model_dump()

        self.assertNotIn("region_filter", params_dict["target"])


class TestBlockDiscretisation(TestCase):
    """Tests for BlockDiscretisation class."""

    def test_default_values(self):
        bd = BlockDiscretisation()
        self.assertEqual(bd.nx, 1)
        self.assertEqual(bd.ny, 1)
        self.assertEqual(bd.nz, 1)

    def test_custom_values(self):
        bd = BlockDiscretisation(nx=3, ny=4, nz=2)
        self.assertEqual(bd.nx, 3)
        self.assertEqual(bd.ny, 4)
        self.assertEqual(bd.nz, 2)

    def test_maximum_values(self):
        bd = BlockDiscretisation(nx=9, ny=9, nz=9)
        self.assertEqual(bd.nx, 9)
        self.assertEqual(bd.ny, 9)
        self.assertEqual(bd.nz, 9)

    def test_model_dump(self):
        bd = BlockDiscretisation(nx=3, ny=3, nz=2)
        result = bd.model_dump()
        self.assertEqual(result, {"nx": 3, "ny": 3, "nz": 2})

    def test_model_dump_defaults(self):
        bd = BlockDiscretisation()
        result = bd.model_dump()
        self.assertEqual(result, {"nx": 1, "ny": 1, "nz": 1})

    def test_validation_nx_too_low(self):
        with self.assertRaises(ValidationError) as ctx:
            BlockDiscretisation(nx=0)
        error_str = str(ctx.exception)
        self.assertIn("nx", error_str)

    def test_validation_ny_too_high(self):
        with self.assertRaises(ValidationError) as ctx:
            BlockDiscretisation(ny=10)
        error_str = str(ctx.exception)
        self.assertIn("ny", error_str)

    def test_validation_nz_negative(self):
        with self.assertRaises(ValidationError) as ctx:
            BlockDiscretisation(nz=-1)
        error_str = str(ctx.exception)
        self.assertIn("nz", error_str)

    def test_validation_non_integer_type(self):
        with self.assertRaises((TypeError, ValueError)):
            BlockDiscretisation(nx=2.5)


class TestKrigingParametersWithBlockDiscretisation(TestCase):
    """Tests for KrigingParameters with block_discretisation support."""

    def test_kriging_params_with_block_discretisation(self):
        source = Source(object=POINTSET_URL, attribute="grade")
        target = Target.new_attribute(GRID_URL, "kriged_grade")
        search = SearchNeighborhood(
            ellipsoid=Ellipsoid(ranges=EllipsoidRanges(major=100, semi_major=100, minor=50)),
            max_samples=20,
        )
        bd = BlockDiscretisation(nx=3, ny=3, nz=2)

        params = KrigingParameters(
            source=source,
            target=target,
            variogram=VARIOGRAM_URL,
            search=search,
            block_discretisation=bd,
        )

        params_dict = params.model_dump()
        self.assertIn("block_discretisation", params_dict)
        self.assertEqual(params_dict["block_discretisation"], {"nx": 3, "ny": 3, "nz": 2})

    def test_kriging_params_without_block_discretisation(self):
        source = Source(object=POINTSET_URL, attribute="grade")
        target = Target.new_attribute(GRID_URL, "kriged_grade")
        search = SearchNeighborhood(
            ellipsoid=Ellipsoid(ranges=EllipsoidRanges(major=100, semi_major=100, minor=50)),
            max_samples=20,
        )

        params = KrigingParameters(
            source=source,
            target=target,
            variogram=VARIOGRAM_URL,
            search=search,
        )

        params_dict = params.model_dump(mode="json", by_alias=True, exclude_none=True)
        self.assertNotIn("block_discretisation", params_dict)

    def test_kriging_params_block_discretisation_with_region_filter(self):
        source = Source(object=POINTSET_URL, attribute="grade")
        target = Target.new_attribute(GRID_URL, "kriged_grade")
        search = SearchNeighborhood(
            ellipsoid=Ellipsoid(ranges=EllipsoidRanges(major=100, semi_major=100, minor=50)),
            max_samples=20,
        )
        bd = BlockDiscretisation(nx=2, ny=2, nz=2)
        region_filter = RegionFilter(attribute="domain_attribute", names=["LMS1"])

        params = KrigingParameters(
            source=source,
            target=target,
            variogram=VARIOGRAM_URL,
            search=search,
            block_discretisation=bd,
            target_region_filter=region_filter,
        )

        params_dict = params.model_dump()
        self.assertIn("block_discretisation", params_dict)
        self.assertEqual(params_dict["block_discretisation"], {"nx": 2, "ny": 2, "nz": 2})
        self.assertIn("region_filter", params_dict["target"])
        self.assertEqual(params_dict["target"]["region_filter"]["names"], ["LMS1"])


class TestObjectReferenceValidation(TestCase):
    """Tests for strict ObjectReference validation on GeoscienceObjectReference fields."""

    def test_valid_object_reference_accepted(self):
        """Source should accept a valid ObjectReference URL string."""
        source = Source(object=POINTSET_URL, attribute="grade")
        self.assertIsInstance(source.object, str)
        self.assertEqual(source.object, POINTSET_URL)

    def test_invalid_url_rejected(self):
        """Source should reject an invalid URL that cannot be parsed as ObjectReference."""
        with self.assertRaises(Exception):
            Source(object="https://example.com/not-valid", attribute="grade")

    def test_plain_string_rejected(self):
        """Source should reject a plain string that is not a valid object reference."""
        with self.assertRaises(Exception):
            Source(object="not_a_url", attribute="grade")

    def test_integer_rejected(self):
        """Source should reject an integer."""
        with self.assertRaises((TypeError, Exception)):
            Source(object=12345, attribute="grade")


class TestSearchNeighborhoodAlias(TestCase):
    """Tests for the search/neighborhood alias on KrigingParameters."""

    def _make_params(self, **kwargs):
        """Create KrigingParameters with default values, overridable via kwargs."""
        search = SearchNeighborhood(
            ellipsoid=Ellipsoid(ranges=EllipsoidRanges(major=100, semi_major=100, minor=50)),
            max_samples=20,
        )
        defaults = dict(
            source=Source(object=POINTSET_URL, attribute="grade"),
            target=Target.new_attribute(GRID_URL, "kriged_grade"),
            variogram=VARIOGRAM_URL,
            search=search,
        )
        defaults.update(kwargs)
        return KrigingParameters(**defaults)

    def test_search_serializes_as_neighborhood_with_by_alias(self):
        """When by_alias=True, the 'search' field should serialize as 'neighborhood'."""
        params = self._make_params()
        params_dict = params.model_dump(mode="json", by_alias=True, exclude_none=True)
        self.assertIn("neighborhood", params_dict)
        self.assertNotIn("search", params_dict)

    def test_search_serializes_as_search_without_by_alias(self):
        """When by_alias=False (default), the field should serialize as 'search'."""
        params = self._make_params()
        params_dict = params.model_dump()
        self.assertIn("search", params_dict)

    def test_can_construct_with_field_name_search(self):
        """Users should be able to construct KrigingParameters with search=..."""
        params = self._make_params()
        self.assertIsNotNone(params.search)

    def test_can_construct_with_alias_neighborhood(self):
        """Users should also be able to construct with neighborhood=..."""
        search = SearchNeighborhood(
            ellipsoid=Ellipsoid(ranges=EllipsoidRanges(major=100, semi_major=100, minor=50)),
            max_samples=20,
        )
        params = KrigingParameters(
            source=Source(object=POINTSET_URL, attribute="grade"),
            target=Target.new_attribute(GRID_URL, "kriged_grade"),
            variogram=VARIOGRAM_URL,
            neighborhood=search,
        )
        self.assertIsNotNone(params.search)
