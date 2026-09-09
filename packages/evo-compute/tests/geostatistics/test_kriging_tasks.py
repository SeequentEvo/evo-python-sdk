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

from types import SimpleNamespace
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
    AllOfFilter,
    BlockDiscretisation,
    CreateAttribute,
    Filter,
    FilterCondition,
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
from evo.compute.tasks.geostatistics.kriging import (
    RECOMMENDED_DIAGNOSTIC_NAMES,
    KrigingDiagnostics,
    KrigingParameters,
    KrigingResult,
    KrigingResultModel,
)

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


def _create_mock_source_attribute(
    name: str, key: str, object_url: str, schema_path: str = "", attribute_type: str = "scalar"
) -> MagicMock:
    """Create a mock Attribute (existing) that passes isinstance checks.

    Uses ``spec=Attribute`` so ``isinstance(mock, Attribute)`` returns True.
    Sets the underlying properties that the adapter functions inspect.
    """
    attr = MagicMock(spec=Attribute)
    attr.name = name
    attr.key = key
    attr.exists = True
    attr.attribute_type = attribute_type

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
        with self.assertRaises(ValidationError) as ctx:
            self.ta.validate_python(pending)
        message = str(ctx.exception)
        self.assertIn("new_attr", message)
        self.assertIn("does not exist", message)

    def test_source_from_pending_lists_available_attributes(self):
        parent = MagicMock()
        parent.__iter__.return_value = iter([SimpleNamespace(name="grade"), SimpleNamespace(name="au_ppm")])
        pending = PendingAttribute(parent, "grde")
        with self.assertRaises(ValidationError) as ctx:
            self.ta.validate_python(pending)
        message = str(ctx.exception)
        self.assertIn("grde", message)
        self.assertIn("grade", message)
        self.assertIn("au_ppm", message)

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


class TestFilter(TestCase):
    """Tests for the generic Filter / FilterCondition models."""

    def test_filter_condition_membership(self):
        """Test a single membership FilterCondition."""
        f = Filter(where=FilterCondition(attribute="domain_attribute", operator="in", values=["LMS1", "LMS2"]))

        result = f.model_dump(mode="json", exclude_none=True)

        self.assertEqual(result["where"]["type"], "condition")
        self.assertEqual(result["where"]["attribute"], "domain_attribute")
        self.assertEqual(result["where"]["operator"], "in")
        self.assertEqual(result["where"]["values"], ["LMS1", "LMS2"])
        self.assertNotIn("threshold", result["where"])

    def test_filter_condition_numeric(self):
        """Test a single numeric FilterCondition."""
        f = Filter(where=FilterCondition(attribute="grade", operator="greater_than_or_equal_to", threshold=0.5))

        result = f.model_dump(mode="json", exclude_none=True)

        self.assertEqual(result["where"]["operator"], "greater_than_or_equal_to")
        self.assertEqual(result["where"]["threshold"], 0.5)
        self.assertNotIn("values", result["where"])

    def test_filter_condition_integer_values(self):
        """Test membership FilterCondition with integer category keys."""
        f = Filter(where=FilterCondition(attribute="domain_code_attribute", operator="not_in", values=[1, 2, 3]))

        result = f.model_dump(mode="json", exclude_none=True)

        self.assertEqual(result["where"]["values"], [1, 2, 3])

    def test_filter_composite_all_of(self):
        """Test a composite all_of (AND) filter expression."""
        f = Filter(
            where=AllOfFilter(
                filters=[
                    FilterCondition(attribute="domain", operator="in", values=[1, 2]),
                    FilterCondition(attribute="grade", operator="greater_than", threshold=0.1),
                ],
            )
        )

        result = f.model_dump(mode="json", exclude_none=True)

        self.assertEqual(result["where"]["type"], "all_of")
        self.assertEqual(len(result["where"]["filters"]), 2)
        self.assertEqual(result["where"]["filters"][0]["values"], [1, 2])
        self.assertEqual(result["where"]["filters"][1]["threshold"], 0.1)

    def test_filter_condition_with_block_model_attribute(self):
        """Test FilterCondition resolves a BlockModelAttribute to an expression."""
        bm_attr = BlockModelAttribute(name="domain", attribute_type="category")

        f = Filter(where=FilterCondition(attribute=bm_attr, operator="in", values=["Zone1"]))

        result = f.model_dump()

        self.assertEqual(result["where"]["attribute"], "attributes[?name=='domain']")
        self.assertEqual(result["where"]["values"], ["Zone1"])

    def test_filter_condition_with_pointset_attribute(self):
        """Test FilterCondition resolves a PointSet Attribute to a key-based expression."""
        mock_attr = _create_mock_source_attribute(
            name="domain",
            key="domain-key",
            object_url=POINTSET_URL,
            schema_path="locations.attributes",
        )

        f = Filter(where=FilterCondition(attribute=mock_attr, operator="in", values=["Domain1"]))

        result = f.model_dump()

        self.assertEqual(result["where"]["attribute"], "locations.attributes[?key=='domain-key']")

    def test_membership_operator_requires_values(self):
        """Membership operators must be paired with values, not threshold."""
        with self.assertRaises(ValueError):
            FilterCondition(attribute="domain_attribute", operator="in", threshold=1.0)

    def test_numeric_operator_requires_threshold(self):
        """Numeric operators must be paired with threshold, not values."""
        with self.assertRaises(ValueError):
            FilterCondition(attribute="grade", operator="greater_than", values=[1])


class TestKrigingParametersWithFilter(TestCase):
    """Tests for KrigingParameters with source/target filter support."""

    def _search(self) -> SearchNeighborhood:
        return SearchNeighborhood(
            ellipsoid=Ellipsoid(ranges=EllipsoidRanges(major=100, semi_major=100, minor=50)),
            max_samples=20,
        )

    def test_kriging_params_with_target_filter(self):
        """Test KrigingParameters serializes a target filter under target.filter."""
        params = KrigingParameters(
            source=Source(object=POINTSET_URL, attribute="grade"),
            target=Target.new_attribute(GRID_URL, "kriged_grade"),
            variogram=VARIOGRAM_URL,
            search=self._search(),
            target_filter=Filter(
                where=FilterCondition(attribute="domain_attribute", operator="in", values=["LMS1", "LMS2"]),
            ),
        )

        params_dict = params.model_dump()

        self.assertIn("filter", params_dict["target"])
        self.assertEqual(params_dict["target"]["filter"]["where"]["attribute"], "domain_attribute")
        self.assertEqual(params_dict["target"]["filter"]["where"]["values"], ["LMS1", "LMS2"])
        self.assertNotIn("filter", params_dict["source"])

    def test_kriging_params_with_source_filter(self):
        """Test KrigingParameters serializes a source filter under source.filter."""
        params = KrigingParameters(
            source=Source(object=POINTSET_URL, attribute="grade"),
            target=Target.new_attribute(GRID_URL, "kriged_grade"),
            variogram=VARIOGRAM_URL,
            search=self._search(),
            source_filter=Filter(
                where=FilterCondition(attribute="grade", operator="greater_than", threshold=0.0),
            ),
        )

        params_dict = params.model_dump()

        self.assertIn("filter", params_dict["source"])
        self.assertEqual(params_dict["source"]["filter"]["where"]["threshold"], 0.0)
        self.assertNotIn("filter", params_dict["target"])

    def test_kriging_params_without_filter(self):
        """Test KrigingParameters omits filters by default."""
        params = KrigingParameters(
            source=Source(object=POINTSET_URL, attribute="grade"),
            target=Target.new_attribute(GRID_URL, "kriged_grade"),
            variogram=VARIOGRAM_URL,
            search=self._search(),
        )

        params_dict = params.model_dump()

        self.assertNotIn("filter", params_dict["target"])
        self.assertNotIn("filter", params_dict["source"])


class TestKrigingDiagnostics(TestCase):
    """Tests for the diagnostics selected on KrigingParameters."""

    def _search(self) -> SearchNeighborhood:
        return SearchNeighborhood(
            ellipsoid=Ellipsoid(ranges=EllipsoidRanges(major=100, semi_major=100, minor=50)),
            max_samples=20,
        )

    def _params(self, **kwargs) -> KrigingParameters:
        return KrigingParameters(
            source=Source(object=POINTSET_URL, attribute="grade"),
            target=Target.new_attribute(GRID_URL, "kriged_grade"),
            variogram=VARIOGRAM_URL,
            search=self._search(),
            **kwargs,
        )

    def test_true_uses_the_recommended_name(self):
        """``True`` selects a diagnostic using its Leapfrog-aligned recommended name."""
        diagnostics = KrigingDiagnostics(kriging_variance=True, num_samples=True)

        self.assertEqual(diagnostics.kriging_variance, CreateAttribute(name="KV"))
        self.assertEqual(diagnostics.num_samples, CreateAttribute(name="NS"))

    def test_every_diagnostic_has_a_recommended_name(self):
        """Every field can be selected with ``True``, so the name table must be complete."""
        fields = set(KrigingDiagnostics.model_fields)

        self.assertEqual(fields, set(RECOMMENDED_DIAGNOSTIC_NAMES))

        diagnostics = KrigingDiagnostics(**dict.fromkeys(fields, True))
        for field, expected in RECOMMENDED_DIAGNOSTIC_NAMES.items():
            self.assertEqual(getattr(diagnostics, field), CreateAttribute(name=expected), field)

    def test_string_creates_an_attribute_with_that_name(self):
        diagnostics = KrigingDiagnostics(kriging_variance="my_variance")

        self.assertEqual(diagnostics.kriging_variance, CreateAttribute(name="my_variance"))

    def test_false_and_none_select_nothing(self):
        diagnostics = KrigingDiagnostics(kriging_variance=False, num_samples=None)

        self.assertIsNone(diagnostics.kriging_variance)
        self.assertIsNone(diagnostics.num_samples)

    def test_existing_attribute_becomes_an_update(self):
        attr = _create_mock_source_attribute("KV", "kv-key", GRID_URL, schema_path="cell_attributes")

        diagnostics = KrigingDiagnostics(kriging_variance=attr)

        self.assertEqual(diagnostics.kriging_variance, UpdateAttribute(reference="cell_attributes[?key=='kv-key']"))

    def test_pending_attribute_becomes_a_create(self):
        pending = _create_pending_attribute("KV")

        diagnostics = KrigingDiagnostics(kriging_variance=pending)

        self.assertEqual(diagnostics.kriging_variance, CreateAttribute(name="KV"))

    def test_explicit_specifications_are_kept(self):
        diagnostics = KrigingDiagnostics(
            kriging_variance=CreateAttribute(name="KV"),
            num_samples=UpdateAttribute(reference="attributes[?name=='NS']"),
        )

        self.assertEqual(diagnostics.kriging_variance, CreateAttribute(name="KV"))
        self.assertEqual(diagnostics.num_samples, UpdateAttribute(reference="attributes[?name=='NS']"))

    def test_unknown_diagnostic_is_rejected(self):
        with self.assertRaises(ValidationError):
            KrigingDiagnostics(variance=True)

    def test_diagnostics_serialize_under_target(self):
        """Selected diagnostics are nested under ``target.diagnostics`` on the wire."""
        params = self._params(diagnostics=KrigingDiagnostics(kriging_variance=True, num_samples="sample_count"))

        params_dict = params.model_dump(mode="json", by_alias=True, exclude_none=True)

        self.assertNotIn("diagnostics", params_dict)
        self.assertEqual(
            params_dict["target"]["diagnostics"],
            {
                "kriging_variance": {"operation": "create", "name": "KV"},
                "num_samples": {"operation": "create", "name": "sample_count"},
            },
        )

    def test_unselected_diagnostics_are_not_sent(self):
        params = self._params(diagnostics=KrigingDiagnostics(kriging_variance=True))

        diagnostics = params.model_dump(mode="json", by_alias=True, exclude_none=True)["target"]["diagnostics"]

        self.assertEqual(list(diagnostics), ["kriging_variance"])

    def test_diagnostics_can_be_given_as_a_dict(self):
        params = self._params(diagnostics={"slope_of_regression": True})

        diagnostics = params.model_dump(mode="json", by_alias=True, exclude_none=True)["target"]["diagnostics"]

        self.assertEqual(diagnostics, {"slope_of_regression": {"operation": "create", "name": "SoR"}})

    def test_diagnostics_coexist_with_a_target_filter(self):
        params = self._params(
            diagnostics=KrigingDiagnostics(kriging_variance=True),
            target_filter=Filter(
                where=FilterCondition(attribute="domain_attribute", operator="in", values=["LMS1"]),
            ),
        )

        target = params.model_dump(mode="json", by_alias=True, exclude_none=True)["target"]

        self.assertIn("diagnostics", target)
        self.assertIn("filter", target)

    def test_params_without_diagnostics_send_nothing(self):
        params = self._params()

        params_dict = params.model_dump(mode="json", by_alias=True, exclude_none=True)

        self.assertNotIn("diagnostics", params_dict["target"])
        self.assertNotIn("diagnostics", params_dict)


class TestKrigingDiagnosticAttributeTypes(TestCase):
    """Tests that an existing attribute can hold the values of the diagnostic written to it."""

    GRID_TYPES = {"boolean": "bool", "integer": "integer", "floating-point": "scalar"}
    BLOCK_MODEL_TYPES = {"boolean": "Boolean", "integer": "Int32", "floating-point": "Float64"}

    def test_each_diagnostic_only_updates_attributes_that_hold_its_values(self):
        """Mirrors the task schema: booleans for ``valid``, integers for the counts, floats for the rest."""
        for diagnostic in KrigingDiagnostics.model_fields:
            if diagnostic == "valid":
                expected = "boolean"
            elif diagnostic.startswith("num_"):
                expected = "integer"
            else:
                expected = "floating-point"

            for kind in self.GRID_TYPES:
                candidates = [
                    _create_mock_source_attribute("existing", "key", GRID_URL, attribute_type=self.GRID_TYPES[kind]),
                    BlockModelAttribute(name="existing", attribute_type=self.BLOCK_MODEL_TYPES[kind]),
                ]
                for attr in candidates:
                    with self.subTest(diagnostic=diagnostic, attribute_type=attr.attribute_type):
                        if kind == expected:
                            diagnostics = KrigingDiagnostics(**{diagnostic: attr})
                            self.assertIsInstance(getattr(diagnostics, diagnostic), UpdateAttribute)
                        else:
                            with self.assertRaises(ValidationError):
                                KrigingDiagnostics(**{diagnostic: attr})

    def test_every_block_model_integer_type_holds_a_count(self):
        for attribute_type in ("Int8", "Int16", "Int32", "Int64", "UInt8", "UInt16", "UInt32", "UInt64"):
            with self.subTest(attribute_type=attribute_type):
                attr = BlockModelAttribute(name="NS", attribute_type=attribute_type)

                self.assertIsNotNone(KrigingDiagnostics(num_samples=attr).num_samples)

    def test_the_error_names_the_attribute_and_suggests_creating_one(self):
        attr = _create_mock_source_attribute("KV", "kv-key", GRID_URL, attribute_type="scalar")

        with self.assertRaises(ValidationError) as ctx:
            KrigingDiagnostics(valid=attr)

        self.assertIn(
            "Diagnostic 'valid' writes boolean values, so it cannot update 'KV', which is a scalar attribute. "
            "Pass a name to create a new attribute instead.",
            str(ctx.exception),
        )

    def test_a_pending_attribute_is_not_checked(self):
        """A new attribute is created with the diagnostic's type, so there is nothing to check."""
        diagnostics = KrigingDiagnostics(valid=_create_pending_attribute("valid"))

        self.assertEqual(diagnostics.valid, CreateAttribute(name="valid"))


class TestKrigingDiagnosticOutputValidation(TestCase):
    """Tests that every kriging output writes to its own attribute on the target object."""

    def _params(self, target: Target | Attribute | None = None, **kwargs) -> KrigingParameters:
        return KrigingParameters(
            source=Source(object=POINTSET_URL, attribute="grade"),
            target=target if target is not None else Target.new_attribute(GRID_URL, "kriged_grade"),
            variogram=VARIOGRAM_URL,
            search=SearchNeighborhood(
                ellipsoid=Ellipsoid(ranges=EllipsoidRanges(major=100, semi_major=100, minor=50)),
                max_samples=20,
            ),
            **kwargs,
        )

    def test_distinct_outputs_are_accepted(self):
        params = self._params(diagnostics=KrigingDiagnostics(kriging_variance=True, num_samples=True))

        self.assertEqual(list(params.model_dump()["target"]["diagnostics"]), ["kriging_variance", "num_samples"])

    def test_two_diagnostics_cannot_share_a_name(self):
        with self.assertRaises(ValidationError) as ctx:
            self._params(diagnostics=KrigingDiagnostics(kriging_variance=True, kriging_efficiency="KV"))

        self.assertIn("both write to the attribute 'KV'", str(ctx.exception))

    def test_a_diagnostic_cannot_take_the_name_of_the_estimate(self):
        with self.assertRaises(ValidationError) as ctx:
            self._params(
                target=Target.new_attribute(GRID_URL, "KV"),
                diagnostics=KrigingDiagnostics(kriging_variance=True),
            )

        self.assertIn("the estimate and diagnostic 'kriging_variance'", str(ctx.exception))

    def test_two_diagnostics_cannot_share_an_update_reference(self):
        reference = "cell_attributes[?name=='KV']"
        with self.assertRaises(ValidationError) as ctx:
            self._params(
                diagnostics=KrigingDiagnostics(
                    kriging_variance=UpdateAttribute(reference=reference),
                    kriging_efficiency=UpdateAttribute(reference=reference),
                )
            )

        self.assertIn(f"both write to the attribute {reference!r}", str(ctx.exception))

    def test_an_update_and_a_create_cannot_share_a_name(self):
        """A typed attribute is updated by key, so its own name is what reveals the collision."""
        attr = _create_mock_source_attribute("KV", "kv-key", GRID_URL, schema_path="cell_attributes")

        with self.assertRaises(ValidationError) as ctx:
            self._params(diagnostics=KrigingDiagnostics(kriging_variance=attr, kriging_efficiency="KV"))

        self.assertIn(
            "diagnostic 'kriging_variance' and diagnostic 'kriging_efficiency' both write to the attribute 'KV'",
            str(ctx.exception),
        )

    def test_a_diagnostic_cannot_take_the_name_of_an_existing_estimate_attribute(self):
        estimate = _create_mock_source_attribute("KV", "kv-key", GRID_URL, schema_path="cell_attributes")

        with self.assertRaises(ValidationError) as ctx:
            self._params(target=estimate, diagnostics=KrigingDiagnostics(kriging_variance=True))

        self.assertIn(
            "the estimate and diagnostic 'kriging_variance' both write to the attribute 'KV'",
            str(ctx.exception),
        )

    def test_a_hand_written_reference_is_only_compared_with_references(self):
        """A reference carries no attribute name, so it cannot be compared with a create name."""
        params = self._params(
            diagnostics=KrigingDiagnostics(
                kriging_variance="KV",
                kriging_efficiency=UpdateAttribute(reference="KV"),
            )
        )

        self.assertEqual(params.diagnostics.kriging_variance, CreateAttribute(name="KV"))

    def test_an_attribute_of_the_target_object_is_accepted(self):
        attr = _create_mock_source_attribute("KV", "kv-key", GRID_URL, schema_path="cell_attributes")

        params = self._params(diagnostics=KrigingDiagnostics(kriging_variance=attr))

        self.assertEqual(
            params.diagnostics.kriging_variance,
            UpdateAttribute(reference="cell_attributes[?key=='kv-key']"),
        )

    def test_an_attribute_of_another_object_is_rejected(self):
        attr = _create_mock_source_attribute("KV", "kv-key", POINTSET_URL, schema_path="locations.attributes")

        with self.assertRaises(ValidationError) as ctx:
            self._params(diagnostics=KrigingDiagnostics(kriging_variance=attr))

        self.assertIn("Diagnostic 'kriging_variance' references an attribute of a different object", str(ctx.exception))

    def test_a_pending_attribute_of_another_object_is_rejected(self):
        parent = MagicMock()
        parent.metadata.url = ObjectReference(POINTSET_URL)
        pending = _create_pending_attribute("KV", parent_obj=parent)

        with self.assertRaises(ValidationError) as ctx:
            self._params(diagnostics=KrigingDiagnostics(kriging_variance=pending))

        self.assertIn("references an attribute of a different object", str(ctx.exception))

    def test_a_versioned_target_url_still_matches(self):
        attr = _create_mock_source_attribute("KV", "kv-key", GRID_URL, schema_path="cell_attributes")

        params = self._params(
            target=Target.new_attribute(f"{GRID_URL}?version=17", "kriged_grade"),
            diagnostics=KrigingDiagnostics(kriging_variance=attr),
        )

        self.assertIsNotNone(params.diagnostics.kriging_variance)

    def test_an_attribute_without_a_parent_object_is_not_checked(self):
        """A standalone block model attribute carries no parent, so it cannot be verified."""
        attr = BlockModelAttribute(name="KV", attribute_type="Float64")

        params = self._params(diagnostics=KrigingDiagnostics(kriging_variance=attr))

        self.assertEqual(
            params.diagnostics.kriging_variance,
            UpdateAttribute(reference="attributes[?name=='KV']"),
        )


class TestKrigingDiagnosticsResult(TestCase):
    """Tests for the diagnostics returned in a kriging result."""

    def _result(self, diagnostics: dict | None) -> KrigingResult:
        target = {
            "reference": GRID_URL,
            "name": "block model",
            "schema_id": "/objects/regular-3d-grid/1.2.0/regular-3d-grid.schema.json",
            "attribute": {"reference": "cell_attributes[?key=='e-key']", "name": "kriged_grade"},
        }
        if diagnostics is not None:
            target["diagnostics"] = diagnostics
        return KrigingResult(MagicMock(), KrigingResultModel(message="ok", target=target))

    def test_requested_diagnostics_are_exposed(self):
        """The service returns every key, with ``null`` for diagnostics that were not requested."""
        result = self._result(
            {
                **dict.fromkeys(RECOMMENDED_DIAGNOSTIC_NAMES, None),
                "kriging_variance": {"reference": "cell_attributes[?key=='kv-key']", "name": "KV"},
            }
        )

        self.assertEqual(list(result.diagnostics), ["kriging_variance"])
        self.assertEqual(result.diagnostics["kriging_variance"].name, "KV")
        self.assertEqual(result.diagnostics["kriging_variance"].reference, "cell_attributes[?key=='kv-key']")

    def test_no_diagnostics_gives_an_empty_mapping(self):
        self.assertEqual(self._result(None).diagnostics, {})

    def test_diagnostics_are_listed_in_the_summary(self):
        result = self._result({"kriging_variance": {"reference": "ref", "name": "KV"}})

        self.assertIn("Diagnostics: KV", str(result))

    def test_summary_omits_diagnostics_when_there_are_none(self):
        self.assertNotIn("Diagnostics", str(self._result(None)))


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

    def test_kriging_params_block_discretisation_with_filter(self):
        source = Source(object=POINTSET_URL, attribute="grade")
        target = Target.new_attribute(GRID_URL, "kriged_grade")
        search = SearchNeighborhood(
            ellipsoid=Ellipsoid(ranges=EllipsoidRanges(major=100, semi_major=100, minor=50)),
            max_samples=20,
        )
        bd = BlockDiscretisation(nx=2, ny=2, nz=2)
        target_filter = Filter(where=FilterCondition(attribute="domain_attribute", operator="in", values=["LMS1"]))

        params = KrigingParameters(
            source=source,
            target=target,
            variogram=VARIOGRAM_URL,
            search=search,
            block_discretisation=bd,
            target_filter=target_filter,
        )

        params_dict = params.model_dump()
        self.assertIn("block_discretisation", params_dict)
        self.assertEqual(params_dict["block_discretisation"], {"nx": 2, "ny": 2, "nz": 2})
        self.assertIn("filter", params_dict["target"])
        self.assertEqual(params_dict["target"]["filter"]["where"]["values"], ["LMS1"])


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
