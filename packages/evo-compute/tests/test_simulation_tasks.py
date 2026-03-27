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

"""Tests for conditional simulation task parameter handling."""

from unittest import TestCase
from unittest.mock import MagicMock

from evo.objects import ObjectReference
from evo.objects.typed.attributes import Attribute, PendingAttribute
from pydantic import ValidationError

from evo.compute.tasks import CreateAttribute, SearchNeighborhood, Source
from evo.compute.tasks.common import Ellipsoid, EllipsoidRanges
from evo.compute.tasks.common.results import TaskAttribute
from evo.compute.tasks.common.runner import TaskRegistry
from evo.compute.tasks.simulation import (
    ConsimLinks,
    ConsimParameters,
    ConsimResult,
    ConsimResultModel,
    ConsimRunner,
    ConsimTarget,
    Distribution,
    LossCalculation,
    LowerTail,
    MaterialCategory,
    QuantileAttribute,
    ReportMeanThresholds,
    SummaryAttributes,
    TailExtrapolation,
    UpperTail,
    ValidationReportContext,
    ValidationReportContextItem,
    ValidationSummary,
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


def _create_mock_source_attribute(name: str, key: str, object_url: str, schema_path: str = "") -> MagicMock:
    """Create a mock Attribute (existing) that passes isinstance checks."""
    attr = MagicMock(spec=Attribute)
    attr.name = name
    attr.key = key
    attr.exists = True

    mock_context = MagicMock()
    mock_context.schema_path = schema_path
    attr._context = mock_context

    mock_obj = MagicMock()
    mock_obj.metadata.url = ObjectReference(object_url)
    attr._obj = mock_obj

    return attr


def _create_pending_attribute(name: str, parent_obj: MagicMock | None = None) -> PendingAttribute:
    """Create a real PendingAttribute with an optional mock parent."""
    mock_parent = MagicMock()
    mock_parent._obj = parent_obj
    return PendingAttribute(mock_parent, name)


# ---------------------------------------------------------------------------
# Distribution Types Tests
# ---------------------------------------------------------------------------


class TestTailExtrapolation(TestCase):
    """Tests for tail extrapolation types."""

    def test_upper_tail_valid(self):
        """Test UpperTail with valid parameters."""
        upper = UpperTail(power=0.5, max=10.0)
        self.assertEqual(upper.power, 0.5)
        self.assertEqual(upper.max, 10.0)

    def test_upper_tail_power_boundary(self):
        """Test UpperTail power must be >0 and <=1."""
        # Valid at boundary
        upper = UpperTail(power=1.0, max=10.0)
        self.assertEqual(upper.power, 1.0)

        # Invalid: power = 0
        with self.assertRaises(ValidationError):
            UpperTail(power=0.0, max=10.0)

        # Invalid: power > 1
        with self.assertRaises(ValidationError):
            UpperTail(power=1.5, max=10.0)

    def test_lower_tail_valid(self):
        """Test LowerTail with valid parameters."""
        lower = LowerTail(power=0.5, min=0.0)
        self.assertEqual(lower.power, 0.5)
        self.assertEqual(lower.min, 0.0)

    def test_tail_extrapolation_upper_only(self):
        """Test TailExtrapolation with only upper tail."""
        tail = TailExtrapolation(upper=UpperTail(power=0.5, max=10.0))
        self.assertIsNotNone(tail.upper)
        self.assertIsNone(tail.lower)

    def test_tail_extrapolation_both_tails(self):
        """Test TailExtrapolation with both tails."""
        tail = TailExtrapolation(
            upper=UpperTail(power=0.5, max=10.0),
            lower=LowerTail(power=0.5, min=0.0),
        )
        self.assertIsNotNone(tail.upper)
        self.assertIsNotNone(tail.lower)


class TestDistribution(TestCase):
    """Tests for Distribution type."""

    def test_distribution_basic(self):
        """Test Distribution with basic configuration."""
        dist = Distribution(
            tail_extrapolation=TailExtrapolation(
                upper=UpperTail(power=0.5, max=10.0),
            ),
        )
        self.assertIsNotNone(dist.tail_extrapolation)
        self.assertIsNone(dist.weights)

    def test_distribution_with_weights(self):
        """Test Distribution with weights attribute."""
        dist = Distribution(
            tail_extrapolation=TailExtrapolation(
                upper=UpperTail(power=0.5, max=10.0),
            ),
            weights="weight_col",
        )
        self.assertEqual(dist.weights, "weight_col")


# ---------------------------------------------------------------------------
# Loss Calculation Tests
# ---------------------------------------------------------------------------


class TestMaterialCategory(TestCase):
    """Tests for MaterialCategory type."""

    def test_material_category_minimal(self):
        """Test MaterialCategory with only required fields."""
        cat = MaterialCategory(
            processing_cost=10.0,
            mining_waste_cost=2.0,
            mining_ore_cost=5.0,
            metal_recovery_fraction=0.9,
        )
        self.assertEqual(cat.processing_cost, 10.0)
        self.assertIsNone(cat.cutoff_grade)
        self.assertIsNone(cat.label)

    def test_material_category_full(self):
        """Test MaterialCategory with all fields."""
        cat = MaterialCategory(
            cutoff_grade=0.5,
            metal_price=1000.0,
            label="Ore",
            processing_cost=10.0,
            mining_waste_cost=2.0,
            mining_ore_cost=5.0,
            metal_recovery_fraction=0.9,
        )
        self.assertEqual(cat.cutoff_grade, 0.5)
        self.assertEqual(cat.label, "Ore")


class TestLossCalculation(TestCase):
    """Tests for LossCalculation type."""

    def test_loss_calculation(self):
        """Test LossCalculation configuration."""
        loss = LossCalculation(
            material_categories=[
                MaterialCategory(
                    label="Waste",
                    processing_cost=0.0,
                    mining_waste_cost=2.0,
                    mining_ore_cost=0.0,
                    metal_recovery_fraction=0.0,
                ),
                MaterialCategory(
                    label="Ore",
                    cutoff_grade=0.5,
                    processing_cost=10.0,
                    mining_waste_cost=0.0,
                    mining_ore_cost=5.0,
                    metal_recovery_fraction=0.9,
                ),
            ],
            target_attribute=CreateAttribute(name="loss_results"),
        )
        self.assertEqual(len(loss.material_categories), 2)
        self.assertEqual(loss.target_attribute.name, "loss_results")


# ---------------------------------------------------------------------------
# Validation Types Tests
# ---------------------------------------------------------------------------


class TestValidationTypes(TestCase):
    """Tests for validation report types."""

    def test_validation_report_context(self):
        """Test ValidationReportContext configuration."""
        context = ValidationReportContext(
            title="Test Report",
            details=[
                ValidationReportContextItem(label="Domain", value="All"),
                ValidationReportContextItem(label="Variable", value="grade"),
            ],
        )
        self.assertEqual(context.title, "Test Report")
        self.assertEqual(len(context.details), 2)

    def test_report_mean_thresholds(self):
        """Test ReportMeanThresholds configuration."""
        thresholds = ReportMeanThresholds(acceptable=5.0, marginal=10.0)
        self.assertEqual(thresholds.acceptable, 5.0)
        self.assertEqual(thresholds.marginal, 10.0)


# ---------------------------------------------------------------------------
# ConsimParameters Tests
# ---------------------------------------------------------------------------


class TestConsimParameters(TestCase):
    """Tests for ConsimParameters."""

    def test_consim_params_minimal(self):
        """Test ConsimParameters with minimal required fields."""
        source = Source(object=POINTSET_URL, attribute="locations.attributes[?name=='grade']")

        search = SearchNeighborhood(
            ellipsoid=Ellipsoid(ranges=EllipsoidRanges(major=100, semi_major=100, minor=50)),
            max_samples=20,
        )

        params = ConsimParameters(
            source=source,
            target_object=GRID_URL,
            variogram=VARIOGRAM_URL,
            search=search,
        )

        self.assertIsNone(params.distribution)
        self.assertEqual(params.kriging_method, "simple")
        self.assertEqual(params.number_of_simulations, 1)
        self.assertEqual(params.number_of_lines, 500)

    def test_consim_params_with_distribution(self):
        """Test ConsimParameters with distribution."""
        source = Source(object=POINTSET_URL, attribute="locations.attributes[?name=='grade']")

        search = SearchNeighborhood(
            ellipsoid=Ellipsoid(ranges=EllipsoidRanges(major=100, semi_major=100, minor=50)),
            max_samples=20,
        )

        dist = Distribution(
            tail_extrapolation=TailExtrapolation(
                upper=UpperTail(power=0.5, max=10.0),
            ),
        )

        params = ConsimParameters(
            source=source,
            target_object=GRID_URL,
            variogram=VARIOGRAM_URL,
            search=search,
            distribution=dist,
            number_of_simulations=100,
            location_wise_quantiles=[0.1, 0.5, 0.9],
        )

        self.assertIsNotNone(params.distribution)
        self.assertEqual(params.number_of_simulations, 100)
        self.assertEqual(params.location_wise_quantiles, [0.1, 0.5, 0.9])

    def test_consim_params_with_validation(self):
        """Test ConsimParameters with validation enabled."""
        source = Source(object=POINTSET_URL, attribute="locations.attributes[?name=='grade']")

        search = SearchNeighborhood(
            ellipsoid=Ellipsoid(ranges=EllipsoidRanges(major=100, semi_major=100, minor=50)),
            max_samples=20,
        )

        params = ConsimParameters(
            source=source,
            target_object=GRID_URL,
            variogram=VARIOGRAM_URL,
            search=search,
            perform_validation=True,
            report_context=ValidationReportContext(
                title="Test",
                details=[],
            ),
            report_mean_thresholds=ReportMeanThresholds(acceptable=5.0, marginal=10.0),
        )

        self.assertTrue(params.perform_validation)
        self.assertIsNotNone(params.report_context)
        self.assertIsNotNone(params.report_mean_thresholds)

    def test_consim_params_serialization_splits_source(self):
        """Test that serialization splits source into source_object and source_attribute."""
        source = Source(object=POINTSET_URL, attribute="locations.attributes[?name=='grade']")

        search = SearchNeighborhood(
            ellipsoid=Ellipsoid(ranges=EllipsoidRanges(major=100, semi_major=100, minor=50)),
            max_samples=20,
        )

        params = ConsimParameters(
            source=source,
            target_object=GRID_URL,
            variogram=VARIOGRAM_URL,
            search=search,
        )

        params_dict = params.model_dump(mode="json", by_alias=True, exclude_none=True)

        # Should have source_object and source_attribute, not source
        self.assertNotIn("source", params_dict)
        self.assertEqual(params_dict["source_object"], POINTSET_URL)
        self.assertEqual(params_dict["source_attribute"], "locations.attributes[?name=='grade']")

    def test_consim_params_variogram_alias(self):
        """Test that variogram is serialized as variogram_model."""
        source = Source(object=POINTSET_URL, attribute="locations.attributes[?name=='grade']")

        search = SearchNeighborhood(
            ellipsoid=Ellipsoid(ranges=EllipsoidRanges(major=100, semi_major=100, minor=50)),
            max_samples=20,
        )

        params = ConsimParameters(
            source=source,
            target_object=GRID_URL,
            variogram=VARIOGRAM_URL,
            search=search,
        )

        params_dict = params.model_dump(mode="json", by_alias=True, exclude_none=True)
        self.assertEqual(params_dict["variogram_model"], VARIOGRAM_URL)

    def test_consim_params_search_alias(self):
        """Test that search is serialized as neighborhood."""
        source = Source(object=POINTSET_URL, attribute="locations.attributes[?name=='grade']")

        search = SearchNeighborhood(
            ellipsoid=Ellipsoid(ranges=EllipsoidRanges(major=100, semi_major=100, minor=50)),
            max_samples=20,
        )

        params = ConsimParameters(
            source=source,
            target_object=GRID_URL,
            variogram=VARIOGRAM_URL,
            search=search,
        )

        params_dict = params.model_dump(mode="json", by_alias=True, exclude_none=True)
        self.assertIn("neighborhood", params_dict)
        self.assertNotIn("search", params_dict)


# ---------------------------------------------------------------------------
# ConsimRunner Registration Tests
# ---------------------------------------------------------------------------


class TestConsimRunnerRegistration(TestCase):
    """Tests for ConsimRunner auto-registration."""

    def test_consim_runner_registered(self):
        """Test ConsimRunner is registered with TaskRegistry."""
        registry = TaskRegistry()
        runner_cls = registry.get_runner(ConsimParameters)
        self.assertIs(runner_cls, ConsimRunner)

    def test_consim_runner_topic_and_task(self):
        """Test ConsimRunner has correct topic and task."""
        self.assertEqual(ConsimRunner.topic, "geostatistics")
        self.assertEqual(ConsimRunner.task, "consim")

    def test_consim_runner_type_params(self):
        """Test ConsimRunner has correct type parameters."""
        self.assertIs(ConsimRunner.params_type, ConsimParameters)
        self.assertIs(ConsimRunner.result_model_type, ConsimResultModel)
        self.assertIs(ConsimRunner.result_type, ConsimResult)


# ---------------------------------------------------------------------------
# ConsimResult Tests
# ---------------------------------------------------------------------------


class TestConsimResult(TestCase):
    """Tests for ConsimResult wrapper."""

    def _create_mock_result_model(self) -> ConsimResultModel:
        """Create a mock result model for testing."""
        return ConsimResultModel(
            target=ConsimTarget(
                reference=GRID_URL,
                name="Test Grid",
                description="Test description",
                schema_id="regular-masked-3d-grid/1.2",
                summary_attributes=SummaryAttributes(
                    mean=TaskAttribute(reference="mean_ref", name="mean"),
                    variance=TaskAttribute(reference="var_ref", name="variance"),
                    min=TaskAttribute(reference="min_ref", name="min"),
                    max=TaskAttribute(reference="max_ref", name="max"),
                ),
                quantile_attributes=[
                    QuantileAttribute(reference="p10_ref", name="p10", quantile=0.1),
                    QuantileAttribute(reference="p50_ref", name="p50", quantile=0.5),
                    QuantileAttribute(reference="p90_ref", name="p90", quantile=0.9),
                ],
                simulations=TaskAttribute(reference="sim_ref", name="simulations"),
                validation_summary=ValidationSummary(reference_mean=1.0, mean=0.98),
            ),
            links=ConsimLinks(dashboard="https://dashboard.example.com"),
        )

    def test_consim_result_properties(self):
        """Test ConsimResult property accessors."""
        model = self._create_mock_result_model()
        context = MagicMock()
        result = ConsimResult(context, model)

        self.assertEqual(result.target_name, "Test Grid")
        self.assertEqual(result.target_reference, GRID_URL)
        self.assertEqual(result.mean_attribute.name, "mean")
        self.assertEqual(result.variance_attribute.name, "variance")
        self.assertEqual(result.min_attribute.name, "min")
        self.assertEqual(result.max_attribute.name, "max")

    def test_consim_result_quantiles(self):
        """Test ConsimResult quantile accessors."""
        model = self._create_mock_result_model()
        context = MagicMock()
        result = ConsimResult(context, model)

        quantiles = result.quantile_attributes
        self.assertEqual(len(quantiles), 3)
        self.assertEqual(quantiles[0].quantile, 0.1)
        self.assertEqual(quantiles[1].quantile, 0.5)
        self.assertEqual(quantiles[2].quantile, 0.9)

    def test_consim_result_validation(self):
        """Test ConsimResult validation accessors."""
        model = self._create_mock_result_model()
        context = MagicMock()
        result = ConsimResult(context, model)

        self.assertIsNotNone(result.validation_summary)
        self.assertEqual(result.validation_summary.reference_mean, 1.0)
        self.assertEqual(result.validation_summary.mean, 0.98)

    def test_consim_result_dashboard_link(self):
        """Test ConsimResult dashboard link accessor."""
        model = self._create_mock_result_model()
        context = MagicMock()
        result = ConsimResult(context, model)

        self.assertEqual(result.dashboard_link, "https://dashboard.example.com")

    def test_consim_result_str(self):
        """Test ConsimResult string representation."""
        model = self._create_mock_result_model()
        context = MagicMock()
        result = ConsimResult(context, model)

        str_repr = str(result)
        self.assertIn("Conditional Simulation", str_repr)
        self.assertIn("Test Grid", str_repr)
        self.assertIn("mean", str_repr)
        self.assertIn("Dashboard", str_repr)

    def test_consim_result_display_name(self):
        """Test ConsimResult has correct display name."""
        self.assertEqual(ConsimResult.TASK_DISPLAY_NAME, "Conditional Simulation")
