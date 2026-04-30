#  Copyright © 2025 Bentley Systems, Incorporated
#  Licensed under the Apache License, Version 2.0 (the "License")

"""Tests for conditioned simulator (consim) task parameter handling."""

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from evo.compute.tasks import SearchNeighborhood
from evo.compute.tasks.common import Ellipsoid, EllipsoidRanges
from evo.compute.tasks.common.results import TaskAttribute
from evo.compute.tasks.common.runner import TaskRegistry
from evo.compute.tasks.conditioned_simulator import (
    BlockDiscretization,
    ConSimLossCalculation,
    ConSimMaterialCategory,
    ConSimParameters,
    ConSimResult,
    ConSimResultModel,
    ConSimRunner,
    ConSimSummaryAttributes,
    ConSimTargetResult,
    ConSimLinks,
    ConSimValidationSummary,
    DistributionParams,
    ReportContext,
    ReportMeanThresholds,
    TailExtrapolationParams,
    UpperTailParams,
    ValidationReportContextItem,
)
from evo.compute.tasks.common import CreateAttribute

# ---------------------------------------------------------------------------
_BASE = "https://hub.test.evo.bentley.com"
_ORG = "00000000-0000-0000-0000-000000000001"
_WS = "00000000-0000-0000-0000-000000000002"


def _obj_url(obj_id: str = "00000000-0000-0000-0000-000000000003") -> str:
    return f"{_BASE}/geoscience-object/orgs/{_ORG}/workspaces/{_WS}/objects/{obj_id}"


POINTSET_URL = _obj_url("00000000-0000-0000-0000-000000000010")
GRID_URL = _obj_url("00000000-0000-0000-0000-000000000020")
VARIOGRAM_URL = _obj_url("00000000-0000-0000-0000-000000000030")


def _search() -> SearchNeighborhood:
    return SearchNeighborhood(
        ellipsoid=Ellipsoid(ranges=EllipsoidRanges(major=300, semi_major=200, minor=100)),
        max_samples=24,
    )


def _params(**kwargs) -> ConSimParameters:
    defaults = dict(
        source_object=POINTSET_URL,
        source_attribute="locations.attributes[?name=='grade']",
        target_object=GRID_URL,
        neighborhood=_search(),
        variogram_model=VARIOGRAM_URL,
    )
    defaults.update(kwargs)
    return ConSimParameters(**defaults)


def _dump(params: ConSimParameters) -> dict:
    return params.model_dump(mode="json", by_alias=True, exclude_none=True)


def _make_result_model() -> ConSimResultModel:
    return ConSimResultModel(
        target=ConSimTargetResult(
            reference=GRID_URL,
            name="SimGrid",
            schema_id="/objects/regular-3d-grid/1.3.0/regular-3d-grid.schema.json",
            summary_attributes=ConSimSummaryAttributes(
                mean=TaskAttribute(reference="ref_mean", name="sim_mean"),
                variance=TaskAttribute(reference="ref_var", name="sim_var"),
                min=TaskAttribute(reference="ref_min", name="sim_min"),
                max=TaskAttribute(reference="ref_max", name="sim_max"),
            ),
        ),
        links=ConSimLinks(),
    )


# ---------------------------------------------------------------------------
class TestConSimRegistration(unittest.TestCase):
    def test_registered(self):
        runner_cls = TaskRegistry().get_runner(ConSimParameters)
        self.assertIs(runner_cls, ConSimRunner)

    def test_topic_and_task(self):
        self.assertEqual(ConSimRunner.topic, "geostatistics")
        self.assertEqual(ConSimRunner.task, "consim")

    def test_runner_types(self):
        self.assertIs(ConSimRunner.params_type, ConSimParameters)
        self.assertIs(ConSimRunner.result_model_type, ConSimResultModel)
        self.assertIs(ConSimRunner.result_type, ConSimResult)


# ---------------------------------------------------------------------------
class TestBlockDiscretization(unittest.TestCase):
    def test_defaults(self):
        bd = BlockDiscretization()
        self.assertEqual(bd.nx, 1)
        self.assertEqual(bd.ny, 1)
        self.assertEqual(bd.nz, 1)

    def test_custom(self):
        bd = BlockDiscretization(nx=3, ny=3, nz=2)
        d = bd.model_dump(mode="json")
        self.assertEqual(d["nx"], 3)
        self.assertEqual(d["nz"], 2)


# ---------------------------------------------------------------------------
class TestConSimParametersSerialization(unittest.TestCase):
    def test_basic(self):
        d = _dump(_params())
        self.assertEqual(d["source_object"], POINTSET_URL)
        self.assertEqual(d["source_attribute"], "locations.attributes[?name=='grade']")
        self.assertEqual(d["target_object"], GRID_URL)
        self.assertEqual(d["variogram_model"], VARIOGRAM_URL)

    def test_defaults(self):
        d = _dump(_params())
        self.assertEqual(d["kriging_method"], "simple")
        self.assertEqual(d["number_of_lines"], 500)
        self.assertEqual(d["number_of_simulations"], 1)
        self.assertEqual(d["random_seed"], 38239342)
        self.assertEqual(d["number_of_simulations_to_save"], 5)
        self.assertFalse(d["perform_validation"])

    def test_block_discretization_default(self):
        d = _dump(_params())
        self.assertEqual(d["block_discretization"]["nx"], 1)

    def test_custom_simulations(self):
        d = _dump(_params(number_of_simulations=10, number_of_simulations_to_save=3))
        self.assertEqual(d["number_of_simulations"], 10)
        self.assertEqual(d["number_of_simulations_to_save"], 3)

    def test_distribution_params(self):
        params = _params(
            distribution=DistributionParams(
                tail_extrapolation=TailExtrapolationParams(
                    upper=UpperTailParams(power=0.5, max=100.0),
                ),
            ),
        )
        d = _dump(params)
        self.assertEqual(d["distribution"]["tail_extrapolation"]["upper"]["power"], 0.5)

    def test_loss_calculation_params(self):
        params = _params(
            loss_calculation=ConSimLossCalculation(
                material_categories=[
                    ConSimMaterialCategory(cutoff_grade=0.0, metal_price=0, label="waste"),
                    ConSimMaterialCategory(cutoff_grade=0.5, metal_price=200, label="ore"),
                ],
                processing_cost=20.0,
                mining_waste_cost=5.0,
                mining_ore_cost=10.0,
                metal_recovery_fraction=0.85,
                target_attribute=CreateAttribute(name="loss_cat"),
            ),
        )
        d = _dump(params)
        lc = d["loss_calculation"]
        self.assertEqual(len(lc["material_categories"]), 2)
        self.assertEqual(lc["processing_cost"], 20.0)

    def test_location_wise_quantiles(self):
        params = _params(location_wise_quantiles=[0.1, 0.5, 0.9])
        d = _dump(params)
        self.assertEqual(d["location_wise_quantiles"], [0.1, 0.5, 0.9])

    def test_report_context(self):
        params = _params(
            perform_validation=True,
            report_context=ReportContext(
                title="Simulation QC",
                details=[ValidationReportContextItem(label="Deposit", value="TestDeposit")],
            ),
        )
        d = _dump(params)
        self.assertTrue(d["perform_validation"])
        self.assertEqual(d["report_context"]["title"], "Simulation QC")

    def test_report_mean_thresholds(self):
        params = _params(
            report_mean_thresholds=ReportMeanThresholds(acceptable=0.02, marginal=0.05),
        )
        d = _dump(params)
        self.assertEqual(d["report_mean_thresholds"]["acceptable"], 0.02)


# ---------------------------------------------------------------------------
class TestConSimResult(unittest.TestCase):
    def _make_result(self) -> ConSimResult:
        return ConSimResult(MagicMock(), _make_result_model())

    def test_target_name(self):
        self.assertEqual(self._make_result().target_name, "SimGrid")

    def test_target_reference(self):
        self.assertEqual(self._make_result().target_reference, GRID_URL)

    def test_summary_attributes(self):
        r = self._make_result()
        self.assertEqual(r.summary_attributes.mean.name, "sim_mean")
        self.assertEqual(r.summary_attributes.variance.name, "sim_var")

    def test_quantile_attributes_empty(self):
        r = self._make_result()
        self.assertEqual(r.quantile_attributes, [])

    def test_validation_summary_none(self):
        r = self._make_result()
        self.assertIsNone(r.validation_summary)

    def test_dashboard_url_none(self):
        r = self._make_result()
        self.assertIsNone(r.dashboard_url)

    def test_str(self):
        s = str(self._make_result())
        self.assertIn("Conditional Simulation", s)
        self.assertIn("sim_mean", s)

    def test_with_validation(self):
        model = _make_result_model()
        model.validation_summary = ConSimValidationSummary(reference_mean=5.5, mean=5.3)
        r = ConSimResult(MagicMock(), model)
        s = str(r)
        self.assertIn("5.5", s)


# ---------------------------------------------------------------------------
class TestConSimRunnerAsync(unittest.IsolatedAsyncioTestCase):
    def _make_context(self):
        ctx = MagicMock()
        ctx.get_connector.return_value = MagicMock()
        ctx.get_org_id.return_value = "test-org"
        return ctx

    async def test_runner_submits_correctly(self):
        ctx = self._make_context()
        params = _params()
        job = AsyncMock()
        job.wait_for_results.return_value = _make_result_model()

        with patch(
            "evo.compute.tasks.common.runner.JobClient.submit",
            new_callable=AsyncMock,
            return_value=job,
        ) as mock_submit:
            result = await ConSimRunner(ctx, params)

        mock_submit.assert_called_once()
        _, kwargs = mock_submit.call_args
        self.assertEqual(kwargs["topic"], "geostatistics")
        self.assertEqual(kwargs["task"], "consim")
        self.assertIsInstance(result, ConSimResult)


if __name__ == "__main__":
    unittest.main()
