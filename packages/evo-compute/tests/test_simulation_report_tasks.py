#  Copyright © 2026 Bentley Systems, Incorporated
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Tests for simulation-report task parameter handling."""

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from evo.compute.tasks import SearchNeighborhood
from evo.compute.tasks.common import Ellipsoid, EllipsoidRanges
from evo.compute.tasks.common.runner import TaskRegistry
from evo.compute.tasks.simulation_report import (
    SimReportContext,
    SimReportContextItem,
    SimReportLinks,
    SimReportThresholds,
    SimReportValidationReport,
    SimReportValidationSummary,
    SimulationReportBlockDiscretization,
    SimulationReportDistribution,
    SimulationReportParameters,
    SimulationReportResult,
    SimulationReportResultModel,
    SimulationReportRunner,
)

# ---------------------------------------------------------------------------
# Test helpers
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
        ellipsoid=Ellipsoid(ranges=EllipsoidRanges(major=200.0, semi_major=150.0, minor=100.0)),
        max_samples=20,
    )


def _params(**kwargs) -> SimulationReportParameters:
    defaults = dict(
        simulation_source=POINTSET_URL,
        source_attribute="locations.attributes[0]",
        simulation_target=GRID_URL,
        point_simulations="cell_attributes[0]",
        point_simulations_normal_score="cell_attributes[1]",
        block_simulations="cell_attributes[2]",
        block_simulations_normal_score="cell_attributes[3]",
        variogram_model=VARIOGRAM_URL,
        neighborhood=_search(),
        number_of_simulations=50,
        random_seed=38239342,
    )
    defaults.update(kwargs)
    return SimulationReportParameters(**defaults)


def _dump(params: SimulationReportParameters) -> dict:
    return params.model_dump(mode="json", by_alias=True, exclude_none=True)


def _make_result_model() -> SimulationReportResultModel:
    return SimulationReportResultModel(
        validation_summary=SimReportValidationSummary(
            reference_mean=2.345,
            mean=2.350,
        ),
        validation_report=SimReportValidationReport(
            reference="https://example.com/report.html",
        ),
        links=SimReportLinks(
            dashboard="https://example.com/dashboard",
        ),
    )


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


class TestSimulationReportRegistration(unittest.TestCase):
    def test_registered(self):
        runner_cls = TaskRegistry().get_runner(SimulationReportParameters)
        self.assertIs(runner_cls, SimulationReportRunner)

    def test_topic_and_task(self):
        self.assertEqual(SimulationReportRunner.topic, "geostatistics")
        self.assertEqual(SimulationReportRunner.task, "simulation-report")

    def test_runner_types(self):
        self.assertIs(SimulationReportRunner.params_type, SimulationReportParameters)
        self.assertIs(SimulationReportRunner.result_model_type, SimulationReportResultModel)
        self.assertIs(SimulationReportRunner.result_type, SimulationReportResult)


# ---------------------------------------------------------------------------
# Parameter serialization tests
# ---------------------------------------------------------------------------


class TestSimulationReportParametersSerialization(unittest.TestCase):
    def test_flat_fields(self):
        d = _dump(_params())
        self.assertEqual(d["simulation_source"], POINTSET_URL)
        self.assertEqual(d["source_attribute"], "locations.attributes[0]")
        self.assertEqual(d["simulation_target"], GRID_URL)
        self.assertEqual(d["point_simulations"], "cell_attributes[0]")
        self.assertEqual(d["point_simulations_normal_score"], "cell_attributes[1]")
        self.assertEqual(d["block_simulations"], "cell_attributes[2]")
        self.assertEqual(d["block_simulations_normal_score"], "cell_attributes[3]")
        self.assertEqual(d["variogram_model"], VARIOGRAM_URL)

    def test_neighborhood(self):
        d = _dump(_params())
        self.assertIn("ellipsoid", d["neighborhood"])
        self.assertEqual(d["neighborhood"]["max_samples"], 20)

    def test_block_discretization_defaults(self):
        d = _dump(_params())
        self.assertEqual(d["block_discretization"]["nx"], 1)
        self.assertEqual(d["block_discretization"]["ny"], 1)
        self.assertEqual(d["block_discretization"]["nz"], 1)

    def test_block_discretization_custom(self):
        d = _dump(_params(
            block_discretization=SimulationReportBlockDiscretization(nx=3, ny=3, nz=3)
        ))
        self.assertEqual(d["block_discretization"]["nx"], 3)

    def test_number_of_simulations(self):
        d = _dump(_params())
        self.assertEqual(d["number_of_simulations"], 50)

    def test_kriging_method_default(self):
        d = _dump(_params())
        self.assertEqual(d["kriging_method"], "simple")

    def test_number_of_lines_default(self):
        d = _dump(_params())
        self.assertEqual(d["number_of_lines"], 500)

    def test_distribution_optional(self):
        d = _dump(_params())
        self.assertNotIn("distribution", d)

    def test_distribution_with_weights(self):
        d = _dump(_params(
            distribution=SimulationReportDistribution(weights="declustering_weights")
        ))
        self.assertEqual(d["distribution"]["weights"], "declustering_weights")

    def test_report_context(self):
        d = _dump(_params(
            report_context=SimReportContext(
                title="Test Report",
                details=[SimReportContextItem(label="Variable", value="Au")],
            )
        ))
        self.assertEqual(d["report_context"]["title"], "Test Report")
        self.assertEqual(d["report_context"]["details"][0]["label"], "Variable")

    def test_report_mean_thresholds(self):
        d = _dump(_params(
            report_mean_thresholds=SimReportThresholds(acceptable=0.05, marginal=0.10)
        ))
        self.assertAlmostEqual(d["report_mean_thresholds"]["acceptable"], 0.05)
        self.assertAlmostEqual(d["report_mean_thresholds"]["marginal"], 0.10)

    def test_none_optional_fields_excluded(self):
        d = _dump(_params())
        self.assertNotIn("distribution", d)
        self.assertNotIn("report_context", d)
        self.assertNotIn("report_mean_thresholds", d)


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


class TestSimulationReportValidation(unittest.TestCase):
    def test_block_discretization_min(self):
        with self.assertRaises(Exception):
            SimulationReportBlockDiscretization(nx=0, ny=1, nz=1)

    def test_block_discretization_max(self):
        with self.assertRaises(Exception):
            SimulationReportBlockDiscretization(nx=10, ny=1, nz=1)

    def test_number_of_simulations_min(self):
        with self.assertRaises(Exception):
            _params(number_of_simulations=0)


# ---------------------------------------------------------------------------
# Result model tests
# ---------------------------------------------------------------------------


class TestSimulationReportResultModel(unittest.TestCase):
    def test_result_model_validate(self):
        data = {
            "validation_summary": {"reference_mean": 1.0, "mean": 1.1},
            "validation_report": {"reference": "https://example.com/report"},
            "links": {"dashboard": "https://example.com/dash"},
        }
        model = SimulationReportResultModel.model_validate(data)
        self.assertIsNotNone(model.validation_summary)
        self.assertAlmostEqual(model.validation_summary.reference_mean, 1.0)

    def test_result_model_minimal(self):
        model = SimulationReportResultModel.model_validate({})
        self.assertIsNone(model.validation_summary)
        self.assertIsNone(model.validation_report)
        self.assertIsNone(model.links)


# ---------------------------------------------------------------------------
# Result wrapper tests
# ---------------------------------------------------------------------------


class TestSimulationReportResult(unittest.TestCase):
    def _make_result(self) -> SimulationReportResult:
        return SimulationReportResult(MagicMock(), _make_result_model())

    def test_validation_summary(self):
        r = self._make_result()
        self.assertAlmostEqual(r.validation_summary.reference_mean, 2.345)
        self.assertAlmostEqual(r.validation_summary.mean, 2.350)

    def test_report_reference(self):
        self.assertEqual(
            self._make_result().report_reference,
            "https://example.com/report.html",
        )

    def test_dashboard_url(self):
        self.assertEqual(
            self._make_result().dashboard_url,
            "https://example.com/dashboard",
        )

    def test_str(self):
        s = str(self._make_result())
        self.assertIn("Simulation Report", s)
        self.assertIn("2.345", s)

    def test_no_summary(self):
        model = SimulationReportResultModel()
        r = SimulationReportResult(MagicMock(), model)
        self.assertIsNone(r.validation_summary)
        self.assertIsNone(r.report_reference)
        self.assertIsNone(r.dashboard_url)


# ---------------------------------------------------------------------------
# Runner behavior tests
# ---------------------------------------------------------------------------


class TestSimulationReportRunnerAsync(unittest.IsolatedAsyncioTestCase):
    def _make_context(self):
        connector = MagicMock()
        context = MagicMock()
        context.get_connector.return_value = connector
        context.get_org_id.return_value = "test-org-id"
        return context

    def _make_job(self):
        job = AsyncMock()
        job.wait_for_results.return_value = _make_result_model()
        return job

    async def test_runner_submits_with_correct_topic_and_task(self):
        context = self._make_context()
        params = _params()

        with patch(
            "evo.compute.tasks.common.runner.JobClient.submit",
            new_callable=AsyncMock,
            return_value=self._make_job(),
        ) as mock_submit:
            result = await SimulationReportRunner(context, params)

        mock_submit.assert_called_once()
        _, kwargs = mock_submit.call_args
        self.assertEqual(kwargs["topic"], "geostatistics")
        self.assertEqual(kwargs["task"], "simulation-report")
        self.assertIsInstance(result, SimulationReportResult)

    async def test_runner_preview_defaults_false(self):
        context = self._make_context()
        params = _params()

        with patch(
            "evo.compute.tasks.common.runner.JobClient.submit",
            new_callable=AsyncMock,
            return_value=self._make_job(),
        ) as mock_submit:
            await SimulationReportRunner(context, params)

        _, kwargs = mock_submit.call_args
        self.assertFalse(kwargs.get("preview", True))


if __name__ == "__main__":
    unittest.main()
