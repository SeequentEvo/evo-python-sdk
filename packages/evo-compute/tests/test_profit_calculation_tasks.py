#  Copyright © 2025 Bentley Systems, Incorporated
#  Licensed under the Apache License, Version 2.0 (the "License")

"""Tests for profit-calculation task parameter handling."""

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from evo.compute.tasks import CreateAttribute, Source, Target, UpdateAttribute
from evo.compute.tasks.common.results import TaskAttribute, TaskTarget
from evo.compute.tasks.common.runner import TaskRegistry
from evo.compute.tasks.loss_calculation import MaterialCategory
from evo.compute.tasks.profit_calculation import (
    ProfitCalculationParameters,
    ProfitCalculationResult,
    ProfitCalculationResultModel,
    ProfitCalculationRunner,
)

# ---------------------------------------------------------------------------
_BASE = "https://hub.test.evo.bentley.com"
_ORG = "00000000-0000-0000-0000-000000000001"
_WS = "00000000-0000-0000-0000-000000000002"


def _obj_url(obj_id: str = "00000000-0000-0000-0000-000000000003") -> str:
    return f"{_BASE}/geoscience-object/orgs/{_ORG}/workspaces/{_WS}/objects/{obj_id}"


GRID_URL = _obj_url("00000000-0000-0000-0000-000000000010")
TARGET_URL = _obj_url("00000000-0000-0000-0000-000000000020")

_CATEGORIES = [
    MaterialCategory(cutoff_grade=0.0, metal_price=0, label="waste"),
    MaterialCategory(cutoff_grade=0.5, metal_price=200, label="ore"),
]


def _params(**kwargs) -> ProfitCalculationParameters:
    defaults = dict(
        source=Source(object=GRID_URL, attribute="cell_attributes[?name=='simulations']"),
        target=Target(object=TARGET_URL, attribute=CreateAttribute(name="profit_cat")),
        material_categories=_CATEGORIES,
        processing_cost=20.0,
        mining_waste_cost=5.0,
        mining_ore_cost=10.0,
        metal_recovery_fraction=0.85,
    )
    defaults.update(kwargs)
    return ProfitCalculationParameters(**defaults)


def _dump(params: ProfitCalculationParameters) -> dict:
    return params.model_dump(mode="json", by_alias=True, exclude_none=True)


def _make_result_model() -> ProfitCalculationResultModel:
    return ProfitCalculationResultModel(
        message="Profit calculation completed successfully.",
        target=TaskTarget(
            reference=TARGET_URL,
            name="MyGrid",
            schema_id="/objects/regular-3d-grid/1.3.0/regular-3d-grid.schema.json",
            attribute=TaskAttribute(reference="cell_attributes[?name=='profit_cat']", name="profit_cat"),
        ),
    )


# ---------------------------------------------------------------------------
class TestProfitCalcRegistration(unittest.TestCase):
    def test_registered(self):
        runner_cls = TaskRegistry().get_runner(ProfitCalculationParameters)
        self.assertIs(runner_cls, ProfitCalculationRunner)

    def test_topic_and_task(self):
        self.assertEqual(ProfitCalculationRunner.topic, "geostatistics")
        self.assertEqual(ProfitCalculationRunner.task, "profit-calculation")

    def test_runner_types(self):
        self.assertIs(ProfitCalculationRunner.params_type, ProfitCalculationParameters)
        self.assertIs(ProfitCalculationRunner.result_model_type, ProfitCalculationResultModel)
        self.assertIs(ProfitCalculationRunner.result_type, ProfitCalculationResult)


# ---------------------------------------------------------------------------
class TestProfitCalcParametersSerialization(unittest.TestCase):
    def test_source_serializes(self):
        d = _dump(_params())
        self.assertEqual(d["source"]["object"], GRID_URL)

    def test_target_create(self):
        d = _dump(_params())
        self.assertEqual(d["target"]["attribute"]["operation"], "create")
        self.assertEqual(d["target"]["attribute"]["name"], "profit_cat")

    def test_material_categories(self):
        d = _dump(_params())
        self.assertEqual(len(d["material_categories"]), 2)
        self.assertEqual(d["material_categories"][1]["metal_price"], 200)

    def test_economic_params(self):
        d = _dump(_params())
        self.assertEqual(d["processing_cost"], 20.0)
        self.assertEqual(d["metal_recovery_fraction"], 0.85)

    def test_reuses_material_category(self):
        """MaterialCategory is imported from loss_calculation module."""
        from evo.compute.tasks.profit_calculation import MaterialCategory as MC
        self.assertIs(MC, MaterialCategory)


# ---------------------------------------------------------------------------
class TestProfitCalcResult(unittest.TestCase):
    def _make_result(self) -> ProfitCalculationResult:
        return ProfitCalculationResult(MagicMock(), _make_result_model())

    def test_message(self):
        self.assertIn("successfully", self._make_result().message)

    def test_target_name(self):
        self.assertEqual(self._make_result().target_name, "MyGrid")

    def test_attribute_name(self):
        self.assertEqual(self._make_result().attribute_name, "profit_cat")

    def test_str(self):
        s = str(self._make_result())
        self.assertIn("Profit Calculation", s)
        self.assertIn("profit_cat", s)


# ---------------------------------------------------------------------------
class TestProfitCalcRunnerAsync(unittest.IsolatedAsyncioTestCase):
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
            result = await ProfitCalculationRunner(ctx, params)

        mock_submit.assert_called_once()
        _, kwargs = mock_submit.call_args
        self.assertEqual(kwargs["topic"], "geostatistics")
        self.assertEqual(kwargs["task"], "profit-calculation")
        self.assertIsInstance(result, ProfitCalculationResult)


if __name__ == "__main__":
    unittest.main()
