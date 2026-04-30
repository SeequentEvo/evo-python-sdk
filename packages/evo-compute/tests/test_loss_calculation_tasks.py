#  Copyright © 2025 Bentley Systems, Incorporated
#  Licensed under the Apache License, Version 2.0 (the "License")

"""Tests for loss-calculation task parameter handling."""

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from evo.compute.tasks import CreateAttribute, Source, Target, UpdateAttribute
from evo.compute.tasks.common.results import TaskAttribute, TaskTarget
from evo.compute.tasks.common.runner import TaskRegistry
from evo.compute.tasks.loss_calculation import (
    LossCalculationParameters,
    LossCalculationResult,
    LossCalculationResultModel,
    LossCalculationRunner,
    MaterialCategory,
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


def _params(**kwargs) -> LossCalculationParameters:
    defaults = dict(
        source=Source(object=GRID_URL, attribute="cell_attributes[?name=='simulations']"),
        target=Target(object=TARGET_URL, attribute=CreateAttribute(name="loss_cat")),
        material_categories=_CATEGORIES,
        processing_cost=20.0,
        mining_waste_cost=5.0,
        mining_ore_cost=10.0,
        metal_recovery_fraction=0.85,
    )
    defaults.update(kwargs)
    return LossCalculationParameters(**defaults)


def _dump(params: LossCalculationParameters) -> dict:
    return params.model_dump(mode="json", by_alias=True, exclude_none=True)


def _make_result_model() -> LossCalculationResultModel:
    return LossCalculationResultModel(
        message="Loss calculation completed successfully.",
        target=TaskTarget(
            reference=TARGET_URL,
            name="MyGrid",
            schema_id="/objects/regular-3d-grid/1.3.0/regular-3d-grid.schema.json",
            attribute=TaskAttribute(reference="cell_attributes[?name=='loss_cat']", name="loss_cat"),
        ),
    )


# ---------------------------------------------------------------------------
class TestLossCalcRegistration(unittest.TestCase):
    def test_registered(self):
        runner_cls = TaskRegistry().get_runner(LossCalculationParameters)
        self.assertIs(runner_cls, LossCalculationRunner)

    def test_topic_and_task(self):
        self.assertEqual(LossCalculationRunner.topic, "geostatistics")
        self.assertEqual(LossCalculationRunner.task, "loss-calculation")

    def test_runner_types(self):
        self.assertIs(LossCalculationRunner.params_type, LossCalculationParameters)
        self.assertIs(LossCalculationRunner.result_model_type, LossCalculationResultModel)
        self.assertIs(LossCalculationRunner.result_type, LossCalculationResult)


# ---------------------------------------------------------------------------
class TestMaterialCategory(unittest.TestCase):
    def test_default_values(self):
        cat = MaterialCategory()
        self.assertEqual(cat.cutoff_grade, 0.0)
        self.assertEqual(cat.metal_price, 0.0)
        self.assertEqual(cat.label, "waste")

    def test_custom_values(self):
        cat = MaterialCategory(cutoff_grade=0.5, metal_price=200, label="ore")
        self.assertEqual(cat.cutoff_grade, 0.5)
        self.assertEqual(cat.metal_price, 200)
        self.assertEqual(cat.label, "ore")

    def test_serialization(self):
        cat = MaterialCategory(cutoff_grade=0.3, metal_price=150, label="low_grade")
        d = cat.model_dump(mode="json")
        self.assertEqual(d["cutoff_grade"], 0.3)
        self.assertEqual(d["metal_price"], 150)
        self.assertEqual(d["label"], "low_grade")


# ---------------------------------------------------------------------------
class TestLossCalcParametersSerialization(unittest.TestCase):
    def test_source_serializes(self):
        d = _dump(_params())
        self.assertEqual(d["source"]["object"], GRID_URL)

    def test_target_create(self):
        d = _dump(_params())
        self.assertEqual(d["target"]["attribute"]["operation"], "create")
        self.assertEqual(d["target"]["attribute"]["name"], "loss_cat")

    def test_target_update(self):
        params = _params(target=Target(object=TARGET_URL, attribute=UpdateAttribute(reference="attr_ref")))
        d = _dump(params)
        self.assertEqual(d["target"]["attribute"]["operation"], "update")

    def test_material_categories_serialized(self):
        d = _dump(_params())
        cats = d["material_categories"]
        self.assertEqual(len(cats), 2)
        self.assertEqual(cats[0]["label"], "waste")
        self.assertEqual(cats[1]["label"], "ore")
        self.assertEqual(cats[1]["cutoff_grade"], 0.5)

    def test_economic_params(self):
        d = _dump(_params())
        self.assertEqual(d["processing_cost"], 20.0)
        self.assertEqual(d["mining_waste_cost"], 5.0)
        self.assertEqual(d["mining_ore_cost"], 10.0)
        self.assertEqual(d["metal_recovery_fraction"], 0.85)


# ---------------------------------------------------------------------------
class TestLossCalcResult(unittest.TestCase):
    def _make_result(self) -> LossCalculationResult:
        return LossCalculationResult(MagicMock(), _make_result_model())

    def test_message(self):
        self.assertIn("successfully", self._make_result().message)

    def test_target_name(self):
        self.assertEqual(self._make_result().target_name, "MyGrid")

    def test_target_reference(self):
        self.assertEqual(self._make_result().target_reference, TARGET_URL)

    def test_attribute_name(self):
        self.assertEqual(self._make_result().attribute_name, "loss_cat")

    def test_str(self):
        s = str(self._make_result())
        self.assertIn("Loss Calculation", s)
        self.assertIn("loss_cat", s)


# ---------------------------------------------------------------------------
class TestLossCalcRunnerAsync(unittest.IsolatedAsyncioTestCase):
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
            result = await LossCalculationRunner(ctx, params)

        mock_submit.assert_called_once()
        _, kwargs = mock_submit.call_args
        self.assertEqual(kwargs["topic"], "geostatistics")
        self.assertEqual(kwargs["task"], "loss-calculation")
        self.assertIsInstance(result, LossCalculationResult)


if __name__ == "__main__":
    unittest.main()
