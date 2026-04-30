#  Copyright © 2025 Bentley Systems, Incorporated
#  Licensed under the Apache License, Version 2.0 (the "License")

"""Tests for continuous-distribution task parameter handling."""

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from evo.compute.tasks import Source
from evo.compute.tasks.common.runner import TaskRegistry
from evo.compute.tasks.continuous_distribution import (
    ContinuousDistributionParameters,
    ContinuousDistributionResult,
    ContinuousDistributionResultModel,
    ContinuousDistributionRunner,
    DistributionOutput,
    DistributionTarget,
    LowerTail,
    TailExtrapolation,
    UpperTail,
)

# ---------------------------------------------------------------------------
_BASE = "https://hub.test.evo.bentley.com"
_ORG = "00000000-0000-0000-0000-000000000001"
_WS = "00000000-0000-0000-0000-000000000002"


def _obj_url(obj_id: str = "00000000-0000-0000-0000-000000000003") -> str:
    return f"{_BASE}/geoscience-object/orgs/{_ORG}/workspaces/{_WS}/objects/{obj_id}"


POINTSET_URL = _obj_url("00000000-0000-0000-0000-000000000010")
DIST_URL = _obj_url("00000000-0000-0000-0000-000000000030")


def _params(**kwargs) -> ContinuousDistributionParameters:
    defaults = dict(
        source=Source(object=POINTSET_URL, attribute="locations.attributes[?name=='grade']"),
        target=DistributionTarget(reference=DIST_URL),
    )
    defaults.update(kwargs)
    return ContinuousDistributionParameters(**defaults)


def _dump(params: ContinuousDistributionParameters) -> dict:
    return params.model_dump(mode="json", by_alias=True, exclude_none=True)


def _make_result_model() -> ContinuousDistributionResultModel:
    return ContinuousDistributionResultModel(
        message="Distribution created successfully.",
        distribution=DistributionOutput(
            reference=DIST_URL,
            name="grade_dist",
            description="Grade distribution",
            schema_id="/objects/distribution/1.0.0/distribution.schema.json",
        ),
    )


# ---------------------------------------------------------------------------
class TestCDRegistration(unittest.TestCase):
    def test_registered(self):
        runner_cls = TaskRegistry().get_runner(ContinuousDistributionParameters)
        self.assertIs(runner_cls, ContinuousDistributionRunner)

    def test_topic_and_task(self):
        self.assertEqual(ContinuousDistributionRunner.topic, "geostatistics")
        self.assertEqual(ContinuousDistributionRunner.task, "continuous-distribution")

    def test_runner_types(self):
        self.assertIs(ContinuousDistributionRunner.params_type, ContinuousDistributionParameters)
        self.assertIs(ContinuousDistributionRunner.result_model_type, ContinuousDistributionResultModel)
        self.assertIs(ContinuousDistributionRunner.result_type, ContinuousDistributionResult)


# ---------------------------------------------------------------------------
class TestTailExtrapolation(unittest.TestCase):
    def test_upper_tail(self):
        t = UpperTail(power=0.5, max=100.0)
        self.assertEqual(t.power, 0.5)
        self.assertEqual(t.max, 100.0)

    def test_lower_tail(self):
        t = LowerTail(power=0.3, min=-5.0)
        self.assertEqual(t.power, 0.3)
        self.assertEqual(t.min, -5.0)

    def test_tail_extrapolation_upper_only(self):
        te = TailExtrapolation(upper=UpperTail(power=0.5, max=100.0))
        d = te.model_dump(mode="json", exclude_none=True)
        self.assertIn("upper", d)
        self.assertNotIn("lower", d)

    def test_tail_extrapolation_both(self):
        te = TailExtrapolation(
            upper=UpperTail(power=0.5, max=100.0),
            lower=LowerTail(power=0.3, min=-5.0),
        )
        d = te.model_dump(mode="json", exclude_none=True)
        self.assertIn("upper", d)
        self.assertIn("lower", d)


# ---------------------------------------------------------------------------
class TestDistributionTarget(unittest.TestCase):
    def test_defaults(self):
        t = DistributionTarget(reference=DIST_URL)
        self.assertFalse(t.overwrite)
        self.assertIsNone(t.description)
        self.assertEqual(t.tags, {})

    def test_with_overwrite_and_tags(self):
        t = DistributionTarget(reference=DIST_URL, overwrite=True, tags={"env": "test"})
        d = t.model_dump(mode="json", exclude_none=True)
        self.assertTrue(d["overwrite"])
        self.assertEqual(d["tags"]["env"], "test")


# ---------------------------------------------------------------------------
class TestCDParametersSerialization(unittest.TestCase):
    def test_basic_serialization(self):
        d = _dump(_params())
        self.assertEqual(d["source"]["object"], POINTSET_URL)
        self.assertEqual(d["target"]["reference"], DIST_URL)
        self.assertNotIn("weights", d)
        self.assertNotIn("tail_extrapolation", d)

    def test_with_weights(self):
        params = _params(
            weights=Source(object=POINTSET_URL, attribute="locations.attributes[?name=='weight']"),
        )
        d = _dump(params)
        self.assertEqual(d["weights"]["object"], POINTSET_URL)

    def test_with_tail_extrapolation(self):
        params = _params(
            tail_extrapolation=TailExtrapolation(
                upper=UpperTail(power=0.5, max=100.0),
                lower=LowerTail(power=0.3, min=-5.0),
            ),
        )
        d = _dump(params)
        self.assertEqual(d["tail_extrapolation"]["upper"]["power"], 0.5)
        self.assertEqual(d["tail_extrapolation"]["lower"]["min"], -5.0)


# ---------------------------------------------------------------------------
class TestCDResult(unittest.TestCase):
    def _make_result(self) -> ContinuousDistributionResult:
        return ContinuousDistributionResult(MagicMock(), _make_result_model())

    def test_message(self):
        self.assertIn("successfully", self._make_result().message)

    def test_distribution_name(self):
        self.assertEqual(self._make_result().distribution_name, "grade_dist")

    def test_distribution_reference(self):
        self.assertEqual(self._make_result().distribution_reference, DIST_URL)

    def test_str(self):
        s = str(self._make_result())
        self.assertIn("Continuous Distribution", s)
        self.assertIn("grade_dist", s)


# ---------------------------------------------------------------------------
class TestCDRunnerAsync(unittest.IsolatedAsyncioTestCase):
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
            result = await ContinuousDistributionRunner(ctx, params)

        mock_submit.assert_called_once()
        _, kwargs = mock_submit.call_args
        self.assertEqual(kwargs["topic"], "geostatistics")
        self.assertEqual(kwargs["task"], "continuous-distribution")
        self.assertIsInstance(result, ContinuousDistributionResult)


if __name__ == "__main__":
    unittest.main()
