#  Copyright © 2025 Bentley Systems, Incorporated
#  Licensed under the Apache License, Version 2.0 (the "License")

"""Tests for KNN task parameter handling."""

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from evo.compute.tasks import SearchNeighborhood
from evo.compute.tasks.common import Ellipsoid, EllipsoidRanges
from evo.compute.tasks.common.runner import TaskRegistry
from evo.compute.tasks.knn import (
    GORefElement,
    KNNModifiedResult,
    KNNParameters,
    KNNResult,
    KNNResultModel,
    KNNRunner,
    KNNSource,
    KNNTarget,
)

# ---------------------------------------------------------------------------
_BASE = "https://hub.test.evo.bentley.com"
_ORG = "00000000-0000-0000-0000-000000000001"
_WS = "00000000-0000-0000-0000-000000000002"


def _obj_url(obj_id: str = "00000000-0000-0000-0000-000000000003") -> str:
    return f"{_BASE}/geoscience-object/orgs/{_ORG}/workspaces/{_WS}/objects/{obj_id}"


POINTSET_URL = _obj_url("00000000-0000-0000-0000-000000000010")
GRID_URL = _obj_url("00000000-0000-0000-0000-000000000020")


def _search() -> SearchNeighborhood:
    return SearchNeighborhood(
        ellipsoid=Ellipsoid(ranges=EllipsoidRanges(major=200, semi_major=150, minor=100)),
        max_samples=20,
    )


def _params(**kwargs) -> KNNParameters:
    defaults = dict(
        source=KNNSource(
            object_reference=POINTSET_URL,
            object_element=[GORefElement(path="/locations/attributes/@name=grade")],
        ),
        target=KNNTarget(
            object_reference=GRID_URL,
            object_element=[GORefElement(path="/cell_attributes/@name=knn_grade")],
        ),
        neighborhood=_search(),
    )
    defaults.update(kwargs)
    return KNNParameters(**defaults)


def _dump(params: KNNParameters) -> dict:
    return params.model_dump(mode="json", by_alias=True, exclude_none=True)


def _make_result_model() -> KNNResultModel:
    return KNNResultModel(
        message="KNN estimation completed.",
        object_modified=KNNModifiedResult(
            object_reference=GRID_URL,
            object_element=[GORefElement(path="/cell_attributes/@name=knn_grade")],
        ),
    )


# ---------------------------------------------------------------------------
class TestKNNRegistration(unittest.TestCase):
    def test_registered(self):
        runner_cls = TaskRegistry().get_runner(KNNParameters)
        self.assertIs(runner_cls, KNNRunner)

    def test_topic_and_task(self):
        self.assertEqual(KNNRunner.topic, "geostat")
        self.assertEqual(KNNRunner.task, "knn")

    def test_runner_types(self):
        self.assertIs(KNNRunner.params_type, KNNParameters)
        self.assertIs(KNNRunner.result_model_type, KNNResultModel)
        self.assertIs(KNNRunner.result_type, KNNResult)


# ---------------------------------------------------------------------------
class TestGORefElement(unittest.TestCase):
    def test_defaults(self):
        e = GORefElement(path="/locations/attributes/@name=grade")
        self.assertEqual(e.type, "element")
        self.assertEqual(e.path, "/locations/attributes/@name=grade")

    def test_serialization(self):
        e = GORefElement(path="/cell_attributes/@name=x")
        d = e.model_dump(mode="json")
        self.assertEqual(d["type"], "element")
        self.assertEqual(d["path"], "/cell_attributes/@name=x")


# ---------------------------------------------------------------------------
class TestKNNParametersSerialization(unittest.TestCase):
    def test_source_old_format(self):
        d = _dump(_params())
        src = d["source"]
        self.assertEqual(src["type"], "geoscience-object-reference")
        self.assertEqual(src["object_reference"], POINTSET_URL)
        self.assertEqual(len(src["object_element"]), 1)
        self.assertEqual(src["object_element"][0]["type"], "element")
        self.assertEqual(src["object_element"][0]["path"], "/locations/attributes/@name=grade")

    def test_target_old_format(self):
        d = _dump(_params())
        tgt = d["target"]
        self.assertEqual(tgt["type"], "geoscience-object-reference")
        self.assertEqual(tgt["object_reference"], GRID_URL)
        self.assertEqual(tgt["object_element"][0]["path"], "/cell_attributes/@name=knn_grade")

    def test_neighborhood(self):
        d = _dump(_params())
        self.assertIn("ellipsoid", d["neighborhood"])
        self.assertEqual(d["neighborhood"]["max_samples"], 20)


# ---------------------------------------------------------------------------
class TestKNNResult(unittest.TestCase):
    def _make_result(self) -> KNNResult:
        return KNNResult(MagicMock(), _make_result_model())

    def test_message(self):
        self.assertIn("completed", self._make_result().message)

    def test_object_reference(self):
        self.assertEqual(self._make_result().object_reference, GRID_URL)

    def test_str(self):
        s = str(self._make_result())
        self.assertIn("KNN", s)
        self.assertIn(GRID_URL, s)


# ---------------------------------------------------------------------------
class TestKNNRunnerAsync(unittest.IsolatedAsyncioTestCase):
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
            result = await KNNRunner(ctx, params)

        mock_submit.assert_called_once()
        _, kwargs = mock_submit.call_args
        self.assertEqual(kwargs["topic"], "geostat")
        self.assertEqual(kwargs["task"], "knn")
        self.assertIsInstance(result, KNNResult)


if __name__ == "__main__":
    unittest.main()
