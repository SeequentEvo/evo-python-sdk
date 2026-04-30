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

"""Tests for normal-score task parameter handling."""

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from evo.compute.tasks import (
    CreateAttribute,
    Source,
    Target,
    UpdateAttribute,
)
from evo.compute.tasks.common.results import TaskAttribute, TaskTarget
from evo.compute.tasks.common.runner import TaskRegistry
from evo.compute.tasks.normal_score import (
    NormalScoreParameters,
    NormalScoreResult,
    NormalScoreResultModel,
    NormalScoreRunner,
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
TARGET_URL = _obj_url("00000000-0000-0000-0000-000000000020")
DIST_URL = _obj_url("00000000-0000-0000-0000-000000000030")


def _params(**kwargs) -> NormalScoreParameters:
    defaults = dict(
        method="forward",
        source=Source(object=POINTSET_URL, attribute="locations.attributes[0]"),
        target=Target(object=TARGET_URL, attribute=CreateAttribute(name="grade_ns")),
        distribution=DIST_URL,
    )
    defaults.update(kwargs)
    return NormalScoreParameters(**defaults)


def _dump(params: NormalScoreParameters) -> dict:
    return params.model_dump(mode="json", by_alias=True, exclude_none=True)


def _make_result_model(
    target_name: str = "MyPointset",
    attr_name: str = "grade_ns",
) -> NormalScoreResultModel:
    return NormalScoreResultModel(
        message="Normal score transform completed successfully.",
        target=TaskTarget(
            reference=TARGET_URL,
            name=target_name,
            description=None,
            schema_id="/objects/pointset/1.3.0/pointset.schema.json",
            attribute=TaskAttribute(
                reference="locations.attributes[?name=='grade_ns']",
                name=attr_name,
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


class TestNormalScoreRegistration(unittest.TestCase):
    def test_registered(self):
        runner_cls = TaskRegistry().get_runner(NormalScoreParameters)
        self.assertIs(runner_cls, NormalScoreRunner)

    def test_topic_and_task(self):
        self.assertEqual(NormalScoreRunner.topic, "geostatistics")
        self.assertEqual(NormalScoreRunner.task, "normal-score")

    def test_runner_types(self):
        self.assertIs(NormalScoreRunner.params_type, NormalScoreParameters)
        self.assertIs(NormalScoreRunner.result_model_type, NormalScoreResultModel)
        self.assertIs(NormalScoreRunner.result_type, NormalScoreResult)


# ---------------------------------------------------------------------------
# Parameter serialization tests
# ---------------------------------------------------------------------------


class TestNormalScoreParametersSerialization(unittest.TestCase):
    def test_forward_method(self):
        d = _dump(_params(method="forward"))
        self.assertEqual(d["method"], "forward")

    def test_backward_method(self):
        d = _dump(_params(method="backward"))
        self.assertEqual(d["method"], "backward")

    def test_source_new_format(self):
        d = _dump(_params())
        self.assertEqual(d["source"]["object"], POINTSET_URL)
        self.assertEqual(d["source"]["attribute"], "locations.attributes[0]")

    def test_target_with_create_attribute(self):
        d = _dump(_params())
        self.assertEqual(d["target"]["object"], TARGET_URL)
        self.assertEqual(d["target"]["attribute"]["name"], "grade_ns")
        self.assertEqual(d["target"]["attribute"]["operation"], "create")

    def test_target_with_update_attribute(self):
        d = _dump(_params(
            target=Target(
                object=TARGET_URL,
                attribute=UpdateAttribute(reference="locations.attributes[?name=='grade_ns']"),
            ),
        ))
        self.assertEqual(d["target"]["attribute"]["operation"], "update")

    def test_distribution_is_url(self):
        d = _dump(_params())
        self.assertEqual(d["distribution"], DIST_URL)

    def test_invalid_method_rejected(self):
        with self.assertRaises(Exception):
            _params(method="invalid")

    def test_full_serialization(self):
        d = _dump(_params())
        self.assertIn("method", d)
        self.assertIn("source", d)
        self.assertIn("target", d)
        self.assertIn("distribution", d)


# ---------------------------------------------------------------------------
# Result model tests
# ---------------------------------------------------------------------------


class TestNormalScoreResultModel(unittest.TestCase):
    def test_result_model_validate(self):
        data = {
            "message": "ok",
            "target": {
                "reference": TARGET_URL,
                "name": "target",
                "schema_id": "/objects/pointset/1.3.0/pointset.schema.json",
                "attribute": {"reference": "attr_ref", "name": "attr_name"},
            },
        }
        model = NormalScoreResultModel.model_validate(data)
        self.assertIsInstance(model, NormalScoreResultModel)
        self.assertEqual(model.message, "ok")

    def test_result_model_target_attribute(self):
        model = _make_result_model()
        self.assertEqual(model.target.attribute.name, "grade_ns")


# ---------------------------------------------------------------------------
# Result wrapper tests
# ---------------------------------------------------------------------------


class TestNormalScoreResult(unittest.TestCase):
    def _make_result(self) -> NormalScoreResult:
        return NormalScoreResult(MagicMock(), _make_result_model())

    def test_message(self):
        self.assertIn("completed", self._make_result().message)

    def test_target_name(self):
        self.assertEqual(self._make_result().target_name, "MyPointset")

    def test_target_reference(self):
        self.assertEqual(self._make_result().target_reference, TARGET_URL)

    def test_attribute_name(self):
        self.assertEqual(self._make_result().attribute_name, "grade_ns")

    def test_str(self):
        s = str(self._make_result())
        self.assertIn("Normal Score", s)
        self.assertIn("grade_ns", s)


# ---------------------------------------------------------------------------
# Runner behavior tests
# ---------------------------------------------------------------------------


class TestNormalScoreRunnerAsync(unittest.IsolatedAsyncioTestCase):
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
            result = await NormalScoreRunner(context, params)

        mock_submit.assert_called_once()
        _, kwargs = mock_submit.call_args
        self.assertEqual(kwargs["topic"], "geostatistics")
        self.assertEqual(kwargs["task"], "normal-score")
        self.assertIsInstance(result, NormalScoreResult)

    async def test_runner_preview_defaults_false(self):
        context = self._make_context()
        params = _params()

        with patch(
            "evo.compute.tasks.common.runner.JobClient.submit",
            new_callable=AsyncMock,
            return_value=self._make_job(),
        ) as mock_submit:
            await NormalScoreRunner(context, params)

        _, kwargs = mock_submit.call_args
        self.assertFalse(kwargs.get("preview", True))


if __name__ == "__main__":
    unittest.main()
