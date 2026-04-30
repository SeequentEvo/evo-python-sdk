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

"""Tests for location-wise task parameter handling."""

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from evo.compute.tasks import Source
from evo.compute.tasks.common.results import TaskAttribute
from evo.compute.tasks.common.runner import TaskRegistry
from evo.compute.tasks.location_wise import (
    LocationWiseParameters,
    LocationWiseResult,
    LocationWiseResultModel,
    LocationWiseRunner,
    LocationWiseTarget,
    LocationWiseTargetResult,
    MeanAboveCutoff,
    ProbabilityAboveCutoff,
)

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

_BASE = "https://hub.test.evo.bentley.com"
_ORG = "00000000-0000-0000-0000-000000000001"
_WS = "00000000-0000-0000-0000-000000000002"


def _obj_url(obj_id: str = "00000000-0000-0000-0000-000000000003") -> str:
    return f"{_BASE}/geoscience-object/orgs/{_ORG}/workspaces/{_WS}/objects/{obj_id}"


GRID_URL = _obj_url("00000000-0000-0000-0000-000000000010")


def _params(**kwargs) -> LocationWiseParameters:
    defaults = dict(
        source=Source(object=GRID_URL, attribute="cell_attributes[0]"),
        target=LocationWiseTarget(object=GRID_URL),
    )
    defaults.update(kwargs)
    return LocationWiseParameters(**defaults)


def _dump(params: LocationWiseParameters) -> dict:
    return params.model_dump(mode="json", by_alias=True, exclude_none=True)


def _make_result_model(attrs=None) -> LocationWiseResultModel:
    if attrs is None:
        attrs = [
            TaskAttribute(reference="ref_p10", name="Quantile: 0.1"),
            TaskAttribute(reference="ref_p50", name="Quantile: 0.5"),
            TaskAttribute(reference="ref_p90", name="Quantile: 0.9"),
        ]
    return LocationWiseResultModel(
        message="Location-wise completed.",
        target=LocationWiseTargetResult(
            reference=GRID_URL,
            name="MyGrid",
            schema_id="/objects/regular-3d-grid/1.3.0/regular-3d-grid.schema.json",
            attributes=attrs,
        ),
    )


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


class TestLocationWiseRegistration(unittest.TestCase):
    def test_registered(self):
        runner_cls = TaskRegistry().get_runner(LocationWiseParameters)
        self.assertIs(runner_cls, LocationWiseRunner)

    def test_topic_and_task(self):
        self.assertEqual(LocationWiseRunner.topic, "geostatistics")
        self.assertEqual(LocationWiseRunner.task, "location-wise")

    def test_runner_types(self):
        self.assertIs(LocationWiseRunner.params_type, LocationWiseParameters)
        self.assertIs(LocationWiseRunner.result_model_type, LocationWiseResultModel)
        self.assertIs(LocationWiseRunner.result_type, LocationWiseResult)


# ---------------------------------------------------------------------------
# Target tests
# ---------------------------------------------------------------------------


class TestLocationWiseTarget(unittest.TestCase):
    def test_target_only_object(self):
        t = LocationWiseTarget(object=GRID_URL)
        d = t.model_dump(mode="json", exclude_none=True)
        self.assertEqual(d["object"], GRID_URL)
        self.assertNotIn("attribute", d)


# ---------------------------------------------------------------------------
# Parameter serialization tests
# ---------------------------------------------------------------------------


class TestLocationWiseParametersSerialization(unittest.TestCase):
    def test_summary_only(self):
        d = _dump(_params(summary=True))
        self.assertTrue(d["summary"])
        self.assertNotIn("quantiles", d)
        self.assertNotIn("probability_above_cutoff", d)
        self.assertNotIn("mean_above_cutoff", d)

    def test_quantiles_only(self):
        d = _dump(_params(quantiles=[0.1, 0.5, 0.9]))
        self.assertEqual(d["quantiles"], [0.1, 0.5, 0.9])
        self.assertNotIn("summary", d)

    def test_probability_above_cutoff(self):
        d = _dump(_params(
            probability_above_cutoff=ProbabilityAboveCutoff(cutoffs=[0.5, 1.0, 2.0])
        ))
        self.assertEqual(d["probability_above_cutoff"]["cutoffs"], [0.5, 1.0, 2.0])

    def test_mean_above_cutoff(self):
        d = _dump(_params(
            mean_above_cutoff=MeanAboveCutoff(cutoffs=[0.5, 1.0])
        ))
        self.assertEqual(d["mean_above_cutoff"]["cutoffs"], [0.5, 1.0])

    def test_all_operations(self):
        d = _dump(_params(
            summary=True,
            quantiles=[0.1, 0.5, 0.9],
            probability_above_cutoff=ProbabilityAboveCutoff(cutoffs=[0.5, 1.0]),
            mean_above_cutoff=MeanAboveCutoff(cutoffs=[0.5]),
        ))
        self.assertTrue(d["summary"])
        self.assertEqual(d["quantiles"], [0.1, 0.5, 0.9])
        self.assertEqual(d["probability_above_cutoff"]["cutoffs"], [0.5, 1.0])
        self.assertEqual(d["mean_above_cutoff"]["cutoffs"], [0.5])

    def test_source_format(self):
        d = _dump(_params(summary=True))
        self.assertEqual(d["source"]["object"], GRID_URL)
        self.assertEqual(d["source"]["attribute"], "cell_attributes[0]")

    def test_target_format(self):
        d = _dump(_params(summary=True))
        self.assertEqual(d["target"]["object"], GRID_URL)

    def test_none_fields_excluded(self):
        d = _dump(_params(summary=True))
        self.assertNotIn("probability_above_cutoff", d)
        self.assertNotIn("mean_above_cutoff", d)
        self.assertNotIn("quantiles", d)


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


class TestProbabilityAboveCutoffValidation(unittest.TestCase):
    def test_empty_cutoffs_rejected(self):
        with self.assertRaises(Exception):
            ProbabilityAboveCutoff(cutoffs=[])

    def test_single_cutoff_ok(self):
        p = ProbabilityAboveCutoff(cutoffs=[1.0])
        self.assertEqual(p.cutoffs, [1.0])


class TestMeanAboveCutoffValidation(unittest.TestCase):
    def test_empty_cutoffs_rejected(self):
        with self.assertRaises(Exception):
            MeanAboveCutoff(cutoffs=[])

    def test_single_cutoff_ok(self):
        m = MeanAboveCutoff(cutoffs=[1.0])
        self.assertEqual(m.cutoffs, [1.0])


# ---------------------------------------------------------------------------
# Result model tests
# ---------------------------------------------------------------------------


class TestLocationWiseResultModel(unittest.TestCase):
    def test_result_model_validate(self):
        data = {
            "message": "ok",
            "target": {
                "reference": GRID_URL,
                "name": "target",
                "schema_id": "/objects/regular-3d-grid/1.3.0/regular-3d-grid.schema.json",
                "attributes": [
                    {"reference": "ref_1", "name": "attr_1"},
                    {"reference": "ref_2", "name": "attr_2"},
                ],
            },
        }
        model = LocationWiseResultModel.model_validate(data)
        self.assertIsInstance(model, LocationWiseResultModel)
        self.assertEqual(model.message, "ok")
        self.assertEqual(len(model.target.attributes), 2)


# ---------------------------------------------------------------------------
# Result wrapper tests
# ---------------------------------------------------------------------------


class TestLocationWiseResult(unittest.TestCase):
    def _make_result(self) -> LocationWiseResult:
        return LocationWiseResult(MagicMock(), _make_result_model())

    def test_message(self):
        self.assertIn("completed", self._make_result().message)

    def test_target_name(self):
        self.assertEqual(self._make_result().target_name, "MyGrid")

    def test_target_reference(self):
        self.assertEqual(self._make_result().target_reference, GRID_URL)

    def test_attribute_names(self):
        r = self._make_result()
        self.assertEqual(r.attribute_names, ["Quantile: 0.1", "Quantile: 0.5", "Quantile: 0.9"])

    def test_summary_attributes(self):
        summary_attrs = [
            TaskAttribute(reference="ref_min", name="min"),
            TaskAttribute(reference="ref_max", name="max"),
            TaskAttribute(reference="ref_mean", name="mean"),
            TaskAttribute(reference="ref_var", name="variance"),
        ]
        model = _make_result_model(attrs=summary_attrs)
        r = LocationWiseResult(MagicMock(), model)
        self.assertEqual(r.attribute_names, ["min", "max", "mean", "variance"])

    def test_str(self):
        s = str(self._make_result())
        self.assertIn("Location-Wise", s)
        self.assertIn("Quantile:", s)


# ---------------------------------------------------------------------------
# Runner behavior tests
# ---------------------------------------------------------------------------


class TestLocationWiseRunnerAsync(unittest.IsolatedAsyncioTestCase):
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
        params = _params(summary=True, quantiles=[0.1, 0.5, 0.9])

        with patch(
            "evo.compute.tasks.common.runner.JobClient.submit",
            new_callable=AsyncMock,
            return_value=self._make_job(),
        ) as mock_submit:
            result = await LocationWiseRunner(context, params)

        mock_submit.assert_called_once()
        _, kwargs = mock_submit.call_args
        self.assertEqual(kwargs["topic"], "geostatistics")
        self.assertEqual(kwargs["task"], "location-wise")
        self.assertIsInstance(result, LocationWiseResult)

    async def test_runner_preview_defaults_false(self):
        context = self._make_context()
        params = _params(summary=True)

        with patch(
            "evo.compute.tasks.common.runner.JobClient.submit",
            new_callable=AsyncMock,
            return_value=self._make_job(),
        ) as mock_submit:
            await LocationWiseRunner(context, params)

        _, kwargs = mock_submit.call_args
        self.assertFalse(kwargs.get("preview", True))


if __name__ == "__main__":
    unittest.main()
