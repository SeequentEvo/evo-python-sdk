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

"""Tests for break-ties task parameter handling."""

from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, MagicMock, patch

from evo.objects import ObjectReference
from evo.objects.typed.attributes import BlockModelPendingAttribute, PendingAttribute

from evo.compute.tasks import (
    CreateAttribute,
    SearchNeighborhood,
    Source,
    Target,
    UpdateAttribute,
)
from evo.compute.tasks.break_ties import (
    BreakTiesParameters,
    BreakTiesResult,
    BreakTiesResultModel,
    BreakTiesRunner,
)
from evo.compute.tasks.common import (
    Ellipsoid,
    EllipsoidRanges,
    Rotation,
)
from evo.compute.tasks.common.results import TaskAttribute, TaskTarget
from evo.compute.tasks.common.runner import TaskRegistry

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


def _search(
    major: float = 200.0, semi_major: float = 150.0, minor: float = 100.0, max_samples: int = 20
) -> SearchNeighborhood:
    return SearchNeighborhood(
        ellipsoid=Ellipsoid(ranges=EllipsoidRanges(major=major, semi_major=semi_major, minor=minor)),
        max_samples=max_samples,
    )


def _make_result_model(target_name: str = "MyPointset", attr_name: str = "grade_bt") -> BreakTiesResultModel:
    return BreakTiesResultModel(
        message="Break ties completed successfully.",
        target=TaskTarget(
            reference=TARGET_URL,
            name=target_name,
            description=None,
            schema_id="/objects/pointset/1.3.0/pointset.schema.json",
            attribute=TaskAttribute(reference="locations.attributes[?name=='grade_bt']", name=attr_name),
        ),
    )


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


class TestBreakTiesRegistration(TestCase):
    def test_break_ties_parameters_registered(self):
        registry = TaskRegistry()
        runner_cls = registry.get_runner(BreakTiesParameters)
        self.assertIsNotNone(runner_cls)
        self.assertIs(runner_cls, BreakTiesRunner)

    def test_runner_topic_and_task(self):
        self.assertEqual(BreakTiesRunner.topic, "geostatistics")
        self.assertEqual(BreakTiesRunner.task, "break-ties")

    def test_runner_types(self):
        self.assertIs(BreakTiesRunner.params_type, BreakTiesParameters)
        self.assertIs(BreakTiesRunner.result_model_type, BreakTiesResultModel)
        self.assertIs(BreakTiesRunner.result_type, BreakTiesResult)


# ---------------------------------------------------------------------------
# Parameter serialization tests
# ---------------------------------------------------------------------------


class TestBreakTiesParametersSerialization(TestCase):
    def _params(self, **kwargs) -> BreakTiesParameters:
        defaults = dict(
            source=Source(object=POINTSET_URL, attribute="locations.attributes[?name=='grade']"),
            target=Target(object=TARGET_URL, attribute=CreateAttribute(name="grade_bt")),
            neighborhood=_search(),
        )
        defaults.update(kwargs)
        return BreakTiesParameters(**defaults)

    def _dump(self, params: BreakTiesParameters) -> dict:
        return params.model_dump(mode="json", by_alias=True, exclude_none=True)

    def test_source_serializes_correctly(self):
        d = self._dump(self._params())
        self.assertEqual(d["source"]["object"], POINTSET_URL)
        self.assertEqual(d["source"]["attribute"], "locations.attributes[?name=='grade']")

    def test_target_create_serializes_correctly(self):
        d = self._dump(self._params())
        self.assertEqual(d["target"]["object"], TARGET_URL)
        self.assertEqual(d["target"]["attribute"]["operation"], "create")
        self.assertEqual(d["target"]["attribute"]["name"], "grade_bt")

    def test_target_update_serializes_correctly(self):
        params = self._params(
            target=Target(
                object=TARGET_URL, attribute=UpdateAttribute(reference="locations.attributes[?name=='grade']")
            )
        )
        d = self._dump(params)
        self.assertEqual(d["target"]["attribute"]["operation"], "update")
        self.assertEqual(d["target"]["attribute"]["reference"], "locations.attributes[?name=='grade']")

    def test_neighborhood_serializes_correctly(self):
        d = self._dump(self._params())
        nbh = d["neighborhood"]
        self.assertIn("ellipsoid", nbh)
        self.assertEqual(nbh["max_samples"], 20)
        ellipsoid = nbh["ellipsoid"]
        self.assertEqual(ellipsoid["ellipsoid_ranges"]["major"], 200.0)
        self.assertEqual(ellipsoid["ellipsoid_ranges"]["semi_major"], 150.0)
        self.assertEqual(ellipsoid["ellipsoid_ranges"]["minor"], 100.0)

    def test_min_samples_included_when_set(self):
        params = self._params(
            neighborhood=SearchNeighborhood(
                ellipsoid=Ellipsoid(ranges=EllipsoidRanges(major=200, semi_major=150, minor=100)),
                max_samples=20,
                min_samples=4,
            )
        )
        d = self._dump(params)
        self.assertEqual(d["neighborhood"]["min_samples"], 4)

    def test_min_samples_absent_when_not_set(self):
        d = self._dump(self._params())
        self.assertNotIn("min_samples", d["neighborhood"])

    def test_default_seed(self):
        d = self._dump(self._params())
        self.assertEqual(d["seed"], 38239342)

    def test_custom_seed(self):
        d = self._dump(self._params(seed=12345))
        self.assertEqual(d["seed"], 12345)

    def test_neighborhood_with_rotation(self):
        params = self._params(
            neighborhood=SearchNeighborhood(
                ellipsoid=Ellipsoid(
                    ranges=EllipsoidRanges(major=300, semi_major=200, minor=100),
                    rotation=Rotation(dip_azimuth=45.0, dip=30.0, pitch=0.0),
                ),
                max_samples=16,
            )
        )
        d = self._dump(params)
        rotation = d["neighborhood"]["ellipsoid"]["rotation"]
        self.assertEqual(rotation["dip_azimuth"], 45.0)
        self.assertEqual(rotation["dip"], 30.0)

    def test_pending_attribute_target(self):
        mock_obj = MagicMock()
        mock_obj.metadata.url = ObjectReference(TARGET_URL)
        mock_parent = MagicMock()
        mock_parent._obj = mock_obj
        pending = PendingAttribute(mock_parent, "new_grade_bt")

        params = self._params(target=pending)
        d = self._dump(params)
        self.assertEqual(d["target"]["object"], TARGET_URL)
        self.assertEqual(d["target"]["attribute"]["operation"], "create")
        self.assertEqual(d["target"]["attribute"]["name"], "new_grade_bt")

    def test_block_model_pending_attribute_target(self):
        mock_bm = MagicMock()
        mock_bm.metadata.url = ObjectReference(TARGET_URL)
        bm_pending = BlockModelPendingAttribute(obj=mock_bm, name="bt_attr")

        params = self._params(target=bm_pending)
        d = self._dump(params)
        self.assertEqual(d["target"]["attribute"]["operation"], "create")
        self.assertEqual(d["target"]["attribute"]["name"], "bt_attr")


# ---------------------------------------------------------------------------
# Result type tests
# ---------------------------------------------------------------------------


class TestBreakTiesResult(TestCase):
    def _make_result(self, model: BreakTiesResultModel | None = None) -> BreakTiesResult:
        context = MagicMock()
        return BreakTiesResult(context, model or _make_result_model())

    def test_message(self):
        result = self._make_result()
        self.assertEqual(result.message, "Break ties completed successfully.")

    def test_target_name(self):
        result = self._make_result()
        self.assertEqual(result.target_name, "MyPointset")

    def test_target_reference(self):
        result = self._make_result()
        self.assertEqual(result.target_reference, TARGET_URL)

    def test_attribute_name(self):
        result = self._make_result()
        self.assertEqual(result.attribute_name, "grade_bt")

    def test_str_representation(self):
        result = self._make_result()
        s = str(result)
        self.assertIn("Break Ties Result", s)
        self.assertIn("MyPointset", s)
        self.assertIn("grade_bt", s)

    def test_task_display_name(self):
        self.assertEqual(BreakTiesResult.TASK_DISPLAY_NAME, "Break Ties")

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
        model = BreakTiesResultModel.model_validate(data)
        self.assertIsInstance(model, BreakTiesResultModel)
        self.assertEqual(model.message, "ok")


# ---------------------------------------------------------------------------
# Runner behavior tests
# ---------------------------------------------------------------------------


class TestBreakTiesRunnerBehavior(
    TestCase.IsolatedAsyncioTestCase if hasattr(TestCase, "IsolatedAsyncioTestCase") else TestCase
):
    pass


class TestBreakTiesRunnerAsync(IsolatedAsyncioTestCase):
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
        params = BreakTiesParameters(
            source=Source(object=POINTSET_URL, attribute="locations.attributes[?name=='grade']"),
            target=Target(object=TARGET_URL, attribute=CreateAttribute(name="grade_bt")),
            neighborhood=_search(),
        )

        with patch(
            "evo.compute.tasks.common.runner.JobClient.submit",
            new_callable=AsyncMock,
            return_value=self._make_job(),
        ) as mock_submit:
            result = await BreakTiesRunner(context, params)

        mock_submit.assert_called_once()
        _, kwargs = mock_submit.call_args
        self.assertEqual(kwargs["topic"], "geostatistics")
        self.assertEqual(kwargs["task"], "break-ties")
        self.assertIsInstance(result, BreakTiesResult)

    async def test_runner_passes_preview_flag(self):
        context = self._make_context()
        params = BreakTiesParameters(
            source=Source(object=POINTSET_URL, attribute="locations.attributes[?name=='grade']"),
            target=Target(object=TARGET_URL, attribute=CreateAttribute(name="grade_bt")),
            neighborhood=_search(),
        )

        with patch(
            "evo.compute.tasks.common.runner.JobClient.submit",
            new_callable=AsyncMock,
            return_value=self._make_job(),
        ) as mock_submit:
            await BreakTiesRunner(context, params, preview=True)

        _, kwargs = mock_submit.call_args
        self.assertTrue(kwargs.get("preview", False))

    async def test_runner_preview_defaults_false(self):
        context = self._make_context()
        params = BreakTiesParameters(
            source=Source(object=POINTSET_URL, attribute="locations.attributes[?name=='grade']"),
            target=Target(object=TARGET_URL, attribute=CreateAttribute(name="grade_bt")),
            neighborhood=_search(),
        )

        with patch(
            "evo.compute.tasks.common.runner.JobClient.submit",
            new_callable=AsyncMock,
            return_value=self._make_job(),
        ) as mock_submit:
            await BreakTiesRunner(context, params)

        _, kwargs = mock_submit.call_args
        self.assertFalse(kwargs.get("preview", True))
