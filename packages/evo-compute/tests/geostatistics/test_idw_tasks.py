#  Copyright © 2026 Bentley Systems, Incorporated
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Tests for the IDW (inverse distance weighting) estimation task SDK client."""

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from pydantic import ValidationError

from evo.compute.tasks import SearchNeighborhood
from evo.compute.tasks.common import (
    Ellipsoid,
    EllipsoidRanges,
    Filter,
    FilterCondition,
    Source,
    Target,
    UpdateAttribute,
)
from evo.compute.tasks.common.results import TaskAttribute, TaskTarget
from evo.compute.tasks.common.runner import TaskRegistry
from evo.compute.tasks.geostatistics.idw import (
    IDWParameters,
    IDWResult,
    IDWResultModel,
    IDWRunner,
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


def _search() -> SearchNeighborhood:
    return SearchNeighborhood(
        ellipsoid=Ellipsoid(ranges=EllipsoidRanges(major=200, semi_major=150, minor=100)),
        max_samples=20,
    )


def _params(**kwargs) -> IDWParameters:
    defaults = dict(
        source=Source(object=POINTSET_URL, attribute="locations.attributes[?name=='grade']"),
        target=Target.new_attribute(GRID_URL, "idw_grade"),
        neighborhood=_search(),
        power=2.0,
    )
    defaults.update(kwargs)
    return IDWParameters(**defaults)


def _dump(params: IDWParameters) -> dict:
    return params.model_dump(mode="json", by_alias=True, exclude_none=True)


def _make_result_model() -> IDWResultModel:
    return IDWResultModel(
        message="IDW estimation completed.",
        target=TaskTarget(
            reference=GRID_URL,
            name="my-grid",
            schema_id="/objects/pointset/1.0.0/pointset.schema.json",
            attribute=TaskAttribute(reference="locations.attributes[0]", name="idw_grade"),
        ),
    )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestIDWRegistration(unittest.TestCase):
    def test_registered(self):
        runner_cls = TaskRegistry().get_runner(IDWParameters)
        self.assertIs(runner_cls, IDWRunner)

    def test_topic_and_task(self):
        self.assertEqual(IDWRunner.topic, "geostatistics")
        self.assertEqual(IDWRunner.task, "idw")

    def test_runner_types(self):
        self.assertIs(IDWRunner.params_type, IDWParameters)
        self.assertIs(IDWRunner.result_model_type, IDWResultModel)
        self.assertIs(IDWRunner.result_type, IDWResult)


# ---------------------------------------------------------------------------
# Parameter serialization
# ---------------------------------------------------------------------------


class TestIDWParametersSerialization(unittest.TestCase):
    def test_basic_fields_present(self):
        d = _dump(_params())
        self.assertEqual(d["source"]["object"], POINTSET_URL)
        self.assertEqual(d["source"]["attribute"], "locations.attributes[?name=='grade']")
        self.assertEqual(d["target"]["object"], GRID_URL)
        self.assertIn("neighborhood", d)
        self.assertEqual(d["power"], 2.0)

    def test_power_is_serialized(self):
        d = _dump(_params(power=3.0))
        self.assertEqual(d["power"], 3.0)

    def test_power_must_be_positive(self):
        with self.assertRaises(ValidationError):
            _params(power=0.0)
        with self.assertRaises(ValidationError):
            _params(power=-1.0)

    def test_different_power_values(self):
        for power in (0.5, 1.0, 2.0, 3.5):
            d = _dump(_params(power=power))
            self.assertAlmostEqual(d["power"], power)

    def test_neighborhood_fields(self):
        d = _dump(_params())
        nbr = d["neighborhood"]
        self.assertIn("ellipsoid", nbr)
        self.assertEqual(nbr["max_samples"], 20)

    def test_new_attribute_target(self):
        d = _dump(_params(target=Target.new_attribute(GRID_URL, "result")))
        self.assertEqual(d["target"]["attribute"]["operation"], "create")
        self.assertEqual(d["target"]["attribute"]["name"], "result")

    def test_update_attribute_target(self):
        from evo.compute.tasks.common.source_target import Target as _Target

        target = _Target(
            object=GRID_URL,
            attribute=UpdateAttribute(operation="update", name="existing", reference="attributes[?key=='abc']"),
        )
        d = _dump(_params(target=target))
        self.assertEqual(d["target"]["attribute"]["operation"], "update")
        self.assertEqual(d["target"]["attribute"]["reference"], "attributes[?key=='abc']")

    def test_no_filter_by_default(self):
        d = _dump(_params())
        self.assertNotIn("filter", d.get("target", {}))
        self.assertNotIn("filter", d.get("source", {}))

    def test_target_filter_injected_into_target(self):
        params = _params(
            target_filter=Filter(
                where=FilterCondition(
                    attribute="attributes[?name=='domain']",
                    operator="in",
                    values=["LMS1", "LMS2"],
                ),
            )
        )
        d = _dump(params)
        f = d["target"]["filter"]
        self.assertEqual(f["where"]["attribute"], "attributes[?name=='domain']")
        self.assertEqual(f["where"]["operator"], "in")
        self.assertEqual(f["where"]["values"], ["LMS1", "LMS2"])
        self.assertNotIn("filter", d["source"])

    def test_target_filter_with_integer_keys(self):
        params = _params(
            target_filter=Filter(
                where=FilterCondition(
                    attribute="attributes[?name=='domain']",
                    operator="in",
                    values=[1, 2, 3],
                ),
            )
        )
        d = _dump(params)
        self.assertEqual(d["target"]["filter"]["where"]["values"], [1, 2, 3])

    def test_source_filter_injected_into_source(self):
        params = _params(
            source_filter=Filter(
                where=FilterCondition(attribute="grade", operator="greater_than", threshold=0.0),
            )
        )
        d = _dump(params)
        f = d["source"]["filter"]
        self.assertEqual(f["where"]["threshold"], 0.0)
        self.assertNotIn("filter", d["target"])

    def test_filter_excluded_from_top_level(self):
        """source_filter / target_filter must not appear at the top-level; only inside source/target."""
        params = _params(
            target_filter=Filter(
                where=FilterCondition(attribute="attributes[?name=='domain']", operator="in", values=["A"]),
            )
        )
        d = _dump(params)
        self.assertNotIn("target_filter", d)
        self.assertNotIn("source_filter", d)

    def test_min_samples_in_neighborhood(self):
        search = SearchNeighborhood(
            ellipsoid=Ellipsoid(ranges=EllipsoidRanges(major=200, semi_major=150, minor=100)),
            max_samples=20,
            min_samples=4,
        )
        d = _dump(_params(neighborhood=search))
        self.assertEqual(d["neighborhood"]["min_samples"], 4)


# ---------------------------------------------------------------------------
# Result handling
# ---------------------------------------------------------------------------


class TestIDWResult(unittest.TestCase):
    def _make(self) -> IDWResult:
        return IDWResult(MagicMock(), _make_result_model())

    def test_message(self):
        self.assertEqual(self._make().message, "IDW estimation completed.")

    def test_target_name(self):
        self.assertEqual(self._make().target_name, "my-grid")

    def test_target_reference(self):
        self.assertEqual(self._make().target_reference, GRID_URL)

    def test_attribute_name(self):
        self.assertEqual(self._make().attribute_name, "idw_grade")

    def test_str(self):
        s = str(self._make())
        self.assertIn("IDW Estimation", s)
        self.assertIn("my-grid", s)
        self.assertIn("idw_grade", s)


# ---------------------------------------------------------------------------
# Async runner
# ---------------------------------------------------------------------------


class TestIDWRunnerAsync(unittest.IsolatedAsyncioTestCase):
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
            result = await IDWRunner(ctx, params)

        mock_submit.assert_called_once()
        _, kwargs = mock_submit.call_args
        self.assertEqual(kwargs["topic"], "geostatistics")
        self.assertEqual(kwargs["task"], "idw")
        self.assertIsInstance(result, IDWResult)

    async def test_runner_passes_power_in_payload(self):
        ctx = self._make_context()
        params = _params(power=3.0)
        job = AsyncMock()
        job.wait_for_results.return_value = _make_result_model()

        captured_payload = {}

        async def capture_submit(**kwargs):
            captured_payload.update(kwargs)
            return job

        with patch("evo.compute.tasks.common.runner.JobClient.submit", side_effect=capture_submit):
            await IDWRunner(ctx, params)

        payload = captured_payload.get("parameters", {})
        self.assertAlmostEqual(payload.get("power"), 3.0)
