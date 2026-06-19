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

"""Tests for the conditional turning-bands simulation task SDK client."""

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from pydantic import ValidationError

from evo.compute.tasks import SearchNeighborhood
from evo.compute.tasks.common import (
    AllOfFilter,
    AnyOfFilter,
    Ellipsoid,
    EllipsoidRanges,
    Filter,
    FilterCondition,
)
from evo.compute.tasks.common.results import TaskAttribute
from evo.compute.tasks.common.runner import TaskRegistry
from evo.compute.tasks.geostatistics.conditional_turning_bands import (
    ConditionalTurningBandsParameters,
    ConditionalTurningBandsResult,
    ConditionalTurningBandsResultModel,
    ConditionalTurningBandsRunner,
    ConditionalTurningBandsTargetResult,
)
from evo.compute.tasks.geostatistics.conditioned_simulator import BlockDiscretization

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
DISTRIBUTION_URL = _obj_url("00000000-0000-0000-0000-000000000040")


def _search() -> SearchNeighborhood:
    return SearchNeighborhood(
        ellipsoid=Ellipsoid(ranges=EllipsoidRanges(major=70, semi_major=70, minor=5)),
        max_samples=40,
    )


def _params(**kwargs) -> ConditionalTurningBandsParameters:
    defaults = dict(
        source=POINTSET_URL,
        source_attribute="locations.attributes[?name=='grade']",
        target=GRID_URL,
        distribution=DISTRIBUTION_URL,
        variogram_model=VARIOGRAM_URL,
        neighborhood=_search(),
    )
    defaults.update(kwargs)
    return ConditionalTurningBandsParameters(**defaults)


def _dump(params: ConditionalTurningBandsParameters) -> dict:
    return params.model_dump(mode="json", by_alias=True, exclude_none=True)


def _make_result_model(*, with_simulations: bool = True) -> ConditionalTurningBandsResultModel:
    return ConditionalTurningBandsResultModel(
        target=ConditionalTurningBandsTargetResult(
            reference=GRID_URL,
            name="my-grid",
            schema_id="/objects/regular-3d-grid/1.3.0/regular-3d-grid.schema.json",
            simulations=TaskAttribute(reference="cell_attributes[0]", name="simulation-results")
            if with_simulations
            else None,
        )
    )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestConditionalTurningBandsRegistration(unittest.TestCase):
    def test_registered(self):
        runner_cls = TaskRegistry().get_runner(ConditionalTurningBandsParameters)
        self.assertIs(runner_cls, ConditionalTurningBandsRunner)

    def test_topic_and_task(self):
        self.assertEqual(ConditionalTurningBandsRunner.topic, "geostatistics")
        self.assertEqual(ConditionalTurningBandsRunner.task, "conditional-turning-bands")

    def test_runner_types(self):
        self.assertIs(ConditionalTurningBandsRunner.params_type, ConditionalTurningBandsParameters)
        self.assertIs(ConditionalTurningBandsRunner.result_model_type, ConditionalTurningBandsResultModel)
        self.assertIs(ConditionalTurningBandsRunner.result_type, ConditionalTurningBandsResult)


# ---------------------------------------------------------------------------
# BlockDiscretization
# ---------------------------------------------------------------------------


class TestBlockDiscretization(unittest.TestCase):
    def test_defaults(self):
        bd = BlockDiscretization()
        self.assertEqual(bd.nx, 1)
        self.assertEqual(bd.ny, 1)
        self.assertEqual(bd.nz, 1)

    def test_custom(self):
        bd = BlockDiscretization(nx=5, ny=5, nz=5)
        d = bd.model_dump(mode="json")
        self.assertEqual(d["nx"], 5)
        self.assertEqual(d["nz"], 5)

    def test_out_of_range_rejected(self):
        with self.assertRaises(ValidationError):
            BlockDiscretization(nx=0)
        with self.assertRaises(ValidationError):
            BlockDiscretization(nx=10)


# ---------------------------------------------------------------------------
# Parameter serialization
# ---------------------------------------------------------------------------


class TestConditionalTurningBandsParametersSerialization(unittest.TestCase):
    def test_required_fields_present(self):
        d = _dump(_params())
        self.assertEqual(d["source"], POINTSET_URL)
        self.assertEqual(d["source_attribute"], "locations.attributes[?name=='grade']")
        self.assertEqual(d["target"], GRID_URL)
        self.assertEqual(d["distribution"], DISTRIBUTION_URL)
        self.assertEqual(d["variogram_model"], VARIOGRAM_URL)

    def test_defaults(self):
        d = _dump(_params())
        self.assertEqual(d["kriging_method"], "simple")
        self.assertEqual(d["number_of_lines"], 500)
        self.assertEqual(d["realizations"], 1)
        self.assertEqual(d["random_seed"], 38239342)

    def test_block_discretization_default(self):
        d = _dump(_params())
        bd = d["block_discretization"]
        self.assertEqual(bd["nx"], 1)
        self.assertEqual(bd["ny"], 1)
        self.assertEqual(bd["nz"], 1)

    def test_block_discretization_custom(self):
        d = _dump(_params(block_discretization=BlockDiscretization(nx=5, ny=5, nz=3)))
        bd = d["block_discretization"]
        self.assertEqual(bd["nx"], 5)
        self.assertEqual(bd["nz"], 3)

    def test_custom_realizations(self):
        d = _dump(_params(realizations=10))
        self.assertEqual(d["realizations"], 10)

    def test_custom_number_of_lines(self):
        d = _dump(_params(number_of_lines=250))
        self.assertEqual(d["number_of_lines"], 250)

    def test_ordinary_kriging_method(self):
        d = _dump(_params(kriging_method="ordinary"))
        self.assertEqual(d["kriging_method"], "ordinary")

    def test_realizations_out_of_range_rejected(self):
        with self.assertRaises(ValidationError):
            _params(realizations=0)
        with self.assertRaises(ValidationError):
            _params(realizations=101)

    def test_number_of_lines_out_of_range_rejected(self):
        with self.assertRaises(ValidationError):
            _params(number_of_lines=0)
        with self.assertRaises(ValidationError):
            _params(number_of_lines=1001)

    def test_no_filters_by_default(self):
        d = _dump(_params())
        self.assertNotIn("filter", d)
        self.assertNotIn("source_filter", d)

    def test_target_filter_single_condition(self):
        params = _params(
            filter=Filter(
                where=FilterCondition(
                    attribute="cell_attributes[?name=='domain']",
                    operator="in",
                    values=["LMS1", "LMS2"],
                )
            )
        )
        d = _dump(params)
        where = d["filter"]["where"]
        self.assertEqual(where["type"], "condition")
        self.assertEqual(where["operator"], "in")
        self.assertEqual(where["values"], ["LMS1", "LMS2"])

    def test_source_filter_numeric_condition(self):
        params = _params(
            source_filter=Filter(
                where=FilterCondition(
                    attribute="locations.attributes[?name=='grade']",
                    operator="greater_than",
                    threshold=0.5,
                )
            )
        )
        d = _dump(params)
        where = d["source_filter"]["where"]
        self.assertEqual(where["operator"], "greater_than")
        self.assertEqual(where["threshold"], 0.5)

    def test_composite_all_of_filter(self):
        params = _params(
            filter=Filter(
                where=AllOfFilter(
                    filters=[
                        FilterCondition(
                            attribute="cell_attributes[?name=='domain']",
                            operator="in",
                            values=[1, 2],
                        ),
                        FilterCondition(
                            attribute="cell_attributes[?name=='grade']",
                            operator="greater_than",
                            threshold=0.1,
                        ),
                    ]
                )
            )
        )
        d = _dump(params)
        where = d["filter"]["where"]
        self.assertEqual(where["type"], "all_of")
        self.assertEqual(len(where["filters"]), 2)

    def test_composite_any_of_filter(self):
        params = _params(
            source_filter=Filter(
                where=AnyOfFilter(
                    filters=[
                        FilterCondition(
                            attribute="locations.attributes[?name=='domain']",
                            operator="in",
                            values=["LMS1"],
                        ),
                        FilterCondition(
                            attribute="locations.attributes[?name=='grade']",
                            operator="less_than",
                            threshold=10.0,
                        ),
                    ]
                )
            )
        )
        d = _dump(params)
        where = d["source_filter"]["where"]
        self.assertEqual(where["type"], "any_of")

    def test_both_filter_and_source_filter(self):
        params = _params(
            filter=Filter(
                where=FilterCondition(
                    attribute="cell_attributes[?name=='domain']",
                    operator="in",
                    values=["LMS1"],
                )
            ),
            source_filter=Filter(
                where=FilterCondition(
                    attribute="locations.attributes[?name=='grade']",
                    operator="greater_than_or_equal_to",
                    threshold=0.5,
                )
            ),
        )
        d = _dump(params)
        self.assertIn("filter", d)
        self.assertIn("source_filter", d)


# ---------------------------------------------------------------------------
# Result handling
# ---------------------------------------------------------------------------


class TestConditionalTurningBandsResult(unittest.TestCase):
    def _make(self, **kwargs) -> ConditionalTurningBandsResult:
        return ConditionalTurningBandsResult(MagicMock(), _make_result_model(**kwargs))

    def test_target_name(self):
        self.assertEqual(self._make().target_name, "my-grid")

    def test_target_reference(self):
        self.assertEqual(self._make().target_reference, GRID_URL)

    def test_simulations_attribute_present(self):
        r = self._make(with_simulations=True)
        self.assertIsNotNone(r.simulations_attribute)
        self.assertEqual(r.simulations_attribute.name, "simulation-results")
        self.assertEqual(r.simulations_attribute.reference, "cell_attributes[0]")

    def test_simulations_attribute_none(self):
        r = self._make(with_simulations=False)
        self.assertIsNone(r.simulations_attribute)

    def test_str_contains_task_name(self):
        s = str(self._make())
        self.assertIn("Conditional Turning-Band Simulation", s)
        self.assertIn("my-grid", s)

    def test_str_contains_simulations_name_when_present(self):
        s = str(self._make(with_simulations=True))
        self.assertIn("simulation-results", s)

    def test_str_no_simulations_line_when_absent(self):
        s = str(self._make(with_simulations=False))
        self.assertNotIn("Simulations:", s)


# ---------------------------------------------------------------------------
# Async runner
# ---------------------------------------------------------------------------


class TestConditionalTurningBandsRunnerAsync(unittest.IsolatedAsyncioTestCase):
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
            result = await ConditionalTurningBandsRunner(ctx, params)

        mock_submit.assert_called_once()
        _, kwargs = mock_submit.call_args
        self.assertEqual(kwargs["topic"], "geostatistics")
        self.assertEqual(kwargs["task"], "conditional-turning-bands")
        self.assertIsInstance(result, ConditionalTurningBandsResult)
