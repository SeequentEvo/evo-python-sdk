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

"""Tests for declustering task parameter handling."""

from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, MagicMock, patch

from pydantic import ValidationError

from evo.compute.tasks import (
    CreateAttribute,
    SearchNeighborhood,
    Target,
    UpdateAttribute,
)
from evo.compute.tasks.common import (
    Ellipsoid,
    EllipsoidRanges,
    Rotation,
)
from evo.compute.tasks.common.results import TaskAttribute, TaskTarget
from evo.compute.tasks.common.runner import TaskRegistry
from evo.compute.tasks.geostatistics.declustering import (
    DeclusteringParameters,
    DeclusteringResult,
    DeclusteringResultModel,
    DeclusteringRunner,
    DeclusteringSource,
    idw,
    knn,
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
TARGET_URL = _obj_url("00000000-0000-0000-0000-000000000030")


def _search(
    major: float = 200.0, semi_major: float = 150.0, minor: float = 100.0, max_samples: int = 20
) -> SearchNeighborhood:
    return SearchNeighborhood(
        ellipsoid=Ellipsoid(ranges=EllipsoidRanges(major=major, semi_major=semi_major, minor=minor)),
        max_samples=max_samples,
    )


def _make_result_model(target_name: str = "MyPointset", attr_name: str = "weights") -> DeclusteringResultModel:
    return DeclusteringResultModel(
        message="Declustering completed successfully.",
        target=TaskTarget(
            reference=TARGET_URL,
            name=target_name,
            description=None,
            schema_id="/objects/pointset/1.3.0/pointset.schema.json",
            attribute=TaskAttribute(reference="locations.attributes[?name=='weights']", name=attr_name),
        ),
    )


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


class TestDeclusteringRegistration(TestCase):
    def test_declustering_parameters_registered(self):
        registry = TaskRegistry()
        runner_cls = registry.get_runner(DeclusteringParameters)
        self.assertIsNotNone(runner_cls)
        self.assertIs(runner_cls, DeclusteringRunner)

    def test_runner_topic_and_task(self):
        self.assertEqual(DeclusteringRunner.topic, "geostatistics")
        self.assertEqual(DeclusteringRunner.task, "declustering")

    def test_runner_types(self):
        self.assertIs(DeclusteringRunner.params_type, DeclusteringParameters)
        self.assertIs(DeclusteringRunner.result_model_type, DeclusteringResultModel)
        self.assertIs(DeclusteringRunner.result_type, DeclusteringResult)


# ---------------------------------------------------------------------------
# Parameter serialization tests
# ---------------------------------------------------------------------------


class TestDeclusteringParametersSerialization(TestCase):
    def _params(self, **kwargs) -> DeclusteringParameters:
        defaults = dict(
            source=POINTSET_URL,
            grid=GRID_URL,
            target=Target(object=TARGET_URL, attribute=CreateAttribute(name="weights")),
            neighborhood=_search(),
        )
        defaults.update(kwargs)
        return DeclusteringParameters(**defaults)

    def _dump(self, params: DeclusteringParameters) -> dict:
        return params.model_dump(mode="json", by_alias=True, exclude_none=True)

    def test_source_serializes_correctly(self):
        d = self._dump(self._params())
        self.assertEqual(d["source"]["object"], POINTSET_URL)

    def test_source_accepts_raw_url(self):
        params = self._params(source=POINTSET_URL)
        self.assertIsInstance(params.source, DeclusteringSource)
        self.assertEqual(str(params.source.object), POINTSET_URL)

    def test_grid_serializes_correctly(self):
        d = self._dump(self._params())
        self.assertEqual(d["grid"]["object"], GRID_URL)

    def test_target_create_serializes_correctly(self):
        d = self._dump(self._params())
        self.assertEqual(d["target"]["object"], TARGET_URL)
        self.assertEqual(d["target"]["attribute"]["operation"], "create")
        self.assertEqual(d["target"]["attribute"]["name"], "weights")

    def test_target_update_serializes_correctly(self):
        params = self._params(
            target=Target(
                object=TARGET_URL, attribute=UpdateAttribute(reference="locations.attributes[?name=='weights']")
            )
        )
        d = self._dump(params)
        self.assertEqual(d["target"]["attribute"]["operation"], "update")

    def test_neighborhood_serializes_correctly(self):
        d = self._dump(self._params())
        nbh = d["neighborhood"]
        self.assertIn("ellipsoid", nbh)
        self.assertEqual(nbh["max_samples"], 20)

    def test_default_power_is_idw(self):
        params = self._params()
        self.assertEqual(params.power, 2.0)

    def test_default_power_in_serialized_output(self):
        d = self._dump(self._params())
        self.assertEqual(d["power"], 2.0)

    def test_custom_power(self):
        d = self._dump(self._params(power=3.0))
        self.assertEqual(d["power"], 3.0)

    def test_knn_mode_power_none(self):
        params = self._params(power=None)
        self.assertIsNone(params.power)

    def test_knn_power_none_preserved_with_exclude_none(self):
        """When power=None, serialization must explicitly include power: null."""
        d = self._dump(self._params(power=None))
        self.assertIn("power", d)
        self.assertIsNone(d["power"])

    def test_negative_power_rejected(self):
        with self.assertRaises(ValidationError):
            self._params(power=-1.0)

    def test_zero_power_rejected(self):
        with self.assertRaises(ValidationError):
            self._params(power=0.0)

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


# ---------------------------------------------------------------------------
# Factory function tests
# ---------------------------------------------------------------------------


class TestDeclusteringFactoryFunctions(TestCase):
    def _target(self):
        return Target(object=TARGET_URL, attribute=CreateAttribute(name="weights"))

    def test_idw_returns_declustering_parameters(self):
        params = idw(source=POINTSET_URL, grid=GRID_URL, target=self._target(), neighborhood=_search())
        self.assertIsInstance(params, DeclusteringParameters)

    def test_idw_default_power(self):
        params = idw(source=POINTSET_URL, grid=GRID_URL, target=self._target(), neighborhood=_search())
        self.assertEqual(params.power, 2.0)

    def test_idw_custom_power(self):
        params = idw(source=POINTSET_URL, grid=GRID_URL, target=self._target(), neighborhood=_search(), power=3.5)
        self.assertEqual(params.power, 3.5)

    def test_idw_serializes_power(self):
        params = idw(source=POINTSET_URL, grid=GRID_URL, target=self._target(), neighborhood=_search())
        d = params.model_dump(mode="json", by_alias=True, exclude_none=True)
        self.assertEqual(d["power"], 2.0)

    def test_knn_returns_declustering_parameters(self):
        params = knn(source=POINTSET_URL, grid=GRID_URL, target=self._target(), neighborhood=_search())
        self.assertIsInstance(params, DeclusteringParameters)

    def test_knn_power_is_none(self):
        params = knn(source=POINTSET_URL, grid=GRID_URL, target=self._target(), neighborhood=_search())
        self.assertIsNone(params.power)

    def test_knn_preserves_null_power_in_serialization(self):
        params = knn(source=POINTSET_URL, grid=GRID_URL, target=self._target(), neighborhood=_search())
        d = params.model_dump(mode="json", by_alias=True, exclude_none=True)
        self.assertIn("power", d)
        self.assertIsNone(d["power"])

    def test_knn_with_single_neighbor(self):
        params = knn(source=POINTSET_URL, grid=GRID_URL, target=self._target(), neighborhood=_search(max_samples=1))
        self.assertIsNone(params.power)
        d = params.model_dump(mode="json", by_alias=True, exclude_none=True)
        self.assertEqual(d["neighborhood"]["max_samples"], 1)

    def test_idw_wraps_source_url(self):
        params = idw(source=POINTSET_URL, grid=GRID_URL, target=self._target(), neighborhood=_search())
        self.assertIsInstance(params.source, DeclusteringSource)

    def test_knn_wraps_source_url(self):
        params = knn(source=POINTSET_URL, grid=GRID_URL, target=self._target(), neighborhood=_search())
        self.assertIsInstance(params.source, DeclusteringSource)


# ---------------------------------------------------------------------------
# Result wrapper tests
# ---------------------------------------------------------------------------


class TestDeclusteringResult(TestCase):
    def _result(self, **kwargs) -> DeclusteringResult:
        model = _make_result_model(**kwargs)
        return DeclusteringResult(context=MagicMock(), model=model)

    def test_message(self):
        r = self._result()
        self.assertEqual(r.message, "Declustering completed successfully.")

    def test_target_name(self):
        r = self._result(target_name="Samples")
        self.assertEqual(r.target_name, "Samples")

    def test_target_reference(self):
        r = self._result()
        self.assertEqual(r.target_reference, TARGET_URL)

    def test_attribute_name(self):
        r = self._result(attr_name="decluster_wt")
        self.assertEqual(r.attribute_name, "decluster_wt")

    def test_display_name(self):
        self.assertEqual(DeclusteringResult.TASK_DISPLAY_NAME, "Declustering")

    def test_str_contains_key_info(self):
        r = self._result(target_name="Samples", attr_name="weights")
        s = str(r)
        self.assertIn("Declustering", s)
        self.assertIn("Samples", s)
        self.assertIn("weights", s)

    def test_schema_parses_valid_id(self):
        r = self._result()
        self.assertEqual(r.schema.sub_classification, "pointset")


# ---------------------------------------------------------------------------
# Runner end-to-end tests
# ---------------------------------------------------------------------------


def _mock_declustering_context():
    mock_connector = MagicMock()
    mock_context = MagicMock()
    mock_context.get_connector.return_value = mock_connector
    mock_context.get_org_id.return_value = "test-org-id"
    return mock_context, mock_connector


def _mock_declustering_job():
    mock_job = AsyncMock()
    mock_job.wait_for_results.return_value = _make_result_model()
    return mock_job


class TestDeclusteringRunnerExecution(IsolatedAsyncioTestCase):
    async def test_runner_submits_job_and_returns_result(self):
        mock_context, _ = _mock_declustering_context()
        params = DeclusteringParameters(
            source=POINTSET_URL,
            grid=GRID_URL,
            target=Target(object=TARGET_URL, attribute=CreateAttribute(name="weights")),
            neighborhood=_search(),
        )

        with patch(
            "evo.compute.tasks.common.runner.JobClient.submit",
            new_callable=AsyncMock,
            return_value=_mock_declustering_job(),
        ):
            result = await DeclusteringRunner(mock_context, params)

        self.assertIsInstance(result, DeclusteringResult)
        self.assertEqual(result.target_name, "MyPointset")

    async def test_runner_knn_mode_sends_null_power(self):
        mock_context, _ = _mock_declustering_context()
        params = DeclusteringParameters(
            source=POINTSET_URL,
            grid=GRID_URL,
            target=Target(object=TARGET_URL, attribute=CreateAttribute(name="weights")),
            neighborhood=_search(),
            power=None,
        )

        with patch(
            "evo.compute.tasks.common.runner.JobClient.submit",
            new_callable=AsyncMock,
            return_value=_mock_declustering_job(),
        ) as mock_submit:
            await DeclusteringRunner(mock_context, params)

        mock_submit.assert_called_once()
        _, kwargs = mock_submit.call_args
        submitted_params = kwargs.get("parameters", {})
        self.assertIn("power", submitted_params)
        self.assertIsNone(submitted_params["power"])
