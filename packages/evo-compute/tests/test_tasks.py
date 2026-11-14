#  Copyright © 2025 Bentley Systems, Incorporated
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Tests for the compute tasks module imports and basic functionality."""

import ast
import asyncio
import importlib.util
import inspect
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from evo.common.interfaces import IFeedback
from evo.common.test_tools import ORG, TestWithConnector
from evo.objects import ObjectSchema
from evo.objects.exceptions import SchemaIDFormatError
from pydantic import ValidationError

from evo.compute import JobClient, TaskJob, TaskResult, _stubgen
from evo.compute.tasks import SearchNeighborhood, run
from evo.compute.tasks.common import Ellipsoid, EllipsoidRanges, Filter, FilterCondition, Source, Target
from evo.compute.tasks.common.results import TaskAttribute, TaskResultList, TaskTarget
from evo.compute.tasks.common.runner import TaskRegistry, run_tasks
from evo.compute.tasks.common.source_target import _convert_object_reference
from evo.compute.tasks.geostatistics.idw import IDWParameters, IDWResult
from evo.compute.tasks.geostatistics.knn import KNNParameters, KNNResult
from evo.compute.tasks.geostatistics.kriging import (
    BlockDiscretization,
    KrigingDiagnostics,
    KrigingMethod,
    KrigingParameters,
    KrigingResult,
    KrigingResultModel,
    KrigingRunner,
    KrigingTargetResult,
    OrdinaryKriging,
    SimpleKriging,
)
from test_engine import FakeContext
from test_jobs import RecordingFeedback
from test_payload_parity import (
    POINTSET_URL,
    TARGET_URL,
    VARIOGRAM_URL,
    _pending_attribute,
    _typed_object,
    payload_differences,
)


class TestTaskRegistry(unittest.TestCase):
    """Tests for the task registry system."""

    def test_kriging_parameters_registered(self):
        """KrigingParameters should be registered with the task registry."""
        registry = TaskRegistry()
        runner_cls = registry.get_runner(KrigingParameters)
        self.assertIsNotNone(runner_cls)
        self.assertIs(runner_cls, KrigingRunner)

    def test_unregistered_type_returns_none(self):
        """Unregistered types should return None from get_runner."""
        registry = TaskRegistry()

        class UnregisteredParams:
            pass

        runner_cls = registry.get_runner(UnregisteredParams)
        self.assertIsNone(runner_cls)

    def test_registry_get_runner_for_params_raises_on_unknown(self):
        """get_runner_for_params should raise TypeError for unregistered types."""
        registry = TaskRegistry()

        class UnknownParams:
            pass

        with self.assertRaises(TypeError) as ctx:
            registry.get_runner_for_params(UnknownParams())

        self.assertIn("UnknownParams", str(ctx.exception))


class TestTaskRunnerSubclass(unittest.TestCase):
    """Tests for the TaskRunner __init_subclass__ mechanism."""

    def test_kriging_runner_has_correct_topic_and_task(self):
        """KrigingRunner should have topic='geostatistics' and task='kriging'."""
        self.assertEqual(KrigingRunner.topic, "geostatistics")
        self.assertEqual(KrigingRunner.task, "kriging")

    def test_kriging_runner_has_correct_types(self):
        """KrigingRunner should have correct params_type, result_model_type, and result_type."""
        self.assertIs(KrigingRunner.params_type, KrigingParameters)
        self.assertIs(KrigingRunner.result_model_type, KrigingResultModel)
        self.assertIs(KrigingRunner.result_type, KrigingResult)

    def test_runner_accepts_preview_kwarg(self):
        """TaskRunner.__init__ should accept a 'preview' keyword argument."""
        mock_context = MagicMock()
        mock_params = MagicMock(spec=KrigingParameters)
        runner = KrigingRunner(mock_context, mock_params, preview=True)
        self.assertTrue(runner._preview)

    def test_runner_preview_defaults_false(self):
        """TaskRunner.__init__ preview should default to False."""
        mock_context = MagicMock()
        mock_params = MagicMock(spec=KrigingParameters)
        runner = KrigingRunner(mock_context, mock_params)
        self.assertFalse(runner._preview)


class TestPreviewFlagSignatures(unittest.TestCase):
    """Tests for the preview flag signatures on run()."""

    def test_run_function_accepts_preview_kwarg(self):
        """The public run() function should accept a 'preview' keyword argument defaulting to False."""
        sig = inspect.signature(run)
        self.assertIn("preview", sig.parameters)
        self.assertEqual(sig.parameters["preview"].default, False)

    def test_run_tasks_accepts_preview_kwarg(self):
        """run_tasks() should accept a 'preview' keyword argument defaulting to False."""
        sig = inspect.signature(run_tasks)
        self.assertIn("preview", sig.parameters)
        self.assertEqual(sig.parameters["preview"].default, False)


def _mock_kriging_context():
    """Create a mock context + connector for kriging preview tests."""
    mock_connector = MagicMock()

    mock_context = MagicMock()
    mock_context.get_connector.return_value = mock_connector
    mock_context.get_org_id.return_value = "test-org-id"
    return mock_context, mock_connector


def _mock_kriging_job():
    """Create a mock job that returns a valid KrigingResult."""
    mock_job = AsyncMock()
    mock_job.wait_for_results.return_value = KrigingResultModel(
        message="ok",
        target=KrigingTargetResult(
            reference="ref",
            name="t",
            description=None,
            schema_id="s",
            attribute=TaskAttribute(reference="ar", name="an"),
        ),
    )
    return mock_job


def _mock_task_job(context):
    backend = _mock_kriging_job()
    backend.wait_for_results.return_value = backend.wait_for_results.return_value.model_dump(mode="json")

    async def hydrate(payload):
        return TaskResult(payload, None, context)

    return TaskJob(backend, hydrate)


class TestPreviewFlagBehavior(unittest.IsolatedAsyncioTestCase):
    """Tests for the preview flag runtime behavior on KrigingRunner."""

    async def test_runner_passes_preview_true_to_submit(self):
        """KrigingRunner should pass preview=True to JobClient.submit."""
        mock_context, mock_connector = _mock_kriging_context()
        mock_params = MagicMock(spec=KrigingParameters)
        mock_params.model_dump.return_value = {"source": {}, "target": {}}

        with patch(
            "evo.compute.engine.ComputeClient.asubmit",
            new_callable=AsyncMock,
            return_value=_mock_task_job(mock_context),
        ) as mock_submit:
            await KrigingRunner(mock_context, mock_params, preview=True)

        mock_submit.assert_awaited_once_with("geostatistics", "kriging", {"source": {}, "target": {}, "preview": True})

    async def test_runner_passes_preview_false_to_submit(self):
        """KrigingRunner should pass preview=False when preview=False."""
        mock_context, mock_connector = _mock_kriging_context()
        mock_params = MagicMock(spec=KrigingParameters)
        mock_params.model_dump.return_value = {"source": {}, "target": {}}

        with patch(
            "evo.compute.engine.ComputeClient.asubmit",
            new_callable=AsyncMock,
            return_value=_mock_task_job(mock_context),
        ) as mock_submit:
            await KrigingRunner(mock_context, mock_params, preview=False)

        mock_submit.assert_awaited_once_with("geostatistics", "kriging", {"source": {}, "target": {}, "preview": False})

    async def test_runner_default_preview_is_false(self):
        """KrigingRunner should default to preview=False when not specified."""
        mock_context, mock_connector = _mock_kriging_context()
        mock_params = MagicMock(spec=KrigingParameters)
        mock_params.model_dump.return_value = {"source": {}, "target": {}}

        with patch(
            "evo.compute.engine.ComputeClient.asubmit",
            new_callable=AsyncMock,
            return_value=_mock_task_job(mock_context),
        ) as mock_submit:
            await KrigingRunner(mock_context, mock_params)

        mock_submit.assert_awaited_once_with("geostatistics", "kriging", {"source": {}, "target": {}, "preview": False})


class TestTaskResultSchemaType(unittest.TestCase):
    """Tests for schema_type property using ObjectSchema parsing."""

    def _make_result(self, schema_id: str):
        attr = TaskAttribute(reference="ref", name="attr")
        target = KrigingTargetResult(
            reference="ref", name="target", description=None, schema_id=schema_id, attribute=attr
        )
        return KrigingResult(
            context=...,
            model=KrigingResultModel(message="ok", target=target),
        )

    def test_schema_type_parses_valid_schema_id(self):
        """schema_type should return the sub_classification for a valid schema ID."""
        result = self._make_result("/objects/regular-masked-3d-grid/1.0.0/regular-masked-3d-grid.schema.json")
        self.assertEqual(result.schema.sub_classification, "regular-masked-3d-grid")

    def test_schema_type_parses_different_schema(self):
        """schema_type should handle different object schema types."""
        result = self._make_result("/objects/block-model/2.1.0/block-model.schema.json")
        self.assertEqual(result.schema.sub_classification, "block-model")

    def test_schema_type_fails_for_malformed_id(self):
        """schema_type should return the raw schema_id when it cannot be parsed."""
        result = self._make_result("some-unparseable-string")
        with self.assertRaises(SchemaIDFormatError):
            result.schema.sub_classification

    def test_schema_type_fails_for_partial_id(self):
        """schema_type should fall back gracefully for partial schema paths."""
        result = self._make_result("schema/1.0.0")
        with self.assertRaises(SchemaIDFormatError):
            result.schema.sub_classification


class TestKrigingMethod(unittest.TestCase):
    """Tests for kriging method classes."""

    def test_ordinary_kriging_singleton(self):
        """KrigingMethod.ORDINARY should be an OrdinaryKriging instance."""
        self.assertIsInstance(KrigingMethod.ORDINARY, OrdinaryKriging)

    def test_simple_kriging_factory(self):
        """KrigingMethod.simple() should create a SimpleKriging instance."""
        method = KrigingMethod.simple(mean=100.0)
        self.assertIsInstance(method, SimpleKriging)
        self.assertEqual(method.mean, 100.0)

    def test_ordinary_kriging_model_dump(self):
        """OrdinaryKriging should serialize to dict with type='ordinary'."""
        d = OrdinaryKriging().model_dump()
        self.assertEqual(d, {"type": "ordinary"})

    def test_simple_kriging_model_dump(self):
        """SimpleKriging should serialize to dict with type='simple' and mean."""
        d = SimpleKriging(mean=50.0).model_dump()
        self.assertEqual(d, {"type": "simple", "mean": 50.0})


class TestTaskTargetModelValidate(unittest.TestCase):
    """Tests for TaskTarget.model_validate (replaces parse_task_target)."""

    def test_model_validate_from_dict(self):
        """TaskTarget.model_validate should parse a raw API response dict."""
        data = {
            "reference": "ref",
            "name": "target_name",
            "description": "desc",
            "schema_id": "/objects/block-model/2.1.0/block-model.schema.json",
            "attribute": {"reference": "attr_ref", "name": "attr_name"},
        }
        target = TaskTarget.model_validate(data)
        self.assertIsInstance(target, TaskTarget)
        self.assertEqual(target.reference, "ref")
        self.assertEqual(target.name, "target_name")
        self.assertEqual(target.attribute.reference, "attr_ref")
        self.assertEqual(target.attribute.name, "attr_name")


class TestTaskResultPydanticModel(unittest.TestCase):
    """Tests that TaskResult works as a Pydantic BaseModel."""

    def test_kriging_result_model_validate(self):
        """KrigingResult should be constructable via model_validate."""
        data = {
            "message": "ok",
            "target": {
                "reference": "ref",
                "name": "target",
                "schema_id": "s",
                "attribute": {"reference": "ar", "name": "an"},
            },
        }
        result = KrigingResultModel.model_validate(data)
        self.assertIsInstance(result, KrigingResultModel)


class TestConvertObjectReference(unittest.TestCase):
    """Tests for _convert_object_reference BeforeValidator."""

    def test_valid_string_accepted(self):
        """A valid ObjectReference URL string should be converted to a validated str."""
        url = "https://hub.test.evo.bentley.com/geoscience-object/orgs/00000000-0000-0000-0000-000000000001/workspaces/00000000-0000-0000-0000-000000000002/objects/00000000-0000-0000-0000-000000000003"
        result = _convert_object_reference(url)
        self.assertIsInstance(result, str)
        self.assertEqual(result, url)

    def test_raises_for_unsupported_type(self):
        """_convert_object_reference should raise TypeError for unsupported types."""
        with self.assertRaises(TypeError):
            _convert_object_reference(12345)

    def test_raises_for_invalid_url(self):
        """_convert_object_reference should raise ValueError for invalid URL strings."""
        with self.assertRaises(ValueError):
            _convert_object_reference("not_a_url")


class TestSearchNeighborhood(unittest.TestCase):
    """Tests for the octant, quadrant and drillhole limits on SearchNeighborhood."""

    LIMITS = {
        "max_empty_octants": 4,
        "max_samples_per_octant": 3,
        "max_empty_quadrants": 1,
        "max_samples_per_quadrant": 5,
        "max_samples_per_drillhole": 2,
        "max_drillholes_per_estimate": 6,
    }

    def _search(self, **limits) -> SearchNeighborhood:
        return SearchNeighborhood(
            ellipsoid=Ellipsoid(ranges=EllipsoidRanges(major=200, semi_major=150, minor=100)),
            max_samples=24,
            **limits,
        )

    def test_limits_are_omitted_unless_set(self):
        self.assertEqual(set(self._search().model_dump(mode="json")), {"ellipsoid", "max_samples"})

    def test_default_rotation_has_stable_wire_round_trips(self):
        original = self._search()
        restored = SearchNeighborhood.model_validate_json(original.model_dump_json())
        self.assertEqual(original.model_dump_json(), restored.model_dump_json())

    def test_limits_are_sent_when_set(self):
        dumped = self._search(**self.LIMITS).model_dump(mode="json")
        self.assertEqual({name: dumped[name] for name in self.LIMITS}, self.LIMITS)

    def test_zero_empty_sectors_is_sent(self):
        dumped = self._search(max_empty_octants=0, max_empty_quadrants=0).model_dump(mode="json")
        self.assertEqual(dumped["max_empty_octants"], 0)
        self.assertEqual(dumped["max_empty_quadrants"], 0)

    def test_limits_outside_the_service_bounds_are_refused(self):
        for name, value in [
            ("max_empty_octants", -1),
            ("max_empty_octants", 9),
            ("max_empty_quadrants", -1),
            ("max_empty_quadrants", 5),
            ("max_samples_per_octant", 0),
            ("max_samples_per_quadrant", 0),
            ("max_samples_per_drillhole", 0),
            ("max_drillholes_per_estimate", 0),
        ]:
            with self.subTest(name=name, value=value), self.assertRaises(ValidationError):
                self._search(**{name: value})


class TestRunTasksDispatch(unittest.IsolatedAsyncioTestCase):
    """Tests that run_tasks dispatches to concrete runner subclasses, not abstract TaskRunner."""

    async def test_run_tasks_uses_engine_and_keeps_native_result(self):
        mock_context, _ = _mock_kriging_context()
        mock_params = MagicMock(spec=KrigingParameters)
        payload = {"source": {}, "target": {}}
        mock_params.model_dump.return_value = payload
        feedback = MagicMock(spec=IFeedback)
        raw_result = _mock_kriging_job().wait_for_results.return_value.model_dump()
        job = MagicMock(spec=TaskJob)
        job.results.return_value = TaskResult(raw_result, None, mock_context)

        with (
            patch("evo.compute.tasks.common.runner._registry.get_runner_for_params", return_value=KrigingRunner),
            patch("evo.compute.engine.ComputeClient", autospec=True) as client_type,
        ):
            client_type.return_value.asubmit.return_value = job
            results = await run_tasks(mock_context, [mock_params], preview=True, fb=feedback)

        client_type.assert_called_once_with(mock_context)
        client_type.return_value.asubmit.assert_awaited_once_with(
            "geostatistics", "kriging", {"source": {}, "target": {}, "preview": True}
        )
        job.results.assert_awaited_once()
        self.assertIsInstance(results[0], KrigingResult)
        self.assertEqual("t", results[0].target_name)
        self.assertEqual("an", results[0].attribute_name)
        self.assertIs(mock_context, results[0]._context)
        self.assertEqual({"source": {}, "target": {}}, payload)
        job.results.await_args.kwargs["fb"].progress(1.0, "remote job message")
        feedback.progress.assert_called_once_with(1.0, "Completed 1/1")

    async def test_unknown_parameter_type_is_rejected_before_any_submission(self):
        mock_context, _ = _mock_kriging_context()
        known = KrigingParameters.model_construct()
        unknown = object()
        with patch("evo.compute.engine.ComputeClient", autospec=True) as client_type:
            with self.assertRaisesRegex(TypeError, "object"):
                await run_tasks(mock_context, [known, unknown])
        client_type.assert_not_called()

    async def test_backend_failure_is_not_hidden(self):
        mock_context, _ = _mock_kriging_context()
        mock_params = MagicMock(spec=KrigingParameters)
        mock_params.model_dump.return_value = {"source": {}, "target": {}}
        with (
            patch("evo.compute.engine.ComputeClient", autospec=True) as client_type,
            patch("evo.compute.tasks.common.runner._registry.get_runner_for_params", return_value=KrigingRunner),
        ):
            client_type.return_value.asubmit.side_effect = RuntimeError("submission failed")
            with self.assertRaisesRegex(RuntimeError, "submission failed"):
                await run_tasks(mock_context, [mock_params])

    async def test_direct_runner_preserves_feedback_polling_and_retry(self):
        context, _ = _mock_kriging_context()
        params = MagicMock(spec=KrigingParameters)
        params.model_dump.return_value = {"source": {}, "target": {}}
        feedback, retry = MagicMock(), MagicMock()
        job = _mock_task_job(context)
        with patch("evo.compute.engine.ComputeClient.asubmit", AsyncMock(return_value=job)):
            result = await KrigingRunner(context, params, polling_interval_seconds=0.25, retry=retry, fb=feedback)
        self.assertIsInstance(result, KrigingResult)
        job._job.wait_for_results.assert_awaited_once_with(polling_interval_seconds=0.25, retry=retry, fb=feedback)

    async def test_run_tasks_dispatches_to_registered_runner(self):
        """run_tasks should use the registered runner subclass, not TaskRunner directly."""
        mock_context, _ = _mock_kriging_context()
        mock_params = MagicMock(spec=KrigingParameters)
        mock_params.model_dump.return_value = {"source": {}, "target": {}}

        with (
            patch(
                "evo.compute.tasks.common.runner._registry.get_runner_for_params",
                return_value=KrigingRunner,
            ),
            patch("evo.compute.engine.ComputeClient", autospec=True) as client_type,
        ):
            client_type.return_value.asubmit.return_value = _mock_task_job(mock_context)
            results = await run_tasks(mock_context, [mock_params], preview=True)

        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], KrigingResult)

    async def test_run_tasks_multiple_parameters(self):
        """run_tasks should handle multiple parameters and return results in order."""
        mock_context, _ = _mock_kriging_context()
        params_list = []
        for _ in range(3):
            p = MagicMock(spec=KrigingParameters)
            p.model_dump.return_value = {"source": {}, "target": {}}
            params_list.append(p)

        with (
            patch(
                "evo.compute.tasks.common.runner._registry.get_runner_for_params",
                return_value=KrigingRunner,
            ),
            patch("evo.compute.engine.ComputeClient", autospec=True) as client_type,
        ):
            client_type.return_value.asubmit.return_value = _mock_task_job(mock_context)
            results = await run_tasks(mock_context, params_list, preview=True)

        self.assertEqual(len(results), 3)
        for result in results:
            self.assertIsInstance(result, KrigingResult)

    async def test_run_tasks_empty_list_returns_empty(self):
        """run_tasks should return an empty list for empty parameters."""
        mock_context, _ = _mock_kriging_context()
        results = await run_tasks(mock_context, [])
        self.assertEqual(results, [])

    async def test_run_tasks_raises_for_unregistered_type(self):
        """run_tasks should raise TypeError for unregistered parameter types."""
        mock_context, _ = _mock_kriging_context()

        class UnknownParams:
            pass

        with self.assertRaises(TypeError):
            await run_tasks(mock_context, [UnknownParams()])


class TestRunReturnsTaskResultList(unittest.IsolatedAsyncioTestCase):
    """Tests that the public run() function returns TaskResultList for multi-param calls."""

    async def test_default_feedback_and_preview_are_preserved(self):
        mock_context, _ = _mock_kriging_context()
        mock_params = MagicMock(spec=KrigingParameters)
        mock_params.model_dump.return_value = {"source": {}, "target": {}}
        feedback = MagicMock(spec=IFeedback)
        job = _mock_task_job(mock_context)
        with (
            patch("evo.compute.engine.ComputeClient", autospec=True) as client_type,
            patch("evo.compute.tasks.create_default_feedback", return_value=feedback) as create_feedback,
            patch("evo.compute.tasks.common.runner._registry.get_runner_for_params", return_value=KrigingRunner),
        ):
            client_type.return_value.asubmit.return_value = job
            result = await run(mock_context, mock_params)
        create_feedback.assert_called_once_with("Tasks")
        client_type.return_value.asubmit.assert_awaited_once_with(
            "geostatistics", "kriging", {"source": {}, "target": {}, "preview": False}
        )
        job._job.wait_for_results.await_args.kwargs["fb"].progress(1.0)
        feedback.progress.assert_called_once_with(1.0, "Completed 1/1")
        self.assertIsInstance(result, KrigingResult)

    async def test_run_single_returns_bare_result(self):
        """run() with a single param should return a bare result, not TaskResultList."""
        mock_context, _ = _mock_kriging_context()
        mock_params = MagicMock(spec=KrigingParameters)
        mock_params.model_dump.return_value = {"source": {}, "target": {}}

        with (
            patch(
                "evo.compute.tasks.common.runner._registry.get_runner_for_params",
                return_value=KrigingRunner,
            ),
            patch("evo.compute.engine.ComputeClient", autospec=True) as client_type,
        ):
            client_type.return_value.asubmit.return_value = _mock_task_job(mock_context)
            result = await run(mock_context, mock_params, preview=True)

        self.assertIsInstance(result, KrigingResult)
        self.assertNotIsInstance(result, TaskResultList)

    async def test_run_list_returns_task_result_list(self):
        """run() with a list of params should return a TaskResultList."""
        mock_context, _ = _mock_kriging_context()
        params_list = []
        for _ in range(3):
            p = MagicMock(spec=KrigingParameters)
            p.model_dump.return_value = {"source": {}, "target": {}}
            params_list.append(p)

        with (
            patch(
                "evo.compute.tasks.common.runner._registry.get_runner_for_params",
                return_value=KrigingRunner,
            ),
            patch("evo.compute.engine.ComputeClient", autospec=True) as client_type,
        ):
            client_type.return_value.asubmit.return_value = _mock_task_job(mock_context)
            results = await run(mock_context, params_list, preview=True)

        self.assertIsInstance(results, TaskResultList)
        self.assertEqual(len(results), 3)
        for r in results:
            self.assertIsInstance(r, KrigingResult)

    async def test_run_empty_list_returns_task_result_list(self):
        """run() with an empty list should return an empty TaskResultList."""
        mock_context, _ = _mock_kriging_context()
        results = await run(mock_context, [], preview=True)
        self.assertIsInstance(results, TaskResultList)
        self.assertEqual(len(results), 0)


class TestNativeFacadeEngineParity(TestWithConnector):
    def setUp(self):
        super().setUp()
        self.context = FakeContext(self.connector, ORG.id)
        specs = [
            spec
            for spec in _stubgen.load_snapshot(_stubgen.DEFAULT_SNAPSHOT_DIR)
            if spec.topic == "geostatistics" and spec.name in {"kriging", "idw", "knn"}
        ]
        self.catalogue = {
            "results": [spec.model_dump(mode="json", by_alias=True) for spec in specs],
            "total": len(specs),
        }
        self.schemas = {
            POINTSET_URL: ObjectSchema.from_id("/objects/pointset/1.2.0/pointset.schema.json"),
            TARGET_URL: ObjectSchema.from_id("/objects/block-model/1.0.0/block-model.schema.json"),
            VARIOGRAM_URL: ObjectSchema.from_id("/objects/variogram/1.1.0/variogram.schema.json"),
        }
        self.feedback = RecordingFeedback()

    def _parameters(self):
        search = SearchNeighborhood(
            ellipsoid=Ellipsoid(ranges=EllipsoidRanges(major=150, semi_major=100, minor=75)),
            max_samples=30,
            min_samples=1,
        )
        common = {
            "source": Source(object=POINTSET_URL, attribute="locations.attributes[?name=='Cu_pct']"),
            "source_filter": Filter(
                where=FilterCondition(
                    attribute="locations.attributes[?name=='Geological_model']", operator="in", values=["Early Diorite"]
                )
            ),
            "target_filter": Filter(
                where=FilterCondition(
                    attribute="attributes[?name=='Geological model']", operator="in", values=["Early Diorite"]
                )
            ),
            "neighborhood": search,
        }
        return [
            KrigingParameters(
                **common,
                target=Target.new_attribute(TARGET_URL, "Cu_OK"),
                variogram=VARIOGRAM_URL,
                block_discretization=BlockDiscretization(nx=5, ny=5, nz=2),
                diagnostics=KrigingDiagnostics(kriging_variance=_pending_attribute("KV", TARGET_URL)),
            ),
            IDWParameters(**common, target=Target.new_attribute(TARGET_URL, "Cu_IDW"), power=2),
            KNNParameters(**common, target=Target.new_attribute(TARGET_URL, "Cu_KNN")),
        ]

    def _object(self, context, reference):
        downloaded = MagicMock()
        downloaded.metadata.schema_id = self.schemas[reference]
        return downloaded

    def _wire_parameters(self, parameters):
        expected = {
            name: params.model_dump(mode="json", by_alias=True, exclude_none=True)
            for name, params in zip(("kriging", "idw", "knn"), parameters, strict=True)
        }
        expected["kriging"]["block_discretisation"] = expected["kriging"].pop("block_discretization")
        return expected

    async def test_direct_runner_and_repeated_batches_share_execution_and_discovery(self):
        params = self._parameters()[0]
        payloads = []
        raw_result = _mock_kriging_job().wait_for_results.return_value.model_dump(mode="json")

        async def submit(**kwargs):
            payloads.append(kwargs["parameters"])
            backend = MagicMock(spec=JobClient)
            backend.wait_for_results.return_value = raw_result
            return backend

        with (
            self.transport.set_http_response(status_code=200, content=json.dumps(self.catalogue)),
            patch("evo.compute.resolution.DownloadedObject.from_context", side_effect=self._object),
            patch("evo.compute.engine.JobClient.submit", side_effect=submit),
            patch.object(KrigingRunner, "_get_result", side_effect=AssertionError("result converted twice")),
        ):
            direct = await KrigingRunner(self.context, params)
            first = await run_tasks(self.context, [params])
            second = await run_tasks(self.context, [params])
        self.assertEqual([KrigingResult] * 3, [type(direct), type(first[0]), type(second[0])])
        self.assertEqual(3, len(payloads))
        self.assertEqual(payloads[0], payloads[1])
        self.assertEqual(payloads[0], payloads[2])
        self.assertEqual({"nx": 5, "ny": 5, "nz": 2}, payloads[0]["block_discretisation"])
        self.assertNotIn("block_discretization", payloads[0])
        self.assertEqual(1, self.transport.request.call_count)

    async def test_mixed_notebook_batch_preserves_payloads_results_and_feedback(self):
        parameters = self._parameters()
        expected = self._wire_parameters(parameters)
        started, completed = [], []
        ready = asyncio.Event()

        async def submit(*, topic, task, parameters, preview, result_type, **kwargs):
            self.assertEqual("geostatistics", topic)
            self.assertTrue(preview)
            self.assertIs(dict, result_type)
            self.assertEqual([], payload_differences(expected[task], parameters))
            backend = MagicMock(spec=JobClient)

            async def wait_for_results(*, fb, polling_interval_seconds, retry):
                self.assertEqual(5.0, polling_interval_seconds)
                self.assertIsNone(retry)
                fb.progress(0.5, "private job message")
                started.append(task)
                if len(started) == 3:
                    ready.set()
                await asyncio.wait_for(ready.wait(), timeout=5)
                completed.append(task)
                fb.progress(1.0, "private job message")
                return {
                    "message": task,
                    "target": {
                        "reference": TARGET_URL,
                        "name": "WP Regular BlockModel",
                        "schema_id": str(self.schemas[TARGET_URL]),
                        "attribute": {"reference": "attributes", "name": parameters["target"]["attribute"]["name"]},
                        "diagnostics": {"kriging_variance": {"reference": "attributes", "name": "KV"}},
                    },
                }

            backend.wait_for_results.side_effect = wait_for_results
            return backend

        with (
            self.transport.set_http_response(status_code=200, content=json.dumps(self.catalogue)),
            patch("evo.compute.resolution.DownloadedObject.from_context", side_effect=self._object),
            patch("evo.compute.engine.JobClient.submit", side_effect=submit) as submitted,
        ):
            results = await run(self.context, parameters, preview=True, fb=self.feedback)

        self.assertEqual(3, submitted.await_count)
        self.assertEqual(1, self.transport.request.call_count)
        self.assertEqual("knn", completed[0])
        self.assertIsInstance(results, TaskResultList)
        self.assertEqual([KrigingResult, IDWResult, KNNResult], [type(result) for result in results])
        self.assertEqual(["kriging", "idw", "knn"], [result.message for result in results])
        self.assertEqual(["Cu_OK", "Cu_IDW", "Cu_KNN"], [result.attribute_name for result in results])
        self.assertEqual("KV", results[0].diagnostics["kriging_variance"].name)
        self.assertTrue(all(result._context is self.context for result in results))
        self.assertIn("WP Regular BlockModel", str(results))
        self.assertNotIn("?", str(results))
        self.assertEqual((1.0, "Completed 3/3"), self.feedback.reports[-1])
        self.assertFalse(any(message == "private job message" for progress, message in self.feedback.reports))

    @unittest.skipUnless(
        all(importlib.util.find_spec(name) is not None for name in ("IPython", "ipywidgets", "plotly")),
        "notebook extras are required",
    )
    async def test_estimation_engine_notebook_runs_both_workflows(self):
        import pandas as pd
        from IPython.core.interactiveshell import InteractiveShell

        path = Path(__file__).resolve().parents[1] / "docs" / "examples" / "estimation_engine.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(45, len(notebook["cells"]))
        self.assertEqual(45, len({cell["id"] for cell in notebook["cells"]}))
        native_parameters = self._parameters()
        expected = self._wire_parameters(native_parameters)
        for parameters in expected.values():
            ranges = parameters["neighborhood"]["ellipsoid"]["ellipsoid_ranges"]
            ranges.update({name: float(value) for name, value in ranges.items()})
        source = SimpleNamespace(
            attributes={
                "Cu_pct": native_parameters[0].source,
                "Geological_model": "locations.attributes[?name=='Geological_model']",
            },
            to_dataframe=AsyncMock(
                return_value=pd.DataFrame(
                    {"Cu_pct": [0.1, 0.2, 0.3], "Geological_model": ["Early Diorite"] * 3, "IDW declus": [1, 2, 1]}
                )
            ),
        )
        source.refresh = AsyncMock(return_value=source)
        frame = pd.DataFrame(
            {
                "Cu_KNN": [0.1, 0.2, 0.3],
                "Cu_IDW": [0.3, 0.5, 0.7],
                "Cu_OK": [0.2, 0.4, 0.6],
                "KV": [0.01, 0.02, 0.03],
                "Geological model": ["Early Diorite"] * 3,
            }
        )
        target = SimpleNamespace(
            attributes={name: _pending_attribute(name, TARGET_URL) for name in ("Cu_OK", "Cu_IDW", "Cu_KNN")},
            to_dataframe=AsyncMock(side_effect=lambda *, columns: frame[columns].copy()),
        )
        target.attributes.update(
            {"KV": _pending_attribute("KV", TARGET_URL), "Geological model": "attributes[?name=='Geological model']"}
        )
        target.refresh = AsyncMock(return_value=target)
        variogram = _typed_object(VARIOGRAM_URL)
        variogram.get_ellipsoid = MagicMock(
            return_value=Ellipsoid(ranges=EllipsoidRanges(major=100, semi_major=200 / 3, minor=50))
        )
        inputs = {
            "WP comps.json": source,
            "Cu_pct Variogram Model.json": variogram,
            "WP Regular BlockModel.json": target,
        }
        submitted, started, barriers = [], {}, {}

        async def submit(*, topic, task, parameters, preview, result_type, **kwargs):
            self.assertEqual("geostatistics", topic)
            self.assertTrue(preview)
            self.assertIs(dict, result_type)
            self.assertEqual([], payload_differences(expected[task], parameters))
            batch = len(submitted) // 3
            submitted.append((batch, task))
            backend = MagicMock(spec=JobClient)
            backend.topic, backend.task = topic, task
            backend.id, backend.url = f"{batch}-{task}", f"https://jobs.test/{batch}/{task}"

            async def wait_for_results(**kwargs):
                started.setdefault(batch, []).append(task)
                event = barriers.setdefault(batch, asyncio.Event())
                if len(started[batch]) == 3:
                    event.set()
                await asyncio.wait_for(event.wait(), timeout=5)
                return {
                    "message": task,
                    "target": {
                        "reference": TARGET_URL,
                        "name": "WP Regular BlockModel",
                        "schema_id": str(self.schemas[TARGET_URL]),
                        "attribute": {"reference": "attributes", "name": parameters["target"]["attribute"]["name"]},
                        "diagnostics": {"kriging_variance": {"reference": "attributes", "name": "KV"}},
                    },
                }

            backend.wait_for_results.side_effect = wait_for_results
            return backend

        shell = InteractiveShell.instance()
        namespace = {"get_ipython": lambda: shell}
        with (
            self.transport.set_http_response(status_code=200, content=json.dumps(self.catalogue)),
            patch("evo.notebooks.ServiceManagerWidget.with_auth_code") as authentication,
            patch("evo.objects.typed.object_from_path", AsyncMock(side_effect=lambda context, path: inputs[path])),
            patch("evo.compute.resolution.DownloadedObject.from_context", side_effect=self._object),
            patch(
                "evo.compute.tasks.geostatistics.kriging.object_from_reference", AsyncMock(return_value=target)
            ) as load_target,
            patch("evo.compute.engine.JobClient.submit", side_effect=submit),
            patch("evo.compute.tasks.create_default_feedback", return_value=self.feedback),
            patch.object(self.connector, "close", AsyncMock()) as close,
            patch.object(shell.extension_manager, "load_extension"),
            patch("plotly.graph_objects.Figure.show") as show,
        ):
            authentication.return_value.login = AsyncMock(return_value=self.context)
            for number, cell in enumerate(notebook["cells"], start=1):
                language = "python" if cell["cell_type"] == "code" else "markdown"
                self.assertEqual(language, cell["metadata"]["language"])
                self.assertEqual(cell["id"], cell["metadata"]["id"])
                if cell["cell_type"] != "code":
                    continue
                self.assertIsNone(cell["execution_count"])
                self.assertEqual([], cell["outputs"])
                code = compile(
                    shell.input_transformer_manager.transform_cell("".join(cell["source"])),
                    f"Cell {number}",
                    "exec",
                    ast.PyCF_ALLOW_TOP_LEVEL_AWAIT,
                )
                if code.co_flags & inspect.CO_COROUTINE:
                    await eval(code, namespace)
                else:
                    exec(code, namespace)
            self.assertEqual(6, len(submitted))
            self.assertEqual({0, 1}, set(started))
            self.assertTrue(all(len(tasks) == 3 for tasks in started.values()))
            self.assertEqual([KrigingResult, IDWResult, KNNResult], [type(result) for result in namespace["results"]])
            self.assertIsInstance(namespace["engine_results"][0], KrigingResult)
            self.assertTrue(all(isinstance(result, TaskResult) for result in namespace["engine_results"][1:]))
            self.assertTrue(all(isinstance(job, TaskJob) for job in namespace["jobs"]))
            self.assertTrue(namespace["comparison_df"]["Values match"].all())
            self.assertEqual(4, len(namespace["fig"].data))
            self.assertAlmostEqual(0.2, namespace["summary_df"].loc["mean", "Declustered Composites"])
            load_target.assert_awaited_once()
            self.assertEqual(2, target.to_dataframe.await_count)
            close.assert_awaited()
            show.assert_called_once()
            namespace["engine_df"].loc[0, "Cu_OK"] += 1
            exec("".join(notebook["cells"][36]["source"]), namespace)
            self.assertFalse(namespace["comparison_df"].loc["Cu_OK", "Values match"])
            namespace["engine_df"]["Cu_OK"] = float("nan")
            namespace["block_model_df"]["Cu_OK"] = float("nan")
            exec("".join(notebook["cells"][36]["source"]), namespace)
            self.assertFalse(namespace["comparison_df"].loc["Cu_OK", "Values match"])
            namespace["engine_df"].index = [2, 1, 0]
            exec("".join(notebook["cells"][36]["source"]), namespace)
            self.assertFalse(namespace["comparison_df"].loc["Cu_IDW", "Values match"])


class TestTaskResultList(unittest.TestCase):
    """Tests for TaskResultList list-like container."""

    def _make_mock_result(self, target_name="Grid", attribute_name="attr", message="ok"):
        r = MagicMock()
        r.TASK_DISPLAY_NAME = "Kriging"
        r.target_name = target_name
        r.attribute_name = attribute_name
        r.message = message
        return r

    def test_len(self):
        results = TaskResultList([self._make_mock_result() for _ in range(3)])
        self.assertEqual(len(results), 3)

    def test_len_empty(self):
        results = TaskResultList([])
        self.assertEqual(len(results), 0)

    def test_getitem_int(self):
        items = [self._make_mock_result(target_name=f"G{i}") for i in range(3)]
        results = TaskResultList(items)
        self.assertIs(results[0], items[0])
        self.assertIs(results[2], items[2])
        self.assertIs(results[-1], items[-1])

    def test_getitem_slice(self):
        items = [self._make_mock_result(target_name=f"G{i}") for i in range(5)]
        results = TaskResultList(items)
        sliced = results[1:3]
        self.assertEqual(len(sliced), 2)
        self.assertIs(sliced[0], items[1])

    def test_iter(self):
        items = [self._make_mock_result() for _ in range(3)]
        results = TaskResultList(items)
        self.assertEqual(list(results), items)

    def test_bool_true(self):
        results = TaskResultList([self._make_mock_result()])
        self.assertTrue(results)

    def test_bool_false(self):
        results = TaskResultList([])
        self.assertFalse(results)

    def test_repr_with_results(self):
        results = TaskResultList([self._make_mock_result() for _ in range(2)])
        self.assertEqual(repr(results), "TaskResultList([2 Kriging result(s)])")

    def test_repr_empty(self):
        results = TaskResultList([])
        self.assertEqual(repr(results), "TaskResultList([])")

    def test_str_with_results(self):
        items = [
            self._make_mock_result(target_name="Grid A", attribute_name="cu_est"),
            self._make_mock_result(target_name="Grid B", attribute_name="au_est"),
        ]
        results = TaskResultList(items)
        s = str(results)
        self.assertIn("2 Kriging Result(s)", s)
        self.assertIn("Grid A", s)
        self.assertIn("cu_est", s)
        self.assertIn("Grid B", s)
        self.assertIn("au_est", s)

    def test_str_empty(self):
        results = TaskResultList([])
        self.assertEqual(str(results), "No results")


if __name__ == "__main__":
    unittest.main()
