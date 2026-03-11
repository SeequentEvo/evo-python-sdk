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

import inspect
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from evo.objects.exceptions import SchemaIDFormatError

from evo.compute.tasks import run
from evo.compute.tasks.common.results import TaskAttribute, TaskResultList, TaskTarget
from evo.compute.tasks.common.runner import TaskRegistry, run_tasks
from evo.compute.tasks.common.source_target import _convert_object_reference
from evo.compute.tasks.kriging import (
    KrigingMethod,
    KrigingParameters,
    KrigingResult,
    KrigingResultModel,
    KrigingRunner,
    OrdinaryKriging,
    SimpleKriging,
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
        target=TaskTarget(
            reference="ref",
            name="t",
            description=None,
            schema_id="s",
            attribute=TaskAttribute(reference="ar", name="an"),
        ),
    )
    return mock_job


class TestPreviewFlagBehavior(unittest.IsolatedAsyncioTestCase):
    """Tests for the preview flag runtime behavior on KrigingRunner."""

    async def test_runner_passes_preview_true_to_submit(self):
        """KrigingRunner should pass preview=True to JobClient.submit."""
        mock_context, mock_connector = _mock_kriging_context()
        mock_params = MagicMock(spec=KrigingParameters)
        mock_params.model_dump.return_value = {"source": {}, "target": {}}

        with patch(
            "evo.compute.tasks.common.runner.JobClient.submit", new_callable=AsyncMock, return_value=_mock_kriging_job()
        ) as mock_submit:
            await KrigingRunner(mock_context, mock_params, preview=True)

        mock_submit.assert_called_once()
        _, kwargs = mock_submit.call_args
        self.assertTrue(kwargs.get("preview", False))

    async def test_runner_passes_preview_false_to_submit(self):
        """KrigingRunner should pass preview=False when preview=False."""
        mock_context, mock_connector = _mock_kriging_context()
        mock_params = MagicMock(spec=KrigingParameters)
        mock_params.model_dump.return_value = {"source": {}, "target": {}}

        with patch(
            "evo.compute.tasks.common.runner.JobClient.submit", new_callable=AsyncMock, return_value=_mock_kriging_job()
        ) as mock_submit:
            await KrigingRunner(mock_context, mock_params, preview=False)

        mock_submit.assert_called_once()
        _, kwargs = mock_submit.call_args
        self.assertFalse(kwargs.get("preview", True))

    async def test_runner_default_preview_is_false(self):
        """KrigingRunner should default to preview=False when not specified."""
        mock_context, mock_connector = _mock_kriging_context()
        mock_params = MagicMock(spec=KrigingParameters)
        mock_params.model_dump.return_value = {"source": {}, "target": {}}

        with patch(
            "evo.compute.tasks.common.runner.JobClient.submit", new_callable=AsyncMock, return_value=_mock_kriging_job()
        ) as mock_submit:
            await KrigingRunner(mock_context, mock_params)

        mock_submit.assert_called_once()
        _, kwargs = mock_submit.call_args
        self.assertFalse(kwargs.get("preview", True))


class TestTaskResultSchemaType(unittest.TestCase):
    """Tests for schema_type property using ObjectSchema parsing."""

    def _make_result(self, schema_id: str):
        attr = TaskAttribute(reference="ref", name="attr")
        target = TaskTarget(reference="ref", name="target", description=None, schema_id=schema_id, attribute=attr)
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


class TestRunTasksDispatch(unittest.IsolatedAsyncioTestCase):
    """Tests that run_tasks dispatches to concrete runner subclasses, not abstract TaskRunner."""

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
            patch(
                "evo.compute.tasks.common.runner.JobClient.submit",
                new_callable=AsyncMock,
                return_value=_mock_kriging_job(),
            ),
        ):
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
            patch(
                "evo.compute.tasks.common.runner.JobClient.submit",
                new_callable=AsyncMock,
                return_value=_mock_kriging_job(),
            ),
        ):
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
            patch(
                "evo.compute.tasks.common.runner.JobClient.submit",
                new_callable=AsyncMock,
                return_value=_mock_kriging_job(),
            ),
        ):
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
            patch(
                "evo.compute.tasks.common.runner.JobClient.submit",
                new_callable=AsyncMock,
                return_value=_mock_kriging_job(),
            ),
        ):
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
