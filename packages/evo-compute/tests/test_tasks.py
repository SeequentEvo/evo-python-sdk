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

from evo.compute.tasks.common.runner import get_task_runner, run_tasks
from evo.compute.tasks.kriging import KrigingParameters, _run_single_kriging


class TestTaskRegistry(unittest.TestCase):
    """Tests for the task registry system."""

    def test_kriging_parameters_registered(self):
        """KrigingParameters should be registered with the task registry."""
        from evo.compute.tasks.common.runner import get_task_runner
        from evo.compute.tasks.kriging import KrigingParameters

        runner = get_task_runner(KrigingParameters)
        self.assertIsNotNone(runner)

    def test_unregistered_type_returns_none(self):
        """Unregistered types should return None from get_task_runner."""
        from evo.compute.tasks.common.runner import get_task_runner

        class UnregisteredParams:
            pass

        runner = get_task_runner(UnregisteredParams)
        self.assertIsNone(runner)

    def test_registry_get_runner_for_params_raises_on_unknown(self):
        """get_runner_for_params should raise TypeError for unregistered types."""
        from evo.compute.tasks.common.runner import TaskRegistry

        registry = TaskRegistry()

        class UnknownParams:
            pass

        with self.assertRaises(TypeError) as ctx:
            registry.get_runner_for_params(UnknownParams())

        self.assertIn("UnknownParams", str(ctx.exception))


class TestPreviewFlagSignatures(unittest.TestCase):
    """Tests for the preview flag signatures on run() and runner functions."""

    def test_registered_runner_accepts_preview_kwarg(self):
        """The registered kriging runner should accept a 'preview' keyword argument."""

        runner = get_task_runner(KrigingParameters)
        sig = inspect.signature(runner)
        self.assertIn("preview", sig.parameters)
        param = sig.parameters["preview"]
        self.assertEqual(param.default, False)
        self.assertEqual(param.kind, inspect.Parameter.KEYWORD_ONLY)

    def test_run_single_kriging_accepts_preview_kwarg(self):
        """_run_single_kriging should accept a 'preview' keyword argument defaulting to False."""
        sig = inspect.signature(_run_single_kriging)
        self.assertIn("preview", sig.parameters)
        self.assertEqual(sig.parameters["preview"].default, False)

    def test_run_function_accepts_preview_kwarg(self):
        """The public run() function should accept a 'preview' keyword argument defaulting to False."""
        from evo.compute.tasks import run

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
    """Create a mock job that returns a valid kriging result."""
    mock_job = AsyncMock()
    mock_job.wait_for_results.return_value = {
        "message": "ok",
        "target": {
            "reference": "ref",
            "name": "t",
            "description": None,
            "schema_id": "s",
            "attribute": {"reference": "ar", "name": "an"},
        },
    }
    return mock_job


class TestPreviewFlagBehavior(unittest.IsolatedAsyncioTestCase):
    """Tests for the preview flag runtime behavior on _run_single_kriging."""

    async def test_run_single_kriging_passes_preview_true_to_submit(self):
        """_run_single_kriging should pass preview=True to JobClient.submit."""
        mock_context, mock_connector = _mock_kriging_context()
        mock_params = MagicMock(spec=KrigingParameters)
        mock_params.to_dict.return_value = {"source": {}, "target": {}}

        with patch(
            "evo.compute.tasks.kriging.JobClient.submit", new_callable=AsyncMock, return_value=_mock_kriging_job()
        ) as mock_submit:
            await _run_single_kriging(mock_context, mock_params, preview=True)

        # Verify preview=True was passed to JobClient.submit
        mock_submit.assert_called_once()
        _, kwargs = mock_submit.call_args
        self.assertTrue(kwargs.get("preview", False))

    async def test_run_single_kriging_passes_preview_false_to_submit(self):
        """_run_single_kriging should pass preview=False to JobClient.submit when preview=False."""
        mock_context, mock_connector = _mock_kriging_context()
        mock_params = MagicMock(spec=KrigingParameters)
        mock_params.to_dict.return_value = {"source": {}, "target": {}}

        with patch(
            "evo.compute.tasks.kriging.JobClient.submit", new_callable=AsyncMock, return_value=_mock_kriging_job()
        ) as mock_submit:
            await _run_single_kriging(mock_context, mock_params, preview=False)

        # Verify preview=False was passed to JobClient.submit
        mock_submit.assert_called_once()
        _, kwargs = mock_submit.call_args
        self.assertFalse(kwargs.get("preview", True))

    async def test_run_single_kriging_default_preview_is_false(self):
        """_run_single_kriging should default to preview=False when not specified."""
        mock_context, mock_connector = _mock_kriging_context()
        mock_params = MagicMock(spec=KrigingParameters)
        mock_params.to_dict.return_value = {"source": {}, "target": {}}

        with patch(
            "evo.compute.tasks.kriging.JobClient.submit", new_callable=AsyncMock, return_value=_mock_kriging_job()
        ) as mock_submit:
            # Call without preview kwarg — should default to False
            await _run_single_kriging(mock_context, mock_params)

        # Verify preview=False was passed to JobClient.submit
        mock_submit.assert_called_once()
        _, kwargs = mock_submit.call_args
        self.assertFalse(kwargs.get("preview", True))


class TestKrigingResultInheritance(unittest.TestCase):
    """Tests that KrigingResult inherits from TaskResult."""

    def test_kriging_result_inherits_from_task_result(self):
        """KrigingResult should be a subclass of TaskResult."""
        from evo.compute.tasks import KrigingResult, TaskResult

        self.assertTrue(issubclass(KrigingResult, TaskResult))


class TestTaskResultsContainer(unittest.TestCase):
    """Tests for the TaskResults container class."""

    def test_task_results_iteration(self):
        """TaskResults should support iteration."""
        from evo.compute.tasks.kriging import KrigingResult, TaskResults, _KrigingAttribute, _KrigingTarget

        # Create mock results
        attr = _KrigingAttribute(reference="ref1", name="attr1")
        target = _KrigingTarget(
            reference="ref1",
            name="target1",
            description="desc",
            schema_id="schema/1.0.0",
            attribute=attr,
        )
        result1 = KrigingResult(message="msg1", target=target)
        result2 = KrigingResult(message="msg2", target=target)

        results = TaskResults([result1, result2])

        # Test len
        self.assertEqual(len(results), 2)

        # Test iteration
        items = list(results)
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0].message, "msg1")
        self.assertEqual(items[1].message, "msg2")

        # Test indexing
        self.assertEqual(results[0].message, "msg1")
        self.assertEqual(results[1].message, "msg2")

    def test_task_results_results_property(self):
        """TaskResults should expose results via .results property."""
        from evo.compute.tasks.kriging import KrigingResult, TaskResults, _KrigingAttribute, _KrigingTarget

        attr = _KrigingAttribute(reference="ref1", name="attr1")
        target = _KrigingTarget(
            reference="ref1",
            name="target1",
            description="desc",
            schema_id="schema/1.0.0",
            attribute=attr,
        )
        result = KrigingResult(message="msg", target=target)

        results = TaskResults([result])

        self.assertEqual(results.results, [result])


class TestKrigingMethod(unittest.TestCase):
    """Tests for kriging method classes."""

    def test_ordinary_kriging_singleton(self):
        """KrigingMethod.ORDINARY should be an OrdinaryKriging instance."""
        from evo.compute.tasks.kriging import KrigingMethod, OrdinaryKriging

        self.assertIsInstance(KrigingMethod.ORDINARY, OrdinaryKriging)

    def test_simple_kriging_factory(self):
        """KrigingMethod.simple() should create a SimpleKriging instance."""
        from evo.compute.tasks.kriging import KrigingMethod, SimpleKriging

        method = KrigingMethod.simple(mean=100.0)
        self.assertIsInstance(method, SimpleKriging)
        self.assertEqual(method.mean, 100.0)

    def test_ordinary_kriging_to_dict(self):
        """OrdinaryKriging should serialize to dict with type='ordinary'."""
        from evo.compute.tasks.kriging import OrdinaryKriging

        d = OrdinaryKriging().to_dict()
        self.assertEqual(d, {"type": "ordinary"})

    def test_simple_kriging_to_dict(self):
        """SimpleKriging should serialize to dict with type='simple' and mean."""
        from evo.compute.tasks.kriging import SimpleKriging

        d = SimpleKriging(mean=50.0).to_dict()
        self.assertEqual(d, {"type": "simple", "mean": 50.0})


if __name__ == "__main__":
    unittest.main()
