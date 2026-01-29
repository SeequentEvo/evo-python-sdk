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

import unittest


class TestTasksModuleImports(unittest.TestCase):
    """Tests that the tasks module exports are correctly configured."""

    def test_top_level_exports(self):
        """Test that expected symbols are exported from evo.compute.tasks."""
        from evo.compute.tasks import (
            # Shared components
            Ellipsoid,
            SearchNeighborhood,
            # Run function and results
            run,
            TaskResult,
            TaskResults,
            KrigingResult,
        )
        # Just verify they're importable
        self.assertIsNotNone(Ellipsoid)
        self.assertIsNotNone(SearchNeighborhood)
        self.assertIsNotNone(run)
        self.assertIsNotNone(TaskResult)
        self.assertIsNotNone(TaskResults)
        self.assertIsNotNone(KrigingResult)

    def test_kriging_submodule_exports(self):
        """Test that kriging-specific exports are in the kriging submodule."""
        from evo.compute.tasks.kriging import (
            KrigingMethod,
            KrigingParameters,
            OrdinaryKriging,
            SimpleKriging,
        )
        # Just verify they're importable
        self.assertIsNotNone(KrigingMethod)
        self.assertIsNotNone(KrigingParameters)
        self.assertIsNotNone(OrdinaryKriging)
        self.assertIsNotNone(SimpleKriging)


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
        from evo.compute.tasks.kriging import TaskResults, KrigingResult, _KrigingTarget, _KrigingAttribute

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
        from evo.compute.tasks.kriging import TaskResults, KrigingResult, _KrigingTarget, _KrigingAttribute

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

