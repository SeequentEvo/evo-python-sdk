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

"""Tests for the evo.widgets IPython extension feedback factory registration."""

import importlib
import importlib.util
import unittest
from unittest.mock import MagicMock, patch

from evo.common.utils import NoFeedback, create_default_feedback, reset_feedback_factory
from evo.widgets import (
    _TASK_RESULT_LIST_TYPE,
    _TASK_RESULT_TYPES,
    _register_feedback_factory,
    _unregister_feedback_factory,
)
from evo.widgets.formatters import _TASK_RESULT_DETAILS


class TestFeedbackFactoryRegistration(unittest.TestCase):
    """Tests for _register_feedback_factory / _unregister_feedback_factory."""

    def tearDown(self) -> None:
        reset_feedback_factory()

    def test_register_feedback_factory_sets_factory(self) -> None:
        """After _register_feedback_factory, create_default_feedback should produce a FeedbackWidget-like object."""
        mock_widget = MagicMock()
        mock_widget_class = MagicMock(return_value=mock_widget)

        # Patch at the import targets inside _register_feedback_factory so that
        # the test works even when evo-sdk-common[notebooks] is not installed.
        with (
            patch.dict("sys.modules", {"evo.notebooks": MagicMock(FeedbackWidget=mock_widget_class)}),
        ):
            _register_feedback_factory()

        result = create_default_feedback("Test Label")
        mock_widget_class.assert_called_once_with("Test Label")
        self.assertIs(result, mock_widget)

    def test_unregister_feedback_factory_restores_default(self) -> None:
        """After _unregister_feedback_factory, create_default_feedback should return NoFeedback."""
        mock_widget_class = MagicMock(return_value=MagicMock())

        with (
            patch.dict("sys.modules", {"evo.notebooks": MagicMock(FeedbackWidget=mock_widget_class)}),
        ):
            _register_feedback_factory()

        # Sanity: factory is active
        self.assertIsNot(create_default_feedback("x"), NoFeedback)

        _unregister_feedback_factory()

        self.assertIs(create_default_feedback("x"), NoFeedback)

    def test_register_feedback_factory_handles_import_error(self) -> None:
        """_register_feedback_factory should silently handle ImportError."""
        with patch.dict("sys.modules", {"evo.notebooks": None}):
            # Should not raise
            _register_feedback_factory()

        # Factory should still be default
        self.assertIs(create_default_feedback("x"), NoFeedback)

    def test_unregister_feedback_factory_handles_import_error(self) -> None:
        """_unregister_feedback_factory should silently handle ImportError."""
        with patch.dict("sys.modules", {"evo.common.utils": None}):
            # Should not raise
            _unregister_feedback_factory()


class TestTaskResultFormatterRegistration(unittest.TestCase):
    """Tests that registered task result paths match the real evo-compute layout."""

    def setUp(self) -> None:
        if importlib.util.find_spec("evo.compute") is None:
            self.skipTest("evo-compute is not installed")

    def test_expected_task_result_details_are_registered(self) -> None:
        expected = {
            "BreakTiesResult": "_target_result_details",
            "ConditionalTurningBandsResult": "_turning_bands_result_details",
            "ConSimResult": "_conditional_simulation_result_details",
            "ContinuousDistributionResult": "_distribution_result_details",
            "DeclusteringResult": "_target_result_details",
            "IDWResult": "_target_result_details",
            "KNNResult": "_target_result_details",
            "KrigingResult": "_target_result_details",
            "LocationWiseResult": "_location_wise_result_details",
            "LossCalculationResult": "_target_result_details",
            "NormalScoreResult": "_target_result_details",
            "ProfitCalculationResult": "_target_result_details",
            "SimulationReportResult": "_simulation_report_result_details",
        }
        self.assertEqual(
            {class_name: details.__name__ for (_, class_name), details in _TASK_RESULT_DETAILS.items()},
            expected,
        )

    def test_registered_task_result_types_resolve(self) -> None:
        """Each registered task result class must exist, or IPython silently skips the formatter."""
        for module_name, class_name in (*_TASK_RESULT_TYPES, _TASK_RESULT_LIST_TYPE):
            with self.subTest(module=module_name, cls=class_name):
                module = importlib.import_module(module_name)
                self.assertTrue(
                    hasattr(module, class_name),
                    f"{module_name}.{class_name} does not exist",
                )

    def test_registered_paths_match_class_modules(self) -> None:
        """The registered module must be the one that actually defines the class."""
        for module_name, class_name in (*_TASK_RESULT_TYPES, _TASK_RESULT_LIST_TYPE):
            with self.subTest(module=module_name, cls=class_name):
                cls = getattr(importlib.import_module(module_name), class_name)
                self.assertEqual(cls.__module__, module_name)


if __name__ == "__main__":
    unittest.main()
