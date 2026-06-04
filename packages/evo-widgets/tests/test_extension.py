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

import unittest
from unittest.mock import MagicMock, patch

from evo.common.utils import NoFeedback, create_default_feedback, reset_feedback_factory
from evo.widgets import _register_feedback_factory, _unregister_feedback_factory


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


if __name__ == "__main__":
    unittest.main()
