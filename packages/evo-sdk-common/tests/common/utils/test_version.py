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

import unittest
from importlib import metadata
from unittest.mock import patch

from evo.common.utils.version import PackageDetails, get_package_details


class TestGetPackageDetails(unittest.TestCase):
    def setUp(self) -> None:
        """Clear the cache before each test."""
        get_package_details.cache_clear()

    def test_get_package_details_with_valid_package(self) -> None:
        """Test that get_package_details returns correct details for a valid package."""
        # Mock the metadata to return a valid package
        mock_metadata = {  
            "name": "test-package",  
            "version": "1.2.3",  
        }

        with patch("evo.common.utils.version.metadata.metadata", return_value=mock_metadata):
            result = get_package_details("test.module")

        self.assertEqual(result.name, "test-package")
        self.assertEqual(result.version, "1.2.3")
        self.assertIsInstance(result, PackageDetails)

    def test_get_package_details_falls_back_to_evo_sdk_common(self) -> None:
        """Test that get_package_details falls back to evo-sdk-common when no package is found."""

        def mock_metadata_side_effect(candidate: str) -> dict[str, str]:
            if candidate == "evo-sdk-common":
                mock_meta = {
                    "name": "evo-sdk-common",
                    "version": "0.1.0",
                }
                return mock_meta
            else:
                raise metadata.PackageNotFoundError

        with patch("evo.common.utils.version.metadata.metadata", side_effect=mock_metadata_side_effect):
            result = get_package_details("nonexistent.module")

        self.assertEqual(result.name, "evo-sdk-common")
        self.assertEqual(result.version, "0.1.0")

    def test_get_package_details_with_no_metadata(self) -> None:
        """Test that get_package_details returns 'unknown' for both when both keys are missing."""
        mock_metadata = {}

        with patch("evo.common.utils.version.metadata.metadata", return_value=mock_metadata):
            result = get_package_details("missing.both")

        self.assertEqual(result.name, "evo-sdk-common")
        self.assertEqual(result.version, "0.0.0")


if __name__ == "__main__":
    unittest.main()
