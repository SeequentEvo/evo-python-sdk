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

"""Tests for BlockSync URL generation utilities."""

import unittest
from uuid import UUID

from evo.blockmodels.urls import (
    get_blocksync_base_url,
    get_blocksync_block_model_url,
)


class TestGetBlocksyncBaseUrl(unittest.TestCase):
    """Tests for get_blocksync_base_url function."""

    def test_integration_environment(self):
        """Integration environment hub URL returns integration BlockSync URL."""
        hub_url = "https://350mt.api.integration.seequent.com"
        result = get_blocksync_base_url(hub_url)
        self.assertEqual(result, "https://blocksync.integration.seequent.com")

    def test_qa_environment(self):
        """QA environment hub URL returns integration BlockSync URL."""
        hub_url = "https://350mt.api.qa.seequent.com"
        result = get_blocksync_base_url(hub_url)
        self.assertEqual(result, "https://blocksync.integration.seequent.com")

    def test_production_environment(self):
        """Production environment hub URL returns production BlockSync URL."""
        hub_url = "https://350mt.api.seequent.com"
        result = get_blocksync_base_url(hub_url)
        self.assertEqual(result, "https://blocksync.seequent.com")


class TestGetBlocksyncBlockModelUrl(unittest.TestCase):
    """Tests for get_blocksync_block_model_url function."""

    def test_generates_correct_url_with_strings(self):
        """BlockSync block model URL is generated correctly with string IDs."""
        result = get_blocksync_block_model_url(
            org_id="org-123",
            workspace_id="ws-456",
            block_model_id="bm-789",
            hub_url="https://350mt.api.integration.seequent.com",
        )
        self.assertEqual(
            result,
            "https://blocksync.integration.seequent.com/org-123/redirect?ws=ws-456&bm=bm-789",
        )

    def test_generates_correct_url_with_uuids(self):
        """BlockSync block model URL is generated correctly with UUID objects."""
        org_id = UUID("829e6621-0ab6-4d7d-96bb-2bb5b407a5fe")
        workspace_id = UUID("783b6eef-01b9-42a7-aaf4-35e153e6fcbe")
        block_model_id = UUID("9100d7dc-44e9-4e61-b427-159635dea22f")

        result = get_blocksync_block_model_url(
            org_id=org_id,
            workspace_id=workspace_id,
            block_model_id=block_model_id,
            hub_url="https://350mt.api.integration.seequent.com",
        )
        self.assertEqual(
            result,
            "https://blocksync.integration.seequent.com/829e6621-0ab6-4d7d-96bb-2bb5b407a5fe"
            "/redirect?ws=783b6eef-01b9-42a7-aaf4-35e153e6fcbe&bm=9100d7dc-44e9-4e61-b427-159635dea22f",
        )

    def test_production_url(self):
        """Production BlockSync URL is generated correctly."""
        result = get_blocksync_block_model_url(
            org_id="org-123",
            workspace_id="ws-456",
            block_model_id="bm-789",
            hub_url="https://mining.api.seequent.com",
        )
        self.assertEqual(
            result,
            "https://blocksync.seequent.com/org-123/redirect?ws=ws-456&bm=bm-789",
        )

    def test_lowercase_conversion(self):
        """IDs are converted to lowercase in the URL."""
        result = get_blocksync_block_model_url(
            org_id="ORG-ABC",
            workspace_id="WS-DEF",
            block_model_id="BM-GHI",
            hub_url="https://350mt.api.seequent.com",
        )
        self.assertEqual(
            result,
            "https://blocksync.seequent.com/org-abc/redirect?ws=ws-def&bm=bm-ghi",
        )


if __name__ == "__main__":
    unittest.main()

