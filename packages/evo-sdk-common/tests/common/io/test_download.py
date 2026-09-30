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

import asyncio
from pathlib import Path
from uuid import UUID

from evo.common import ICache, ResourceMetadata
from evo.common.io import Download
from evo.common.test_tools import TestWithDownloadHandler, UrlGenerator, utc_datetime

from .common import TEST_DATA


class MyResourceMetadata(ResourceMetadata):
    @property
    def url(self) -> str:
        return self.environment.hub_url + f"path/{self.name}"


class MyDownload(Download[MyResourceMetadata]):
    def __init__(self, metadata: MyResourceMetadata, url_generator: UrlGenerator) -> None:
        super().__init__()
        self._generator = url_generator
        self._metadata = metadata

    @property
    def metadata(self) -> MyResourceMetadata:
        return self._metadata

    @property
    def label(self) -> str:
        return "test_download"

    def _get_cache_location(self, cache: ICache) -> Path:
        return cache.get_location(self.metadata.environment, "downloads") / self.metadata.name

    async def get_download_url(self) -> str:
        return await self._generator.get_new_url()


class TestDownload(TestWithDownloadHandler):
    def setUp(self) -> None:
        super().setUp()
        self.setup_download_handler(TEST_DATA)
        self.metadata = MyResourceMetadata(
            environment=self.environment,
            id=UUID(int=4321),
            name="test_data_download.csv",
            created_at=utc_datetime(2024),
            created_by=None,
        )
        self.download = MyDownload(self.metadata, self.url_generator)

    async def test_to_path(self) -> None:
        """Test downloading a file to a path."""
        test_data_file = self.CACHE_DIR / "test_data_download.csv"
        self.assertFalse(test_data_file.exists())
        await self.download.download_to_path(test_data_file, self.transport)
        self.assert_download_requests(self.url_generator.current_url)
        self.assertTrue(test_data_file.exists())
        self.assertEqual(TEST_DATA, test_data_file.read_bytes())

    async def test_to_cache(self) -> None:
        """Test downloading a file to the cache."""
        expected_data_file = self.download._get_cache_location(self.cache)
        self.assertFalse(expected_data_file.exists())
        actual_data_file = await self.download.download_to_cache(self.cache, self.transport)
        self.assert_download_requests(self.url_generator.current_url)
        self.assertEqual(expected_data_file, actual_data_file)
        self.assertTrue(expected_data_file.exists())
        self.assertEqual(TEST_DATA, expected_data_file.read_bytes())

    async def test_concurrent_downloads_to_cache(self) -> None:
        """Concurrent requests for the same cached file only download it once."""
        download = MyDownload(
            MyResourceMetadata(
                environment=self.environment,
                id=UUID(int=4322),
                name="test_concurrent_download.csv",
                created_at=utc_datetime(2024),
                created_by=None,
            ),
            self.url_generator,
        )
        expected_data_file = download._get_cache_location(self.cache)

        locations = await asyncio.gather(
            download.download_to_cache(self.cache, self.transport),
            download.download_to_cache(self.cache, self.transport),
        )

        self.assertEqual([expected_data_file, expected_data_file], locations)
        self.assert_download_requests(self.url_generator.current_url)
        self.assertEqual(TEST_DATA, expected_data_file.read_bytes())
