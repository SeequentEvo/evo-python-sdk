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

import json
import random
from uuid import UUID

from data import load_test_data
from evo.common.test_tools import TestWithConnector, TestWithDownloadHandler, TestWithUploadHandler
from evo.objects import ObjectDataDownload, ObjectDataUpload, ObjectMetadata

# The test data for these tests does need to be real parquet data, we just need enough content to test
# chunked upload and download.
TEST_DATA = random.randbytes(1024 * 1024 * 5)  # 5MB

OBJECT_ID = UUID(int=5)
VERSION_ID = "123456"
DATA_NAME = "0000000000000000000000000000000000000000000000000000000000000001"
INITIAL_URL = "https://unit.test/initial/url"


class TestObjectDataDownload(TestWithConnector, TestWithDownloadHandler):
    def setUp(self) -> None:
        TestWithConnector.setUp(self)
        TestWithDownloadHandler.setUp(self)
        self.setup_download_handler(TEST_DATA)
        self.metadata = ObjectMetadata(
            environment=self.environment,
            id=OBJECT_ID,
            name="object.json",
            created_by=...,
            created_at=...,
            modified_at=...,
            modified_by=...,
            parent="/some/path",
            schema_id=...,
            version_id=VERSION_ID,
            stage=None,
        )
        self.download = ObjectDataDownload(
            connector=self.connector, metadata=self.metadata, name=DATA_NAME, initial_url=INITIAL_URL
        )

    def test_label(self) -> None:
        expected = f"{self.metadata.path} (ref={DATA_NAME})"
        self.assertEqual(expected, self.download.label)

    def test_metadata(self) -> None:
        self.assertEqual(self.metadata, self.download.metadata)

    def test_get_cache_location(self) -> None:
        expected = self.cache.get_location(self.environment, "geoscience-object") / str(DATA_NAME)
        self.assertEqual(expected, self.download._get_cache_location(self.cache))

    async def test_get_download_url(self) -> None:
        # Test the initial URL is used first.
        first = await self.download.get_download_url()
        self.assertEqual(INITIAL_URL, first)

        # No requests should be made when using the initial URL.
        self.transport.assert_no_requests()

        # Test that a new URL is generated when the initial URL is used up.
        get_object_response = load_test_data("get_object.json")
        with self.transport.set_http_response(
            status_code=200,
            content=json.dumps(get_object_response),
            headers={"Content-Type": "application/json"},
        ):
            second = await self.download.get_download_url()
        self.assertEqual(get_object_response["links"]["data"][1]["download_url"], second)

    async def test_download_to_path(self) -> None:
        dest = self.cache.root / "test_download_to_path.parquet"
        await self.download.download_to_path(dest, self.transport)

        # Check the data was downloaded correctly.
        self.assertEqual(TEST_DATA, dest.read_bytes())
        self.assert_download_requests(INITIAL_URL)

    async def test_download_to_cache(self) -> None:
        dest = await self.download.download_to_cache(self.cache, self.transport)

        # Check the download location is correct.
        expected = self.cache.get_location(self.environment, "geoscience-object") / str(DATA_NAME)
        self.assertEqual(expected, dest)

        # Check the data was downloaded correctly.
        self.assertEqual(TEST_DATA, dest.read_bytes())
        self.assert_download_requests(INITIAL_URL)


class TestObjectDataUpload(TestWithConnector, TestWithUploadHandler):
    def setUp(self) -> None:
        TestWithConnector.setUp(self)
        TestWithUploadHandler.setUp(self)
        self.upload = ObjectDataUpload(
            connector=self.connector,
            environment=self.environment,
            name=DATA_NAME,
            initial_url=INITIAL_URL,
        )

    def test_label(self) -> None:
        expected = DATA_NAME
        self.assertEqual(expected, self.upload.label)

    async def test_get_upload_url(self) -> None:
        # Test the initial URL is used first.
        first = await self.upload.get_upload_url()
        self.assertEqual(INITIAL_URL, first)

        # No requests should be made when using the initial URL.
        self.transport.assert_no_requests()

        # Test that a new URL is generated when the initial URL is used up.
        put_data_response = load_test_data("put_data.json")
        with self.transport.set_http_response(
            status_code=200,
            content=json.dumps(put_data_response),
            headers={"Content-Type": "application/json"},
        ):
            second = await self.upload.get_upload_url()
        self.assertEqual(put_data_response[0]["upload_url"], second)

    async def test_upload_from_path(self) -> None:
        source = self.cache.root / "test_upload_from_path.parquet"
        source.write_bytes(TEST_DATA)
        await self.upload.upload_from_path(source, self.transport)

        # Check the file was uploaded correctly.
        uploaded = await self.handler.get_committed()
        self.assertEqual(TEST_DATA, uploaded)
