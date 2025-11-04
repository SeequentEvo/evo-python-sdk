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
from uuid import UUID

from data import load_test_data
from evo.common import ServiceUser
from evo.common.test_tools import TestWithConnector, TestWithDownloadHandler, TestWithUploadHandler, utc_datetime
from evo.files import FileAPIDownload, FileAPIUpload, FileMetadata

TEST_DATA = """
x,y,z
10689.54607,100813.2874,442.172449
10690.87819,100812.6204,437.399555
10692.21062,100811.9502,432.627182
10693.54337,100811.277,427.85533
10694.87638,100810.6009,423.083966
10696.06353,100809.9976,418.831067
10725.47983,100777.2912,448.471682
10727.51038,100775.9784,444.022761
10729.54666,100774.6657,439.576449
10731.58199,100773.3566,435.128615
""".lstrip().encode("utf-8")

FILE_ID = UUID(int=5)
VERSION_ID = "123456"
INITIAL_URL = "https://unit.test/initial/url"


class TestFileAPIDownload(TestWithConnector, TestWithDownloadHandler):
    def setUp(self) -> None:
        TestWithConnector.setUp(self)
        TestWithDownloadHandler.setUp(self)
        self.setup_download_handler(TEST_DATA)
        self.metadata = FileMetadata(
            environment=self.environment,
            id=FILE_ID,
            name="file.csv",
            created_by=ServiceUser(
                id=UUID(int=16),
                name="x y",
                email="test@example.com",
            ),
            created_at=utc_datetime(2020, 1, 1, 1, 30),
            modified_by=ServiceUser(
                id=UUID(int=16),
                name="x y",
                email="test@example.com",
            ),
            modified_at=utc_datetime(2020, 1, 1, 1, 30),
            parent="/some/path",
            version_id=VERSION_ID,
            size=len(TEST_DATA),
        )
        self.download = FileAPIDownload(connector=self.connector, metadata=self.metadata, initial_url=INITIAL_URL)

    def test_label(self) -> None:
        expected = f"{self.metadata.id}?version_id={self.metadata.version_id}"
        self.assertEqual(expected, self.download.label)

    def test_metadata(self) -> None:
        self.assertEqual(self.metadata, self.download.metadata)

    def test_get_cache_location(self) -> None:
        expected = self.cache.get_location(self.environment, "filev2") / str(FILE_ID) / VERSION_ID / self.metadata.name
        self.assertEqual(expected, self.download._get_cache_location(self.cache))

    async def test_get_download_url(self) -> None:
        # Test the initial URL is used first.
        first = await self.download.get_download_url()
        self.assertEqual(INITIAL_URL, first)

        # No requests should be made when using the initial URL.
        self.transport.assert_no_requests()

        # Test that a new URL is generated when the initial URL is used up.
        get_file_response = load_test_data("get_file.json")
        with self.transport.set_http_response(
            status_code=200, content=json.dumps(get_file_response), headers={"Content-Type": "application/json"}
        ):
            second = await self.download.get_download_url()
        self.assertEqual(get_file_response["download"], second)

    async def test_download_to_path(self) -> None:
        dest = self.cache.root / "test_download_to_path.csv"
        await self.download.download_to_path(dest, self.transport)

        # Check the file was downloaded correctly.
        self.assertEqual(TEST_DATA, dest.read_bytes())
        self.assert_download_requests(INITIAL_URL)

    async def test_download_to_cache(self) -> None:
        dest = await self.download.download_to_cache(self.cache, self.transport)

        # Check the download location is correct.
        expected = self.cache.get_location(self.environment, "filev2") / str(FILE_ID) / VERSION_ID / self.metadata.name
        self.assertEqual(expected, dest)

        # Check the file was downloaded correctly.
        self.assertEqual(TEST_DATA, dest.read_bytes())
        self.assert_download_requests(INITIAL_URL)


class TestFileAPIUpload(TestWithConnector, TestWithUploadHandler):
    def setUp(self) -> None:
        TestWithConnector.setUp(self)
        TestWithUploadHandler.setUp(self)
        self.upload = FileAPIUpload(
            connector=self.connector,
            environment=self.environment,
            file_id=FILE_ID,
            version_id=VERSION_ID,
            initial_url=INITIAL_URL,
        )

    def test_label(self) -> None:
        expected = f"{FILE_ID}?version_id={VERSION_ID}"
        self.assertEqual(expected, self.upload.label)

    def test_file_id(self) -> None:
        self.assertEqual(self.upload.file_id, FILE_ID)

    def test_version_id(self) -> None:
        self.assertEqual(self.upload.version_id, VERSION_ID)

    async def test_get_upload_url(self) -> None:
        # Test the initial URL is used first.
        first = await self.upload.get_upload_url()
        self.assertEqual(INITIAL_URL, first)

        # No requests should be made when using the initial URL.
        self.transport.assert_no_requests()

        # Test that a new URL is generated when the initial URL is used up.
        update_file_response = load_test_data("update_file.json")
        with self.transport.set_http_response(
            status_code=200, content=json.dumps(update_file_response), headers={"Content-Type": "application/json"}
        ):
            second = await self.upload.get_upload_url()
        self.assertEqual(update_file_response["upload"], second)

    async def test_upload_from_path(self) -> None:
        source = self.cache.root / "test_upload_from_path.csv"
        source.write_bytes(TEST_DATA)
        await self.upload.upload_from_path(source, self.transport)

        # Check the file was uploaded correctly.
        uploaded = await self.handler.get_committed()
        self.assertEqual(TEST_DATA, uploaded)
