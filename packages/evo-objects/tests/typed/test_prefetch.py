#  Copyright © 2026 Bentley Systems, Incorporated
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase

from evo.objects.typed._prefetch import collect_data_ids, prefetch_object_data


class _Cache:
    def __init__(self, path: Path) -> None:
        self.path = path

    def get_location(self, **_kwargs) -> Path:
        return self.path


class _Download:
    def __init__(self, identifier: str, owner: _Object) -> None:
        self.identifier = identifier
        self.owner = owner

    async def download_to_cache(self, cache: _Cache, _transport, fb=None) -> None:
        self.owner.active += 1
        self.owner.max_active = max(self.owner.max_active, self.owner.active)
        try:
            await asyncio.sleep(0.001)
            (cache.path / self.identifier).touch()
            self.owner.downloaded.append(self.identifier)
        finally:
            self.owner.active -= 1


class _Object:
    def __init__(self, document: dict, cache: _Cache) -> None:
        self.document = document
        self.cache = cache
        self.downloaded: list[str] = []
        self.requested: list[str] = []
        self.active = 0
        self.max_active = 0

    def as_dict(self) -> dict:
        return self.document

    def get_cache(self) -> _Cache:
        return self.cache

    def get_environment(self):
        return object()

    def get_connector(self):
        return SimpleNamespace(transport=object())

    def prepare_data_download(self, identifiers):
        self.requested.extend(identifiers)
        return (_Download(identifier, self) for identifier in identifiers)


class TestPrefetch(IsolatedAsyncioTestCase):
    async def test_prefetch_deduplicates_honours_concurrency_and_warms_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = _Cache(Path(directory))
            obj = _Object({"first": {"data": "a"}, "second": [{"data": "b"}, {"data": "a"}]}, cache)
            await prefetch_object_data(obj, max_concurrent=1)
            self.assertListEqual(obj.requested, ["a", "b"])
            self.assertListEqual(obj.downloaded, ["a", "b"])
            self.assertEqual(obj.max_active, 1)

            await prefetch_object_data(obj)
            self.assertListEqual(obj.requested, ["a", "b"])

    async def test_prefetch_collects_category_values_and_lookup_and_accepts_subset(self):
        document = {
            "category": {"values": {"data": "values"}, "lookup": {"data": "lookup"}},
            "other": {"data": "other"},
        }
        self.assertListEqual(collect_data_ids(document), ["values", "lookup", "other"])
        with tempfile.TemporaryDirectory() as directory:
            obj = _Object(document, _Cache(Path(directory)))
            await prefetch_object_data(obj, data_ids=["lookup"])
            self.assertListEqual(obj.downloaded, ["lookup"])

    async def test_prefetch_empty_document_is_a_no_op_and_rejects_invalid_concurrency(self):
        with tempfile.TemporaryDirectory() as directory:
            obj = _Object({}, _Cache(Path(directory)))
            await prefetch_object_data(obj)
            self.assertListEqual(obj.requested, [])
            with self.assertRaises(ValueError):
                await prefetch_object_data(obj, max_concurrent=0)
