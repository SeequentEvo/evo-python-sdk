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

"""Helpers for warming typed-object data in the local cache."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from evo.common import IFeedback
from evo.common.utils import NoFeedback
from evo.objects import DownloadedObject
from evo.objects.io import _CACHE_SCOPE


def collect_data_ids(documents: Any) -> list[str]:
    """Collect de-duplicated ``data`` references from document values."""
    ids: list[str] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                if key == "data" and isinstance(nested, str) and nested not in ids:
                    ids.append(nested)
                else:
                    visit(nested)
        elif isinstance(value, (list, tuple)):
            for nested in value:
                visit(nested)

    visit(documents)
    return ids


async def prefetch_object_data(
    obj: DownloadedObject,
    *,
    data_ids: Sequence[str] | None = None,
    max_concurrent: int = 8,
    fb: IFeedback = NoFeedback,
) -> None:
    """Warm cache entries referenced by an object, downloading each ID at most once."""
    if max_concurrent < 1:
        raise ValueError("max_concurrent must be at least 1")
    cache = obj.get_cache()
    if cache is None:
        raise ValueError("prefetch requires an IContext with a cache")
    identifiers = list(dict.fromkeys(data_ids if data_ids is not None else collect_data_ids(obj.as_dict())))
    if not identifiers:
        return
    cache_location = cache.get_location(environment=obj.get_environment(), scope=_CACHE_SCOPE)
    identifiers = [identifier for identifier in identifiers if not (cache_location / identifier).exists()]
    if not identifiers:
        return
    contexts = list(obj.prepare_data_download(identifiers))
    semaphore = asyncio.Semaphore(max_concurrent)

    async def download(context: Any) -> None:
        async with semaphore:
            await context.download_to_cache(cache, obj.get_connector().transport, fb=fb)

    await asyncio.gather(*(download(context) for context in contexts))
