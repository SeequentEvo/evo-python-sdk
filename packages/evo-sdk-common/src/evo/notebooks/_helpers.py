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

import base64
from pathlib import Path
from typing import TypeAlias

from evo.common.utils import Cache

from . import assets
from ._consts import DEFAULT_CACHE_LOCATION

FileName: TypeAlias = str | Path

_MEDIA_TYPES = {".gif": "image/gif", ".png": "image/png"}


def init_cache(cache_location: FileName = DEFAULT_CACHE_LOCATION) -> Cache:
    """Initialise the storage location for the notebook environment.

    Configures the cache location and creates a `.gitignore` file in the root of the cache directory.

    :param cache_location: The location for the cache directory.

    :returns: A Cache instance.
    """
    cache = Cache(cache_location, mkdir=True)
    ignorefile = cache.root / ".gitignore"
    ignorefile.write_text("*\n")
    return cache


def read_asset_text(filename: str) -> str:
    """Read a bundled text asset, such as the widget ESM or stylesheet.

    :param filename: The name of the file in the assets directory.

    :returns: The contents of the file.
    """
    return assets.get(filename).read_text(encoding="utf-8")


def asset_data_uri(filename: str) -> str:
    """Encode a bundled image asset as a data URI so it can be embedded directly in widget markup.

    :param filename: The name of the file in the assets directory.

    :returns: The data URI for the image.
    """
    encoded = base64.b64encode(assets.get(filename).read_bytes()).decode("ascii")
    return f"data:{_MEDIA_TYPES[Path(filename).suffix.lower()]};base64,{encoded}"
