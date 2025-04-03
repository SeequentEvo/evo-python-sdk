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

import re

import dotenv

from evo import logging
from evo.common import ICache

__all__ = ["DotEnv"]

logger = logging.getLogger(__name__)

_KEY = re.compile(r"^[A-Z0-9_\-\.]+$", flags=re.IGNORECASE)


class DotEnv:
    """A simple wrapper around `python-dotenv` to manage and cache environment variables."""

    def __init__(self, cache: ICache, filename: str = ".env") -> None:
        self._file = cache.root / filename
        self._file.touch(exist_ok=True)
        self._cache = dotenv.dotenv_values(self._file, encoding="utf-8")

    def get(self, key: str, default: str | None = None) -> str | None:
        logger.debug(f"Getting environment variable: {key}")
        return self._cache.get(key, default)

    def set(self, key: str, value: str | None) -> None:
        logger.debug(f"Updating environment variable: {key}={value!r}")
        if not _KEY.match(key):
            raise ValueError(f"Invalid key: {key!r}")
        elif value is not None and not isinstance(value, str):
            raise ValueError(f"Invalid value for key {key}: {value!r}")
        elif value is None:
            dotenv.unset_key(self._file, key)
            self._cache.pop(key, None)
        else:
            dotenv.set_key(self._file, key, value, quote_mode="always", encoding="utf-8")
            self._cache[key] = value
