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

"""Coordinate reference system (CRS) types and parsing helpers."""

from __future__ import annotations

import re
from typing import Any, Final

from pydantic_core import core_schema

__all__ = [
    "EpsgCode",
    "parse_crs",
]

MIN_EPSG_CODE: Final[int] = 1024
"""The lowest valid EPSG code."""

MAX_EPSG_CODE: Final[int] = 32767
"""The highest valid EPSG code."""

UNSPECIFIED_EPSG_CODE: Final[int] = 404000
"""The EPSG code used to signal that no CRS is specified."""

_UNSPECIFIED_STRINGS: Final[frozenset[str]] = frozenset({"", "unspecified"})

# "EPSG:1234", "EPSG::1234", "epsg 1234"
_EPSG_PREFIXED = re.compile(r"^epsg[\s:]+(-?\d+)$", re.IGNORECASE)

# "urn:ogc:def:crs:EPSG::1234", "urn:x-ogc:def:crs:EPSG:6.6:1234"
_EPSG_URN = re.compile(r"^urn:[\w\-]+:def:crs:epsg:[^:]*:(-?\d+)$", re.IGNORECASE)

# A bare numeric string, e.g. "1234"
_EPSG_BARE = re.compile(r"^(-?\d+)$")


class EpsgCode(int):
    """An integer representing an EPSG code."""

    def __new__(cls, value: int | str) -> EpsgCode:
        if isinstance(value, str):
            try:
                value = int(value)
            except ValueError as ve:
                raise ValueError(f"Cannot convert '{value}' to an integer EPSG code") from ve

        if not (MIN_EPSG_CODE <= value <= MAX_EPSG_CODE):
            raise ValueError(f"EPSG code must be between {MIN_EPSG_CODE} and {MAX_EPSG_CODE}, got {value}")

        return int.__new__(cls, value)

    def __repr__(self) -> str:
        return f"EpsgCode({int(self)})"

    def __str__(self) -> str:
        return f"EPSG:{int(self)}"

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: Any) -> core_schema.CoreSchema:
        return core_schema.no_info_after_validator_function(
            cls,
            core_schema.int_schema(),
            serialization=core_schema.plain_serializer_function_ser_schema(int),
        )


def parse_crs(value: int | str | None) -> EpsgCode | str | None:
    """Normalise a coordinate reference system reference.

    Accepts an ``int``, ``"EPSG:1234"``, ``"urn:ogc:def:crs:EPSG::1234"``, a bare numeric string, or
    an OGC WKT string. ``None``, ``"unspecified"``, ``""`` and EPSG ``404000`` all map to ``None``.
    Any other ``str`` is passed through unchanged as OGC WKT.

    :param value: The CRS reference to normalise.

    :return: An :class:`EpsgCode`, an OGC WKT ``str``, or ``None`` when no CRS is specified.

    :raises TypeError: If the value is not an ``int``, ``str`` or ``None``.
    :raises ValueError: If an EPSG-shaped value is outside the valid range
        [``MIN_EPSG_CODE``, ``MAX_EPSG_CODE``].
    """
    if value is None:
        return None

    if isinstance(value, bool):
        raise TypeError(f"CRS value must be an int, str or None, got {type(value).__name__}")

    if isinstance(value, int):
        return None if value == UNSPECIFIED_EPSG_CODE else EpsgCode(value)

    if not isinstance(value, str):
        raise TypeError(f"CRS value must be an int, str or None, got {type(value).__name__}")

    stripped = value.strip()
    if stripped.lower() in _UNSPECIFIED_STRINGS:
        return None

    for pattern in (_EPSG_PREFIXED, _EPSG_URN, _EPSG_BARE):
        if (match := pattern.match(stripped)) is not None:
            return parse_crs(int(match.group(1)))

    # Anything else is assumed to be an OGC WKT string, and is passed through unchanged.
    return value
