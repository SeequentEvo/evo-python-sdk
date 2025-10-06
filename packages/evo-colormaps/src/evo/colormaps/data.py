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

from __future__ import annotations

__all__ = [
    "Association",
    "AssociationMetadata",
    "CategoryColormap",
    "ColormapMetadata",
    "ContinuousColormap",
    "DiscreteColormap",
]

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from evo.common import ResourceMetadata, ServiceUser


def _validate_colors(colors: list[list[int]], minimum: int = 2, maximum: int = 1024):
    if not minimum <= len(colors) <= maximum:
        raise ValueError("Colors must have between 2 and 1024 entries")

    for color in colors:
        if len(color) != 3 or any(not isinstance(value, int) or value < 0 or value > 255 for value in color):
            raise ValueError("Each color must have exactly 3 values, each in the range 0-255")


@dataclass(frozen=True, kw_only=True)
class ContinuousColormap:
    """Continuous colormap data."""

    colors: list[list[int]]
    """A list of colors. Each color is an RGB representation of the color as list of 3 integers in the range 0-255.
    The list must have between 2 and 1024 colors."""

    attribute_controls: list[float]

    gradient_controls: list[float]

    def __post_init__(self):
        _validate_colors(self.colors, minimum=2, maximum=1024)


@dataclass(frozen=True, kw_only=True)
class DiscreteColormap:
    """Discrete colormap data."""

    colors: list[list[int]]
    """A list of colors. Each color is an RGB representation of the color as list of 3 integers in the range 0-255.
    The list must have between 1 and 1024 colors."""

    end_inclusive: list[bool]

    end_points: list[float]

    def __post_init__(self):
        _validate_colors(self.colors, minimum=1, maximum=1024)


@dataclass(frozen=True, kw_only=True)
class CategoryColormap:
    """Category colormap data."""

    colors: list[list[int]]
    """A list of colors. Each color is an RGB representation of the color as list of 3 integers in the range 0-255.
    The list must have between 1 and 10000 colors."""

    map: list[str]

    def __post_init__(self):
        _validate_colors(self.colors, minimum=1, maximum=10000)


@dataclass(frozen=True, kw_only=True)
class ColormapMetadata(ResourceMetadata):
    """Colormap metadata."""

    modified_at: datetime
    """The last modified timestamp."""

    modified_by: ServiceUser
    """The user that last modified the colormap."""

    colormap: ContinuousColormap | DiscreteColormap | CategoryColormap

    self_link: str
    """The URL of the colormap."""

    def url(self) -> str:
        return self.self_link


@dataclass(frozen=True, kw_only=True)
class Association:
    """An association between an attribute and a colormap."""

    attribute_id: str
    """The ID of the attribute."""

    colormap_id: UUID
    """The ID of the colormap."""


@dataclass(frozen=True, kw_only=True)
class AssociationMetadata(ResourceMetadata):
    """Association metadata."""

    modified_at: datetime
    """The last modified timestamp."""

    modified_by: ServiceUser
    """The user that last modified the association."""

    self_link: str
    """The URL of the association."""

    colormap_id: UUID
    """The ID of the colormap."""

    attribute_id: str
    """The ID of the attribute."""

    object_id: UUID
    """The ID of the object."""

    colormap_uri: str
    """The URI of the colormap."""

    def url(self) -> str:
        return self.self_link
