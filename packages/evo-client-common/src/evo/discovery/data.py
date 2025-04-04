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

from dataclasses import dataclass
from uuid import UUID

__all__ = [
    "Hub",
    "Organization",
]


@dataclass(frozen=True, kw_only=True)
class Hub:
    """Hub metadata."""

    url: str
    """Hub URL."""

    code: str
    """Hub shortcode."""

    display_name: str
    """Hub display name."""

    services: tuple[str, ...]
    """List of service codes."""


@dataclass(frozen=True, kw_only=True)
class Organization:
    """License holder organization metadata."""

    id: UUID
    """Organization ID."""

    display_name: str
    """Organization display name."""

    hubs: tuple[Hub, ...]
