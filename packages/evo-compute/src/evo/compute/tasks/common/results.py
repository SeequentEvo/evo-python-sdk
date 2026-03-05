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

"""Common result types for compute tasks.

This module provides base result classes that all compute task types can inherit
from. These were originally defined in the kriging module but are generic enough
for any task type.
"""

from __future__ import annotations

from pydantic import BaseModel

__all__ = [
    "TaskAttribute",
    "TaskTarget",
]


class TaskAttribute(BaseModel):
    """Attribute information from a task result."""

    reference: str
    name: str


class TaskTarget(BaseModel):
    """Target information from a task result."""

    reference: str
    name: str
    description: str | None = None
    schema_id: str
    attribute: TaskAttribute
