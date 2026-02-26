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

"""Type definitions for typed block model access.

These types are re-exported from evo.common.typed for backward compatibility.
"""

from evo.common.typed import BoundingBox, Point3, Size3d, Size3i

__all__ = [
    "BoundingBox",
    "Point3",
    "Size3d",
    "Size3i",
]
