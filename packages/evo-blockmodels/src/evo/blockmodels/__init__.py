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

from .client import BlockModelAPIClient

__all__ = [
    "BlockModelAPIClient",
]

# Typed access requires the 'utils' extra (pyarrow, pandas).
# Install with: pip install evo-blockmodels[utils]
try:
    from .typed import (  # noqa: F401
        BaseTypedBlockModel,
        BoundingBox,
        Point3,
        RegularBlockModel,
        RegularBlockModelData,
        Size3d,
        Size3i,
        Units,
        get_available_units,
    )
except ImportError:
    _UTILS_AVAILABLE = False
else:
    _UTILS_AVAILABLE = True
    __all__ += [
        "BaseTypedBlockModel",
        "BoundingBox",
        "Point3",
        "RegularBlockModel",
        "RegularBlockModelData",
        "Size3d",
        "Size3i",
        "Units",
        "get_available_units",
    ]
