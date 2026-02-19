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

"""Typed access for block models with pandas DataFrame support."""

from .regular_block_model import RegularBlockModel, RegularBlockModelData
from .report import (
    Aggregation,
    MassUnits,
    Report,
    ReportCategorySpec,
    ReportColumnSpec,
    ReportResult,
    ReportSpecificationData,
)
from .types import BoundingBox, Point3, Size3d, Size3i
from .units import UnitInfo, Units, UnitType, get_available_units

__all__ = [
    "Aggregation",
    "BoundingBox",
    "MassUnits",
    "Point3",
    "RegularBlockModel",
    "RegularBlockModelData",
    "Report",
    "ReportCategorySpec",
    "ReportColumnSpec",
    "ReportResult",
    "ReportSpecificationData",
    "Size3d",
    "Size3i",
    "UnitInfo",
    "UnitType",
    "Units",
    "get_available_units",
]
