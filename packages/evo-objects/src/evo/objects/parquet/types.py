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

import sys
from typing import TypeAlias

if sys.version_info >= (3, 12):
    from typing import NotRequired, TypedDict
else:
    from typing_extensions import NotRequired, TypedDict

__all__ = [
    "ArrayTableInfo",
    "LookupTableInfo",
    "TableInfo",
]


class _BaseTableInfo(TypedDict):
    data: str
    length: int


class ArrayTableInfo(_BaseTableInfo):
    data_type: str
    width: NotRequired[int]


class LookupTableInfo(_BaseTableInfo):
    keys_data_type: str
    values_data_type: str


TableInfo: TypeAlias = ArrayTableInfo | LookupTableInfo
