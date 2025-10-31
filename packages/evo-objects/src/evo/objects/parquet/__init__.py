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

try:
    import pyarrow  # noqa: F401
except ImportError:
    raise ImportError("The 'pyarrow' package is required to use ParquetLoader") from None

from .loader import ParquetDownloader, ParquetLoader
from .types import ArrayTableInfo, AttributeInfo, CategoryInfo, LookupTableInfo, TableInfo

__all__ = [
    "ArrayTableInfo",
    "AttributeInfo",
    "CategoryInfo",
    "LookupTableInfo",
    "ParquetDownloader",
    "ParquetLoader",
    "TableInfo",
]
