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
    # Import the table type for backwards compatibility. This should be removed in a future release.
    from pyarrow import Table  # noqa: F401
except ImportError:
    raise ImportError("pyarrow is required to use the utils package in evo-objects")

try:
    # Import the dataframe type for backwards compatibility. This should be removed in a future release.
    from pandas import DataFrame  # noqa: F401
except ImportError:
    DataFrame = None  # type: ignore

from .data import ObjectDataClient
from .table_formats import all_known_formats, get_known_format
from .tables import ArrowTableFormat, BaseTableFormat, KnownTableFormat

# We _used_ to export Table and DataFrame from this package as custom protocols, but we are using the actual
# pyarrow.Table and pandas.DataFrame types now. We are importing these types here from pyarrow and pandas
# for backwards compatibility, but they are no longer explicitly exported as the exports should be
# removed in a future release.

__all__ = [
    "ArrowTableFormat",
    "BaseTableFormat",
    "KnownTableFormat",
    "ObjectDataClient",
    "all_known_formats",
    "get_known_format",
]
