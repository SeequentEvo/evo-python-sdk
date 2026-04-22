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

"""DEPRECATED: Use evo.widgets instead.

This module is deprecated and will be removed in a future version.
Please migrate to the new anywidget-based widgets in `evo.widgets`:

    # Old (deprecated):
    from evo.notebooks import ServiceManagerWidget

    # New (recommended):
    from evo.widgets import ServiceManagerWidget

The new widgets in `evo.widgets` provide improved compatibility across
different Jupyter environments through the anywidget framework.
"""

import warnings

warnings.warn(
    "evo.notebooks is deprecated and will be removed in a future version. "
    "Please use evo.widgets instead, which provides improved anywidget-based widgets. "
    "See the migration guide at https://developer.seequent.com/docs/guides/migration/evo-widgets",
    DeprecationWarning,
    stacklevel=2,
)

from .widgets import (  # noqa: E402
    FeedbackWidget,
    OrgSelectorWidget,
    ServiceManagerWidget,
    WorkspaceSelectorWidget,
)

__all__ = [
    "FeedbackWidget",
    "OrgSelectorWidget",
    "ServiceManagerWidget",
    "WorkspaceSelectorWidget",
]
