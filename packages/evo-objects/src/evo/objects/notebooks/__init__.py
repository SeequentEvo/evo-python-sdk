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

"""Notebook visualization tools for geoscience objects.

This module provides interactive visualization functions for use in Jupyter notebooks.
Requires the 'viz' optional dependency: pip install evo-objects[viz]
"""

from .variogram_plot import (
    plot_variogram,
    plot_variogram_2d,
    plot_variogram_3d,
    plot_variogram_ellipsoids,
    plot_variogram_model,
)

__all__ = [
    "plot_variogram",
    "plot_variogram_2d",
    "plot_variogram_3d",
    "plot_variogram_ellipsoids",
    "plot_variogram_model",
]
