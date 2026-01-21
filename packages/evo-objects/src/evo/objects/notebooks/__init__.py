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

"""Ellipsoid and variogram data generation for visualization.

This module provides functions to generate mesh/wireframe data for 3D ellipsoids
and curve data for 2D variogram plots. The output is numpy arrays that can be
used with any plotting library (Plotly, K3D, matplotlib, etc.).

No external plotting dependencies are required - just numpy.

Example with Plotly:
    >>> from evo.objects.notebooks import generate_ellipsoid_wireframe
    >>> import plotly.graph_objects as go
    >>>
    >>> data = generate_ellipsoid_wireframe(ranges=(200, 150, 100), rotation=(45, 30, 0))
    >>> fig = go.Figure(data=[go.Scatter3d(x=data.x, y=data.y, z=data.z, mode="lines")])
    >>> fig.show()

Example with K3D:
    >>> from evo.objects.notebooks import generate_ellipsoid_wireframe
    >>> import k3d
    >>> import numpy as np
    >>>
    >>> data = generate_ellipsoid_wireframe(ranges=(200, 150, 100))
    >>> mask = ~np.isnan(data.x)
    >>> vertices = np.column_stack([data.x[mask], data.y[mask], data.z[mask]]).astype(np.float32)
    >>> plot = k3d.plot()
    >>> plot += k3d.line(vertices, color=0x0000ff, width=0.5)
    >>> plot.display()
"""

from .ellipsoid import (
    Ellipsoid,
    EllipsoidData,
    VariogramCurveData,
    generate_ellipsoid_mesh,
    generate_ellipsoid_wireframe,
    generate_variogram_curves,
)

__all__ = [
    "Ellipsoid",
    "EllipsoidData",
    "VariogramCurveData",
    "generate_ellipsoid_mesh",
    "generate_ellipsoid_wireframe",
    "generate_variogram_curves",
]
