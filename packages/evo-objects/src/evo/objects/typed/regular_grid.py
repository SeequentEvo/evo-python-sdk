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

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

import pandas as pd

from evo.common.interfaces import IFeedback
from evo.common.utils import NoFeedback
from evo.objects import SchemaVersion

from ._grid import BaseRegular3DGrid, BaseRegular3DGridData, Cells3D, Vertices3D
from ._model import SchemaLocation
from .attributes import Attributes
from .exceptions import ObjectValidationError

__all__ = [
    "Regular3DGrid",
    "Regular3DGridData",
]


@dataclass(kw_only=True, frozen=True)
class Regular3DGridData(BaseRegular3DGridData):
    vertex_data: pd.DataFrame | None = None

    def __post_init__(self):
        if self.cell_data is not None and self.cell_data.shape[0] != self.size.total_size:
            raise ObjectValidationError(
                f"The number of rows in the cell_data dataframe ({self.cell_data.shape[0]}) does not match the number of cells in the grid ({self.size.nx * self.size.ny * self.size.nz})."
            )
        vertices_expected_length = (self.size.nx + 1) * (self.size.ny + 1) * (self.size.nz + 1)
        if self.vertex_data is not None and self.vertex_data.shape[0] != vertices_expected_length:
            raise ObjectValidationError(
                f"The number of rows in the vertex_data dataframe ({self.vertex_data.shape[0]}) does not match the number of vertices in the grid ({self.size.nx * self.size.ny * self.size.nz})."
            )


class Regular3DGrid(BaseRegular3DGrid):
    """A GeoscienceObject representing a regular 3D grid.

    The object contains a dataset for both the cells and the vertices of the grid.

    Each of these datasets only contain attribute columns. The actual geometry of the grid is defined by
    the properties: origin, size, cell_size, and rotation.
    """

    _data_class = Regular3DGridData

    sub_classification = "regular-3d-grid"
    creation_schema_version = SchemaVersion(major=1, minor=3, patch=0)

    cells: Annotated[Cells3D, SchemaLocation("")]
    vertices: Annotated[Vertices3D, SchemaLocation("")]

    @property
    def attributes(self) -> "Attributes":
        """The cell attributes of this grid (alias for cells.attributes)."""
        return self.cells.attributes

    @property
    def vertex_attributes(self) -> "Attributes":
        """The vertex attributes of this grid."""
        return self.vertices.attributes

    async def to_dataframe(self, *keys: str, fb: IFeedback = NoFeedback) -> pd.DataFrame:
        """Get a dataframe containing the cell attributes.

        :param keys: Optional list of attribute keys to include. If not provided, all attributes are included.
        :param fb: Optional feedback object to report download progress.
        :return: A DataFrame with cell attribute columns.
        """
        return await self.cells.to_dataframe(*keys, fb=fb)
