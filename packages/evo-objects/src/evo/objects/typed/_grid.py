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

"""Base classes for 3D grid objects."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Annotated
from uuid import UUID

import pandas as pd

from evo.common import IFeedback
from evo.common.utils import NoFeedback

from ._model import DataLocation, SchemaLocation, SchemaModel
from .attributes import Attributes, BlockModelAttribute
from .exceptions import ObjectValidationError
from .spatial import BaseSpatialObject, BaseSpatialObjectData
from .types import BoundingBox, Point3, Rotation, Size3d, Size3i

__all__ = [
    "Base3DGrid",
    "Base3DGridData",
    "BaseRegular3DGrid",
    "BaseRegular3DGridData",
    "BlockModelData",
    "BlockModelGeometry",
    "RegularBlockModelData",
]


@dataclass(kw_only=True, frozen=True)
class Base3DGridData(BaseSpatialObjectData):
    """Base class for all 3D grid data.

    Contains the common properties shared by all grid types: origin, size, rotation, and cell_data.
    """

    origin: Point3
    size: Size3i
    cell_data: pd.DataFrame | None = None
    rotation: Rotation | None = None


@dataclass(kw_only=True, frozen=True)
class BaseRegular3DGridData(Base3DGridData):
    """Base class for regular 3D grid data (both masked and non-masked).

    Contains the common properties shared by Regular3DGridData and RegularMasked3DGridData.
    Adds cell_size to the base grid properties.
    """

    cell_size: Size3d

    def compute_bounding_box(self) -> BoundingBox:
        return BoundingBox.from_regular_grid(self.origin, self.size, self.cell_size, self.rotation)


class Base3DGrid(BaseSpatialObject, ABC):
    """Base class for all 3D grid objects.

    Contains the common properties shared by all grid types: origin, size, and rotation.
    The bounding box is dynamically computed from the grid properties.
    """

    origin: Annotated[Point3, SchemaLocation("origin")]
    size: Annotated[Size3i, SchemaLocation("size")]
    rotation: Annotated[Rotation | None, SchemaLocation("rotation")]

    @abstractmethod
    def compute_bounding_box(self) -> BoundingBox:
        """Compute the bounding box for the grid."""
        # This class does not have enough information about the grid cell sizes to compute the bounding box.
        raise NotImplementedError(
            "Subclasses must implement compute_bounding_box to derive bounding box from grid properties."
        )

    @property
    def bounding_box(self) -> BoundingBox:
        return self.compute_bounding_box()

    @bounding_box.setter
    def bounding_box(self, value: BoundingBox) -> None:
        raise AttributeError("Cannot set bounding_box on this object, as it is dynamically derived from the data.")

    async def update(self):
        """Update the object on the geoscience object service, including recomputing the bounding box."""
        self._bounding_box = self.compute_bounding_box()
        await super().update()


class BaseRegular3DGrid(Base3DGrid):
    """Base class for regular 3D grid objects (both masked and non-masked).

    Contains the common properties shared by Regular3DGrid and RegularMasked3DGrid.
    Adds cell_size to the base grid properties.
    """

    cell_size: Annotated[Size3d, SchemaLocation("cell_size")]

    def compute_bounding_box(self) -> BoundingBox:
        return BoundingBox.from_regular_grid(self.origin, self.size, self.cell_size, self.rotation)


class Cells3D(SchemaModel):
    """A dataset representing the cells of a non-masked 3D grid.

    The order of the cells are in column-major order, i.e. for a unrotated grid: x changes fastest, then y, then z.
    """

    _grid_size: Annotated[Size3i, SchemaLocation("size")]
    attributes: Annotated[Attributes, SchemaLocation("cell_attributes"), DataLocation("cell_data")]

    @property
    def size(self) -> Size3i:
        return self._grid_size

    @property
    def expected_length(self) -> int:
        return self.size.total_size

    async def to_dataframe(self, *keys: str, fb: IFeedback = NoFeedback) -> pd.DataFrame:
        """Load a DataFrame containing the cell attribute values.

        :param keys: Optional list of attribute keys to include. If not provided, all attributes are included.
        :param fb: Optional feedback object to report download progress.
        :return: A DataFrame with cell attribute columns.
        """
        return await self.attributes.to_dataframe(*keys, fb=fb)

    async def from_dataframe(self, df: pd.DataFrame, fb: IFeedback = NoFeedback) -> None:
        """Set the cell attributes from a DataFrame."""
        if df.shape[0] != self.expected_length:
            raise ObjectValidationError(
                f"The number of rows in the dataframe ({df.shape[0]}) does not match the number of cells in the grid ({self.expected_length})."
            )
        await self.attributes.set_attributes(df, fb=fb)

    def validate(self) -> None:
        """Validate that all attributes have the correct length."""
        self.attributes.validate_lengths(self.expected_length)


class Vertices3D(SchemaModel):
    """A dataset representing the vertices of a non-masked 3D grid.

    The order of the vertices are in column-major order, i.e. for a unrotated grid: x changes fastest, then y, then z.
    """

    _grid_size: Annotated[Size3i, SchemaLocation("size")]
    attributes: Annotated[Attributes, SchemaLocation("vertex_attributes"), DataLocation("vertex_data")]

    @property
    def size(self) -> Size3i:
        grid_size = self._grid_size
        return Size3i(nx=grid_size.nx + 1, ny=grid_size.ny + 1, nz=grid_size.nz + 1)

    @property
    def expected_length(self) -> int:
        return self.size.total_size

    async def to_dataframe(self, *keys: str, fb: IFeedback = NoFeedback) -> pd.DataFrame:
        """Load a DataFrame containing the vertex attribute values.

        :param keys: Optional list of attribute keys to include. If not provided, all attributes are included.
        :param fb: Optional feedback object to report download progress.
        :return: A DataFrame with vertex attribute columns.
        """
        return await self.attributes.to_dataframe(*keys, fb=fb)

    async def from_dataframe(self, df: pd.DataFrame, fb: IFeedback = NoFeedback) -> None:
        """Set the vertex attributes from a DataFrame."""
        if df.shape[0] != self.expected_length:
            raise ObjectValidationError(
                f"The number of rows in the dataframe ({df.shape[0]}) does not match the number of vertices in the grid ({self.expected_length})."
            )
        await self.attributes.set_attributes(df, fb=fb)

    def validate(self) -> None:
        """Validate that all attributes have the correct length."""
        self.attributes.validate_lengths(self.expected_length)


@dataclass(frozen=True, kw_only=True)
class BlockModelGeometry:
    """The geometry definition of a regular block model."""

    model_type: str
    origin: Point3
    n_blocks: Size3i
    block_size: Size3d
    rotation: Rotation | None = None


@dataclass(frozen=True, kw_only=True)
class RegularBlockModelData:
    """Data for creating a regular block model.

    This creates a new block model in the Block Model Service and a corresponding
    Geoscience Object reference.

    :param name: The name of the block model.
    :param origin: The origin point (x, y, z) of the block model.
    :param n_blocks: The number of blocks in each dimension (nx, ny, nz).
    :param block_size: The size of each block (dx, dy, dz).
    :param cell_data: DataFrame with block data. Must contain (x, y, z) or (i, j, k) columns.
    :param description: Optional description.
    :param coordinate_reference_system: Coordinate reference system (e.g., "EPSG:28354").
    :param size_unit_id: Unit for block sizes (e.g., "m").
    :param units: Dictionary mapping column names to unit IDs.
    """

    name: str
    origin: Point3
    n_blocks: Size3i
    block_size: Size3d
    cell_data: pd.DataFrame | None = None
    description: str | None = None
    coordinate_reference_system: str | None = None
    size_unit_id: str | None = None
    units: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class BlockModelData(BaseSpatialObjectData):
    """Data for creating a BlockModel reference.

    A BlockModel is a reference to a block model stored in the Block Model Service.
    This creates a Geoscience Object that points to an existing block model.

    :param name: The name of the block model reference object.
    :param block_model_uuid: The UUID of the block model in the Block Model Service.
    :param block_model_version_uuid: Optional specific version UUID to reference.
    :param geometry: The geometry definition of the block model.
    :param attributes: List of attributes available on the block model.
    :param coordinate_reference_system: Optional CRS for the block model.
    """

    block_model_uuid: UUID
    block_model_version_uuid: UUID | None = None
    geometry: BlockModelGeometry
    attributes: list[BlockModelAttribute] = field(default_factory=list)

    def compute_bounding_box(self) -> BoundingBox:
        """Compute the bounding box from the geometry."""
        return BoundingBox.from_regular_grid(
            self.geometry.origin, self.geometry.n_blocks, self.geometry.block_size, self.geometry.rotation
        )
