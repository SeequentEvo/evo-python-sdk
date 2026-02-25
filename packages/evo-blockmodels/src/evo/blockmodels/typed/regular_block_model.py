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

"""Typed access for regular block models."""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID

import pandas as pd

from evo.common import IContext, IFeedback, StaticContext
from evo.common.utils import NoFeedback

from ..client import BlockModelAPIClient
from ..data import BlockModel, RegularGridDefinition, Version
from ..endpoints.models import BBox, BBoxXYZ, RotationAxis
from ._utils import dataframe_to_pyarrow
from .base import BaseTypedBlockModel
from .types import Point3, Size3d, Size3i

__all__ = [
    "RegularBlockModel",
    "RegularBlockModelData",
]


@dataclass(frozen=True, kw_only=True)
class RegularBlockModelData:
    """Data class for creating a new regular block model.

    :param name: The name of the block model.
    :param origin: The origin point of the block model grid.
    :param n_blocks: The number of blocks in each dimension (nx, ny, nz).
    :param block_size: The size of each block in each dimension (dx, dy, dz).
    :param rotations: List of rotations as (axis, angle) tuples. Angle is in degrees,
        positive angles indicate clockwise rotation when looking down the axis.
    :param cell_data: Optional DataFrame containing block attribute data.
        Must include geometry columns (i, j, k) or (x, y, z) and attribute columns.
    :param description: Optional description of the block model.
    :param coordinate_reference_system: Optional coordinate reference system (e.g., "EPSG:4326").
    :param size_unit_id: Optional unit identifier for block sizes (e.g., "m").
    :param units: Optional dictionary mapping column names to unit identifiers.
    """

    name: str
    origin: Point3
    n_blocks: Size3i
    block_size: Size3d
    rotations: list[tuple[RotationAxis, float]] = field(default_factory=list)
    cell_data: pd.DataFrame | None = None
    description: str | None = None
    coordinate_reference_system: str | None = None
    size_unit_id: str | None = None
    units: dict[str, str] = field(default_factory=dict)


class RegularBlockModel(BaseTypedBlockModel):
    """A typed wrapper for regular block models providing pandas DataFrame access.

    This class provides a high-level interface for creating, retrieving, and updating
    regular block models with typed access to grid properties and cell data.

    Example usage:

        # Create a new block model
        data = RegularBlockModelData(
            name="My Block Model",
            origin=Point3(0, 0, 0),
            n_blocks=Size3i(10, 10, 10),
            block_size=Size3d(1.0, 1.0, 1.0),
            cell_data=my_dataframe,
        )
        block_model = await RegularBlockModel.create(context, data)

        # Retrieve an existing block model
        block_model = await RegularBlockModel.get(context, bm_id)
        df = block_model.cell_data

        # Update attributes
        new_version = await block_model.update_attributes(
            updated_dataframe,
            new_columns=["new_col"],
        )
    """

    def __init__(
        self,
        client: BlockModelAPIClient,
        metadata: BlockModel,
        version: Version,
        cell_data: pd.DataFrame,
        context: IContext | None = None,
    ) -> None:
        """Initialize a RegularBlockModel instance.

        :param client: The BlockModelAPIClient used for API operations.
        :param metadata: The block model metadata.
        :param version: The current version information.
        :param cell_data: The cell data as a pandas DataFrame.
        :param context: Optional IContext for report and other operations.
        """
        super().__init__(client=client, metadata=metadata, version=version, cell_data=cell_data)
        self._context = context

    @property
    def n_blocks(self) -> Size3i:
        """The number of blocks in each dimension."""
        grid_def = self._metadata.grid_definition
        if not isinstance(grid_def, RegularGridDefinition):
            raise TypeError("Block model is not a regular grid")
        return Size3i(
            nx=grid_def.n_blocks[0],
            ny=grid_def.n_blocks[1],
            nz=grid_def.n_blocks[2],
        )

    @property
    def block_size(self) -> Size3d:
        """The size of each block in each dimension."""
        grid_def = self._metadata.grid_definition
        if not isinstance(grid_def, RegularGridDefinition):
            raise TypeError("Block model is not a regular grid")
        return Size3d(
            dx=grid_def.block_size[0],
            dy=grid_def.block_size[1],
            dz=grid_def.block_size[2],
        )

    @property
    def rotations(self) -> list[tuple[RotationAxis, float]]:
        """The rotations applied to the block model."""
        return list(self._metadata.grid_definition.rotations)

    def _get_context(self) -> IContext:
        """Get the IContext for this block model."""
        if self._context is not None:
            return self._context
        # Build a context from the client's internal state
        return StaticContext.from_environment(
            environment=self._client._environment,
            connector=self._client._connector,
            cache=self._client._cache,
        )

    @classmethod
    async def create(
        cls,
        context: IContext,
        data: RegularBlockModelData,
        path: str | None = None,
        fb: IFeedback = NoFeedback,
    ) -> RegularBlockModel:
        """Create a new regular block model.

        :param context: The context containing environment, connector, and cache.
        :param data: The data defining the block model to create.
        :param path: Optional path for the block model in the workspace.
        :param fb: Optional feedback interface for progress reporting.
        :return: A RegularBlockModel instance representing the created block model.
        :raises ValueError: If the data is invalid.
        """
        client = BlockModelAPIClient.from_context(context)

        # Create the grid definition
        grid_definition = RegularGridDefinition(
            model_origin=[data.origin.x, data.origin.y, data.origin.z],
            rotations=list(data.rotations),
            n_blocks=[data.n_blocks.nx, data.n_blocks.ny, data.n_blocks.nz],
            block_size=[data.block_size.dx, data.block_size.dy, data.block_size.dz],
        )

        fb.progress(0.0, "Creating block model...")

        # Convert DataFrame to PyArrow Table if cell data is provided
        initial_data = None
        if data.cell_data is not None:
            initial_data = dataframe_to_pyarrow(data.cell_data)

        # Create the block model with initial data (if provided)
        bm, version = await client.create_block_model(
            name=data.name,
            description=data.description,
            grid_definition=grid_definition,
            object_path=path,
            coordinate_reference_system=data.coordinate_reference_system,
            size_unit_id=data.size_unit_id,
            initial_data=initial_data,
            units=data.units if data.units else None,
        )

        fb.progress(1.0, "Block model created successfully")

        # Retrieve the cell data (or create empty DataFrame)
        if data.cell_data is not None:
            cell_data = data.cell_data.copy()
        else:
            cell_data = pd.DataFrame()

        return cls(
            client=client,
            metadata=bm,
            version=version,
            cell_data=cell_data,
            context=context,
        )

    @classmethod
    async def get(
        cls,
        context: IContext,
        bm_id: UUID,
        version_id: UUID | None = None,
        columns: list[str] | None = None,
        bbox: BBox | BBoxXYZ | None = None,
        fb: IFeedback = NoFeedback,
    ) -> RegularBlockModel:
        """Retrieve an existing regular block model.

        :param context: The context containing environment, connector, and cache.
        :param bm_id: The UUID of the block model to retrieve.
        :param version_id: Optional version UUID. Defaults to the latest version.
        :param columns: Optional list of columns to retrieve. Defaults to all columns ["*"].
        :param bbox: Optional bounding box to filter the data.
        :param fb: Optional feedback interface for progress reporting.
        :return: A RegularBlockModel instance.
        :raises ValueError: If the block model is not a regular grid.
        """
        client = BlockModelAPIClient.from_context(context)

        fb.progress(0.0, "Retrieving block model metadata...")

        # Get block model metadata
        bm = await client.get_block_model(bm_id)

        # Verify it's a regular grid
        if not isinstance(bm.grid_definition, RegularGridDefinition):
            raise ValueError(f"Block model {bm_id} is not a regular grid. Got {type(bm.grid_definition).__name__}")

        fb.progress(0.2, "Querying block model data...")

        # Default to all columns if not specified
        if columns is None:
            columns = ["*"]

        # Query the block model data
        table = await client.query_block_model_as_table(
            bm_id=bm_id,
            columns=columns,
            bbox=bbox,
            version_uuid=version_id,
        )

        fb.progress(0.8, "Converting data...")

        # Convert to DataFrame
        cell_data = table.to_pandas()

        # Get version information
        versions = await client.list_versions(bm_id)
        if version_id is not None:
            version = next(
                (v for v in versions if v.version_uuid == version_id),
                versions[0] if versions else None,
            )
        else:
            version = versions[0] if versions else None

        fb.progress(1.0, "Block model retrieved successfully")

        return cls(
            client=client,
            metadata=bm,
            version=version,
            cell_data=cell_data,
            context=context,
        )
