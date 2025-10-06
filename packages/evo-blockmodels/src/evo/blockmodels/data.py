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

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from evo.common import ResourceMetadata
from evo.workspaces import ServiceUser

from .endpoints.models import BBox, BBoxXYZ, Column, RotationAxis

__all__ = ["BaseGridDefinition", "BlockModel", "RegularGridDefinition", "Version"]


@dataclass(frozen=True, kw_only=True)
class BaseGridDefinition:
    """Base class for grid definitions."""

    model_origin: list[float]
    """Origin of the block model."""
    rotations: list[tuple[RotationAxis, float]]
    """Rotation of the block model, as a list of intrinsic rotations.

    The angle is in degrees, with positive angles indicating a clockwise rotation around the axis, when looking down the
    axis towards the negative direction.
    """

    def __post_init__(self):
        if len(self.model_origin) != 3:
            raise ValueError("model_origin must have 3 elements")
        if len(self.rotations) > 3:
            raise ValueError("rotations must have less than or equal to 3 elements")


@dataclass(frozen=True, kw_only=True)
class RegularGridDefinition(BaseGridDefinition):
    """A regular grid definition, defining the grid's origin, block size, number of block, and orientation."""

    n_blocks: list[int]
    """Number of blocks in the grid along each axis."""
    block_size: list[float]
    """Size of the blocks along each axis."""

    def __post_init__(self):
        super().__post_init__()
        if len(self.n_blocks) != 3:
            raise ValueError("n_blocks must have 3 elements")
        if len(self.block_size) != 3:
            raise ValueError("block_size must have 3 elements")


@dataclass(frozen=True, kw_only=True)
class BlockModel(ResourceMetadata):
    geoscience_object_id: UUID | None
    """
    UUID of the Geoscience Object Service object associated with the block model
    """

    description: str | None
    """
    Description of the block model.
    """

    grid_definition: BaseGridDefinition
    """
    Definition of the block model grid.
    """

    coordinate_reference_system: str | None
    """
    Coordinate reference system used in the block model.
    """

    size_unit_id: str | None
    """
    Unit ID denoting the length unit used for the block model's blocks.
    """

    bbox: BBoxXYZ
    """
    Axis-aligned bounding box of the block model.

    This is the smallest box that fully contains all blocks within the block model, regardless of whether they contain data.
    It is defined by the minimum and maximum coordinates along each axis.
    """

    last_updated_at: datetime
    """
    Date and time of the last block model update, including metadata updates
    """

    last_updated_by: ServiceUser
    """
    User who last updated the block model, including metadata updates
    """

    @property
    def url(self) -> str:
        """The url of the block model version."""
        return "{hub_url}/blockmodel/orgs/{org_id}/workspaces/{workspace_id}/block-models/{bm_id}".format(
            hub_url=self.environment.hub_url.rstrip("/"),
            org_id=self.environment.org_id,
            workspace_id=self.environment.workspace_id,
            bm_id=self.id,
        )


@dataclass(frozen=True, kw_only=True)
class Version:
    bm_uuid: UUID
    """UUID of the block model this version belongs to."""

    version_id: int
    """
    Identifier for the version within a block model as a monotonically increasing integer, where 1 is
    the `version_id` for the version created upon creation of the block model.
    """

    version_uuid: UUID
    """
    A universally unique identifier for the version
    """

    parent_version_id: int
    """
    Previous version. 0 if this is the first version.
    """

    base_version_id: int | None
    """
    Version the update was applied to. This will be the same as `parent_version_id`, except for
    updates made by Leapfrog, where it is the current local version when the block model is published. This is None if this
    is the first version.
    """

    geoscience_version_id: str | None
    """
    ID of the Geoscience Object Service object version associated with this block model version
    """

    created_at: datetime
    """
    User who performed the action that created the version
    """

    created_by: ServiceUser
    """
    User who performed the action that created the version
    """

    comment: str
    """
    User-supplied comment for this version
    """

    bbox: BBox | None = None
    """
    Bounding box of data updated between this version and last version. Will be None for the initial version, and updates
    that only delete and rename columns.
    """

    columns: list[Column]
    """
    Columns within this version
    """
