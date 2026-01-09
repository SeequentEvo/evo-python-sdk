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

"""Typed access for block models.

A BlockModel is a Geoscience Object that references a block model stored in the
Block Model Service. It acts as a proxy, providing typed access to the block model's
geometry, attributes, and data through the Block Model Service API.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from uuid import UUID

import pandas as pd
from pydantic import TypeAdapter

from evo.common import IContext, IFeedback
from evo.common.utils import NoFeedback

from evo.objects import SchemaVersion

from .base import BaseSpatialObject, BaseSpatialObjectData, ConstructableObject
from ._property import SchemaProperty
from .types import BoundingBox, EpsgCode, Point3, Size3d, Size3i

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

if TYPE_CHECKING:
    from evo.blockmodels import BlockModelAPIClient
    from evo.blockmodels.data import BlockModel as BlockModelMetadata, Version

__all__ = [
    "BlockModel",
    "BlockModelAttribute",
    "BlockModelData",
    "BlockModelGeometry",
    "RegularBlockModelData",
]


@dataclass(frozen=True, kw_only=True)
class BlockModelGeometry:
    """The geometry definition of a regular block model."""

    model_type: str
    origin: Point3
    n_blocks: Size3i
    block_size: Size3d
    rotation: tuple[float, float, float] | None = None


@dataclass(frozen=True, kw_only=True)
class BlockModelAttribute:
    """An attribute on a block model."""

    name: str
    attribute_type: str
    block_model_column_uuid: UUID | None = None
    unit: str | None = None


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
    :param crs: Coordinate reference system (e.g., "EPSG:28354").
    :param size_unit_id: Unit for block sizes (e.g., "m").
    :param units: Dictionary mapping column names to unit IDs.
    """

    name: str
    origin: Point3
    n_blocks: Size3i
    block_size: Size3d
    cell_data: pd.DataFrame | None = None
    description: str | None = None
    crs: str | None = None
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
        geom = self.geometry
        return BoundingBox(
            min_x=geom.origin.x,
            max_x=geom.origin.x + geom.n_blocks.nx * geom.block_size.dx,
            min_y=geom.origin.y,
            max_y=geom.origin.y + geom.n_blocks.ny * geom.block_size.dy,
            min_z=geom.origin.z,
            max_z=geom.origin.z + geom.n_blocks.nz * geom.block_size.dz,
        )


def _parse_geometry(geometry_dict: dict) -> BlockModelGeometry:
    """Parse geometry from the schema format."""
    model_type = geometry_dict.get("model_type", "regular")
    origin = geometry_dict.get("origin", [0, 0, 0])
    n_blocks = geometry_dict.get("n_blocks", [1, 1, 1])
    block_size = geometry_dict.get("block_size", [1, 1, 1])
    rotation = geometry_dict.get("rotation")

    rotation_tuple = None
    if rotation:
        rotation_tuple = (
            rotation.get("dip_azimuth", 0),
            rotation.get("dip", 0),
            rotation.get("pitch", 0),
        )

    return BlockModelGeometry(
        model_type=model_type,
        origin=Point3(x=origin[0], y=origin[1], z=origin[2]),
        n_blocks=Size3i(nx=n_blocks[0], ny=n_blocks[1], nz=n_blocks[2]),
        block_size=Size3d(dx=block_size[0], dy=block_size[1], dz=block_size[2]),
        rotation=rotation_tuple,
    )


def _serialize_geometry(geometry: BlockModelGeometry) -> dict:
    """Serialize geometry to the schema format."""
    result = {
        "model_type": geometry.model_type,
        "origin": [geometry.origin.x, geometry.origin.y, geometry.origin.z],
        "n_blocks": [geometry.n_blocks.nx, geometry.n_blocks.ny, geometry.n_blocks.nz],
        "block_size": [geometry.block_size.dx, geometry.block_size.dy, geometry.block_size.dz],
    }
    if geometry.rotation:
        result["rotation"] = {
            "dip_azimuth": geometry.rotation[0],
            "dip": geometry.rotation[1],
            "pitch": geometry.rotation[2],
        }
    return result


def _parse_attributes(attributes_list: list[dict]) -> list[BlockModelAttribute]:
    """Parse attributes from the schema format."""
    result = []
    for attr in attributes_list:
        col_uuid = attr.get("block_model_column_uuid")
        result.append(
            BlockModelAttribute(
                name=attr.get("name", ""),
                attribute_type=attr.get("attribute_type", "Float64"),
                block_model_column_uuid=UUID(col_uuid) if col_uuid else None,
                unit=attr.get("unit"),
            )
        )
    return result


def _serialize_attributes(attributes: list[BlockModelAttribute]) -> list[dict]:
    """Serialize attributes to the schema format."""
    result = []
    for attr in attributes:
        attr_dict = {
            "name": attr.name,
            "attribute_type": attr.attribute_type,
        }
        if attr.block_model_column_uuid:
            attr_dict["block_model_column_uuid"] = str(attr.block_model_column_uuid)
        if attr.unit:
            attr_dict["unit"] = attr.unit
        result.append(attr_dict)
    return result


class BlockModel(BaseSpatialObject, ConstructableObject[BlockModelData]):
    """A GeoscienceObject representing a block model.

    This object acts as a proxy, allowing you to access block model data and attributes
    through the Block Model Service while the reference itself is stored as a Geoscience Object.

    Example usage:

        # Create a new regular block model
        data = RegularBlockModelData(
            name="My Block Model",
            origin=Point3(x=0, y=0, z=0),
            n_blocks=Size3i(nx=10, ny=10, nz=5),
            block_size=Size3d(dx=2.5, dy=5.0, dz=5.0),
            cell_data=my_dataframe,
        )
        bm = await BlockModel.create_regular(context, data)

        # Get an existing block model
        bm = await BlockModel.from_reference(context, reference)

        # Access geometry
        print(f"Origin: {bm.geometry.origin}")
        print(f"Size: {bm.geometry.n_blocks}")

        # Access data through the Block Model Service
        df = await bm.get_data(columns=["*"])

        # Create a new attribute on the block model
        await bm.add_attribute(data_df, "new_attribute")
    """

    _data_class = BlockModelData

    sub_classification = "block-model"
    creation_schema_version = SchemaVersion(major=1, minor=0, patch=0)

    # Schema properties
    block_model_uuid: UUID = SchemaProperty(
        "block_model_uuid",
        TypeAdapter(UUID),
    )

    block_model_version_uuid: UUID | None = SchemaProperty(
        "block_model_version_uuid",
        TypeAdapter(UUID | None),
    )

    _geometry_raw: dict = SchemaProperty(
        "geometry",
        TypeAdapter(dict),
    )

    _attributes_raw: list[dict] = SchemaProperty(
        "attributes",
        TypeAdapter(list[dict]),
        default_factory=list,
    )

    @property
    def geometry(self) -> BlockModelGeometry:
        """The geometry definition of the block model."""
        return _parse_geometry(self._geometry_raw)

    @property
    def attributes(self) -> list[BlockModelAttribute]:
        """The attributes available on this block model."""
        return _parse_attributes(self._attributes_raw)

    def get_attribute(self, name: str) -> BlockModelAttribute | None:
        """Get an attribute by name.

        :param name: The name of the attribute.
        :return: The attribute, or None if not found.
        """
        for attr in self.attributes:
            if attr.name == name:
                return attr
        return None

    def _get_block_model_client(self) -> BlockModelAPIClient:
        """Get a BlockModelAPIClient for the current context."""
        from evo.blockmodels import BlockModelAPIClient

        return BlockModelAPIClient.from_context(self._context)

    async def get_block_model_metadata(self) -> BlockModelMetadata:
        """Get the full block model metadata from the Block Model Service.

        :return: The BlockModel metadata from the Block Model Service.
        """
        client = self._get_block_model_client()
        return await client.get_block_model(self.block_model_uuid)

    async def get_versions(self) -> list[Version]:
        """Get all versions of this block model.

        :return: List of versions, ordered from newest to oldest.
        """
        client = self._get_block_model_client()
        return await client.list_versions(self.block_model_uuid)

    async def get_data(
        self,
        columns: list[str] | None = None,
        version_uuid: UUID | None = None,
        fb: IFeedback = NoFeedback,
    ) -> pd.DataFrame:
        """Get block model data as a DataFrame.

        :param columns: List of column names to retrieve. Defaults to all columns ["*"].
        :param version_uuid: Specific version to query. Defaults to the referenced version
            or latest if no version is referenced.
        :param fb: Optional feedback interface for progress reporting.
        :return: DataFrame containing the block model data with user-friendly column names.
        """
        from evo.blockmodels.endpoints.models import ColumnHeaderType

        client = self._get_block_model_client()

        fb.progress(0.0, "Querying block model data...")

        # Use referenced version if no specific version requested
        if version_uuid is None:
            version_uuid = self.block_model_version_uuid

        # Default to all columns
        if columns is None:
            columns = ["*"]

        table = await client.query_block_model_as_table(
            bm_id=self.block_model_uuid,
            columns=columns,
            version_uuid=version_uuid,
            column_headers=ColumnHeaderType.name,  # Use column titles, not UUIDs
        )

        fb.progress(0.9, "Converting data...")

        result = table.to_pandas()

        fb.progress(1.0, "Data retrieved")
        return result

    async def add_attribute(
        self,
        data: pd.DataFrame,
        attribute_name: str,
        unit: str | None = None,
        fb: IFeedback = NoFeedback,
    ) -> Version:
        """Add a new attribute to the block model.

        The DataFrame must contain geometry columns (i, j, k) or (x, y, z) and the
        attribute column to add.

        :param data: DataFrame containing geometry columns and the new attribute.
        :param attribute_name: Name of the attribute column in the DataFrame to add.
        :param unit: Optional unit ID for the attribute (must be a valid unit ID from the Block Model Service).
        :param fb: Optional feedback interface for progress reporting.
        :return: The new version created by adding the attribute.
        """
        from evo.blockmodels.typed._utils import dataframe_to_pyarrow

        client = self._get_block_model_client()

        fb.progress(0.0, "Preparing attribute data...")

        # Convert to PyArrow table with proper uint32 casting for i,j,k
        table = dataframe_to_pyarrow(data)

        fb.progress(0.2, "Uploading attribute...")

        units = {attribute_name: unit} if unit else None
        version = await client.add_new_columns(
            bm_id=self.block_model_uuid,
            data=table,
            units=units,
        )

        fb.progress(1.0, "Attribute added")
        return version

    async def update_attributes(
        self,
        data: pd.DataFrame,
        new_columns: list[str] | None = None,
        update_columns: set[str] | None = None,
        delete_columns: set[str] | None = None,
        units: dict[str, str] | None = None,
        fb: IFeedback = NoFeedback,
    ) -> Version:
        """Update attributes on the block model.

        :param data: DataFrame containing geometry columns and attribute data.
        :param new_columns: List of new column names to add.
        :param update_columns: Set of existing column names to update.
        :param delete_columns: Set of column names to delete.
        :param units: Dictionary mapping column names to unit IDs (must be valid unit IDs from the Block Model Service).
        :param fb: Optional feedback interface for progress reporting.
        :return: The new version created by the update.
        """
        from evo.blockmodels.typed._utils import dataframe_to_pyarrow

        client = self._get_block_model_client()

        fb.progress(0.0, "Preparing update...")

        # Convert to PyArrow table with proper uint32 casting for i,j,k
        table = dataframe_to_pyarrow(data)

        fb.progress(0.2, "Uploading changes...")

        version = await client.update_block_model_columns(
            bm_id=self.block_model_uuid,
            data=table,
            new_columns=new_columns or [],
            update_columns=update_columns,
            delete_columns=delete_columns,
            units=units,
        )

        fb.progress(1.0, "Attributes updated")
        return version

    @classmethod
    async def _data_to_dict(cls, data: BlockModelData, context: IContext) -> dict[str, Any]:
        """Convert BlockModelData to a dictionary for creating the Geoscience Object."""
        if cls.creation_schema_version is None:
            raise NotImplementedError("creation_schema_version must be defined")

        result: dict[str, Any] = {
            "schema": f"/objects/block-model/{cls.creation_schema_version}/block-model.schema.json",
            "name": data.name,
            "block_model_uuid": str(data.block_model_uuid),
            "geometry": _serialize_geometry(data.geometry),
        }

        if data.description:
            result["description"] = data.description

        if data.block_model_version_uuid:
            result["block_model_version_uuid"] = str(data.block_model_version_uuid)

        if data.coordinate_reference_system:
            if isinstance(data.coordinate_reference_system, EpsgCode):
                result["coordinate_reference_system"] = {"epsg_code": int(data.coordinate_reference_system)}
            else:
                result["coordinate_reference_system"] = {"ogc_wkt": data.coordinate_reference_system}

        if data.attributes:
            result["attributes"] = _serialize_attributes(data.attributes)

        # Compute and set bounding box
        bbox = data.compute_bounding_box()
        result["bounding_box"] = {
            "min_x": bbox.min_x,
            "max_x": bbox.max_x,
            "min_y": bbox.min_y,
            "max_y": bbox.max_y,
            "min_z": bbox.min_z,
            "max_z": bbox.max_z,
        }

        return result

    @classmethod
    async def create_regular(
        cls,
        context: IContext,
        data: RegularBlockModelData,
        path: str | None = None,
        fb: IFeedback = NoFeedback,
    ) -> Self:
        """Create a new regular block model.

        This creates a block model in the Block Model Service and a corresponding
        Geoscience Object reference.

        :param context: The context containing environment, connector, and cache.
        :param data: The data defining the regular block model to create.
        :param path: Optional path for the Geoscience Object.
        :param fb: Optional feedback interface for progress reporting.
        :return: A new BlockModel instance.
        """
        from evo.blockmodels import RegularBlockModel as BMRegularBlockModel
        from evo.blockmodels import RegularBlockModelData as BMRegularBlockModelData
        from evo.blockmodels.typed import Point3 as BMPoint3, Size3i as BMSize3i, Size3d as BMSize3d
        from evo.objects import ObjectReference

        fb.progress(0.0, "Creating block model...")

        # Convert to evo-blockmodels data format
        bm_data = BMRegularBlockModelData(
            name=data.name,
            description=data.description,
            origin=BMPoint3(data.origin.x, data.origin.y, data.origin.z),
            n_blocks=BMSize3i(data.n_blocks.nx, data.n_blocks.ny, data.n_blocks.nz),
            block_size=BMSize3d(data.block_size.dx, data.block_size.dy, data.block_size.dz),
            cell_data=data.cell_data,
            crs=data.crs,
            size_unit_id=data.size_unit_id,
            units=data.units,
        )

        # Create the block model via Block Model Service
        bm = await BMRegularBlockModel.create(context, bm_data, path=path)

        fb.progress(0.6, "Loading block model reference...")

        # Load the Geoscience Object that was created
        goose_id = bm.metadata.geoscience_object_id
        if goose_id is None:
            raise RuntimeError("Block model was created but geoscience_object_id is not set")

        object_ref = ObjectReference.new(
            environment=context.get_environment(),
            object_id=goose_id,
        )

        result = await cls.from_reference(context, object_ref)

        fb.progress(1.0, "Block model created")
        return result

    @classmethod
    async def from_block_model(
        cls,
        context: IContext,
        block_model_uuid: UUID,
        name: str | None = None,
        version_uuid: UUID | None = None,
        path: str | None = None,
        fb: IFeedback = NoFeedback,
    ) -> Self:
        """Create a BlockModel from an existing block model in the Block Model Service.

        This fetches the block model metadata from the Block Model Service and creates
        a corresponding Geoscience Object reference.

        :param context: The context containing environment, connector, and cache.
        :param block_model_uuid: UUID of the block model in the Block Model Service.
        :param name: Optional name for the reference object. Defaults to the block model name.
        :param version_uuid: Optional specific version to reference.
        :param path: Optional path for the Geoscience Object.
        :param fb: Optional feedback interface for progress reporting.
        :return: A new BlockModel instance.
        """
        from evo.blockmodels import BlockModelAPIClient
        from evo.blockmodels.data import RegularGridDefinition

        client = BlockModelAPIClient.from_context(context)

        fb.progress(0.0, "Fetching block model metadata...")

        # Get block model metadata
        bm = await client.get_block_model(block_model_uuid)

        fb.progress(0.3, "Fetching version information...")

        # Get version info if not specified
        if version_uuid is None:
            versions = await client.list_versions(block_model_uuid)
            if versions:
                version_uuid = versions[0].version_uuid

        fb.progress(0.5, "Creating reference object...")

        # Build geometry from the block model
        grid_def = bm.grid_definition
        if not isinstance(grid_def, RegularGridDefinition):
            raise ValueError(f"Only regular block models are supported, got {type(grid_def).__name__}")

        rotation_tuple = None
        if grid_def.rotations:
            # Convert rotations to (dip_azimuth, dip, pitch) - simplified
            rotation_tuple = (0.0, 0.0, 0.0)  # Default, would need proper conversion

        geometry = BlockModelGeometry(
            model_type="regular",
            origin=Point3(x=grid_def.model_origin[0], y=grid_def.model_origin[1], z=grid_def.model_origin[2]),
            n_blocks=Size3i(nx=grid_def.n_blocks[0], ny=grid_def.n_blocks[1], nz=grid_def.n_blocks[2]),
            block_size=Size3d(dx=grid_def.block_size[0], dy=grid_def.block_size[1], dz=grid_def.block_size[2]),
            rotation=rotation_tuple,
        )

        # Build attributes from version info
        attributes: list[BlockModelAttribute] = []
        if version_uuid:
            versions = await client.list_versions(block_model_uuid)
            version = next((v for v in versions if v.version_uuid == version_uuid), None)
            if version and version.columns:
                for col in version.columns:
                    # Try to parse col_id as UUID, but it might not be valid for system columns
                    col_uuid = None
                    if col.col_id:
                        try:
                            col_uuid = UUID(col.col_id)
                        except ValueError:
                            # Not a valid UUID (e.g., system column), skip
                            pass
                    attributes.append(
                        BlockModelAttribute(
                            name=col.title,
                            attribute_type=col.data_type.value if col.data_type else "Float64",
                            block_model_column_uuid=col_uuid,
                        )
                    )

        # Determine CRS
        crs: EpsgCode | str | None = None
        if bm.coordinate_reference_system:
            if bm.coordinate_reference_system.startswith("EPSG:"):
                try:
                    crs = EpsgCode(int(bm.coordinate_reference_system.split(":")[1]))
                except ValueError:
                    crs = bm.coordinate_reference_system
            else:
                crs = bm.coordinate_reference_system

        ref_data = BlockModelData(
            name=name or bm.name,
            block_model_uuid=block_model_uuid,
            block_model_version_uuid=version_uuid,
            geometry=geometry,
            attributes=attributes,
            coordinate_reference_system=crs,
        )

        fb.progress(0.8, "Saving reference...")

        result = await cls.create(context, ref_data, path=path)

        fb.progress(1.0, "Block model reference created")
        return result

