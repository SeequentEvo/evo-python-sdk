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
from typing import Annotated, Literal
from uuid import UUID

from evo.common import IContext, IFeedback
from evo.common.utils import NoFeedback
from evo.objects import ObjectReference, SchemaVersion

from . import object_from_uuid
from ._model import SchemaLocation
from .spatial import BaseSpatialObject, BaseSpatialObjectData
from .types import BoundingBox, Point3, Size3d, Size3i

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

# Optional dependency: evo-blockmodels
try:
    import pandas as pd
except ImportError:
    _PD_AVAILABLE = False
else:
    _PD_AVAILABLE = True

try:
    from evo.blockmodels import BlockModelAPIClient
    from evo.blockmodels import RegularBlockModel as BMRegularBlockModel
    from evo.blockmodels import RegularBlockModelData as BMRegularBlockModelData
    from evo.blockmodels.data import BlockModel as BlockModelMetadata
    from evo.blockmodels.data import Version
    from evo.blockmodels.typed import Report, ReportSpecificationData
    from evo.blockmodels.typed.base import BaseTypedBlockModel
except ImportError:
    _BLOCKMODELS_AVAILABLE = False
else:
    _BLOCKMODELS_AVAILABLE = True


__all__ = [
    "BlockModel",
    "BlockModelAttribute",
    "BlockModelAttributes",
    "BlockModelData",
    "BlockModelGeometry",
    "BlockModelPendingAttribute",
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


class BlockModelAttribute:
    """An attribute on a block model.

    This class represents an existing attribute on a block model. It stores a reference
    to the parent BlockModel via `_obj`, similar to how `Attribute` in dataset.py works.
    """

    def __init__(
        self,
        name: str,
        attribute_type: str,
        block_model_column_uuid: UUID | None = None,
        unit: str | None = None,
        obj: "BlockModel | None" = None,
    ):
        self._name = name
        self._attribute_type = attribute_type
        self._block_model_column_uuid = block_model_column_uuid
        self._unit = unit
        self._obj = obj  # Reference to parent BlockModel, similar to Attribute._obj

    @property
    def name(self) -> str:
        """The name of this attribute."""
        return self._name

    @property
    def attribute_type(self) -> str:
        """The type of this attribute."""
        return self._attribute_type

    @property
    def block_model_column_uuid(self) -> UUID | None:
        """The UUID of the column in the block model service."""
        return self._block_model_column_uuid

    @property
    def unit(self) -> str | None:
        """The unit of this attribute."""
        return self._unit

    @property
    def exists(self) -> bool:
        """Whether this attribute exists on the block model.

        :return: True for existing attributes.
        """
        return True

    def __repr__(self) -> str:
        return f"BlockModelAttribute(name={self._name!r}, attribute_type={self._attribute_type!r}, unit={self._unit!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BlockModelAttribute):
            return NotImplemented
        return (
            self._name == other._name
            and self._attribute_type == other._attribute_type
            and self._block_model_column_uuid == other._block_model_column_uuid
            and self._unit == other._unit
        )

    def __hash__(self) -> int:
        return hash((self._name, self._attribute_type, self._block_model_column_uuid, self._unit))


class BlockModelPendingAttribute:
    """A placeholder for an attribute that doesn't exist yet on a Block Model.

    This is returned when accessing an attribute by name that doesn't exist.
    It can be used as a target for compute tasks, which will create the attribute.

    Stores a reference to the parent BlockModel via `_obj`, similar to how
    `BlockModelAttribute` and `Attribute` (in dataset.py) work.
    """

    def __init__(self, obj: "BlockModel", name: str) -> None:
        """
        :param obj: The BlockModel this pending attribute belongs to.
        :param name: The name of the attribute to create.
        """
        self._obj = obj  # Reference to parent BlockModel
        self._name = name

    @property
    def name(self) -> str:
        """The name of this attribute."""
        return self._name

    @property
    def exists(self) -> bool:
        """Whether this attribute exists on the block model.

        :return: False for pending attributes.
        """
        return False

    def __repr__(self) -> str:
        return f"BlockModelPendingAttribute(name={self._name!r}, exists=False)"


class BlockModelAttributes:
    """A collection of attributes on a block model with pretty-printing support."""

    def __init__(self, attributes: list[BlockModelAttribute], block_model: "BlockModel | None" = None):
        self._block_model = block_model
        # Set _obj reference on each attribute to the parent BlockModel
        self._attributes = []
        for attr in attributes:
            # Create a new attribute with _obj reference to the block model
            attr_with_obj = BlockModelAttribute(
                name=attr.name,
                attribute_type=attr.attribute_type,
                block_model_column_uuid=attr.block_model_column_uuid,
                unit=attr.unit,
                obj=block_model,
            )
            self._attributes.append(attr_with_obj)

    def __iter__(self):
        return iter(self._attributes)

    def __len__(self):
        return len(self._attributes)

    def __getitem__(self, index_or_name: int | str) -> BlockModelAttribute | BlockModelPendingAttribute:
        if isinstance(index_or_name, str):
            for attr in self._attributes:
                if attr.name == index_or_name:
                    return attr
            # Return a BlockModelPendingAttribute for non-existent attributes accessed by name
            # Pass the block model directly as _obj
            return BlockModelPendingAttribute(self._block_model, index_or_name)
        return self._attributes[index_or_name]

    def __repr__(self) -> str:
        names = [attr.name for attr in self._attributes]
        return f"BlockModelAttributes({names})"


if _BLOCKMODELS_AVAILABLE and _PD_AVAILABLE:

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
        # Try to parse as UUID, but handle invalid formats gracefully
        parsed_uuid = None
        if col_uuid:
            try:
                parsed_uuid = UUID(col_uuid)
            except (ValueError, AttributeError):
                # col_uuid is not a valid UUID format, skip it
                pass
        result.append(
            BlockModelAttribute(
                name=attr.get("name", ""),
                attribute_type=attr.get("attribute_type", "Float64"),
                block_model_column_uuid=parsed_uuid,
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


class BlockModel(BaseSpatialObject):
    """A GeoscienceObject representing a block model.

    This object acts as a proxy, allowing you to access block model data and attributes
    through the Block Model Service while the reference itself is stored as a Geoscience Object.

    Metadata-only operations (geometry, attributes, name) always work. Data operations
    (to_dataframe, add_attribute, create_report, etc.) require the evo-blockmodels package
    to be installed: ``pip install evo-objects[blockmodels]``

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

        # Access data through the Block Model Service (requires evo-blockmodels)
        df = await bm.to_dataframe(columns=["*"])

        # Create a new attribute on the block model (requires evo-blockmodels)
        await bm.add_attribute(data_df, "new_attribute")
    """

    _data_class = BlockModelData

    sub_classification = "block-model"
    creation_schema_version = SchemaVersion(major=1, minor=0, patch=0)

    # Schema properties
    block_model_uuid: Annotated[UUID, SchemaLocation("block_model_uuid")]

    block_model_version_uuid: Annotated[UUID | None, SchemaLocation("block_model_version_uuid")]

    _geometry_raw: Annotated[dict, SchemaLocation("geometry")]

    _attributes_raw: Annotated[list[dict], SchemaLocation("attributes")] = []

    @property
    def geometry(self) -> BlockModelGeometry:
        """The geometry definition of the block model."""
        return _parse_geometry(self._geometry_raw)

    @property
    def attributes(self) -> BlockModelAttributes:
        """The attributes available on this block model."""
        return BlockModelAttributes(_parse_attributes(self._attributes_raw), block_model=self)

    def get_attribute(self, name: str) -> BlockModelAttribute | None:
        """Get an attribute by name.

        :param name: The name of the attribute.
        :return: The attribute, or None if not found.
        """
        for attr in self.attributes:
            if attr.name == name:
                return attr
        return None

    if _BLOCKMODELS_AVAILABLE and _PD_AVAILABLE:

        def _get_block_model_client(self) -> BlockModelAPIClient:
            """Get a BlockModelAPIClient for the current context."""
            return BlockModelAPIClient.from_context(self._api_context)

        async def _get_or_create_typed_block_model(self) -> BaseTypedBlockModel:
            """Lazily create a typed block model delegate for data operations.

            All data operations are delegated to a BaseTypedBlockModel instance
            (currently RegularBlockModel), avoiding code duplication between the
            reference object and typed objects in evo-blockmodels.
            """
            if not hasattr(self, "_typed_bm") or self._typed_bm is None:
                client = self._get_block_model_client()
                bm_metadata = await client.get_block_model(self.block_model_uuid)
                versions = await client.list_versions(self.block_model_uuid)
                version = versions[0] if versions else None

                self._typed_bm = BMRegularBlockModel(
                    client=client,
                    metadata=bm_metadata,
                    version=version,
                    cell_data=pd.DataFrame(),
                    context=self._api_context,
                )

            return self._typed_bm

        async def get_block_model_metadata(self) -> BlockModelMetadata:
            """Get the full block model metadata from the Block Model Service.

            :return: The BlockModel metadata from the Block Model Service.
            """
            typed_bm = await self._get_or_create_typed_block_model()
            return await typed_bm.get_block_model_metadata()

        async def get_versions(self) -> list[Version]:
            """Get all versions of this block model.

            :return: List of versions, ordered from newest to oldest.
            """
            typed_bm = await self._get_or_create_typed_block_model()
            return await typed_bm.get_versions()

        async def to_dataframe(
            self,
            columns: list[str] | None = None,
            version_uuid: UUID | None | Literal["latest"] = "latest",
            fb: IFeedback = NoFeedback,
        ) -> pd.DataFrame:
            """Get block model data as a DataFrame.

            This is the preferred method for accessing block model data. It retrieves
            the data from the Block Model Service and returns it as a pandas DataFrame.

            :param columns: List of column names to retrieve. Defaults to all columns ["*"].
            :param version_uuid: Specific version to query. Use "latest" (default) to get the latest version,
                or None to use the version referenced by this object.
            :param fb: Optional feedback interface for progress reporting.
            :return: DataFrame containing the block model data with user-friendly column names.

            Example:
                >>> df = await block_model.to_dataframe()
                >>> df.head()
            """
            typed_bm = await self._get_or_create_typed_block_model()
            return await typed_bm.to_dataframe(columns=columns, version_uuid=version_uuid, fb=fb)

        async def refresh(self) -> BlockModel:
            """Refresh this block model object with the latest data from the server.

            Use this after a remote operation (like kriging) has updated the block model
            to see the newly added attributes.

            :return: A new BlockModel instance with refreshed data.

            Example:
                >>> # After running kriging that adds attributes...
                >>> block_model = await block_model.refresh()
                >>> block_model.attributes  # Now shows the new attributes
            """
            # Refresh the typed block model delegate if it exists, so it's immediately up-to-date
            if hasattr(self, "_typed_bm") and self._typed_bm is not None:
                await self._typed_bm.refresh()
            return await object_from_uuid(self._api_context, self.metadata.id)

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
            typed_bm = await self._get_or_create_typed_block_model()
            return await typed_bm.add_attribute(data, attribute_name, unit=unit, fb=fb)

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
            typed_bm = await self._get_or_create_typed_block_model()
            return await typed_bm.update_attributes(
                data,
                new_columns=new_columns,
                update_columns=update_columns,
                delete_columns=delete_columns,
                units=units,
                fb=fb,
            )

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
            fb.progress(0.0, "Creating block model...")

            # Convert to evo-blockmodels data format
            bm_data = BMRegularBlockModelData(
                name=data.name,
                description=data.description,
                origin=Point3(data.origin.x, data.origin.y, data.origin.z),
                n_blocks=Size3i(data.n_blocks.nx, data.n_blocks.ny, data.n_blocks.nz),
                block_size=Size3d(data.block_size.dx, data.block_size.dy, data.block_size.dz),
                cell_data=data.cell_data,
                coordinate_reference_system=data.crs,
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

        async def set_attribute_units(
            self,
            units: dict[str, str],
            fb: IFeedback = NoFeedback,
        ) -> BlockModel:
            """Set units for attributes on this block model.

            This is required before creating reports, as reports need columns to have
            units defined.

            :param units: Dictionary mapping attribute names to unit IDs (e.g., {"Au": "g/t", "density": "t/m3"}).
            :param fb: Optional feedback interface for progress reporting.
            :return: The updated BlockModel instance (refreshed from server).

            Example:
                >>> from evo.blockmodels import Units
                >>> block_model = await block_model.set_attribute_units({
                ...     "Au": Units.GRAMS_PER_TONNE,
                ...     "density": Units.TONNES_PER_CUBIC_METRE,
                ... })
            """
            typed_bm = await self._get_or_create_typed_block_model()
            await typed_bm.set_attribute_units(units, fb=fb)
            return await self.refresh()

        async def create_report(
            self,
            data: ReportSpecificationData,
            fb: IFeedback = NoFeedback,
        ) -> Report:
            """Create a new report specification for this block model.

            Reports require:
            1. Columns to have units set (use `set_attribute_units()` first)
            2. At least one category column for grouping (e.g., domain, rock type)

            :param data: The report specification data.
            :param fb: Optional feedback interface for progress reporting.
            :return: A Report instance representing the created report.

            Example:
                >>> from evo.blockmodels.typed import ReportSpecificationData, ReportColumnSpec, ReportCategorySpec
                >>> report = await block_model.create_report(ReportSpecificationData(
                ...     name="Gold Resource Report",
                ...     columns=[ReportColumnSpec(column_name="Au", aggregation="WEIGHTED_MEAN", output_unit_id="g/t")],
                ...     categories=[ReportCategorySpec(column_name="domain")],
                ...     mass_unit_id="t",
                ...     density_value=2.7,
                ...     density_unit_id="t/m3",
                ... ))
                >>> report  # Pretty-prints with BlockSync link
            """
            typed_bm = await self._get_or_create_typed_block_model()
            return await typed_bm.create_report(data, fb=fb)

        async def list_reports(self, fb: IFeedback = NoFeedback) -> list[Report]:
            """List all report specifications for this block model.

            :param fb: Optional feedback interface for progress reporting.
            :return: List of Report instances.
            """
            typed_bm = await self._get_or_create_typed_block_model()
            return await typed_bm.list_reports(fb=fb)
