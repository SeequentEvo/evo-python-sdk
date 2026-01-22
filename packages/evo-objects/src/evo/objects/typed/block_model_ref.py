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
from typing import TYPE_CHECKING, Any, Literal
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
    from evo.blockmodels.typed import Report, ReportSpecificationData

__all__ = [
    "BlockModel",
    "BlockModelAttribute",
    "BlockModelAttributes",
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


class BlockModelAttributes:
    """A collection of attributes on a block model with pretty-printing support."""

    def __init__(self, attributes: list[BlockModelAttribute]):
        self._attributes = attributes

    def __iter__(self):
        return iter(self._attributes)

    def __len__(self):
        return len(self._attributes)

    def __getitem__(self, index_or_name: int | str) -> BlockModelAttribute:
        if isinstance(index_or_name, str):
            for attr in self._attributes:
                if attr.name == index_or_name:
                    return attr
            raise KeyError(f"Attribute '{index_or_name}' not found")
        return self._attributes[index_or_name]

    def __repr__(self) -> str:
        names = [attr.name for attr in self._attributes]
        return f"BlockModelAttributes({names})"

    def _repr_html_(self) -> str:
        """Return an HTML representation for Jupyter notebooks."""
        from .._html_styles import STYLESHEET, build_nested_table

        if len(self._attributes) == 0:
            return f'{STYLESHEET}<div class="evo-object">No attributes available.</div>'

        headers = ["Name", "Type", "Unit"]
        rows = [[attr.name, attr.attribute_type, attr.unit or ""] for attr in self._attributes]
        table_html = build_nested_table(headers, rows)
        return f'{STYLESHEET}<div class="evo-object">{table_html}</div>'


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
    def attributes(self) -> BlockModelAttributes:
        """The attributes available on this block model."""
        return BlockModelAttributes(_parse_attributes(self._attributes_raw))

    def get_attribute(self, name: str) -> BlockModelAttribute | None:
        """Get an attribute by name.

        :param name: The name of the attribute.
        :return: The attribute, or None if not found.
        """
        for attr in self.attributes:
            if attr.name == name:
                return attr
        return None

    def _repr_html_(self) -> str:
        """Return an HTML representation for Jupyter notebooks."""
        from .._html_styles import STYLESHEET, build_nested_table, build_table_row, build_table_row_vtop, build_title

        doc = self.as_dict()

        # Get basic info
        name = doc.get("name", "Unnamed")

        # Build title links for viewer and portal
        title_links = [("Portal", self.portal_url), ("Viewer", self.viewer_url)]

        # Build basic rows
        rows = [
            ("Block Model UUID:", str(self.block_model_uuid)),
        ]

        # Add geometry info
        geom = self.geometry
        geom_rows = [
            ["<strong>Origin:</strong>", f"({geom.origin.x:.2f}, {geom.origin.y:.2f}, {geom.origin.z:.2f})"],
            ["<strong>N Blocks:</strong>", f"({geom.n_blocks.nx}, {geom.n_blocks.ny}, {geom.n_blocks.nz})"],
            ["<strong>Block Size:</strong>", f"({geom.block_size.dx:.2f}, {geom.block_size.dy:.2f}, {geom.block_size.dz:.2f})"],
        ]
        if geom.rotation:
            geom_rows.append(["<strong>Rotation:</strong>", f"({geom.rotation[0]:.2f}, {geom.rotation[1]:.2f}, {geom.rotation[2]:.2f})"])
        geom_table = build_nested_table(["Property", "Value"], geom_rows)
        rows.append(("Geometry:", geom_table))

        # Add bounding box if present (as nested table)
        if bbox := doc.get("bounding_box"):
            bbox_rows = [
                ["<strong>X:</strong>", f"{bbox.get('min_x', 0):.2f}", f"{bbox.get('max_x', 0):.2f}"],
                ["<strong>Y:</strong>", f"{bbox.get('min_y', 0):.2f}", f"{bbox.get('max_y', 0):.2f}"],
                ["<strong>Z:</strong>", f"{bbox.get('min_z', 0):.2f}", f"{bbox.get('max_z', 0):.2f}"],
            ]
            bbox_table = build_nested_table(["", "Min", "Max"], bbox_rows)
            rows.append(("Bounding Box:", bbox_table))

        # Add CRS if present
        if crs := doc.get("coordinate_reference_system"):
            crs_str = f"EPSG:{crs.get('epsg_code')}" if isinstance(crs, dict) else str(crs)
            rows.append(("CRS:", crs_str))

        # Build the main table (handle nested tables with vtop alignment)
        table_rows = []
        for label, value in rows:
            if label in ("Bounding Box:", "Geometry:"):
                table_rows.append(build_table_row_vtop(label, value))
            else:
                table_rows.append(build_table_row(label, value))

        main_table = f'<table>{"".join(table_rows)}</table>'

        # Build attributes section
        attributes_html = ""
        attrs = self.attributes
        if attrs:
            attr_rows = [[attr.name, attr.attribute_type, attr.unit or ""] for attr in attrs]
            attrs_table = build_nested_table(["Name", "Type", "Unit"], attr_rows)
            attributes_html = f'<div style="margin-top: 8px;"><strong>Attributes ({len(attrs)}):</strong></div>{attrs_table}'

        return f'{STYLESHEET}<div class="evo-object">{build_title(name, title_links)}{main_table}{attributes_html}</div>'

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
        version_uuid: UUID | None | Literal["latest"] = "latest",
        fb: IFeedback = NoFeedback,
    ) -> pd.DataFrame:
        """Get block model data as a DataFrame.

        :param columns: List of column names to retrieve. Defaults to all columns ["*"].
        :param version_uuid: Specific version to query. Use "latest" (default) to get the latest version,
            or None to use the version referenced by this object.
        :param fb: Optional feedback interface for progress reporting.
        :return: DataFrame containing the block model data with user-friendly column names.
        """
        from evo.blockmodels.endpoints.models import ColumnHeaderType

        client = self._get_block_model_client()

        fb.progress(0.0, "Querying block model data...")

        # Determine which version to query
        query_version: UUID | None = None
        if version_uuid == "latest":
            # Get the latest version (pass None to block model service)
            query_version = None
        elif version_uuid is None:
            # Use the referenced version
            query_version = self.block_model_version_uuid
        else:
            # Use the explicitly provided version
            query_version = version_uuid

        # Default to all columns
        if columns is None:
            columns = ["*"]

        table = await client.query_block_model_as_table(
            bm_id=self.block_model_uuid,
            columns=columns,
            version_uuid=query_version,
            column_headers=ColumnHeaderType.name,  # Use column titles, not UUIDs
        )

        fb.progress(0.9, "Converting data...")

        result = table.to_pandas()

        fb.progress(1.0, "Data retrieved")
        return result

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
        return await self.get_data(columns=columns, version_uuid=version_uuid, fb=fb)

    async def refresh(self) -> "BlockModel":
        """Refresh this block model object with the latest data from the server.

        Use this after a remote operation (like kriging) has updated the block model
        to see the newly added attributes.

        This method:
        1. Reloads the geoscience object metadata
        2. Fetches the latest version's attributes from the Block Model Service

        :return: A new BlockModel instance with refreshed data.

        Example:
            >>> # After running kriging that adds attributes...
            >>> block_model = await block_model.refresh()
            >>> block_model.attributes  # Now shows the new attributes
        """
        from . import object_from_reference

        # Reload the object from the server using its reference URL
        refreshed = await object_from_reference(self._context, self.metadata.url)

        # Also fetch the latest attributes from the Block Model Service
        # The geoscience object might not have the latest attributes if they were
        # added by the Block Model Service (e.g., via kriging)
        try:
            client = refreshed._get_block_model_client()
            versions = await client.list_versions(refreshed.block_model_uuid)
            if versions:
                latest_version = versions[0]  # List is ordered newest to oldest
                # Convert Column objects to the expected _attributes_raw format
                # Filter out geometry columns (i, j, k, x, y, z)
                geometry_cols = {"i", "j", "k", "x", "y", "z"}
                updated_attrs = []
                for col in latest_version.columns:
                    if col.title.lower() not in geometry_cols:
                        # Only include col_id if it looks like a valid UUID (36 chars with dashes)
                        col_uuid = None
                        if col.col_id and len(col.col_id) == 36 and '-' in col.col_id:
                            col_uuid = col.col_id
                        updated_attrs.append({
                            "name": col.title,
                            "attribute_type": col.data_type.value if hasattr(col.data_type, 'value') else str(col.data_type),
                            "block_model_column_uuid": col_uuid,
                            "unit": col.unit_id,
                        })
                # Update the _attributes_raw on the refreshed object
                refreshed._attributes_raw = updated_attrs
        except Exception:
            # If we can't fetch from Block Model Service, just use what we got from geoscience object
            pass

        return refreshed

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

    async def set_attribute_units(
        self,
        units: dict[str, str],
        fb: IFeedback = NoFeedback,
    ) -> "BlockModel":
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
        fb.progress(0.0, "Updating attribute units...")

        client = self._get_block_model_client()

        fb.progress(0.3, "Applying unit updates...")

        # Use the client's update_column_metadata method
        await client.update_column_metadata(
            bm_id=self.block_model_uuid,
            column_updates=units,
        )

        fb.progress(0.9, "Refreshing block model...")

        # Refresh to get updated metadata
        result = await self.refresh()

        fb.progress(1.0, "Units updated")
        return result

    def _get_column_id_map(self) -> dict[str, UUID]:
        """Get a mapping of column names to their UUIDs.

        :return: Dictionary mapping column names to UUIDs.
        """
        result = {}
        for attr in self.attributes:
            if attr.block_model_column_uuid:
                result[attr.name] = attr.block_model_column_uuid
        return result

    async def create_report(
        self,
        data: "ReportSpecificationData",
        fb: IFeedback = NoFeedback,
    ) -> "Report":
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
        from evo.blockmodels.typed import Report, ReportSpecificationData as RSD

        fb.progress(0.0, "Preparing report specification...")

        # Refresh to ensure we have latest column information
        refreshed = await self.refresh()
        column_id_map = refreshed._get_column_id_map()

        fb.progress(0.2, "Creating report...")

        report = await Report.create(
            context=self._context,
            block_model_uuid=self.block_model_uuid,
            data=data,
            column_id_map=column_id_map,
            fb=fb,
        )

        return report

    async def list_reports(self, fb: IFeedback = NoFeedback) -> list["Report"]:
        """List all report specifications for this block model.

        :param fb: Optional feedback interface for progress reporting.
        :return: List of Report instances.
        """
        from evo.blockmodels.typed import Report

        fb.progress(0.0, "Fetching reports...")

        client = self._get_block_model_client()
        environment = self._context.get_environment()

        result = await client._reports_api.list_block_model_report_specifications(
            workspace_id=str(environment.workspace_id),
            org_id=str(environment.org_id),
            bm_id=str(self.block_model_uuid),
        )

        fb.progress(1.0, f"Found {result.total} reports")

        return [
            Report(self._context, self.block_model_uuid, spec)
            for spec in result.results
        ]
