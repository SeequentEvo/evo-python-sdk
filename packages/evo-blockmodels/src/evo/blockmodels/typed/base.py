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

"""Base class for typed block model access.

Provides shared functionality for all block model types (regular, sub-blocked, octree, etc.)
including data access, attribute management, reports, and versioning.
"""

from abc import ABC, abstractmethod
from typing import Literal
from uuid import UUID

import pandas as pd

from evo.common import IContext, IFeedback
from evo.common.utils import NoFeedback

from ..client import BlockModelAPIClient
from ..data import BlockModel, Version
from ..endpoints.models import ColumnHeaderType
from ._utils import dataframe_to_pyarrow, get_attribute_columns, pyarrow_to_dataframe
from .report import Report, ReportSpecificationData
from .types import Point3

__all__ = [
    "BaseTypedBlockModel",
]


class BaseTypedBlockModel(ABC):
    """Abstract base class for typed block model wrappers.

    Provides shared functionality for all block model types including:
    - Data access via pandas DataFrames
    - Attribute management (add, update, delete, set units)
    - Report creation and listing
    - Version management
    - Metadata access

    Subclasses must implement grid-type-specific properties and factory methods.
    """

    def __init__(
        self,
        client: BlockModelAPIClient,
        metadata: BlockModel,
        version: Version,
        cell_data: pd.DataFrame,
    ) -> None:
        """Initialize a BaseTypedBlockModel instance.

        :param client: The BlockModelAPIClient used for API operations.
        :param metadata: The block model metadata.
        :param version: The current version information.
        :param cell_data: The cell data as a pandas DataFrame.
        """
        self._client = client
        self._metadata = metadata
        self._version = version
        self._cell_data = cell_data

    @property
    def id(self) -> UUID:
        """The unique identifier of the block model."""
        return self._metadata.id

    @property
    def name(self) -> str:
        """The name of the block model."""
        return self._metadata.name

    @property
    def description(self) -> str | None:
        """The description of the block model."""
        return self._metadata.description

    @property
    def origin(self) -> Point3:
        """The origin point of the block model grid."""
        grid_def = self._metadata.grid_definition
        return Point3(
            x=grid_def.model_origin[0],
            y=grid_def.model_origin[1],
            z=grid_def.model_origin[2],
        )

    @property
    def metadata(self) -> BlockModel:
        """The full block model metadata."""
        return self._metadata

    @property
    def version(self) -> Version:
        """The current version information."""
        return self._version

    @property
    def cell_data(self) -> pd.DataFrame:
        """The cell data as a pandas DataFrame."""
        return self._cell_data

    # ---- Data access ----

    async def to_dataframe(
        self,
        columns: list[str] | None = None,
        version_uuid: UUID | None | Literal["latest"] = "latest",
        fb: IFeedback = NoFeedback,
    ) -> pd.DataFrame:
        """Get block model data as a DataFrame.

        Retrieves data from the Block Model Service and returns it as a pandas DataFrame
        with user-friendly column names.

        :param columns: List of column names to retrieve. Defaults to all columns ["*"].
        :param version_uuid: Specific version to query. Use "latest" (default) for the latest version,
            or None to use the version referenced by this object.
        :param fb: Optional feedback interface for progress reporting.
        :return: DataFrame containing the block model data.

        Example:
            >>> df = await block_model.to_dataframe()
            >>> df.head()
        """
        fb.progress(0.0, "Querying block model data...")

        # Determine which version to query
        query_version: UUID | None = None
        if version_uuid == "latest":
            query_version = None
        elif version_uuid is None:
            query_version = self._version.version_uuid
        else:
            query_version = version_uuid

        if columns is None:
            columns = ["*"]

        table = await self._client.query_block_model_as_table(
            bm_id=self._metadata.id,
            columns=columns,
            version_uuid=query_version,
            column_headers=ColumnHeaderType.name,
        )

        fb.progress(0.9, "Converting data...")
        result = table.to_pandas()
        fb.progress(1.0, "Data retrieved")
        return result

    # ---- Attribute management ----

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
        fb.progress(0.0, "Preparing attribute data...")

        table = dataframe_to_pyarrow(data)

        fb.progress(0.2, "Uploading attribute...")

        units = {attribute_name: unit} if unit else None
        version = await self._client.add_new_columns(
            bm_id=self._metadata.id,
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
        """Update attributes in the block model.

        :param data: DataFrame containing the updated data with geometry columns.
        :param new_columns: List of new column names to add.
        :param update_columns: Set of existing column names to update.
        :param delete_columns: Set of column names to delete.
        :param units: Optional dictionary mapping column names to unit identifiers.
        :param fb: Optional feedback interface for progress reporting.
        :return: The new version created by the update.
        """
        fb.progress(0.0, "Preparing attribute update...")

        table = dataframe_to_pyarrow(data)

        fb.progress(0.2, "Uploading updated data...")

        if new_columns is None and update_columns is None:
            new_columns = get_attribute_columns(data)

        version = await self._client.update_block_model_columns(
            bm_id=self._metadata.id,
            data=table,
            new_columns=new_columns or [],
            update_columns=update_columns,
            delete_columns=delete_columns,
            units=units,
        )

        fb.progress(0.4, "Data uploaded, processing...")

        self._version = version
        self._cell_data = data.copy()

        fb.progress(1.0, "Attributes updated successfully")
        return version

    async def set_attribute_units(
        self,
        units: dict[str, str],
        fb: IFeedback = NoFeedback,
    ) -> Version:
        """Set units for attributes on this block model.

        This is required before creating reports, as reports need columns to have
        units defined.

        :param units: Dictionary mapping attribute names to unit IDs (e.g., {"Au": "g/t", "density": "t/m3"}).
        :param fb: Optional feedback interface for progress reporting.
        :return: The new version created by the metadata update.

        Example:
            >>> from evo.blockmodels import Units
            >>> version = await block_model.set_attribute_units({
            ...     "Au": Units.GRAMS_PER_TONNE,
            ...     "density": Units.TONNES_PER_CUBIC_METRE,
            ... })
        """
        fb.progress(0.0, "Updating attribute units...")

        version = await self._client.update_column_metadata(
            bm_id=self._metadata.id,
            column_updates=units,
        )

        fb.progress(0.9, "Refreshing metadata...")

        self._version = version

        fb.progress(1.0, "Units updated")
        return version

    # ---- Version management ----

    async def get_versions(self) -> list[Version]:
        """Get all versions of this block model.

        :return: List of versions, ordered from newest to oldest.
        """
        return await self._client.list_versions(self._metadata.id)

    async def get_block_model_metadata(self) -> BlockModel:
        """Get the full block model metadata from the Block Model Service.

        :return: The BlockModel metadata from the Block Model Service.
        """
        return await self._client.get_block_model(self._metadata.id)

    # ---- Reports ----

    def _get_column_id_map(self) -> dict[str, UUID]:
        """Get a mapping of column names to their UUIDs from the current version.

        :return: Dictionary mapping column names to UUIDs.
        """
        result = {}
        if self._version and self._version.columns:
            for col in self._version.columns:
                if col.col_id:
                    try:
                        result[col.title] = UUID(col.col_id)
                    except ValueError:
                        pass
        return result

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
        """
        fb.progress(0.0, "Preparing report specification...")

        # Refresh to ensure we have latest column information
        await self.refresh(fb=NoFeedback)
        column_id_map = self._get_column_id_map()

        fb.progress(0.2, "Creating report...")

        context = self._get_context()
        report = await Report.create(
            context=context,
            block_model_uuid=self._metadata.id,
            data=data,
            column_id_map=column_id_map,
            fb=fb,
            block_model_name=self.name,
        )

        return report

    async def list_reports(self, fb: IFeedback = NoFeedback) -> list[Report]:
        """List all report specifications for this block model.

        :param fb: Optional feedback interface for progress reporting.
        :return: List of Report instances.
        """
        fb.progress(0.0, "Fetching reports...")

        environment = self._metadata.environment
        context = self._get_context()

        result = await self._client._reports_api.list_block_model_report_specifications(
            workspace_id=str(environment.workspace_id),
            org_id=str(environment.org_id),
            bm_id=str(self._metadata.id),
        )

        fb.progress(1.0, f"Found {result.total} reports")

        return [Report(context, self._metadata.id, spec, block_model_name=self.name) for spec in result.results]

    # ---- Refresh ----

    async def refresh(self, fb: IFeedback = NoFeedback) -> None:
        """Refresh the block model data from the server.

        :param fb: Optional feedback interface for progress reporting.
        """
        fb.progress(0.0, "Refreshing block model...")

        self._metadata = await self._client.get_block_model(self._metadata.id)

        table = await self._client.query_block_model_as_table(
            bm_id=self._metadata.id,
            columns=["*"],
        )
        self._cell_data = pyarrow_to_dataframe(table)

        versions = await self._client.list_versions(self._metadata.id)
        if versions:
            self._version = versions[0]

        fb.progress(1.0, "Block model refreshed")

    # ---- Internal helpers ----

    @abstractmethod
    def _get_context(self) -> IContext:
        """Get the IContext for this block model.

        Subclasses must implement this to provide the context used for report creation
        and other operations that require it.
        """
        ...
