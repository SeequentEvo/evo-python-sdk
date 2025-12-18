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

import asyncio
import uuid
from uuid import UUID

from evo import logging
from evo.common import APIConnector, BaseAPIClient, Environment, HealthCheckType, ICache, IContext, ServiceHealth
from evo.common.data import ServiceUser
from evo.common.utils import get_service_health

from ._types import Table
from ._utils import convert_dtype, extract_payload
from .data import (
    BaseGridDefinition,
    BlockModel,
    FlexibleGridDefinition,
    FullySubBlockedGridDefinition,
    OctreeGridDefinition,
    RegularGridDefinition,
    SubBlockedGridDefinition,
    Version,
)
from .endpoints import models
from .endpoints.api import ColumnOperationsApi, JobsApi, MetadataApi, OperationsApi, VersionsApi
from .endpoints.models import (
    AnyUrl,
    BBox,
    BBoxXYZ,
    BlockSize,
    ColumnHeaderType,
    CreateData,
    GeometryColumns,
    JobErrorPayload,
    JobResponse,
    JobStatus,
    Location,
    OctreeSubblocks,
    OutputOptionsParquet,
    QueryCriteria,
    QueryDownload,
    RegularSubblocks,
    Rotation,
    RotationAxis,
    Size3D,
    SizeOptionsFlexible,
    SizeOptionsFullySubBlocked,
    SizeOptionsOctree,
    SizeOptionsRegular,
)
from .exceptions import CacheNotConfiguredException, JobFailedException, MissingColumnInTable
from .io import BlockModelDownload, BlockModelUpload, get_cache_location_for_upload

logger = logging.getLogger("blockmodel.client")

__all__ = [
    "BlockModelAPIClient",
]


def _job_id_from_url(job_url: AnyUrl) -> UUID:
    job_id = job_url.path.split("/")[-1]
    return UUID(job_id)


def _version_from_model(version: models.Version) -> Version:
    return Version(
        bm_uuid=version.bm_uuid,
        version_id=version.version_id,
        version_uuid=version.version_uuid,
        created_at=version.created_at,
        created_by=ServiceUser.from_model(version.created_by),
        comment=version.comment,
        bbox=version.bbox,
        base_version_id=version.base_version_id,
        parent_version_id=version.parent_version_id,
        columns=version.mapping.columns,
        geoscience_version_id=version.geoscience_version_id,
    )


_GEOMETRY_COLUMNS = {
    "i",
    "j",
    "k",
    "sidx",
    "start_si",
    "start_sj",
    "start_sk",
    "end_si",
    "end_sj",
    "end_sk",
    "x",
    "y",
    "z",
    "dx",
    "dy",
    "dz",
}


class BlockModelAPIClient(BaseAPIClient):
    def __init__(self, environment: Environment, connector: APIConnector, cache: ICache | None = None) -> None:
        """
        Constructor for the Block Model Service client.

        Some methods need a cache to store temporary files. If you want to use these methods, you must provide a cache.

        :param environment: The environment object.
        :param connector: The connector object.
        :param cache: The cache to use for storing temporary files.
        """
        super().__init__(environment, connector)
        self._versions_api = VersionsApi(connector)
        self._jobs_api = JobsApi(connector)
        self._operations_api = OperationsApi(connector)
        self._column_operations_api = ColumnOperationsApi(connector)
        self._metadata_api = MetadataApi(connector)
        self._cache = cache

    @classmethod
    def from_context(cls, context: IContext) -> BlockModelAPIClient:
        """Create a BlockModelAPIClient from the given context.

        The context must have a hub_url, org_id, and workspace_id set.

        :param context: The context to create the client from.
        :return: A BlockModelAPIClient instance.
        """
        return cls(
            environment=context.get_environment(),
            connector=context.get_connector(),
            cache=context.get_cache(),
        )

    async def get_service_health(self, check_type: HealthCheckType = HealthCheckType.FULL) -> ServiceHealth:
        """Get the health of the service.

        :param check_type: The type of health check to perform.

        :return: A ServiceHealth object.

        :raises EvoAPIException: If the API returns an unexpected status code.
        :raises ClientValueError: If the response is not a valid service health check response.
        """
        return await get_service_health(self._connector, "blockmodel", check_type=check_type)

    def _bm_from_model(self, model: models.BlockModel | models.BlockModelAndJobURL) -> BlockModel:
        match model.size_options:
            case SizeOptionsRegular(n_blocks=n_blocks, block_size=block_size):
                grid_definition = RegularGridDefinition(
                    model_origin=[model.model_origin.x, model.model_origin.y, model.model_origin.z],
                    rotations=[(rotation.axis, rotation.angle) for rotation in model.block_rotation],
                    n_blocks=[n_blocks.nx, n_blocks.ny, n_blocks.nz],
                    block_size=[block_size.x, block_size.y, block_size.z],
                )

            case SizeOptionsFullySubBlocked(
                n_parent_blocks=n_parent_blocks, n_subblocks_per_parent=n_subblocks, parent_block_size=parent_block_size
            ):
                grid_definition = FullySubBlockedGridDefinition(
                    model_origin=[model.model_origin.x, model.model_origin.y, model.model_origin.z],
                    rotations=[(rotation.axis, rotation.angle) for rotation in model.block_rotation],
                    n_parent_blocks=[n_parent_blocks.nx, n_parent_blocks.ny, n_parent_blocks.nz],
                    n_subblocks_per_parent=[n_subblocks.nx, n_subblocks.ny, n_subblocks.nz],
                    parent_block_size=[parent_block_size.x, parent_block_size.y, parent_block_size.z],
                )

            case SizeOptionsFlexible(
                n_parent_blocks=n_parent_blocks, n_subblocks_per_parent=n_subblocks, parent_block_size=parent_block_size
            ):
                grid_definition = FlexibleGridDefinition(
                    model_origin=[model.model_origin.x, model.model_origin.y, model.model_origin.z],
                    rotations=[(rotation.axis, rotation.angle) for rotation in model.block_rotation],
                    n_parent_blocks=[n_parent_blocks.nx, n_parent_blocks.ny, n_parent_blocks.nz],
                    n_subblocks_per_parent=[n_subblocks.nx, n_subblocks.ny, n_subblocks.nz],
                    parent_block_size=[parent_block_size.x, parent_block_size.y, parent_block_size.z],
                )

            case SizeOptionsOctree(
                n_parent_blocks=n_parent_blocks, n_subblocks_per_parent=n_subblocks, parent_block_size=parent_block_size
            ):
                grid_definition = OctreeGridDefinition(
                    model_origin=[model.model_origin.x, model.model_origin.y, model.model_origin.z],
                    rotations=[(rotation.axis, rotation.angle) for rotation in model.block_rotation],
                    n_parent_blocks=[n_parent_blocks.nx, n_parent_blocks.ny, n_parent_blocks.nz],
                    n_subblocks_per_parent=[n_subblocks.nx, n_subblocks.ny, n_subblocks.nz],
                    parent_block_size=[parent_block_size.x, parent_block_size.y, parent_block_size.z],
                )

            case _:
                # Fall back to an informative error for unknown types
                raise NotImplementedError(f"Size options type '{type(model.size_options).__name__!r}' is not supported")

        return BlockModel(
            environment=self._environment,
            id=model.bm_uuid,
            name=model.name,
            created_at=model.created_at,
            created_by=ServiceUser.from_model(model.created_by),
            description=model.description,
            grid_definition=grid_definition,
            coordinate_reference_system=model.coordinate_reference_system,
            size_unit_id=model.size_unit_id,
            bbox=model.bbox,
            last_updated_at=model.last_updated_at,
            last_updated_by=ServiceUser.from_model(model.last_updated_by),
            geoscience_object_id=model.geoscience_object_id,
        )

    async def _poll_job_url(self, bm_id: UUID, job_id: UUID) -> JobResponse:
        while True:
            response = await self._jobs_api.get_job_status(
                job_id=str(job_id),
                workspace_id=str(self._environment.workspace_id),
                org_id=str(self._environment.org_id),
                bm_id=str(bm_id),
            )
            if response.job_status == JobStatus.COMPLETE:
                return response
            elif response.job_status == JobStatus.FAILED:
                payload = extract_payload(job_id, response, JobErrorPayload)
                raise JobFailedException(job_id, payload)
            await asyncio.sleep(1)

    async def _create_block_model(
        self,
        name: str,
        grid_definition: BaseGridDefinition,
        description: str | None = None,
        object_path: str | None = None,
        coordinate_reference_system: str | None = None,
        size_unit_id: str | None = None,
    ):
        match grid_definition:
            case RegularGridDefinition(n_blocks=n_blocks, block_size=block_size):
                size_options = SizeOptionsRegular(
                    model_type="regular",
                    n_blocks=Size3D(nx=n_blocks[0], ny=n_blocks[1], nz=n_blocks[2]),
                    block_size=BlockSize(x=block_size[0], y=block_size[1], z=block_size[2]),
                )
            case FullySubBlockedGridDefinition(
                n_parent_blocks=n_parent_blocks, n_subblocks_per_parent=n_subblocks, parent_block_size=parent_block_size
            ):
                size_options = SizeOptionsFullySubBlocked(
                    model_type="fully-sub-blocked",
                    n_parent_blocks=Size3D(nx=n_parent_blocks[0], ny=n_parent_blocks[1], nz=n_parent_blocks[2]),
                    n_subblocks_per_parent=RegularSubblocks(nx=n_subblocks[0], ny=n_subblocks[1], nz=n_subblocks[2]),
                    parent_block_size=BlockSize(x=parent_block_size[0], y=parent_block_size[1], z=parent_block_size[2]),
                )
            case FlexibleGridDefinition(
                n_parent_blocks=n_parent_blocks, n_subblocks_per_parent=n_subblocks, parent_block_size=parent_block_size
            ):
                size_options = SizeOptionsFlexible(
                    model_type="flexible",
                    n_parent_blocks=Size3D(nx=n_parent_blocks[0], ny=n_parent_blocks[1], nz=n_parent_blocks[2]),
                    n_subblocks_per_parent=RegularSubblocks(nx=n_subblocks[0], ny=n_subblocks[1], nz=n_subblocks[2]),
                    parent_block_size=BlockSize(x=parent_block_size[0], y=parent_block_size[1], z=parent_block_size[2]),
                )
            case OctreeGridDefinition(
                n_parent_blocks=n_parent_blocks, n_subblocks_per_parent=n_subblocks, parent_block_size=parent_block_size
            ):
                size_options = SizeOptionsOctree(
                    model_type="variable-octree",
                    n_parent_blocks=Size3D(nx=n_parent_blocks[0], ny=n_parent_blocks[1], nz=n_parent_blocks[2]),
                    n_subblocks_per_parent=OctreeSubblocks(nx=n_subblocks[0], ny=n_subblocks[1], nz=n_subblocks[2]),
                    parent_block_size=BlockSize(x=parent_block_size[0], y=parent_block_size[1], z=parent_block_size[2]),
                )
            case _:
                raise NotImplementedError(f"Unknown grid definition type '{type(grid_definition).__name__}'")
        create_result = await self._operations_api.create_block_model(
            workspace_id=str(self._environment.workspace_id),
            org_id=str(self._environment.org_id),
            create_data=CreateData(
                name=name,
                description=description,
                object_path=object_path,
                coordinate_reference_system=coordinate_reference_system,
                size_unit_id=size_unit_id,
                model_origin=Location(
                    x=grid_definition.model_origin[0],
                    y=grid_definition.model_origin[1],
                    z=grid_definition.model_origin[2],
                ),
                block_rotation=[
                    Rotation(axis=RotationAxis(axis), angle=angle) for axis, angle in grid_definition.rotations
                ],
                size_options=size_options,
            ),
        )
        job_id = _job_id_from_url(create_result.job_url)
        job_status = await self._poll_job_url(create_result.bm_uuid, job_id)
        version = extract_payload(job_id, job_status, models.Version)
        return create_result, _version_from_model(version)

    async def _upload_data(self, bm_id: uuid.UUID, job_id: uuid.UUID, upload_url: str, data: Table) -> models.Version:
        """Upload data to a block model service, marks the upload as complete and waits for the job to complete."""
        # Write the data to a temporary file
        import pyarrow.parquet

        cache_location = get_cache_location_for_upload(self._cache, self._environment, job_id)
        pyarrow.parquet.write_table(data, cache_location)

        # Upload the data
        upload = BlockModelUpload(self._connector, self._environment, bm_id, job_id, upload_url)
        await upload.upload_from_path(cache_location, self._connector.transport)

        # Notify the service that the upload is complete
        await self._column_operations_api.notify_upload_complete(
            org_id=str(self._environment.org_id),
            workspace_id=str(self._environment.workspace_id),
            bm_id=str(bm_id),
            job_id=str(job_id),
        )

        # Poll the job URL until it is complete
        job_status = await self._poll_job_url(bm_id, job_id)
        return extract_payload(job_id, job_status, models.Version)

    async def _update_model_no_data(
        self, bm_id: UUID, columns: models.UpdateColumnsLiteInput, comment: str | None = None
    ) -> Version:
        """Helper to apply an UpdateColumnsLiteInput and return the resulting Version.
        This is for column operations where new data is not required.
        """
        update_response = await self._column_operations_api.update_block_model_from_latest_version(
            org_id=str(self._environment.org_id),
            workspace_id=str(self._environment.workspace_id),
            bm_id=str(bm_id),
            update_data_lite_input=models.UpdateDataLiteInput(
                columns=columns,
                comment=comment,
            ),
        )

        job_id = _job_id_from_url(update_response.job_url)
        await self._column_operations_api.notify_upload_complete(
            org_id=str(self._environment.org_id),
            workspace_id=str(self._environment.workspace_id),
            bm_id=str(bm_id),
            job_id=str(job_id),
        )
        job_status = await self._poll_job_url(bm_id, job_id)
        version = extract_payload(job_id, job_status, models.Version)
        return _version_from_model(version)

    async def list_block_models(self) -> list[BlockModel]:
        """List block models in the current workspace.

        Returns a list of `BlockModel`s for the workspace referenced by the client's
        `Environment`. Limited to the first 100 results.
        """
        response = await self._metadata_api.list_block_models(
            workspace_id=str(self._environment.workspace_id),
            org_id=str(self._environment.org_id),
        )

        return [self._bm_from_model(m) for m in response.results]

    async def list_all_block_models(self, page_limit: int | None = 100) -> list[BlockModel]:
        """Return all block models for the current workspace, following paginated responses.

        This method will page through the `list_block_models` endpoint using `offset` and `limit`
        until all entries are retrieved. The `page_limit` is clamped to the service maximum (100).

        :param page_limit: Maximum items to request per page (1..100). Defaults to 100.
        :return: A list of `BlockModel` dataclasses for the workspace.
        """
        if page_limit is None:
            page_limit = 100
        if page_limit <= 0:
            raise ValueError("page_limit must be > 0")
        # Service enforces maximum limit of 100
        page_limit = min(page_limit, 100)

        offset = 0
        all_models: list[BlockModel] = []

        while True:
            page = await self._metadata_api.list_block_models(
                workspace_id=str(self._environment.workspace_id),
                org_id=str(self._environment.org_id),
                offset=offset,
                limit=page_limit,
            )

            # Convert and append
            for m in page.results:
                all_models.append(self._bm_from_model(m))

            # Determine stopping condition
            if page.count == 0:
                break
            if page.total is not None:
                if offset + page.count >= page.total:
                    break
            # Advance offset
            offset += page.count

        return all_models

    async def list_versions(self, bm_id: UUID) -> list[Version]:
        """List versions of a block model.

        Returns a list of `Version`s for the block model referenced by `bm_id`,
        limited to the first 100 results (or the service maximum).

        Versions are ordered from newest to oldest.

        :param bm_id: The ID of the block model.
        :return: A list of `Version` dataclasses for the block model, ordered newest to oldest.
        """
        response = await self._versions_api.list_block_model_versions(
            workspace_id=str(self._environment.workspace_id),
            org_id=str(self._environment.org_id),
            bm_id=str(bm_id),
        )

        return [_version_from_model(v) for v in response.results]

    async def list_all_versions(self, bm_id: UUID, page_limit: int | None = 100) -> list[Version]:
        """Return all versions of a block model, following paginated responses.

        This method will page through the `list_block_model_versions` endpoint using `offset` and `limit`
        until all entries are retrieved. The `page_limit` is clamped to the service maximum (100).

        Versions are ordered from newest to oldest.

        :param bm_id: The ID of the block model.
        :param page_limit: Maximum items to request per page (1..100). Defaults to 100.
        :return: A list of `Version` dataclasses for the block model, ordered newest to oldest.
        """
        if page_limit is None:
            page_limit = 100
        if page_limit <= 0:
            raise ValueError("page_limit must be > 0")
        # Service enforces maximum limit of 100
        page_limit = min(page_limit, 100)

        offset = 0
        all_versions: list[Version] = []

        while True:
            page = await self._versions_api.list_block_model_versions(
                workspace_id=str(self._environment.workspace_id),
                org_id=str(self._environment.org_id),
                bm_id=str(bm_id),
                offset=offset,
                limit=page_limit,
            )

            # Convert and append
            for v in page.results:
                all_versions.append(_version_from_model(v))

            # Determine stopping condition
            if page.count == 0:
                break
            if page.total is not None:
                if offset + page.count >= page.total:
                    break
            # Advance offset
            offset += page.count

        return all_versions

    async def create_block_model(
        self,
        name: str,
        grid_definition: BaseGridDefinition,
        description: str | None = None,
        object_path: str | None = None,
        coordinate_reference_system: str | None = None,
        size_unit_id: str | None = None,
        initial_data: Table | None = None,
        units: dict[str, str] | None = None,
    ) -> tuple[BlockModel, Version]:
        r"""Create a block model.

        Optionally, takes initial data to populate the block model with. Units for the columns within the initial data
        can be provided in the `units` dictionary. This requires the `pyarrow` package to be installed, and the 'cache'
        parameter to be set in the constructor.

        If `initial_data` is provided, this method will wait for the initial data to be successfully uploaded and
        processed before returning. This then returns both the block model and the version created from the initial data
        update.

        Otherwise, if `initial_data` is not provided, this waits for the block model creation job to complete before
        returning. This then returns both the block model and the initial block model version.

        :param name: Name of the block model. This may not contain `/` nor `\`.
        :param grid_definition: Definition of the block model grid.
        :param description: Description of the block model.
        :param object_path: Path of the folder in Geoscience Object Service to create the reference object in.
        :param coordinate_reference_system: Coordinate reference system used in the block model.
        :param size_unit_id: Unit ID denoting the length unit used for the block model's blocks.
        :param initial_data: The initial data to populate the block model with.
        :param units: A dictionary mapping column names within `initial_data` to units.
        :return: A tuple containing the created block model and the version of the block model.
        """
        if units is not None and initial_data is None:
            raise ValueError("units can only be provided if initial_data is provided")
        if initial_data is not None and self._cache is None:
            raise CacheNotConfiguredException(
                "Cache must be configured to use this method. Please set the 'cache' parameter in the constructor."
            )
        create_result, version = await self._create_block_model(
            name, grid_definition, description, object_path, coordinate_reference_system, size_unit_id
        )

        if initial_data is not None:
            if isinstance(grid_definition, SubBlockedGridDefinition):
                geometry_change = True
            else:
                geometry_change = None
            version = await self._add_new_columns(create_result.bm_uuid, initial_data, units, geometry_change)
        return self._bm_from_model(create_result), version

    async def _add_new_columns(
        self,
        bm_id: UUID,
        data: Table,
        units: dict[str, str] | None = None,
        geometry_change: bool | None = None,
    ):
        """Add new columns to an existing block model.

        For sub-blocked models, this will not change the sub-blocking structure. Thus the block within the data must match existing sub-blocks in the model.

        Units for the columns can be provided in the `units` dictionary.

        This method requires the `pyarrow` package to be installed, and the 'cache' parameter to be set in the constructor.

        :param bm_id: The ID of the block model to add columns to.
        :param data: The data containing the new columns to add.
        :param units: A dictionary mapping column names within `data` to units.
        :param geometry_change: Whether the geometry of the block model is changing.
        :raises CacheNotConfiguredException: If the cache is not configured.
        :return: The new version of the block model with the added columns.
        """
        if self._cache is None:
            raise CacheNotConfiguredException(
                "Cache must be configured to use this method. Please set the 'cache' parameter in the constructor."
            )

        schema = data.schema
        if units is None:
            units = {}
        columns = models.UpdateColumnsLiteInput(
            new=[
                models.ColumnLite(title=name, data_type=convert_dtype(data_type), unit_id=units.get(name))
                for name, data_type in zip(schema.names, schema.types)
                if name not in _GEOMETRY_COLUMNS
            ],
            update=[],
            delete=[],
            rename=[],
        )
        update_response = await self._column_operations_api.update_block_model_from_latest_version(
            org_id=str(self._environment.org_id),
            workspace_id=str(self._environment.workspace_id),
            bm_id=str(bm_id),
            update_data_lite_input=models.UpdateDataLiteInput(
                columns=columns,
                update_type=models.UpdateType.replace,
                geometry_change=geometry_change,
            ),
        )
        version = await self._upload_data(bm_id, update_response.job_uuid, str(update_response.upload_url), data)
        return _version_from_model(version)

    async def add_new_subblocked_columns(
        self,
        bm_id: UUID,
        data: Table,
        units: dict[str, str] | None = None,
    ):
        """Add new columns to an existing sub-blocked block model. This will not change the sub-blocking structure, thus the provided data must match existing sub-blocks in the model.

        Units for the columns can be provided in the `units` dictionary.

        This method requires the `pyarrow` package to be installed, and the 'cache' parameter to be set in the constructor.

        :param bm_id: The ID of the block model to add columns to.
        :param data: The data containing the new columns to add.
        :param units: A dictionary mapping column names within `data` to units.
        :raises CacheNotConfiguredException: If the cache is not configured.
        :return: The new version of the block model with the added columns.
        """
        return await self._add_new_columns(bm_id, data, units, geometry_change=False)

    async def add_new_columns(
        self,
        bm_id: UUID,
        data: Table,
        units: dict[str, str] | None = None,
    ):
        """Add new columns to an existing regular block model.

        Units for the columns can be provided in the `units` dictionary.

        This method requires the `pyarrow` package to be installed, and the 'cache' parameter to be set in the constructor.

        :param bm_id: The ID of the block model to add columns to.
        :param data: The data containing the new columns to add.
        :param units: A dictionary mapping column names within `data` to units.
        :raises CacheNotConfiguredException: If the cache is not configured.
        :return: The new version of the block model with the added columns.
        """
        return await self._add_new_columns(bm_id, data, units, geometry_change=None)

    async def update_block_model_columns(
        self,
        bm_id: UUID,
        data: Table,
        new_columns: list[str],
        update_columns: set[str] | None = None,
        delete_columns: set[str] | None = None,
        units: dict[str, str] | None = None,
    ) -> Version:
        """Add, update, or delete block model columns.

        Units for the columns can be provided in the `units` dictionary.

        This method requires the `pyarrow` package to be installed, and the 'cache' parameter to be set in the constructor.

        :param bm_id: The ID of the block model to add columns to.
        :param data: The data containing the new columns to add.
        :param new_columns: A list of new column names to add to the block model.
        :param update_columns: A set of column names to update in the block model.
        :param delete_columns: A set of column names to delete from the block model.
        :param units: A dictionary mapping column names within `data` to units.
        :raises CacheNotConfiguredException: If the cache is not configured.
        :return: The new version of the block model with the added columns.
        """
        if self._cache is None:
            raise CacheNotConfiguredException(
                "Cache must be configured to use this method. Please set the 'cache' parameter in the constructor."
            )

        schema = data.schema
        if units is None:
            units = {}
        data_type_map = {name: data_type for name, data_type in zip(schema.names, schema.types)}

        if update_columns is None:
            update_columns = set()

        if delete_columns is None:
            delete_columns = set()

        # Check for any new or updated columns that are not in the data
        missing = (set(new_columns) | update_columns) - data_type_map.keys()
        if missing:
            raise MissingColumnInTable(f"Columns {missing} are not present in the provided table.")

        columns = models.UpdateColumnsLiteInput(
            new=[
                models.ColumnLite(
                    title=new_column, data_type=convert_dtype(data_type_map[new_column]), unit_id=units.get(new_column)
                )
                for new_column in new_columns
            ],
            update=list(update_columns),
            delete=list(delete_columns),
            rename=[],
        )
        update_response = await self._column_operations_api.update_block_model_from_latest_version(
            org_id=str(self._environment.org_id),
            workspace_id=str(self._environment.workspace_id),
            bm_id=str(bm_id),
            update_data_lite_input=models.UpdateDataLiteInput(
                columns=columns,
                update_type=models.UpdateType.replace,
            ),
        )
        version = await self._upload_data(bm_id, update_response.job_uuid, str(update_response.upload_url), data)
        return _version_from_model(version)

    async def update_column_metadata(
        self,
        bm_id: UUID,
        column_updates: dict[str, str | None],
        comment: str | None = None,
    ) -> Version:
        """Update metadata (e.g., units) for existing block model columns.

        This method updates column properties without requiring data upload or cache configuration.

        :param bm_id: The ID of the block model to update.
        :param column_updates: A dictionary mapping column titles to their new unit IDs.
                               Set the unit ID to None to remove the unit.
                               Example: {"Cu": "%[mass]", "Au": None}
        :param comment: An optional comment describing the metadata changes. This is max 250 characters.
        :return: The new version of the block model with updated metadata.
        """
        update_metadata_list = [
            models.UpdateMetadataLite(title=col_title, values=models.UpdateMetadataValues(unit_id=unit_id))
            for col_title, unit_id in column_updates.items()
        ]

        columns = models.UpdateColumnsLiteInput(
            new=[],
            update=[],
            delete=[],
            rename=[],
            update_metadata=update_metadata_list,
        )

        return await self._update_model_no_data(bm_id, columns, comment=comment)

    async def rename_block_model_columns(
        self,
        bm_id: UUID,
        column_renames: dict[str, str],
        comment: str | None = None,
    ) -> Version:
        """Rename existing block model columns.

        This method renames columns without requiring data upload or cache configuration.

        :param bm_id: The ID of the block model to update.
        :param column_renames: A dictionary mapping current column titles to their new titles.
                               Example: {"Cu": "Copper", "Au": "Gold"}
        :param comment: An optional comment describing the rename operation. This is max 250 characters.
        :return: The new version of the block model with renamed columns.
        """
        rename_list = [
            models.RenameLite(title=old_title, new_title=new_title) for old_title, new_title in column_renames.items()
        ]

        columns = models.UpdateColumnsLiteInput(
            new=[],
            update=[],
            delete=[],
            rename=rename_list,
        )

        return await self._update_model_no_data(bm_id, columns, comment=comment)

    async def delete_block_model_columns(
        self,
        bm_id: UUID,
        column_titles: list[str],
        comment: str | None = None,
    ) -> Version:
        """Delete existing columns from a block model.

        This method deletes columns without requiring data upload or cache configuration.

        :param bm_id: The ID of the block model to update.
        :param column_titles: The titles of the columns to delete.
        :param comment: An optional comment describing the rename operation. This is max 250 characters.
        :return: The new version of the block model with renamed columns.
        """
        columns = models.UpdateColumnsLiteInput(
            new=[],
            update=[],
            delete=column_titles,
            rename=[],
        )

        return await self._update_model_no_data(bm_id, columns, comment=comment)

    async def query_block_model_as_table(
        self,
        bm_id: UUID,
        columns: list[str | UUID],
        bbox: BBox | BBoxXYZ | None = None,
        version_uuid: UUID | None = None,
        geometry_columns: GeometryColumns = GeometryColumns.coordinates,
        column_headers: ColumnHeaderType = ColumnHeaderType.id,
        exclude_null_rows: bool = True,
    ) -> Table:
        """Query a block model and return the result as a PyArrow Table.

        This requires the `pyarrow` package to be installed, and the 'cache' parameter to be set in the constructor.

        :param bm_id: The ID of the block model to query.
        :param columns: The columns to query, can either be the title or the ID of the column.
        :param bbox: The bounding box to query, if None (the default) the entire block model is queried.
        :param version_uuid: The version UUID to query, if None (the default) the latest version is queried.
        :param geometry_columns: Whether rows in the returned table should include coordinates, or block indices of the
            block, that the row belongs to.
        :param column_headers: Whether the names of the columns in the returned column should be the title or the ID of
            the block model column.
        :param exclude_null_rows: Whether to exclude rows where all values are null within the queried columns.
        :return: The result as a PyArrow Table.
        :raises JobFailedException: If the job failed.
        """
        import pyarrow.parquet

        if self._cache is None:
            raise CacheNotConfiguredException(
                "Cache must be configured to use this method. Please set the 'cache' parameter in the constructor."
            )

        query_result = await self._versions_api.query_block_model_latest_as_post(
            bm_id=str(bm_id),
            workspace_id=str(self._environment.workspace_id),
            org_id=str(self._environment.org_id),
            query_criteria=QueryCriteria(
                columns=[str(c) for c in columns],  # Convert any UUIDs to strings
                bbox=bbox,
                version_uuid=version_uuid,
                geometry_columns=geometry_columns,
                output_options=OutputOptionsParquet(
                    file_format="parquet",
                    column_headers=column_headers,
                    exclude_null_rows=exclude_null_rows,
                ),
            ),
        )

        # Poll the job URL until it is complete
        job_id = _job_id_from_url(query_result.job_url)
        job = await self._poll_job_url(bm_id, job_id)
        payload = extract_payload(job_id, job, QueryDownload)

        # Download the result to a temporary file
        download = BlockModelDownload(
            self._connector, self._environment, query_result, job_id, str(payload.download_url)
        )
        path = await download.download_to_cache(self._cache, self._connector.transport)

        # Read the PyArrow Table from the temporary file
        return pyarrow.parquet.read_table(path)
