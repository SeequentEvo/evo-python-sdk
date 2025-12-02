import uuid
from datetime import datetime, timezone

from evo.blockmodels import BlockModelAPIClient
from evo.blockmodels.data import (
    FlexibleGridDefinition,
    FullySubBlockedGridDefinition,
    OctreeGridDefinition,
    RegularGridDefinition,
)
from evo.blockmodels.endpoints import models as api_models
from evo.common import Environment
from evo.common.test_tools import BASE_URL, ORG, WORKSPACE_ID, TestWithConnector, TestWithStorage


class TestBMSizeOptionsConversion(TestWithConnector, TestWithStorage):
    def setUp(self) -> None:
        TestWithConnector.setUp(self)
        TestWithStorage.setUp(self)
        self.environment = Environment(hub_url=BASE_URL, org_id=ORG.id, workspace_id=WORKSPACE_ID)
        self.client = BlockModelAPIClient(connector=self.connector, environment=self.environment)

    def _create_bm(self, name: str, size_options: object) -> api_models.BlockModel:
        now = datetime.now(timezone.utc)
        return api_models.BlockModel(
            bbox=api_models.BBoxXYZ(
                x_minmax=api_models.FloatRange(min=0.0, max=1.0),
                y_minmax=api_models.FloatRange(min=0.0, max=1.0),
                z_minmax=api_models.FloatRange(min=0.0, max=1.0),
            ),
            block_rotation=[api_models.Rotation(axis=api_models.RotationAxis.x, angle=0.0)],
            bm_uuid=uuid.uuid4(),
            coordinate_reference_system=None,
            created_at=now,
            created_by=api_models.UserInfo(email="a@b.c", id=uuid.uuid4(), name="creator"),
            description=None,
            fill_subblocks=False,
            geoscience_object_id=None,
            last_updated_at=now,
            last_updated_by=api_models.UserInfo(email="u@b.c", id=uuid.uuid4(), name="updater"),
            model_origin=api_models.Location(x=0.0, y=0.0, z=0.0),
            name=name,
            normalized_rotation=[0.0, 0.0, 0.0],
            org_uuid=uuid.uuid4(),
            size_options=size_options,
            size_unit_id=None,
            workspace_id=uuid.uuid4(),
        )

    async def test_regular_size_options_map_to_regular_grid_definition(self) -> None:
        size_options = api_models.SizeOptionsRegular(
            model_type="regular",
            n_blocks=api_models.Size3D(nx=4, ny=5, nz=6),
            block_size=api_models.BlockSize(x=1.0, y=2.0, z=3.0),
        )
        endpoint_bm = self._create_bm("regular", size_options)

        bm = self.client._bm_from_model(endpoint_bm)
        assert isinstance(bm.grid_definition, RegularGridDefinition)
        assert bm.grid_definition.n_blocks == [4, 5, 6]
        assert bm.grid_definition.block_size == [1.0, 2.0, 3.0]

    async def test_fully_subblocked_maps_to_fullysubblocked_definition(self) -> None:
        size_options = api_models.SizeOptionsFullySubBlocked(
            model_type="fully-sub-blocked",
            n_parent_blocks=api_models.Size3D(nx=2, ny=3, nz=4),
            n_subblocks_per_parent=api_models.RegularSubblocks(nx=2, ny=2, nz=2),
            parent_block_size=api_models.BlockSize(x=4.0, y=4.0, z=4.0),
        )
        endpoint_bm = self._create_bm("full", size_options)

        bm = self.client._bm_from_model(endpoint_bm)
        assert isinstance(bm.grid_definition, FullySubBlockedGridDefinition)
        gd = bm.grid_definition
        assert gd.n_parent_blocks == [2, 3, 4]
        assert gd.n_subblocks_per_parent == [2, 2, 2]

    async def test_flexible_maps_to_flexible_definition(self) -> None:
        size_options = api_models.SizeOptionsFlexible(
            model_type="flexible",
            n_parent_blocks=api_models.Size3D(nx=1, ny=2, nz=3),
            n_subblocks_per_parent=api_models.RegularSubblocks(nx=3, ny=3, nz=3),
            parent_block_size=api_models.BlockSize(x=3.0, y=3.0, z=3.0),
        )
        endpoint_bm = self._create_bm("flex", size_options)

        bm = self.client._bm_from_model(endpoint_bm)
        assert isinstance(bm.grid_definition, FlexibleGridDefinition)
        gd = bm.grid_definition
        assert gd.n_parent_blocks == [1, 2, 3]
        assert gd.n_subblocks_per_parent == [3, 3, 3]

    async def test_octree_maps_to_octree_definition(self) -> None:
        size_options = api_models.SizeOptionsOctree(
            model_type="variable-octree",
            n_parent_blocks=api_models.Size3D(nx=1, ny=1, nz=1),
            n_subblocks_per_parent=api_models.OctreeSubblocks(nx=2, ny=2, nz=2),
            parent_block_size=api_models.BlockSize(x=2.0, y=2.0, z=2.0),
        )
        endpoint_bm = self._create_bm("oct", size_options)

        bm = self.client._bm_from_model(endpoint_bm)
        assert isinstance(bm.grid_definition, OctreeGridDefinition)
        gd = bm.grid_definition
        assert gd.n_parent_blocks == [1, 1, 1]
        assert gd.n_subblocks_per_parent == [2, 2, 2]
