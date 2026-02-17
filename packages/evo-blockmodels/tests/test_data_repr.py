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

"""Tests for pretty printing of block model related data classes."""

import unittest
from datetime import datetime, timezone
from uuid import uuid4

from evo.blockmodels.data import (
    BlockModel,
    FlexibleGridDefinition,
    FullySubBlockedGridDefinition,
    OctreeGridDefinition,
    RegularGridDefinition,
    Version,
)
from evo.blockmodels.endpoints.models import (
    BBox,
    BBoxXYZ,
    Column,
    DataType,
    FloatRange,
    IntRange,
    RotationAxis,
)
from evo.common import Environment
from evo.workspaces import ServiceUser


class TestVersionRepr(unittest.TestCase):
    """Test pretty printing for Version class."""

    def _create_test_version(
        self,
        version_id: int = 3,
        with_bbox: bool = True,
        with_comment: str = "Test comment",
    ) -> Version:
        """Create a Version object for testing."""
        columns = [
            Column(col_id="i", data_type=DataType.UInt32, title="i", unit_id=None),
            Column(col_id="j", data_type=DataType.UInt32, title="j", unit_id=None),
            Column(col_id="k", data_type=DataType.UInt32, title="k", unit_id=None),
            Column(col_id=str(uuid4()), data_type=DataType.Float64, title="grade", unit_id="g/t"),
            Column(col_id=str(uuid4()), data_type=DataType.Float64, title="density", unit_id="t/m3"),
        ]

        bbox = None
        if with_bbox:
            bbox = BBox(
                i_minmax=IntRange(min=0, max=9),
                j_minmax=IntRange(min=0, max=9),
                k_minmax=IntRange(min=0, max=4),
            )

        return Version(
            bm_uuid=uuid4(),
            version_id=version_id,
            version_uuid=uuid4(),
            parent_version_id=version_id - 1 if version_id > 1 else 0,
            base_version_id=version_id - 1 if version_id > 1 else None,
            geoscience_version_id="1770234750628962917",
            created_at=datetime(2026, 2, 4, 19, 52, 30, 120561, tzinfo=timezone.utc),
            created_by=ServiceUser(id=uuid4(), name="Denis Simo", email="Denis.Simo@bentley.com"),
            comment=with_comment,
            bbox=bbox,
            columns=columns,
        )

    def test_repr_returns_concise_string(self) -> None:
        """Test that __repr__ returns a concise, readable string."""
        version = self._create_test_version()
        repr_str = repr(version)

        # Should contain key info
        self.assertIn("Version(id=3", repr_str)
        self.assertIn("created=2026-02-04 19:52:30", repr_str)
        self.assertIn("by=Denis Simo", repr_str)
        self.assertIn("bbox=i[0-9] j[0-9] k[0-4]", repr_str)
        self.assertIn("columns=['i', 'j', 'k', 'grade', 'density']", repr_str)

    def test_repr_without_bbox(self) -> None:
        """Test that __repr__ works when bbox is None."""
        version = self._create_test_version(with_bbox=False)
        repr_str = repr(version)

        self.assertIn("Version(id=3", repr_str)
        self.assertNotIn("bbox=i[", repr_str)

    def test_repr_with_email_fallback(self) -> None:
        """Test that repr falls back to email when name is None."""
        columns = [
            Column(col_id="i", data_type=DataType.UInt32, title="i", unit_id=None),
        ]
        version = Version(
            bm_uuid=uuid4(),
            version_id=1,
            version_uuid=uuid4(),
            parent_version_id=0,
            base_version_id=None,
            geoscience_version_id="123",
            created_at=datetime(2026, 2, 4, 19, 52, 30, tzinfo=timezone.utc),
            created_by=ServiceUser(id=uuid4(), name=None, email="test@example.com"),
            comment="",
            bbox=None,
            columns=columns,
        )
        repr_str = repr(version)
        self.assertIn("by=test@example.com", repr_str)


class TestGridDefinitionRepr(unittest.TestCase):
    """Test repr for grid definition classes."""

    def test_regular_grid_definition_default_repr(self) -> None:
        """Test that RegularGridDefinition has a readable default repr."""
        grid = RegularGridDefinition(
            model_origin=[0.0, 0.0, 0.0],
            rotations=[(RotationAxis.x, 0.0)],
            n_blocks=[10, 10, 5],
            block_size=[1.0, 1.0, 2.0],
        )
        repr_str = repr(grid)

        # Default dataclass repr should include key fields
        self.assertIn("RegularGridDefinition", repr_str)
        self.assertIn("n_blocks=[10, 10, 5]", repr_str)
        self.assertIn("block_size=[1.0, 1.0, 2.0]", repr_str)

    def test_fully_subblocked_grid_definition_default_repr(self) -> None:
        """Test that FullySubBlockedGridDefinition has a readable default repr."""
        grid = FullySubBlockedGridDefinition(
            model_origin=[0.0, 0.0, 0.0],
            rotations=[],
            n_parent_blocks=[5, 5, 5],
            n_subblocks_per_parent=[2, 2, 2],
            parent_block_size=[10.0, 10.0, 10.0],
        )
        repr_str = repr(grid)

        self.assertIn("FullySubBlockedGridDefinition", repr_str)
        self.assertIn("n_parent_blocks=[5, 5, 5]", repr_str)
        self.assertIn("n_subblocks_per_parent=[2, 2, 2]", repr_str)

    def test_flexible_grid_definition_default_repr(self) -> None:
        """Test that FlexibleGridDefinition has a readable default repr."""
        grid = FlexibleGridDefinition(
            model_origin=[100.0, 200.0, 300.0],
            rotations=[(RotationAxis.z, 45.0)],
            n_parent_blocks=[8, 8, 4],
            n_subblocks_per_parent=[4, 4, 2],
            parent_block_size=[5.0, 5.0, 10.0],
        )
        repr_str = repr(grid)

        self.assertIn("FlexibleGridDefinition", repr_str)
        self.assertIn("n_parent_blocks=[8, 8, 4]", repr_str)

    def test_octree_grid_definition_default_repr(self) -> None:
        """Test that OctreeGridDefinition has a readable default repr."""
        grid = OctreeGridDefinition(
            model_origin=[0.0, 0.0, 0.0],
            rotations=[],
            n_parent_blocks=[4, 4, 4],
            n_subblocks_per_parent=[8, 8, 8],
            parent_block_size=[100.0, 100.0, 100.0],
        )
        repr_str = repr(grid)

        self.assertIn("OctreeGridDefinition", repr_str)
        self.assertIn("n_parent_blocks=[4, 4, 4]", repr_str)


class TestBlockModelRepr(unittest.TestCase):
    """Test repr for BlockModel class."""

    def _create_test_environment(self) -> Environment:
        """Create a test environment."""
        return Environment(
            hub_url="https://example.evo.bentley.com",
            org_id=uuid4(),
            workspace_id=uuid4(),
        )

    def _create_test_block_model(self) -> BlockModel:
        """Create a BlockModel object for testing."""
        return BlockModel(
            id=uuid4(),
            name="Test Block Model",
            environment=self._create_test_environment(),
            created_at=datetime(2026, 1, 15, 10, 30, 0, tzinfo=timezone.utc),
            created_by=ServiceUser(id=uuid4(), name="Creator", email="creator@example.com"),
            geoscience_object_id=uuid4(),
            description="A test block model for unit testing",
            grid_definition=RegularGridDefinition(
                model_origin=[1000.0, 2000.0, -500.0],
                rotations=[(RotationAxis.z, 15.0)],
                n_blocks=[100, 100, 50],
                block_size=[5.0, 5.0, 10.0],
            ),
            coordinate_reference_system="EPSG:32650",
            size_unit_id="m",
            bbox=BBoxXYZ(
                x_minmax=FloatRange(min=1000.0, max=1500.0),
                y_minmax=FloatRange(min=2000.0, max=2500.0),
                z_minmax=FloatRange(min=-500.0, max=0.0),
            ),
            last_updated_at=datetime(2026, 2, 1, 14, 45, 30, tzinfo=timezone.utc),
            last_updated_by=ServiceUser(id=uuid4(), name="Updater", email="updater@example.com"),
        )

    def test_block_model_default_repr(self) -> None:
        """Test that BlockModel has a readable default repr."""
        bm = self._create_test_block_model()
        repr_str = repr(bm)

        # Default dataclass repr should include key fields
        self.assertIn("BlockModel", repr_str)
        self.assertIn("Test Block Model", repr_str)
        self.assertIn("RegularGridDefinition", repr_str)


class TestBBoxRepr(unittest.TestCase):
    """Test repr for BBox and BBoxXYZ classes."""

    def test_bbox_repr(self) -> None:
        """Test that BBox has a readable repr."""
        bbox = BBox(
            i_minmax=IntRange(min=0, max=99),
            j_minmax=IntRange(min=0, max=99),
            k_minmax=IntRange(min=0, max=49),
        )
        repr_str = repr(bbox)

        self.assertIn("BBox", repr_str)
        self.assertIn("i_minmax", repr_str)
        self.assertIn("j_minmax", repr_str)
        self.assertIn("k_minmax", repr_str)

    def test_bbox_xyz_repr(self) -> None:
        """Test that BBoxXYZ has a readable repr."""
        bbox = BBoxXYZ(
            x_minmax=FloatRange(min=0.0, max=1000.0),
            y_minmax=FloatRange(min=0.0, max=1000.0),
            z_minmax=FloatRange(min=-500.0, max=0.0),
        )
        repr_str = repr(bbox)

        self.assertIn("BBoxXYZ", repr_str)
        self.assertIn("x_minmax", repr_str)
        self.assertIn("y_minmax", repr_str)
        self.assertIn("z_minmax", repr_str)


class TestColumnRepr(unittest.TestCase):
    """Test repr for Column class."""

    def test_column_repr_with_unit(self) -> None:
        """Test that Column with unit has a readable repr."""
        col = Column(
            col_id="abc123",
            data_type=DataType.Float64,
            title="grade",
            unit_id="g/t",
        )
        repr_str = repr(col)

        self.assertIn("Column", repr_str)
        self.assertIn("grade", repr_str)
        self.assertIn("Float64", repr_str)
        self.assertIn("g/t", repr_str)

    def test_column_repr_without_unit(self) -> None:
        """Test that Column without unit has a readable repr."""
        col = Column(
            col_id="i",
            data_type=DataType.UInt32,
            title="i",
            unit_id=None,
        )
        repr_str = repr(col)

        self.assertIn("Column", repr_str)
        self.assertIn("UInt32", repr_str)


if __name__ == "__main__":
    unittest.main()
