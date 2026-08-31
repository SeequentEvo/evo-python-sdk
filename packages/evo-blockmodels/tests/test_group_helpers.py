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

import unittest
import uuid
from datetime import datetime, timezone

from evo.blockmodels.data import QUALIFIED_TITLE_SEPARATOR, MissingColumnPolicy, Version
from evo.blockmodels.endpoints import models
from evo.common import ServiceUser

DATE = datetime(2021, 1, 1, tzinfo=timezone.utc)
USER = ServiceUser.from_model(models.IMSUserInfo(email="test@test.com", name="Test User", id=uuid.uuid4()))

PARENT_UUID = uuid.uuid4()
CHILD_UUID = uuid.uuid4()
COL_IN_CHILD = uuid.uuid4()


def _resolved_group(group_uuid, title, parent_group_uuid=None):
    return models.ResolvedGroup(
        group_uuid=group_uuid,
        title=title,
        parent_group_uuid=parent_group_uuid,
        missing_column_policy=MissingColumnPolicy.SET_NULL,
        resolved_missing_column_policy=MissingColumnPolicy.SET_NULL,
        tags={"source": "assay"},
    )


def _make_version() -> Version:
    return Version(
        bm_uuid=uuid.uuid4(),
        version_id=2,
        version_uuid=uuid.uuid4(),
        parent_version_id=1,
        base_version_id=1,
        geoscience_version_id="3",
        created_at=DATE,
        created_by=USER,
        comment="",
        columns=[
            models.Column(
                col_id=str(COL_IN_CHILD), title="Cu", data_type=models.DataType.Float64, group_uuid=CHILD_UUID
            ),
            models.Column(col_id=str(uuid.uuid4()), title="Au", data_type=models.DataType.Float64),
        ],
        groups=[
            _resolved_group(PARENT_UUID, "Assays"),
            _resolved_group(CHILD_UUID, "Primary", parent_group_uuid=PARENT_UUID),
        ],
    )


class TestVersionGroupHelpers(unittest.TestCase):
    def setUp(self) -> None:
        self.version = _make_version()

    def test_group_by_uuid(self) -> None:
        group = self.version.group_by_uuid(CHILD_UUID)
        self.assertIsNotNone(group)
        self.assertEqual(group.title, "Primary")
        self.assertIsNone(self.version.group_by_uuid(uuid.uuid4()))

    def test_qualified_group_title_nested(self) -> None:
        child = self.version.group_by_uuid(CHILD_UUID)
        self.assertEqual(
            self.version.qualified_group_title(child),
            f"Assays{QUALIFIED_TITLE_SEPARATOR}Primary",
        )

    def test_qualified_group_title_top_level(self) -> None:
        parent = self.version.group_by_uuid(PARENT_UUID)
        self.assertEqual(self.version.qualified_group_title(parent), "Assays")

    def test_qualified_group_title_custom_separator(self) -> None:
        child = self.version.group_by_uuid(CHILD_UUID)
        self.assertEqual(self.version.qualified_group_title(child, separator="/"), "Assays/Primary")

    def test_group_by_qualified_title_roundtrip(self) -> None:
        # A caller who created "Assays▸Primary" by title finds it again by its qualified title.
        group = self.version.group_by_qualified_title(f"Assays{QUALIFIED_TITLE_SEPARATOR}Primary")
        self.assertIsNotNone(group)
        self.assertEqual(group.group_uuid, CHILD_UUID)
        self.assertIsNone(self.version.group_by_qualified_title("Missing"))

    def test_group_for_column_by_object(self) -> None:
        column = self.version.columns[0]
        group = self.version.group_for_column(column)
        self.assertIsNotNone(group)
        self.assertEqual(group.group_uuid, CHILD_UUID)
        self.assertEqual(group.resolved_missing_column_policy, MissingColumnPolicy.SET_NULL)

    def test_group_for_column_by_title(self) -> None:
        group = self.version.group_for_column("Cu")
        self.assertIsNotNone(group)
        self.assertEqual(group.group_uuid, CHILD_UUID)

    def test_group_for_ungrouped_column(self) -> None:
        self.assertIsNone(self.version.group_for_column("Au"))

    def test_group_for_unknown_column(self) -> None:
        self.assertIsNone(self.version.group_for_column("does_not_exist"))

    def test_qualified_group_title_tolerates_broken_parent_chain(self) -> None:
        # A dangling parent reference must not loop or raise; it just stops walking.
        orphan = models.ResolvedGroup(
            group_uuid=uuid.uuid4(),
            title="Orphan",
            parent_group_uuid=uuid.uuid4(),
            missing_column_policy=MissingColumnPolicy.INHERIT,
            resolved_missing_column_policy=MissingColumnPolicy.USE_PREVIOUS,
        )
        object.__setattr__(self.version, "groups", [*self.version.groups, orphan])
        self.assertEqual(self.version.qualified_group_title(orphan), "Orphan")


if __name__ == "__main__":
    unittest.main()
