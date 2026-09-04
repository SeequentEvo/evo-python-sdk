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

    def test_group_for_ungrouped_column(self) -> None:
        # columns[1] ("Au") has no group_uuid.
        self.assertIsNone(self.version.group_for_column(self.version.columns[1]))

    def test_group_for_column_with_dangling_group(self) -> None:
        # A column referencing a group that is not on this version resolves to None rather than guessing.
        orphan_column = models.Column(
            col_id=str(uuid.uuid4()), title="Cu", data_type=models.DataType.Float64, group_uuid=uuid.uuid4()
        )
        self.assertIsNone(self.version.group_for_column(orphan_column))

    def test_group_for_column_is_unambiguous_across_groups(self) -> None:
        # Two columns share the title "Cu" but live in different groups; each resolves to its own group
        # because the lookup uses the column's group_uuid, not its (non-unique) title.
        other_uuid = uuid.uuid4()
        version = Version(
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
                    col_id=str(uuid.uuid4()), title="Cu", data_type=models.DataType.Float64, group_uuid=CHILD_UUID
                ),
                models.Column(
                    col_id=str(uuid.uuid4()), title="Cu", data_type=models.DataType.Float64, group_uuid=other_uuid
                ),
            ],
            groups=[
                _resolved_group(CHILD_UUID, "Primary"),
                _resolved_group(other_uuid, "Secondary"),
            ],
        )
        self.assertEqual(version.group_for_column(version.columns[0]).group_uuid, CHILD_UUID)
        self.assertEqual(version.group_for_column(version.columns[1]).group_uuid, other_uuid)

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


class TestGroupInputModelsRejectExtras(unittest.TestCase):
    def test_group_definition_forbids_unknown_fields(self) -> None:
        from pydantic import ValidationError

        from evo.blockmodels.data import GroupDefinition

        with self.assertRaises(ValidationError):
            GroupDefinition(title="Assays", parnet_group="typo")

    def test_group_metadata_update_forbids_title_field(self) -> None:
        # The wire rename field is ``new_title``; passing ``title`` (the wire name) must be rejected
        # so it can't silently collide with the new_title -> title remap.
        from pydantic import ValidationError

        from evo.blockmodels.data import GroupMetadataUpdate

        with self.assertRaises(ValidationError):
            GroupMetadataUpdate(title="X")

    def test_column_metadata_update_accepts_group_field(self) -> None:
        # A column's group is metadata and can be changed without re-uploading data, so ``group`` is a
        # supported field. Only fields explicitly set are forwarded onto the wire.
        from evo.blockmodels.data import ColumnMetadataUpdate

        update = ColumnMetadataUpdate(group="Assays")
        self.assertEqual(update.group, "Assays")
        self.assertEqual(update.model_dump(exclude_unset=True), {"group": "Assays"})

        # An empty string ungroups the column.
        self.assertEqual(ColumnMetadataUpdate(group="").model_dump(exclude_unset=True), {"group": ""})

    def test_column_metadata_update_forbids_unknown_field(self) -> None:
        from pydantic import ValidationError

        from evo.blockmodels.data import ColumnMetadataUpdate

        with self.assertRaises(ValidationError):
            ColumnMetadataUpdate(not_a_field="x")


class TestQualifyColumnTitles(unittest.TestCase):
    def test_get_qualified_title_builds_qualified_and_bare_titles(self) -> None:
        from evo.blockmodels.data import get_qualified_title

        self.assertEqual(get_qualified_title("Assays", "Cu"), f"Assays{QUALIFIED_TITLE_SEPARATOR}Cu")
        self.assertEqual(
            get_qualified_title("Assays\u25b8Primary", "Cu"), f"Assays\u25b8Primary{QUALIFIED_TITLE_SEPARATOR}Cu"
        )
        # An ungrouped column keeps its bare title.
        self.assertEqual(get_qualified_title(None, "Cu"), "Cu")
        self.assertEqual(get_qualified_title("", "Cu"), "Cu")

    def test_qualify_column_titles_renames_and_builds_column_groups(self) -> None:
        import pyarrow

        from evo.blockmodels.data import qualify_column_titles

        data = pyarrow.table({"i": [1], "Cu": [2.0], "Au": [3.0], "rock": ["x"]})
        renamed, column_groups = qualify_column_titles(data, {"Cu": "Assays", "Au": "Assays"})

        self.assertEqual(
            renamed.schema.names,
            ["i", f"Assays{QUALIFIED_TITLE_SEPARATOR}Cu", f"Assays{QUALIFIED_TITLE_SEPARATOR}Au", "rock"],
        )
        self.assertEqual(
            column_groups,
            {f"Assays{QUALIFIED_TITLE_SEPARATOR}Cu": "Assays", f"Assays{QUALIFIED_TITLE_SEPARATOR}Au": "Assays"},
        )
        # Untouched columns keep their plain title and are absent from column_groups.
        self.assertEqual(renamed.column("i").to_pylist(), [1])

    def test_qualify_column_titles_treats_empty_group_as_ungrouped(self) -> None:
        import pyarrow

        from evo.blockmodels.data import qualify_column_titles

        data = pyarrow.table({"Cu": [2.0]})
        renamed, column_groups = qualify_column_titles(data, {"Cu": ""})

        self.assertEqual(renamed.schema.names, ["Cu"])
        self.assertEqual(column_groups, {})

    def test_qualify_column_titles_rejects_unknown_columns(self) -> None:
        import pyarrow

        from evo.blockmodels.data import qualify_column_titles

        data = pyarrow.table({"Cu": [2.0]})
        with self.assertRaises(KeyError):
            qualify_column_titles(data, {"Au": "Assays"})
