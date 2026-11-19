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

import json
import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from types import SimpleNamespace

import pandas as pd

from evo.blockmodels.endpoints import models
from evo.blockmodels.typed import (
    Aggregation,
    MassUnits,
    Report,
    ReportCategorySpec,
    ReportCellStatus,
    ReportColumnSpec,
    ReportResult,
    ReportSpecificationData,
    SpatialRelation,
    VolumeFilter,
)
from evo.common import StaticContext
from evo.common.test_tools import TestWithConnector, TestWithStorage

BM_UUID = uuid.uuid4()
RS_UUID = uuid.uuid4()
RESULT_UUID = uuid.uuid4()
VERSION_UUID = uuid.uuid4()
COL_UUID = uuid.uuid4()
CAT_COL_UUID = uuid.uuid4()
DATE = datetime(2026, 1, 1)
MESH_UUID = uuid.uuid4()


class TestReportSpecificationData(TestWithConnector):
    """Tests for ReportSpecificationData dataclass."""

    def test_basic_creation(self) -> None:
        """Test creating a basic report specification."""
        data = ReportSpecificationData(
            name="Test Report",
            columns=[
                ReportColumnSpec(column_name="Au", aggregation="MASS_AVERAGE", output_unit_id="g/t"),
            ],
            mass_unit_id="t",
        )
        self.assertEqual(data.name, "Test Report")
        self.assertEqual(len(data.columns), 1)
        self.assertEqual(data.columns[0].column_name, "Au")
        self.assertEqual(data.mass_unit_id, "t")
        self.assertTrue(data.autorun)
        self.assertTrue(data.run_now)

    def test_with_categories(self) -> None:
        """Test creating a report specification with categories."""
        data = ReportSpecificationData(
            name="Test Report",
            columns=[
                ReportColumnSpec(column_name="Au", output_unit_id="g/t"),
            ],
            categories=[
                ReportCategorySpec(column_name="domain", label="Domain", values=["LMS1", "LMS2"]),
            ],
            mass_unit_id="t",
            density_value=2.7,
            density_unit_id="t/m3",
        )
        self.assertEqual(len(data.categories), 1)
        self.assertEqual(data.categories[0].column_name, "domain")
        self.assertEqual(data.categories[0].values, ["LMS1", "LMS2"])
        self.assertEqual(data.density_value, 2.7)

    def test_column_spec_default_label(self) -> None:
        """Test that column spec label defaults to column name."""
        spec = ReportColumnSpec(column_name="Au")
        self.assertEqual(spec._get_label(), "Au")

        spec_with_label = ReportColumnSpec(column_name="Au", label="Gold Grade")
        self.assertEqual(spec_with_label._get_label(), "Gold Grade")


class TestReportResult(TestWithConnector):
    """Tests for ReportResult class."""

    def test_to_dataframe(self) -> None:
        """Test converting a report result to DataFrame."""
        result = ReportResult(
            result_uuid=RESULT_UUID,
            report_specification_uuid=RS_UUID,
            block_model_uuid=BM_UUID,
            version_id=1,
            version_uuid=VERSION_UUID,
            created_at=DATE,
            categories=[{"label": "Domain", "col_id": str(CAT_COL_UUID)}],
            columns=[{"label": "Au Grade", "unit_id": "g/t"}],
            result_sets=[
                {
                    "cutoff_value": 0.5,
                    "rows": [
                        {"categories": ["LMS1"], "values": [{"value": 2.5, "status": "OK"}]},
                        {"categories": ["LMS2"], "values": [{"value": 3.2, "status": "OK"}]},
                    ],
                },
            ],
        )

        df = result.to_dataframe()
        self.assertEqual(len(df), 2)
        self.assertIn("cutoff", df.columns)
        self.assertIn("Domain", df.columns)
        self.assertIn("Au Grade", df.columns)
        self.assertEqual(df.iloc[0]["Domain"], "LMS1")
        self.assertEqual(df.iloc[0]["Au Grade"], 2.5)

    def _new_schema_result(self) -> ReportResult:
        return ReportResult(
            result_uuid=RESULT_UUID,
            report_specification_uuid=RS_UUID,
            block_model_uuid=BM_UUID,
            version_id=1,
            version_uuid=VERSION_UUID,
            created_at=DATE,
            categories=[{"label": "Domain", "col_id": str(CAT_COL_UUID)}],
            columns=[{"label": "Au Grade", "unit_id": "g/t"}],
            result_sets=[
                {
                    "cutoff_value": 0.5,
                    "rows": [
                        {"categories": ["LMS1"], "values": [{"value": 2.5, "status": "OK"}]},
                        {"categories": ["LMS2"], "values": [{"value": None, "status": "NO_DATA"}]},
                        {"categories": ["LMS3"], "values": [{"value": None, "status": "INVALID"}]},
                    ],
                },
            ],
        )

    def test_to_dataframe_new_schema(self) -> None:
        """New ReportCell schema: DataFrame contains numeric value (or None) by default."""
        df = self._new_schema_result().to_dataframe()
        self.assertEqual(len(df), 3)
        self.assertEqual(df.iloc[0]["Au Grade"], 2.5)
        # null/invalid cells become None, rendered as NaN in the numeric column.
        self.assertTrue(pd.isna(df.iloc[1]["Au Grade"]))
        self.assertTrue(pd.isna(df.iloc[2]["Au Grade"]))
        # Status is not included unless requested.
        self.assertNotIn("Au Grade:status", df.columns)

    def test_to_dataframe_include_status(self) -> None:
        """include_status=True adds a companion status column per value column."""
        df = self._new_schema_result().to_dataframe(include_status=True)
        self.assertIn("Au Grade:status", df.columns)
        self.assertEqual(df.iloc[0]["Au Grade:status"], "OK")
        self.assertEqual(df.iloc[1]["Au Grade:status"], "NO_DATA")
        self.assertEqual(df.iloc[2]["Au Grade:status"], "INVALID")

    def test_repr(self) -> None:
        """Test string representation of report result."""
        result = ReportResult(
            result_uuid=RESULT_UUID,
            report_specification_uuid=RS_UUID,
            block_model_uuid=BM_UUID,
            version_id=1,
            version_uuid=VERSION_UUID,
            created_at=DATE,
            categories=[{"label": "Domain", "col_id": str(CAT_COL_UUID)}],
            columns=[{"label": "Au Grade", "unit_id": "g/t"}],
            result_sets=[
                {"cutoff_value": None, "rows": [{"categories": ["LMS1"], "values": [{"value": 2.5, "status": "OK"}]}]}
            ],
        )

        repr_str = repr(result)
        self.assertIn("ReportResult", repr_str)
        self.assertIn("version=1", repr_str)
        self.assertIn("rows=1", repr_str)


class TestReport(TestWithConnector, TestWithStorage):
    """Tests for Report class."""

    def setUp(self) -> None:
        TestWithConnector.setUp(self)
        TestWithStorage.setUp(self)

    def _mock_specification(self) -> models.ReportSpecificationWithLastRunInfo:
        return models.ReportSpecificationWithLastRunInfo(
            report_specification_uuid=RS_UUID,
            bm_uuid=BM_UUID,
            name="Test Report",
            description="A test report",
            revision=1,
            autorun=True,
            mass_unit_id="t",
            columns=[
                models.ReportColumn(
                    col_id=COL_UUID,
                    label="Au Grade",
                    aggregation=models.ReportAggregation.MASS_AVERAGE,
                    output_unit_id="g/t",
                )
            ],
            categories=[
                models.ReportCategory(
                    col_id=CAT_COL_UUID,
                    label="Domain",
                    values=["LMS1", "LMS2", "LMS3"],
                )
            ],
        )

    def test_report_properties(self) -> None:
        """Test Report properties."""
        spec = self._mock_specification()
        context = StaticContext.from_environment(self.environment, self.connector, self.cache)
        report = Report(context, BM_UUID, spec)

        self.assertEqual(report.id, RS_UUID)
        self.assertEqual(report.name, "Test Report")
        self.assertEqual(report.description, "A test report")
        self.assertEqual(report.block_model_uuid, BM_UUID)
        self.assertEqual(report.revision, 1)


class TestReportColumnSpec(TestWithConnector):
    """Tests for ReportColumnSpec dataclass."""

    def test_defaults(self) -> None:
        """Test default values."""
        spec = ReportColumnSpec(column_name="Au")
        self.assertEqual(spec.aggregation, "SUM")
        self.assertIsNone(spec.label)
        self.assertIsNone(spec.output_unit_id)

    def test_custom_values(self) -> None:
        """Test custom values."""
        spec = ReportColumnSpec(
            column_name="Au",
            aggregation=Aggregation.MASS_AVERAGE,
            label="Gold Grade",
            output_unit_id="g/t",
        )
        self.assertEqual(spec.aggregation, Aggregation.MASS_AVERAGE)
        self.assertEqual(spec.label, "Gold Grade")
        self.assertEqual(spec.output_unit_id, "g/t")


class TestReportCategorySpec(TestWithConnector):
    """Tests for ReportCategorySpec dataclass."""

    def test_defaults(self) -> None:
        """Test default values."""
        spec = ReportCategorySpec(column_name="domain")
        self.assertIsNone(spec.label)
        self.assertIsNone(spec.values)

    def test_with_values(self) -> None:
        """Test with explicit values."""
        spec = ReportCategorySpec(
            column_name="domain",
            label="Domain",
            values=["LMS1", "LMS2", "LMS3"],
        )
        self.assertEqual(spec.label, "Domain")
        self.assertEqual(spec.values, ["LMS1", "LMS2", "LMS3"])


class TestMassUnits(TestWithConnector):
    """Tests for MassUnits helper class."""

    def test_mass_unit_constants(self) -> None:
        """Test that MassUnits provides expected constants."""
        self.assertEqual(MassUnits.TONNES, "t")
        self.assertEqual(MassUnits.KILOGRAMS, "kg")
        self.assertEqual(MassUnits.GRAMS, "g")
        self.assertEqual(MassUnits.OUNCES, "oz")
        self.assertEqual(MassUnits.POUNDS, "lb")

    def test_use_in_report_spec(self) -> None:
        """Test using MassUnits in ReportSpecificationData."""
        data = ReportSpecificationData(
            name="Test Report",
            columns=[ReportColumnSpec(column_name="Au")],
            mass_unit_id=MassUnits.TONNES,
        )
        self.assertEqual(data.mass_unit_id, "t")


class TestReportCellSchema(TestWithConnector):
    """Tests for parsing the report-cell schema in the generated wire models."""

    def test_report_cell_new_schema(self) -> None:
        """A ReportCell object parses into value + status."""
        cell = models.ReportCell.model_validate({"value": 2.5, "status": "OK"})
        self.assertEqual(cell.value, 2.5)
        self.assertEqual(cell.status, ReportCellStatus.OK)

        invalid = models.ReportCell.model_validate({"value": None, "status": "INVALID"})
        self.assertIsNone(invalid.value)
        self.assertEqual(invalid.status, ReportCellStatus.INVALID)

    def test_report_row_new_schema(self) -> None:
        """ReportRow accepts a list of ReportCell objects."""
        row = models.ReportRow.model_validate(
            {
                "categories": ["LMS1"],
                "values": [{"value": 2.0, "status": "OK"}, {"value": None, "status": "NO_DATA"}],
            }
        )
        self.assertEqual(row.values[0].value, 2.0)
        self.assertEqual(row.values[0].status, ReportCellStatus.OK)
        self.assertIsNone(row.values[1].value)
        self.assertEqual(row.values[1].status, ReportCellStatus.NO_DATA)

    def test_comparison_value_new_schema(self) -> None:
        """ReportComparisonValue accepts ReportCell fields, including a null percent."""
        value = models.ReportComparisonValue.model_validate(
            {
                "from_value": {"value": 0.0, "status": "OK"},
                "to_value": {"value": 3.0, "status": "OK"},
                "difference": {"value": 3.0, "status": "OK"},
                "percent": {"value": None, "status": "PERCENT_CHANGE_FROM_ZERO"},
            }
        )
        self.assertEqual(value.from_value.value, 0.0)
        self.assertIsNone(value.percent.value)
        self.assertEqual(value.percent.status, ReportCellStatus.PERCENT_CHANGE_FROM_ZERO)

    def test_report_cell_isinstance_reused(self) -> None:
        """Validating an existing ReportCell instance returns an equivalent cell."""
        original = models.ReportCell(value=1.5, status=ReportCellStatus.OK)
        revalidated = models.ReportCell.model_validate(original)
        self.assertEqual(revalidated.value, 1.5)
        self.assertEqual(revalidated.status, ReportCellStatus.OK)


def _mesh_metadata(sub_classification: str = "triangle-mesh", workspace_id: uuid.UUID | None = None) -> SimpleNamespace:
    """Duck-typed stand-in for evo-objects ObjectMetadata."""
    return SimpleNamespace(
        id=MESH_UUID,
        version_id="1700000000000",
        schema_id=SimpleNamespace(sub_classification=sub_classification),
        environment=SimpleNamespace(workspace_id=workspace_id or uuid.uuid4()),
    )


def _api_volume_filter_options(selector: str | None = None) -> dict:
    mesh = {"geoscience_object_id": str(MESH_UUID), "geoscience_version_id": "1700000000000"}
    if selector is not None:
        mesh["selector"] = selector
    return {
        "volume_filter": {"mesh": mesh, "spatial_relation": "OUTSIDE"},
        "volume_filter_block_inclusion": "CENTROID_IN",
    }


class TestVolumeFilter(TestWithConnector):
    """Tests for VolumeFilter conversion and construction."""

    def test_to_api(self) -> None:
        options = VolumeFilter(mesh_id=MESH_UUID, relation=SpatialRelation.OUTSIDE)._to_api()
        self.assertEqual(
            options.model_dump(mode="json", exclude_unset=True),
            {
                "volume_filter": {
                    "mesh": {"geoscience_object_id": str(MESH_UUID), "geoscience_version_id": None},
                    "spatial_relation": "OUTSIDE",
                },
                "volume_filter_block_inclusion": "CENTROID_IN",
            },
        )

    def test_from_api_round_trip(self) -> None:
        options = models.VolumeFilterOptionsOutput.model_validate(_api_volume_filter_options())
        self.assertEqual(
            VolumeFilter._from_api(options),
            VolumeFilter(mesh_id=MESH_UUID, relation=SpatialRelation.OUTSIDE, mesh_version_id="1700000000000"),
        )
        self.assertIsNone(VolumeFilter._from_api(None))

    def test_from_api_unsupported_expressions(self) -> None:
        mesh = {"mesh": {"geoscience_object_id": str(MESH_UUID)}, "spatial_relation": "INSIDE"}
        group = {
            "volume_filter": {"operator": "UNION", "volumes": [mesh, mesh]},
            "volume_filter_block_inclusion": "CENTROID_IN",
        }
        for payload in (group, _api_volume_filter_options(selector="/parts/0")):
            with self.subTest(payload=payload), self.assertRaises(NotImplementedError):
                VolumeFilter._from_api(models.VolumeFilterOptionsOutput.model_validate(payload))

    def test_from_object(self) -> None:
        workspace_id = uuid.uuid4()
        metadata = _mesh_metadata(workspace_id=workspace_id)

        latest = VolumeFilter.from_object(metadata)
        self.assertEqual(latest.mesh_id, MESH_UUID)
        self.assertEqual(latest.relation, SpatialRelation.INSIDE)
        self.assertIsNone(latest.mesh_version_id)

        pinned = VolumeFilter.from_object(SimpleNamespace(metadata=metadata), "OUTSIDE", pin_version=True)
        self.assertEqual(pinned.relation, SpatialRelation.OUTSIDE)
        self.assertEqual(pinned.mesh_version_id, "1700000000000")

        reference = SimpleNamespace(object_id=str(MESH_UUID), workspace_id=workspace_id, version_id=None)
        self.assertEqual(VolumeFilter.from_object(reference).mesh_id, MESH_UUID)

    def test_from_object_rejects_invalid_objects(self) -> None:
        with self.assertRaisesRegex(ValueError, "triangle-mesh"):
            VolumeFilter.from_object(_mesh_metadata(sub_classification="pointset"))
        with self.assertRaisesRegex(ValueError, "no object ID"):
            VolumeFilter.from_object(SimpleNamespace())

    def test_workspace_check(self) -> None:
        workspace_id = uuid.uuid4()
        volume_filter = VolumeFilter.from_object(_mesh_metadata(workspace_id=workspace_id))
        volume_filter._check_workspace(workspace_id)
        with self.assertRaisesRegex(ValueError, "same workspace"):
            volume_filter._check_workspace(uuid.uuid4())
        # Filters created from a bare ID can't be checked client-side.
        VolumeFilter(mesh_id=MESH_UUID)._check_workspace(uuid.uuid4())


class TestReportVolumeFilter(TestWithConnector, TestWithStorage):
    """Tests for sending and reading volume filters through Report."""

    def setUp(self) -> None:
        TestWithConnector.setUp(self)
        TestWithStorage.setUp(self)
        self.context = StaticContext.from_environment(self.environment, self.connector, self.cache)

    def _spec_response(self, **overrides) -> dict:
        return {
            "report_specification_uuid": str(RS_UUID),
            "bm_uuid": str(BM_UUID),
            "name": "Test Report",
            "description": "A test report",
            "revision": 1,
            "autorun": True,
            "mass_unit_id": "t",
            "density_value": 2.7,
            "density_unit_id": "t/m3",
            "columns": [
                {"col_id": str(COL_UUID), "label": "Au", "aggregation": "MASS_AVERAGE", "output_unit_id": "g/t"}
            ],
            "categories": [{"col_id": str(CAT_COL_UUID), "label": "Domain", "values": ["LMS1"]}],
            **overrides,
        }

    def _last_request(self) -> tuple[Mapping[str, str], dict]:
        kwargs = self.transport.request.call_args.kwargs
        body = kwargs["body"]
        return kwargs["headers"], json.loads(body) if isinstance(body, (str, bytes)) else body

    async def _create(self, volume_filter: VolumeFilter | None, preview: bool) -> Report:
        data = ReportSpecificationData(
            name="Test Report",
            columns=[ReportColumnSpec(column_name="Au", output_unit_id="g/t")],
            categories=[ReportCategorySpec(column_name="Domain")],
            mass_unit_id="t",
            density_value=2.7,
            density_unit_id="t/m3",
            volume_filter=volume_filter,
        )
        with self.transport.set_http_response(201, json.dumps(self._spec_response())):
            return await Report.create(
                self.context, BM_UUID, data, {"Au": COL_UUID, "Domain": CAT_COL_UUID}, preview=preview
            )

    async def test_create_without_volume_filter(self) -> None:
        await self._create(None, preview=False)
        headers, body = self._last_request()
        self.assertNotIn("volume_filter_options", body)
        self.assertNotIn("API-Preview", headers)

    async def test_create_with_volume_filter(self) -> None:
        report = await self._create(VolumeFilter(mesh_id=MESH_UUID), preview=True)
        headers, body = self._last_request()
        self.assertEqual(headers["API-Preview"], "opt-in")
        self.assertEqual(body["volume_filter_options"]["volume_filter"]["mesh"]["geoscience_object_id"], str(MESH_UUID))
        self.assertEqual(body["volume_filter_options"]["volume_filter"]["spatial_relation"], "INSIDE")
        self.assertTrue(report._preview)

    async def test_create_rejects_mesh_from_other_workspace(self) -> None:
        volume_filter = VolumeFilter.from_object(_mesh_metadata(workspace_id=uuid.uuid4()))
        with self.assertRaisesRegex(ValueError, "same workspace"):
            await self._create(volume_filter, preview=True)
        self.transport.request.assert_not_called()

    def _report(self, **overrides) -> Report:
        spec = models.ReportSpecificationWithJobUrl.model_validate(self._spec_response(**overrides))
        return Report(self.context, BM_UUID, spec, preview=True)

    async def _update(self, report: Report, response: dict, method: str = "update", **kwargs) -> dict:
        with self.transport.set_http_response(200, json.dumps(response)):
            await getattr(report, method)(**kwargs)
        headers, body = self._last_request()
        self.assertEqual(headers["API-Preview"], "opt-in")
        return body

    async def test_update_sets_filter_and_keeps_spec(self) -> None:
        report = self._report()
        response = self._spec_response(revision=2, volume_filter_options=_api_volume_filter_options())
        body = await self._update(report, response, volume_filter=VolumeFilter(mesh_id=MESH_UUID))

        self.assertEqual(body["name"], "Test Report")
        self.assertEqual(body["description"], "A test report")
        self.assertEqual(body["density_value"], 2.7)
        self.assertEqual(body["columns"][0]["col_id"], str(COL_UUID))
        self.assertEqual(body["volume_filter_options"]["volume_filter"]["mesh"]["geoscience_object_id"], str(MESH_UUID))
        self.assertEqual(report.revision, 2)
        self.assertEqual(report.volume_filter.relation, SpatialRelation.OUTSIDE)

    async def test_update_without_volume_filter_keeps_it(self) -> None:
        report = self._report(volume_filter_options=_api_volume_filter_options())

        body = await self._update(report, self._spec_response(name="Renamed"), name="Renamed")

        self.assertEqual(body["name"], "Renamed")
        self.assertEqual(body["description"], "A test report")
        self.assertEqual(body["volume_filter_options"]["volume_filter"]["spatial_relation"], "OUTSIDE")

    async def test_remove_volume_filter(self) -> None:
        report = self._report(volume_filter_options=_api_volume_filter_options())

        body = await self._update(report, self._spec_response(), method="remove_volume_filter")

        self.assertIsNone(body["volume_filter_options"])
        self.assertEqual(body["name"], "Test Report")
        self.assertIsNone(report.volume_filter)

    def test_result_volume_filter_and_warnings(self) -> None:
        result = models.ReportResult.model_validate(
            {
                "bm_uuid": str(BM_UUID),
                "report_result_uuid": str(RESULT_UUID),
                "report_specification_uuid": str(RS_UUID),
                "report_specification_name": "Test Report",
                "report_specification_revision": 1,
                "report_result_created_at": DATE.replace(tzinfo=timezone.utc).isoformat(),
                "version_id": 1,
                "version_uuid": str(VERSION_UUID),
                "version_created_at": DATE.replace(tzinfo=timezone.utc).isoformat(),
                "version_created_by": {"id": str(uuid.uuid4()), "name": None, "email": None},
                "categories": [],
                "value_columns": [],
                "referenced_columns": [],
                "result_sets": [],
                "volume_filter_options": _api_volume_filter_options(),
                "warnings": [
                    {
                        "message": "The volume filter excluded all blocks",
                        "occurrences": 1,
                        "warning_type": "MESH_FILTER_EXCLUDED_ALL_BLOCKS",
                    }
                ],
            }
        )
        typed = ReportResult._from_api_result(result)
        self.assertEqual(typed.volume_filter.mesh_id, MESH_UUID)
        self.assertEqual(typed.warnings[0]["warning_type"], "MESH_FILTER_EXCLUDED_ALL_BLOCKS")
        self.assertIn("warnings=1", repr(typed))
