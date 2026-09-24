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

import uuid
from datetime import datetime

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
        """Test converting a legacy (bare-number) report result to DataFrame."""
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
                        {"categories": ["LMS1"], "values": [2.5]},
                        {"categories": ["LMS2"], "values": [3.2]},
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
        self.assertNotIn("Au Grade status", df.columns)

    def test_to_dataframe_include_status(self) -> None:
        """include_status=True adds a companion status column per value column."""
        df = self._new_schema_result().to_dataframe(include_status=True)
        self.assertIn("Au Grade status", df.columns)
        self.assertEqual(df.iloc[0]["Au Grade status"], "OK")
        self.assertEqual(df.iloc[1]["Au Grade status"], "NO_DATA")
        self.assertEqual(df.iloc[2]["Au Grade status"], "INVALID")

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
            result_sets=[{"cutoff_value": None, "rows": [{"categories": ["LMS1"], "values": [2.5]}]}],
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

    def test_report_cell_legacy_number(self) -> None:
        """A bare number (legacy schema) is upgraded to a ReportCell with OK status."""
        cell = models.ReportCell.model_validate(2.5)
        self.assertEqual(cell.value, 2.5)
        self.assertEqual(cell.status, ReportCellStatus.OK)

    def test_report_cell_legacy_null(self) -> None:
        """A bare null (legacy comparison percent) is upgraded to a ReportCell."""
        cell = models.ReportCell.model_validate(None)
        self.assertIsNone(cell.value)
        self.assertEqual(cell.status, ReportCellStatus.OK)

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

    def test_report_row_legacy_schema(self) -> None:
        """ReportRow accepts a legacy list of bare numbers and upgrades each to a ReportCell."""
        row = models.ReportRow.model_validate({"categories": ["LMS1"], "values": [2.0, 5.2]})
        self.assertEqual([c.value for c in row.values], [2.0, 5.2])
        self.assertTrue(all(c.status == ReportCellStatus.OK for c in row.values))

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

    def test_comparison_value_legacy_schema(self) -> None:
        """ReportComparisonValue accepts the legacy bare-number schema (percent may be null)."""
        value = models.ReportComparisonValue.model_validate(
            {"from_value": 0.0, "to_value": 3.0, "difference": 3.0, "percent": None}
        )
        self.assertEqual(value.from_value.value, 0.0)
        self.assertEqual(value.from_value.status, ReportCellStatus.OK)
        self.assertIsNone(value.percent.value)
        self.assertEqual(value.percent.status, ReportCellStatus.OK)

    def test_report_cell_isinstance_reused(self) -> None:
        """Validating an existing ReportCell instance returns an equivalent cell."""
        original = models.ReportCell(value=1.5, status=ReportCellStatus.OK)
        revalidated = models.ReportCell.model_validate(original)
        self.assertEqual(revalidated.value, 1.5)
        self.assertEqual(revalidated.status, ReportCellStatus.OK)
