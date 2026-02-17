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

"""Typed access for block model reports.

Reports provide resource estimation summaries for block models, allowing you to
calculate tonnages, grades, and metal content by category (e.g., geological domains).

Reports require:
1. Columns to have units set (e.g., grade in g/t, density in t/m³)
2. At least one category column for grouping (e.g., domain, rock type)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any
from uuid import UUID

import pandas as pd

from evo.common import IContext, IFeedback
from evo.common.utils import NoFeedback

if TYPE_CHECKING:
    from ..client import BlockModelAPIClient
    from ..endpoints.models import (
        ReportResult as APIReportResult,
    )
    from ..endpoints.models import (
        ReportSpecificationWithJobUrl,
        ReportSpecificationWithLastRunInfo,
    )

__all__ = [
    "Aggregation",
    "MassUnits",
    "Report",
    "ReportCategorySpec",
    "ReportColumnSpec",
    "ReportResult",
    "ReportSpecificationData",
]


class Aggregation(str, Enum):
    """Aggregation methods for report columns.

    Use these values for the `aggregation` parameter in `ReportColumnSpec`.

    Example:
        >>> col = ReportColumnSpec(
        ...     column_name="Au",
        ...     aggregation=Aggregation.MASS_AVERAGE,
        ...     output_unit_id="g/t",
        ... )
    """

    SUM = "SUM"
    """Sum of values - use for metal content, volume, tonnage, etc."""

    MASS_AVERAGE = "MASS_AVERAGE"
    """Mass-weighted average - use for grades, densities, quality metrics, etc."""


@dataclass(frozen=True, kw_only=True)
class ReportColumnSpec:
    """Specification for a column in a report.

    :param column_name: The name of the column in the block model.
    :param aggregation: How to aggregate the column values. Use `Aggregation` enum:
        - `Aggregation.MASS_AVERAGE` - Mass-weighted average (for grades)
        - `Aggregation.SUM` - Sum of values (for metal content)
    :param label: Display label for the column in the report.
    :param output_unit_id: Unit ID for the output values. Use `Units` class constants:
        - `Units.GRAMS_PER_TONNE` - g/t (grades)
        - `Units.PERCENT` - % (grades)
        - `Units.PPM` - ppm (grades)
        - `Units.KILOGRAMS` - kg (metal content)
        - `Units.TONNES` - t (metal content)
        - `Units.TROY_OUNCES` - oz_tr (metal content)

    Example:
        >>> from evo.blockmodels import Units
        >>> from evo.blockmodels.typed import Aggregation, ReportColumnSpec
        >>>
        >>> # For grade columns, use MASS_AVERAGE
        >>> grade_col = ReportColumnSpec(
        ...     column_name="Au",
        ...     aggregation=Aggregation.MASS_AVERAGE,
        ...     label="Au Grade",
        ...     output_unit_id=Units.GRAMS_PER_TONNE,
        ... )
        >>> # For metal content columns, use SUM
        >>> metal_col = ReportColumnSpec(
        ...     column_name="Au_metal",
        ...     aggregation=Aggregation.SUM,
        ...     label="Au Metal",
        ...     output_unit_id=Units.KILOGRAMS,
        ... )
    """

    column_name: str
    aggregation: Aggregation = Aggregation.SUM
    label: str | None = None
    output_unit_id: str | None = None

    def _get_label(self) -> str:
        """Get the label, defaulting to column_name if not set."""
        return self.label or self.column_name


@dataclass(frozen=True, kw_only=True)
class ReportCategorySpec:
    """Specification for a category column in a report.

    Category columns are used to group blocks for reporting (e.g., by domain, rock type).

    :param column_name: The name of the category column in the block model.
    :param label: Display label for the category in the report.
    :param values: Optional list of category values to include. If None, all values are included.
    """

    column_name: str
    label: str | None = None
    values: list[str] | None = None

    def _get_label(self) -> str:
        """Get the label, defaulting to column_name if not set."""
        return self.label or self.column_name


class MassUnits:
    """Common mass unit IDs for reports.

    Use these constants for the `mass_unit_id` parameter in `ReportSpecificationData`.

    Example:
        >>> report_data = ReportSpecificationData(
        ...     name="My Report",
        ...     columns=[...],
        ...     mass_unit_id=MassUnits.TONNES,
        ... )
    """

    TONNES = "t"
    """Metric tonnes"""

    KILOGRAMS = "kg"
    """Kilograms"""

    GRAMS = "g"
    """Grams"""

    OUNCES = "oz"
    """Troy ounces"""

    POUNDS = "lb"
    """Pounds"""


@dataclass(frozen=True, kw_only=True)
class ReportSpecificationData:
    """Data for creating a report specification.

    A report specification defines how to calculate resource estimates from a block model.
    It includes which columns to report on, how to categorize blocks, and density/mass settings.

    :param name: The name of the report.
    :param columns: List of columns to include in the report with their aggregation settings.
        Use `ReportColumnSpec` to define each column.
    :param mass_unit_id: Unit ID for mass output. Common values:
        - "t" (tonnes) - use `MassUnits.TONNES`
        - "kg" (kilograms) - use `MassUnits.KILOGRAMS`
        - "oz" (ounces) - use `MassUnits.OUNCES`
    :param categories: List of category columns for grouping blocks.
        Use `ReportCategorySpec` to define each category.
    :param description: Optional description of the report.
    :param density_value: Fixed density value (requires `density_unit_id`).
        Do NOT use with `density_column_name`.
    :param density_unit_id: Unit ID for fixed density (e.g., "t/m3").
        Only use with `density_value`, NOT with `density_column_name`.
    :param density_column_name: Name of the column containing block densities.
        Do NOT use with `density_value` or `density_unit_id`.
    :param cutoff_column_name: Name of the column to use for cut-off evaluation.
    :param cutoff_values: List of cut-off values to evaluate.
    :param autorun: Whether to automatically run the report when block model is updated.
    :param run_now: Whether to run the report immediately after creation.

    Example with density column:
        >>> data = ReportSpecificationData(
        ...     name="Gold Resource Report",
        ...     columns=[
        ...         ReportColumnSpec(
        ...             column_name="Au",
        ...             aggregation="MASS_AVERAGE",  # Use for grades
        ...             label="Au Grade",
        ...             output_unit_id="g/t",
        ...         ),
        ...     ],
        ...     categories=[
        ...         ReportCategorySpec(column_name="domain", label="Domain"),
        ...     ],
        ...     mass_unit_id=MassUnits.TONNES,
        ...     density_column_name="density",  # Unit comes from column
        ... )

    Example with fixed density:
        >>> data = ReportSpecificationData(
        ...     name="Gold Resource Report",
        ...     columns=[...],
        ...     categories=[...],
        ...     mass_unit_id=MassUnits.TONNES,
        ...     density_value=2.7,  # Fixed density
        ...     density_unit_id="t/m3",  # Required with density_value
        ... )
    """

    name: str
    columns: list[ReportColumnSpec]
    mass_unit_id: str

    categories: list[ReportCategorySpec] = field(default_factory=list)
    description: str | None = None
    density_value: float | None = None
    density_unit_id: str | None = None
    density_column_name: str | None = None
    cutoff_column_name: str | None = None
    cutoff_values: list[float] | None = None
    autorun: bool = True
    run_now: bool = True


@dataclass(frozen=True, kw_only=True)
class ReportResult:
    """A result from running a report.

    Contains the calculated values for each category and cut-off combination.
    """

    result_uuid: UUID
    report_specification_uuid: UUID
    block_model_uuid: UUID
    version_id: int
    version_uuid: UUID | None
    created_at: datetime
    categories: list[dict[str, Any]]
    columns: list[dict[str, Any]]
    result_sets: list[dict[str, Any]]

    @classmethod
    def _from_api_result(cls, result: "APIReportResult") -> "ReportResult":
        """Create a ReportResult from an API response."""
        return cls(
            result_uuid=result.report_result_uuid,
            report_specification_uuid=result.report_specification_uuid,
            block_model_uuid=result.bm_uuid,
            version_id=result.version_id,
            version_uuid=result.version_uuid,
            created_at=result.report_result_created_at,
            categories=[cat.model_dump() for cat in result.categories],
            columns=[col.model_dump() for col in result.value_columns],
            result_sets=[rs.model_dump() for rs in result.result_sets],
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the report result to a pandas DataFrame.

        Returns a DataFrame with one row per category/cut-off combination,
        containing the aggregated values for each report column.

        :return: DataFrame with report results.
        """
        rows = []
        column_labels = [col.get("label", f"Column {i}") for i, col in enumerate(self.columns)]

        for result_set in self.result_sets:
            cutoff = result_set.get("cutoff_value")
            for row_data in result_set.get("rows", []):
                row = {"cutoff": cutoff}
                # Add category values
                cat_values = row_data.get("categories", [])
                for i, cat in enumerate(self.categories):
                    cat_label = cat.get("label", f"Category {i}")
                    row[cat_label] = cat_values[i] if i < len(cat_values) else None
                # Add column values
                values = row_data.get("values", [])
                for i, label in enumerate(column_labels):
                    row[label] = values[i] if i < len(values) else None
                rows.append(row)

        return pd.DataFrame(rows)

    def __repr__(self) -> str:
        """Return a string representation of the report result."""
        df = self.to_dataframe()
        return (
            f"ReportResult(version={self.version_id}, "
            f"created={self.created_at.strftime('%Y-%m-%d %H:%M:%S')}, "
            f"rows={len(df)})\n{df.to_string()}"
        )


class Report:
    """A typed wrapper for block model report specifications.

    Reports provide resource estimation summaries for block models. They calculate
    tonnages, grades, and metal content grouped by categories (e.g., geological domains).

    Example usage:

        # Create a report from a block model
        report = await block_model.create_report(ReportSpecificationData(
            name="Resource Report",
            columns=[ReportColumnSpec(column_name="Au", output_unit_id="g/t")],
            categories=[ReportCategorySpec(column_name="domain")],
            mass_unit_id="t",
            density_value=2.7,
            density_unit_id="t/m3",
        ))

        # Pretty-print shows BlockSync link
        report

        # Get the latest result
        result = await report.get_latest_result()
        df = result.to_dataframe()
    """

    def __init__(
        self,
        context: IContext,
        block_model_uuid: UUID,
        specification: "ReportSpecificationWithLastRunInfo | ReportSpecificationWithJobUrl",
        block_model_name: str | None = None,
    ) -> None:
        """Initialize a Report instance.

        :param context: The context containing environment, connector, and cache.
        :param block_model_uuid: The UUID of the block model this report is for.
        :param specification: The report specification from the API.
        :param block_model_name: The name of the block model (for display purposes).
        """
        self._context = context
        self._block_model_uuid = block_model_uuid
        self._specification = specification
        self._block_model_name = block_model_name

    @property
    def id(self) -> UUID:
        """The unique identifier of the report specification."""
        return self._specification.report_specification_uuid

    @property
    def name(self) -> str:
        """The name of the report."""
        return self._specification.name

    @property
    def description(self) -> str | None:
        """The description of the report."""
        return self._specification.description

    @property
    def block_model_uuid(self) -> UUID:
        """The UUID of the block model this report is for."""
        return self._block_model_uuid

    @property
    def revision(self) -> int:
        """The revision number of the report specification."""
        return self._specification.revision

    def _get_client(self) -> "BlockModelAPIClient":
        """Get a BlockModelAPIClient for the current context."""
        from ..client import BlockModelAPIClient

        return BlockModelAPIClient.from_context(self._context)


    @classmethod
    async def create(
        cls,
        context: IContext,
        block_model_uuid: UUID,
        data: ReportSpecificationData,
        column_id_map: dict[str, UUID],
        fb: IFeedback = NoFeedback,
        block_model_name: str | None = None,
    ) -> "Report":
        """Create a new report specification.

        :param context: The context containing environment, connector, and cache.
        :param block_model_uuid: The UUID of the block model to create the report for.
        :param data: The report specification data.
        :param column_id_map: Mapping of column names to their UUIDs in the block model.
        :param fb: Optional feedback interface for progress reporting.
        :param block_model_name: The name of the block model (for display purposes).
        :return: A Report instance representing the created report.
        """
        from ..endpoints.models import (
            CreateReportSpecification,
            ReportAggregation,
            ReportCategory,
            ReportColumn,
        )

        fb.progress(0.0, "Creating report specification...")

        # Build columns list
        columns = []
        for col_spec in data.columns:
            col_id = column_id_map.get(col_spec.column_name)
            if col_id is None:
                raise ValueError(f"Column '{col_spec.column_name}' not found in block model")
            columns.append(
                ReportColumn(
                    col_id=col_id,
                    label=col_spec._get_label(),
                    aggregation=ReportAggregation(col_spec.aggregation.value),
                    output_unit_id=col_spec.output_unit_id or "",
                )
            )

        # Build categories list
        categories = None
        if data.categories:
            categories = []
            for cat_spec in data.categories:
                col_id = column_id_map.get(cat_spec.column_name)
                if col_id is None:
                    raise ValueError(f"Category column '{cat_spec.column_name}' not found in block model")
                categories.append(
                    ReportCategory(
                        col_id=col_id,
                        label=cat_spec._get_label(),
                        values=cat_spec.values,
                    )
                )

        # Build cutoff settings
        cutoff_col_id = None
        if data.cutoff_column_name:
            cutoff_col_id = column_id_map.get(data.cutoff_column_name)
            if cutoff_col_id is None:
                raise ValueError(f"Cut-off column '{data.cutoff_column_name}' not found in block model")

        # Build density settings
        density_col_id = None
        if data.density_column_name:
            density_col_id = column_id_map.get(data.density_column_name)
            if density_col_id is None:
                raise ValueError(f"Density column '{data.density_column_name}' not found in block model")

        # Create the specification
        spec = CreateReportSpecification(
            name=data.name,
            description=data.description,
            columns=columns,
            categories=categories,
            mass_unit_id=data.mass_unit_id,
            density_value=data.density_value,
            density_unit_id=data.density_unit_id,
            density_col_id=density_col_id,
            cutoff_col_id=cutoff_col_id,
            cutoff_values=data.cutoff_values,
            autorun=data.autorun,
        )

        fb.progress(0.3, "Submitting to Block Model Service...")

        # Call the API
        from ..client import BlockModelAPIClient

        client = BlockModelAPIClient.from_context(context)
        environment = context.get_environment()

        result = await client._reports_api.create_report_specification(
            workspace_id=str(environment.workspace_id),
            org_id=str(environment.org_id),
            bm_id=str(block_model_uuid),
            create_report_specification=spec,
            run_now=data.run_now,
        )

        fb.progress(1.0, "Report specification created")

        return cls(context, block_model_uuid, result, block_model_name=block_model_name)

    async def run(self, version_uuid: UUID | None = None, fb: IFeedback = NoFeedback) -> ReportResult:
        """Run the report to generate a new result.

        :param version_uuid: Optional specific version UUID to run the report on.
            If None, runs on the latest version.
        :param fb: Optional feedback interface for progress reporting.
        :return: The generated report result.
        """
        from ..endpoints.models import ReportingJobSpec

        fb.progress(0.0, "Running report...")

        client = self._get_client()
        environment = self._context.get_environment()

        # Create job spec
        job_spec = ReportingJobSpec(version_uuid=version_uuid)

        # Run the job
        job_result = await client._reports_api.run_reporting_job(
            rs_id=str(self.id),
            workspace_id=str(environment.workspace_id),
            org_id=str(environment.org_id),
            bm_id=str(self._block_model_uuid),
            reporting_job_spec=job_spec,
        )

        fb.progress(0.5, "Fetching result...")

        # Get the full result
        result = await client._reports_api.get_report_result(
            rs_id=str(self.id),
            report_result_uuid=str(job_result.report_result_uuid),
            workspace_id=str(environment.workspace_id),
            org_id=str(environment.org_id),
            bm_id=str(self._block_model_uuid),
        )

        fb.progress(1.0, "Report complete")

        return ReportResult._from_api_result(result)

    async def refresh(
        self,
        fb: IFeedback = NoFeedback,
        timeout_seconds: float = 120.0,
        poll_interval_seconds: float = 2.0,
    ) -> ReportResult:
        """Get the most recent result for this report, waiting if necessary.

        If no results exist yet (e.g., report is still running), this method will
        poll until a result is available or the timeout is reached.

        :param fb: Optional feedback interface for progress reporting.
        :param timeout_seconds: Maximum time to wait for results (default 120 seconds).
        :param poll_interval_seconds: Time between polling attempts (default 2 seconds).
        :return: The latest report result.
        :raises TimeoutError: If no results are available within the timeout period.
        """
        import asyncio
        import time

        fb.progress(0.0, "Fetching latest result...")

        client = self._get_client()
        environment = self._context.get_environment()

        start_time = time.time()
        attempt = 0

        while True:
            attempt += 1
            elapsed = time.time() - start_time

            # List results (ordered newest first)
            results_list = await client._reports_api.get_report_results_list(
                rs_id=str(self.id),
                workspace_id=str(environment.workspace_id),
                org_id=str(environment.org_id),
                bm_id=str(self._block_model_uuid),
                limit=1,
            )

            if results_list.results:
                # Get the full result
                latest = results_list.results[0]
                result = await client._reports_api.get_report_result(
                    rs_id=str(self.id),
                    report_result_uuid=str(latest.report_result_uuid),
                    workspace_id=str(environment.workspace_id),
                    org_id=str(environment.org_id),
                    bm_id=str(self._block_model_uuid),
                )

                fb.progress(1.0, "Result fetched")
                return ReportResult._from_api_result(result)

            # Check timeout
            if elapsed >= timeout_seconds:
                raise TimeoutError(
                    f"No report results available after {timeout_seconds} seconds. "
                    "The report may still be running - try again later."
                )

            # Report progress and wait
            progress = min(0.9, elapsed / timeout_seconds)
            fb.progress(progress, f"Waiting for results (attempt {attempt})...")
            await asyncio.sleep(poll_interval_seconds)

    async def list_results(self, limit: int = 50, fb: IFeedback = NoFeedback) -> list[ReportResult]:
        """List all results for this report.

        :param limit: Maximum number of results to return.
        :param fb: Optional feedback interface for progress reporting.
        :return: List of report results, ordered newest first.
        """
        fb.progress(0.0, "Fetching results...")

        client = self._get_client()
        environment = self._context.get_environment()

        # List results
        results_list = await client._reports_api.get_report_results_list(
            rs_id=str(self.id),
            workspace_id=str(environment.workspace_id),
            org_id=str(environment.org_id),
            bm_id=str(self._block_model_uuid),
            limit=limit,
        )

        fb.progress(0.5, f"Fetching {len(results_list.results)} results...")

        # Get full results
        results = []
        for i, summary in enumerate(results_list.results):
            result = await client._reports_api.get_report_result(
                rs_id=str(self.id),
                report_result_uuid=str(summary.report_result_uuid),
                workspace_id=str(environment.workspace_id),
                org_id=str(environment.org_id),
                bm_id=str(self._block_model_uuid),
            )
            results.append(ReportResult._from_api_result(result))
            fb.progress(0.5 + 0.5 * (i + 1) / len(results_list.results), f"Fetched {i + 1}/{len(results_list.results)}")

        return results
