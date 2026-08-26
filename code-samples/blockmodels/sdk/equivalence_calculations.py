"""Calculate zinc-equivalent values from block-model attributes.

The default column names and values replicate the Leapfrog calculation shown in
the accompanying project screenshots. Results use ``NaN`` where Leapfrog would
return ``outside``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EquivalenceSettings:
    """Prices, recoveries, defaults, and source-column names for equivalence."""

    silver_default_ppm: float = 0.02
    gold_default_ppm: float = 0.01
    lead_default_percent: float = 0.15
    zinc_default_percent: float = 0.15
    silver_price_oz_aud: float = 95.0
    gold_price_oz_aud: float = 6488.0
    lead_price_tonne_aud: float = 863.0
    zinc_price_tonne_aud: float = 5335.0
    silver_recovery: float = 0.4
    gold_recovery: float = 0.8
    lead_recovery: float = 0.8
    zinc_recovery: float = 0.3
    min_type_column: str = "MIN_TYPE (Refined Model)"
    included_min_types: tuple[str, ...] = (
        "MIN_TYPE_Selection: 230",
        "MIN_TYPE_Selection: 235",
    )
    silver_estimate_column: str = "Comb_Est_Ag_ID"
    gold_estimate_column: str = "Comb_Est_Au_ID"
    lead_estimate_column: str = "Comb_Est_Pb_ID"
    zinc_estimate_column: str = "Comb_Est_Zn_ID"


def calculate_zinc_equivalent(
    blocks: pd.DataFrame,
    settings: EquivalenceSettings | None = None,
) -> pd.DataFrame:
    """Return a copy of ``blocks`` with equivalent-grade calculation columns.

    Included blocks have a ``MIN_TYPE`` in ``settings.included_min_types``.
    For included blocks, a finite estimate is used; otherwise the metal's default
    grade is used. Excluded blocks receive ``NaN`` for all derived columns.
    """
    settings = settings or EquivalenceSettings()
    required_columns = [
        settings.min_type_column,
        settings.silver_estimate_column,
        settings.gold_estimate_column,
        settings.lead_estimate_column,
        settings.zinc_estimate_column,
    ]
    missing_columns = [column for column in required_columns if column not in blocks]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise KeyError(f"Block data is missing required columns: {missing}")

    result = blocks.copy()
    included = result[settings.min_type_column].isin(settings.included_min_types)

    result["Equivalence_Include"] = np.where(included, "Include", "Exclude")
    result["Ag_Used"] = _use_estimate_or_default(
        result[settings.silver_estimate_column], included, settings.silver_default_ppm
    )
    result["Au_Used"] = _use_estimate_or_default(
        result[settings.gold_estimate_column], included, settings.gold_default_ppm
    )
    result["Pb_Used"] = _use_estimate_or_default(
        result[settings.lead_estimate_column], included, settings.lead_default_percent
    )
    result["Zn_Used"] = _use_estimate_or_default(
        result[settings.zinc_estimate_column], included, settings.zinc_default_percent
    )

    result["Ag_Equiv_Calc"] = (
        result["Ag_Used"] * settings.silver_price_oz_aud / 31.10348 * settings.silver_recovery
    )
    result["Au_Equiv_Calc"] = (
        result["Au_Used"] * settings.gold_price_oz_aud / 31.10348 * settings.gold_recovery
    )
    result["Pb_Equiv_Calc"] = (
        result["Pb_Used"] * settings.lead_price_tonne_aud / 100 * settings.lead_recovery
    )
    result["Zn_Equiv_Calc"] = (
        result["Zn_Used"] * settings.zinc_price_tonne_aud / 100 * settings.zinc_recovery
    )
    result["Zn_Equiv%"] = (
        result[["Ag_Equiv_Calc", "Au_Equiv_Calc", "Pb_Equiv_Calc", "Zn_Equiv_Calc"]].sum(axis=1)
        / (settings.zinc_price_tonne_aud / 100 * settings.zinc_recovery)
    )
    result.loc[~included, [
        "Ag_Equiv_Calc",
        "Au_Equiv_Calc",
        "Pb_Equiv_Calc",
        "Zn_Equiv_Calc",
        "Zn_Equiv%",
    ]] = np.nan

    return result


def _use_estimate_or_default(
    values: pd.Series,
    included: pd.Series,
    default: float,
) -> pd.Series:
    """Apply Leapfrog's normal-value/default/outside decision to one metal."""
    numeric_values = pd.to_numeric(values, errors="coerce")
    valid_values = numeric_values.where(np.isfinite(numeric_values))
    return valid_values.where(included & valid_values.notna(), np.where(included, default, np.nan))