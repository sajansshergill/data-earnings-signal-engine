"""
Parallel trends diagnostics for the DiD panel.
"""

from __future__ import annotations

import pandas as pd
import statsmodels.formula.api as smf


def pretrend_summary(
    df: pd.DataFrame,
    outcome_col: str = "log_gmv",
    treatment_col: str = "is_treatment",
    entity_col: str = "firm_id",
    time_col: str = "quarter",
    treatment_quarter: int = 7,
) -> pd.DataFrame:
    """Estimate treatment-control outcome gaps for pre-treatment quarters."""
    required = {outcome_col, treatment_col, entity_col, time_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    pre = df[df[time_col] < treatment_quarter].copy()
    if pre.empty:
        return pd.DataFrame(columns=["quarter", "treated_mean", "control_mean", "gap"])

    grouped = (
        pre.groupby([time_col, treatment_col])[outcome_col]
        .mean()
        .unstack(treatment_col)
        .rename(columns={0: "control_mean", 1: "treated_mean"})
    )
    for col in ("control_mean", "treated_mean"):
        if col not in grouped:
            grouped[col] = pd.NA
    grouped["gap"] = grouped["treated_mean"] - grouped["control_mean"]
    return grouped.reset_index().rename(columns={time_col: "quarter"})


def parallel_trends_p_value(
    df: pd.DataFrame,
    outcome_col: str = "log_gmv",
    treatment_col: str = "is_treatment",
    entity_col: str = "firm_id",
    time_col: str = "quarter",
    treatment_quarter: int = 7,
) -> float:
    """Return the joint F-test p-value for pre-period treatment interactions."""
    required = {outcome_col, treatment_col, entity_col, time_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    pre = df[df[time_col] < treatment_quarter].copy()
    if pre.empty or pre[time_col].nunique() < 2:
        return 1.0

    quarters = sorted(pre[time_col].unique())
    interaction_cols: list[str] = []
    for quarter in quarters[:-1]:
        col = f"treat_x_q{quarter}"
        pre[col] = ((pre[treatment_col] == 1) & (pre[time_col] == quarter)).astype(int)
        interaction_cols.append(col)

    if not interaction_cols:
        return 1.0

    formula = (
        f"{outcome_col} ~ {' + '.join(interaction_cols)} "
        f"+ C({time_col}) + C({entity_col})"
    )
    model = smf.ols(formula, data=pre).fit(cov_type="HC1")
    f_test = model.f_test([f"{col} = 0" for col in interaction_cols])
    return float(f_test.pvalue)
