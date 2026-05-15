"""
Lightweight synthetic-control baseline.

This is intentionally small: it provides a transparent comparator for the DiD
estimate by weighting control firms to match a treated firm's pre-period path.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass(frozen=True)
class SyntheticControlResult:
    treated_firm: str
    weights: pd.Series
    path: pd.DataFrame
    pre_rmse: float
    post_effect: float


def fit_synthetic_control(
    df: pd.DataFrame,
    treated_firm: str,
    outcome_col: str = "log_gmv",
    treatment_col: str = "is_treatment",
    entity_col: str = "firm_id",
    time_col: str = "quarter",
    treatment_quarter: int = 7,
) -> SyntheticControlResult:
    """Fit non-negative control weights that sum to one."""
    required = {outcome_col, treatment_col, entity_col, time_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    pivot = df.pivot(index=time_col, columns=entity_col, values=outcome_col).sort_index()
    if treated_firm not in pivot.columns:
        raise ValueError(f"Unknown treated firm: {treated_firm}")

    control_firms = (
        df.loc[df[treatment_col] == 0, entity_col]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    if not control_firms:
        raise ValueError("At least one control firm is required")

    pre_periods = [q for q in pivot.index if q < treatment_quarter]
    treated_pre = pivot.loc[pre_periods, treated_firm].to_numpy()
    controls_pre = pivot.loc[pre_periods, control_firms].to_numpy()

    n_controls = len(control_firms)
    initial = np.repeat(1.0 / n_controls, n_controls)
    bounds = [(0.0, 1.0)] * n_controls
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)

    def objective(weights: np.ndarray) -> float:
        synthetic = controls_pre @ weights
        return float(np.mean((treated_pre - synthetic) ** 2))

    opt = minimize(objective, initial, method="SLSQP", bounds=bounds, constraints=constraints)
    if not opt.success:
        raise RuntimeError(f"Synthetic control optimisation failed: {opt.message}")

    weights = pd.Series(opt.x, index=control_firms, name="weight")
    synthetic_path = pivot[control_firms] @ weights
    treated_path = pivot[treated_firm]
    path = pd.DataFrame(
        {
            "quarter": pivot.index,
            "treated": treated_path.to_numpy(),
            "synthetic": synthetic_path.to_numpy(),
        }
    )
    path["effect"] = path["treated"] - path["synthetic"]

    pre_rmse = float(np.sqrt(np.mean(path.loc[path["quarter"] < treatment_quarter, "effect"] ** 2)))
    post_effect = float(path.loc[path["quarter"] >= treatment_quarter, "effect"].mean())

    return SyntheticControlResult(
        treated_firm=treated_firm,
        weights=weights,
        path=path,
        pre_rmse=pre_rmse,
        post_effect=post_effect,
    )
