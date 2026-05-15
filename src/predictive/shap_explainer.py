"""
Small SHAP summary helpers used by notebooks and the dashboard.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def summarize_shap_values(
    shap_values: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    """Return mean absolute SHAP value by feature, sorted descending."""
    values = np.asarray(shap_values)
    if values.ndim != 2:
        raise ValueError("shap_values must be a 2D array")
    if values.shape[1] != len(feature_names):
        raise ValueError("feature_names length must match SHAP value columns")

    return (
        pd.DataFrame(
            {
                "feature": feature_names,
                "mean_shap_abs": np.abs(values).mean(axis=0),
            }
        )
        .sort_values("mean_shap_abs", ascending=False)
        .reset_index(drop=True)
    )


def top_shap_features(
    shap_values: np.ndarray,
    feature_names: list[str],
    n: int = 3,
) -> list[tuple[str, float]]:
    """Return the top ``n`` features as ``(feature, mean_abs_shap)`` tuples."""
    summary = summarize_shap_values(shap_values, feature_names)
    return list(summary.head(n).itertuples(index=False, name=None))
