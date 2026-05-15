"""
engineer.py
───────────
Transforms the raw transaction panel into model-ready features.

All operations are vectorized (no Python-level loops on DataFrames).
This mirrors Spark-style thinking: every transformation is expressible
as a column operation or a group aggregation — no row-by-row logic.

Features produced
-----------------
cohort_retention_rate   Repeat customers Q(t) / new customers Q(t-1)
spend_velocity          ΔAOV / Δtime, rolling 2-quarter window
new_returning_ratio     New customers / returning customers
signal_momentum         EW-smoothed GMV growth (α = 0.3)
seasonality_adj_gmv     GMV de-seasonalised via ratio-to-moving-average
gmv_growth_yoy          YoY GMV growth (Q vs Q-4)
log_gmv                 log(GMV) — stabilises variance for modelling
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path


# ── Constants ──────────────────────────────────────────────────────────────────

EWM_ALPHA: float = 0.3          # EWM decay for signal_momentum
SEASONALITY_WINDOW: int = 4     # Rolling window for seasonal adjustment (4Q = 1 year)
MIN_RETURNING: int = 1          # Floor to avoid division by zero


# ── Main entry point ───────────────────────────────────────────────────────────

def engineer_features(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature transformations to the raw panel.

    Parameters
    ----------
    panel : pd.DataFrame
        Output of panel_generator.generate_panel()

    Returns
    -------
    pd.DataFrame
        Original columns plus all engineered features.
        Rows with insufficient history for lag-based features are retained
        but the lag columns will contain NaN (handled by downstream models).
    """
    df = panel.copy()
    df = df.sort_values(["firm_id", "quarter"]).reset_index(drop=True)

    df = _add_cohort_retention_rate(df)
    df = _add_spend_velocity(df)
    df = _add_new_returning_ratio(df)
    df = _add_signal_momentum(df)
    df = _add_seasonality_adjusted_gmv(df)
    df = _add_gmv_growth_yoy(df)
    df = _add_log_gmv(df)

    return df


# ── Feature transformations ────────────────────────────────────────────────────

def _add_cohort_retention_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    cohort_retention_rate = n_returning_customers(t) / n_new_customers(t-1)

    Interpretation: how well the firm retains customers acquired last quarter.
    High retention → durable revenue base → positive earnings signal.
    """
    df = df.sort_values(["firm_id", "quarter"])

    # Lag new customers by 1 quarter within each firm
    df["_lag_new_customers"] = (
        df.groupby("firm_id")["n_new_customers"]
        .shift(1)
    )

    df["cohort_retention_rate"] = (
        df["n_returning_customers"] / df["_lag_new_customers"].clip(lower=1)
    )

    df = df.drop(columns=["_lag_new_customers"])
    return df


def _add_spend_velocity(df: pd.DataFrame) -> pd.DataFrame:
    """
    spend_velocity = ΔAOV over rolling 2-quarter window.

    Captures whether customers are trading up (positive) or down (negative).
    Normalised by baseline AOV so it's comparable across firms.
    """
    df = df.sort_values(["firm_id", "quarter"])

    lag1_aov = df.groupby("firm_id")["avg_order_value"].shift(1)
    lag2_aov = df.groupby("firm_id")["avg_order_value"].shift(2)

    # (AOV_t - AOV_{t-2}) / AOV_{t-2}  — 2-quarter % change
    df["spend_velocity"] = (df["avg_order_value"] - lag2_aov) / lag2_aov.clip(lower=1e-6)

    return df


def _add_new_returning_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    new_returning_ratio = n_new_customers / n_returning_customers

    High ratio → firm is still in growth/acquisition mode.
    Low ratio → mature, retention-driven revenue — different risk profile.
    """
    df["new_returning_ratio"] = (
        df["n_new_customers"]
        / df["n_returning_customers"].clip(lower=MIN_RETURNING)
    )
    return df


def _add_signal_momentum(df: pd.DataFrame) -> pd.DataFrame:
    """
    signal_momentum = EWM(GMV growth rate, α=0.3)

    Exponential weighting means recent quarters carry more weight.
    This is the top SHAP feature in the earnings surprise classifier —
    momentum in alternative data leads earnings by 1–2 quarters.
    """
    df = df.sort_values(["firm_id", "quarter"])

    # Quarter-over-quarter GMV growth rate per firm
    lag_gmv = df.groupby("firm_id")["gmv"].shift(1)
    gmv_growth = (df["gmv"] - lag_gmv) / lag_gmv.clip(lower=1e-6)

    # Apply EWM per firm group
    df["signal_momentum"] = (
        df.assign(_gmv_growth=gmv_growth)
        .groupby("firm_id")["_gmv_growth"]
        .transform(lambda s: s.ewm(alpha=EWM_ALPHA, adjust=False).mean())
    )

    return df


def _add_seasonality_adjusted_gmv(df: pd.DataFrame) -> pd.DataFrame:
    """
    seasonality_adj_gmv = GMV / rolling_mean(GMV, 4Q)

    Ratio-to-moving-average decomposition: a value > 1 means this quarter
    is above trend; < 1 means below trend. Removes shared macro seasonality
    so the model sees idiosyncratic firm-level signal.
    """
    df = df.sort_values(["firm_id", "quarter"])

    rolling_mean = (
        df.groupby("firm_id")["gmv"]
        .transform(lambda s: s.rolling(window=SEASONALITY_WINDOW, min_periods=2).mean())
    )

    df["seasonality_adj_gmv"] = df["gmv"] / rolling_mean.clip(lower=1e-6)

    return df


def _add_gmv_growth_yoy(df: pd.DataFrame) -> pd.DataFrame:
    """
    gmv_growth_yoy = (GMV_t - GMV_{t-4}) / GMV_{t-4}

    Year-over-year growth removes within-year seasonality naturally.
    Institutional investors think in YoY terms; this aligns signal framing.
    """
    df = df.sort_values(["firm_id", "quarter"])

    lag4_gmv = df.groupby("firm_id")["gmv"].shift(4)

    df["gmv_growth_yoy"] = (df["gmv"] - lag4_gmv) / lag4_gmv.clip(lower=1e-6)

    return df


def _add_log_gmv(df: pd.DataFrame) -> pd.DataFrame:
    """
    log_gmv = log(GMV)

    Log transformation stabilises variance (GMV is right-skewed).
    Standard practice before using GMV as a regressor in the DiD model.
    """
    df["log_gmv"] = np.log(df["gmv"].clip(lower=1.0))
    return df


# ── Feature metadata ───────────────────────────────────────────────────────────

ENGINEERED_FEATURES: list[str] = [
    "cohort_retention_rate",
    "spend_velocity",
    "new_returning_ratio",
    "signal_momentum",
    "seasonality_adj_gmv",
    "gmv_growth_yoy",
    "log_gmv",
]

MODEL_FEATURES: list[str] = [
    # Engineered signals
    "cohort_retention_rate",
    "spend_velocity",
    "new_returning_ratio",
    "signal_momentum",
    "seasonality_adj_gmv",
    "gmv_growth_yoy",
    "log_gmv",
    # Raw panel features useful for the classifier
    "avg_order_value",
    "n_customers",
    "n_transactions",
]

TARGET: str = "earnings_beat"
TREATMENT_COL: str = "is_treatment"
POST_COL: str = "post_event"
TIME_COL: str = "quarter"
ENTITY_COL: str = "firm_id"


# ── I/O helpers ────────────────────────────────────────────────────────────────

def save_features(df: pd.DataFrame, output_dir: Path | str = "data/processed") -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "features.parquet"
    df.to_parquet(out_path, index=False, engine="pyarrow")
    return out_path


def load_features(path: Path | str = "data/processed/features.parquet") -> pd.DataFrame:
    return pd.read_parquet(path, engine="pyarrow")


# ── Diagnostic summary ─────────────────────────────────────────────────────────

def describe_features(df: pd.DataFrame) -> None:
    """Print null counts and descriptive stats for all engineered features."""
    print("=" * 60)
    print("  FEATURE SUMMARY")
    print("=" * 60)
    print(f"  Shape: {df.shape}")
    print()

    feature_df = df[ENGINEERED_FEATURES]
    null_counts = feature_df.isnull().sum()

    print(f"  {'Feature':<28} {'Nulls':>6}  {'Mean':>10}  {'Std':>10}")
    print("  " + "-" * 56)
    for col in ENGINEERED_FEATURES:
        nulls = null_counts[col]
        mean = df[col].mean()
        std = df[col].std()
        print(f"  {col:<28} {nulls:>6}  {mean:>10.4f}  {std:>10.4f}")
    print("=" * 60)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.simulation.panel_generator import generate_panel, PanelConfig

    cfg = PanelConfig()
    print("Generating panel...")
    panel = generate_panel(cfg)

    print("Engineering features...")
    features = engineer_features(panel)
    describe_features(features)

    path = save_features(features)
    print(f"\nSaved → {path}")