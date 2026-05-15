"""
panel_generator.py
──────────────────
Generates a synthetic transaction-level panel for 50 firms × 12 quarters.

Design decisions
----------------
- Treatment assignment is random but fixed at generation time (seed-controlled).
- Revenue inflection is injected at Q7 for treatment firms only.
- Noise is Gaussian, calibrated so pre-treatment parallel trends hold in expectation
  (necessary for DiD validity downstream).
- Output is Parquet for downstream compatibility with Spark / DuckDB.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class PanelConfig:
    n_firms: int = 50
    n_quarters: int = 12
    n_treatment_firms: int = 10
    treatment_quarter: int = 7          # 1-indexed; event happens at start of Q7
    treatment_effect: float = 0.15      # +15% GMV lift for treatment firms post-event
    base_gmv_mean: float = 10_000_000  # $10M baseline GMV per firm per quarter
    base_gmv_std: float = 2_000_000    # cross-firm heterogeneity
    noise_std_frac: float = 0.04       # quarter-level noise as fraction of GMV
    avg_order_value_mean: float = 85.0
    avg_order_value_std: float = 15.0
    new_customer_rate: float = 0.20    # fraction of customers who are new each Q
    seed: int = 42
    output_dir: Path = field(default_factory=lambda: Path("data/raw"))


# ── Core generator ─────────────────────────────────────────────────────────────

def generate_panel(config: PanelConfig | None = None) -> pd.DataFrame:
    """
    Generate the synthetic firm × quarter panel.

    Returns
    -------
    pd.DataFrame with columns:
        firm_id, quarter, is_treatment, post_event,
        gmv, n_transactions, avg_order_value,
        n_customers, n_new_customers, n_returning_customers,
        cohort_size, earnings_beat
    """
    if config is None:
        config = PanelConfig()

    rng = np.random.default_rng(config.seed)

    firms = _build_firms(config, rng)
    records = _build_records(firms, config, rng)
    panel = pd.DataFrame(records)
    panel = _add_earnings_labels(panel, config, rng)
    panel = panel.sort_values(["firm_id", "quarter"]).reset_index(drop=True)

    return panel


def _build_firms(config: PanelConfig, rng: np.random.Generator) -> pd.DataFrame:
    """Assign firm-level baseline characteristics and treatment status."""
    firm_ids = [f"firm_{i:03d}" for i in range(config.n_firms)]
    treatment_idx = rng.choice(config.n_firms, size=config.n_treatment_firms, replace=False)
    is_treatment = np.zeros(config.n_firms, dtype=bool)
    is_treatment[treatment_idx] = True

    # Firm-level GMV baselines — draw once, held fixed across quarters
    base_gmv = rng.normal(
        loc=config.base_gmv_mean,
        scale=config.base_gmv_std,
        size=config.n_firms,
    ).clip(min=1_000_000)  # floor at $1M

    base_aov = rng.normal(
        loc=config.avg_order_value_mean,
        scale=config.avg_order_value_std,
        size=config.n_firms,
    ).clip(min=20.0)

    return pd.DataFrame({
        "firm_id": firm_ids,
        "is_treatment": is_treatment,
        "base_gmv": base_gmv,
        "base_aov": base_aov,
    })


def _build_records(
    firms: pd.DataFrame,
    config: PanelConfig,
    rng: np.random.Generator,
) -> list[dict]:
    """Build one record per firm × quarter."""
    records = []
    quarters = list(range(1, config.n_quarters + 1))

    # Shared time trend — affects all firms equally (absorbed by time FE in DiD)
    time_trend = 1 + 0.02 * (np.array(quarters) - 1)  # +2% growth per quarter

    # Shared macro shock at Q5 — tests parallel trends pre-treatment (Q7)
    macro_shock = np.ones(config.n_quarters)
    if config.n_quarters >= 5:
        macro_shock[4] = 1.08  # +8% macro lift at Q5 for all firms

    for _, firm in firms.iterrows():
        for q_idx, q in enumerate(quarters):
            post_event = (q >= config.treatment_quarter) and firm["is_treatment"]

            # GMV = baseline × time_trend × macro × treatment_lift × noise
            treatment_multiplier = (1 + config.treatment_effect) if post_event else 1.0
            noise = rng.normal(1.0, config.noise_std_frac)

            gmv = (
                firm["base_gmv"]
                * time_trend[q_idx]
                * macro_shock[q_idx]
                * treatment_multiplier
                * noise
            )

            # AOV drifts slightly over time (consumers trade up)
            aov_drift = 1 + 0.005 * q_idx
            avg_order_value = firm["base_aov"] * aov_drift * rng.normal(1.0, 0.03)
            avg_order_value = max(avg_order_value, 10.0)

            n_transactions = int(gmv / avg_order_value)
            cohort_size = int(n_transactions * rng.uniform(0.6, 0.85))  # unique customers

            n_new = int(cohort_size * (config.new_customer_rate + rng.normal(0, 0.02)))
            n_new = max(0, min(n_new, cohort_size))
            n_returning = cohort_size - n_new

            records.append({
                "firm_id": firm["firm_id"],
                "quarter": q,
                "is_treatment": int(firm["is_treatment"]),
                "post_event": int(post_event),
                "gmv": round(gmv, 2),
                "n_transactions": n_transactions,
                "avg_order_value": round(avg_order_value, 2),
                "n_customers": cohort_size,
                "n_new_customers": n_new,
                "n_returning_customers": n_returning,
                "cohort_size": cohort_size,
                "base_gmv": round(firm["base_gmv"], 2),
            })

    return records


def _add_earnings_labels(
    panel: pd.DataFrame,
    config: PanelConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Attach a binary earnings_beat label per firm × quarter.

    Logic:
    - Treatment firms post-event beat with higher probability (signal is real).
    - Control firms and pre-event quarters: beat with base rate ~50%.
    - Adds realistic noise so the classifier has something to learn.
    """
    beat_prob = np.where(
        (panel["is_treatment"] == 1) & (panel["post_event"] == 1),
        0.72,   # treatment firms post-event beat more often
        0.48,   # base rate roughly coin-flip
    )
    panel["earnings_beat"] = rng.binomial(1, beat_prob).astype(int)
    return panel


# ── I/O helpers ───────────────────────────────────────────────────────────────

def save_panel(panel: pd.DataFrame, config: PanelConfig | None = None) -> Path:
    """Save panel to Parquet. Returns the output path."""
    if config is None:
        config = PanelConfig()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = config.output_dir / "transaction_panel.parquet"
    panel.to_parquet(out_path, index=False, engine="pyarrow")
    return out_path


def load_panel(path: Path | str | None = None) -> pd.DataFrame:
    """Load panel from Parquet."""
    if path is None:
        path = Path("data/raw/transaction_panel.parquet")
    return pd.read_parquet(path, engine="pyarrow")


# ── Summary stats ─────────────────────────────────────────────────────────────

def describe_panel(panel: pd.DataFrame) -> None:
    """Print a quick sanity-check summary of the generated panel."""
    n_firms = panel["firm_id"].nunique()
    n_quarters = panel["quarter"].nunique()
    n_treatment = panel.loc[panel["is_treatment"] == 1, "firm_id"].nunique()
    n_control = n_firms - n_treatment

    avg_gmv_pre_treat = panel.loc[
        (panel["is_treatment"] == 1) & (panel["quarter"] < 7), "gmv"
    ].mean()
    avg_gmv_post_treat = panel.loc[
        (panel["is_treatment"] == 1) & (panel["quarter"] >= 7), "gmv"
    ].mean()

    beat_rate_treatment_post = panel.loc[
        (panel["is_treatment"] == 1) & (panel["post_event"] == 1), "earnings_beat"
    ].mean()
    beat_rate_control = panel.loc[panel["is_treatment"] == 0, "earnings_beat"].mean()

    print("=" * 55)
    print("  PANEL SUMMARY")
    print("=" * 55)
    print(f"  Firms:           {n_firms} ({n_treatment} treatment, {n_control} control)")
    print(f"  Quarters:        {n_quarters}")
    print(f"  Total obs:       {len(panel):,}")
    print(f"  Avg GMV pre  (treatment):  ${avg_gmv_pre_treat:>12,.0f}")
    print(f"  Avg GMV post (treatment):  ${avg_gmv_post_treat:>12,.0f}")
    print(f"  Implied GMV lift:          {(avg_gmv_post_treat/avg_gmv_pre_treat - 1)*100:.1f}%")
    print(f"  Beat rate — treatment/post: {beat_rate_treatment_post:.2%}")
    print(f"  Beat rate — control:        {beat_rate_control:.2%}")
    print("=" * 55)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = PanelConfig()
    print("Generating panel...")
    panel = generate_panel(cfg)
    path = save_panel(panel, cfg)
    describe_panel(panel)
    print(f"\nSaved → {path}")