"""
did_model.py
────────────
Difference-in-Differences with Two-Way Fixed Effects (TWFE).

Model
-----
    log_GMV_{it} = α_i + λ_t + β·(Treat_i × Post_t) + ε_{it}

    α_i  — firm fixed effects   (absorb time-invariant confounders)
    λ_t  — time fixed effects   (absorb macro-level shocks)
    β    — ATT: Average Treatment effect on the Treated

Identification assumption: parallel trends.
We formally test this by regressing on pre-treatment period interactions
and checking whether pre-treatment coefficients are jointly zero.

Reference: Callaway & Sant'Anna (2021). Difference-in-differences with
multiple time periods. Journal of Econometrics, 225(2), 200–230.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


# ── Result container ───────────────────────────────────────────────────────────

@dataclass
class DiDResults:
    beta: float                         # ATT point estimate
    se: float                           # Robust standard error
    t_stat: float
    p_value: float
    ci_lower: float                     # 95% confidence interval
    ci_upper: float
    n_obs: int
    n_firms: int
    n_treatment_firms: int
    r_squared: float
    parallel_trends_p: float            # p-value for pre-trend test (want > 0.05)
    event_study_df: pd.DataFrame        # Quarter-level ATT estimates for plotting
    model_summary: str                  # Full statsmodels summary as string

    @property
    def is_significant(self) -> bool:
        return self.p_value < 0.05

    @property
    def parallel_trends_holds(self) -> bool:
        """True if we fail to reject H0 of no pre-treatment trends."""
        return self.parallel_trends_p > 0.05

    def __str__(self) -> str:
        sig = "***" if self.p_value < 0.01 else ("**" if self.p_value < 0.05 else "")
        pt_status = "✓ holds" if self.parallel_trends_holds else "✗ VIOLATED"
        return (
            f"DiD Results\n"
            f"{'─'*45}\n"
            f"  ATT (β):          {self.beta:+.4f}{sig}\n"
            f"  Std. Error:       {self.se:.4f}\n"
            f"  t-statistic:      {self.t_stat:.3f}\n"
            f"  p-value:          {self.p_value:.4f}\n"
            f"  95% CI:           [{self.ci_lower:.4f}, {self.ci_upper:.4f}]\n"
            f"  R²:               {self.r_squared:.4f}\n"
            f"  N obs:            {self.n_obs:,}\n"
            f"  Parallel trends:  {pt_status} (p={self.parallel_trends_p:.3f})\n"
            f"{'─'*45}"
        )


# ── Main estimator ─────────────────────────────────────────────────────────────

def run_did(
    df: pd.DataFrame,
    outcome_col: str = "log_gmv",
    treatment_col: str = "is_treatment",
    post_col: str = "post_event",
    entity_col: str = "firm_id",
    time_col: str = "quarter",
    treatment_quarter: int = 7,
) -> DiDResults:
    """
    Estimate ATT via TWFE DiD.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered panel (output of engineer_features).
    outcome_col : str
        Dependent variable. Default: log_gmv.
    treatment_col : str
        Binary indicator: 1 = treatment firm.
    post_col : str
        Binary indicator: 1 = post-treatment period.
    entity_col : str
        Firm identifier column.
    time_col : str
        Quarter column (integer 1–12).
    treatment_quarter : int
        First quarter of treatment (used for event study).

    Returns
    -------
    DiDResults
    """
    df = _prepare(df, outcome_col, treatment_col, post_col, entity_col, time_col)

    # ── TWFE via OLS with dummies ──────────────────────────────────────────────
    # We create the interaction term manually; C() dummies handle FE.
    df["did_interaction"] = df[treatment_col] * df[post_col]

    formula = (
        f"{outcome_col} ~ did_interaction "
        f"+ C({entity_col}) + C({time_col})"
    )

    model = smf.ols(formula, data=df).fit(
        cov_type="HC1"      # heteroskedasticity-robust SEs
    )

    beta = model.params["did_interaction"]
    se = model.bse["did_interaction"]
    t_stat = model.tvalues["did_interaction"]
    p_value = model.pvalues["did_interaction"]
    ci = model.conf_int(alpha=0.05).loc["did_interaction"]

    # ── Parallel trends test ───────────────────────────────────────────────────
    pt_p = _parallel_trends_test(df, outcome_col, treatment_col, time_col, treatment_quarter)

    # ── Event study ───────────────────────────────────────────────────────────
    event_study_df = _event_study(df, outcome_col, treatment_col, time_col, treatment_quarter)

    return DiDResults(
        beta=beta,
        se=se,
        t_stat=t_stat,
        p_value=p_value,
        ci_lower=ci.iloc[0],
        ci_upper=ci.iloc[1],
        n_obs=int(model.nobs),
        n_firms=df[entity_col].nunique(),
        n_treatment_firms=int(df.loc[df[treatment_col] == 1, entity_col].nunique()),
        r_squared=model.rsquared,
        parallel_trends_p=pt_p,
        event_study_df=event_study_df,
        model_summary=model.summary().as_text(),
    )


# ── Parallel trends test ───────────────────────────────────────────────────────

def _parallel_trends_test(
    df: pd.DataFrame,
    outcome_col: str,
    treatment_col: str,
    time_col: str,
    treatment_quarter: int,
) -> float:
    """
    Test for pre-treatment parallel trends.

    Regress the outcome on Treatment × Quarter_t for each pre-treatment
    quarter t. If pre-trends are absent, all interaction coefficients
    should be jointly zero (F-test p > 0.05).

    Returns the F-test p-value.
    """
    pre_df = df[df[time_col] < treatment_quarter].copy()
    if pre_df.empty or pre_df[time_col].nunique() < 2:
        return 1.0  # Not enough pre-periods to test

    # Create treatment × quarter_t dummies for each pre-period
    pre_quarters = sorted(pre_df[time_col].unique())
    interaction_cols = []

    for q in pre_quarters[:-1]:  # drop one quarter as reference
        col = f"treat_x_q{q}"
        pre_df[col] = (pre_df[treatment_col] == 1) & (pre_df[time_col] == q)
        pre_df[col] = pre_df[col].astype(int)
        interaction_cols.append(col)

    if not interaction_cols:
        return 1.0

    formula = (
        f"{outcome_col} ~ {' + '.join(interaction_cols)} "
        f"+ C({time_col}) + C(firm_id)"
    )

    try:
        model = smf.ols(formula, data=pre_df).fit(cov_type="HC1")
        f_test = model.f_test([f"{col} = 0" for col in interaction_cols])
        return float(f_test.pvalue)
    except Exception:
        return 1.0


# ── Event study ────────────────────────────────────────────────────────────────

def _event_study(
    df: pd.DataFrame,
    outcome_col: str,
    treatment_col: str,
    time_col: str,
    treatment_quarter: int,
) -> pd.DataFrame:
    """
    Estimate quarter-level ATT for an event study plot.

    Creates Treatment × Quarter_t interactions for every quarter,
    normalising to the quarter just before treatment (t = treatment_quarter - 1).
    """
    df = df.copy()
    quarters = sorted(df[time_col].unique())
    ref_quarter = treatment_quarter - 1

    interaction_cols = []
    for q in quarters:
        if q == ref_quarter:
            continue
        col = f"treat_x_q{q}"
        df[col] = ((df[treatment_col] == 1) & (df[time_col] == q)).astype(int)
        interaction_cols.append((q, col))

    formula = (
        f"{outcome_col} ~ {' + '.join(c for _, c in interaction_cols)} "
        f"+ C({time_col}) + C(firm_id)"
    )

    try:
        model = smf.ols(formula, data=df).fit(cov_type="HC1")
        ci = model.conf_int(alpha=0.05)

        records = []
        for q, col in interaction_cols:
            if col in model.params:
                records.append({
                    "quarter": q,
                    "relative_quarter": q - treatment_quarter,
                    "estimate": model.params[col],
                    "ci_lower": ci.loc[col].iloc[0],
                    "ci_upper": ci.loc[col].iloc[1],
                    "se": model.bse[col],
                })

        # Add reference quarter (normalised to 0)
        records.append({
            "quarter": ref_quarter,
            "relative_quarter": -1,
            "estimate": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "se": 0.0,
        })

        return pd.DataFrame(records).sort_values("quarter").reset_index(drop=True)

    except Exception as e:
        print(f"[warn] Event study failed: {e}")
        return pd.DataFrame()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _prepare(
    df: pd.DataFrame,
    outcome_col: str,
    treatment_col: str,
    post_col: str,
    entity_col: str,
    time_col: str,
) -> pd.DataFrame:
    """Validate and clean the DataFrame before modelling."""
    required = [outcome_col, treatment_col, post_col, entity_col, time_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.dropna(subset=[outcome_col]).copy()
    df[treatment_col] = df[treatment_col].astype(int)
    df[post_col] = df[post_col].astype(int)
    return df


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_event_study(
    results: DiDResults,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Render the event study plot (ATT per quarter with 95% CIs).

    The canonical YipitData diagnostic: flat pre-trends, post-treatment lift.
    """
    es = results.event_study_df
    if es.empty:
        raise ValueError("Event study DataFrame is empty — cannot plot.")

    fig, ax = plt.subplots(figsize=(10, 5))

    pre = es[es["relative_quarter"] < 0]
    post = es[es["relative_quarter"] >= 0]

    for subset, color, label in [
        (pre, "#4C72B0", "Pre-treatment"),
        (post, "#DD8452", "Post-treatment"),
    ]:
        ax.errorbar(
            subset["relative_quarter"],
            subset["estimate"],
            yerr=[
                subset["estimate"] - subset["ci_lower"],
                subset["ci_upper"] - subset["estimate"],
            ],
            fmt="o-",
            color=color,
            capsize=4,
            linewidth=1.8,
            markersize=6,
            label=label,
        )

    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.axvline(-0.5, color="red", linewidth=1.0, linestyle=":", alpha=0.7,
               label="Treatment onset")

    ax.set_xlabel("Quarter relative to treatment", fontsize=11)
    ax.set_ylabel("Estimated ATT (log GMV)", fontsize=11)
    ax.set_title(
        f"Event Study — DiD Treatment Effect\n"
        f"ATT = {results.beta:+.3f} (p={results.p_value:.3f})  |  "
        f"Parallel trends p={results.parallel_trends_p:.3f}",
        fontsize=12,
    )
    ax.legend(framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved event study plot → {save_path}")

    return fig


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.simulation.panel_generator import generate_panel, PanelConfig
    from src.features.engineer import engineer_features

    cfg = PanelConfig()
    panel = generate_panel(cfg)
    features = engineer_features(panel)

    print("Running DiD model...")
    results = run_did(features)
    print(results)

    fig = plot_event_study(results, save_path=Path("outputs/figures/event_study.png"))
    print("Event study plot saved.")