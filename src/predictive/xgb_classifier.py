"""
xgb_classifier.py
─────────────────
XGBoost earnings surprise classifier with walk-forward cross-validation.

Design decisions
----------------
- Walk-forward CV: train on Q1–Q(t-1), test on Qt. No look-ahead leakage.
- Recall-prioritised threshold: for institutional investors, missing a true
  beat (FN) is more costly than acting on a false signal (FP).
- SHAP integration: waterfall and beeswarm plots per prediction.
- DiD residuals are included as a meta-feature — links causal inference
  to the predictive layer, which is architecturally novel.

Reference
---------
Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system.
KDD '16. https://doi.org/10.1145/2939672.2939785
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)


# ── Config ─────────────────────────────────────────────────────────────────────

@dataclass
class XGBConfig:
    n_estimators: int = 300
    max_depth: int = 4
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 5
    gamma: float = 0.1
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    scale_pos_weight: float = 1.0     # Adjust for class imbalance if needed
    seed: int = 42
    recall_target: float = 0.80       # Optimise threshold to hit this recall floor
    eval_metric: str = "auc"


# ── Result container ───────────────────────────────────────────────────────────

@dataclass
class ClassifierResults:
    auc: float
    avg_precision: float
    precision_at_threshold: float
    recall_at_threshold: float
    f1_at_threshold: float
    threshold: float
    feature_importance: pd.DataFrame    # feature, mean_shap_abs
    oof_predictions: pd.DataFrame       # firm_id, quarter, y_true, y_prob, y_pred
    model: xgb.XGBClassifier
    shap_values: np.ndarray
    shap_explainer: Any
    X_test_last: pd.DataFrame           # Last fold test set (for waterfall plot)
    feature_cols: list[str]

    def __str__(self) -> str:
        return (
            f"Classifier Results\n"
            f"{'─'*45}\n"
            f"  AUC:                {self.auc:.4f}\n"
            f"  Avg Precision:      {self.avg_precision:.4f}\n"
            f"  Threshold:          {self.threshold:.4f}\n"
            f"  Precision @ thr:    {self.precision_at_threshold:.4f}\n"
            f"  Recall @ thr:       {self.recall_at_threshold:.4f}\n"
            f"  F1 @ thr:           {self.f1_at_threshold:.4f}\n"
            f"{'─'*45}"
        )


# ── Feature columns ────────────────────────────────────────────────────────────

DEFAULT_FEATURES: list[str] = [
    "cohort_retention_rate",
    "spend_velocity",
    "new_returning_ratio",
    "signal_momentum",
    "seasonality_adj_gmv",
    "gmv_growth_yoy",
    "log_gmv",
    "avg_order_value",
    "n_customers",
    "n_transactions",
]

TARGET: str = "earnings_beat"


# ── Main entry point ───────────────────────────────────────────────────────────

def train_and_evaluate(
    df: pd.DataFrame,
    feature_cols: Optional[list[str]] = None,
    did_residuals: Optional[pd.Series] = None,
    config: Optional[XGBConfig] = None,
    min_train_quarters: int = 4,
) -> ClassifierResults:
    """
    Walk-forward train/evaluate the XGBoost earnings surprise classifier.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered panel with all DEFAULT_FEATURES and TARGET.
    feature_cols : list[str], optional
        Override default feature list.
    did_residuals : pd.Series, optional
        DiD residuals indexed like df — added as meta-feature if provided.
    config : XGBConfig, optional
    min_train_quarters : int
        Minimum quarters needed in training set before first eval fold.

    Returns
    -------
    ClassifierResults
    """
    if config is None:
        config = XGBConfig()
    if feature_cols is None:
        feature_cols = DEFAULT_FEATURES

    df = _prepare(df, feature_cols, did_residuals)
    feature_cols_final = [c for c in feature_cols if c in df.columns]
    if did_residuals is not None and "did_residual" in df.columns:
        feature_cols_final = feature_cols_final + ["did_residual"]

    quarters = sorted(df["quarter"].unique())
    oof_records: list[dict] = []
    all_shap_values: list[np.ndarray] = []
    all_X_test: list[pd.DataFrame] = []

    model = None
    shap_values_out = None
    explainer = None

    for i, test_quarter in enumerate(quarters):
        if i < min_train_quarters:
            continue

        train = df[df["quarter"] < test_quarter]
        test = df[df["quarter"] == test_quarter]

        if train.empty or test.empty:
            continue

        X_train = train[feature_cols_final].fillna(0)
        y_train = train[TARGET]
        X_test = test[feature_cols_final].fillna(0)
        y_test = test[TARGET]

        model = _build_model(config)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        y_prob = model.predict_proba(X_test)[:, 1]

        for j, (idx, row) in enumerate(test.iterrows()):
            oof_records.append({
                "firm_id": row["firm_id"],
                "quarter": row["quarter"],
                "y_true": int(row[TARGET]),
                "y_prob": float(y_prob[j]),
            })

        # Collect SHAP-style contributions. Prefer shap when available, but fall
        # back to XGBoost's native pred_contribs for environments where shap's
        # compiled dependencies lag the installed NumPy version.
        explainer, sv = _shap_values(model, X_test)
        all_shap_values.append(sv)
        all_X_test.append(X_test)

    if model is None:
        raise RuntimeError("No folds evaluated — check min_train_quarters vs n_quarters.")

    oof_df = pd.DataFrame(oof_records)
    auc = roc_auc_score(oof_df["y_true"], oof_df["y_prob"])
    avg_precision = average_precision_score(oof_df["y_true"], oof_df["y_prob"])

    threshold = _find_threshold(oof_df["y_true"], oof_df["y_prob"], config.recall_target)
    oof_df["y_pred"] = (oof_df["y_prob"] >= threshold).astype(int)

    precision, recall, f1 = _prf_at_threshold(oof_df["y_true"], oof_df["y_prob"], threshold)

    # SHAP: aggregate across folds
    shap_values_out = np.vstack(all_shap_values)
    X_test_last = all_X_test[-1]

    feature_importance = pd.DataFrame({
        "feature": feature_cols_final,
        "mean_shap_abs": np.abs(shap_values_out).mean(axis=0),
    }).sort_values("mean_shap_abs", ascending=False).reset_index(drop=True)

    return ClassifierResults(
        auc=auc,
        avg_precision=avg_precision,
        precision_at_threshold=precision,
        recall_at_threshold=recall,
        f1_at_threshold=f1,
        threshold=threshold,
        feature_importance=feature_importance,
        oof_predictions=oof_df,
        model=model,
        shap_values=shap_values_out,
        shap_explainer=explainer,
        X_test_last=X_test_last,
        feature_cols=feature_cols_final,
    )


# ── Model builder ──────────────────────────────────────────────────────────────

def _build_model(config: XGBConfig) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        learning_rate=config.learning_rate,
        subsample=config.subsample,
        colsample_bytree=config.colsample_bytree,
        min_child_weight=config.min_child_weight,
        gamma=config.gamma,
        reg_alpha=config.reg_alpha,
        reg_lambda=config.reg_lambda,
        scale_pos_weight=config.scale_pos_weight,
        eval_metric=config.eval_metric,
        random_state=config.seed,
    )


def _load_shap():
    try:
        import shap

        return shap
    except ImportError:
        return None


def _shap_values(
    model: xgb.XGBClassifier,
    X: pd.DataFrame,
) -> tuple[Any, np.ndarray]:
    shap_lib = _load_shap()
    if shap_lib is not None:
        explainer = shap_lib.TreeExplainer(model)
        return explainer, np.asarray(explainer.shap_values(X))

    booster = model.get_booster()
    contribs = booster.predict(xgb.DMatrix(X), pred_contribs=True)
    return None, np.asarray(contribs[:, :-1])


# ── Threshold optimisation ─────────────────────────────────────────────────────

def _find_threshold(
    y_true: pd.Series,
    y_prob: pd.Series,
    recall_target: float,
) -> float:
    """
    Find the highest-precision threshold that achieves at least recall_target.
    Falls back to 0.5 if no threshold satisfies the constraint.

    Design note: for institutional investors, false negatives (missing
    a real earnings beat) carry asymmetric downside. We prioritise recall.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)

    # thresholds has one fewer element than precisions/recalls
    valid = np.flatnonzero(recalls[:-1] >= recall_target)
    if len(valid) > 0:
        best_idx = valid[np.argmax(precisions[:-1][valid])]
        return float(thresholds[best_idx])

    return 0.5  # fallback


def _prf_at_threshold(
    y_true: pd.Series,
    y_prob: pd.Series,
    threshold: float,
) -> tuple[float, float, float]:
    y_pred = (y_prob >= threshold).astype(int)
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    return float(precision), float(recall), float(f1)


# ── Data prep ──────────────────────────────────────────────────────────────────

def _prepare(
    df: pd.DataFrame,
    feature_cols: list[str],
    did_residuals: Optional[pd.Series],
) -> pd.DataFrame:
    df = df.copy()
    if did_residuals is not None:
        df["did_residual"] = did_residuals.values
    available = [c for c in feature_cols if c in df.columns]
    missing = set(feature_cols) - set(available)
    if missing:
        print(f"[warn] Missing feature columns (will skip): {missing}")
    return df


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_shap_beeswarm(
    results: ClassifierResults,
    save_path: Optional[Path] = None,
    max_display: int = 10,
) -> plt.Figure:
    """Global SHAP beeswarm (feature importance across all OOF predictions)."""
    shap_lib = _load_shap()
    if shap_lib is None:
        raise ImportError("shap is not available; cannot render beeswarm plot.")

    fig, ax = plt.subplots(figsize=(9, 6))

    shap_lib.summary_plot(
        results.shap_values,
        pd.concat([results.X_test_last] * max(1, len(results.shap_values) // len(results.X_test_last)),
                  ignore_index=True).iloc[:len(results.shap_values)],
        feature_names=results.feature_cols,
        max_display=max_display,
        show=False,
        plot_type="dot",
    )

    plt.title("SHAP Feature Importance — Earnings Surprise Classifier", fontsize=12)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved SHAP beeswarm → {save_path}")

    return plt.gcf()


def plot_shap_waterfall(
    results: ClassifierResults,
    obs_idx: int = 0,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """SHAP waterfall for a single prediction — used in white paper figure callout."""
    shap_lib = _load_shap()
    if shap_lib is None or results.shap_explainer is None:
        raise ImportError("shap is not available; cannot render waterfall plot.")

    X = results.X_test_last
    if obs_idx >= len(X):
        obs_idx = 0

    sv = results.shap_explainer(X.iloc[[obs_idx]])

    fig, ax = plt.subplots(figsize=(9, 5))
    shap_lib.plots.waterfall(sv[0], show=False, max_display=10)
    plt.title(f"SHAP Waterfall — Observation {obs_idx}", fontsize=11)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved SHAP waterfall → {save_path}")

    return plt.gcf()


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.simulation.panel_generator import generate_panel, PanelConfig
    from src.features.engineer import engineer_features

    cfg = PanelConfig()
    panel = generate_panel(cfg)
    features = engineer_features(panel)

    print("Training earnings surprise classifier...")
    results = train_and_evaluate(features)
    print(results)
    print("\nTop features by mean |SHAP|:")
    print(results.feature_importance.head(5).to_string(index=False))