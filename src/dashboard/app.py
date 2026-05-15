"""
Streamlit dashboard for the synthetic earnings signal pipeline.

Run with:
    streamlit run src/dashboard/app.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.causal.did_model import DiDResults, run_did
from src.features.engineer import ENGINEERED_FEATURES, engineer_features
from src.llm.white_paper_agent import build_context
from src.predictive.xgb_classifier import ClassifierResults, XGBConfig, train_and_evaluate
from src.simulation.panel_generator import PanelConfig, generate_panel


@dataclass(frozen=True)
class DashboardResults:
    config: PanelConfig
    features: pd.DataFrame
    did_results: DiDResults
    classifier_results: ClassifierResults
    white_paper_context: dict


@st.cache_resource(show_spinner=False)
def _run_pipeline(
    n_firms: int,
    n_quarters: int,
    n_treatment_firms: int,
    treatment_quarter: int,
    treatment_effect: float,
    seed: int,
    n_estimators: int,
) -> DashboardResults:
    cfg = PanelConfig(
        n_firms=n_firms,
        n_quarters=n_quarters,
        n_treatment_firms=n_treatment_firms,
        treatment_quarter=treatment_quarter,
        treatment_effect=treatment_effect,
        seed=seed,
    )
    panel = generate_panel(cfg)
    features = engineer_features(panel)
    did_results = run_did(features, treatment_quarter=treatment_quarter)
    classifier_results = train_and_evaluate(
        features,
        config=XGBConfig(n_estimators=n_estimators, max_depth=3, learning_rate=0.06),
    )

    top_features = list(
        classifier_results.feature_importance[["feature", "mean_shap_abs"]]
        .head(3)
        .itertuples(index=False, name=None)
    )
    white_paper_context = build_context(
        did_beta=did_results.beta,
        did_se=did_results.se,
        did_ci_lower=did_results.ci_lower,
        did_ci_upper=did_results.ci_upper,
        did_p_value=did_results.p_value,
        parallel_trends_p=did_results.parallel_trends_p,
        auc=classifier_results.auc,
        precision=classifier_results.precision_at_threshold,
        recall=classifier_results.recall_at_threshold,
        threshold=classifier_results.threshold,
        top_features=top_features,
        n_firms=did_results.n_firms,
        n_quarters=n_quarters,
        treatment_effect_pct=float(np.exp(did_results.beta) - 1),
    )
    return DashboardResults(
        config=cfg,
        features=features,
        did_results=did_results,
        classifier_results=classifier_results,
        white_paper_context=white_paper_context,
    )


def _fmt_pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def _fmt_dollars(value: float) -> str:
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:.1f}M"
    if abs(value) >= 1_000:
        return f"${value / 1_000:.1f}K"
    return f"${value:,.0f}"


def _panel_summary(features: pd.DataFrame) -> pd.DataFrame:
    return (
        features.groupby(["quarter", "is_treatment"], as_index=False)
        .agg(
            avg_gmv=("gmv", "mean"),
            avg_customers=("n_customers", "mean"),
            beat_rate=("earnings_beat", "mean"),
            avg_momentum=("signal_momentum", "mean"),
        )
        .assign(group=lambda df: np.where(df["is_treatment"] == 1, "Treatment", "Control"))
    )


def _plot_gmv_trend(features: pd.DataFrame, treatment_quarter: int) -> go.Figure:
    summary = _panel_summary(features)
    fig = px.line(
        summary,
        x="quarter",
        y="avg_gmv",
        color="group",
        markers=True,
        title="Average GMV by Quarter",
        labels={"avg_gmv": "Average GMV", "quarter": "Quarter", "group": "Group"},
    )
    fig.add_vline(
        x=treatment_quarter - 0.5,
        line_dash="dash",
        line_color="#D62728",
        annotation_text="Treatment onset",
    )
    fig.update_layout(legend_title_text="", yaxis_tickprefix="$", hovermode="x unified")
    return fig


def _plot_event_study(results: DiDResults) -> go.Figure:
    es = results.event_study_df.copy()
    es["period"] = np.where(es["relative_quarter"] < 0, "Pre", "Post")
    fig = go.Figure()
    for period, color in [("Pre", "#4C78A8"), ("Post", "#F58518")]:
        subset = es[es["period"] == period]
        fig.add_trace(
            go.Scatter(
                x=subset["relative_quarter"],
                y=subset["estimate"],
                error_y={
                    "type": "data",
                    "array": subset["ci_upper"] - subset["estimate"],
                    "arrayminus": subset["estimate"] - subset["ci_lower"],
                    "visible": True,
                },
                mode="lines+markers",
                name=f"{period}-treatment",
                line={"color": color},
            )
        )
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.add_vline(x=-0.5, line_dash="dot", line_color="#D62728")
    fig.update_layout(
        title="Event Study: ATT by Quarter",
        xaxis_title="Quarter Relative to Treatment",
        yaxis_title="Estimated ATT (log GMV)",
        hovermode="x unified",
    )
    return fig


def _plot_feature_importance(results: ClassifierResults) -> go.Figure:
    top = results.feature_importance.head(10).sort_values("mean_shap_abs")
    fig = px.bar(
        top,
        x="mean_shap_abs",
        y="feature",
        orientation="h",
        title="Top Feature Contributions",
        labels={"mean_shap_abs": "Mean |SHAP|", "feature": ""},
    )
    fig.update_layout(showlegend=False)
    return fig


def _plot_prediction_distribution(results: ClassifierResults) -> go.Figure:
    oof = results.oof_predictions.copy()
    oof["label"] = np.where(oof["y_true"] == 1, "Beat", "Miss")
    fig = px.histogram(
        oof,
        x="y_prob",
        color="label",
        nbins=25,
        barmode="overlay",
        opacity=0.72,
        title="Out-of-Fold Probability Distribution",
        labels={"y_prob": "Predicted Beat Probability", "label": "Actual"},
    )
    fig.add_vline(
        x=results.threshold,
        line_dash="dash",
        line_color="#D62728",
        annotation_text=f"Threshold {results.threshold:.2f}",
    )
    return fig


def _draft_white_paper_preview(context: dict) -> str:
    causal = context["causal_inference"]
    predictive = context["predictive_model"]
    top_feature = predictive["top_features"][0]
    return (
        "The pipeline estimates revenue inflection using a Difference-in-Differences "
        "specification with firm and quarter fixed effects, following the identification "
        f"logic in {causal['reference']}. The estimated treatment effect is "
        f"beta = {causal['beta']}, with a 95% confidence interval from "
        f"{causal['ci_lower']} to {causal['ci_upper']} and p = {causal['p_value']}. "
        f"The parallel-trends diagnostic is {causal['parallel_trends_status']} "
        f"(p = {causal['parallel_trends_p']}), so the evidence is framed subject to "
        "that identifying assumption.\n\n"
        f"The predictive layer uses {predictive['method']} and reports AUC = "
        f"{predictive['auc']}, precision = {predictive['precision']}, and recall = "
        f"{predictive['recall']} at an operating threshold of "
        f"{predictive['operating_threshold']}. The largest model contribution is "
        f"{top_feature['feature']} with mean |SHAP| = {top_feature['mean_shap_abs']}. "
        "Figure 1 should show the event-study path, while Figure 2 should explain the "
        "top model contributions."
    )


def _render_sidebar() -> dict[str, int | float]:
    st.sidebar.header("Simulation Controls")
    n_firms = st.sidebar.slider("Firms", min_value=20, max_value=120, value=50, step=10)
    n_quarters = st.sidebar.slider("Quarters", min_value=8, max_value=16, value=12, step=1)
    max_treatment = max(2, n_firms // 2)
    n_treatment = st.sidebar.slider(
        "Treatment firms",
        min_value=2,
        max_value=max_treatment,
        value=min(10, max_treatment),
    )
    treatment_quarter = st.sidebar.slider(
        "Treatment quarter",
        min_value=3,
        max_value=max(3, n_quarters - 2),
        value=min(7, max(3, n_quarters - 2)),
    )
    treatment_effect = st.sidebar.slider(
        "Injected GMV lift",
        min_value=0.05,
        max_value=0.40,
        value=0.15,
        step=0.01,
        format="%.2f",
    )
    n_estimators = st.sidebar.slider("XGBoost trees", min_value=25, max_value=200, value=75, step=25)
    seed = st.sidebar.number_input("Random seed", min_value=0, value=42, step=1)

    st.sidebar.divider()
    st.sidebar.caption("Changing any control reruns the synthetic pipeline with cached results.")
    return {
        "n_firms": n_firms,
        "n_quarters": n_quarters,
        "n_treatment_firms": n_treatment,
        "treatment_quarter": treatment_quarter,
        "treatment_effect": treatment_effect,
        "seed": int(seed),
        "n_estimators": n_estimators,
    }


def main() -> None:
    st.set_page_config(page_title="Earnings Signal Engine", layout="wide")
    st.title("Alternative Data Earnings Signal Engine")
    st.caption(
        "Synthetic transaction panel, causal inference, earnings-surprise prediction, "
        "and white-paper readiness in one analyst dashboard."
    )

    controls = _render_sidebar()
    with st.spinner("Running simulation, DiD, and walk-forward classifier..."):
        results = _run_pipeline(**controls)

    features = results.features
    did = results.did_results
    clf = results.classifier_results
    implied_lift = float(np.exp(did.beta) - 1)
    post_beat_rate = features.loc[
        (features["is_treatment"] == 1) & (features["post_event"] == 1),
        "earnings_beat",
    ].mean()

    metric_cols = st.columns(5)
    metric_cols[0].metric("Panel Rows", f"{len(features):,}")
    metric_cols[1].metric("DiD ATT", f"{did.beta:+.3f}", delta=_fmt_pct(implied_lift))
    metric_cols[2].metric("Parallel Trends p", f"{did.parallel_trends_p:.3f}")
    metric_cols[3].metric("Classifier AUC", f"{clf.auc:.3f}")
    metric_cols[4].metric("Treatment/Post Beat Rate", _fmt_pct(post_beat_rate))

    tabs = st.tabs(["Overview", "Panel Data", "Causal Impact", "Predictive Model", "White Paper"])

    with tabs[0]:
        left, right = st.columns([1.2, 0.8])
        with left:
            st.plotly_chart(_plot_gmv_trend(features, results.config.treatment_quarter), use_container_width=True)
        with right:
            st.subheader("Run Summary")
            st.write(
                f"The synthetic panel contains **{results.config.n_firms} firms** across "
                f"**{results.config.n_quarters} quarters**, with "
                f"**{results.config.n_treatment_firms} treated firms** receiving a "
                f"simulated **{_fmt_pct(results.config.treatment_effect)} GMV lift** "
                f"starting in quarter **{results.config.treatment_quarter}**."
            )
            st.write(
                f"The DiD estimate implies an observed log-GMV lift of **{did.beta:+.3f}** "
                f"({_fmt_pct(implied_lift)} in level terms), and the classifier reaches "
                f"**{clf.auc:.3f} AUC** in walk-forward validation."
            )
            st.dataframe(
                _panel_summary(features)[["quarter", "group", "avg_gmv", "beat_rate", "avg_momentum"]],
                use_container_width=True,
                hide_index=True,
            )

    with tabs[1]:
        st.subheader("Transaction Panel and Engineered Signals")
        selected_features = st.multiselect(
            "Feature columns",
            options=ENGINEERED_FEATURES,
            default=["cohort_retention_rate", "signal_momentum", "gmv_growth_yoy", "log_gmv"],
        )
        display_cols = [
            "firm_id",
            "quarter",
            "is_treatment",
            "post_event",
            "gmv",
            "n_customers",
            "earnings_beat",
            *selected_features,
        ]
        st.dataframe(features[display_cols], use_container_width=True, hide_index=True)
        st.download_button(
            "Download Feature Panel CSV",
            data=features.to_csv(index=False).encode("utf-8"),
            file_name="engineered_panel.csv",
            mime="text/csv",
        )

    with tabs[2]:
        left, right = st.columns([1.15, 0.85])
        with left:
            st.plotly_chart(_plot_event_study(did), use_container_width=True)
        with right:
            st.subheader("DiD Readout")
            st.code(str(did), language="text")
            st.write(
                "Interpretation: positive post-treatment coefficients indicate the treated "
                "firms' log GMV rose relative to controls after absorbing firm and quarter effects."
            )
            st.dataframe(did.event_study_df, use_container_width=True, hide_index=True)

    with tabs[3]:
        left, right = st.columns([1, 1])
        with left:
            st.plotly_chart(_plot_feature_importance(clf), use_container_width=True)
        with right:
            st.plotly_chart(_plot_prediction_distribution(clf), use_container_width=True)
        st.subheader("Out-of-Fold Predictions")
        st.dataframe(clf.oof_predictions, use_container_width=True, hide_index=True)

    with tabs[4]:
        st.subheader("LLM Input Context")
        st.json(results.white_paper_context, expanded=False)
        st.subheader("Local Preview")
        st.info(
            "This preview uses the same structured numbers that would be sent to the LLM, "
            "but it does not call the Anthropic API."
        )
        st.write(_draft_white_paper_preview(results.white_paper_context))


if __name__ == "__main__":
    main()
