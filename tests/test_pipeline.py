"""
tests/test_pipeline.py
──────────────────────
Pytest suite covering panel simulation, feature engineering,
causal inference, and eval harness logic.
 
Run: pytest tests/ -v --tb=short
"""
 
from __future__ import annotations
 
import math
import re
from unittest.mock import MagicMock, patch
 
import numpy as np
import pandas as pd
import pytest
 
from src.simulation.panel_generator import (
    PanelConfig,
    describe_panel,
    generate_panel,
)
from src.features.engineer import (
    ENGINEERED_FEATURES,
    engineer_features,
)
from src.causal.did_model import run_did
from src.eval.harness import (
    APPROVAL_THRESHOLD,
    EvalScore,
    _count_syllables,
    _score_causal_language,
    _score_citation_discipline,
    _score_factual_accuracy,
    _score_figure_consistency,
    _score_hedging,
    _score_readability,
)
 
 
# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════
 
@pytest.fixture(scope="session")
def cfg() -> PanelConfig:
    return PanelConfig(n_firms=20, n_quarters=12, n_treatment_firms=4, seed=0)
 
 
@pytest.fixture(scope="session")
def raw_panel(cfg) -> pd.DataFrame:
    return generate_panel(cfg)
 
 
@pytest.fixture(scope="session")
def features(raw_panel) -> pd.DataFrame:
    return engineer_features(raw_panel)
 
 
@pytest.fixture(scope="session")
def did_results(features):
    return run_did(features)
 
 
# ══════════════════════════════════════════════════════════════════════════════
# Panel Generator Tests
# ══════════════════════════════════════════════════════════════════════════════
 
class TestPanelGenerator:
 
    def test_shape(self, raw_panel, cfg):
        assert len(raw_panel) == cfg.n_firms * cfg.n_quarters
 
    def test_treatment_count(self, raw_panel, cfg):
        n_treatment = raw_panel.loc[raw_panel["is_treatment"] == 1, "firm_id"].nunique()
        assert n_treatment == cfg.n_treatment_firms
 
    def test_no_negative_gmv(self, raw_panel):
        assert (raw_panel["gmv"] > 0).all()
 
    def test_no_negative_customers(self, raw_panel):
        assert (raw_panel["n_customers"] >= 0).all()
        assert (raw_panel["n_new_customers"] >= 0).all()
        assert (raw_panel["n_returning_customers"] >= 0).all()
 
    def test_customers_sum_to_cohort(self, raw_panel):
        total = raw_panel["n_new_customers"] + raw_panel["n_returning_customers"]
        assert (total == raw_panel["cohort_size"]).all()
 
    def test_post_event_only_treatment(self, raw_panel):
        """post_event should only be 1 for treatment firms."""
        control_post = raw_panel.loc[raw_panel["is_treatment"] == 0, "post_event"]
        assert (control_post == 0).all()
 
    def test_treatment_effect_is_positive(self, raw_panel, cfg):
        """Treatment firms should have higher avg GMV post-event."""
        avg_pre = raw_panel.loc[
            (raw_panel["is_treatment"] == 1) & (raw_panel["quarter"] < cfg.treatment_quarter),
            "gmv",
        ].mean()
        avg_post = raw_panel.loc[
            (raw_panel["is_treatment"] == 1) & (raw_panel["quarter"] >= cfg.treatment_quarter),
            "gmv",
        ].mean()
        assert avg_post > avg_pre
 
    def test_earnings_beat_is_binary(self, raw_panel):
        assert set(raw_panel["earnings_beat"].unique()).issubset({0, 1})
 
    def test_reproducibility(self, cfg):
        p1 = generate_panel(cfg)
        p2 = generate_panel(cfg)
        pd.testing.assert_frame_equal(p1, p2)
 
    def test_describe_panel_runs(self, raw_panel, capsys):
        describe_panel(raw_panel)
        captured = capsys.readouterr()
        assert "PANEL SUMMARY" in captured.out
 
 
# ══════════════════════════════════════════════════════════════════════════════
# Feature Engineering Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestFeatureEngineering:
    def test_all_features_present(self, features):
        for col in ENGINEERED_FEATURES:
            assert col in features.columns, f"Missing feature: {col}"
            
    def test_log_gmv_positive(self, features):
        assert (features["log_gmv"] > 0).all()
        
    def test_new_returning_ratio_non_negative(self, features):
        assert (features["new_returning_ratio"] >= 0).all()
        
    def test_seasonality_adj_positive(self, features):
        valid = features["seasonality_adj_gmv"].dropna()
        assert (valid > 0).all()
        
    def test_no_extra_rows(self, raw_panel, features):
        """Feature engineering must not change raw count."""
        assert len(features) == len(raw_panel)
        
    def test_sorted_by_firm_quarter(self, features):
        for firm_id, grp in features.groupby("firm_id"):
            quarters = grp["quarter"].tolist()
            assert quarters == sorted(quarters), f"Unsorted quarters for {firm_id}"
            
    def test_cohort_retention_null_in_first_quarter(self, features):
        """Q1 has no lag -> retention rate must be NaN."""
        q1 = features[features["quarter"] == 1]
        assert q1["cohort_retention_rate"].isna().all()
        
    def test_gmv_growth_yoy_null_before_q5(self, features):
        """YoY requires lag-4 -> Q1-Q4 must be NaN."""
        early = features[features["quarter"] <= 4]
        assert early["gmv_growth_yoy"].isna().all()
        
    def test_signal_momentum_bounded(self, features):
        """EMW of growth rates shouldn't explode beyong ±5x."""
        valid = features["signal_momentum"].dropna()
        assert (valid.abs() < 5).all()
        
    def test_vectorised_no_object_loops(self, raw_panel):
        """Smoke test: engineer_features runs fast (<5s) on 20-firm panel."""
        import time
        start = time.time()
        engineer_features(raw_panel)
        elapsed = time.time() - start
        assert elapsed < 5.0, f"Feature Engineering too slow: {elapsed:.1f}s"
        
 
# ══════════════════════════════════════════════════════════════════════════════
# Causal Inference Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestDiDModel:
    def test_beta_is_positive(self, did_results):
        """With a ±15% injected treatment effect, ATT should be positive."""
        assert did_results.beta > 0
        
    def test_ci_width_reasonable(self, did_results):
        width = did_results.ci_upper - did_results.ci_lower
        assert 0 < width < 2.0, f"CI width implausible: {width:.4f}"
        
    def test_parallel_trends_p_is_float(self, did_results):
        assert isinstance(did_results.parallel_trends_p, float)
        assert 0.0 <= did_results.parallel_trends_p <= 1.0
        
    def test_r_squared_reasonable(self, did_results, cfg):
        assert 0.0 <= did_results.r_squared <= 1.0
        
    def test_n_obs_correct(self, did_results, cfg):
        assert did_results.n_obs == cfg.n_firms * cfg.n_quarters
        
    def test_event_study_df_not_empty(self, did_results):
        assert not did_results.event_study_df.empty
        
    def test_event_study_has_all_quarters(self, did_results, cfg):
        n_quarters_in_es = did_results.event_study_df["quarter"].nunique()
        # All quarters should appear (inclusing reference quarter)
        assert n_quarters_in_es == cfg.n_quarters
        
    def test_significance_at_injected_effect(self, did_results):
        """
        With a 15% treatment effect on 20 firms x 12 quarters,
        the estimate should be statistically significant.
        """
        assert did_results.p_value < 0.05
        
    def test_str_output(self, did_results):
        s = str(did_results)
        assert "ATT" in s
        assert "CI" in s
        
# ══════════════════════════════════════════════════════════════════════════════
# Eval Harness Tests (unit-level)
# ══════════════════════════════════════════════════════════════════════════════
class TestEvalHarness:
    SAMPLE_TEXT = (
        "Using a Difference-in-Differences design, the evidence is consistent with "
        "an attributable to treatment revenue lift. The DiD beta estimate is 0.15 "
        "with a confidence interval from 0.09 to 0.20, and Figure 1 shows the event "
        "study. Chen and Guestrin document the XGBoost method used for the predictive "
        "model, where AUC is 0.80 and recall is 0.70. Figure 2 reports SHAP evidence. "
        "Our results suggest this estimate is subject to the parallel trends assumption."
    )

    class _Did:
        beta = 0.15
        ci_lower = 0.09
        ci_upper = 0.20

    class _Classifier:
        auc = 0.80
        recall_at_threshold = 0.70

    def test_factual_accuracy_finds_expected_numbers(self):
        score, flags = _score_factual_accuracy(
            self.SAMPLE_TEXT,
            self._Did(),
            self._Classifier(),
        )
        assert score == 1.0
        assert flags == []

    def test_citation_discipline_scores_known_references(self):
        score, flags = _score_citation_discipline(self.SAMPLE_TEXT)
        assert score == 1.0
        assert flags == []

    def test_causal_language_requires_did_anchor(self):
        score, flags = _score_causal_language(
            "Marketing drives revenue without model support.",
            self._Did(),
        )
        assert score == 0.0
        assert flags

    def test_hedging_rewards_uncertainty_language(self):
        score, flags = _score_hedging(self.SAMPLE_TEXT)
        assert score == 1.0
        assert flags == []

    def test_figure_consistency_requires_expected_figures(self):
        score, flags = _score_figure_consistency("Figure 1 is present.")
        assert score == 0.5
        assert flags

    def test_readability_returns_unit_interval(self):
        score, flags = _score_readability("The model is clear. The result is useful.")
        assert 0.0 <= score <= 1.0
        assert isinstance(flags, list)

    def test_count_syllables_minimum_one(self):
        assert _count_syllables("rhythm") >= 1

    def test_eval_score_approval_flag(self):
        score = EvalScore(
            run_id="unit",
            factual_accuracy=1.0,
            citation_discipline=1.0,
            causal_language=1.0,
            hedging=1.0,
            figure_consistency=1.0,
            readability=1.0,
            composite=1.0,
            approved=True,
        )
        assert score.approved
        assert "APPROVED" in str(score)