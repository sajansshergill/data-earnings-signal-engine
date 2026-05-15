from src.causal.did_model import run_did
from src.causal.parallel_trends import parallel_trends_p_value, pretrend_summary
from src.features.engineer import engineer_features
from src.simulation.panel_generator import PanelConfig, generate_panel


def test_run_did_returns_positive_effect_on_simulated_panel():
    panel = generate_panel(PanelConfig(n_firms=12, n_treatment_firms=3, seed=3))
    features = engineer_features(panel)

    result = run_did(features)

    assert result.beta > 0
    assert result.n_obs == len(features)
    assert not result.event_study_df.empty


def test_parallel_trends_helpers_return_diagnostics():
    panel = generate_panel(PanelConfig(n_firms=12, n_treatment_firms=3, seed=3))
    features = engineer_features(panel)

    summary = pretrend_summary(features)
    p_value = parallel_trends_p_value(features)

    assert {"quarter", "treated_mean", "control_mean", "gap"}.issubset(summary.columns)
    assert 0.0 <= p_value <= 1.0
