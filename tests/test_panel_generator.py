import pandas as pd

from src.simulation.panel_generator import PanelConfig, generate_panel


def test_generate_panel_respects_config_shape():
    cfg = PanelConfig(n_firms=6, n_quarters=4, n_treatment_firms=2, seed=7)
    panel = generate_panel(cfg)

    assert len(panel) == 24
    assert panel["firm_id"].nunique() == 6
    assert panel["quarter"].nunique() == 4
    assert panel.loc[panel["is_treatment"] == 1, "firm_id"].nunique() == 2


def test_generate_panel_is_reproducible():
    cfg = PanelConfig(n_firms=6, n_quarters=4, n_treatment_firms=2, seed=7)

    pd.testing.assert_frame_equal(generate_panel(cfg), generate_panel(cfg))
