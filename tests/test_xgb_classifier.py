import numpy as np
import pandas as pd

from src.predictive.shap_explainer import summarize_shap_values, top_shap_features
from src.predictive.walk_forward_cv import iter_walk_forward_splits, validate_no_lookahead
from src.predictive.xgb_classifier import _find_threshold, _prf_at_threshold


def test_walk_forward_splits_are_chronological():
    df = pd.DataFrame({"quarter": [1, 1, 2, 2, 3, 3, 4, 4]})
    splits = list(iter_walk_forward_splits(df, min_train_periods=2))

    validate_no_lookahead(splits)
    assert [split.test_quarter for split in splits] == [3, 4]
    assert splits[0].train_quarters == (1, 2)


def test_threshold_and_metrics_are_valid():
    y_true = pd.Series([0, 1, 1, 0])
    y_prob = pd.Series([0.1, 0.9, 0.8, 0.2])

    threshold = _find_threshold(y_true, y_prob, recall_target=1.0)
    precision, recall, f1 = _prf_at_threshold(y_true, y_prob, threshold)

    assert 0.0 <= threshold <= 1.0
    assert precision == 1.0
    assert recall == 1.0
    assert f1 == 1.0


def test_shap_summary_orders_features_by_importance():
    values = np.array([[1.0, -0.5], [0.5, 2.0]])

    summary = summarize_shap_values(values, ["a", "b"])
    top = top_shap_features(values, ["a", "b"], n=1)

    assert summary.iloc[0]["feature"] == "b"
    assert top == [("b", 1.25)]
