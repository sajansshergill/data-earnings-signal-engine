from src.eval.rubric import RUBRIC, rubric_weights, validate_rubric_weights


def test_rubric_weights_sum_to_one():
    validate_rubric_weights()
    assert abs(sum(rubric_weights().values()) - 1.0) < 1e-9


def test_rubric_contains_expected_dimensions():
    names = {dimension.name for dimension in RUBRIC}

    assert {
        "factual_accuracy",
        "citation_discipline",
        "causal_language",
        "hedging",
        "figure_consistency",
        "readability",
    }.issubset(names)
