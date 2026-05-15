"""
Shared rubric metadata for LLM white paper evaluation.

The scoring implementation lives in ``harness.py``. This module keeps dimension
descriptions and weights in a small structured form for dashboards, notebooks,
and docs.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RubricDimension:
    name: str
    weight: float
    description: str
    pass_criterion: str


RUBRIC: tuple[RubricDimension, ...] = (
    RubricDimension(
        name="factual_accuracy",
        weight=0.25,
        description="Numbers in the generated section match model outputs.",
        pass_criterion="DiD beta, CI bounds, AUC, and recall are present within tolerance.",
    ),
    RubricDimension(
        name="citation_discipline",
        weight=0.20,
        description="Methodological claims cite known references.",
        pass_criterion="At least two recognised references are present.",
    ),
    RubricDimension(
        name="causal_language",
        weight=0.20,
        description="Causal verbs are anchored to the DiD estimate or design.",
        pass_criterion="Every causal verb appears near a DiD, beta, or difference-in-differences anchor.",
    ),
    RubricDimension(
        name="hedging",
        weight=0.15,
        description="Uncertainty and assumptions are explicitly qualified.",
        pass_criterion="At least four hedging phrases or uncertainty markers appear.",
    ),
    RubricDimension(
        name="figure_consistency",
        weight=0.10,
        description="Figure callouts match the expected artifacts.",
        pass_criterion="Figure 1 and Figure 2 are both referenced.",
    ),
    RubricDimension(
        name="readability",
        weight=0.10,
        description="The section remains readable for investor audiences.",
        pass_criterion="Approximate Flesch-Kincaid grade is 14 or lower.",
    ),
)


def rubric_weights() -> dict[str, float]:
    """Return rubric weights keyed by dimension name."""
    return {dimension.name: dimension.weight for dimension in RUBRIC}


def validate_rubric_weights() -> None:
    """Raise if rubric weights do not sum to one."""
    total = sum(dimension.weight for dimension in RUBRIC)
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"Rubric weights must sum to 1.0, got {total:.4f}")
