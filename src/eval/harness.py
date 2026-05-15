"""
harness.py
──────────
Rubric-based evaluation of LLM-generated white paper sections.

Six dimensions scored 0.0–1.0; composite is weighted average.
Outputs below 0.75 composite are flagged for human review.
All scores logged to DuckDB for drift tracking over time.

Dimensions
----------
1. Factual Accuracy     (0.25) — numbers match model outputs exactly
2. Citation Discipline  (0.20) — methodological claims cite known references
3. Causal Language      (0.20) — causal verbs anchored to DiD β
4. Hedging              (0.15) — uncertainty explicitly stated
5. Figure Consistency   (0.10) — figure callouts match expected figure names
6. Readability          (0.10) — Flesch-Kincaid grade ≤ 14
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import duckdb
import numpy as np


# ── Weights ────────────────────────────────────────────────────────────────────

DIMENSION_WEIGHTS: dict[str, float] = {
    "factual_accuracy": 0.25,
    "citation_discipline": 0.20,
    "causal_language": 0.20,
    "hedging": 0.15,
    "figure_consistency": 0.10,
    "readability": 0.10,
}

APPROVAL_THRESHOLD: float = 0.75

# Known valid references (partial match)
VALID_REFERENCES = [
    "callaway",
    "sant'anna",
    "sant anna",
    "chen",
    "guestrin",
    "lundberg",
    "abadie",
]

# Causal verbs that require DiD anchoring
CAUSAL_VERBS = [
    r"\bcauses?\b",
    r"\bdrives?\b",
    r"\bleads? to\b",
    r"\bresults? in\b",
    r"\battributable to\b",
    r"\bresponsible for\b",
]

# Hedging phrases — at least N of these should appear
HEDGE_PHRASES = [
    r"consistent with",
    r"suggest[s]?",
    r"evidence",
    r"confidence interval",
    r"subject to",
    r"assumption",
    r"estimate[s]?",
    r"approximately",
    r"we find",
    r"our results",
]

EXPECTED_FIGURES = ["figure 1", "figure 2"]


# ── Result container ───────────────────────────────────────────────────────────

@dataclass
class EvalScore:
    run_id: str
    factual_accuracy: float
    citation_discipline: float
    causal_language: float
    hedging: float
    figure_consistency: float
    readability: float
    composite: float
    approved: bool
    flags: list[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(tz=timezone.utc).isoformat())

    def __str__(self) -> str:
        status = "✓ APPROVED" if self.approved else "✗ FLAGGED — needs human review"
        lines = [
            f"Eval Run: {self.run_id}",
            "─" * 48,
            f"  {'Factual Accuracy':<26} {self.factual_accuracy:.2f}",
            f"  {'Citation Discipline':<26} {self.citation_discipline:.2f}",
            f"  {'Causal Language':<26} {self.causal_language:.2f}",
            f"  {'Hedging':<26} {self.hedging:.2f}",
            f"  {'Figure Consistency':<26} {self.figure_consistency:.2f}",
            f"  {'Readability':<26} {self.readability:.2f}",
            "─" * 48,
            f"  {'Composite':<26} {self.composite:.2f}  {status}",
        ]
        if self.flags:
            lines.append("  Flags:")
            for f in self.flags:
                lines.append(f"    • {f}")
        lines.append("─" * 48)
        return "\n".join(lines)


# ── Main evaluator ─────────────────────────────────────────────────────────────

def evaluate_output(
    wp_section,           # WhitePaperSection
    did_results,          # DiDResults
    classifier_results,   # ClassifierResults
    db_path: Path | str = "outputs/eval_logs/eval_scores.duckdb",
) -> EvalScore:
    """
    Score a white paper section against the 6-dimension rubric.

    Parameters
    ----------
    wp_section : WhitePaperSection
        Output of generate_white_paper_section().
    did_results : DiDResults
        Ground truth for numeric fact-checking.
    classifier_results : ClassifierResults
        Ground truth for AUC / precision / recall fact-checking.
    db_path : Path
        DuckDB file for score logging.

    Returns
    -------
    EvalScore
    """
    text = wp_section.text
    flags: list[str] = []

    # ── 1. Factual Accuracy ────────────────────────────────────────────────────
    fa_score, fa_flags = _score_factual_accuracy(text, did_results, classifier_results)
    flags.extend(fa_flags)

    # ── 2. Citation Discipline ─────────────────────────────────────────────────
    cd_score, cd_flags = _score_citation_discipline(text)
    flags.extend(cd_flags)

    # ── 3. Causal Language Precision ──────────────────────────────────────────
    cl_score, cl_flags = _score_causal_language(text, did_results)
    flags.extend(cl_flags)

    # ── 4. Hedging ────────────────────────────────────────────────────────────
    hd_score, hd_flags = _score_hedging(text)
    flags.extend(hd_flags)

    # ── 5. Figure Consistency ──────────────────────────────────────────────────
    fc_score, fc_flags = _score_figure_consistency(text)
    flags.extend(fc_flags)

    # ── 6. Readability ────────────────────────────────────────────────────────
    rd_score, rd_flags = _score_readability(text)
    flags.extend(rd_flags)

    composite = (
        DIMENSION_WEIGHTS["factual_accuracy"] * fa_score
        + DIMENSION_WEIGHTS["citation_discipline"] * cd_score
        + DIMENSION_WEIGHTS["causal_language"] * cl_score
        + DIMENSION_WEIGHTS["hedging"] * hd_score
        + DIMENSION_WEIGHTS["figure_consistency"] * fc_score
        + DIMENSION_WEIGHTS["readability"] * rd_score
    )

    run_id = wp_section.run_id or str(uuid.uuid4())[:8]
    score = EvalScore(
        run_id=run_id,
        factual_accuracy=fa_score,
        citation_discipline=cd_score,
        causal_language=cl_score,
        hedging=hd_score,
        figure_consistency=fc_score,
        readability=rd_score,
        composite=composite,
        approved=composite >= APPROVAL_THRESHOLD,
        flags=flags,
    )

    _log_to_duckdb(score, db_path)
    return score


# ── Dimension scorers ──────────────────────────────────────────────────────────

def _score_factual_accuracy(
    text: str,
    did_results,
    classifier_results,
) -> tuple[float, list[str]]:
    """
    Check that key numbers from model outputs appear in the text.

    Strategy: for each critical number, check whether a string representation
    matching to 2 decimal places appears anywhere in the text.
    Tolerance: ±0.01 in any numeric substring.
    """
    flags = []
    checks_passed = 0
    checks_total = 0

    critical_numbers = {
        "DiD beta": did_results.beta,
        "CI lower": did_results.ci_lower,
        "CI upper": did_results.ci_upper,
        "AUC": classifier_results.auc,
        "recall": classifier_results.recall_at_threshold,
    }

    # Extract all numeric substrings from text
    text_numbers = [float(m) for m in re.findall(r"-?\d+\.?\d*", text)]

    for label, expected in critical_numbers.items():
        checks_total += 1
        if any(abs(n - expected) < 0.015 for n in text_numbers):
            checks_passed += 1
        else:
            flags.append(f"Factual: '{label}' ({expected:.4f}) not found in text")

    score = checks_passed / checks_total if checks_total > 0 else 0.0
    return score, flags


def _score_citation_discipline(text: str) -> tuple[float, list[str]]:
    """
    Check whether at least 1 valid reference appears in the text.
    Award full marks for ≥ 2 references; partial for 1; zero for none.
    """
    flags = []
    text_lower = text.lower()
    found = [ref for ref in VALID_REFERENCES if ref in text_lower]

    if len(found) >= 2:
        return 1.0, flags
    elif len(found) == 1:
        flags.append("Citation: only 1 reference found; ≥ 2 expected")
        return 0.6, flags
    else:
        flags.append("Citation: no recognised references found in text")
        return 0.0, flags


def _score_causal_language(text: str, did_results) -> tuple[float, list[str]]:
    """
    Check whether causal verbs appear near the DiD estimate.

    Heuristic: for every causal verb found, check whether β or 'DiD'
    or 'difference-in-differences' appears within 200 characters.
    """
    flags = []
    text_lower = text.lower()

    causal_hits = []
    for pattern in CAUSAL_VERBS:
        for match in re.finditer(pattern, text_lower):
            causal_hits.append(match.start())

    if not causal_hits:
        # No causal verbs at all — acceptable (passive voice), full marks
        return 1.0, flags

    did_anchors = list(re.finditer(r"(did|difference.in.differences|\bβ\b|beta)", text_lower))
    anchor_positions = [m.start() for m in did_anchors]

    anchored = 0
    for pos in causal_hits:
        if any(abs(pos - ap) <= 200 for ap in anchor_positions):
            anchored += 1
        else:
            flags.append(
                f"Causal: unanchored causal verb at char {pos} — "
                "not within 200 chars of DiD reference"
            )

    score = anchored / len(causal_hits) if causal_hits else 1.0
    return score, flags


def _score_hedging(text: str) -> tuple[float, list[str]]:
    """
    Count hedging phrases. Expect ≥ 4 for full marks.
    """
    flags = []
    text_lower = text.lower()
    found = sum(1 for p in HEDGE_PHRASES if re.search(p, text_lower))

    if found >= 4:
        return 1.0, flags
    elif found >= 2:
        flags.append(f"Hedging: only {found} hedging phrases (target ≥ 4)")
        return 0.6, flags
    else:
        flags.append(f"Hedging: insufficient hedging ({found} phrases); text may overstate certainty")
        return 0.2, flags


def _score_figure_consistency(text: str) -> tuple[float, list[str]]:
    """
    Check whether expected figure callouts (Figure 1, Figure 2) appear.
    """
    flags = []
    text_lower = text.lower()
    found = [fig for fig in EXPECTED_FIGURES if fig in text_lower]

    score = len(found) / len(EXPECTED_FIGURES)
    if score < 1.0:
        missing = [fig for fig in EXPECTED_FIGURES if fig not in text_lower]
        flags.append(f"Figures: missing callouts: {missing}")

    return score, flags


def _score_readability(text: str) -> tuple[float, list[str]]:
    """
    Approximate Flesch-Kincaid grade level.

    FK Grade = 0.39 * (words/sentences) + 11.8 * (syllables/words) - 15.59
    Target: FK Grade ≤ 14 (readable by sophisticated, non-academic readers).
    """
    flags = []
    sentences = max(len(re.split(r"[.!?]+", text)), 1)
    words_list = re.findall(r"\b\w+\b", text)
    words = max(len(words_list), 1)
    syllables = sum(_count_syllables(w) for w in words_list)

    fk_grade = 0.39 * (words / sentences) + 11.8 * (syllables / words) - 15.59

    if fk_grade <= 14:
        return 1.0, flags
    elif fk_grade <= 16:
        flags.append(f"Readability: FK grade {fk_grade:.1f} (target ≤ 14)")
        return 0.6, flags
    else:
        flags.append(f"Readability: FK grade {fk_grade:.1f} — text may be inaccessible")
        return 0.2, flags


def _count_syllables(word: str) -> int:
    """Approximate syllable count via vowel groups."""
    word = word.lower()
    vowels = re.findall(r"[aeiouy]+", word)
    count = len(vowels)
    if word.endswith("e") and count > 1:
        count -= 1
    return max(count, 1)


# ── DuckDB logging ─────────────────────────────────────────────────────────────

def _log_to_duckdb(score: EvalScore, db_path: Path | str) -> None:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(db_path))
    con.execute("""
        CREATE TABLE IF NOT EXISTS eval_scores (
            run_id VARCHAR,
            timestamp VARCHAR,
            factual_accuracy DOUBLE,
            citation_discipline DOUBLE,
            causal_language DOUBLE,
            hedging DOUBLE,
            figure_consistency DOUBLE,
            readability DOUBLE,
            composite DOUBLE,
            approved BOOLEAN,
            flags VARCHAR
        )
    """)

    con.execute("""
        INSERT INTO eval_scores VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, [
        score.run_id,
        score.timestamp,
        score.factual_accuracy,
        score.citation_discipline,
        score.causal_language,
        score.hedging,
        score.figure_consistency,
        score.readability,
        score.composite,
        score.approved,
        "; ".join(score.flags),
    ])
    con.close()


def load_eval_history(
    db_path: Path | str = "outputs/eval_logs/eval_scores.duckdb",
) -> "pd.DataFrame":
    """Load all eval scores from DuckDB as a DataFrame."""
    import pandas as pd
    con = duckdb.connect(str(db_path))
    df = con.execute("SELECT * FROM eval_scores ORDER BY timestamp DESC").df()
    con.close()
    return df


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.simulation.panel_generator import generate_panel, PanelConfig
    from src.features.engineer import engineer_features
    from src.causal.did_model import run_did
    from src.predictive.xgb_classifier import train_and_evaluate
    from src.llm.white_paper_agent import generate_from_results
    import datetime

    cfg = PanelConfig()
    panel = generate_panel(cfg)
    features = engineer_features(panel)
    did_results = run_did(features)
    classifier_results = train_and_evaluate(features)

    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    print("Generating white paper section...")
    wp = generate_from_results(did_results, classifier_results, run_id=run_id)

    print("Evaluating output...")
    score = evaluate_output(wp, did_results, classifier_results)
    print(score)