"""
white_paper_agent.py
────────────────────
LLM agent that ingests model outputs and generates an investor-facing
white paper section with equation citations and confidence framing.

The agent receives structured context (DiD estimates, SHAP values, model
metrics, parallel trends test) and produces a 400–600 word section
formatted for institutional investor and Fortune 500 readers.

Design notes
------------
- System prompt enforces citation discipline: every causal claim must
  reference the DiD estimate; every predictive claim must reference AUC
  or SHAP magnitude.
- Hedging is required by the prompt: the agent must qualify confidence
  intervals and model limitations.
- Output is plain text (markdown-lite) — downstream eval harness scores it
  before it surfaces to any user.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import anthropic
from dotenv import load_dotenv

load_dotenv()


# ── Result container ───────────────────────────────────────────────────────────

@dataclass
class WhitePaperSection:
    text: str
    input_context: dict
    model_used: str
    prompt_version: str = "v2"
    run_id: Optional[str] = None

    def save(self, output_dir: Path | str = "outputs/white_papers") -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        fname = f"white_paper_{self.run_id or 'latest'}.txt"
        out_path = output_dir / fname
        out_path.write_text(self.text, encoding="utf-8")
        return out_path

    def __str__(self) -> str:
        return self.text


# ── Context builder ────────────────────────────────────────────────────────────

def build_context(
    did_beta: float,
    did_se: float,
    did_ci_lower: float,
    did_ci_upper: float,
    did_p_value: float,
    parallel_trends_p: float,
    auc: float,
    precision: float,
    recall: float,
    threshold: float,
    top_features: list[tuple[str, float]],   # [(feature_name, mean_shap_abs), ...]
    n_firms: int,
    n_quarters: int,
    treatment_effect_pct: Optional[float] = None,
) -> dict:
    """
    Build the structured context dict passed to the LLM agent.

    All numeric inputs come directly from DiDResults and ClassifierResults —
    the agent never infers numbers; it narrates what the models produced.
    """
    return {
        "causal_inference": {
            "method": "Difference-in-Differences with Two-Way Fixed Effects",
            "beta": round(did_beta, 4),
            "se": round(did_se, 4),
            "ci_lower": round(did_ci_lower, 4),
            "ci_upper": round(did_ci_upper, 4),
            "p_value": round(did_p_value, 4),
            "parallel_trends_p": round(parallel_trends_p, 3),
            "parallel_trends_status": "holds" if parallel_trends_p > 0.05 else "violated",
            "implied_gmv_lift_pct": round(treatment_effect_pct * 100, 1) if treatment_effect_pct else None,
            "reference": "Callaway & Sant'Anna (2021). Journal of Econometrics, 225(2), 200–230.",
        },
        "predictive_model": {
            "method": "XGBoost with walk-forward cross-validation",
            "auc": round(auc, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "operating_threshold": round(threshold, 4),
            "top_features": [
                {"feature": f, "mean_shap_abs": round(s, 4)}
                for f, s in top_features[:3]
            ],
            "reference": "Chen & Guestrin (2016). XGBoost. KDD '16.",
        },
        "data": {
            "n_firms": n_firms,
            "n_quarters": n_quarters,
            "panel_type": "transaction-level alternative data (e-commerce GMV)",
        },
    }


# ── System prompt ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a quantitative research analyst at an alternative data firm.
Your task is to write one section of an investor-facing white paper based on the
structured model results you receive.

Audience: institutional investors and Fortune 500 strategy teams. They are
statistically sophisticated and expect precise, hedged language.

REQUIREMENTS — follow every rule:

1. LENGTH: 400–600 words. No more. No less.

2. STRUCTURE: Three paragraphs.
   - Paragraph 1 (Methodology): Describe the causal identification strategy.
     Cite the reference provided. Include the model specification in plain English.
   - Paragraph 2 (Results): Report the ATT estimate with its confidence interval.
     Use the exact numbers from the context. Reference Figure 1 (event study) and
     Figure 2 (SHAP waterfall). Name the top predictive feature by name.
   - Paragraph 3 (Limitations & Confidence): Qualify the findings honestly.
     State what the parallel trends test result means for credibility.
     Acknowledge model assumptions.

3. CAUSAL LANGUAGE: Only use causal language ("causes", "drives", "attributable to")
   when anchored to the DiD estimate. Never write "X causes Y" without the DiD β.

4. HEDGING: Uncertainty must be explicit. State confidence intervals. Use phrases
   like "the evidence is consistent with", "our estimates suggest", "subject to
   the parallel trends assumption".

5. CITATIONS: Every methodological claim must cite the reference provided in context.
   Format: Author (Year). No invented citations.

6. NUMBERS: Use only the exact numbers from the structured context provided.
   Do not round differently. Do not infer additional statistics.

7. NO BULLET POINTS. Prose only. No headers within the section.

Output the white paper section text only — no preamble, no meta-commentary."""


# ── User prompt builder ────────────────────────────────────────────────────────

def _build_user_prompt(context: dict) -> str:
    return f"""Write the white paper section based on the following model results:

{json.dumps(context, indent=2)}

Reminder: 400–600 words, three paragraphs, no bullet points, causal language
only anchored to DiD β, cite all methodological claims, use exact numbers above."""


# ── Agent ──────────────────────────────────────────────────────────────────────

def generate_white_paper_section(
    context: dict,
    run_id: Optional[str] = None,
    api_key: Optional[str] = None,
) -> WhitePaperSection:
    """
    Call the Claude API to generate the investor white paper section.

    Parameters
    ----------
    context : dict
        Output of build_context().
    run_id : str, optional
        Unique identifier for this run (used in output filename).
    api_key : str, optional
        Anthropic API key. Falls back to ANTHROPIC_API_KEY env var.

    Returns
    -------
    WhitePaperSection
    """
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY not set. Add it to .env or pass api_key= directly."
        )

    client = anthropic.Anthropic(api_key=key)

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": _build_user_prompt(context)}
        ],
    )

    text = message.content[0].text.strip()

    return WhitePaperSection(
        text=text,
        input_context=context,
        model_used=message.model,
        prompt_version="v2",
        run_id=run_id,
    )


# ── Convenience wrapper ────────────────────────────────────────────────────────

def generate_from_results(
    did_results,          # DiDResults
    classifier_results,   # ClassifierResults
    run_id: Optional[str] = None,
    api_key: Optional[str] = None,
) -> WhitePaperSection:
    """
    Convenience wrapper: build context from result objects and call the agent.

    Parameters
    ----------
    did_results : DiDResults
        Output of run_did().
    classifier_results : ClassifierResults
        Output of train_and_evaluate().
    """
    top_features = list(zip(
        classifier_results.feature_importance["feature"].tolist(),
        classifier_results.feature_importance["mean_shap_abs"].tolist(),
    ))

    # Implied GMV lift: exp(β) - 1 since outcome is log_gmv
    import math
    implied_pct = math.exp(did_results.beta) - 1

    context = build_context(
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
        n_quarters=did_results.n_obs // did_results.n_firms,
        treatment_effect_pct=implied_pct,
    )

    return generate_white_paper_section(context, run_id=run_id, api_key=api_key)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.simulation.panel_generator import generate_panel, PanelConfig
    from src.features.engineer import engineer_features
    from src.causal.did_model import run_did
    from src.predictive.xgb_classifier import train_and_evaluate
    import datetime

    cfg = PanelConfig()
    panel = generate_panel(cfg)
    features = engineer_features(panel)
    did_results = run_did(features)
    classifier_results = train_and_evaluate(features)

    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    print("Generating white paper section...")
    wp = generate_from_results(did_results, classifier_results, run_id=run_id)

    path = wp.save()
    print(f"\nSaved → {path}")
    print("\n" + "="*60)
    print(wp.text)