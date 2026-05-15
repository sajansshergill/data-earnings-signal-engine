"""
Prompt helpers for the white paper agent.

The main agent keeps the production prompt close to the Anthropic call. This
module exposes reusable prompt text for notebooks, tests, and manual review.
"""

from __future__ import annotations

import json
from textwrap import dedent


PROMPT_VERSION = "v2"

WHITE_PAPER_REQUIREMENTS = (
    "400-600 words; three paragraphs; no bullet points; cite provided references; "
    "anchor causal language to the DiD estimate; use exact numeric inputs."
)


def build_white_paper_user_prompt(context: dict) -> str:
    """Build the user prompt passed to the white paper LLM."""
    return dedent(
        f"""
        Write an investor-facing white paper section from these model results.

        {json.dumps(context, indent=2, sort_keys=True)}

        Requirements: {WHITE_PAPER_REQUIREMENTS}
        """
    ).strip()


def build_eval_repair_prompt(text: str, flags: list[str]) -> str:
    """Ask an LLM to revise a section based on eval harness flags."""
    flag_text = "\n".join(f"- {flag}" for flag in flags) or "- No flags"
    return dedent(
        f"""
        Revise the white paper section so it satisfies the evaluation rubric.

        Current section:
        {text}

        Evaluation flags:
        {flag_text}

        Keep the same facts and do not invent numbers or citations.
        """
    ).strip()
