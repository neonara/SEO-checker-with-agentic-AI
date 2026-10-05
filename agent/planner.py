"""Planner: the entry point of the pipeline. Decides which specialists run.

This used to be a model call. It returned the same seven specialists on
practically every audit, so it cost a request (and its share of the free-tier
quota) to restate a rule -- and when it did deviate, it was by adding the
competitive specialist to an audit that had no competitor to compare with.
The rule is now applied directly."""
from __future__ import annotations

from .config import PLANNER_MODEL, FALLBACK_MODEL

CORE_SPECIALISTS = [
    "technical_seo", "content", "performance", "security", "links", "accessibility", "best_practices",
]


def run_planner(url: str, competitor_url: str | None, has_history: bool, model: str = PLANNER_MODEL,
                 fallback_model: str | None = FALLBACK_MODEL, key_index: int = 0, log_fn=None) -> dict:
    """Signature kept from the model-backed planner so callers don't change;
    the model/key arguments are unused."""
    specialists = list(CORE_SPECIALISTS)
    reasoning = "All core checks."
    if competitor_url:
        specialists.append("competitive")
        reasoning = "All core checks, plus the competitive check because a competitor URL was given."
    return {"specialists": specialists, "reasoning": reasoning}
