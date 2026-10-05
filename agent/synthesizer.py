"""Synthesizer agent: takes every specialist's independent JSON findings and
merges them into one coherent, weighted report. It also incorporates
historical trend data from memory when available."""
from __future__ import annotations

import json

from .base_agent import ToolAgent
from .config import DEFAULT_MODEL, FALLBACK_MODEL
from .compaction import maybe_compact_specialist_reports

SYNTHESIZER_SYSTEM_PROMPT = """You are the synthesizer agent in a multi-agent SEO audit system.
You receive independent JSON reports from several specialist agents (technical SEO, content,
performance, security, links, and optionally competitive analysis), each with their own 0-100
score and findings. Your job:

1. Assign each category a "weight" (fractions summing to 1.0) reflecting its relative importance
   to overall SEO health -- typically technical SEO and content matter most, but adjust based on
   what the specialists actually found (e.g. weight security higher if a critical vulnerability
   was found).
2. Compute an overall_score (0-100) as the weighted average, then round it.
3. Assign a letter grade: A (90-100), B (80-89), C (70-79), D (60-69), F (<60).
4. Write a 2-4 sentence plain-language summary a non-technical site owner would understand.
5. Pull out 3-6 "quick_wins" -- the highest-impact, lowest-effort fixes across all categories.
6. Write a short, honest "data_limitations" note (e.g. no real Core Web Vitals/Lighthouse data,
   no JS rendering, link-checking was a sample not a full crawl).
7. If previous audit history is provided, weave a brief trend observation into the summary
   (e.g. "up 8 points since the last audit on 2026-06-01").

Respond with ONLY a single JSON object (no prose, no markdown fences) matching exactly:
{
  "url": string,
  "overall_score": number,
  "grade": "A"|"B"|"C"|"D"|"F",
  "summary": string,
  "categories": [
    { "name": string, "score": number, "weight": number, "findings": [ {"severity":"good"|"warning"|"critical","issue":string,"recommendation":string} ] }
  ],
  "quick_wins": [string],
  "data_limitations": string
}

Do not invent findings not present in the specialist reports. Do not simply average blindly --
use judgment, but stay grounded in the evidence provided. Each specialist report has a "category"
field with its exact, fixed name -- use those names VERBATIM as the "name" field for each category
in your output. Do not rename, rephrase, or invent alternative category names.
"""


def run_synthesizer(
    url: str, specialist_reports: dict[str, dict], previous_audit: dict | None,
    model: str = DEFAULT_MODEL, fallback_model: str | None = FALLBACK_MODEL,
    key_index: int = 0, log_fn=None, deadline: float | None = None
) -> dict:
    agent = ToolAgent(name="Synthesizer", system_prompt=SYNTHESIZER_SYSTEM_PROMPT, model=model,
                       fallback_model=fallback_model, max_output_tokens=3500, starting_key_index=key_index,
                       log_fn=log_fn, deadline=deadline)

    specialist_reports = maybe_compact_specialist_reports(specialist_reports, log_fn=log_fn)

    payload = {
        "url": url,
        "specialist_reports": specialist_reports,
    }
    if previous_audit:
        payload["previous_audit"] = {
            "timestamp": previous_audit.get("_timestamp"),
            "overall_score": previous_audit.get("overall_score"),
            "grade": previous_audit.get("grade"),
        }

    return agent.run(json.dumps(payload, indent=2))