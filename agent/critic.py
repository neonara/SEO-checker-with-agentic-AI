"""
Critic agent: implements a reflection loop. Rather than trusting the
synthesizer's first draft, a separate critical pass checks it for
hallucinated claims, miscalibrated scores, and internal inconsistency, and
either approves it or sends it back with concrete revision instructions.
This catches errors a single-pass pipeline would silently ship.
"""
from __future__ import annotations

import json

from .base_agent import ToolAgent
from .synthesizer import run_synthesizer
from .config import CRITIC_MODEL, MAX_REFLECTION_ROUNDS, DEFAULT_MODEL, FALLBACK_MODEL
from .compaction import maybe_compact_specialist_reports, maybe_compact_report

CRITIC_SYSTEM_PROMPT = """You are the critic agent in a multi-agent SEO audit system. You review
a draft report produced by a synthesizer agent against the raw specialist findings it was built
from. Check for:
- Findings or numbers in the draft that are NOT supported by the underlying specialist reports
  (hallucination)
- Scores that seem miscalibrated given the findings (e.g. a "critical" finding present but the
  category score is still 95)
- Internal inconsistency (e.g. weights don't sum to ~1.0, overall_score doesn't roughly match
  the weighted average of category scores)
- Missing categories that a specialist actually reported on
- Vague, non-actionable recommendations

Respond with ONLY a JSON object (no prose, no markdown fences):
{
  "approved": boolean,
  "issues": [string],   // empty list if approved
  "instructions_for_revision": string  // empty string if approved; otherwise concrete, specific fixes
}
Be a genuinely skeptical reviewer -- approving a flawed report defeats the point of this step.
List AT MOST the 6 most important issues, ranked by severity -- do not enumerate every minor or
repetitive inconsistency. Keep each issue to one concise sentence. Your entire response must be
valid, complete JSON; being concise matters more than being exhaustive.
"""


def critique(draft: dict, specialist_reports: dict, model: str = CRITIC_MODEL,
             fallback_model: str | None = FALLBACK_MODEL, key_index: int = 0, log_fn=None,
             deadline: float | None = None) -> dict:
    agent = ToolAgent(name="Critic", system_prompt=CRITIC_SYSTEM_PROMPT, model=model,
                       fallback_model=fallback_model, max_output_tokens=3500, starting_key_index=key_index,
                       log_fn=log_fn, deadline=deadline)
    draft = maybe_compact_report(draft, log_fn=log_fn)
    specialist_reports = maybe_compact_specialist_reports(specialist_reports, log_fn=log_fn)
    payload = {"draft_report": draft, "specialist_reports": specialist_reports}
    return agent.run(json.dumps(payload, indent=2))


def reflect_and_revise(
    url: str,
    specialist_reports: dict[str, dict],
    previous_audit: dict | None,
    synthesizer_model: str = DEFAULT_MODEL,
    critic_model: str = CRITIC_MODEL,
    fallback_model: str | None = FALLBACK_MODEL,
    starting_key_index: int = 0,
    log_fn=None,
    deadline: float | None = None,
) -> tuple[dict, list[dict]]:
    """Run synthesizer -> critic -> (revise if needed) up to MAX_REFLECTION_ROUNDS times.
    Returns (final_report, reflection_log).

    Synthesizer and critic each run multiple times per audit (once per
    reflection round), and previously always started on the same API key
    (index 0) every time -- meaning that key's 70B quota took the full brunt
    of every sequential call in this stage while other configured keys sat
    unused here. Each successive call now moves to the next key instead.

    On the LAST round, if the critic still rejects the draft, we stop there
    rather than running one more, unreviewed synthesis pass -- that extra
    call was previously happening silently, wasting an API call and
    occasionally producing a broken/degenerate draft (e.g. an empty category
    with a null score) that nothing ever re-checked before it reached the user."""
    reflection_log = []
    key_cycle = starting_key_index

    draft = run_synthesizer(url, specialist_reports, previous_audit, model=synthesizer_model,
                             fallback_model=fallback_model, key_index=key_cycle, log_fn=log_fn,
                             deadline=deadline)
    key_cycle += 1

    for round_num in range(1, MAX_REFLECTION_ROUNDS + 1):
        review = critique(draft, specialist_reports, model=critic_model,
                           fallback_model=fallback_model, key_index=key_cycle, log_fn=log_fn,
                           deadline=deadline)
        key_cycle += 1
        reflection_log.append({"round": round_num, "review": review})

        if review.get("approved"):
            break

        if log_fn:
            log_fn(f"[Critic] round {round_num}: revision requested -- {review.get('issues')}")

        if round_num == MAX_REFLECTION_ROUNDS:
            # No rounds left to re-review a further revision -- stop here so the
            # draft we return is exactly the one reflection_log's last entry
            # describes, rather than a further, unreviewed synthesizer pass.
            if log_fn:
                log_fn("[Critic] max reflection rounds reached; proceeding with the "
                       "last-reviewed draft.")
            break

        # Feed critic's instructions back into a re-run of the synthesizer
        revised_reports = dict(specialist_reports)
        revised_reports["_critic_feedback"] = {
            "previous_draft": draft,
            "issues": review.get("issues"),
            "instructions": review.get("instructions_for_revision"),
        }
        draft = run_synthesizer(url, revised_reports, previous_audit, model=synthesizer_model,
                                 fallback_model=fallback_model, key_index=key_cycle, log_fn=log_fn,
                                 deadline=deadline)
        key_cycle += 1

    return draft, reflection_log