"""
Deterministic ground-truth reconciliation.

LLMs -- especially smaller fallback ones -- sometimes misreport facts a tool
already computed for them with total certainty, or restate numbers in prose
that contradict deterministic data computed elsewhere in the same report.
Prompt instructions reduce this but can't eliminate it, so this module fixes
specific, repeatedly-observed failure classes in code instead of trusting
the model to self-correct.
"""
from __future__ import annotations

import math
import re

_SSL_KEYWORDS = ("certificate", "ssl", "https/ssl", "tls")
_SSL_VALIDITY_KEYWORDS = ("expir", "valid", "not set")


def _is_ssl_finding(finding: dict) -> bool:
    text = (finding.get("issue", "") + " " + finding.get("recommendation", "")).lower()
    return any(k in text for k in _SSL_KEYWORDS) and any(k in text for k in _SSL_VALIDITY_KEYWORDS)


def _latest_ssl_tool_result(tool_call_log: list[dict]) -> dict | None:
    for entry in reversed(tool_call_log):
        if entry.get("name") == "check_ssl_certificate":
            result = entry.get("result") or {}
            if result.get("ok"):
                return result
    return None


def reconcile_ssl_findings(specialist_result: dict, tool_call_log: list[dict], log_fn=None) -> dict:
    """Replace any SSL-expiry/validity findings in specialist_result with a
    single, deterministically-correct one built from the actual tool output.
    No-op if the specialist never called check_ssl_certificate."""
    ssl_result = _latest_ssl_tool_result(tool_call_log)
    if ssl_result is None:
        return specialist_result

    findings = specialist_result.get("findings", [])
    kept = [f for f in findings if not _is_ssl_finding(f)]
    removed_count = len(findings) - len(kept)

    if ssl_result.get("has_valid_ssl") is False:
        canonical = {
            "severity": "critical",
            "issue": f"Could not verify an SSL certificate: {ssl_result.get('error', 'connection failed')}.",
            "recommendation": "Investigate why the HTTPS/SSL handshake is failing.",
        }
    elif ssl_result.get("is_expired") is True:
        canonical = {
            "severity": "critical",
            "issue": f"SSL certificate has expired. {ssl_result.get('ssl_status_summary', '')}".strip(),
            "recommendation": "Renew the SSL certificate immediately.",
        }
    else:
        canonical = {
            "severity": "good",
            "issue": f"SSL certificate is valid. {ssl_result.get('ssl_status_summary', '')}".strip(),
            "recommendation": "No action needed; renew before it approaches expiry.",
        }

    kept.append(canonical)
    specialist_result["findings"] = kept

    if removed_count and log_fn:
        log_fn(
            f"  -> Corrected {removed_count} inaccurate SSL finding(s) in "
            f"'{specialist_result.get('category', '?')}' using verified tool data "
            f"(is_expired={ssl_result.get('is_expired')})"
        )

    return specialist_result


# --- Competitive specialist / on-page & technical overlap ---------------

_ONPAGE_OVERLAP_TOPIC_KEYWORDS = ("title tag", "title length", "title is", "meta description")
_ONPAGE_OVERLAP_CONTEXT_KEYWORDS = ("character", "length", "within", "recommend", "missing", "lacks", "omission")

# Canonical tags are unambiguously Technical SEO's domain (crawlability/
# duplicate-content signal) -- unlike title/meta findings, real-world
# phrasing here rarely uses the context words above (e.g. "canonical link
# element present and correctly points to the primary URL" has none of
# them), so this needs no context requirement: any mention at all is overlap.
_STANDALONE_OVERLAP_TOPICS = ("canonical",)


def _is_onpage_overlap_finding(finding: dict) -> bool:
    text = (finding.get("issue", "") + " " + finding.get("recommendation", "")).lower()
    if any(k in text for k in _STANDALONE_OVERLAP_TOPICS):
        return True
    mentions_topic = any(k in text for k in _ONPAGE_OVERLAP_TOPIC_KEYWORDS)
    mentions_context = any(k in text for k in _ONPAGE_OVERLAP_CONTEXT_KEYWORDS)
    return mentions_topic and mentions_context


def strip_competitive_onpage_overlap(specialist_result: dict, log_fn=None) -> dict:
    """Remove any competitive-specialist finding that re-judges title tag,
    meta description, or canonical tag presence -- those are the Content and
    Technical SEO specialists' jobs, and the competitive specialist doesn't
    share their ground-truth data."""
    findings = specialist_result.get("findings", [])
    kept = [f for f in findings if not _is_onpage_overlap_finding(f)]
    removed_count = len(findings) - len(kept)

    specialist_result["findings"] = kept

    if removed_count and log_fn:
        log_fn(
            f"  -> Removed {removed_count} duplicate/conflicting on-page finding(s) from "
            f"'{specialist_result.get('category', '?')}' -- basic on-page/technical facts "
            f"(title, meta description, or canonical tag) are the Content/Technical SEO "
            f"specialist's domain, not competitive's."
        )

    return specialist_result


# --- Summary text vs. real trend data ------------------------------------
#
# The synthesizer sometimes writes a specific point-delta claim in its prose
# summary ("up 3 points", "improved by 8%") that contradicts the actual,
# deterministically-computed trend sitting right next to it in the same
# report -- sometimes getting the *direction* backwards, and sometimes
# agreeing on direction but stating the wrong *magnitude* (e.g. "increased
# by 1.5 points" when the real delta is +5). Rather than trying to
# surgically rewrite the model's prose (fragile), append a clear, correct
# note so the real number is never far from the wrong one.

_UP_WORDS = ("improved", "increase", "up ", "risen", "grew", "gained")
_DOWN_WORDS = ("decreased", "decline", "down ", "dropped", "fell", "worsened", "regressed")
_POINT_NUMBER_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:point|points|%|percent)", re.IGNORECASE)


def fix_summary_trend_mismatch(report: dict, log_fn=None) -> None:
    """If the summary text's claimed trend direction OR magnitude contradicts
    the actual computed score_delta, append a correcting note rather than
    trusting the model's restated arithmetic."""
    trend = report.get("trend")
    summary = report.get("summary", "")
    if not trend or not summary:
        return

    delta = trend.get("score_delta")
    if delta is None:
        return

    summary_lower = summary.lower()
    claims_up = any(w in summary_lower for w in _UP_WORDS)
    claims_down = any(w in summary_lower for w in _DOWN_WORDS)

    direction_mismatch = (claims_up and delta < 0) or (claims_down and delta > 0)

    magnitude_mismatch = False
    if (claims_up or claims_down) and not direction_mismatch:
        match = _POINT_NUMBER_RE.search(summary_lower)
        if match:
            claimed_magnitude = float(match.group(1))
            real_magnitude = abs(delta)
            # Tolerate small rounding differences; flag anything meaningfully off.
            tolerance = max(1.0, real_magnitude * 0.25)
            if abs(claimed_magnitude - real_magnitude) > tolerance:
                magnitude_mismatch = True

    if direction_mismatch or magnitude_mismatch:
        direction = "up" if delta > 0 else ("down" if delta < 0 else "flat")
        reason = "direction" if direction_mismatch else "magnitude"
        if log_fn:
            log_fn(f"  -> Summary text's trend claim has a {reason} mismatch vs actual "
                   f"score_delta ({delta:+g}); appending a correction.")
        report["summary"] = (
            summary.rstrip()
            + f" (Note: the summary above may misstate the change -- the actual verified "
              f"change since the last audit is {direction} {abs(delta):g} points.)"
        )


# --- Fabricated "previous audit" comparisons -----------------------------

_PREVIOUS_AUDIT_PHRASES = (
    "previous audit", "last audit", "prior audit", "earlier audit",
    "since the last", "compared to the previous", "compared to the last",
)


def fix_fabricated_trend_claim(report: dict, log_fn=None) -> None:
    """If no real trend data exists (no prior audit for this domain), but the
    summary still claims a comparison to a previous audit, append a
    correction rather than leaving a fabricated data point unchallenged."""
    if report.get("trend"):
        return  # a real previous audit exists -- fix_summary_trend_mismatch handles that case

    summary = report.get("summary", "")
    if not summary:
        return

    summary_lower = summary.lower()
    if any(phrase in summary_lower for phrase in _PREVIOUS_AUDIT_PHRASES):
        if log_fn:
            log_fn("  -> Summary claims a comparison to a previous audit, but no prior audit "
                   "exists for this domain; appending a correction.")
        report["summary"] = (
            summary.rstrip()
            + " (Note: this is the first recorded audit for this domain -- there is no actual "
              "previous audit to compare against, so any such comparison above is fabricated "
              "and should be disregarded.)"
        )


# --- Likely bot-blocked fetches --------------------------------------------
#
# When a site's anti-bot/WAF protection returns 401/403/429/503 instead of
# the real page, that response gets cached and every specialist that reuses
# it ends up scoring a block/challenge page as if it were the real site --
# "no headings", "no images", "missing meta description" etc. are typical
# symptoms of this, not real content quality problems. Prompting specialists
# to notice fetch_page's likely_blocked flag isn't reliable enough on its
# own, so this deterministically overrides the specialist's output instead
# of trusting the model to correctly discount a blocked fetch.

def _latest_fetch_page_result(tool_call_log: list[dict]) -> dict | None:
    for entry in reversed(tool_call_log):
        if entry.get("name") == "fetch_page":
            result = entry.get("result") or {}
            if result.get("ok"):
                return result
    return None


def reconcile_likely_blocked(specialist_result: dict, tool_call_log: list[dict], log_fn=None) -> dict:
    """If this specialist's fetch_page result was flagged likely_blocked,
    override its score/findings with a single, honest 'could not verify'
    finding rather than trusting whatever it inferred from a block page."""
    fetch_result = _latest_fetch_page_result(tool_call_log)
    if not fetch_result or not fetch_result.get("likely_blocked"):
        return specialist_result

    status = fetch_result.get("status_code")
    category = specialist_result.get("category", "?")

    if log_fn:
        log_fn(
            f"  -> '{category}' fetch returned status {status} (likely bot-blocking) -- "
            f"overriding its findings with a single block-warning finding rather than "
            f"trusting scores/findings derived from what is probably a block page, not "
            f"the real site."
        )

    specialist_result["_likely_blocked"] = True
    specialist_result["_blocked_status"] = status
    specialist_result["score"] = None
    specialist_result["findings"] = [{
        "severity": "critical",
        "issue": (
            f"Could not reliably audit real page content for this category -- the request "
            f"returned HTTP {status}, which commonly indicates bot/WAF blocking rather than "
            f"a genuine site issue."
        ),
        "recommendation": (
            "Manually verify the page loads correctly in a real browser. This category's "
            "automated findings are not reliable until access is confirmed; it has been "
            "excluded from the overall score rather than scored against a likely block page."
        ),
    }]
    specialist_result["raw_evidence_notes"] = (
        f"fetch_page returned status {status} and was flagged likely_blocked; "
        f"findings suppressed and replaced with this notice."
    )
    return specialist_result


# --- Real Core Web Vitals data actually being used -----------------------
#
# check_core_web_vitals runs a genuine Lighthouse audit, but a model can
# still write vague, numberless findings ("page load time is higher than
# expected") instead of citing what it actually measured -- especially the
# smaller fallback model. If the specialist's findings don't reference any
# of the real numbers, inject a guaranteed-accurate finding built directly
# from the tool result, so real data always makes it into the report
# regardless of whether the model chose to use it well.

def _latest_cwv_result(tool_call_log: list[dict]) -> dict | None:
    for entry in reversed(tool_call_log):
        if entry.get("name") == "check_core_web_vitals":
            result = entry.get("result") or {}
            if result.get("ok"):
                return result
    return None


def _pin_measured_score(specialist_result: dict, measured, log_fn=None) -> None:
    """Lighthouse already scored this category 0-100. Use that number as the
    category score instead of the model's own estimate, which moves between
    runs on an unchanged site. Recorded under `_measured_score` too, so the
    orchestrator can re-apply it over whatever the synthesizer writes.
    Skipped for a blocked result, whose score is deliberately empty."""
    if not isinstance(measured, (int, float)) or specialist_result.get("_likely_blocked"):
        return
    claimed = specialist_result.get("score")
    if log_fn and isinstance(claimed, (int, float)) and abs(claimed - measured) >= 1:
        log_fn(f"  -> '{specialist_result.get('category', '?')}' score set to the measured "
               f"Lighthouse score {measured} (model said {claimed}).")
    specialist_result["score"] = measured
    specialist_result["_measured_score"] = measured


def reconcile_core_web_vitals(specialist_result: dict, tool_call_log: list[dict], log_fn=None) -> dict:
    """No-op if check_core_web_vitals wasn't called or failed. If it
    succeeded but the specialist's findings don't cite the real numbers,
    append a canonical finding built from the actual measured data."""
    cwv = _latest_cwv_result(tool_call_log)
    if cwv is None:
        return specialist_result

    lab = cwv.get("lab_data", {})
    # Set whenever the tool succeeded, not only when a finding gets injected
    # below: the orchestrator reads this to correct a stale "no real Core Web
    # Vitals data" note, and that note is just as wrong when the model did
    # cite the real numbers.
    specialist_result["_real_cwv_available"] = True
    _pin_measured_score(specialist_result, lab.get("performance_score_0_100"), log_fn)

    findings = specialist_result.get("findings", [])
    findings_text = " ".join(
        f"{f.get('issue', '')} {f.get('recommendation', '')}" for f in findings
    ).lower()

    checkable_values = [lab.get("lcp_ms"), lab.get("performance_score_0_100")]
    mentions_real_data = any(
        v is not None and (str(round(v)) in findings_text or str(v) in findings_text)
        for v in checkable_values
    )
    if mentions_real_data:
        return specialist_result  # model already cited real data, nothing to force

    score = lab.get("performance_score_0_100")
    lcp, cls = lab.get("lcp_ms"), lab.get("cls")
    thresholds = cwv.get("good_thresholds_2026", {})
    severity = "critical" if (score is not None and score < 50) else (
        "warning" if (score is not None and score < 90) else "good"
    )
    canonical = {
        "severity": severity,
        "issue": (
            f"Real Lighthouse performance score: {score}/100. LCP={lcp}ms "
            f"(good is under {thresholds.get('lcp_ms')}ms), CLS={cls} "
            f"(good is under {thresholds.get('cls')})."
        ),
        "recommendation": (
            "Prioritize LCP and CLS fixes if either exceeds the good thresholds above "
            "(e.g. optimize largest above-the-fold image/text, reserve space for dynamic content)."
        ),
    }
    findings.append(canonical)
    specialist_result["findings"] = findings

    if log_fn:
        log_fn(
            f"  -> Injected real Core Web Vitals data into "
            f"'{specialist_result.get('category', '?')}' since the model's findings "
            f"didn't cite the actual measured numbers."
        )

    return specialist_result


# --- Real accessibility audit data actually being used --------------------
#
# Same failure mode as Core Web Vitals: check_accessibility_and_best_practices
# runs a genuine Lighthouse accessibility audit, but a model can still write
# vague findings ("could improve accessibility") instead of citing the real
# score and the specific failing checks -- inject a canonical finding built
# directly from the tool result if the model's own findings don't cite it.
#
# NOTE: this must check *topic* overlap, not just whether the literal score
# number appears in the text, AND that overlap check needs to tolerate
# normal English variation (plurals, word order), not just exact substrings.
# Observed in the wild, twice, in two different ways:
#   1. The model wrote four findings clearly sourced from this exact tool
#      call but never spelled out the score number -- a literal-digit-only
#      check treated that as "no real data cited" and appended a duplicate.
#   2. The model wrote "required child roles" for an audit whose id/title
#      say "required children", and "List items not contained within..."
#      for an audit id "listitem" (one fused word) -- exact-substring
#      matching missed both as "not covered" and injected duplicates of
#      findings that were, in substance, already there.
# The matcher below stems words (including the "children"/"child" irregular
# case) and requires only a majority-of-keywords overlap, using both the
# audit's id AND its human-readable title as the keyword source, so
# ordinary rewording no longer produces a false "not covered".

# --- Real Lighthouse audit data (accessibility / best practices) actually
#     being used -----------------------------------------------------------
#
# Same failure mode as Core Web Vitals: check_accessibility_and_best_practices
# and check_best_practices run genuine Lighthouse audits, but a model can
# still write vague findings ("could improve accessibility") instead of
# citing the real score and the specific failing checks -- inject a
# canonical finding built directly from the tool result if the model's own
# findings don't cite it.
#
# NOTE: this must check *topic* overlap, not just whether the literal score
# number appears in the text, AND that overlap check needs to tolerate
# normal English variation (plurals, word order), not just exact substrings.
# Observed in the wild, twice, in two different ways (both against the
# accessibility audit, but the same matcher is shared with best-practices
# below since the failure mode is generic to "any Lighthouse audit list"):
#   1. The model wrote four findings clearly sourced from this exact tool
#      call but never spelled out the score number -- a literal-digit-only
#      check treated that as "no real data cited" and appended a duplicate.
#   2. The model wrote "required child roles" for an audit whose id/title
#      say "required children", and "List items not contained within..."
#      for an audit id "listitem" (one fused word) -- exact-substring
#      matching missed both as "not covered" and injected duplicates of
#      findings that were, in substance, already there.
# The matcher below stems words (including the "children"/"child" irregular
# case) and requires only a majority-of-keywords overlap, using both the
# audit's id AND its human-readable title as the keyword source, so
# ordinary rewording no longer produces a false "not covered".

_LIGHTHOUSE_AUDIT_STOPWORDS = {
    "with", "that", "some", "those", "this", "have", "has", "are", "not",
    "the", "and", "for", "within", "which", "their", "does", "will",
}
_LIGHTHOUSE_AUDIT_IRREGULAR_STEMS = {"children": "child"}


def _lighthouse_stem(word: str) -> str:
    w = word.lower()
    if w in _LIGHTHOUSE_AUDIT_IRREGULAR_STEMS:
        return _LIGHTHOUSE_AUDIT_IRREGULAR_STEMS[w]
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("es") and len(w) > 4:
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        return w[:-1]
    return w


def _lighthouse_tokenize(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z]+", text.lower())
    return {_lighthouse_stem(w) for w in words if len(w) >= 3}


def _latest_tool_result(tool_call_log: list[dict], tool_name: str) -> dict | None:
    for entry in reversed(tool_call_log):
        if entry.get("name") == tool_name:
            result = entry.get("result") or {}
            if result.get("ok"):
                return result
    return None


def keyword_topic_covered(keywords: list[str] | set[str], text: str | None = None, *,
                           tokens: set[str] | None = None, threshold_fraction: float = 0.5) -> bool:
    """Public, general-purpose version of the stemmed/majority-keyword
    overlap check: True if a majority of the given keywords (stemmed, to
    tolerate plurals like "children"/"child") appear somewhere in the
    target text, without requiring an exact phrase match. This is the same
    matching logic behind _lighthouse_audit_topic_already_covered below --
    exposed here as a public, reusable primitive because the eval harness
    needs the identical "does this text substantively mention topic X"
    judgment to grade whether a report caught a known issue, and
    reimplementing it a second time risks reintroducing the exact bugs
    already fixed once (plurals, word order, fused words like "listitem"
    all producing false "not covered" negatives with naive substring
    matching).

    Pass either `text` (raw, will be tokenized) or a pre-tokenized `tokens`
    set (to avoid re-tokenizing the same findings text once per audit, as
    the internal Lighthouse-reconciliation callers below do)."""
    stemmed_keywords = {_lighthouse_stem(k) for k in keywords if len(k) >= 3}
    if not stemmed_keywords:
        return False
    if tokens is None:
        tokens = _lighthouse_tokenize(text or "")
    matched = sum(1 for k in stemmed_keywords if k in tokens)
    threshold = max(1, math.ceil(len(stemmed_keywords) * threshold_fraction))
    return matched >= threshold


def _lighthouse_audit_topic_already_covered(audit: dict, findings_tokens: set[str]) -> bool:
    """True if the model's findings already substantively reference this
    specific failing Lighthouse audit -- judged by stemmed-keyword overlap
    (from both the audit's id and its human-readable title) against the
    model's own findings text, rather than requiring an exact substring
    match. A majority of the audit's significant keywords being present is
    treated as "covered"; ordinary rewording (plurals, word order,
    "children" vs "child") no longer produces a false "not covered"."""
    id_words = re.split(r"[-_]", audit.get("id", ""))
    title_words = re.findall(r"[a-zA-Z]+", audit.get("title", ""))
    candidates = [w for w in id_words + title_words if len(w) >= 4 and w.lower() not in _LIGHTHOUSE_AUDIT_STOPWORDS]
    return keyword_topic_covered(candidates, tokens=findings_tokens)


def _mentions_score(findings_text: str, score) -> bool:
    """True if the findings text cites this exact score number in a way
    that plausibly means "the score", not just any nearby digit. A raw
    substring check is too loose -- observed in the wild (well, in a
    combined-findings scenario): a score of 50 was treated as "already
    cited" because the digits "50" happened to appear inside an unrelated
    "30-50% of real WCAG issues" caveat elsewhere in the text. Requiring
    the number not be immediately adjacent to another digit or a '%' sign
    rules out both "150" containing "50" and "30-50%" ranges/percentages
    while still matching ordinary phrasing like "score: 50/100" or "a 50
    out of 100"."""
    if score is None:
        return False
    pattern = rf"(?<!\d){re.escape(str(round(score)))}(?!\d)(?!%)"
    return re.search(pattern, findings_text) is not None


def _reconcile_lighthouse_category_data(
    specialist_result: dict, tool_call_log: list[dict], *,
    tool_name: str, score_field: str, failing_audits_field: str,
    category_label: str, availability_flag: str, floor_note: str, log_fn=None,
) -> dict:
    """Shared implementation behind reconcile_accessibility_data and
    reconcile_best_practices_data -- both categories have the identical
    shape of failure mode and identical fix, so this is the one place that
    logic lives (rather than two copies that could drift out of sync, the
    way the accessibility-only version already had two real bugs fixed in
    it)."""
    tool_result = _latest_tool_result(tool_call_log, tool_name)
    if tool_result is None:
        return specialist_result

    score = tool_result.get(score_field)
    _pin_measured_score(specialist_result, score, log_fn)

    findings = specialist_result.get("findings", [])
    findings_text = " ".join(
        f"{f.get('issue', '')} {f.get('recommendation', '')}" for f in findings
    ).lower()
    findings_tokens = _lighthouse_tokenize(findings_text)

    mentions_real_score = _mentions_score(findings_text, score)
    failing = tool_result.get(failing_audits_field, [])
    uncovered = [a for a in failing if not _lighthouse_audit_topic_already_covered(a, findings_tokens)]

    if mentions_real_score:
        return specialist_result  # explicitly grounded already, nothing to force
    if failing and not uncovered:
        return specialist_result  # every failing check is already substantively covered

    severity = "critical" if (score is not None and score < 50) else (
        "warning" if (score is not None and score < 90) else "good"
    )
    # A high score can genuinely coexist with a "failing" audit -- Lighthouse
    # has zero-weight/informational audits (e.g. "missing source maps") that
    # don't count toward the category score. That's real, accurate data, but
    # presenting it as "OK ... score: 100/100. Failing checks include: X" is
    # self-contradictory to read regardless of the underlying nuance being
    # correct. Never label a finding "good" while it's also naming a failing
    # check -- observed for real: a 100/100 Best Practices score with one
    # zero-weight failing audit got shipped as an "OK" finding that also
    # said "Failing checks include: Missing source maps...".
    if uncovered and severity == "good":
        severity = "warning"

    if uncovered:
        examples = "; ".join(a.get("title", a.get("id", "?")) for a in uncovered[:5])
        issue = f"Real Lighthouse {category_label} score: {score}/100. Failing checks include: {examples}."
        recommendation = (
            f"Fix the specific failing checks above first (highest-impact, most concrete). {floor_note}"
        )
    else:
        issue = f"Real Lighthouse {category_label} score: {score}/100. No automated check failures detected."
        recommendation = f"No action required for automated checks. {floor_note}"

    canonical = {"severity": severity, "issue": issue, "recommendation": recommendation}
    findings.append(canonical)
    specialist_result["findings"] = findings
    specialist_result[availability_flag] = True

    if log_fn:
        log_fn(
            f"  -> Injected real {category_label} audit data into "
            f"'{specialist_result.get('category', '?')}' since the model's findings "
            f"didn't cite the actual measured score."
        )

    return specialist_result


def reconcile_accessibility_data(specialist_result: dict, tool_call_log: list[dict], log_fn=None) -> dict:
    """No-op if check_accessibility_and_best_practices wasn't called or
    failed, or if the model's findings already substantively reflect the
    real audit data (whether or not they spell out the exact score number).
    Otherwise append a canonical finding built from the actual audit data
    -- limited to whichever specific failing checks the model didn't
    already cover, so a partially-grounded response gets topped up rather
    than duplicated."""
    return _reconcile_lighthouse_category_data(
        specialist_result, tool_call_log,
        tool_name="check_accessibility_and_best_practices",
        score_field="accessibility_score_0_100",
        failing_audits_field="failing_accessibility_audits",
        category_label="accessibility",
        availability_flag="_real_accessibility_available",
        floor_note=(
            "Automated checks like this catch roughly 30-50% of real WCAG issues, so also "
            "consider manual or assistive-tech (screen reader) testing beyond this list."
        ),
        log_fn=log_fn,
    )


def reconcile_best_practices_data(specialist_result: dict, tool_call_log: list[dict], log_fn=None) -> dict:
    """Same mechanism as reconcile_accessibility_data, for the Best
    Practices category: no-op if check_best_practices wasn't called/failed
    or the model already substantively covered the real failing checks;
    otherwise tops up whatever it missed."""
    return _reconcile_lighthouse_category_data(
        specialist_result, tool_call_log,
        tool_name="check_best_practices",
        score_field="best_practices_score_0_100",
        failing_audits_field="failing_best_practices_audits",
        category_label="Best Practices",
        availability_flag="_real_best_practices_available",
        floor_note="These are automated checks -- always worth a manual spot-check too.",
        log_fn=log_fn,
    )


_STALE_NO_CWV_PHRASES = ("no real core web vitals", "no core web vitals", "not include real core web vitals")


def fix_stale_cwv_data_limitations(report: dict, had_real_cwv: bool, log_fn=None) -> None:
    """If real Core Web Vitals data WAS obtained this run, but data_limitations
    still claims otherwise (stale boilerplate text), correct it."""
    if not had_real_cwv:
        return
    dl = report.get("data_limitations", "")
    dl_lower = dl.lower()
    if any(phrase in dl_lower for phrase in _STALE_NO_CWV_PHRASES):
        if log_fn:
            log_fn("  -> data_limitations claims no real Core Web Vitals data, but "
                   "check_core_web_vitals succeeded this run; correcting.")
        report["data_limitations"] = (
            dl.rstrip()
            + " (Note: real Core Web Vitals/Lighthouse data WAS obtained this run via the "
              "PageSpeed Insights API -- the claim above about lacking it is stale/inaccurate; "
              "see the Page Speed category for the actual measured numbers.)"
        )