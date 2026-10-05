"""Tests for agent/postprocess.py -- the deterministic ground-truth layer.

Each test here corresponds to a specific, documented bug in PROJECT_HANDOFF.md
(hallucinated SSL status, competitive-specialist overlap, summary/trend
mismatches, bot-blocked fetches poisoning findings, CWV data being ignored).
"""
from agent import postprocess


# --------------------------------------------------------------------------
# reconcile_ssl_findings
# --------------------------------------------------------------------------

class TestReconcileSslFindings:
    def test_noop_when_ssl_tool_never_called(self):
        result = {"category": "Web Security", "findings": [{"severity": "good", "issue": "fine", "recommendation": ""}]}
        out = postprocess.reconcile_ssl_findings(result, tool_call_log=[])
        assert out["findings"] == result["findings"]

    def test_valid_cert_overrides_model_claim_of_expiry(self, sample_ssl_tool_call_log_valid):
        # Model hallucinated an expired-certificate finding despite the tool
        # proving the cert is valid -- this is the exact bug from the handoff.
        result = {
            "category": "Web Security",
            "findings": [
                {"severity": "critical", "issue": "SSL certificate has expired.", "recommendation": "Renew it."},
                {"severity": "good", "issue": "Unrelated: HSTS header present.", "recommendation": ""},
            ],
        }
        out = postprocess.reconcile_ssl_findings(result, sample_ssl_tool_call_log_valid)
        ssl_findings = [f for f in out["findings"] if "certificate" in f["issue"].lower()]
        assert len(ssl_findings) == 1
        assert ssl_findings[0]["severity"] == "good"
        assert "VALID" in ssl_findings[0]["issue"]
        # Unrelated finding must survive untouched.
        assert any("HSTS" in f["issue"] for f in out["findings"])

    def test_expired_cert_produces_critical_finding(self, sample_ssl_tool_call_log_expired):
        result = {"category": "Web Security", "findings": []}
        out = postprocess.reconcile_ssl_findings(result, sample_ssl_tool_call_log_expired)
        assert len(out["findings"]) == 1
        assert out["findings"][0]["severity"] == "critical"
        assert "EXPIRED" in out["findings"][0]["issue"]

    def test_failed_handshake_produces_critical_finding(self):
        log = [{"name": "check_ssl_certificate", "result": {
            "ok": True, "has_valid_ssl": False, "error": "connection refused",
        }}]
        out = postprocess.reconcile_ssl_findings({"findings": []}, log)
        assert out["findings"][0]["severity"] == "critical"
        assert "connection refused" in out["findings"][0]["issue"]

    def test_uses_latest_ssl_call_if_multiple(self):
        log = [
            {"name": "check_ssl_certificate", "result": {"ok": True, "has_valid_ssl": True, "is_expired": True, "ssl_status_summary": "OLD"}},
            {"name": "check_ssl_certificate", "result": {"ok": True, "has_valid_ssl": True, "is_expired": False, "ssl_status_summary": "Certificate is VALID and NOT expired (NEW)."}},
        ]
        out = postprocess.reconcile_ssl_findings({"findings": []}, log)
        assert out["findings"][0]["severity"] == "good"
        assert "NEW" in out["findings"][0]["issue"]

    def test_ignores_failed_ssl_tool_calls(self):
        log = [{"name": "check_ssl_certificate", "result": {"ok": False, "error": "timeout"}}]
        result = {"findings": [{"severity": "good", "issue": "unrelated", "recommendation": ""}]}
        out = postprocess.reconcile_ssl_findings(result, log)
        # ok: False result should be treated as if the tool was never called.
        assert out["findings"] == result["findings"]


# --------------------------------------------------------------------------
# strip_competitive_onpage_overlap
# --------------------------------------------------------------------------

class TestStripCompetitiveOnpageOverlap:
    def test_removes_title_length_judgment(self):
        result = {"category": "Competitive", "findings": [
            {"severity": "warning", "issue": "The title is well within the 50-60 character recommendation.", "recommendation": ""},
            {"severity": "good", "issue": "Structured data usage matches industry norms.", "recommendation": ""},
        ]}
        out = postprocess.strip_competitive_onpage_overlap(result)
        assert len(out["findings"]) == 1
        assert "Structured data" in out["findings"][0]["issue"]

    def test_removes_any_canonical_mention_unconditionally(self):
        result = {"findings": [
            {"severity": "good", "issue": "Canonical link element present and correctly points to the primary URL.", "recommendation": ""},
        ]}
        out = postprocess.strip_competitive_onpage_overlap(result)
        assert out["findings"] == []

    def test_meta_description_requires_context_word(self):
        # "meta description" alone with no length/quality context shouldn't
        # match by topic-keyword alone unless a context word is present too.
        result = {"findings": [
            {"severity": "good", "issue": "meta description is missing entirely", "recommendation": ""},
        ]}
        out = postprocess.strip_competitive_onpage_overlap(result)
        assert out["findings"] == []  # "missing" is a context keyword -> stripped

    def test_keeps_findings_with_no_overlap(self):
        result = {"findings": [
            {"severity": "warning", "issue": "Competitor uses more schema.org markup types.", "recommendation": "Add Product schema."},
        ]}
        out = postprocess.strip_competitive_onpage_overlap(result)
        assert len(out["findings"]) == 1


# --------------------------------------------------------------------------
# fix_summary_trend_mismatch
# --------------------------------------------------------------------------

class TestFixSummaryTrendMismatch:
    def test_direction_mismatch_appends_note(self):
        report = {
            "trend": {"score_delta": -5.0, "previous_score": 80, "previous_timestamp": "t"},
            "summary": "The site has improved significantly since the last audit.",
        }
        postprocess.fix_summary_trend_mismatch(report)
        assert "Note:" in report["summary"]

    def test_magnitude_mismatch_appends_note(self):
        report = {
            "trend": {"score_delta": 8.0, "previous_score": 70, "previous_timestamp": "t"},
            "summary": "Scores improved by 1.5 points since the last audit.",
        }
        postprocess.fix_summary_trend_mismatch(report)
        assert "Note:" in report["summary"]

    def test_matching_direction_and_magnitude_is_left_alone(self):
        report = {
            "trend": {"score_delta": 5.0, "previous_score": 75, "previous_timestamp": "t"},
            "summary": "The score increased by 5 points since the last audit.",
        }
        postprocess.fix_summary_trend_mismatch(report)
        assert "Note:" not in report["summary"]

    def test_noop_without_trend(self):
        report = {"summary": "The site improved a lot."}
        postprocess.fix_summary_trend_mismatch(report)
        assert report["summary"] == "The site improved a lot."

    def test_noop_without_directional_claim(self):
        report = {
            "trend": {"score_delta": -5.0},
            "summary": "The site has several accessibility issues to fix.",
        }
        postprocess.fix_summary_trend_mismatch(report)
        assert "Note:" not in report["summary"]


# --------------------------------------------------------------------------
# fix_fabricated_trend_claim
# --------------------------------------------------------------------------

class TestFixFabricatedTrendClaim:
    def test_flags_fabricated_comparison_on_first_audit(self):
        report = {"summary": "Compared to the previous audit, this site has improved."}
        postprocess.fix_fabricated_trend_claim(report)
        assert "fabricated" in report["summary"].lower()

    def test_noop_when_real_trend_exists(self):
        report = {"trend": {"score_delta": 2.0}, "summary": "Compared to the previous audit, up slightly."}
        postprocess.fix_fabricated_trend_claim(report)
        assert "fabricated" not in report["summary"].lower()

    def test_noop_when_no_previous_audit_phrase_used(self):
        report = {"summary": "This site has strong technical SEO fundamentals."}
        postprocess.fix_fabricated_trend_claim(report)
        assert report["summary"] == "This site has strong technical SEO fundamentals."


# --------------------------------------------------------------------------
# reconcile_likely_blocked
# --------------------------------------------------------------------------

class TestReconcileLikelyBlocked:
    def test_overrides_findings_when_blocked(self):
        log = [{"name": "fetch_page", "result": {"ok": True, "status_code": 403, "likely_blocked": True}}]
        result = {"category": "Content", "score": 20, "findings": [{"severity": "critical", "issue": "No H1 found", "recommendation": "Add one"}]}
        out = postprocess.reconcile_likely_blocked(result, log)
        assert out["score"] is None
        assert len(out["findings"]) == 1
        assert "block" in out["findings"][0]["issue"].lower()

    def test_noop_when_not_blocked(self):
        log = [{"name": "fetch_page", "result": {"ok": True, "status_code": 200, "likely_blocked": False}}]
        result = {"score": 90, "findings": [{"severity": "good", "issue": "fine", "recommendation": ""}]}
        out = postprocess.reconcile_likely_blocked(result, log)
        assert out["score"] == 90
        assert out["findings"] == result["findings"]

    def test_noop_when_fetch_page_never_called(self):
        result = {"score": 90, "findings": []}
        out = postprocess.reconcile_likely_blocked(result, [])
        assert out["score"] == 90

    def test_uses_latest_fetch_page_call(self):
        log = [
            {"name": "fetch_page", "result": {"ok": True, "status_code": 200, "likely_blocked": False}},
            {"name": "fetch_page", "result": {"ok": True, "status_code": 503, "likely_blocked": True}},
        ]
        result = {"score": 90, "findings": []}
        out = postprocess.reconcile_likely_blocked(result, log)
        assert out["score"] is None


# --------------------------------------------------------------------------
# reconcile_core_web_vitals
# --------------------------------------------------------------------------

class TestReconcileCoreWebVitals:
    CWV_LOG = [{
        "name": "check_core_web_vitals",
        "result": {
            "ok": True,
            "lab_data": {"performance_score_0_100": 42, "lcp_ms": 4200, "cls": 0.25},
            "good_thresholds_2026": {"lcp_ms": 2500, "cls": 0.1, "inp_ms": 200},
        },
    }]

    def test_injects_canonical_finding_when_model_ignored_real_data(self):
        result = {"category": "Page Speed", "findings": [
            {"severity": "warning", "issue": "Page load time is higher than expected.", "recommendation": "Optimize images."},
        ]}
        out = postprocess.reconcile_core_web_vitals(result, self.CWV_LOG)
        assert out["_real_cwv_available"] is True
        assert any("42" in f["issue"] for f in out["findings"])
        assert any(f["severity"] == "critical" for f in out["findings"])  # score 42 < 50

    def test_noop_when_model_already_cites_real_numbers(self):
        result = {"findings": [
            {"severity": "critical", "issue": "Lighthouse performance score is 42/100 with LCP at 4200ms.", "recommendation": "Fix it."},
        ]}
        out = postprocess.reconcile_core_web_vitals(result, self.CWV_LOG)
        assert len(out["findings"]) == 1  # nothing appended
        # The flag means "real data was obtained", not "a finding was injected"
        # (see TestMeasuredScoresAndAvailabilityFlags for the bug that was).
        assert out["_real_cwv_available"] is True

    def test_noop_when_cwv_tool_never_called(self):
        result = {"findings": []}
        out = postprocess.reconcile_core_web_vitals(result, [])
        assert out["findings"] == []

    def test_noop_when_cwv_tool_failed(self):
        log = [{"name": "check_core_web_vitals", "result": {"ok": False, "error": "timeout"}}]
        result = {"findings": []}
        out = postprocess.reconcile_core_web_vitals(result, log)
        assert out["findings"] == []

    def test_good_severity_for_high_score(self):
        log = [{"name": "check_core_web_vitals", "result": {
            "ok": True, "lab_data": {"performance_score_0_100": 96, "lcp_ms": 1200, "cls": 0.02},
            "good_thresholds_2026": {"lcp_ms": 2500, "cls": 0.1, "inp_ms": 200},
        }}]
        out = postprocess.reconcile_core_web_vitals({"findings": []}, log)
        assert out["findings"][0]["severity"] == "good"


# --------------------------------------------------------------------------
# fix_stale_cwv_data_limitations
# --------------------------------------------------------------------------

class TestFixStaleCwvDataLimitations:
    def test_corrects_stale_claim_when_real_data_was_obtained(self):
        report = {"data_limitations": "No real Core Web Vitals data was available for this audit."}
        postprocess.fix_stale_cwv_data_limitations(report, had_real_cwv=True)
        assert "Note:" in report["data_limitations"]

    def test_noop_when_no_real_cwv_data_obtained(self):
        report = {"data_limitations": "No real Core Web Vitals data was available."}
        postprocess.fix_stale_cwv_data_limitations(report, had_real_cwv=False)
        assert "Note:" not in report["data_limitations"]

    def test_noop_when_text_is_not_stale(self):
        report = {"data_limitations": "Link checking was a sample, not a full crawl."}
        postprocess.fix_stale_cwv_data_limitations(report, had_real_cwv=True)
        assert "Note:" not in report["data_limitations"]


# --------------------------------------------------------------------------
# reconcile_accessibility_data
# --------------------------------------------------------------------------

class TestReconcileAccessibilityData:
    A11Y_LOG = [{
        "name": "check_accessibility_and_best_practices",
        "result": {
            "ok": True,
            "accessibility_score_0_100": 71,
            "failing_accessibility_audits": [
                {"id": "color-contrast", "title": "Insufficient color contrast", "description": "..."},
                {"id": "image-alt", "title": "Images missing alt attributes", "description": "..."},
            ],
        },
    }]

    def test_injects_canonical_finding_when_model_didnt_cite_score(self):
        result = {"category": "Accessibility", "findings": [
            {"severity": "warning", "issue": "Accessibility could be improved in a few places.", "recommendation": "Review manually."},
        ]}
        out = postprocess.reconcile_accessibility_data(result, self.A11Y_LOG)
        assert out["_real_accessibility_available"] is True
        assert any("71" in f["issue"] for f in out["findings"])
        assert any("color-contrast" in f["issue"] or "Insufficient color contrast" in f["issue"] for f in out["findings"])

    def test_noop_when_model_already_cites_real_score(self):
        result = {"findings": [
            {"severity": "warning", "issue": "Lighthouse accessibility score is 71/100.", "recommendation": "Fix contrast issues."},
        ]}
        out = postprocess.reconcile_accessibility_data(result, self.A11Y_LOG)
        assert len(out["findings"]) == 1
        assert "_real_accessibility_available" not in out

    def test_noop_when_tool_never_called(self):
        result = {"findings": []}
        out = postprocess.reconcile_accessibility_data(result, [])
        assert out["findings"] == []

    def test_noop_when_tool_failed(self):
        log = [{"name": "check_accessibility_and_best_practices", "result": {"ok": False, "error": "timeout"}}]
        result = {"findings": []}
        out = postprocess.reconcile_accessibility_data(result, log)
        assert out["findings"] == []

    def test_critical_severity_for_low_score(self):
        log = [{"name": "check_accessibility_and_best_practices", "result": {
            "ok": True, "accessibility_score_0_100": 30, "failing_accessibility_audits": [],
        }}]
        out = postprocess.reconcile_accessibility_data({"findings": []}, log)
        assert out["findings"][0]["severity"] == "critical"

    def test_good_severity_for_high_score_with_no_failures(self):
        log = [{"name": "check_accessibility_and_best_practices", "result": {
            "ok": True, "accessibility_score_0_100": 96, "failing_accessibility_audits": [],
        }}]
        out = postprocess.reconcile_accessibility_data({"findings": []}, log)
        assert out["findings"][0]["severity"] == "good"
        assert "No automated check failures" in out["findings"][0]["issue"]

    def test_warning_severity_for_mid_range_score(self):
        log = [{"name": "check_accessibility_and_best_practices", "result": {
            "ok": True, "accessibility_score_0_100": 75, "failing_accessibility_audits": [],
        }}]
        out = postprocess.reconcile_accessibility_data({"findings": []}, log)
        assert out["findings"][0]["severity"] == "warning"

    def test_no_duplicate_when_model_already_covered_all_failures_without_stating_score(self):
        """Regression test for a real observed bug: the model wrote accurate,
        specific findings clearly sourced from this exact tool call (matching
        each failing audit's topic) but never spelled out the literal score
        number -- that must NOT be treated as 'didn't cite real data' and
        must NOT produce a duplicate finding repeating the same four issues."""
        log = [{"name": "check_accessibility_and_best_practices", "result": {
            "ok": True,
            "accessibility_score_0_100": 88,
            "failing_accessibility_audits": [
                {"id": "color-contrast", "title": "Background and foreground colors do not have a sufficient contrast ratio.", "description": ""},
                {"id": "frame-title", "title": "`<frame>` or `<iframe>` elements do not have a title", "description": ""},
                {"id": "heading-order", "title": "Heading elements are not in a sequentially-descending order", "description": ""},
                {"id": "link-name", "title": "Links do not have a discernible name", "description": ""},
            ],
        }}]
        result = {"category": "Accessibility", "findings": [
            {"severity": "warning", "issue": "Insufficient color contrast", "recommendation": "Improve color contrast between background and foreground elements"},
            {"severity": "warning", "issue": "Missing frame titles", "recommendation": "Add titles to frame and iframe elements"},
            {"severity": "warning", "issue": "Incorrect heading order", "recommendation": "Ensure heading elements are in a sequentially-descending order"},
            {"severity": "warning", "issue": "Links without discernible names", "recommendation": "Make links accessible by adding discernible and unique link text"},
        ]}
        out = postprocess.reconcile_accessibility_data(result, log)
        assert len(out["findings"]) == 4  # nothing appended
        assert "_real_accessibility_available" not in out

    def test_tops_up_only_the_uncovered_failing_checks(self):
        """Partial coverage: model covered one failing check but missed the
        other -- only the uncovered one should be added, not both again."""
        log = [{"name": "check_accessibility_and_best_practices", "result": {
            "ok": True,
            "accessibility_score_0_100": 60,
            "failing_accessibility_audits": [
                {"id": "color-contrast", "title": "Insufficient color contrast", "description": ""},
                {"id": "image-alt", "title": "Images are missing alt attributes", "description": ""},
            ],
        }}]
        result = {"findings": [
            {"severity": "warning", "issue": "Color contrast is too low in several places.", "recommendation": "Fix it."},
        ]}
        out = postprocess.reconcile_accessibility_data(result, log)
        assert len(out["findings"]) == 2
        injected = out["findings"][1]
        assert "image" in injected["issue"].lower()
        assert "color" not in injected["issue"].lower()  # already-covered one not repeated

    def test_no_duplicate_for_plural_and_word_order_variation(self):
        """Regression test for a second real observed bug: the model wrote
        'ARIA parent roles missing required child roles' (singular 'child')
        against an audit whose title says '...required children', and 'List
        items not contained within...' against an audit id 'listitem' (one
        fused word). Exact-substring matching missed both as 'uncovered'
        and injected a duplicate. Stemmed/majority-keyword matching must
        catch both as already covered."""
        log = [{"name": "check_accessibility_and_best_practices", "result": {
            "ok": True,
            "accessibility_score_0_100": 77,
            "failing_accessibility_audits": [
                {
                    "id": "aria-required-children",
                    "title": "Elements with an ARIA [role] that require children to contain a "
                              "specific [role] are missing some or all of those required children.",
                    "description": "",
                },
                {
                    "id": "listitem",
                    "title": "List items (<li>) are not contained within <ul>, <ol> or <menu> "
                              "parent elements.",
                    "description": "",
                },
            ],
        }}]
        result = {"category": "Accessibility", "findings": [
            {"severity": "warning", "issue": "ARIA parent roles missing required child roles",
             "recommendation": "Add required child roles to ARIA parent roles to ensure proper accessibility functions"},
            {"severity": "warning", "issue": "List items not contained within ul, ol, or menu parent elements",
             "recommendation": "Contain list items within ul, ol, or menu parent elements to ensure proper screen reader announcement"},
        ]}
        out = postprocess.reconcile_accessibility_data(result, log)
        assert len(out["findings"]) == 2  # nothing appended
        assert "_real_accessibility_available" not in out

    def test_majority_overlap_is_enough_not_exact_match(self):
        """A finding doesn't need to restate every keyword -- a solid
        majority of the audit's significant keywords being present should
        count as covered."""
        log = [{"name": "check_accessibility_and_best_practices", "result": {
            "ok": True,
            "accessibility_score_0_100": 80,
            "failing_accessibility_audits": [
                {"id": "button-name", "title": "Buttons do not have an accessible name", "description": ""},
            ],
        }}]
        result = {"findings": [
            {"severity": "warning", "issue": "Some buttons are missing accessible names.", "recommendation": "Add labels."},
        ]}
        out = postprocess.reconcile_accessibility_data(result, log)
        assert len(out["findings"]) == 1  # covered, nothing appended

    def test_genuinely_unrelated_finding_still_triggers_injection(self):
        """Sanity check the improved matcher doesn't become too lenient --
        a finding about a completely different topic must still count as
        'uncovered' and trigger injection."""
        log = [{"name": "check_accessibility_and_best_practices", "result": {
            "ok": True,
            "accessibility_score_0_100": 40,
            "failing_accessibility_audits": [
                {"id": "color-contrast", "title": "Background and foreground colors do not have a sufficient contrast ratio", "description": ""},
            ],
        }}]
        result = {"findings": [
            {"severity": "good", "issue": "The site has a valid sitemap.", "recommendation": "No action needed."},
        ]}
        out = postprocess.reconcile_accessibility_data(result, log)
        assert len(out["findings"]) == 2
        assert out["_real_accessibility_available"] is True


# --------------------------------------------------------------------------
# reconcile_best_practices_data
# --------------------------------------------------------------------------

class TestReconcileBestPracticesData:
    BP_LOG = [{
        "name": "check_best_practices",
        "result": {
            "ok": True,
            "best_practices_score_0_100": 67,
            "failing_best_practices_audits": [
                {"id": "no-vulnerable-libraries", "title": "Includes JS libraries with known vulnerabilities", "description": "..."},
                {"id": "deprecations", "title": "Uses deprecated APIs", "description": "..."},
            ],
        },
    }]

    def test_injects_canonical_finding_when_model_didnt_cite_score(self):
        result = {"category": "Best Practices", "findings": [
            {"severity": "warning", "issue": "A few best practice issues were found.", "recommendation": "Review manually."},
        ]}
        out = postprocess.reconcile_best_practices_data(result, self.BP_LOG)
        assert out["_real_best_practices_available"] is True
        assert any("67" in f["issue"] for f in out["findings"])

    def test_noop_when_model_already_cites_real_score(self):
        result = {"findings": [
            {"severity": "warning", "issue": "Best Practices score is 67/100.", "recommendation": "Fix vulnerable libraries."},
        ]}
        out = postprocess.reconcile_best_practices_data(result, self.BP_LOG)
        assert len(out["findings"]) == 1
        assert "_real_best_practices_available" not in out

    def test_noop_when_model_already_covers_all_failing_checks(self):
        """Same matcher, same guarantee as accessibility: substantively
        covered findings shouldn't get duplicated just because the exact
        score digit wasn't stated."""
        result = {"findings": [
            {"severity": "warning", "issue": "Includes vulnerable JavaScript libraries.", "recommendation": "Update them."},
            {"severity": "warning", "issue": "Uses deprecated APIs.", "recommendation": "Migrate off them."},
        ]}
        out = postprocess.reconcile_best_practices_data(result, self.BP_LOG)
        assert len(out["findings"]) == 2
        assert "_real_best_practices_available" not in out

    def test_noop_when_tool_never_called(self):
        result = {"findings": []}
        out = postprocess.reconcile_best_practices_data(result, [])
        assert out["findings"] == []

    def test_noop_when_tool_failed(self):
        log = [{"name": "check_best_practices", "result": {"ok": False, "error": "timeout"}}]
        result = {"findings": []}
        out = postprocess.reconcile_best_practices_data(result, log)
        assert out["findings"] == []

    def test_does_not_interfere_with_accessibility_reconciliation(self):
        """The two reconciliation functions must be independent -- one
        specialist's tool_call_log has both real tool calls, but each
        reconciliation function should only act on its own tool's data."""
        log = [
            {"name": "check_accessibility_and_best_practices", "result": {
                "ok": True, "accessibility_score_0_100": 90, "failing_accessibility_audits": [],
            }},
            {"name": "check_best_practices", "result": {
                "ok": True, "best_practices_score_0_100": 50,
                "failing_best_practices_audits": [{"id": "deprecations", "title": "Uses deprecated APIs", "description": ""}],
            }},
        ]
        result = {"findings": []}
        result = postprocess.reconcile_accessibility_data(result, log)
        result = postprocess.reconcile_best_practices_data(result, log)
        assert len(result["findings"]) == 2
        assert result["_real_accessibility_available"] is True
        assert result["_real_best_practices_available"] is True

    def test_coincidental_percentage_digit_doesnt_count_as_score_citation(self):
        """Regression test for a real bug found while testing: a coincidental
        digit match (score=50 vs. the boilerplate phrase '30-50% of real
        WCAG issues' left over from a prior finding) must not be treated as
        'the model already cited the score' -- the number has to plausibly
        refer to the score itself, not just appear anywhere nearby."""
        log = [{"name": "check_best_practices", "result": {
            "ok": True,
            "best_practices_score_0_100": 50,
            "failing_best_practices_audits": [{"id": "deprecations", "title": "Uses deprecated APIs", "description": ""}],
        }}]
        result = {"findings": [
            {"severity": "good", "issue": "Accessibility is solid overall.",
             "recommendation": "Automated checks catch roughly 30-50% of real WCAG issues."},
        ]}
        out = postprocess.reconcile_best_practices_data(result, log)
        assert len(out["findings"]) == 2  # injection still happened
        assert out["_real_best_practices_available"] is True

    def test_score_cited_as_slash_100_is_recognized(self):
        log = [{"name": "check_best_practices", "result": {
            "ok": True, "best_practices_score_0_100": 67, "failing_best_practices_audits": [],
        }}]
        result = {"findings": [
            {"severity": "warning", "issue": "Best Practices score: 67/100.", "recommendation": "Review manually."},
        ]}
        out = postprocess.reconcile_best_practices_data(result, log)
        assert len(out["findings"]) == 1

    def test_no_ok_severity_when_a_failing_check_is_still_named(self):
        """Regression test for a real observed bug: Lighthouse can report a
        perfect 100/100 score while still flagging one zero-weight/
        informational audit as failing (e.g. missing source maps, which
        doesn't count toward the score). That combination is real and
        accurate, but labeling the resulting finding 'good'/OK while it
        also says 'Failing checks include: X' reads as self-contradictory.
        A finding that names a failing check must never be 'good'."""
        log = [{"name": "check_best_practices", "result": {
            "ok": True,
            "best_practices_score_0_100": 100,
            "failing_best_practices_audits": [
                {"id": "source-maps", "title": "Missing source maps for large first-party JavaScript", "description": ""},
            ],
        }}]
        result = {"findings": []}
        out = postprocess.reconcile_best_practices_data(result, log)
        assert out["findings"][0]["severity"] != "good"
        assert "Failing checks include" in out["findings"][0]["issue"]

    def test_still_good_when_perfect_score_and_no_failing_checks_at_all(self):
        """Sanity check the fix isn't overcorrecting -- a genuinely clean
        100/100 with nothing failing should still say 'good'."""
        log = [{"name": "check_best_practices", "result": {
            "ok": True, "best_practices_score_0_100": 100, "failing_best_practices_audits": [],
        }}]
        out = postprocess.reconcile_best_practices_data({"findings": []}, log)
        assert out["findings"][0]["severity"] == "good"
        assert "No automated check failures detected" in out["findings"][0]["issue"]

class TestMeasuredScoresAndAvailabilityFlags:
    def _cwv_log(self, score=43, lcp=5200.0):
        return [{"name": "check_core_web_vitals", "args": {"url": "https://example.com"},
                 "result": {"ok": True, "lab_data": {"performance_score_0_100": score, "lcp_ms": lcp, "cls": 0.3},
                            "good_thresholds_2026": {"lcp_ms": 2500, "cls": 0.1}}}]

    def test_cwv_flag_is_set_when_the_model_already_cited_the_real_numbers(self):
        """Reproduces a real bug: the flag was only set on the path that
        injects a finding, so a specialist that DID cite the real data left
        it unset and the stale "no real Core Web Vitals data" note was never
        corrected -- exactly in the case where the model behaved well."""
        from agent.postprocess import reconcile_core_web_vitals
        result = {"category": "Page Speed", "score": 60, "findings": [
            {"severity": "critical", "issue": "Lighthouse performance score is 43/100, LCP 5200 ms.",
             "recommendation": "Optimize the hero image."}]}
        out = reconcile_core_web_vitals(result, self._cwv_log())
        assert len(out["findings"]) == 1  # nothing injected
        assert out["_real_cwv_available"] is True

    def test_performance_score_is_pinned_to_the_measured_lighthouse_score(self):
        from agent.postprocess import reconcile_core_web_vitals
        result = {"category": "Page Speed", "score": 85, "findings": []}
        out = reconcile_core_web_vitals(result, self._cwv_log(score=43))
        assert out["score"] == 43
        assert out["_measured_score"] == 43

    def test_accessibility_score_is_pinned_even_when_already_grounded(self):
        from agent.postprocess import reconcile_accessibility_data
        log = [{"name": "check_accessibility_and_best_practices", "args": {"url": "https://example.com"},
                "result": {"ok": True, "accessibility_score_0_100": 71, "failing_accessibility_audits": []}}]
        result = {"category": "Accessibility", "score": 95, "findings": [
            {"severity": "warning", "issue": "Lighthouse accessibility score: 71/100.", "recommendation": "x"}]}
        out = reconcile_accessibility_data(result, log)
        assert out["score"] == 71
        assert len(out["findings"]) == 1  # already grounded: nothing injected

    def test_blocked_result_keeps_its_empty_score(self):
        from agent.postprocess import reconcile_likely_blocked, reconcile_core_web_vitals
        log = [{"name": "fetch_page", "args": {"url": "https://example.com"},
                "result": {"ok": True, "status_code": 403, "likely_blocked": True}}] + self._cwv_log()
        result = reconcile_likely_blocked({"category": "Page Speed", "score": 20, "findings": []}, log)
        out = reconcile_core_web_vitals(result, log)
        assert out["_likely_blocked"] is True
        assert out["score"] is None
