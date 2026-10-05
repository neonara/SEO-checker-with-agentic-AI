from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agent import orchestrator


# --------------------------------------------------------------------------
# _reconcile_overall_score
# --------------------------------------------------------------------------

class TestReconcileOverallScore:
    def _log(self):
        return lambda msg: None

    def test_recomputes_wrong_overall_score(self):
        report = {
            "overall_score": 50.0,  # wrong -- model's bad arithmetic
            "grade": "F",
            "categories": [
                {"name": "A", "score": 100.0, "weight": 0.5},
                {"name": "B", "score": 80.0, "weight": 0.5},
            ],
        }
        orchestrator._reconcile_overall_score(report, self._log())
        assert report["overall_score"] == 90.0
        assert report["grade"] == "A"

    def test_leaves_correct_score_untouched(self):
        report = {
            "overall_score": 90.0,
            "grade": "A",
            "categories": [
                {"name": "A", "score": 100.0, "weight": 0.5},
                {"name": "B", "score": 80.0, "weight": 0.5},
            ],
        }
        orchestrator._reconcile_overall_score(report, self._log())
        assert report["overall_score"] == 90.0

    def test_normalizes_weights_that_dont_sum_to_one(self):
        report = {
            "overall_score": 0,
            "categories": [
                {"name": "A", "score": 100.0, "weight": 0.6},
                {"name": "B", "score": 50.0, "weight": 0.6},  # sums to 1.2
            ],
        }
        orchestrator._reconcile_overall_score(report, self._log())
        total_weight = sum(c["weight"] for c in report["categories"])
        assert abs(total_weight - 1.0) < 0.01

    def test_excludes_categories_with_missing_score_from_average(self):
        report = {
            "overall_score": 0,
            "categories": [
                {"name": "A", "score": 100.0, "weight": 0.5},
                {"name": "B", "score": None, "weight": 0.5},  # failed specialist
            ],
        }
        orchestrator._reconcile_overall_score(report, self._log())
        assert report["overall_score"] == 100.0  # only A counted

    def test_noop_when_no_categories(self):
        report = {"overall_score": 42, "categories": []}
        orchestrator._reconcile_overall_score(report, self._log())
        assert report["overall_score"] == 42

    def test_grade_boundaries(self):
        for score, expected_grade in [(95, "A"), (85, "B"), (75, "C"), (65, "D"), (30, "F")]:
            report = {"overall_score": 0, "categories": [{"name": "A", "score": score, "weight": 1.0}]}
            orchestrator._reconcile_overall_score(report, self._log())
            assert report["grade"] == expected_grade, f"score {score} should be grade {expected_grade}"


# --------------------------------------------------------------------------
# _recover_or_drop_empty_categories
# --------------------------------------------------------------------------

class TestRecoverOrDropEmptyCategories:
    def _log(self):
        return lambda msg: None

    def test_recovers_findings_from_specialist_report(self):
        report = {"categories": [{"name": "Link Health", "score": 80, "findings": []}]}
        specialist_reports = {
            "links": {"category": "Link Health", "score": 80, "findings": [
                {"severity": "warning", "issue": "2 broken links found.", "recommendation": "Fix them."},
            ]},
        }
        orchestrator._recover_or_drop_empty_categories(report, specialist_reports, self._log())
        assert len(report["categories"]) == 1
        assert report["categories"][0]["findings"]

    def test_drops_category_with_no_recoverable_findings(self):
        report = {"categories": [{"name": "Link Health", "score": None, "findings": []}]}
        specialist_reports = {
            "links": {"category": "Link Health", "score": None, "findings": [],
                      "raw_evidence_notes": "Specialist failed to complete: timeout"},
        }
        orchestrator._recover_or_drop_empty_categories(report, specialist_reports, self._log())
        assert report["categories"] == []

    def test_keeps_categories_that_already_have_findings(self):
        report = {"categories": [{"name": "Technical SEO", "score": 90, "findings": [
            {"severity": "good", "issue": "fine", "recommendation": ""},
        ]}]}
        orchestrator._recover_or_drop_empty_categories(report, {}, self._log())
        assert len(report["categories"]) == 1

    def test_recovers_score_too_if_draft_score_missing(self):
        report = {"categories": [{"name": "Link Health", "score": None, "findings": []}]}
        specialist_reports = {
            "links": {"category": "Link Health", "score": 65, "findings": [
                {"severity": "warning", "issue": "issue", "recommendation": "fix"},
            ]},
        }
        orchestrator._recover_or_drop_empty_categories(report, specialist_reports, self._log())
        assert report["categories"][0]["score"] == 65


# --------------------------------------------------------------------------
# Full pipeline, everything mocked -- verifies stage wiring, not real agent behavior
# --------------------------------------------------------------------------

class TestRunFullAuditWiring:
    def test_full_pipeline_wires_stages_together(self, monkeypatch):
        monkeypatch.setattr(orchestrator.memory, "get_last_audit", lambda url: None)
        monkeypatch.setattr(orchestrator.memory, "save_audit", lambda url, report: 1)

        monkeypatch.setattr(orchestrator, "run_planner", lambda *a, **kw: {
            "specialists": ["technical_seo"], "reasoning": "test",
        })

        fake_specialist_result = {
            "category": "Technical SEO", "score": 88, "findings": [
                {"severity": "good", "issue": "All good.", "recommendation": ""},
            ],
            "raw_evidence_notes": "checked stuff",
        }
        fake_agent = MagicMock()
        fake_agent.run.return_value = dict(fake_specialist_result)
        fake_agent.tool_call_log = []
        monkeypatch.setattr(orchestrator, "build_specialist", lambda *a, **kw: fake_agent)

        fake_draft = {
            "url": "https://example.com",
            "overall_score": 88.0,
            "grade": "B",
            "summary": "Solid technical foundation.",
            "categories": [{"name": "Technical SEO", "score": 88.0, "weight": 1.0, "findings": [
                {"severity": "good", "issue": "All good.", "recommendation": ""},
            ]}],
            "quick_wins": [],
            "data_limitations": "",
        }
        monkeypatch.setattr(orchestrator, "reflect_and_revise", lambda *a, **kw: (
            dict(fake_draft), [{"round": 1, "review": {"approved": True, "issues": [], "instructions_for_revision": ""}}]
        ))

        result = orchestrator.run_full_audit("https://example.com", use_memory=True, mode="quick")

        assert result["review_status"] == "approved"
        assert result["overall_score"] == 88.0
        assert "_specialist_reports" in result
        assert "_reflection_log" in result
        # No previous audit -> schema validation still includes a "trend" key
        # (its default), but it must be None, not a fabricated trend block.
        assert result["trend"] is None

    def test_not_approved_report_surfaces_unresolved_issues(self, monkeypatch):
        monkeypatch.setattr(orchestrator.memory, "get_last_audit", lambda url: None)
        monkeypatch.setattr(orchestrator.memory, "save_audit", lambda url, report: 1)
        monkeypatch.setattr(orchestrator, "run_planner", lambda *a, **kw: {"specialists": ["technical_seo"], "reasoning": "t"})

        fake_agent = MagicMock()
        fake_agent.run.return_value = {"category": "Technical SEO", "score": 40, "findings": [
            {"severity": "critical", "issue": "Broken.", "recommendation": "Fix."},
        ]}
        fake_agent.tool_call_log = []
        monkeypatch.setattr(orchestrator, "build_specialist", lambda *a, **kw: fake_agent)

        fake_draft = {
            "url": "https://example.com", "overall_score": 40.0, "grade": "F",
            "summary": "Needs work.",
            "categories": [{"name": "Technical SEO", "score": 40.0, "weight": 1.0, "findings": [
                {"severity": "critical", "issue": "Broken.", "recommendation": "Fix."},
            ]}],
            "quick_wins": [], "data_limitations": "",
        }
        monkeypatch.setattr(orchestrator, "reflect_and_revise", lambda *a, **kw: (
            dict(fake_draft), [{"round": 1, "review": {"approved": False, "issues": ["score too low for findings"], "instructions_for_revision": "..."}}]
        ))

        result = orchestrator.run_full_audit("https://example.com", use_memory=True, mode="quick")
        assert result["review_status"] == "not_approved"
        assert result["unresolved_review_issues"] == ["score too low for findings"]

    def test_previous_audit_produces_trend_block(self, monkeypatch):
        previous = {"overall_score": 70.0, "_timestamp": "2026-01-01T00:00:00+00:00"}
        monkeypatch.setattr(orchestrator.memory, "get_last_audit", lambda url: previous)
        monkeypatch.setattr(orchestrator.memory, "save_audit", lambda url, report: 1)
        monkeypatch.setattr(orchestrator, "run_planner", lambda *a, **kw: {"specialists": ["technical_seo"], "reasoning": "t"})

        fake_agent = MagicMock()
        fake_agent.run.return_value = {"category": "Technical SEO", "score": 85, "findings": []}
        fake_agent.tool_call_log = []
        monkeypatch.setattr(orchestrator, "build_specialist", lambda *a, **kw: fake_agent)

        fake_draft = {
            "url": "https://example.com", "overall_score": 85.0, "grade": "B",
            "summary": "Improved since last time.",
            "categories": [{"name": "Technical SEO", "score": 85.0, "weight": 1.0, "findings": []}],
            "quick_wins": [], "data_limitations": "",
        }
        monkeypatch.setattr(orchestrator, "reflect_and_revise", lambda *a, **kw: (
            dict(fake_draft), [{"round": 1, "review": {"approved": True, "issues": [], "instructions_for_revision": ""}}]
        ))

        result = orchestrator.run_full_audit("https://example.com", use_memory=True, mode="quick")
        assert result["trend"]["previous_score"] == 70.0
        assert result["trend"]["score_delta"] == 15.0


# --------------------------------------------------------------------------
# Accessibility specialist wiring
# --------------------------------------------------------------------------

class TestAccessibilitySpecialistWiring:
    def test_accessibility_has_a_canonical_category_name(self):
        assert orchestrator.CANONICAL_CATEGORY_NAMES.get("accessibility") == "Accessibility"

    def test_accessibility_is_in_the_fallback_specialist_list(self, monkeypatch):
        """If the planner fails to return a usable list, run_full_audit falls
        back to a hardcoded default -- accessibility must be in it, or the
        new specialist silently never runs when the planner misbehaves."""
        monkeypatch.setattr(orchestrator.memory, "get_last_audit", lambda url: None)
        monkeypatch.setattr(orchestrator.memory, "save_audit", lambda url, report: 1)
        monkeypatch.setattr(orchestrator, "run_planner", lambda *a, **kw: {"specialists": [], "reasoning": "t"})

        seen_keys = []

        def fake_build_specialist(key, **kw):
            seen_keys.append(key)
            agent = MagicMock()
            agent.run.return_value = {"category": key, "score": 80, "findings": []}
            agent.tool_call_log = []
            return agent

        monkeypatch.setattr(orchestrator, "build_specialist", fake_build_specialist)
        monkeypatch.setattr(orchestrator, "reflect_and_revise", lambda *a, **kw: (
            {"url": "https://example.com", "overall_score": 80.0, "grade": "B", "summary": "s",
             "categories": [], "quick_wins": [], "data_limitations": ""},
            [{"round": 1, "review": {"approved": True, "issues": [], "instructions_for_revision": ""}}],
        ))

        orchestrator.run_full_audit("https://example.com", use_memory=True, mode="quick")
        assert "accessibility" in seen_keys


# --------------------------------------------------------------------------
# _drop_issues_resolved_by_reconciliation
# --------------------------------------------------------------------------

class TestDropIssuesResolvedByReconciliation:
    def test_drops_weight_sum_complaint(self):
        issues = ["The weights of the categories do not sum to approximately 1.0."]
        assert orchestrator._drop_issues_resolved_by_reconciliation(issues) == []

    def test_drops_weight_sum_complaint_regardless_of_exact_wording(self):
        issues = ["Category weights don't sum to 1.0, which is a problem."]
        assert orchestrator._drop_issues_resolved_by_reconciliation(issues) == []

    def test_keeps_substantive_score_calibration_complaints(self):
        issues = ["The overall score does not accurately reflect the severity of critical findings."]
        assert orchestrator._drop_issues_resolved_by_reconciliation(issues) == issues

    def test_keeps_unrelated_issues_and_drops_only_weight_sum_one(self):
        issues = [
            "The weights of the categories do not sum to approximately 1.0.",
            "Some recommendations are vague and non-actionable.",
        ]
        result = orchestrator._drop_issues_resolved_by_reconciliation(issues)
        assert result == ["Some recommendations are vague and non-actionable."]

    def test_empty_list_stays_empty(self):
        assert orchestrator._drop_issues_resolved_by_reconciliation([]) == []

    def test_end_to_end_final_report_never_shows_stale_weight_complaint(self, monkeypatch):
        """The exact scenario observed in the wild: critic's last review
        complains weights don't sum to 1.0, but by the time the final
        report is built, _reconcile_overall_score has already normalized
        them -- the stale complaint must not appear in the printed
        unresolved_review_issues."""
        monkeypatch.setattr(orchestrator.memory, "get_last_audit", lambda url: None)
        monkeypatch.setattr(orchestrator.memory, "save_audit", lambda url, report: 1)
        monkeypatch.setattr(orchestrator, "run_planner", lambda *a, **kw: {"specialists": ["technical_seo"], "reasoning": "t"})

        fake_agent = MagicMock()
        fake_agent.run.return_value = {"category": "Technical SEO", "score": 70, "findings": [
            {"severity": "warning", "issue": "Some issue.", "recommendation": "Fix it."},
        ]}
        fake_agent.tool_call_log = []
        monkeypatch.setattr(orchestrator, "build_specialist", lambda *a, **kw: fake_agent)

        # Draft's weights intentionally don't sum to 1.0 -- _reconcile_overall_score
        # will normalize them before the report is returned.
        fake_draft = {
            "url": "https://example.com", "overall_score": 70.0, "grade": "C",
            "summary": "s",
            "categories": [{"name": "Technical SEO", "score": 70.0, "weight": 0.6, "findings": [
                {"severity": "warning", "issue": "Some issue.", "recommendation": "Fix it."},
            ]}],
            "quick_wins": [], "data_limitations": "",
        }
        monkeypatch.setattr(orchestrator, "reflect_and_revise", lambda *a, **kw: (
            dict(fake_draft), [{"round": 1, "review": {
                "approved": False,
                "issues": ["The weights of the categories do not sum to approximately 1.0.", "Vague recommendations."],
                "instructions_for_revision": "...",
            }}],
        ))

        result = orchestrator.run_full_audit("https://example.com", use_memory=True, mode="quick")
        assert result["review_status"] == "not_approved"
        assert result["unresolved_review_issues"] == ["Vague recommendations."]
        assert sum(c["weight"] for c in result["categories"]) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Best Practices specialist wiring
# --------------------------------------------------------------------------

class TestBestPracticesSpecialistWiring:
    def test_best_practices_has_a_canonical_category_name(self):
        assert orchestrator.CANONICAL_CATEGORY_NAMES.get("best_practices") == "Best Practices"

    def test_best_practices_is_in_the_fallback_specialist_list(self, monkeypatch):
        monkeypatch.setattr(orchestrator.memory, "get_last_audit", lambda url: None)
        monkeypatch.setattr(orchestrator.memory, "save_audit", lambda url, report: 1)
        monkeypatch.setattr(orchestrator, "run_planner", lambda *a, **kw: {"specialists": [], "reasoning": "t"})

        seen_keys = []

        def fake_build_specialist(key, **kw):
            seen_keys.append(key)
            agent = MagicMock()
            agent.run.return_value = {"category": key, "score": 80, "findings": []}
            agent.tool_call_log = []
            return agent

        monkeypatch.setattr(orchestrator, "build_specialist", fake_build_specialist)
        monkeypatch.setattr(orchestrator, "reflect_and_revise", lambda *a, **kw: (
            {"url": "https://example.com", "overall_score": 80.0, "grade": "B", "summary": "s",
             "categories": [], "quick_wins": [], "data_limitations": ""},
            [{"round": 1, "review": {"approved": True, "issues": [], "instructions_for_revision": ""}}],
        ))

        orchestrator.run_full_audit("https://example.com", use_memory=True, mode="quick")
        assert "best_practices" in seen_keys


class TestStartingKeyIndex:
    """Confirms run_full_audit's starting_key_index actually changes which
    key each stage starts on -- this is what lets a caller running many
    audits back-to-back (like the eval harness) spread them across all
    configured keys instead of every single one starting on key 0."""

    def test_planner_receives_the_starting_key_index(self, monkeypatch):
        monkeypatch.setattr(orchestrator.memory, "get_last_audit", lambda url: None)
        monkeypatch.setattr(orchestrator.memory, "save_audit", lambda url, report: 1)
        monkeypatch.setattr(orchestrator, "GROQ_API_KEYS", ["key0", "key1", "key2"])

        captured = {}

        def fake_planner(*a, **kw):
            captured["key_index"] = kw.get("key_index")
            return {"specialists": ["technical_seo"], "reasoning": "t"}

        monkeypatch.setattr(orchestrator, "run_planner", fake_planner)

        fake_agent = MagicMock()
        fake_agent.run.return_value = {"category": "Technical SEO", "score": 80, "findings": []}
        fake_agent.tool_call_log = []
        monkeypatch.setattr(orchestrator, "build_specialist", lambda *a, **kw: fake_agent)
        monkeypatch.setattr(orchestrator, "reflect_and_revise", lambda *a, **kw: (
            {"url": "https://example.com", "overall_score": 80.0, "grade": "B", "summary": "s",
             "categories": [], "quick_wins": [], "data_limitations": ""},
            [{"round": 1, "review": {"approved": True, "issues": [], "instructions_for_revision": ""}}],
        ))

        orchestrator.run_full_audit("https://example.com", starting_key_index=2)
        assert captured["key_index"] == 2

    def test_specialist_key_indices_are_offset_by_starting_key_index(self, monkeypatch):
        monkeypatch.setattr(orchestrator.memory, "get_last_audit", lambda url: None)
        monkeypatch.setattr(orchestrator.memory, "save_audit", lambda url, report: 1)
        monkeypatch.setattr(orchestrator, "GROQ_API_KEYS", ["key0", "key1", "key2"])
        monkeypatch.setattr(orchestrator, "run_planner", lambda *a, **kw: {
            "specialists": ["technical_seo", "content"], "reasoning": "t",
        })

        seen_key_indices = []

        def fake_build_specialist(key, key_index=0, **kw):
            seen_key_indices.append(key_index)
            agent = MagicMock()
            agent.run.return_value = {"category": key, "score": 80, "findings": []}
            agent.tool_call_log = []
            return agent

        monkeypatch.setattr(orchestrator, "build_specialist", fake_build_specialist)
        monkeypatch.setattr(orchestrator, "reflect_and_revise", lambda *a, **kw: (
            {"url": "https://example.com", "overall_score": 80.0, "grade": "B", "summary": "s",
             "categories": [], "quick_wins": [], "data_limitations": ""},
            [{"round": 1, "review": {"approved": True, "issues": [], "instructions_for_revision": ""}}],
        ))

        orchestrator.run_full_audit("https://example.com", starting_key_index=1)
        # 2 specialists, starting_key_index=1, 3 keys -> expect indices {1, 2}, not {0, 1}
        assert sorted(seen_key_indices) == [1, 2]

    def test_default_starting_key_index_is_zero_backward_compatible(self, monkeypatch):
        monkeypatch.setattr(orchestrator.memory, "get_last_audit", lambda url: None)
        monkeypatch.setattr(orchestrator.memory, "save_audit", lambda url, report: 1)
        monkeypatch.setattr(orchestrator, "GROQ_API_KEYS", ["key0", "key1"])

        captured = {}

        def fake_planner(*a, **kw):
            captured["key_index"] = kw.get("key_index")
            return {"specialists": ["technical_seo"], "reasoning": "t"}

        monkeypatch.setattr(orchestrator, "run_planner", fake_planner)
        fake_agent = MagicMock()
        fake_agent.run.return_value = {"category": "Technical SEO", "score": 80, "findings": []}
        fake_agent.tool_call_log = []
        monkeypatch.setattr(orchestrator, "build_specialist", lambda *a, **kw: fake_agent)
        monkeypatch.setattr(orchestrator, "reflect_and_revise", lambda *a, **kw: (
            {"url": "https://example.com", "overall_score": 80.0, "grade": "B", "summary": "s",
             "categories": [], "quick_wins": [], "data_limitations": ""},
            [{"round": 1, "review": {"approved": True, "issues": [], "instructions_for_revision": ""}}],
        ))

        orchestrator.run_full_audit("https://example.com")  # no starting_key_index passed
        assert captured["key_index"] == 0


# --------------------------------------------------------------------------
# Regression: a category with fabricated findings but no valid score
# --------------------------------------------------------------------------

class TestRecoverOrDropInvalidScoreCategories:
    """Reproduces a real bug: when a specialist genuinely fails, the
    synthesizer/critic could still end up with a category that has
    findings (sometimes fabricated) but an invalid score, which fails
    schema validation and would otherwise ship a category showing a
    None/blank score in the final report."""

    def _log(self):
        return lambda msg: None

    def test_drops_category_with_findings_but_no_valid_score_when_specialist_also_failed(self):
        report = {"categories": [{
            "name": "Technical SEO", "score": None,
            "findings": [{"severity": "critical", "issue": "Invalid SSL certificate", "recommendation": "Fix it"}],
        }]}
        specialist_reports = {
            "technical_seo": {"category": "technical_seo", "score": None, "findings": [],
                               "raw_evidence_notes": "This specialist failed to complete..."},
        }
        orchestrator._recover_or_drop_empty_categories(report, specialist_reports, self._log())
        assert report["categories"] == []

    def test_recovers_real_score_when_specialist_score_is_valid(self):
        report = {"categories": [{
            "name": "Technical SEO", "score": None,
            "findings": [{"severity": "warning", "issue": "Missing meta description", "recommendation": "Add one"}],
        }]}
        specialist_reports = {
            "technical_seo": {"category": "technical_seo", "score": 80, "findings": [
                {"severity": "warning", "issue": "Missing meta description", "recommendation": "Add one"},
            ]},
        }
        orchestrator._recover_or_drop_empty_categories(report, specialist_reports, self._log())
        assert len(report["categories"]) == 1
        assert report["categories"][0]["score"] == 80

    def test_keeps_category_untouched_when_findings_and_score_both_already_valid(self):
        report = {"categories": [{
            "name": "Technical SEO", "score": 90,
            "findings": [{"severity": "good", "issue": "All good", "recommendation": ""}],
        }]}
        orchestrator._recover_or_drop_empty_categories(report, {}, self._log())
        assert len(report["categories"]) == 1
        assert report["categories"][0]["score"] == 90

    def test_final_report_never_has_a_null_score_category_after_reconciliation(self):
        """End-to-end guard: after both reconciliation steps, no category
        in the final report should have a non-numeric score -- this is
        exactly what caused the observed Pydantic schema validation
        failure in the wild."""
        report = {"categories": [
            {"name": "Web Security", "score": 60, "findings": [{"severity": "critical", "issue": "x", "recommendation": "y"}]},
            {"name": "Technical SEO", "score": None,
             "findings": [{"severity": "critical", "issue": "Invalid SSL certificate", "recommendation": "Fix it"}]},
        ]}
        specialist_reports = {
            "technical_seo": {"category": "technical_seo", "score": None, "findings": [],
                               "raw_evidence_notes": "This specialist failed to complete..."},
        }
        orchestrator._recover_or_drop_empty_categories(report, specialist_reports, self._log())
        orchestrator._reconcile_overall_score(report, self._log())
        for cat in report["categories"]:
            assert isinstance(cat["score"], (int, float))


class TestFailedSpecialistPlaceholderDoesNotLeakRawJson:
    """Reproduces a real bug: when a specialist fails because the model's
    JSON was truncated/invalid, the raised error message used to embed
    that raw, never-successfully-parsed text verbatim into
    raw_evidence_notes -- which the synthesizer AND critic both then
    treated as real, validated specialist data (citing specific findings/
    scores that existed only inside a JSON blob that was never actually
    parsed)."""

    def test_failure_placeholder_never_contains_raw_json_braces(self):
        from unittest.mock import patch as mock_patch

        # Simulate the exact failure mode: agent.run() raises with a raw,
        # truncated JSON blob embedded in the message (as base_agent.py's
        # ToolAgent.run() actually does on a JSON-repair failure).
        raw_json_blob = (
            '[Technical SEO Specialist] model did not return valid JSON after a repair attempt:\n'
            '{\n  "category": "crawlability and indexability",\n  "score": 0,\n'
            '  "findings": [\n    { "severity": "critical", "issue": "Invalid SSL certificate", '
            '"recommendation": "Verify SSL certificate" }\n  ],\n'
            '  "raw_evidence_notes": "Failed to retrieve HTML"'
        )

        # The other specialists succeed: an audit where too little came back
        # is refused outright now (see TestRefusesToGradeWithoutEvidence).
        with mock_patch("agent.orchestrator.run_planner", return_value={"specialists": ["technical_seo", "content", "security", "links"], "reasoning": "t"}), \
             mock_patch("agent.orchestrator.memory.get_last_audit", return_value=None), \
             mock_patch("agent.orchestrator.memory.save_audit", return_value=1):

            def fake_build_specialist(key, **kw):
                agent = MagicMock()
                agent.tool_call_log = []
                if key == "technical_seo":
                    agent.run.side_effect = RuntimeError(raw_json_blob)
                else:
                    agent.run.return_value = {"category": key, "score": 80, "findings": []}
                return agent

            with mock_patch("agent.orchestrator.build_specialist", side_effect=fake_build_specialist), \
                 mock_patch("agent.orchestrator.reflect_and_revise", return_value=(
                     {"url": "https://example.com", "overall_score": 0.0, "grade": "F", "summary": "s",
                      "categories": [], "quick_wins": [], "data_limitations": ""},
                     [{"round": 1, "review": {"approved": True, "issues": [], "instructions_for_revision": ""}}],
                 )):
                result = orchestrator.run_full_audit("https://example.com", use_memory=True, mode="quick")

        notes = result["_specialist_reports"]["technical_seo"]["raw_evidence_notes"]
        assert "{" not in notes
        assert "Invalid SSL certificate" not in notes
        assert "failed" in notes.lower()

# --------------------------------------------------------------------------
# Refusing to grade without evidence / reporting what did not run
# --------------------------------------------------------------------------

_APPROVED = [{"round": 1, "review": {"approved": True, "issues": [], "instructions_for_revision": ""}}]
_FINDING = {"severity": "warning", "issue": "Something real.", "recommendation": "Fix it."}


def _patch_pipeline(monkeypatch, specialist_keys, build_specialist, draft, previous=None):
    """Wire run_full_audit to fakes. Returns the list save_audit calls land in
    and a counter of synthesis runs."""
    saved, synth_calls = [], []
    monkeypatch.setattr(orchestrator.memory, "get_last_audit", lambda url: previous)
    monkeypatch.setattr(orchestrator.memory, "save_audit", lambda url, report, **kw: saved.append(report) or 1)
    monkeypatch.setattr(orchestrator, "run_planner", lambda *a, **kw: {"specialists": specialist_keys, "reasoning": "t"})
    monkeypatch.setattr(orchestrator, "build_specialist", build_specialist)

    def fake_reflect(*a, **kw):
        synth_calls.append(1)
        return draft, _APPROVED

    monkeypatch.setattr(orchestrator, "reflect_and_revise", fake_reflect)
    return saved, synth_calls


def _blocked_fetch_log(status=403):
    return [{"name": "fetch_page", "args": {"url": "https://example.com"},
             "result": {"ok": True, "status_code": status, "likely_blocked": True}}]


class TestRefusesToGradeWithoutEvidence:
    """Reproduces a real gap: a site that blocked the scanner (or a run where
    the specialists all failed) still went through the synthesizer, which
    wrote an overall score with no category behind it. That invented grade
    was returned as a finished report and saved as audit history."""

    KEYS = ["technical_seo", "content", "performance", "security"]
    INVENTED = {"url": "https://example.com", "overall_score": 72.0, "grade": "C", "summary": "s",
                "categories": [], "quick_wins": [], "data_limitations": ""}

    def test_blocked_site_fails_instead_of_shipping_an_invented_grade(self, monkeypatch):
        def build(key, **kw):
            agent = MagicMock()
            agent.run.return_value = {"category": key, "score": 15, "findings": [_FINDING]}
            agent.tool_call_log = _blocked_fetch_log(403)
            return agent

        saved, synth_calls = _patch_pipeline(monkeypatch, self.KEYS, build, self.INVENTED)
        with pytest.raises(orchestrator.AuditFailed) as excinfo:
            orchestrator.run_full_audit("https://example.com", mode="quick")

        assert excinfo.value.code == "blocked"
        assert "403" in excinfo.value.public_message
        assert synth_calls == [], "no model call should be spent synthesizing a blocked audit"
        assert saved == [], "a blocked audit must not be stored as history"

    def test_all_specialists_failing_fails_the_audit(self, monkeypatch):
        def build(key, **kw):
            agent = MagicMock()
            agent.run.side_effect = RuntimeError("model returned garbage: {raw}")
            agent.tool_call_log = []
            return agent

        saved, synth_calls = _patch_pipeline(monkeypatch, self.KEYS, build, self.INVENTED)
        with pytest.raises(orchestrator.AuditFailed) as excinfo:
            orchestrator.run_full_audit("https://example.com", mode="quick")

        assert excinfo.value.code == "too_few_checks"
        assert "garbage" not in excinfo.value.public_message
        assert synth_calls == [] and saved == []

    def test_audit_failed_is_a_runtime_error_for_the_cli(self):
        assert issubclass(orchestrator.AuditFailed, RuntimeError)


class TestSkippedCategories:
    def test_report_lists_the_checks_that_did_not_run(self, monkeypatch):
        def build(key, **kw):
            agent = MagicMock()
            agent.tool_call_log = []
            if key == "links":
                agent.run.side_effect = RuntimeError("boom")
            else:
                agent.run.return_value = {"category": key, "score": 80, "findings": [_FINDING]}
            return agent

        draft = {"url": "https://example.com", "overall_score": 80.0, "grade": "B", "summary": "s",
                 "categories": [
                     {"name": name, "score": 80, "weight": 0.25, "findings": [_FINDING]}
                     for name in ("Technical SEO", "On-Page Content", "Web Security")
                 ] + [{"name": "Link Health", "score": None, "weight": 0.25, "findings": []}],
                 "quick_wins": [], "data_limitations": ""}
        _patch_pipeline(monkeypatch, ["technical_seo", "content", "security", "links"], build, draft)
        report = orchestrator.run_full_audit("https://example.com", mode="quick")

        assert report["skipped_categories"] == [
            {"name": "Link Health", "reason": "The check failed to complete."},
        ]
        assert "boom" not in str(report["skipped_categories"])

    def test_nothing_skipped_gives_an_empty_list(self, monkeypatch):
        def build(key, **kw):
            agent = MagicMock()
            agent.tool_call_log = []
            agent.run.return_value = {"category": key, "score": 80, "findings": [_FINDING]}
            return agent

        draft = {"url": "https://example.com", "overall_score": 80.0, "grade": "B", "summary": "s",
                 "categories": [{"name": "Technical SEO", "score": 80, "weight": 1.0, "findings": [_FINDING]}],
                 "quick_wins": [], "data_limitations": ""}
        _patch_pipeline(monkeypatch, ["technical_seo"], build, draft)
        assert orchestrator.run_full_audit("https://example.com", mode="quick")["skipped_categories"] == []


class TestStableScores:
    def _log(self):
        return lambda msg: None

    def test_known_categories_get_fixed_weights_whatever_the_model_chose(self):
        """The synthesizer picked its own weights each run, so one unchanged
        site scored differently between runs. Same scores, two different
        model weightings -> must give the same overall score."""
        def report(w_tech, w_speed):
            return {"overall_score": 0, "categories": [
                {"name": "Technical SEO", "score": 90.0, "weight": w_tech},
                {"name": "Page Speed", "score": 40.0, "weight": w_speed},
            ]}

        a, b = report(0.8, 0.2), report(0.2, 0.8)
        orchestrator._reconcile_overall_score(a, self._log())
        orchestrator._reconcile_overall_score(b, self._log())
        assert a["overall_score"] == b["overall_score"] == 67.5  # (90*.22 + 40*.18) / .40
        assert abs(sum(c["weight"] for c in a["categories"]) - 1.0) < 0.01

    def test_fixed_weights_cover_every_canonical_category_and_sum_to_one_without_competitive(self):
        assert set(orchestrator.CATEGORY_WEIGHTS) == set(orchestrator.CANONICAL_CATEGORY_NAMES.values())
        core = sum(w for name, w in orchestrator.CATEGORY_WEIGHTS.items()
                   if name != orchestrator.CANONICAL_CATEGORY_NAMES["competitive"])
        assert abs(core - 1.0) < 1e-9

    def test_measured_lighthouse_score_overrides_the_draft(self):
        report = {"categories": [{"name": "Page Speed", "score": 85, "findings": [_FINDING]},
                                 {"name": "Technical SEO", "score": 70, "findings": [_FINDING]}]}
        specialist_reports = {"performance": {"score": 43, "_measured_score": 43},
                              "technical_seo": {"score": 60}}
        orchestrator._pin_measured_scores(report, specialist_reports, self._log())
        assert report["categories"][0]["score"] == 43
        assert report["categories"][1]["score"] == 70  # nothing measured: the draft stands


class TestTrendBaseline:
    def test_previous_audit_without_a_numeric_score_gives_no_trend_and_no_crash(self, monkeypatch):
        """A stored report can carry overall_score None; the delta arithmetic
        used to raise TypeError at the very end of an otherwise finished audit."""
        def build(key, **kw):
            agent = MagicMock()
            agent.tool_call_log = []
            agent.run.return_value = {"category": key, "score": 80, "findings": [_FINDING]}
            return agent

        draft = {"url": "https://example.com", "overall_score": 80.0, "grade": "B", "summary": "s",
                 "categories": [{"name": "Technical SEO", "score": 80, "weight": 1.0, "findings": [_FINDING]}],
                 "quick_wins": [], "data_limitations": ""}
        _patch_pipeline(monkeypatch, ["technical_seo"], build, draft,
                        previous={"overall_score": None, "_timestamp": "2026-09-01T00:00:00+00:00"})
        report = orchestrator.run_full_audit("https://example.com", mode="quick")
        assert not report.get("trend")


class TestPlannerAndFailureReasons:
    def test_planner_is_a_rule_not_a_model_call(self, monkeypatch):
        from agent import planner, base_agent
        monkeypatch.setattr(base_agent.ToolAgent, "run", lambda *a, **k: pytest.fail("planner called a model"))
        plan = planner.run_planner("https://example.com", None, has_history=False)
        assert plan["specialists"] == planner.CORE_SPECIALISTS
        with_rival = planner.run_planner("https://example.com", "https://rival.com", has_history=True)
        assert with_rival["specialists"] == planner.CORE_SPECIALISTS + ["competitive"]

    @pytest.mark.parametrize("exc, code", [
        (RuntimeError("[X] hit a Groq rate/quota limit requiring a wait longer than 15 minutes"), "quota"),
        (orchestrator.AuditFailed("timeout", "The scan ran out of time."), "timeout"),
        (ValueError("something else"), "too_few_checks"),
    ])
    def test_audit_failure_names_the_real_cause(self, monkeypatch, exc, code):
        """"Too few checks finished" is useless to a visitor when the actual
        cause is our quota or our time budget -- and only those two give the
        rate-limit slot back."""
        def build(key, **kw):
            agent = MagicMock()
            agent.run.side_effect = exc
            agent.tool_call_log = []
            return agent

        _patch_pipeline(monkeypatch, ["technical_seo", "content", "security"], build, {})
        with pytest.raises(orchestrator.AuditFailed) as excinfo:
            orchestrator.run_full_audit("https://example.com", mode="quick")
        assert excinfo.value.code == code

    def test_report_records_how_it_was_run_and_is_saved_under_the_audit_id(self, monkeypatch):
        def build(key, **kw):
            agent = MagicMock()
            agent.tool_call_log = []
            agent.run.return_value = {"category": key, "score": 80, "findings": [_FINDING]}
            return agent

        draft = {"url": "https://example.com", "overall_score": 80.0, "grade": "B", "summary": "s",
                 "categories": [{"name": "Technical SEO", "score": 80, "weight": 1.0, "findings": [_FINDING]}],
                 "quick_wins": [], "data_limitations": ""}
        _patch_pipeline(monkeypatch, ["technical_seo"], build, draft)
        saved_kwargs = {}
        monkeypatch.setattr(orchestrator.memory, "save_audit", lambda url, report, **kw: saved_kwargs.update(kw) or 1)

        report = orchestrator.run_full_audit("https://example.com", mode="quick",
                                             competitor_url="https://rival.com", audit_id="f" * 32)
        assert saved_kwargs == {"public_id": "f" * 32}
        assert report["mode"] == "quick"
        assert report["competitor_url"] == "https://rival.com"
        assert isinstance(report["duration_seconds"], float)
