"""Tests for api.py. run_full_audit is always faked -- no audit, no network."""
import socket
import threading
import time

import pytest
from fastapi.testclient import TestClient

import api
from agent import memory, netguard, tools

# Captured before any fixture swaps it out: netguard.socket IS this module.
REAL_GETADDRINFO = socket.getaddrinfo

FAKE_REPORT = {
    "url": "https://example.com",
    "overall_score": 82.0,
    "grade": "B",
    "summary": "Fine.",
    "categories": [],
    "review_status": "approved",
    "_specialist_reports": {"content": {"raw_evidence_notes": "internal"}},
    "_reflection_log": [{"round": 1}],
}


def _public_dns(host, *args, **kwargs):
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0))]


@pytest.fixture
def client(monkeypatch, tmp_db_path):
    """A client against a clean job store and an empty database, with the
    outbound guard on (as in the Docker image) and every hostname resolving
    to a public address."""
    api._jobs.clear()
    api._recent_starts.clear()
    monkeypatch.setenv("SEO_AGENT_BLOCK_PRIVATE_HOSTS", "1")
    monkeypatch.setattr(netguard.socket, "getaddrinfo", _public_dns)
    monkeypatch.setattr(api.config, "GROQ_API_KEYS", ["test-key"])
    monkeypatch.setattr(api, "run_full_audit", lambda url, **kwargs: dict(FAKE_REPORT, url=url))
    yield TestClient(api.app)
    # Let background job threads finish before wiping the store under them.
    deadline = time.time() + 5
    while time.time() < deadline and any(j["status"] == "running" for j in api._jobs.values()):
        time.sleep(0.01)
    api._jobs.clear()
    api._recent_starts.clear()


def wait_for_job(client, job_id, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/audit/{job_id}").json()
        if job["status"] != "running":
            return job
        time.sleep(0.01)
    raise AssertionError("job did not finish in time")


def start(client, url="example.com", **extra):
    return client.post("/api/audit", json={"url": url, **extra})


@pytest.fixture
def blocked_audits(monkeypatch):
    """Make every audit hang until release.set(), to hold jobs in 'running'."""
    release = threading.Event()

    def slow(url, **kwargs):
        release.wait(timeout=5)
        return dict(FAKE_REPORT, url=url)

    monkeypatch.setattr(api, "run_full_audit", slow)
    yield release
    release.set()


class TestAuditLifecycle:
    def test_start_then_poll_until_done(self, client):
        resp = start(client)
        assert resp.status_code == 200
        job = wait_for_job(client, resp.json()["job_id"])
        assert job["status"] == "done"
        assert job["url"] == "https://example.com"
        assert job["report"]["overall_score"] == 82.0
        assert job["finished_at"] >= job["started_at"]

    def test_internal_working_data_is_not_returned(self, client):
        job = wait_for_job(client, start(client).json()["job_id"])
        assert not [k for k in job["report"] if k.startswith("_")]

    def test_live_logs_are_exposed(self, client, monkeypatch):
        def chatty(url, log_fn=None, **kwargs):
            log_fn("Stage 1/4: Planning audit scope... (mode: auto)")
            log_fn("Stage 2/4: Dispatching 7 specialist agents")
            return dict(FAKE_REPORT)

        monkeypatch.setattr(api, "run_full_audit", chatty)
        job = wait_for_job(client, start(client).json()["job_id"])
        assert [m[:9] for m in job["logs"]] == ["Stage 1/4", "Stage 2/4"]

    def test_mode_and_competitor_are_passed_through(self, client, monkeypatch):
        seen = {}

        def capture(url, **kwargs):
            seen.update(url=url, **kwargs)
            return dict(FAKE_REPORT)

        monkeypatch.setattr(api, "run_full_audit", capture)
        wait_for_job(client, start(client, mode="quick", competitor_url="rival.com").json()["job_id"])
        assert seen["url"] == "https://example.com"
        assert seen["mode"] == "quick"
        assert seen["competitor_url"] == "https://rival.com"

    def test_blank_competitor_is_treated_as_none(self, client, monkeypatch):
        seen = {}
        monkeypatch.setattr(api, "run_full_audit", lambda url, **kw: seen.update(kw) or dict(FAKE_REPORT))
        wait_for_job(client, start(client, competitor_url="   ").json()["job_id"])
        assert seen["competitor_url"] is None

    def test_failed_audit_reports_error(self, client, monkeypatch):
        def boom(url, **kwargs):
            raise RuntimeError("All models exhausted")

        monkeypatch.setattr(api, "run_full_audit", boom)
        job = wait_for_job(client, start(client).json()["job_id"])
        assert job["status"] == "error"
        assert job["error_code"] == "internal"
        assert job["report"] is None

    def test_server_without_an_api_key_refuses_up_front(self, client, monkeypatch):
        monkeypatch.setattr(api.config, "GROQ_API_KEYS", [])
        resp = start(client)
        assert resp.status_code == 503
        assert api._jobs == {}
        assert api._recent_starts == {}

    def test_unknown_job_is_404(self, client):
        assert client.get("/api/audit/doesnotexist").status_code == 404

    def test_invalid_mode_is_rejected(self, client):
        assert start(client, mode="turbo").status_code == 422

    def test_tool_caches_are_emptied_once_nothing_is_running(self, client):
        tools._page_cache["https://example.com"] = "<html>stale</html>"
        tools._lighthouse_cache[("https://example.com", "mobile")] = {"stale": True}
        wait_for_job(client, start(client).json()["job_id"])
        assert tools._page_cache == {}
        assert tools._lighthouse_cache == {}


class TestUrlValidation:
    @pytest.mark.parametrize("url", ["notaurl", "   ", "ftp://example.com", "https://"])
    def test_non_addresses_are_rejected(self, client, url):
        resp = start(client, url=url)
        assert resp.status_code == 400
        assert api._jobs == {}

    def test_empty_url_is_rejected(self, client):
        assert start(client, url="").status_code == 422

    @pytest.mark.parametrize("url", [
        "http://127.0.0.1",
        "http://localhost",
        "http://169.254.169.254/latest/meta-data/",
        "http://10.0.0.5",
        "http://[::1]/",
    ])
    def test_internal_addresses_are_rejected(self, client, monkeypatch, url):
        monkeypatch.setattr(netguard.socket, "getaddrinfo", REAL_GETADDRINFO)
        resp = start(client, url=url)
        assert resp.status_code == 400
        assert api._jobs == {}

    def test_name_pointing_at_private_network_is_rejected(self, client, monkeypatch):
        monkeypatch.setattr(netguard.socket, "getaddrinfo",
                            lambda *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("192.168.1.10", 0))])
        resp = start(client, url="intranet.example.com")
        assert resp.status_code == 400
        assert "not a public internet address" in resp.json()["detail"]

    def test_non_web_port_is_rejected(self, client):
        assert start(client, url="example.com:5432").status_code == 400

    def test_internal_competitor_url_is_rejected(self, client, monkeypatch):
        monkeypatch.setattr(netguard.socket, "getaddrinfo", REAL_GETADDRINFO)
        resp = start(client, url="93.184.216.34", competitor_url="http://127.0.0.1")
        assert resp.status_code == 400
        assert resp.json()["detail"].startswith("Competitor: ")

    def test_internal_addresses_are_allowed_when_guard_is_off(self, client, monkeypatch):
        monkeypatch.delenv("SEO_AGENT_BLOCK_PRIVATE_HOSTS")
        assert start(client, url="http://127.0.0.1").status_code == 200


class TestLimits:
    def test_per_ip_hourly_limit(self, client):
        for i in range(api.RATE_LIMIT_PER_HOUR):
            resp = start(client, url=f"site{i}.example.com")
            assert resp.status_code == 200
            wait_for_job(client, resp.json()["job_id"])

        resp = start(client, url="onemore.example.com")
        assert resp.status_code == 429
        assert "per hour" in resp.json()["detail"]
        assert int(resp.headers["Retry-After"]) > 0

    def test_limit_resets_after_the_window(self, client):
        for i in range(api.RATE_LIMIT_PER_HOUR):
            wait_for_job(client, start(client, url=f"site{i}.example.com").json()["job_id"])
        long_ago = time.time() - api.RATE_LIMIT_WINDOW_SECONDS - 1
        for starts in api._recent_starts.values():
            for i in range(len(starts)):
                starts[i] = long_ago

        assert start(client, url="again.example.com").status_code == 200

    def test_rejected_requests_do_not_use_up_the_limit(self, client):
        for _ in range(api.RATE_LIMIT_PER_HOUR + 3):
            assert start(client, url="http://127.0.0.1:22").status_code == 400
        assert start(client).status_code == 200

    def test_limit_can_be_disabled(self, client, monkeypatch):
        monkeypatch.setattr(api, "RATE_LIMIT_PER_HOUR", 0)
        for i in range(8):
            resp = start(client, url=f"site{i}.example.com")
            assert resp.status_code == 200
            wait_for_job(client, resp.json()["job_id"])

    def test_concurrent_audit_cap(self, client, blocked_audits):
        for i in range(api.MAX_CONCURRENT_AUDITS):
            assert start(client, url=f"site{i}.example.com").status_code == 200

        resp = start(client, url="overflow.example.com")
        assert resp.status_code == 429
        assert "busy" in resp.json()["detail"]

        # A request turned away for being busy must not cost the client one
        # of its hourly audits.
        assert len(api._recent_starts["testclient"]) == api.MAX_CONCURRENT_AUDITS

    def test_same_audit_already_running_returns_the_same_job(self, client, blocked_audits):
        first = start(client).json()["job_id"]
        second = start(client).json()["job_id"]
        assert first == second
        assert len(api._jobs) == 1
        assert len(api._recent_starts["testclient"]) == 1

    def test_same_url_in_another_mode_is_a_separate_job(self, client, blocked_audits):
        first = start(client, mode="quick").json()["job_id"]
        second = start(client, mode="deep").json()["job_id"]
        assert first != second

    def test_finished_job_is_not_reused(self, client):
        first = start(client).json()["job_id"]
        wait_for_job(client, first)
        assert start(client).json()["job_id"] != first


class TestEviction:
    def test_old_finished_jobs_are_dropped_when_a_new_one_starts(self, client):
        old = start(client, url="old.example.com").json()["job_id"]
        wait_for_job(client, old)
        api._jobs[old]["finished_at"] = time.time() - api.JOB_TTL_SECONDS - 1

        new = start(client, url="new.example.com").json()["job_id"]
        assert client.get(f"/api/audit/{old}").status_code == 404
        assert client.get(f"/api/audit/{new}").status_code == 200

    def test_recent_finished_jobs_are_kept(self, client):
        first = start(client, url="first.example.com").json()["job_id"]
        wait_for_job(client, first)
        start(client, url="second.example.com")
        assert client.get(f"/api/audit/{first}").status_code == 200

    def test_running_jobs_are_never_dropped(self, client, blocked_audits):
        running = start(client, url="slow.example.com").json()["job_id"]
        api._jobs[running]["started_at"] = time.time() - api.JOB_TTL_SECONDS * 10
        start(client, url="other.example.com")
        assert client.get(f"/api/audit/{running}").json()["status"] == "running"


class TestOtherRoutes:
    def test_health(self, client):
        assert client.get("/api/health").json() == {"ok": True}

    def test_history_returns_stored_audits(self, client, tmp_db_path):
        memory.save_audit("https://example.com", {"overall_score": 70, "grade": "C"})
        memory.save_audit("https://example.com", {"overall_score": 80, "grade": "B"})
        body = client.get("/api/history/example.com").json()
        assert body["domain"] == "example.com"
        assert [row["overall_score"] for row in body["history"]] == [80, 70]

    def test_history_for_unknown_domain_is_empty(self, client, tmp_db_path):
        assert client.get("/api/history/never-seen.example.com").json()["history"] == []

    @pytest.mark.parametrize("limit", [0, 51, -1])
    def test_history_limit_is_bounded(self, client, tmp_db_path, limit):
        assert client.get(f"/api/history/example.com?limit={limit}").status_code == 422

    def test_page_is_served_at_root(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    def test_api_routes_are_not_shadowed_by_the_page(self, client):
        assert client.get("/api/health").headers["content-type"].startswith("application/json")


class TestVisitorSafeErrors:
    """The job's error used to be str(exception) and its logs the raw
    progress lines. Provider errors name the account's organization id and
    its billing page; neither belongs in front of a visitor."""

    GROQ_TEXT = ("[Synthesizer] hit a Groq rate/quota limit requiring a wait longer than 15 minutes. "
                 "Upgrade at https://console.groq.com/settings/billing.\n\nGroq's message: Rate limit "
                 "reached for model `openai/gpt-oss-120b` in organization `org_01abcXYZ` on tokens per day")

    def test_provider_error_text_never_reaches_the_visitor(self, client, monkeypatch):
        def boom(url, **kwargs):
            raise RuntimeError(self.GROQ_TEXT)

        monkeypatch.setattr(api, "run_full_audit", boom)
        job = wait_for_job(client, start(client).json()["job_id"])
        assert job["status"] == "error"
        assert job["error_code"] == "quota"
        assert "org_01abcXYZ" not in job["error"]
        assert "console.groq.com" not in job["error"]

    def test_unexpected_exception_becomes_a_generic_message(self, client, monkeypatch):
        def boom(url, **kwargs):
            raise KeyError("/app/agent/secret_path.py")

        monkeypatch.setattr(api, "run_full_audit", boom)
        job = wait_for_job(client, start(client).json()["job_id"])
        assert job["error_code"] == "internal"
        assert "secret_path" not in job["error"]

    def test_audit_failed_message_is_passed_through_with_its_code(self, client, monkeypatch):
        def blocked(url, **kwargs):
            raise api.AuditFailed("blocked", "This site blocked the scanner (it answered with HTTP 403).")

        monkeypatch.setattr(api, "run_full_audit", blocked)
        job = wait_for_job(client, start(client).json()["job_id"])
        assert job["error_code"] == "blocked"
        assert "403" in job["error"]

    def test_log_lines_are_redacted_and_bounded(self, client, monkeypatch):
        def chatty(url, log_fn=None, **kwargs):
            log_fn("  -> links specialist FAILED: " + self.GROQ_TEXT)
            log_fn("x" * 5000)
            return dict(FAKE_REPORT)

        monkeypatch.setattr(api, "run_full_audit", chatty)
        job = wait_for_job(client, start(client).json()["job_id"])
        assert "org_01abcXYZ" not in " ".join(job["logs"])
        assert "Groq's message" not in " ".join(job["logs"])
        assert job["logs"][0].startswith("  -> links specialist FAILED")
        assert len(job["logs"][1]) <= api.MAX_LOG_LINE_CHARS + 3

    def test_log_is_capped(self, client, monkeypatch):
        def flood(url, log_fn=None, **kwargs):
            for i in range(api.MAX_JOB_LOG_LINES + 50):
                log_fn(f"line {i}")
            return dict(FAKE_REPORT)

        monkeypatch.setattr(api, "run_full_audit", flood)
        job = wait_for_job(client, start(client).json()["job_id"])
        assert len(job["logs"]) == api.MAX_JOB_LOG_LINES

    def test_private_job_fields_are_not_returned(self, client):
        job = wait_for_job(client, start(client).json()["job_id"])
        assert "client_ip" not in job
        assert "counted_at" not in job


class TestRateLimitRefund:
    def _fail_with(self, monkeypatch, exc):
        def boom(url, **kwargs):
            raise exc
        monkeypatch.setattr(api, "run_full_audit", boom)

    def test_server_side_failure_gives_the_slot_back(self, client, monkeypatch):
        """A scan that died on our quota used to cost the visitor one of
        their hourly audits all the same."""
        self._fail_with(monkeypatch, RuntimeError("hit a Groq rate/quota limit"))
        for i in range(api.RATE_LIMIT_PER_HOUR + 3):
            resp = start(client, url=f"site{i}.example.com")
            assert resp.status_code == 200
            wait_for_job(client, resp.json()["job_id"])
        assert len(api._recent_starts.get("testclient", [])) == 0

    def test_timeout_gives_the_slot_back(self, client, monkeypatch):
        self._fail_with(monkeypatch, api.AuditFailed("timeout", "The scan ran out of time."))
        wait_for_job(client, start(client).json()["job_id"])
        assert len(api._recent_starts.get("testclient", [])) == 0

    def test_blocked_site_still_counts(self, client, monkeypatch):
        self._fail_with(monkeypatch, api.AuditFailed("blocked", "This site blocked the scanner."))
        wait_for_job(client, start(client).json()["job_id"])
        assert len(api._recent_starts["testclient"]) == 1

    def test_successful_audit_still_counts(self, client):
        wait_for_job(client, start(client).json()["job_id"])
        assert len(api._recent_starts["testclient"]) == 1


class TestStoredReports:
    """Jobs live in memory, so a finished report used to become a dead link
    after an hour or after any deploy -- although the report itself was
    sitting in SQLite."""

    def _saving_audit(self, monkeypatch):
        def audit(url, audit_id=None, mode="auto", competitor_url=None, **kwargs):
            report = dict(FAKE_REPORT, url=url, mode=mode, competitor_url=competitor_url, duration_seconds=12.5)
            memory.save_audit(url, report, public_id=audit_id)
            return report
        monkeypatch.setattr(api, "run_full_audit", audit)

    def test_job_id_is_passed_to_the_audit_as_its_public_id(self, client, monkeypatch):
        seen = {}
        monkeypatch.setattr(api, "run_full_audit", lambda url, **kw: seen.update(kw) or dict(FAKE_REPORT))
        job_id = start(client).json()["job_id"]
        wait_for_job(client, job_id)
        assert seen["audit_id"] == job_id
        assert seen["deadline_seconds"] == api.AUDIT_TIMEOUT_SECONDS

    def test_finished_report_survives_losing_the_in_memory_job(self, client, monkeypatch):
        self._saving_audit(monkeypatch)
        job_id = start(client, mode="quick").json()["job_id"]
        live = wait_for_job(client, job_id)
        api._jobs.clear()  # what a restart or the TTL does

        resp = client.get(f"/api/audit/{job_id}")
        assert resp.status_code == 200
        stored = resp.json()
        assert stored["status"] == "done"
        assert stored["url"] == "https://example.com"
        assert stored["mode"] == "quick"
        assert stored["report"]["overall_score"] == live["report"]["overall_score"]
        assert stored["finished_at"] - stored["started_at"] == pytest.approx(12.5)
        assert stored["logs"] == []
        assert not [k for k in stored["report"] if k.startswith("_")]

    def test_report_stored_before_this_feature_still_opens(self, client):
        memory.save_audit("https://old.example.com", {"overall_score": 61, "grade": "D"}, public_id="a" * 32)
        stored = client.get("/api/audit/" + "a" * 32).json()
        assert stored["url"] == "https://old.example.com"
        assert stored["mode"] is None and stored["started_at"] is None

    def test_history_rows_carry_the_public_id(self, client):
        memory.save_audit("https://example.com", {"overall_score": 70, "grade": "C"}, public_id="b" * 32)
        row = client.get("/api/history/example.com").json()["history"][0]
        assert row["public_id"] == "b" * 32

    def test_malformed_id_is_404_without_a_lookup(self, client, monkeypatch):
        monkeypatch.setattr(memory, "get_audit_by_public_id",
                            lambda _id: pytest.fail("looked up a malformed id"))
        assert client.get("/api/audit/1").status_code == 404
        assert client.get("/api/audit/' OR 1=1 --").status_code == 404


class TestPdf:
    REPORT = dict(
        FAKE_REPORT,
        summary="Missing <title> tag & no <h1>; see <img src='http://127.0.0.1/x'>.",
        categories=[{"name": "Technical SEO", "score": 70, "weight": 1.0, "findings": [
            {"severity": "critical", "issue": "The <title> element is missing.",
             "recommendation": "Add <title>Your page</title> inside <head>."}]}],
        quick_wins=["Add a <meta name=\"description\"> tag."],
        skipped_categories=[{"name": "Link Health", "reason": "The check failed to complete."}],
    )

    def test_pdf_of_a_finished_audit(self, client, monkeypatch):
        """Report text is model output about someone else's site and is full
        of angle brackets. Unescaped, reportlab parsed it as markup: the
        export failed on "<title>", and "<img src=...>" would have made the
        server fetch that address."""
        monkeypatch.setattr(api, "run_full_audit", lambda url, **kw: dict(self.REPORT))
        job_id = start(client).json()["job_id"]
        wait_for_job(client, job_id)

        resp = client.get(f"/api/audit/{job_id}/pdf")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/pdf"
        assert "seo-report-example.com.pdf" in resp.headers["content-disposition"]
        assert resp.content.startswith(b"%PDF")

    def test_no_pdf_for_an_unknown_or_unfinished_audit(self, client, blocked_audits):
        assert client.get("/api/audit/" + "c" * 32 + "/pdf").status_code == 404
        running = start(client).json()["job_id"]
        assert client.get(f"/api/audit/{running}/pdf").status_code == 404
