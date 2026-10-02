"""Tests for agent/tools.py. All network calls (requests.get/head, sockets)
are mocked -- these tests never touch the real network."""
from __future__ import annotations

import ssl
import socket
import types
from unittest.mock import MagicMock, patch

import pytest
import requests

from agent import tools


SAMPLE_HTML = """
<html>
<head>
  <title>Example Page Title</title>
  <meta name="description" content="A short description of the example page.">
  <link rel="canonical" href="https://example.com/">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <script type="application/ld+json">{"@type": "Organization", "name": "Example"}</script>
</head>
<body>
  <h1>Main Heading</h1>
  <h2>Sub heading</h2>
  <img src="/logo.png" alt="Logo">
  <img src="/banner.png">
  <a href="/about">About</a>
  <a href="https://external.com/page">External</a>
  <a href="#section">Anchor</a>
  <p>Some body text goes here for word count purposes.</p>
  <meta property="og:title" content="Example Page">
</body>
</html>
"""


def _fake_response(status_code=200, text="", headers=None, url="https://example.com"):
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.text = text
    resp.content = text.encode()
    resp.headers = headers or {"Content-Type": "text/html"}
    resp.url = url
    resp.history = []
    return resp


class TestNormalizeUrl:
    def test_adds_https_when_missing(self):
        assert tools.normalize_url("example.com") == "https://example.com"

    def test_leaves_existing_scheme(self):
        assert tools.normalize_url("http://example.com") == "http://example.com"
        assert tools.normalize_url("https://example.com") == "https://example.com"


class TestFetchPage:
    def setup_method(self):
        tools._page_cache.clear()

    def test_successful_fetch_caches_html_and_returns_metadata(self):
        resp = _fake_response(200, text="<html>hi</html>")
        with patch.object(tools.requests, "get", return_value=resp):
            result = tools.fetch_page("example.com")
        assert result["ok"] is True
        assert result["status_code"] == 200
        assert "likely_blocked" not in result
        assert tools._page_cache["https://example.com"] == "<html>hi</html>"

    def test_flags_likely_blocked_on_403(self):
        resp = _fake_response(403, text="Access Denied")
        with patch.object(tools.requests, "get", return_value=resp):
            result = tools.fetch_page("example.com")
        assert result["likely_blocked"] is True
        assert "block" in result["block_warning"].lower()

    def test_flags_likely_blocked_on_503_and_429_and_401(self):
        for code in (401, 429, 503):
            resp = _fake_response(code, text="blocked")
            with patch.object(tools.requests, "get", return_value=resp):
                result = tools.fetch_page("example.com")
            assert result["likely_blocked"] is True, f"status {code} should be flagged"

    def test_does_not_flag_normal_4xx_like_404(self):
        resp = _fake_response(404, text="not found")
        with patch.object(tools.requests, "get", return_value=resp):
            result = tools.fetch_page("example.com")
        assert "likely_blocked" not in result

    def test_network_error_returns_not_ok(self):
        with patch.object(tools.requests, "get", side_effect=requests.exceptions.ConnectionError("refused")):
            result = tools.fetch_page("example.com")
        assert result["ok"] is False
        assert "refused" in result["error"]


class TestParseSeoElements:
    def setup_method(self):
        tools._page_cache.clear()

    def test_extracts_all_signals_from_sample_html(self):
        tools._page_cache["https://example.com"] = SAMPLE_HTML
        result = tools.parse_seo_elements("example.com")
        assert result["title"] == "Example Page Title"
        assert result["meta_description"].startswith("A short description")
        assert result["canonical_url"] == "https://example.com/"
        assert result["has_viewport_meta"] is True
        assert result["heading_counts"]["h1"] == 1
        assert result["heading_counts"]["h2"] == 1
        assert result["h1_texts"] == ["Main Heading"]
        assert result["total_images"] == 2
        assert result["images_missing_alt_count"] == 1
        assert result["internal_link_count"] == 1
        assert result["external_link_count"] == 1
        assert "Organization" in result["structured_data_types_found"]
        assert result["open_graph_tags"]["og:title"] == "Example Page"
        assert result["word_count"] > 0

    def test_returns_error_when_html_not_cached_and_fetch_fails(self):
        with patch.object(tools.requests, "get", side_effect=requests.exceptions.ConnectionError("nope")):
            result = tools.parse_seo_elements("neverfetched.example.com")
        assert result["ok"] is False


class TestAnalyzeSecurityHeaders:
    def test_rejects_malformed_tool_call_shaped_input(self):
        result = tools.analyze_security_headers({"function": "foo", "args": {}})
        assert result["ok"] is False

    def test_rejects_dict_containing_url_key(self):
        result = tools.analyze_security_headers({"url": "https://example.com"})
        assert result["ok"] is False

    def test_rejects_empty_or_non_dict(self):
        assert tools.analyze_security_headers({})["ok"] is False
        assert tools.analyze_security_headers(None)["ok"] is False

    def test_identifies_present_and_missing_headers(self):
        headers = {
            "Strict-Transport-Security": "max-age=63072000",
            "Content-Type": "text/html",
        }
        result = tools.analyze_security_headers(headers)
        assert "strict-transport-security" in result["present_headers"]
        missing_names = {m["header"] for m in result["missing_headers"]}
        assert "content-security-policy" in missing_names
        assert "x-frame-options" in missing_names

    def test_case_insensitive_header_matching(self):
        headers = {"STRICT-TRANSPORT-SECURITY": "max-age=1"}
        result = tools.analyze_security_headers(headers)
        assert "strict-transport-security" in result["present_headers"]


class TestCheckLinksStatus:
    def test_flags_broken_links(self):
        ok_resp = _fake_response(200)
        broken_resp = _fake_response(404)
        with patch.object(tools.requests, "head", side_effect=[ok_resp, broken_resp]):
            result = tools.check_links_status(["https://example.com/a", "https://example.com/b"])
        assert result["checked"] == 2
        assert result["broken_count"] == 1

    def test_high_failure_rate_warning_triggers_at_threshold(self):
        broken = _fake_response(500)
        with patch.object(tools.requests, "head", return_value=broken):
            result = tools.check_links_status(["https://a.com", "https://b.com", "https://c.com"])
        assert result["broken_count"] == 3
        assert result["high_failure_rate_warning"] is not None
        assert "blocking" in result["high_failure_rate_warning"].lower()

    def test_no_warning_when_most_links_are_fine(self):
        ok = _fake_response(200)
        with patch.object(tools.requests, "head", return_value=ok):
            result = tools.check_links_status(["https://a.com", "https://b.com", "https://c.com"])
        assert result["high_failure_rate_warning"] is None

    def test_falls_back_to_get_when_head_is_4xx(self):
        head_resp = _fake_response(405)  # method not allowed
        get_resp = _fake_response(200)
        with patch.object(tools.requests, "head", return_value=head_resp), \
             patch.object(tools.requests, "get", return_value=get_resp):
            result = tools.check_links_status(["https://example.com/a"])
        assert result["results"][0]["status_code"] == 200
        assert result["results"][0]["broken"] is False

    def test_request_exception_counts_as_broken(self):
        with patch.object(tools.requests, "head", side_effect=requests.exceptions.Timeout("slow")):
            result = tools.check_links_status(["https://example.com/a"])
        assert result["results"][0]["broken"] is True
        assert result["results"][0]["status_code"] is None

    def test_caps_at_ten_urls(self):
        urls = [f"https://example.com/{i}" for i in range(15)]
        ok = _fake_response(200)
        with patch.object(tools.requests, "head", return_value=ok):
            result = tools.check_links_status(urls)
        assert result["checked"] == 10


class TestCheckSslCertificate:
    def _fake_ssl_context(self, not_after: str):
        cert = {
            "notAfter": not_after,
            "issuer": [[("organizationName", "Fake CA")]],
            "subject": [[("commonName", "example.com")]],
        }
        fake_ssock = MagicMock()
        fake_ssock.getpeercert.return_value = cert
        fake_ssock.__enter__.return_value = fake_ssock
        fake_ssock.__exit__.return_value = False

        fake_ctx = MagicMock()
        fake_ctx.wrap_socket.return_value = fake_ssock
        return fake_ctx

    def test_valid_unexpired_certificate(self):
        future = "Jan  1 00:00:00 2099 GMT"
        fake_ctx = self._fake_ssl_context(future)
        fake_sock = MagicMock()
        fake_sock.__enter__.return_value = fake_sock
        fake_sock.__exit__.return_value = False

        with patch.object(tools.ssl, "create_default_context", return_value=fake_ctx), \
             patch.object(tools.socket, "create_connection", return_value=fake_sock):
            result = tools.check_ssl_certificate("example.com")

        assert result["ok"] is True
        assert result["has_valid_ssl"] is True
        assert result["is_expired"] is False
        assert "VALID" in result["ssl_status_summary"]

    def test_expired_certificate(self):
        past = "Jan  1 00:00:00 2020 GMT"
        fake_ctx = self._fake_ssl_context(past)
        fake_sock = MagicMock()
        fake_sock.__enter__.return_value = fake_sock
        fake_sock.__exit__.return_value = False

        with patch.object(tools.ssl, "create_default_context", return_value=fake_ctx), \
             patch.object(tools.socket, "create_connection", return_value=fake_sock):
            result = tools.check_ssl_certificate("example.com")

        assert result["is_expired"] is True
        assert "EXPIRED" in result["ssl_status_summary"]

    def test_connection_failure_returns_has_valid_ssl_false(self):
        with patch.object(tools.socket, "create_connection", side_effect=OSError("refused")):
            result = tools.check_ssl_certificate("example.com")
        assert result["ok"] is True
        assert result["has_valid_ssl"] is False
        assert "refused" in result["error"]


class TestCheckCoreWebVitals:
    def setup_method(self):
        tools._lighthouse_cache.clear()

    LIGHTHOUSE_PAYLOAD = {
        "lighthouseResult": {
            "categories": {"performance": {"score": 0.42}},
            "audits": {
                "largest-contentful-paint": {"numericValue": 4200},
                "cumulative-layout-shift": {"numericValue": 0.25},
                "total-blocking-time": {"numericValue": 300},
                "speed-index": {"numericValue": 5000},
                "first-contentful-paint": {"numericValue": 1800},
                "interactive": {"numericValue": 6000},
            },
        },
        "loadingExperience": {},
    }

    def test_successful_response_parses_lab_data(self):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = self.LIGHTHOUSE_PAYLOAD
        with patch.object(tools.requests, "get", return_value=resp):
            result = tools.check_core_web_vitals("example.com")
        assert result["ok"] is True
        assert result["lab_data"]["performance_score_0_100"] == 42
        assert result["lab_data"]["lcp_ms"] == 4200
        assert result["field_data"] is None  # no loadingExperience.metrics in payload

    def test_retries_on_transient_500_then_succeeds(self):
        bad = MagicMock(status_code=500, text="server error")
        good = MagicMock(status_code=200)
        good.json.return_value = self.LIGHTHOUSE_PAYLOAD
        with patch.object(tools.requests, "get", side_effect=[bad, good]), \
             patch.object(tools.time, "sleep", return_value=None):
            result = tools.check_core_web_vitals("example.com")
        assert result["ok"] is True

    def test_gives_up_after_max_attempts(self):
        bad = MagicMock(status_code=500, text="server error")
        with patch.object(tools.requests, "get", return_value=bad), \
             patch.object(tools.time, "sleep", return_value=None):
            result = tools.check_core_web_vitals("example.com")
        assert result["ok"] is False
        assert "fall back" in result["note"].lower()

    def test_non_retryable_status_fails_immediately(self):
        bad = MagicMock(status_code=400, text="bad request")
        with patch.object(tools.requests, "get", return_value=bad) as mock_get:
            result = tools.check_core_web_vitals("example.com")
        assert result["ok"] is False
        assert mock_get.call_count == 1  # no retry on a non-5xx failure

    def test_second_call_for_same_url_uses_cache_not_a_new_request(self):
        resp = MagicMock(status_code=200)
        resp.json.return_value = self.LIGHTHOUSE_PAYLOAD
        with patch.object(tools.requests, "get", return_value=resp) as mock_get:
            tools.check_core_web_vitals("example.com")
            tools.check_core_web_vitals("example.com")
        assert mock_get.call_count == 1

    def test_different_strategy_is_not_cached_together(self):
        resp = MagicMock(status_code=200)
        resp.json.return_value = self.LIGHTHOUSE_PAYLOAD
        with patch.object(tools.requests, "get", return_value=resp) as mock_get:
            tools.check_core_web_vitals("example.com", strategy="mobile")
            tools.check_core_web_vitals("example.com", strategy="desktop")
        assert mock_get.call_count == 2

    def test_failed_call_is_not_cached_so_a_later_call_retries(self):
        bad = MagicMock(status_code=500, text="server error")
        good = MagicMock(status_code=200)
        good.json.return_value = self.LIGHTHOUSE_PAYLOAD
        with patch.object(tools.time, "sleep", return_value=None):
            with patch.object(tools.requests, "get", return_value=bad):
                first = tools.check_core_web_vitals("example.com")
            assert first["ok"] is False
            with patch.object(tools.requests, "get", return_value=good):
                second = tools.check_core_web_vitals("example.com")
            assert second["ok"] is True


class TestCheckAccessibilityAndBestPractices:
    def setup_method(self):
        tools._lighthouse_cache.clear()

    LIGHTHOUSE_PAYLOAD = {
        "lighthouseResult": {
            "categories": {
                "accessibility": {
                    "score": 0.71,
                    "auditRefs": [
                        {"id": "color-contrast"},
                        {"id": "image-alt"},
                        {"id": "label"},
                        {"id": "document-title"},
                    ],
                },
            },
            "audits": {
                "color-contrast": {"score": 0, "title": "Background/foreground colors have insufficient contrast", "description": "Low-contrast text is hard to read for many users."},
                "image-alt": {"score": 0, "title": "Image elements do not have [alt] attributes", "description": "Screen readers rely on alt text."},
                "label": {"score": 1, "title": "Form elements have associated labels", "description": "Labels help assistive tech."},
                "document-title": {"score": None, "title": "Document has a <title> element", "description": "Informative, not scored."},
            },
        },
    }

    def test_extracts_score_and_failing_audits_only(self):
        resp = MagicMock(status_code=200)
        resp.json.return_value = self.LIGHTHOUSE_PAYLOAD
        with patch.object(tools.requests, "get", return_value=resp):
            result = tools.check_accessibility_and_best_practices("example.com")
        assert result["ok"] is True
        assert result["accessibility_score_0_100"] == 71
        failing_ids = {a["id"] for a in result["failing_accessibility_audits"]}
        assert failing_ids == {"color-contrast", "image-alt"}  # score==1 and score==None excluded

    def test_caps_failing_audits_at_twelve(self):
        many_refs = [{"id": f"check-{i}"} for i in range(20)]
        many_audits = {f"check-{i}": {"score": 0, "title": f"Check {i}", "description": ""} for i in range(20)}
        payload = {"lighthouseResult": {"categories": {"accessibility": {"score": 0.1, "auditRefs": many_refs}}, "audits": many_audits}}
        resp = MagicMock(status_code=200)
        resp.json.return_value = payload
        with patch.object(tools.requests, "get", return_value=resp):
            result = tools.check_accessibility_and_best_practices("example.com")
        assert len(result["failing_accessibility_audits"]) == 12

    def test_ok_false_on_api_failure(self):
        bad = MagicMock(status_code=500, text="server error")
        with patch.object(tools.requests, "get", return_value=bad), \
             patch.object(tools.time, "sleep", return_value=None):
            result = tools.check_accessibility_and_best_practices("example.com")
        assert result["ok"] is False

    def test_missing_accessibility_category_gives_none_score_not_crash(self):
        payload = {"lighthouseResult": {"categories": {}, "audits": {}}}
        resp = MagicMock(status_code=200)
        resp.json.return_value = payload
        with patch.object(tools.requests, "get", return_value=resp):
            result = tools.check_accessibility_and_best_practices("example.com")
        assert result["ok"] is True
        assert result["accessibility_score_0_100"] is None
        assert result["failing_accessibility_audits"] == []

    def test_shares_cache_with_check_core_web_vitals(self):
        """The whole point of sharing the cache: calling both tools for the
        same url+strategy should cost exactly one PageSpeed API request."""
        payload = {
            "lighthouseResult": {
                "categories": {
                    "performance": {"score": 0.9},
                    "accessibility": {"score": 0.8, "auditRefs": []},
                },
                "audits": {},
            },
            "loadingExperience": {},
        }
        resp = MagicMock(status_code=200)
        resp.json.return_value = payload
        with patch.object(tools.requests, "get", return_value=resp) as mock_get:
            perf = tools.check_core_web_vitals("example.com")
            a11y = tools.check_accessibility_and_best_practices("example.com")
        assert mock_get.call_count == 1
        assert perf["ok"] is True
        assert a11y["ok"] is True
        assert a11y["accessibility_score_0_100"] == 80


class TestFetchRobotsAndSitemap:
    def test_fetch_robots_txt_detects_sitemap_reference(self):
        resp = _fake_response(200, text="User-agent: *\nSitemap: https://example.com/sitemap.xml")
        with patch.object(tools.requests, "get", return_value=resp):
            result = tools.fetch_robots_txt("example.com")
        assert result["exists"] is True
        assert result["mentions_sitemap"] is True

    def test_fetch_robots_txt_missing(self):
        resp = _fake_response(404, text="")
        with patch.object(tools.requests, "get", return_value=resp):
            result = tools.fetch_robots_txt("example.com")
        assert result["exists"] is False

    def test_fetch_sitemap_found(self):
        xml = '<?xml version="1.0"?><urlset><url><loc>a</loc></url><url><loc>b</loc></url></urlset>'
        resp = _fake_response(200, text=xml, headers={"Content-Type": "application/xml"})
        with patch.object(tools.requests, "get", return_value=resp):
            result = tools.fetch_sitemap("example.com")
        assert result["exists"] is True
        assert result["entry_count_estimate"] == 2

    def test_fetch_sitemap_not_found(self):
        resp = _fake_response(404, text="")
        with patch.object(tools.requests, "get", return_value=resp):
            result = tools.fetch_sitemap("example.com")
        assert result["exists"] is False


class TestCheckBestPractices:
    def setup_method(self):
        tools._lighthouse_cache.clear()

    LIGHTHOUSE_PAYLOAD = {
        "lighthouseResult": {
            "categories": {
                "best-practices": {
                    "score": 0.67,
                    "auditRefs": [
                        {"id": "is-on-https"},
                        {"id": "no-vulnerable-libraries"},
                        {"id": "deprecations"},
                        {"id": "charset"},
                    ],
                },
            },
            "audits": {
                "is-on-https": {"score": 0, "title": "Does not use HTTPS", "description": "..."},
                "no-vulnerable-libraries": {"score": 0, "title": "Includes front-end JavaScript libraries with known security vulnerabilities", "description": "..."},
                "deprecations": {"score": 0, "title": "Uses deprecated APIs", "description": "..."},
                "charset": {"score": 1, "title": "Properly defines charset", "description": "..."},
            },
        },
    }

    def test_extracts_score_and_failing_audits(self):
        resp = MagicMock(status_code=200)
        resp.json.return_value = self.LIGHTHOUSE_PAYLOAD
        with patch.object(tools.requests, "get", return_value=resp):
            result = tools.check_best_practices("example.com")
        assert result["ok"] is True
        assert result["best_practices_score_0_100"] == 67
        ids = {a["id"] for a in result["failing_best_practices_audits"]}
        assert ids == {"no-vulnerable-libraries", "deprecations"}  # is-on-https excluded, charset passed

    def test_excludes_https_related_audits_even_when_failing(self):
        """is-on-https is Security's domain (dedicated SSL check), not
        Best Practices' -- must never appear here, even if Lighthouse
        itself reports it as a failure, to avoid contradicting the
        Security specialist's more authoritative SSL check."""
        resp = MagicMock(status_code=200)
        resp.json.return_value = self.LIGHTHOUSE_PAYLOAD
        with patch.object(tools.requests, "get", return_value=resp):
            result = tools.check_best_practices("example.com")
        ids = {a["id"] for a in result["failing_best_practices_audits"]}
        assert "is-on-https" not in ids
        assert "redirects-http" not in ids

    def test_ok_false_on_api_failure(self):
        bad = MagicMock(status_code=500, text="server error")
        with patch.object(tools.requests, "get", return_value=bad), \
             patch.object(tools.time, "sleep", return_value=None):
            result = tools.check_best_practices("example.com")
        assert result["ok"] is False

    def test_shares_cache_with_accessibility_and_cwv(self):
        """All three Lighthouse-backed tools should share one API call for
        the same url+strategy."""
        payload = {
            "lighthouseResult": {
                "categories": {
                    "performance": {"score": 0.9},
                    "accessibility": {"score": 0.8, "auditRefs": []},
                    "best-practices": {"score": 0.7, "auditRefs": []},
                },
                "audits": {},
            },
            "loadingExperience": {},
        }
        resp = MagicMock(status_code=200)
        resp.json.return_value = payload
        with patch.object(tools.requests, "get", return_value=resp) as mock_get:
            tools.check_core_web_vitals("example.com")
            tools.check_accessibility_and_best_practices("example.com")
            tools.check_best_practices("example.com")
        assert mock_get.call_count == 1

    def test_caps_failing_audits_at_twelve(self):
        many_refs = [{"id": f"check-{i}"} for i in range(20)]
        many_audits = {f"check-{i}": {"score": 0, "title": f"Check {i}", "description": ""} for i in range(20)}
        payload = {"lighthouseResult": {"categories": {"best-practices": {"score": 0.1, "auditRefs": many_refs}}, "audits": many_audits}}
        resp = MagicMock(status_code=200)
        resp.json.return_value = payload
        with patch.object(tools.requests, "get", return_value=resp):
            result = tools.check_best_practices("example.com")
        assert len(result["failing_best_practices_audits"]) == 12

class TestOutboundGuard:
    """agent/netguard.py wired into the tools. Off unless the env var is set,
    which is why every other test in this file runs unguarded."""

    def setup_method(self):
        tools._page_cache.clear()

    @pytest.fixture
    def guard_on(self, monkeypatch):
        monkeypatch.setenv("SEO_AGENT_BLOCK_PRIVATE_HOSTS", "1")

    @pytest.fixture
    def public_dns(self, monkeypatch):
        monkeypatch.setattr(
            tools.netguard.socket, "getaddrinfo",
            lambda *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0))],
        )

    def test_guard_is_off_by_default(self, monkeypatch):
        monkeypatch.delenv("SEO_AGENT_BLOCK_PRIVATE_HOSTS", raising=False)
        resp = _fake_response(200, text="<html>dev</html>", url="http://127.0.0.1")
        with patch.object(tools.requests, "get", return_value=resp) as mock_get:
            result = tools.fetch_page("http://127.0.0.1")
        assert result["ok"] is True
        assert "hooks" not in mock_get.call_args.kwargs

    def test_fetch_page_refuses_internal_address_without_requesting_it(self, guard_on):
        with patch.object(tools.requests, "get") as mock_get:
            result = tools.fetch_page("http://169.254.169.254/latest/meta-data/")
        assert result["ok"] is False
        assert "not a public internet address" in result["error"]
        mock_get.assert_not_called()
        assert tools._page_cache == {}

    def test_parse_seo_elements_cannot_reach_internal_address_either(self, guard_on):
        with patch.object(tools.requests, "get") as mock_get:
            result = tools.parse_seo_elements("http://127.0.0.1")
        assert result["ok"] is False
        mock_get.assert_not_called()

    def test_public_address_is_fetched_with_redirect_check_attached(self, guard_on, public_dns):
        resp = _fake_response(200, text="<html>hi</html>")
        with patch.object(tools.requests, "get", return_value=resp) as mock_get:
            result = tools.fetch_page("example.com")
        assert result["ok"] is True
        assert mock_get.call_args.kwargs["hooks"] == {"response": tools._refuse_private_redirect}

    def test_redirect_to_internal_address_is_refused(self, guard_on):
        hop = MagicMock(is_redirect=True, url="https://example.com/go", headers={"location": "http://127.0.0.1/admin"})
        with pytest.raises(requests.exceptions.InvalidURL, match="Redirect refused"):
            tools._refuse_private_redirect(hop)

    def test_relative_redirect_is_resolved_against_the_current_url(self, guard_on, public_dns):
        hop = MagicMock(is_redirect=True, url="https://example.com/a", headers={"location": "/b"})
        assert tools._refuse_private_redirect(hop) is None

    def test_non_redirect_response_passes_the_hook(self, guard_on):
        assert tools._refuse_private_redirect(MagicMock(is_redirect=False)) is None

    def test_refused_redirect_surfaces_as_a_normal_fetch_error(self, guard_on, public_dns):
        refused = requests.exceptions.InvalidURL("Redirect refused: Host '127.0.0.1' is not a public internet address.")
        with patch.object(tools.requests, "get", side_effect=refused):
            result = tools.fetch_page("example.com")
        assert result["ok"] is False
        assert "Redirect refused" in result["error"]

    def test_robots_and_sitemap_refuse_internal_address(self, guard_on):
        with patch.object(tools.requests, "get") as mock_get:
            robots = tools.fetch_robots_txt("http://10.0.0.5")
            sitemap = tools.fetch_sitemap("http://10.0.0.5")
        assert robots["ok"] is False
        assert sitemap["ok"] is False and sitemap["exists"] is False
        mock_get.assert_not_called()

    def test_ssl_check_refuses_internal_address_without_connecting(self, guard_on):
        with patch.object(tools.socket, "create_connection") as mock_connect:
            result = tools.check_ssl_certificate("https://192.168.1.1")
        assert result["ok"] is False
        mock_connect.assert_not_called()

    def test_internal_links_are_skipped_not_reported_as_broken(self, guard_on):
        with patch.object(tools.requests, "head") as mock_head:
            result = tools.check_links_status(["http://127.0.0.1/a", "http://10.0.0.5/b"])
        mock_head.assert_not_called()
        assert result["broken_count"] == 0
        assert all(r["broken"] is False and "skipped" in r for r in result["results"])

    def test_public_links_are_still_checked(self, guard_on, public_dns):
        with patch.object(tools.requests, "head", return_value=_fake_response(404)), \
             patch.object(tools.requests, "get", return_value=_fake_response(404)):
            result = tools.check_links_status(["https://example.com/gone"])
        assert result["broken_count"] == 1


class TestClearCaches:
    def test_empties_both_caches(self):
        tools._page_cache["https://example.com"] = "<html></html>"
        tools._lighthouse_cache[("https://example.com", "mobile")] = {"x": 1}
        tools.clear_caches()
        assert tools._page_cache == {}
        assert tools._lighthouse_cache == {}
