"""Shared fixtures for the test suite.

Groq's client is never actually called in these tests -- ToolAgent.client.chat.
completions.create is monkeypatched with fakes built here, so the suite runs
with zero network access and zero API keys.
"""
from __future__ import annotations

import types
import httpx
import pytest
import groq


# --------------------------------------------------------------------------
# Fake Groq wire objects
# --------------------------------------------------------------------------

class FakeFunction:
    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments


class FakeToolCall:
    def __init__(self, id: str, name: str, arguments: str):
        self.id = id
        self.type = "function"
        self.function = FakeFunction(name, arguments)


class FakeMessage:
    def __init__(self, content: str | None = "", tool_calls: list | None = None):
        self.content = content
        self.tool_calls = tool_calls or []


class FakeChoice:
    def __init__(self, message: FakeMessage):
        self.message = message


class FakeCompletion:
    def __init__(self, message: FakeMessage):
        self.choices = [FakeChoice(message)]


def make_completion(content: str | None = "", tool_calls: list | None = None) -> FakeCompletion:
    """Build a fake chat-completion response with a plain text (or tool-call) reply."""
    return FakeCompletion(FakeMessage(content=content, tool_calls=tool_calls))


def make_tool_call_completion(name: str, arguments: str, call_id: str = "call_1") -> FakeCompletion:
    return make_completion(content="", tool_calls=[FakeToolCall(call_id, name, arguments)])


def make_groq_status_error(status_code: int, message: str) -> groq.APIStatusError:
    """Build a real groq.APIStatusError (or RateLimitError for 429) with a
    given status code and message, the way the real SDK would raise it."""
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(status_code, request=request, text=message)
    if status_code == 429:
        return groq.RateLimitError(message, response=response, body=None)
    return groq.APIStatusError(message, response=response, body=None)


@pytest.fixture
def fake_completion_factory():
    return make_completion


@pytest.fixture
def fake_tool_call_completion_factory():
    return make_tool_call_completion


@pytest.fixture
def fake_groq_error_factory():
    return make_groq_status_error


# --------------------------------------------------------------------------
# Sample domain data reused across postprocess / orchestrator / schema tests
# --------------------------------------------------------------------------

@pytest.fixture
def sample_finding_good():
    return {"severity": "good", "issue": "Everything checks out.", "recommendation": "No action needed."}


@pytest.fixture
def sample_ssl_tool_call_log_valid():
    return [{
        "name": "check_ssl_certificate",
        "args": {"domain_or_url": "example.com"},
        "result": {
            "ok": True,
            "has_valid_ssl": True,
            "is_expired": False,
            "days_until_expiry": 60,
            "ssl_status_summary": "Certificate is VALID and NOT expired (60 day(s) remaining until it expires on Oct 1 2026).",
        },
    }]


@pytest.fixture
def sample_ssl_tool_call_log_expired():
    return [{
        "name": "check_ssl_certificate",
        "args": {"domain_or_url": "example.com"},
        "result": {
            "ok": True,
            "has_valid_ssl": True,
            "is_expired": True,
            "days_until_expiry": -10,
            "ssl_status_summary": "CERTIFICATE IS EXPIRED (expired 10 day(s) ago).",
        },
    }]


@pytest.fixture
def sample_report():
    return {
        "url": "https://example.com",
        "overall_score": 82.0,
        "grade": "B",
        "summary": "The site is in decent shape overall.",
        "categories": [
            {"name": "Technical SEO", "score": 90.0, "weight": 0.3, "findings": []},
            {"name": "On-Page Content", "score": 74.0, "weight": 0.3, "findings": []},
            {"name": "Page Speed", "score": 80.0, "weight": 0.2, "findings": []},
            {"name": "Web Security", "score": 85.0, "weight": 0.1, "findings": []},
            {"name": "Link Health", "score": 88.0, "weight": 0.1, "findings": []},
        ],
        "quick_wins": ["Add missing alt text to images."],
        "data_limitations": "Link checking was a small sample, not a full crawl.",
    }


@pytest.fixture
def tmp_db_path(tmp_path, monkeypatch):
    """Point agent.memory at an isolated, temporary SQLite file for this test."""
    from agent import memory as memory_module
    db_path = tmp_path / "audit_history.db"
    monkeypatch.setattr(memory_module, "DB_PATH", str(db_path))
    return str(db_path)


@pytest.fixture(autouse=True)
def _no_background_lighthouse(monkeypatch):
    """run_full_audit warms the Lighthouse cache in a background thread.
    Nothing in the suite may reach the network, so that is a no-op here;
    tests of the prefetch itself call agent.tools._get_lighthouse_data."""
    monkeypatch.setattr("agent.tools.prefetch_lighthouse", lambda *a, **k: None)
