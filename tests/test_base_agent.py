"""Tests for agent/base_agent.py.

The Groq client itself is always mocked -- ToolAgent.client.chat.completions.create
is monkeypatched per-test, so nothing here makes a real API call or needs a key.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from agent import base_agent
from agent.base_agent import ToolAgent, _extract_json, _seconds_to_wait, _shrink_ratio_from_error, _shrink_largest_message


# --------------------------------------------------------------------------
# Pure helper functions
# --------------------------------------------------------------------------

class TestExtractJson:
    def test_plain_json(self):
        assert _extract_json('{"a": 1}') == {"a": 1}

    def test_strips_markdown_fences(self):
        assert _extract_json('```json\n{"a": 1}\n```') == {"a": 1}

    def test_strips_bare_fences(self):
        assert _extract_json('```\n{"a": 1}\n```') == {"a": 1}

    def test_strips_surrounding_whitespace(self):
        assert _extract_json('   {"a": 1}   ') == {"a": 1}


class TestSecondsToWait:
    def test_parses_plain_seconds(self, fake_groq_error_factory):
        err = fake_groq_error_factory(429, "rate limited, please try again in 11.065s")
        wait = _seconds_to_wait(err, attempt=0)
        assert wait == pytest.approx(12.065, abs=0.01)

    def test_parses_minutes_and_seconds(self, fake_groq_error_factory):
        err = fake_groq_error_factory(429, "please try again in 6m53.856s")
        wait = _seconds_to_wait(err, attempt=0)
        assert wait == pytest.approx(6 * 60 + 53.856 + 1.0, abs=0.01)

    def test_parses_hours_minutes_seconds(self, fake_groq_error_factory, monkeypatch):
        # This wait (~64 minutes) exceeds the real MAX_AUTO_WAIT_SECONDS (15 min),
        # so raise the cap just for this test to isolate the *parsing* behavior
        # from the "give up on long waits" behavior (covered separately below).
        monkeypatch.setattr(base_agent, "MAX_AUTO_WAIT_SECONDS", 100000)
        err = fake_groq_error_factory(429, "please try again in 1h4m12.576s")
        wait = _seconds_to_wait(err, attempt=0)
        assert wait == pytest.approx(3600 + 4 * 60 + 12.576 + 1.0, abs=0.01)

    def test_returns_none_when_wait_exceeds_max(self, fake_groq_error_factory):
        err = fake_groq_error_factory(429, "please try again in 2h0m0s")
        assert _seconds_to_wait(err, attempt=0) is None

    def test_falls_back_to_exponential_backoff_when_unparseable(self, fake_groq_error_factory):
        err = fake_groq_error_factory(429, "rate limited")
        assert _seconds_to_wait(err, attempt=0) == 1
        assert _seconds_to_wait(err, attempt=3) == 8

    def test_backoff_capped_at_30(self, fake_groq_error_factory):
        err = fake_groq_error_factory(429, "rate limited")
        assert _seconds_to_wait(err, attempt=10) == 30


class TestShrinkRatioFromError:
    def test_computes_ratio_from_limit_and_requested(self, fake_groq_error_factory):
        err = fake_groq_error_factory(413, "Limit 6000, but Requested 12000 tokens")
        ratio = _shrink_ratio_from_error(err)
        assert 0.3 <= ratio <= 0.85

    def test_default_ratio_when_unparseable(self, fake_groq_error_factory):
        err = fake_groq_error_factory(413, "request too large")
        assert _shrink_ratio_from_error(err) == 0.6


class TestShrinkLargestMessage:
    def test_truncates_the_largest_message(self):
        messages = [
            {"role": "system", "content": "short"},
            {"role": "user", "content": "x" * 1000},
        ]
        changed = _shrink_largest_message(messages)
        assert changed is True
        assert len(messages[1]["content"]) < 1000
        assert "truncated" in messages[1]["content"]

    def test_returns_false_when_nothing_worth_shrinking(self):
        messages = [{"role": "system", "content": "short"}]
        assert _shrink_largest_message(messages) is False

    def test_ignores_non_string_content(self):
        messages = [{"role": "assistant", "content": None, "tool_calls": []}]
        assert _shrink_largest_message(messages) is False


# --------------------------------------------------------------------------
# ToolAgent.run() -- text/JSON path (no tools)
# --------------------------------------------------------------------------

class TestToolAgentRunTextOnly:
    def _agent(self, log_fn=None):
        return ToolAgent(name="Test", system_prompt="sys", model="fake-model", fallback_model=None, log_fn=log_fn)

    def test_returns_parsed_json_when_no_tool_calls(self, fake_completion_factory):
        agent = self._agent()
        agent._client = MagicMock()
        agent._client.chat.completions.create.return_value = fake_completion_factory('{"result": "ok"}')
        result = agent.run("do the thing")
        assert result == {"result": "ok"}

    def test_returns_raw_text_when_expect_json_false(self, fake_completion_factory):
        agent = self._agent()
        agent._client = MagicMock()
        agent._client.chat.completions.create.return_value = fake_completion_factory("plain text reply")
        result = agent.run("do the thing", expect_json=False)
        assert result == "plain text reply"

    def test_repairs_once_on_invalid_json_then_succeeds(self, fake_completion_factory):
        agent = self._agent()
        agent._client = MagicMock()
        agent._client.chat.completions.create.side_effect = [
            fake_completion_factory("not valid json{{{"),
            fake_completion_factory('{"result": "fixed"}'),
        ]
        result = agent.run("do the thing")
        assert result == {"result": "fixed"}
        assert agent._client.chat.completions.create.call_count == 2

    def test_raises_after_one_failed_repair_attempt(self, fake_completion_factory):
        agent = self._agent()
        agent._client = MagicMock()
        agent._client.chat.completions.create.side_effect = [
            fake_completion_factory("still not json"),
            fake_completion_factory("still not json again"),
        ]
        with pytest.raises(RuntimeError):
            agent.run("do the thing")


# --------------------------------------------------------------------------
# ToolAgent.run() -- tool-calling path
# --------------------------------------------------------------------------

class TestToolAgentRunWithTools:
    def test_executes_tool_and_feeds_result_back(self, fake_tool_call_completion_factory, fake_completion_factory, monkeypatch):
        called_with = {}

        def fake_fetch_page(args):
            called_with.update(args)
            return {"ok": True, "status_code": 200}

        monkeypatch.setitem(base_agent.TOOL_IMPL, "fetch_page", fake_fetch_page)

        agent = ToolAgent(
            name="Test", system_prompt="sys",
            client_tools=[{"type": "function", "function": {"name": "fetch_page"}}],
            model="fake-model", fallback_model=None,
        )
        agent._client = MagicMock()
        agent._client.chat.completions.create.side_effect = [
            fake_tool_call_completion_factory("fetch_page", json.dumps({"url": "example.com"})),
            fake_completion_factory('{"done": true}'),
        ]
        result = agent.run("audit example.com")
        assert result == {"done": True}
        assert called_with == {"url": "example.com"}

    def test_tool_execution_error_is_captured_not_raised(self, fake_tool_call_completion_factory, fake_completion_factory, monkeypatch):
        def failing_tool(args):
            raise ValueError("boom")

        monkeypatch.setitem(base_agent.TOOL_IMPL, "fetch_page", failing_tool)

        agent = ToolAgent(name="Test", system_prompt="sys",
                           client_tools=[{"type": "function", "function": {"name": "fetch_page"}}],
                           model="fake-model", fallback_model=None)
        agent._client = MagicMock()
        agent._client.chat.completions.create.side_effect = [
            fake_tool_call_completion_factory("fetch_page", json.dumps({"url": "x"})),
            fake_completion_factory('{"done": true}'),
        ]
        result = agent.run("audit x")
        assert result == {"done": True}
        # tool_call_log should record the captured failure, not blow up the run
        assert agent.tool_call_log[0]["result"]["ok"] is False

    def test_exceeds_max_iterations_raises(self, fake_tool_call_completion_factory, monkeypatch):
        monkeypatch.setitem(base_agent.TOOL_IMPL, "fetch_page", lambda args: {"ok": True})
        agent = ToolAgent(name="Test", system_prompt="sys",
                           client_tools=[{"type": "function", "function": {"name": "fetch_page"}}],
                           model="fake-model", fallback_model=None, max_iterations=2)
        agent._client = MagicMock()
        agent._client.chat.completions.create.return_value = fake_tool_call_completion_factory(
            "fetch_page", json.dumps({"url": "x"})
        )
        with pytest.raises(RuntimeError, match="exceeded"):
            agent.run("audit x")


# --------------------------------------------------------------------------
# _call_with_retry -- rate limit / fallback-model / key-rotation / shrink logic
# --------------------------------------------------------------------------

class TestCallWithRetry:
    def _agent(self, fallback_model="fallback-model", starting_key_index=0):
        return ToolAgent(name="Test", system_prompt="sys", model="primary-model",
                          fallback_model=fallback_model, starting_key_index=starting_key_index)

    def test_falls_back_to_fallback_model_on_429_without_waiting(self, fake_groq_error_factory, fake_completion_factory, monkeypatch):
        agent = self._agent()
        agent._client = MagicMock()
        rate_err = fake_groq_error_factory(429, "rate limited, please try again in 5s")
        agent._client.chat.completions.create.side_effect = [rate_err, fake_completion_factory("ok")]

        sleep_calls = []
        monkeypatch.setattr(base_agent.time, "sleep", lambda s: sleep_calls.append(s))

        result = agent._call_with_retry(model="primary-model", messages=[])
        assert result.choices[0].message.content == "ok"
        assert agent.model == "fallback-model"
        assert sleep_calls == []  # instant fallback, no sleep

    def test_shrinks_payload_on_413(self, fake_groq_error_factory, fake_completion_factory):
        agent = self._agent(fallback_model=None)
        agent._client = MagicMock()
        too_large_err = fake_groq_error_factory(413, "Payload too large. Limit 6000. Requested 12000.")
        agent._client.chat.completions.create.side_effect = [too_large_err, fake_completion_factory("ok")]

        messages = [{"role": "user", "content": "x" * 2000}]
        result = agent._call_with_retry(model="primary-model", messages=messages)
        assert result.choices[0].message.content == "ok"
        assert len(messages[0]["content"]) < 2000

    def test_gives_up_after_max_shrink_attempts(self, fake_groq_error_factory):
        agent = self._agent(fallback_model=None)
        agent._client = MagicMock()
        too_large_err = fake_groq_error_factory(413, "reduce your message size")
        agent._client.chat.completions.create.side_effect = too_large_err

        messages = [{"role": "user", "content": "x" * 5000}]
        with pytest.raises(RuntimeError, match="too large"):
            agent._call_with_retry(model="primary-model", messages=messages)

    def test_rotates_api_key_when_wait_too_long_and_fallback_exhausted(self, fake_groq_error_factory, fake_completion_factory, monkeypatch):
        monkeypatch.setattr(base_agent, "GROQ_API_KEYS", ["key1", "key2"])
        # _rotate_api_key() clears self._client so the next call rebuilds a
        # real Groq() client via the `client` property -- mock the Groq
        # constructor itself so that rebuild never touches the network.
        mock_client = MagicMock()
        monkeypatch.setattr(base_agent, "Groq", MagicMock(return_value=mock_client))

        agent = self._agent(fallback_model=None, starting_key_index=0)
        agent._client = MagicMock()  # first call uses this pre-set mock
        long_wait_err = fake_groq_error_factory(429, "please try again in 2h0m0s")
        agent._client.chat.completions.create.side_effect = [long_wait_err]
        mock_client.chat.completions.create.side_effect = [fake_completion_factory("ok")]

        result = agent._call_with_retry(model="primary-model", messages=[])
        assert result.choices[0].message.content == "ok"
        assert agent._key_index == 1

    def test_raises_clear_error_when_no_more_keys_and_wait_too_long(self, fake_groq_error_factory, monkeypatch):
        monkeypatch.setattr(base_agent, "GROQ_API_KEYS", ["key1"])
        agent = self._agent(fallback_model=None, starting_key_index=0)
        agent._client = MagicMock()
        long_wait_err = fake_groq_error_factory(429, "please try again in 2h0m0s")
        agent._client.chat.completions.create.side_effect = long_wait_err

        with pytest.raises(RuntimeError, match="quota"):
            agent._call_with_retry(model="primary-model", messages=[])

    def test_non_429_non_413_status_error_reraised_immediately(self, fake_groq_error_factory):
        agent = self._agent()
        agent._client = MagicMock()
        server_err = fake_groq_error_factory(500, "internal server error")
        agent._client.chat.completions.create.side_effect = server_err

        import groq
        with pytest.raises(groq.APIStatusError):
            agent._call_with_retry(model="primary-model", messages=[])

    def test_waits_when_short_wait_and_no_fallback_available(self, fake_groq_error_factory, fake_completion_factory, monkeypatch):
        agent = self._agent(fallback_model=None)
        agent._client = MagicMock()
        short_wait_err = fake_groq_error_factory(429, "please try again in 3s")
        agent._client.chat.completions.create.side_effect = [short_wait_err, fake_completion_factory("ok")]

        sleep_calls = []
        monkeypatch.setattr(base_agent.time, "sleep", lambda s: sleep_calls.append(s))

        result = agent._call_with_retry(model="primary-model", messages=[])
        assert result.choices[0].message.content == "ok"
        assert sleep_calls == [pytest.approx(4.0, abs=0.01)]


class TestKeyRotationWraps:
    """Reproduces a real bug: rotation only ever moved forward
    (`_key_index + 1 >= len(keys)` -> give up). Specialists are spread across
    the keys round-robin, so every agent that started on the LAST key had no
    key to rotate to, even with untouched quota on the first one."""

    def _agent(self, **kw):
        return base_agent.ToolAgent(name="T", system_prompt="s", model="primary-model", fallback_model=None, **kw)

    def test_agent_starting_on_the_last_key_rotates_to_the_first(self, fake_groq_error_factory, fake_completion_factory, monkeypatch):
        monkeypatch.setattr(base_agent, "GROQ_API_KEYS", ["key0", "key1"])
        fresh_client = MagicMock()
        fresh_client.chat.completions.create.side_effect = [fake_completion_factory("ok")]
        monkeypatch.setattr(base_agent, "Groq", MagicMock(return_value=fresh_client))

        agent = self._agent(starting_key_index=1)
        agent._client = MagicMock()
        agent._client.chat.completions.create.side_effect = [fake_groq_error_factory(429, "please try again in 2h0m0s")]

        result = agent._call_with_retry(model="primary-model", messages=[])
        assert result.choices[0].message.content == "ok"
        assert agent._key_index == 0

    def test_gives_up_after_trying_every_key_once(self, fake_groq_error_factory, monkeypatch):
        monkeypatch.setattr(base_agent, "GROQ_API_KEYS", ["key0", "key1", "key2"])
        exhausted = MagicMock()
        exhausted.chat.completions.create.side_effect = fake_groq_error_factory(429, "please try again in 2h0m0s")
        groq_ctor = MagicMock(return_value=exhausted)
        monkeypatch.setattr(base_agent, "Groq", groq_ctor)

        agent = self._agent(starting_key_index=1)
        agent._client = exhausted
        with pytest.raises(RuntimeError, match="quota"):
            agent._call_with_retry(model="primary-model", messages=[])
        assert exhausted.chat.completions.create.call_count == 3  # each key exactly once


class TestDeadline:
    def _agent(self, **kw):
        return base_agent.ToolAgent(name="T", system_prompt="s", model="primary-model", fallback_model=None, **kw)

    def test_does_not_sleep_past_the_deadline(self, fake_groq_error_factory, monkeypatch):
        """A web audit used to sleep through rate-limit waits of up to 15
        minutes each while holding one of the two scan slots."""
        from agent.errors import AuditFailed
        sleep_calls = []
        monkeypatch.setattr(base_agent.time, "sleep", lambda s: sleep_calls.append(s))

        agent = self._agent(deadline=base_agent.time.monotonic() + 30)
        agent._client = MagicMock()
        agent._client.chat.completions.create.side_effect = fake_groq_error_factory(429, "please try again in 5m0s")

        with pytest.raises(AuditFailed) as excinfo:
            agent._call_with_retry(model="primary-model", messages=[])
        assert excinfo.value.code == "timeout"
        assert sleep_calls == []

    def test_no_model_call_is_started_after_the_deadline(self):
        from agent.errors import AuditFailed
        agent = self._agent(deadline=base_agent.time.monotonic() - 1)
        agent._client = MagicMock()
        with pytest.raises(AuditFailed):
            agent._call_with_retry(model="primary-model", messages=[])
        agent._client.chat.completions.create.assert_not_called()

    def test_short_wait_inside_the_deadline_still_sleeps(self, fake_groq_error_factory, fake_completion_factory, monkeypatch):
        sleep_calls = []
        monkeypatch.setattr(base_agent.time, "sleep", lambda s: sleep_calls.append(s))
        agent = self._agent(deadline=base_agent.time.monotonic() + 600)
        agent._client = MagicMock()
        agent._client.chat.completions.create.side_effect = [
            fake_groq_error_factory(429, "please try again in 3s"), fake_completion_factory("ok")]
        assert agent._call_with_retry(model="primary-model", messages=[]).choices[0].message.content == "ok"
        assert len(sleep_calls) == 1

    def test_no_deadline_means_no_limit(self):
        assert self._agent()._out_of_time(10_000) is False
