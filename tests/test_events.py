"""Tests for pando.events — AgentEvent parsing."""

from __future__ import annotations

import pytest

from pando.events import AgentEvent, parse_agent_event
from pando.exceptions import PandoError


class TestParseAgentEvent:
    def test_content_delta(self):
        params = {
            "type": "content_delta",
            "sessionId": "sess-123",
            "delta": "Hello, world!",
        }
        event = parse_agent_event(params)

        assert isinstance(event, AgentEvent)
        assert event.type == "content_delta"
        assert event.session_id == "sess-123"
        assert event.delta == "Hello, world!"
        assert event.tool_call is None
        assert event.tool_result is None
        assert event.message is None
        assert event.error is None

    def test_thinking_delta(self):
        params = {
            "type": "thinking_delta",
            "sessionId": "sess-abc",
            "delta": "Let me think...",
        }
        event = parse_agent_event(params)
        assert event.type == "thinking_delta"
        assert event.delta == "Let me think..."

    def test_tool_call_event(self):
        tool_call_data = {"name": "bash", "input": {"command": "ls -la"}}
        params = {
            "type": "tool_call",
            "sessionId": "sess-456",
            "toolCall": tool_call_data,
        }
        event = parse_agent_event(params)
        assert event.type == "tool_call"
        assert event.tool_call == tool_call_data
        assert event.delta is None

    def test_tool_result_event(self):
        tool_result_data = {"content": "file1.txt\nfile2.txt"}
        params = {
            "type": "tool_result",
            "sessionId": "sess-789",
            "toolResult": tool_result_data,
        }
        event = parse_agent_event(params)
        assert event.type == "tool_result"
        assert event.tool_result == tool_result_data

    def test_response_event(self):
        message_data = {"role": "assistant", "content": "Done!"}
        params = {
            "type": "response",
            "sessionId": "sess-101",
            "message": message_data,
        }
        event = parse_agent_event(params)
        assert event.type == "response"
        assert event.message == message_data
        assert event.delta is None

    def test_error_event(self):
        params = {
            "type": "error",
            "sessionId": "sess-err",
            "error": "Something went wrong",
        }
        event = parse_agent_event(params)
        assert event.type == "error"
        assert event.error == "Something went wrong"

    def test_summarize_event(self):
        params = {"type": "summarize", "sessionId": "sess-sum"}
        event = parse_agent_event(params)
        assert event.type == "summarize"
        assert event.session_id == "sess-sum"

    def test_missing_session_id_defaults_to_empty(self):
        params = {"type": "content_delta", "delta": "hi"}
        event = parse_agent_event(params)
        assert event.session_id == ""

    def test_missing_type_raises_pando_error(self):
        with pytest.raises(PandoError, match="missing 'type'"):
            parse_agent_event({"sessionId": "sess-x", "delta": "hi"})

    def test_extra_fields_captured(self):
        params = {
            "type": "content_delta",
            "sessionId": "s",
            "delta": "x",
            "customField": "value",
            "anotherField": 42,
        }
        event = parse_agent_event(params)
        assert event.extra == {"customField": "value", "anotherField": 42}

    def test_empty_extra_fields(self):
        params = {"type": "response", "sessionId": "s"}
        event = parse_agent_event(params)
        assert event.extra == {}
