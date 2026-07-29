"""Tests for the AG-UI client (``pando.agui``).

They exercise the synchronous path and the pure parsing helpers, so the whole
module runs without ``httpx``.
"""

from __future__ import annotations

import io
import json

import pytest

from pando.agui import (
    AguiMessage,
    AguiTool,
    PandoAguiClient,
    PandoAguiError,
    PandoPermissionRequest,
    PandoState,
    parse_sse,
)
from pando.exceptions import PandoConnectionError


def frame(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


class FakeResponse(io.BytesIO):
    """A urlopen result: a byte stream usable as a context manager."""

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def install_fake_urlopen(monkeypatch, body: str, capture: dict) -> None:
    def fake_urlopen(request, timeout=None):  # noqa: ANN001 - urllib signature
        capture["url"] = request.full_url
        capture["method"] = request.get_method()
        capture["headers"] = dict(request.headers)
        capture["body"] = request.data.decode("utf-8") if request.data else None
        capture["timeout"] = timeout
        return FakeResponse(body.encode("utf-8"))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)


# ---------------------------------------------------------------------------
# Request shape
# ---------------------------------------------------------------------------


def test_run_sync_posts_run_agent_input_with_bearer_token(monkeypatch):
    capture: dict = {}
    install_fake_urlopen(
        monkeypatch,
        frame({"type": "RUN_FINISHED", "threadId": "t", "runId": "r"}),
        capture,
    )

    client = PandoAguiClient(base_url="http://localhost:8090/", token="secret")
    events = list(client.run_sync(prompt="hello", thread_id="t", run_id="r"))

    assert capture["url"] == "http://localhost:8090/api/v1/agui/coder"
    assert capture["method"] == "POST"
    # urllib normalises header names to title case.
    headers = {k.lower(): v for k, v in capture["headers"].items()}
    assert headers["authorization"] == "Bearer secret"
    assert headers["accept"] == "text/event-stream"

    body = json.loads(capture["body"])
    assert body["threadId"] == "t"
    assert body["runId"] == "r"
    assert len(body["messages"]) == 1
    assert body["messages"][0]["role"] == "user"
    assert body["messages"][0]["content"] == "hello"
    assert [e.type for e in events] == ["RUN_FINISHED"]


def test_no_token_sends_no_authorization_header(monkeypatch):
    capture: dict = {}
    install_fake_urlopen(monkeypatch, frame({"type": "RUN_FINISHED"}), capture)

    list(PandoAguiClient(base_url="http://x").run_sync(prompt="hi"))

    assert "authorization" not in {k.lower() for k in capture["headers"]}


def test_tools_and_context_are_serialised(monkeypatch):
    capture: dict = {}
    install_fake_urlopen(monkeypatch, frame({"type": "RUN_FINISHED"}), capture)

    client = PandoAguiClient(base_url="http://x")
    list(
        client.run_sync(
            messages=[AguiMessage(id="m1", role="user", content="chart it")],
            tools=[AguiTool(name="show_chart", parameters={"type": "object"})],
            context=[{"description": "page", "value": "/dashboard"}],
            state={"selected": 3},
        )
    )

    body = json.loads(capture["body"])
    assert body["tools"] == [{"name": "show_chart", "parameters": {"type": "object"}}]
    assert body["context"] == [{"description": "page", "value": "/dashboard"}]
    assert body["state"] == {"selected": 3}
    assert body["messages"] == [{"id": "m1", "role": "user", "content": "chart it"}]


def test_agent_override_changes_the_endpoint(monkeypatch):
    capture: dict = {}
    install_fake_urlopen(monkeypatch, frame({"type": "RUN_FINISHED"}), capture)

    client = PandoAguiClient(base_url="http://x", agent="coder")
    list(client.run_sync(prompt="hi", agent="task"))

    assert capture["url"] == "http://x/api/v1/agui/task"
    assert client.agent_url() == "http://x/api/v1/agui/coder"


# ---------------------------------------------------------------------------
# SSE parsing
# ---------------------------------------------------------------------------


def test_parse_sse_reassembles_events_split_across_chunks():
    full = (
        frame({"type": "RUN_STARTED", "threadId": "t", "runId": "r"})
        + frame({"type": "TEXT_MESSAGE_CONTENT", "messageId": "m", "delta": "hel"})
        + frame({"type": "TEXT_MESSAGE_CONTENT", "messageId": "m", "delta": "lo"})
        + frame({"type": "RUN_FINISHED", "threadId": "t", "runId": "r"})
    )
    cut = 70  # falls inside the second event
    events = list(parse_sse([full[:cut].encode(), full[cut:].encode()]))

    assert [e.type for e in events] == [
        "RUN_STARTED",
        "TEXT_MESSAGE_CONTENT",
        "TEXT_MESSAGE_CONTENT",
        "RUN_FINISHED",
    ]
    assert "".join(e.delta for e in events if e.type == "TEXT_MESSAGE_CONTENT") == "hello"


def test_parse_sse_skips_malformed_frames_and_done_marker():
    body = (
        "data: not json\n\n"
        + frame({"type": "RUN_FINISHED"})
        + "data: [DONE]\n\n"
        + ": a comment\n\n"
    )
    assert [e.type for e in parse_sse([body])] == ["RUN_FINISHED"]


def test_parse_sse_yields_a_trailing_frame_without_blank_line():
    body = 'data: {"type": "RUN_FINISHED"}'
    assert [e.type for e in parse_sse([body])] == ["RUN_FINISHED"]


def test_event_fields_are_readable_in_both_spellings():
    (event,) = parse_sse(
        [frame({"type": "TOOL_CALL_START", "toolCallId": "c1", "toolCallName": "view"})]
    )
    assert event.tool_call_id == "c1"
    assert event.toolCallName == "view"
    assert event.get("tool_call_name") == "view"
    assert event.get("missing", "fallback") == "fallback"
    assert "toolCallId" in event
    with pytest.raises(AttributeError):
        _ = event.nope


# ---------------------------------------------------------------------------
# run_text and errors
# ---------------------------------------------------------------------------


def test_run_text_sync_concatenates_assistant_text(monkeypatch):
    body = (
        frame({"type": "TEXT_MESSAGE_CONTENT", "messageId": "m", "delta": "ok "})
        + frame({"type": "TOOL_CALL_START", "toolCallId": "c", "toolCallName": "view"})
        + frame({"type": "TEXT_MESSAGE_CONTENT", "messageId": "m", "delta": "done"})
        + frame({"type": "RUN_FINISHED"})
    )
    install_fake_urlopen(monkeypatch, body, {})

    assert PandoAguiClient(base_url="http://x").run_text_sync("hi") == "ok done"


def test_run_text_sync_raises_on_run_error(monkeypatch):
    install_fake_urlopen(
        monkeypatch, frame({"type": "RUN_ERROR", "message": "model refused"}), {}
    )

    with pytest.raises(PandoAguiError, match="model refused"):
        PandoAguiClient(base_url="http://x").run_text_sync("hi")


def test_http_error_becomes_pando_agui_error_with_status(monkeypatch):
    import urllib.error

    def fake_urlopen(request, timeout=None):  # noqa: ANN001
        raise urllib.error.HTTPError(
            request.full_url,
            401,
            "Unauthorized",
            {},
            io.BytesIO(json.dumps({"error": "invalid or missing token"}).encode()),
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    with pytest.raises(PandoAguiError) as excinfo:
        list(PandoAguiClient(base_url="http://x").run_sync(prompt="hi"))
    assert excinfo.value.status == 401
    assert "invalid or missing token" in str(excinfo.value)


def test_connection_failure_becomes_pando_connection_error(monkeypatch):
    import urllib.error

    def fake_urlopen(request, timeout=None):  # noqa: ANN001
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    with pytest.raises(PandoConnectionError):
        list(PandoAguiClient(base_url="http://x").run_sync(prompt="hi"))


# ---------------------------------------------------------------------------
# Discovery and state
# ---------------------------------------------------------------------------


def test_info_sync_parses_the_discovery_document(monkeypatch):
    capture: dict = {}
    payload = {
        "protocol": "ag-ui",
        "version": "0.1",
        "path": "/api/v1/agui",
        "agents": [
            {
                "name": "coder",
                "url": "http://x/api/v1/agui/coder",
                "model": {"id": "claude", "contextWindow": 200000},
            }
        ],
        "capabilities": {
            "frontendTools": True,
            "humanInTheLoop": True,
            "sharedState": True,
            "interrupts": True,
        },
    }
    install_fake_urlopen(monkeypatch, json.dumps(payload), capture)

    info = PandoAguiClient(base_url="http://x", token="t").info_sync()

    assert capture["url"] == "http://x/api/v1/agui/info"
    assert capture["method"] == "GET"
    assert info.protocol == "ag-ui"
    assert [a.name for a in info.agents] == ["coder"]
    assert info.agents[0].model is not None
    assert info.agents[0].model.context_window == 200000
    assert info.capabilities.frontend_tools is True


def test_state_snapshot_parses_into_pando_state():
    snapshot = {
        "thread": "t",
        "session": "s",
        "agent": "coder",
        "model": {"id": "claude", "provider": "anthropic"},
        "todos": [{"id": "1", "content": "ship", "status": "pending"}],
        "tokenUsage": {
            "promptTokens": 10,
            "completionTokens": 4,
            "contextWindow": 200000,
            "estimated": False,
        },
        "files": [{"path": "/a.go", "name": "a.go", "action": "edit"}],
        "subAgents": [{"id": "sa1", "status": "running", "role": "worker"}],
    }
    state = PandoState.from_wire(snapshot)

    assert state.model.provider == "anthropic"
    assert state.token_usage is not None
    assert state.token_usage.prompt_tokens == 10
    assert state.files[0].action == "edit"
    assert state.sub_agents[0].role == "worker"
    assert state.raw["thread"] == "t"


def test_permission_request_parses_tool_arguments():
    request = PandoPermissionRequest.from_arguments(
        json.dumps({"toolName": "bash", "action": "run", "path": "/repo"})
    )
    assert request.tool_name == "bash"
    assert request.action == "run"
    assert request.path == "/repo"
