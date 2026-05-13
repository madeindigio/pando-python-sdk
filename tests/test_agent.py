"""Tests for pando.agent and pando.session — ACP stdio mode."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pando.agent import PandoAgent
from pando.events import AgentEvent
from pando.exceptions import (
    PandoConnectionError,
    PandoRPCError,
    PandoSessionError,
)
from pando.session import PandoSession
from pando.transport import _JsonRpcTransport, QUEUE_SENTINEL


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------


def _make_transport(connected: bool = True) -> MagicMock:
    """Create a mock _JsonRpcTransport."""
    transport = AsyncMock(spec=_JsonRpcTransport)
    transport.is_connected.return_value = connected
    transport.connect = AsyncMock()
    transport.disconnect = AsyncMock()
    transport.call = AsyncMock(return_value={})
    transport.notify = AsyncMock()
    transport.get_session_queue = AsyncMock(return_value=asyncio.Queue())
    transport.remove_session_queue = AsyncMock()
    return transport


def _make_agent(transport: MagicMock | None = None) -> PandoAgent:
    """Create a PandoAgent with a mock transport."""
    agent = PandoAgent(cwd="/test/project")
    agent._connected = True
    agent._pando_path = "/usr/bin/pando"
    agent._transport = transport or _make_transport()
    return agent


# ---------------------------------------------------------------------------
# PandoAgent tests
# ---------------------------------------------------------------------------


class TestPandoAgentConnect:
    @pytest.mark.asyncio
    async def test_connect_finds_binary_and_starts_transport(self):
        with (
            patch("pando.agent._find_pando_binary", return_value="/usr/bin/pando"),
            patch("pando.agent._JsonRpcTransport") as MockTransport,
        ):
            mock_transport = _make_transport()
            MockTransport.return_value = mock_transport

            agent = PandoAgent(cwd="/my/project")
            await agent.connect()

        assert agent._connected
        mock_transport.connect.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_connect_idempotent(self):
        """Calling connect() twice should not reconnect."""
        with patch("pando.agent._find_pando_binary", return_value="/usr/bin/pando"):
            with patch("pando.agent._JsonRpcTransport") as MockTransport:
                mock_transport = _make_transport()
                MockTransport.return_value = mock_transport

                agent = PandoAgent()
                await agent.connect()
                await agent.connect()  # second call

        assert mock_transport.connect.await_count == 1

    @pytest.mark.asyncio
    async def test_disconnect_clears_transport(self):
        transport = _make_transport()
        agent = _make_agent(transport)

        await agent.disconnect()

        transport.disconnect.assert_awaited_once()
        assert not agent._connected
        assert agent._transport is None


class TestPandoAgentSessions:
    @pytest.mark.asyncio
    async def test_create_session_returns_pando_session(self):
        transport = _make_transport()
        transport.call = AsyncMock(return_value={"sessionId": "uuid-1234"})

        agent = _make_agent(transport)
        agent._cwd = None  # No cwd so params is None
        session = await agent.create_session("Test session")

        assert isinstance(session, PandoSession)
        assert session.id == "uuid-1234"
        transport.call.assert_awaited_with("session/create", None)

    @pytest.mark.asyncio
    async def test_create_session_with_cwd(self):
        transport = _make_transport()
        transport.call = AsyncMock(return_value={"sessionId": "abc"})

        agent = _make_agent(transport)
        agent._cwd = "/my/project"
        session = await agent.create_session()

        assert session.id == "abc"
        transport.call.assert_awaited_with("session/create", {"cwd": "/my/project"})

    @pytest.mark.asyncio
    async def test_create_session_raises_on_missing_session_id(self):
        transport = _make_transport()
        transport.call = AsyncMock(return_value={})  # no sessionId

        agent = _make_agent(transport)
        with pytest.raises(PandoRPCError, match="no sessionId"):
            await agent.create_session()

    @pytest.mark.asyncio
    async def test_list_sessions(self):
        sessions_raw = [
            {"sessionId": "s1", "title": "First", "updatedAt": "2026-01-01"},
            {"sessionId": "s2", "title": "Second"},
        ]
        transport = _make_transport()
        transport.call = AsyncMock(return_value={"sessions": sessions_raw})

        agent = _make_agent(transport)
        sessions = await agent.list_sessions()

        assert len(sessions) == 2
        assert sessions[0].session_id == "s1"
        assert sessions[0].title == "First"
        assert sessions[1].session_id == "s2"

    @pytest.mark.asyncio
    async def test_list_sessions_empty(self):
        transport = _make_transport()
        transport.call = AsyncMock(return_value={"sessions": []})

        agent = _make_agent(transport)
        sessions = await agent.list_sessions()
        assert sessions == []


class TestPandoAgentSettings:
    @pytest.mark.asyncio
    async def test_set_model(self):
        transport = _make_transport()
        agent = _make_agent(transport)
        await agent.set_model("claude-sonnet-4-6")
        transport.call.assert_awaited_with("model/set", {"modelId": "claude-sonnet-4-6"})

    @pytest.mark.asyncio
    async def test_list_models(self):
        transport = _make_transport()
        transport.call = AsyncMock(return_value={"models": [{"id": "m1"}, {"id": "m2"}]})
        agent = _make_agent(transport)
        models = await agent.list_models()
        assert models == [{"id": "m1"}, {"id": "m2"}]

    @pytest.mark.asyncio
    async def test_set_persona(self):
        transport = _make_transport()
        agent = _make_agent(transport)
        await agent.set_persona("software-engineer")
        transport.call.assert_awaited_with("persona/set", {"name": "software-engineer"})

    @pytest.mark.asyncio
    async def test_get_persona(self):
        transport = _make_transport()
        transport.call = AsyncMock(return_value={"active": "qa"})
        agent = _make_agent(transport)
        persona = await agent.get_persona()
        assert persona == "qa"

    @pytest.mark.asyncio
    async def test_list_personas(self):
        transport = _make_transport()
        transport.call = AsyncMock(
            return_value={"personas": ["assistant", "software-engineer", "qa"]}
        )
        agent = _make_agent(transport)
        personas = await agent.list_personas()
        assert "qa" in personas
        assert "assistant" in personas

    @pytest.mark.asyncio
    async def test_list_tools(self):
        transport = _make_transport()
        transport.call = AsyncMock(return_value={"tools": [{"name": "bash"}, {"name": "edit"}]})
        agent = _make_agent(transport)
        tools = await agent.list_tools()
        assert len(tools) == 2

    @pytest.mark.asyncio
    async def test_require_transport_raises_when_disconnected(self):
        agent = PandoAgent()
        # Not connected
        with pytest.raises(PandoConnectionError, match="not connected"):
            agent._require_transport()


class TestPandoAgentAsyncContextManager:
    @pytest.mark.asyncio
    async def test_async_context_manager_connects_and_disconnects(self):
        with (
            patch("pando.agent._find_pando_binary", return_value="/usr/bin/pando"),
            patch("pando.agent._JsonRpcTransport") as MockTransport,
        ):
            mock_transport = _make_transport()
            MockTransport.return_value = mock_transport

            async with PandoAgent(cwd="/test") as agent:
                assert agent._connected

            mock_transport.disconnect.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_async_context_manager_disconnects_on_exception(self):
        with (
            patch("pando.agent._find_pando_binary", return_value="/usr/bin/pando"),
            patch("pando.agent._JsonRpcTransport") as MockTransport,
        ):
            mock_transport = _make_transport()
            MockTransport.return_value = mock_transport

            with pytest.raises(ValueError):
                async with PandoAgent() as agent:
                    raise ValueError("test error")

            mock_transport.disconnect.assert_awaited_once()


# ---------------------------------------------------------------------------
# PandoSession tests
# ---------------------------------------------------------------------------


class TestPandoSessionSend:
    @pytest.mark.asyncio
    async def test_send_streams_events_until_response(self):
        queue: asyncio.Queue = asyncio.Queue()
        # Pre-fill queue with events
        await queue.put(AgentEvent(type="content_delta", session_id="s1", delta="Hello "))
        await queue.put(AgentEvent(type="content_delta", session_id="s1", delta="world"))
        await queue.put(AgentEvent(type="response", session_id="s1"))

        transport = _make_transport()
        transport.call = AsyncMock(return_value={})
        transport.get_session_queue = AsyncMock(return_value=queue)

        session = PandoSession(session_id="s1", transport=transport)

        events = []
        async for event in session.send("Hello"):
            events.append(event)

        assert len(events) == 3
        assert events[0].type == "content_delta"
        assert events[0].delta == "Hello "
        assert events[2].type == "response"

    @pytest.mark.asyncio
    async def test_send_stops_on_error_event(self):
        queue: asyncio.Queue = asyncio.Queue()
        await queue.put(AgentEvent(type="error", session_id="s2", error="Agent failed"))

        transport = _make_transport()
        transport.call = AsyncMock(return_value={})
        transport.get_session_queue = AsyncMock(return_value=queue)

        session = PandoSession(session_id="s2", transport=transport)

        events = []
        async for event in session.send("test"):
            events.append(event)

        assert len(events) == 1
        assert events[0].type == "error"
        assert events[0].error == "Agent failed"

    @pytest.mark.asyncio
    async def test_send_raises_connection_error_on_sentinel(self):
        queue: asyncio.Queue = asyncio.Queue()
        await queue.put(QUEUE_SENTINEL)

        transport = _make_transport()
        transport.call = AsyncMock(return_value={})
        transport.get_session_queue = AsyncMock(return_value=queue)

        session = PandoSession(session_id="s3", transport=transport)

        with pytest.raises(PandoConnectionError, match="exited"):
            async for _ in session.send("test"):
                pass

    @pytest.mark.asyncio
    async def test_ask_collects_full_response(self):
        queue: asyncio.Queue = asyncio.Queue()
        await queue.put(AgentEvent(type="content_delta", session_id="s4", delta="Hello "))
        await queue.put(AgentEvent(type="content_delta", session_id="s4", delta="world!"))
        await queue.put(AgentEvent(type="response", session_id="s4"))

        transport = _make_transport()
        transport.call = AsyncMock(return_value={})
        transport.get_session_queue = AsyncMock(return_value=queue)

        session = PandoSession(session_id="s4", transport=transport)
        text = await session.ask("hi")

        assert text == "Hello world!"

    @pytest.mark.asyncio
    async def test_ask_raises_on_error_event(self):
        queue: asyncio.Queue = asyncio.Queue()
        await queue.put(AgentEvent(type="error", session_id="s5", error="Oops"))

        transport = _make_transport()
        transport.call = AsyncMock(return_value={})
        transport.get_session_queue = AsyncMock(return_value=queue)

        session = PandoSession(session_id="s5", transport=transport)

        with pytest.raises(PandoSessionError, match="Agent error: Oops"):
            await session.ask("test")

    @pytest.mark.asyncio
    async def test_set_persona(self):
        transport = _make_transport()
        session = PandoSession(session_id="s6", transport=transport)
        await session.set_persona("qa")
        transport.call.assert_awaited_with(
            "persona/set_session", {"sessionId": "s6", "name": "qa"}
        )

    @pytest.mark.asyncio
    async def test_cancel(self):
        transport = _make_transport()
        session = PandoSession(session_id="s7", transport=transport)
        await session.cancel()
        transport.notify.assert_awaited_with("cancel", {"sessionId": "s7"})

    @pytest.mark.asyncio
    async def test_close(self):
        transport = _make_transport()
        transport.call = AsyncMock(return_value={})
        session = PandoSession(session_id="s8", transport=transport)
        await session.close()
        transport.call.assert_awaited_with("session/close", {"sessionId": "s8"})
        transport.remove_session_queue.assert_awaited_with("s8")

    @pytest.mark.asyncio
    async def test_send_prompt_rpc_error_raises_session_error(self):
        transport = _make_transport()
        transport.call = AsyncMock(side_effect=PandoRPCError(-32602, "Session not found"))
        transport.get_session_queue = AsyncMock(return_value=asyncio.Queue())

        session = PandoSession(session_id="s9", transport=transport)

        with pytest.raises(PandoSessionError, match="Failed to send prompt"):
            async for _ in session.send("test"):
                pass
