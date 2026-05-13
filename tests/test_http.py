"""Tests for pando.http — PandoHttpClient REST mode.

These tests mock httpx to avoid requiring a real pando server.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Skip entire module if httpx is not installed
httpx = pytest.importorskip("httpx")

from pando.http import PandoHttpClient, _parse_session_info, _raise_for_status
from pando.exceptions import PandoConnectionError, PandoError
from pando.types import MessageChunk, ModelInfo, SessionInfo


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_response(status_code: int, body: dict | list | str) -> MagicMock:
    """Create a mock httpx.Response."""
    resp = MagicMock()
    resp.status_code = status_code
    if isinstance(body, (dict, list)):
        resp.json.return_value = body
        resp.text = json.dumps(body)
    else:
        resp.json.side_effect = ValueError("Not JSON")
        resp.text = str(body)
    return resp


def _make_async_client() -> AsyncMock:
    """Create a mock httpx.AsyncClient."""
    client = AsyncMock()
    client.aclose = AsyncMock()
    return client


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestRaiseForStatus:
    def test_success_does_not_raise(self):
        resp = _make_response(200, {"ok": True})
        _raise_for_status(resp)  # Should not raise

    def test_404_raises_pando_error(self):
        resp = _make_response(404, {"error": "Not found"})
        with pytest.raises(PandoError, match="404"):
            _raise_for_status(resp)

    def test_500_raises_pando_error(self):
        resp = _make_response(500, "Internal server error")
        with pytest.raises(PandoError, match="500"):
            _raise_for_status(resp)

    def test_201_does_not_raise(self):
        resp = _make_response(201, {"id": "new"})
        _raise_for_status(resp)  # Should not raise


class TestParseSessionInfo:
    def test_parse_with_id(self):
        data = {"id": "abc", "title": "Test Session", "updatedAt": "2026-01-01"}
        info = _parse_session_info(data)
        assert info.session_id == "abc"
        assert info.title == "Test Session"
        assert info.updated_at == "2026-01-01"

    def test_parse_with_session_id_key(self):
        data = {"sessionId": "xyz", "title": "Another"}
        info = _parse_session_info(data)
        assert info.session_id == "xyz"

    def test_parse_missing_fields_defaults(self):
        info = _parse_session_info({})
        assert info.session_id == ""
        assert info.title == ""
        assert info.updated_at == ""
        assert info.cwd == ""


# ---------------------------------------------------------------------------
# PandoHttpClient tests
# ---------------------------------------------------------------------------


class TestPandoHttpClientConnect:
    @pytest.mark.asyncio
    async def test_connect_creates_http_client(self):
        with patch("pando.http.httpx.AsyncClient") as MockClient:
            mock_client = _make_async_client()
            MockClient.return_value = mock_client

            client = PandoHttpClient(base_url="http://localhost:8765")
            await client.connect()

        assert client._client is mock_client
        assert client._sessions is not None
        assert client._models is not None
        assert client._personas is not None

    @pytest.mark.asyncio
    async def test_close_calls_aclose(self):
        with patch("pando.http.httpx.AsyncClient") as MockClient:
            mock_client = _make_async_client()
            MockClient.return_value = mock_client

            client = PandoHttpClient()
            await client.connect()
            await client.close()

        mock_client.aclose.assert_awaited_once()
        assert client._client is None

    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        with patch("pando.http.httpx.AsyncClient") as MockClient:
            mock_client = _make_async_client()
            MockClient.return_value = mock_client

            async with PandoHttpClient() as client:
                assert client._client is mock_client

            mock_client.aclose.assert_awaited_once()

    def test_sessions_raises_when_not_connected(self):
        client = PandoHttpClient()
        with pytest.raises(PandoConnectionError):
            _ = client.sessions

    def test_models_raises_when_not_connected(self):
        client = PandoHttpClient()
        with pytest.raises(PandoConnectionError):
            _ = client.models

    def test_personas_raises_when_not_connected(self):
        client = PandoHttpClient()
        with pytest.raises(PandoConnectionError):
            _ = client.personas


class TestSessionsClient:
    @pytest.mark.asyncio
    async def test_create_session(self):
        with patch("pando.http.httpx.AsyncClient") as MockClient:
            mock_client = _make_async_client()
            MockClient.return_value = mock_client
            mock_client.post = AsyncMock(
                return_value=_make_response(201, {"id": "new-session-id", "title": "My Task"})
            )

            async with PandoHttpClient() as client:
                session = await client.sessions.create("My Task")

        assert isinstance(session, SessionInfo)
        assert session.session_id == "new-session-id"
        assert session.title == "My Task"

    @pytest.mark.asyncio
    async def test_list_sessions(self):
        sessions_data = [
            {"id": "s1", "title": "First"},
            {"id": "s2", "title": "Second"},
        ]
        with patch("pando.http.httpx.AsyncClient") as MockClient:
            mock_client = _make_async_client()
            MockClient.return_value = mock_client
            mock_client.get = AsyncMock(
                return_value=_make_response(200, sessions_data)
            )

            async with PandoHttpClient() as client:
                sessions = await client.sessions.list()

        assert len(sessions) == 2
        assert sessions[0].session_id == "s1"
        assert sessions[1].title == "Second"

    @pytest.mark.asyncio
    async def test_list_sessions_with_wrapper_dict(self):
        data = {"sessions": [{"id": "s3", "title": "Third"}]}
        with patch("pando.http.httpx.AsyncClient") as MockClient:
            mock_client = _make_async_client()
            MockClient.return_value = mock_client
            mock_client.get = AsyncMock(return_value=_make_response(200, data))

            async with PandoHttpClient() as client:
                sessions = await client.sessions.list()

        assert len(sessions) == 1
        assert sessions[0].session_id == "s3"

    @pytest.mark.asyncio
    async def test_get_messages(self):
        messages = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]
        with patch("pando.http.httpx.AsyncClient") as MockClient:
            mock_client = _make_async_client()
            MockClient.return_value = mock_client
            mock_client.get = AsyncMock(return_value=_make_response(200, messages))

            async with PandoHttpClient() as client:
                result = await client.sessions.get_messages("s1")

        assert len(result) == 2
        assert result[0]["role"] == "user"


class TestModelsClient:
    @pytest.mark.asyncio
    async def test_list_models(self):
        models_data = [
            {"id": "claude-sonnet-4-6", "name": "Claude Sonnet", "provider": "claude"},
            {"id": "gpt-5.4", "name": "GPT-5.4", "provider": "copilot"},
        ]
        with patch("pando.http.httpx.AsyncClient") as MockClient:
            mock_client = _make_async_client()
            MockClient.return_value = mock_client
            mock_client.get = AsyncMock(return_value=_make_response(200, models_data))

            async with PandoHttpClient() as client:
                models = await client.models.list()

        assert len(models) == 2
        assert isinstance(models[0], ModelInfo)
        assert models[0].id == "claude-sonnet-4-6"
        assert models[1].provider == "copilot"

    @pytest.mark.asyncio
    async def test_set_active_model(self):
        with patch("pando.http.httpx.AsyncClient") as MockClient:
            mock_client = _make_async_client()
            MockClient.return_value = mock_client
            mock_client.put = AsyncMock(return_value=_make_response(200, {"ok": True}))

            async with PandoHttpClient() as client:
                await client.models.set_active("claude-sonnet-4-6")

        mock_client.put.assert_awaited_once()
        call_kwargs = mock_client.put.call_args
        assert "json" in call_kwargs.kwargs
        assert call_kwargs.kwargs["json"]["modelId"] == "claude-sonnet-4-6"


class TestPersonasClient:
    @pytest.mark.asyncio
    async def test_list_personas(self):
        with patch("pando.http.httpx.AsyncClient") as MockClient:
            mock_client = _make_async_client()
            MockClient.return_value = mock_client
            mock_client.get = AsyncMock(
                return_value=_make_response(
                    200, ["assistant", "software-engineer", "qa", "system-engineer"]
                )
            )

            async with PandoHttpClient() as client:
                personas = await client.personas.list()

        assert "assistant" in personas
        assert "qa" in personas
        assert len(personas) == 4

    @pytest.mark.asyncio
    async def test_get_active_persona(self):
        with patch("pando.http.httpx.AsyncClient") as MockClient:
            mock_client = _make_async_client()
            MockClient.return_value = mock_client
            mock_client.get = AsyncMock(
                return_value=_make_response(200, {"active": "software-engineer"})
            )

            async with PandoHttpClient() as client:
                active = await client.personas.get_active()

        assert active == "software-engineer"

    @pytest.mark.asyncio
    async def test_set_active_persona(self):
        with patch("pando.http.httpx.AsyncClient") as MockClient:
            mock_client = _make_async_client()
            MockClient.return_value = mock_client
            mock_client.put = AsyncMock(return_value=_make_response(200, {"ok": True}))

            async with PandoHttpClient() as client:
                await client.personas.set_active("qa")

        mock_client.put.assert_awaited_once()


class TestImportWithoutHttpx:
    """Verify that importing PandoHttpClient without httpx raises a clear error."""

    def test_require_httpx_import_error(self):
        """_require_httpx() should raise ImportError if httpx is not installed."""
        with patch("pando.http._HTTPX_AVAILABLE", False):
            from pando.http import _require_httpx
            with pytest.raises(ImportError, match="httpx"):
                _require_httpx()
