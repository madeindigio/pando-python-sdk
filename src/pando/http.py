"""PandoHttpClient — HTTP REST client for the pando API server.

This module provides :class:`PandoHttpClient` which communicates with a running
``pando serve`` (or ``pando app``) HTTP server via the REST API.

The HTTP client requires the ``httpx`` optional dependency::

    pip install pando-sdk[http]

Usage::

    from pando import PandoHttpClient

    async with PandoHttpClient(base_url="http://localhost:8765") as client:
        session = await client.sessions.create("My task")
        async for chunk in client.sessions.send_message(session.session_id, "Fix lint"):
            if chunk.delta:
                print(chunk.delta, end="", flush=True)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator

from pando.exceptions import PandoConnectionError, PandoError
from pando.types import MessageChunk, ModelInfo, SessionInfo

logger = logging.getLogger(__name__)

# httpx is an optional dependency
try:
    import httpx

    _HTTPX_AVAILABLE = True
except ImportError:
    _HTTPX_AVAILABLE = False


def _require_httpx() -> None:
    if not _HTTPX_AVAILABLE:
        raise ImportError(
            "The 'httpx' package is required for PandoHttpClient. "
            "Install it with: pip install pando-sdk[http]"
        )


# ---------------------------------------------------------------------------
# Sub-resource clients
# ---------------------------------------------------------------------------


class _SessionsClient:
    """Sub-client for session-related endpoints.

    Do not instantiate this directly. Access via :attr:`PandoHttpClient.sessions`.
    """

    def __init__(self, http: "httpx.AsyncClient", base_url: str) -> None:
        self._http = http
        self._base = base_url

    async def create(self, title: str = "") -> SessionInfo:
        """Create a new session.

        Args:
            title: Optional human-readable session title.

        Returns:
            :class:`~pando.types.SessionInfo` for the new session.
        """
        payload: dict[str, Any] = {}
        if title:
            payload["title"] = title

        resp = await self._http.post(f"{self._base}/api/v1/sessions", json=payload)
        _raise_for_status(resp)
        data = resp.json()
        return _parse_session_info(data)

    async def list(self) -> list[SessionInfo]:
        """List all sessions.

        Returns:
            A list of :class:`~pando.types.SessionInfo` objects.
        """
        resp = await self._http.get(f"{self._base}/api/v1/sessions")
        _raise_for_status(resp)
        data = resp.json()
        # Response may be a list or {"sessions": [...]}
        items = data if isinstance(data, list) else data.get("sessions", [])
        return [_parse_session_info(s) for s in items]

    async def get_messages(self, session_id: str) -> list[dict]:
        """Get message history for a session.

        Args:
            session_id: The session identifier.

        Returns:
            A list of message dictionaries.
        """
        resp = await self._http.get(
            f"{self._base}/api/v1/sessions/{session_id}/messages"
        )
        _raise_for_status(resp)
        data = resp.json()
        return data if isinstance(data, list) else data.get("messages", [])

    async def send_message(
        self,
        session_id: str,
        content: str,
    ) -> AsyncGenerator[MessageChunk, None]:
        """Send a message to a session and stream the response via SSE.

        This is an async generator that yields :class:`~pando.types.MessageChunk`
        objects as they arrive from the server-sent events (SSE) stream.

        Example::

            async for chunk in client.sessions.send_message(session_id, "Fix lint"):
                if chunk.delta:
                    print(chunk.delta, end="", flush=True)

        Args:
            session_id: The session identifier.
            content: The user message text.

        Yields:
            :class:`~pando.types.MessageChunk` objects.
        """
        payload = {"content": content}
        url = f"{self._base}/api/v1/sessions/{session_id}/messages"

        async with self._http.stream("POST", url, json=payload) as response:
            _raise_for_status(response)

            async for line in response.aiter_lines():
                line = line.strip()
                if not line:
                    continue

                # SSE format: "data: {...}"
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        logger.debug("Failed to parse SSE data: %s", data_str[:200])
                        continue

                    event_type = data.get("type", "")
                    delta = data.get("delta") or data.get("content")
                    if isinstance(delta, list):
                        # Anthropic-style content blocks
                        delta = "".join(
                            b.get("text", "") for b in delta if isinstance(b, dict)
                        )
                    elif not isinstance(delta, str):
                        delta = None

                    yield MessageChunk(delta=delta, type=event_type, raw=data)

                elif line.startswith("{"):
                    # Raw JSON line (non-SSE format)
                    try:
                        data = json.loads(line)
                        event_type = data.get("type", "")
                        delta = data.get("delta") or data.get("content", "")
                        yield MessageChunk(delta=delta or None, type=event_type, raw=data)
                    except json.JSONDecodeError:
                        pass


class _ModelsClient:
    """Sub-client for model-related endpoints.

    Do not instantiate this directly. Access via :attr:`PandoHttpClient.models`.
    """

    def __init__(self, http: "httpx.AsyncClient", base_url: str) -> None:
        self._http = http
        self._base = base_url

    async def list(self) -> list[ModelInfo]:
        """List available models.

        Returns:
            A list of :class:`~pando.types.ModelInfo` objects.
        """
        resp = await self._http.get(f"{self._base}/api/v1/models")
        _raise_for_status(resp)
        data = resp.json()
        items = data if isinstance(data, list) else data.get("models", [])
        result = []
        for m in items:
            if isinstance(m, dict):
                result.append(
                    ModelInfo(
                        id=m.get("id", ""),
                        name=m.get("name", ""),
                        provider=m.get("provider", ""),
                    )
                )
        return result

    async def set_active(self, model_id: str) -> None:
        """Set the active model.

        Args:
            model_id: The model identifier to activate.
        """
        resp = await self._http.put(
            f"{self._base}/api/v1/models/active",
            json={"modelId": model_id},
        )
        _raise_for_status(resp)


class _PersonasClient:
    """Sub-client for persona-related endpoints.

    Do not instantiate this directly. Access via :attr:`PandoHttpClient.personas`.
    """

    def __init__(self, http: "httpx.AsyncClient", base_url: str) -> None:
        self._http = http
        self._base = base_url

    async def list(self) -> list[str]:
        """List available personas.

        Returns:
            A list of persona name strings.
        """
        resp = await self._http.get(f"{self._base}/api/v1/personas")
        _raise_for_status(resp)
        data = resp.json()
        items = data if isinstance(data, list) else data.get("personas", [])
        return [p if isinstance(p, str) else p.get("name", "") for p in items]

    async def get_active(self) -> str:
        """Get the currently active persona.

        Returns:
            The active persona name.
        """
        resp = await self._http.get(f"{self._base}/api/v1/personas/active")
        _raise_for_status(resp)
        data = resp.json()
        if isinstance(data, str):
            return data
        return data.get("active", "") or data.get("name", "")

    async def set_active(self, name: str) -> None:
        """Set the active persona.

        Args:
            name: Persona name to activate.
        """
        resp = await self._http.put(
            f"{self._base}/api/v1/personas/active",
            json={"name": name},
        )
        _raise_for_status(resp)


# ---------------------------------------------------------------------------
# Main HTTP client
# ---------------------------------------------------------------------------


class PandoHttpClient:
    """Async HTTP client for the pando REST API.

    This client communicates with a running ``pando serve`` process via HTTP.
    It provides access to sessions, models, and personas through sub-clients.

    Requires the ``httpx`` optional dependency::

        pip install pando-sdk[http]

    Usage as an async context manager::

        async with PandoHttpClient(base_url="http://localhost:8765") as client:
            sessions = await client.sessions.list()

    Manual lifecycle::

        client = PandoHttpClient(base_url="http://localhost:8765")
        await client.connect()
        # ... use client ...
        await client.close()

    Args:
        base_url: Base URL of the pando server (e.g. ``"http://localhost:8765"``).
        verify_ssl: Whether to verify SSL certificates. Set to ``False`` for
            self-signed certificates in development.
        timeout: Default request timeout in seconds.
        token: Optional API token for authenticated endpoints.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8765",
        verify_ssl: bool = True,
        timeout: float = 60.0,
        token: str | None = None,
    ) -> None:
        _require_httpx()
        self._base_url = base_url.rstrip("/")
        self._verify_ssl = verify_ssl
        self._timeout = timeout
        self._token = token
        self._client: httpx.AsyncClient | None = None

        # Sub-clients (initialized after connect)
        self._sessions: _SessionsClient | None = None
        self._models: _ModelsClient | None = None
        self._personas: _PersonasClient | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Initialize the underlying HTTP client."""
        headers: dict[str, str] = {}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"

        self._client = httpx.AsyncClient(
            verify=self._verify_ssl,
            timeout=self._timeout,
            headers=headers,
        )
        self._sessions = _SessionsClient(self._client, self._base_url)
        self._models = _ModelsClient(self._client, self._base_url)
        self._personas = _PersonasClient(self._client, self._base_url)

    async def close(self) -> None:
        """Close the underlying HTTP client and release resources."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "PandoHttpClient":
        await self.connect()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Sub-client properties
    # ------------------------------------------------------------------

    @property
    def sessions(self) -> _SessionsClient:
        """Access session-related endpoints."""
        if self._sessions is None:
            raise PandoConnectionError(
                "PandoHttpClient is not connected. Use it as an async context manager "
                "or call connect() first."
            )
        return self._sessions

    @property
    def models(self) -> _ModelsClient:
        """Access model-related endpoints."""
        if self._models is None:
            raise PandoConnectionError(
                "PandoHttpClient is not connected. Use it as an async context manager "
                "or call connect() first."
            )
        return self._models

    @property
    def personas(self) -> _PersonasClient:
        """Access persona-related endpoints."""
        if self._personas is None:
            raise PandoConnectionError(
                "PandoHttpClient is not connected. Use it as an async context manager "
                "or call connect() first."
            )
        return self._personas

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    async def health(self) -> dict:
        """Query the server health endpoint.

        Returns:
            A dictionary with health status information.

        Raises:
            :class:`~pando.exceptions.PandoConnectionError`: If the server is
                unreachable.
        """
        if self._client is None:
            raise PandoConnectionError("Not connected")
        try:
            resp = await self._client.get(f"{self._base_url}/health")
            _raise_for_status(resp)
            return resp.json()
        except httpx.ConnectError as exc:
            raise PandoConnectionError(
                f"Cannot connect to pando server at {self._base_url}: {exc}"
            ) from exc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _raise_for_status(response: "httpx.Response") -> None:
    """Raise a :class:`~pando.exceptions.PandoError` for non-2xx responses."""
    if response.status_code < 200 or response.status_code >= 300:
        try:
            detail = response.json()
            if isinstance(detail, dict):
                msg = detail.get("error") or detail.get("message") or str(detail)
            else:
                msg = str(detail)
        except Exception:
            msg = response.text[:200]
        raise PandoError(f"HTTP {response.status_code}: {msg}")


def _parse_session_info(data: dict) -> SessionInfo:
    """Parse a raw session dict into a :class:`~pando.types.SessionInfo`."""
    return SessionInfo(
        session_id=data.get("id") or data.get("sessionId") or "",
        title=data.get("title") or "",
        updated_at=data.get("updatedAt") or data.get("updated_at") or "",
        cwd=data.get("cwd") or "",
    )
