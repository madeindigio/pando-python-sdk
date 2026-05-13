"""PandoAgent — ACP stdio JSON-RPC mode client.

This module provides :class:`PandoAgent`, which communicates with a pando
process running in ACP server mode (``pando acp``) via newline-delimited
JSON-RPC 2.0 over stdin/stdout.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Awaitable, Callable
from typing import Any

from pando.exceptions import PandoConnectionError, PandoRPCError
from pando.session import PandoSession
from pando.transport import _JsonRpcTransport, _find_pando_binary
from pando.types import PermissionRequest, SessionInfo

logger = logging.getLogger(__name__)


class PandoAgent:
    """Client for pando's ACP stdio JSON-RPC interface.

    :class:`PandoAgent` spawns a ``pando acp`` subprocess and communicates
    with it via newline-delimited JSON-RPC 2.0 over stdin/stdout. It manages
    multiple concurrent sessions and provides both async and sync APIs.

    Usage as an **async** context manager::

        async with PandoAgent(cwd="/path/to/project") as agent:
            session = await agent.create_session("My task")
            async for event in session.send("Refactor the auth module"):
                if event.type == "content_delta":
                    print(event.delta, end="", flush=True)

    Usage as a **sync** context manager::

        with PandoAgent(cwd="/path/to/project") as agent:
            session = agent.create_session_sync("My task")
            for event in session.send_sync("Fix all lint errors"):
                if event.type == "content_delta":
                    print(event.delta, end="")

    Manual lifecycle::

        agent = PandoAgent(cwd="/path/to/project")
        await agent.connect()
        # ... use agent ...
        await agent.disconnect()

    Args:
        cwd: Working directory for the pando subprocess. Defaults to the
            current working directory.
        model: Optional model override (e.g. ``"claude-sonnet-4-6"``).
        persona: Optional initial persona to set after connecting.
        pando_path: Explicit path to the pando binary. Falls back to
            ``PANDO_PATH`` env var and then PATH lookup.
        on_tool_permission: Optional callback invoked when the agent emits a
            tool permission request. The callback receives a
            :class:`~pando.types.PermissionRequest` and must return ``True``
            to approve or ``False`` to deny. The callback may be a regular
            function or a coroutine function.
        debug: If ``True``, passes ``--debug`` to the pando subprocess.
    """

    def __init__(
        self,
        cwd: str | None = None,
        model: str | None = None,
        persona: str | None = None,
        pando_path: str | None = None,
        on_tool_permission: Callable[[PermissionRequest], bool | Awaitable[bool]] | None = None,
        debug: bool = False,
    ) -> None:
        self._cwd = cwd
        self._model = model
        self._persona = persona
        self._pando_path: str | None = None
        self._raw_pando_path = pando_path
        self._on_tool_permission = on_tool_permission
        self._debug = debug

        self._transport: _JsonRpcTransport | None = None
        self._connected = False

        # Sync mode: a dedicated thread runs an event loop.
        self._sync_thread: threading.Thread | None = None
        self._sync_loop: asyncio.AbstractEventLoop | None = None
        self._sync_mode = False

    # ------------------------------------------------------------------
    # Async context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "PandoAgent":
        await self.connect()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.disconnect()

    # ------------------------------------------------------------------
    # Sync context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "PandoAgent":
        """Start the agent in synchronous mode.

        This creates a dedicated background thread with an event loop so that
        async operations can be driven from the calling (sync) thread.
        """
        self._sync_mode = True
        ready = threading.Event()

        def _run_loop() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._sync_loop = loop
            ready.set()
            loop.run_forever()

        self._sync_thread = threading.Thread(target=_run_loop, daemon=True, name="pando-sync-loop")
        self._sync_thread.start()
        ready.wait(timeout=5)

        # Connect the agent on the background loop.
        future = asyncio.run_coroutine_threadsafe(self.connect(), self._sync_loop)
        future.result()  # propagate exceptions

        return self

    def __exit__(self, *_: Any) -> None:
        """Stop the agent and shut down the background event loop."""
        if self._sync_loop:
            future = asyncio.run_coroutine_threadsafe(self.disconnect(), self._sync_loop)
            future.result()
            self._sync_loop.call_soon_threadsafe(self._sync_loop.stop)

        if self._sync_thread:
            self._sync_thread.join(timeout=5)

        self._sync_loop = None
        self._sync_thread = None
        self._sync_mode = False

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Connect to the pando subprocess (ACP stdio mode).

        This method is called automatically when using the context manager.

        Raises:
            :class:`~pando.exceptions.PandoBinaryNotFoundError`: If pando cannot be found.
            :class:`~pando.exceptions.PandoConnectionError`: If the subprocess fails to start.
        """
        if self._connected:
            return

        self._pando_path = _find_pando_binary(self._raw_pando_path)

        extra_args: list[str] = []
        if self._debug:
            extra_args.append("--debug")

        self._transport = _JsonRpcTransport(
            pando_path=self._pando_path,
            cwd=self._cwd,
            extra_args=extra_args,
        )
        await self._transport.connect()
        self._connected = True
        logger.debug("PandoAgent connected (cwd=%s)", self._cwd)

        # Apply initial model if specified.
        if self._model:
            try:
                await self.set_model(self._model)
            except Exception as exc:
                logger.warning("Failed to set initial model %r: %s", self._model, exc)

        # Apply initial persona if specified.
        if self._persona:
            try:
                await self.set_persona(self._persona)
            except Exception as exc:
                logger.warning("Failed to set initial persona %r: %s", self._persona, exc)

    async def disconnect(self) -> None:
        """Disconnect from the pando subprocess and release resources."""
        if not self._connected:
            return
        self._connected = False
        if self._transport:
            await self._transport.disconnect()
            self._transport = None
        logger.debug("PandoAgent disconnected")

    def _require_transport(self) -> _JsonRpcTransport:
        if self._transport is None or not self._connected:
            raise PandoConnectionError(
                "PandoAgent is not connected. Use it as a context manager or call connect() first."
            )
        return self._transport

    # ------------------------------------------------------------------
    # Session management (async)
    # ------------------------------------------------------------------

    async def create_session(self, title: str = "") -> PandoSession:
        """Create a new conversation session on the pando server.

        Args:
            title: Optional human-readable title for the session.

        Returns:
            A :class:`~pando.session.PandoSession` ready to accept prompts.

        Raises:
            :class:`~pando.exceptions.PandoConnectionError`: If not connected.
            :class:`~pando.exceptions.PandoRPCError`: If the server returns an error.
        """
        transport = self._require_transport()

        params: dict = {}
        if self._cwd:
            params["cwd"] = self._cwd

        result = await transport.call("session/create", params or None)
        session_id = result.get("sessionId") or result.get("session_id", "")
        if not session_id:
            raise PandoRPCError(-32603, "session/create returned no sessionId")

        # Pre-register the session queue before any events could arrive.
        await transport.get_session_queue(session_id)

        logger.debug("Session created: %s (title=%r)", session_id, title)
        return PandoSession(
            session_id=session_id,
            transport=transport,
            sync_loop=self._sync_loop,
        )

    async def load_session(self, session_id: str) -> PandoSession:
        """Load an existing session by ID.

        This replays the session history to the client and registers the
        session for future prompts.

        Args:
            session_id: The session identifier to load.

        Returns:
            A :class:`~pando.session.PandoSession`.

        Raises:
            :class:`~pando.exceptions.PandoRPCError`: If the session is not found.
        """
        transport = self._require_transport()
        params: dict = {"sessionId": session_id}
        if self._cwd:
            params["cwd"] = self._cwd

        await transport.call("session/load", params)
        await transport.get_session_queue(session_id)

        logger.debug("Session loaded: %s", session_id)
        return PandoSession(
            session_id=session_id,
            transport=transport,
            sync_loop=self._sync_loop,
        )

    async def list_sessions(self) -> list[SessionInfo]:
        """List all sessions known to the pando server.

        Returns:
            A list of :class:`~pando.types.SessionInfo` objects.
        """
        transport = self._require_transport()
        result = await transport.call("session/list")

        sessions_raw = result.get("sessions", [])
        sessions: list[SessionInfo] = []
        for s in sessions_raw:
            if not isinstance(s, dict):
                continue
            sessions.append(
                SessionInfo(
                    session_id=s.get("sessionId", ""),
                    title=s.get("title") or "",
                    updated_at=s.get("updatedAt") or "",
                    cwd=s.get("cwd") or "",
                )
            )
        return sessions

    # ------------------------------------------------------------------
    # Agent-level settings (async)
    # ------------------------------------------------------------------

    async def set_model(self, model_id: str) -> None:
        """Set the active model for the agent.

        Args:
            model_id: Model identifier (e.g. ``"claude-sonnet-4-6"``).
        """
        transport = self._require_transport()
        await transport.call("model/set", {"modelId": model_id})
        logger.debug("Model set to %r", model_id)

    async def list_models(self) -> list[dict]:
        """List all available models.

        Returns:
            A list of model info dictionaries.
        """
        transport = self._require_transport()
        result = await transport.call("model/list")
        return result.get("models", [])

    async def set_persona(self, name: str) -> None:
        """Set the active persona globally.

        Args:
            name: Persona name (e.g. ``"software-engineer"``, ``"qa"``).
        """
        transport = self._require_transport()
        await transport.call("persona/set", {"name": name})
        logger.debug("Persona set to %r", name)

    async def get_persona(self) -> str:
        """Get the currently active persona name.

        Returns:
            Active persona name, or empty string if none is set.
        """
        transport = self._require_transport()
        result = await transport.call("persona/get")
        return result.get("active", "")

    async def list_personas(self) -> list[str]:
        """List all available personas.

        Returns:
            A list of persona name strings.
        """
        transport = self._require_transport()
        result = await transport.call("persona/list")
        return result.get("personas", [])

    async def list_tools(self) -> list[dict]:
        """List all tools available to the agent.

        Returns:
            A list of tool info dictionaries.
        """
        transport = self._require_transport()
        result = await transport.call("tool/list")
        return result.get("tools", [])

    # ------------------------------------------------------------------
    # Sync API wrappers
    # ------------------------------------------------------------------

    def _sync_call(self, coro: Any) -> Any:
        """Run a coroutine on the sync event loop and return the result."""
        if self._sync_loop is None:
            raise RuntimeError(
                "Sync methods require PandoAgent to be used as a synchronous context manager."
            )
        return asyncio.run_coroutine_threadsafe(coro, self._sync_loop).result()

    def create_session_sync(self, title: str = "") -> PandoSession:
        """Synchronous version of :meth:`create_session`."""
        return self._sync_call(self.create_session(title))

    def load_session_sync(self, session_id: str) -> PandoSession:
        """Synchronous version of :meth:`load_session`."""
        return self._sync_call(self.load_session(session_id))

    def list_sessions_sync(self) -> list[SessionInfo]:
        """Synchronous version of :meth:`list_sessions`."""
        return self._sync_call(self.list_sessions())

    def set_model_sync(self, model_id: str) -> None:
        """Synchronous version of :meth:`set_model`."""
        self._sync_call(self.set_model(model_id))

    def list_models_sync(self) -> list[dict]:
        """Synchronous version of :meth:`list_models`."""
        return self._sync_call(self.list_models())

    def set_persona_sync(self, name: str) -> None:
        """Synchronous version of :meth:`set_persona`."""
        self._sync_call(self.set_persona(name))

    def get_persona_sync(self) -> str:
        """Synchronous version of :meth:`get_persona`."""
        return self._sync_call(self.get_persona())

    def list_personas_sync(self) -> list[str]:
        """Synchronous version of :meth:`list_personas`."""
        return self._sync_call(self.list_personas())

    def list_tools_sync(self) -> list[dict]:
        """Synchronous version of :meth:`list_tools`."""
        return self._sync_call(self.list_tools())
