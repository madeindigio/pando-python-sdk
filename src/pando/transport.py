"""JSON-RPC 2.0 transport over pando's ACP stdio interface.

This module implements :class:`_JsonRpcTransport` which:

* Spawns ``pando acp`` (or ``pando --acp-server``) as an asyncio subprocess.
* Writes newline-delimited JSON requests to the process stdin.
* Reads newline-delimited JSON responses and notifications from stdout.
* Correlates responses to outstanding requests by ``id``.
* Dispatches ``agent/event`` notifications to per-session asyncio queues.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
from asyncio import subprocess as aiosubprocess
from typing import Any, Callable

from pando.events import AgentEvent, parse_agent_event
from pando.exceptions import (
    PandoBinaryNotFoundError,
    PandoConnectionError,
    PandoRPCError,
)

logger = logging.getLogger(__name__)

# Sentinel placed into a session queue when the transport terminates so that
# any async-for loop waiting on the queue will terminate cleanly.
_QUEUE_SENTINEL = object()

# Maximum line size when reading stdout (10 MiB).
_MAX_LINE_SIZE = 10 * 1024 * 1024


def _find_pando_binary(pando_path: str | None = None) -> str:
    """Locate the pando binary.

    Resolution order:
    1. ``pando_path`` argument.
    2. ``PANDO_PATH`` environment variable.
    3. ``pando`` found via :func:`shutil.which`.

    Args:
        pando_path: Explicit path to the pando binary.

    Returns:
        Absolute path to the pando binary.

    Raises:
        :class:`~pando.exceptions.PandoBinaryNotFoundError`: If pando cannot be found.
    """
    # 1. Explicit argument
    if pando_path:
        if os.path.isfile(pando_path) and os.access(pando_path, os.X_OK):
            return pando_path
        raise PandoBinaryNotFoundError(
            f"pando binary not found or not executable at: {pando_path}"
        )

    # 2. Environment variable
    env_path = os.environ.get("PANDO_PATH")
    if env_path:
        if os.path.isfile(env_path) and os.access(env_path, os.X_OK):
            return env_path
        raise PandoBinaryNotFoundError(
            f"PANDO_PATH is set but binary not found or not executable at: {env_path}"
        )

    # 3. PATH lookup
    found = shutil.which("pando")
    if found:
        return found

    raise PandoBinaryNotFoundError(
        "Could not find 'pando' binary. "
        "Install pando and ensure it is on your PATH, set the PANDO_PATH "
        "environment variable, or pass pando_path= to PandoAgent / PandoClient."
    )


class _JsonRpcTransport:
    """Asyncio JSON-RPC 2.0 transport over the pando stdio ACP interface.

    This class manages the lifecycle of a pando subprocess running in ACP
    server mode and provides a coroutine-based API for sending requests and
    receiving responses / notifications.

    Args:
        pando_path: Path to the pando binary.
        cwd: Working directory for the pando subprocess.
        extra_args: Additional CLI arguments passed to pando.
        on_event: Optional coroutine function called for every :class:`~pando.events.AgentEvent`
            notification received from the server (beyond session queues).
    """

    def __init__(
        self,
        pando_path: str,
        cwd: str | None = None,
        extra_args: list[str] | None = None,
        on_event: Callable[[AgentEvent], Any] | None = None,
    ) -> None:
        self._pando_path = pando_path
        self._cwd = cwd
        self._extra_args: list[str] = extra_args or []
        self._on_event = on_event

        self._process: aiosubprocess.Process | None = None
        self._reader_task: asyncio.Task | None = None

        # Maps request id -> Future[dict] for pending RPC calls.
        self._pending: dict[int, asyncio.Future[dict]] = {}

        # Maps sessionId -> Queue[AgentEvent | sentinel] for streaming events.
        self._session_queues: dict[str, asyncio.Queue] = {}
        self._session_queues_lock = asyncio.Lock()

        # Auto-incrementing request ID counter.
        self._next_id = 1
        self._id_lock = asyncio.Lock()

        self._closed = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Spawn the pando subprocess and start the reader loop.

        Raises:
            :class:`~pando.exceptions.PandoConnectionError`: If the subprocess
                cannot be started.
        """
        cmd = [self._pando_path, "acp"] + self._extra_args
        logger.debug("Spawning pando: %s (cwd=%s)", cmd, self._cwd)

        try:
            self._process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=aiosubprocess.PIPE,
                stdout=aiosubprocess.PIPE,
                stderr=aiosubprocess.DEVNULL,
                cwd=self._cwd,
                limit=_MAX_LINE_SIZE,
            )
        except (OSError, FileNotFoundError) as exc:
            raise PandoConnectionError(f"Failed to start pando process: {exc}") from exc

        self._reader_task = asyncio.ensure_future(self._reader_loop())
        logger.debug("pando subprocess started (pid=%d)", self._process.pid)

    async def disconnect(self) -> None:
        """Terminate the pando subprocess and clean up resources."""
        if self._closed:
            return
        self._closed = True

        # Cancel the reader loop first.
        if self._reader_task and not self._reader_task.done():
            self._reader_task.cancel()
            try:
                await self._reader_task
            except (asyncio.CancelledError, Exception):
                pass

        # Terminate the subprocess.
        if self._process and self._process.returncode is None:
            try:
                self._process.terminate()
                try:
                    await asyncio.wait_for(self._process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    self._process.kill()
                    await self._process.wait()
            except Exception as exc:
                logger.debug("Error terminating pando process: %s", exc)

        # Fail all pending requests.
        error = PandoConnectionError("pando process disconnected")
        for future in self._pending.values():
            if not future.done():
                future.set_exception(error)
        self._pending.clear()

        # Signal all session queues that the transport is done.
        for q in self._session_queues.values():
            await q.put(_QUEUE_SENTINEL)
        self._session_queues.clear()

        logger.debug("pando transport disconnected")

    def is_connected(self) -> bool:
        """Return True if the subprocess is alive."""
        return (
            self._process is not None
            and self._process.returncode is None
            and not self._closed
        )

    # ------------------------------------------------------------------
    # RPC
    # ------------------------------------------------------------------

    async def _next_request_id(self) -> int:
        async with self._id_lock:
            rid = self._next_id
            self._next_id += 1
            return rid

    async def call(self, method: str, params: dict | None = None) -> dict:
        """Send a JSON-RPC request and await the response.

        Args:
            method: JSON-RPC method name.
            params: Optional params object.

        Returns:
            The ``result`` field from the JSON-RPC response.

        Raises:
            :class:`~pando.exceptions.PandoConnectionError`: If the transport is
                not connected.
            :class:`~pando.exceptions.PandoRPCError`: If the server returns an
                error response.
        """
        if not self.is_connected():
            raise PandoConnectionError("Not connected to pando. Call connect() first.")

        rid = await self._next_request_id()
        request: dict = {"jsonrpc": "2.0", "id": rid, "method": method}
        if params is not None:
            request["params"] = params

        loop = asyncio.get_event_loop()
        future: asyncio.Future[dict] = loop.create_future()
        self._pending[rid] = future

        line = json.dumps(request) + "\n"
        logger.debug("RPC -> %s", line.rstrip())

        assert self._process and self._process.stdin
        try:
            self._process.stdin.write(line.encode())
            await self._process.stdin.drain()
        except (OSError, BrokenPipeError) as exc:
            self._pending.pop(rid, None)
            raise PandoConnectionError(f"Failed to write to pando stdin: {exc}") from exc

        result = await future
        return result

    async def notify(self, method: str, params: dict | None = None) -> None:
        """Send a JSON-RPC notification (no response expected).

        Args:
            method: JSON-RPC method name.
            params: Optional params object.
        """
        if not self.is_connected():
            raise PandoConnectionError("Not connected to pando. Call connect() first.")

        request: dict = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            request["params"] = params

        line = json.dumps(request) + "\n"
        logger.debug("NOTIFY -> %s", line.rstrip())

        assert self._process and self._process.stdin
        try:
            self._process.stdin.write(line.encode())
            await self._process.stdin.drain()
        except (OSError, BrokenPipeError) as exc:
            raise PandoConnectionError(f"Failed to write notification to pando stdin: {exc}") from exc

    # ------------------------------------------------------------------
    # Session event queues
    # ------------------------------------------------------------------

    async def get_session_queue(self, session_id: str) -> asyncio.Queue:
        """Get or create an :class:`asyncio.Queue` for the given session.

        All ``agent/event`` notifications for ``session_id`` are placed into
        this queue.

        Args:
            session_id: The session identifier.

        Returns:
            The asyncio Queue for this session's events.
        """
        async with self._session_queues_lock:
            if session_id not in self._session_queues:
                self._session_queues[session_id] = asyncio.Queue()
            return self._session_queues[session_id]

    async def remove_session_queue(self, session_id: str) -> None:
        """Remove the event queue for a session (e.g. when the session closes)."""
        async with self._session_queues_lock:
            self._session_queues.pop(session_id, None)

    # ------------------------------------------------------------------
    # Reader loop
    # ------------------------------------------------------------------

    async def _reader_loop(self) -> None:
        """Read newline-delimited JSON from the process stdout and dispatch messages."""
        assert self._process and self._process.stdout

        try:
            while True:
                try:
                    raw = await self._process.stdout.readline()
                except asyncio.LimitOverrunError:
                    logger.warning("pando output line exceeded max buffer size; skipping")
                    await self._process.stdout.read(n=_MAX_LINE_SIZE)
                    continue

                if not raw:
                    # EOF — the process has closed its stdout.
                    logger.debug("pando stdout closed (EOF)")
                    break

                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue

                logger.debug("RPC <- %s", line)
                await self._dispatch_line(line)

        except asyncio.CancelledError:
            logger.debug("Reader loop cancelled")
            raise
        except Exception as exc:
            logger.warning("Unexpected error in reader loop: %s", exc)
        finally:
            # The process has gone away — clean up pending futures and queues.
            if not self._closed:
                error = PandoConnectionError("pando process exited unexpectedly")
                for future in self._pending.values():
                    if not future.done():
                        future.set_exception(error)
                self._pending.clear()
                async with self._session_queues_lock:
                    for q in self._session_queues.values():
                        await q.put(_QUEUE_SENTINEL)
                logger.debug("Reader loop finished; all pending requests failed")

    async def _dispatch_line(self, line: str) -> None:
        """Parse a single JSON line and dispatch to the appropriate handler."""
        try:
            msg = json.loads(line)
        except json.JSONDecodeError as exc:
            logger.warning("Invalid JSON from pando: %s (%s)", line[:200], exc)
            return

        if not isinstance(msg, dict):
            logger.warning("Unexpected JSON type from pando: %r", type(msg))
            return

        method = msg.get("method")
        msg_id = msg.get("id")

        if method == "agent/event":
            # Async notification — route to session queues.
            await self._handle_agent_event(msg)
        elif msg_id is not None:
            # Response to an outstanding request.
            await self._handle_response(msg)
        else:
            logger.debug("Unhandled message from pando: %s", line[:200])

    async def _handle_response(self, msg: dict) -> None:
        """Resolve or reject the future associated with this response."""
        rid = msg.get("id")
        future = self._pending.pop(rid, None)
        if future is None:
            logger.debug("Received response for unknown request id=%r", rid)
            return

        if future.done():
            return

        if "error" in msg:
            err = msg["error"]
            code = err.get("code", -32000)
            message = err.get("message", "Unknown RPC error")
            future.set_exception(PandoRPCError(code, message))
        else:
            future.set_result(msg.get("result") or {})

    async def _handle_agent_event(self, msg: dict) -> None:
        """Parse and dispatch an ``agent/event`` notification."""
        params = msg.get("params", {})
        if not isinstance(params, dict):
            logger.warning("agent/event params is not a dict: %r", params)
            return

        try:
            event = parse_agent_event(params)
        except Exception as exc:
            logger.warning("Failed to parse agent/event: %s", exc)
            return

        # Dispatch to session-specific queue if one exists.
        async with self._session_queues_lock:
            queue = self._session_queues.get(event.session_id)

        if queue is not None:
            await queue.put(event)

        # Also invoke the global on_event callback if provided.
        if self._on_event is not None:
            try:
                result = self._on_event(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as exc:
                logger.warning("on_event callback raised an error: %s", exc)


# Re-export the sentinel so session.py can use it.
QUEUE_SENTINEL = _QUEUE_SENTINEL
