"""PandoSession — represents a single conversation session in ACP stdio mode."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator, Generator
from typing import TYPE_CHECKING, Any

from pando.events import AgentEvent
from pando.exceptions import PandoConnectionError, PandoRPCError, PandoSessionError
from pando.transport import QUEUE_SENTINEL

if TYPE_CHECKING:
    from pando.transport import _JsonRpcTransport

logger = logging.getLogger(__name__)

# Event types that indicate the completion of a prompt/send call.
_TERMINAL_EVENT_TYPES = frozenset({"response", "error", "summarize"})


class PandoSession:
    """A single conversation session managed by a :class:`~pando.agent.PandoAgent`.

    Obtain a session via :meth:`~pando.agent.PandoAgent.create_session` rather
    than constructing this class directly.

    Args:
        session_id: The session identifier returned by the pando server.
        transport: The underlying JSON-RPC transport.
        sync_loop: Event loop used by the sync wrapper (set by the agent).
    """

    def __init__(
        self,
        session_id: str,
        transport: _JsonRpcTransport,
        sync_loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        self._id = session_id
        self._transport = transport
        self._sync_loop = sync_loop

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def id(self) -> str:
        """The unique session identifier."""
        return self._id

    # ------------------------------------------------------------------
    # Async API
    # ------------------------------------------------------------------

    async def send(self, prompt: str) -> AsyncGenerator[AgentEvent, None]:
        """Send a prompt and stream the resulting agent events.

        This is an async generator. Iterate over it to receive events as they
        arrive from the pando process. Iteration stops after a ``response``
        or ``error`` event is received.

        Example::

            async for event in session.send("Refactor the auth module"):
                match event.type:
                    case "content_delta":
                        print(event.delta, end="", flush=True)
                    case "response":
                        print()  # newline after response

        Args:
            prompt: The user message to send to the agent.

        Yields:
            :class:`~pando.events.AgentEvent` instances as they are received.

        Raises:
            :class:`~pando.exceptions.PandoConnectionError`: If the pando process
                exits unexpectedly.
            :class:`~pando.exceptions.PandoRPCError`: If the server returns an
                error response to the prompt/send RPC.
            :class:`~pando.exceptions.PandoSessionError`: If the session is not
                found on the server.
        """
        queue = await self._transport.get_session_queue(self._id)

        # Send the prompt — the response arrives asynchronously as agent/event notifications.
        try:
            await self._transport.call(
                "prompt/send",
                {"sessionId": self._id, "prompt": [{"type": "text", "text": prompt}]},
            )
        except PandoRPCError as exc:
            raise PandoSessionError(f"Failed to send prompt to session {self._id}: {exc}") from exc

        # Stream events until we hit a terminal event.
        while True:
            try:
                item = await queue.get()
            except asyncio.CancelledError:
                raise PandoConnectionError("Session event stream was cancelled")

            if item is QUEUE_SENTINEL:
                raise PandoConnectionError("pando process exited while waiting for events")

            event: AgentEvent = item
            yield event

            if event.type in _TERMINAL_EVENT_TYPES:
                break

    async def ask(self, prompt: str) -> str:
        """Send a prompt and collect the full response text.

        This is a convenience wrapper around :meth:`send` that accumulates all
        ``content_delta`` events and returns the final text.

        Args:
            prompt: The user message to send.

        Returns:
            The full assistant response text.

        Raises:
            :class:`~pando.exceptions.PandoConnectionError`: If the connection drops.
            :class:`~pando.exceptions.PandoSessionError`: On session errors.
        """
        parts: list[str] = []
        async for event in self.send(prompt):
            if event.type == "content_delta" and event.delta:
                parts.append(event.delta)
            elif event.type == "error" and event.error:
                raise PandoSessionError(f"Agent error: {event.error}")
        return "".join(parts)

    async def set_persona(self, name: str) -> None:
        """Set the persona for this session.

        Args:
            name: Persona name (e.g. ``"software-engineer"``, ``"qa"``).

        Raises:
            :class:`~pando.exceptions.PandoRPCError`: If the persona is unknown.
        """
        await self._transport.call(
            "persona/set_session",
            {"sessionId": self._id, "name": name},
        )
        logger.debug("Session %s persona set to %r", self._id, name)

    async def set_model(self, model_id: str) -> None:
        """Set the model for this session.

        Args:
            model_id: Model identifier (e.g. ``"claude-sonnet-4-6"``).

        Raises:
            :class:`~pando.exceptions.PandoRPCError`: If the model is unknown.
        """
        await self._transport.call(
            "session/set_model",
            {"sessionId": self._id, "modelId": model_id},
        )
        logger.debug("Session %s model set to %r", self._id, model_id)

    async def cancel(self) -> None:
        """Cancel any in-progress task for this session.

        Sends a ``cancel`` JSON-RPC notification. This is fire-and-forget; the
        agent may still emit a few more events before stopping.
        """
        try:
            await self._transport.notify(
                "cancel",
                {"sessionId": self._id},
            )
            logger.debug("Cancel notification sent for session %s", self._id)
        except PandoConnectionError:
            # Already disconnected — ignore.
            pass

    async def close(self) -> None:
        """Close this session and clean up the event queue."""
        try:
            await self._transport.call(
                "session/close",
                {"sessionId": self._id},
            )
        except (PandoConnectionError, PandoRPCError) as exc:
            logger.debug("Error closing session %s: %s", self._id, exc)
        finally:
            await self._transport.remove_session_queue(self._id)

    # ------------------------------------------------------------------
    # Sync API
    # ------------------------------------------------------------------

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        """Return the sync event loop, raising if not available."""
        if self._sync_loop is None:
            raise RuntimeError(
                "Sync methods are only available when using PandoAgent as a synchronous "
                "context manager (with PandoAgent(...) as agent: ...)."
            )
        return self._sync_loop

    def send_sync(self, prompt: str) -> Generator[AgentEvent, None, None]:
        """Synchronous version of :meth:`send`.

        Yields events one at a time by driving the asyncio event loop from the
        calling thread. This method is available when using :class:`~pando.agent.PandoAgent`
        as a synchronous context manager.

        Example::

            with PandoAgent(cwd="/my/project") as agent:
                session = agent.create_session_sync("My task")
                for event in session.send_sync("Fix lint errors"):
                    if event.type == "content_delta":
                        print(event.delta, end="")

        Args:
            prompt: The user message to send.

        Yields:
            :class:`~pando.events.AgentEvent` instances.
        """
        loop = self._get_loop()

        # We cannot use a regular generator that suspends across run_until_complete
        # calls with a running loop. Instead we collect events into a sync queue.
        import queue as _queue

        sync_q: _queue.Queue[Any] = _queue.Queue()
        _DONE = object()

        async def _producer() -> None:
            try:
                async for event in self.send(prompt):
                    sync_q.put(event)
            except Exception as exc:
                sync_q.put(exc)
            finally:
                sync_q.put(_DONE)

        # Schedule the producer coroutine on the background loop.
        asyncio.run_coroutine_threadsafe(_producer(), loop)

        while True:
            item = sync_q.get()
            if item is _DONE:
                break
            if isinstance(item, BaseException):
                raise item
            yield item

    def ask_sync(self, prompt: str) -> str:
        """Synchronous version of :meth:`ask`.

        Args:
            prompt: The user message.

        Returns:
            Full response text.
        """
        loop = self._get_loop()
        return asyncio.run_coroutine_threadsafe(self.ask(prompt), loop).result()

    def set_persona_sync(self, name: str) -> None:
        """Synchronous version of :meth:`set_persona`."""
        loop = self._get_loop()
        asyncio.run_coroutine_threadsafe(self.set_persona(name), loop).result()

    def cancel_sync(self) -> None:
        """Synchronous version of :meth:`cancel`."""
        loop = self._get_loop()
        asyncio.run_coroutine_threadsafe(self.cancel(), loop).result()

    def close_sync(self) -> None:
        """Synchronous version of :meth:`close`."""
        loop = self._get_loop()
        asyncio.run_coroutine_threadsafe(self.close(), loop).result()
