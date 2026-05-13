"""Agent event types and parsing utilities for the Pando SDK."""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Event dataclass
# ---------------------------------------------------------------------------


@dataclass
class AgentEvent:
    """A single event emitted by a pando agent session.

    Events are delivered as JSON-RPC 2.0 notifications with method
    ``agent/event``. Each event has a ``type`` field that determines which
    other fields are populated.

    Event types:
        ``content_delta``
            Incremental text content from the assistant. Read :attr:`delta`.
        ``thinking_delta``
            Incremental reasoning/thinking text. Read :attr:`delta`.
        ``tool_call``
            The agent is about to invoke a tool. Read :attr:`tool_call`.
        ``tool_result``
            A tool has returned a result. Read :attr:`tool_result`.
        ``response``
            The agent has finished generating a complete message. Read :attr:`message`.
        ``error``
            An error occurred. Read :attr:`error`.
        ``summarize``
            The session is being summarized (context window management).

    Attributes:
        type: Event type string (see above).
        session_id: The session that emitted this event.
        delta: Incremental text, populated for ``content_delta`` and ``thinking_delta``.
        tool_call: Tool call data dict, populated for ``tool_call``.
        tool_result: Tool result data dict, populated for ``tool_result``.
        message: Full assistant message dict, populated for ``response``.
        error: Error message string, populated for ``error``.
    """

    type: str
    session_id: str
    delta: str | None = None
    tool_call: dict | None = None
    tool_result: dict | None = None
    message: dict | None = None
    error: str | None = None
    # Extra fields from raw params not captured above
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def parse_agent_event(params: dict) -> AgentEvent:
    """Parse an ``agent/event`` notification params dict into an :class:`AgentEvent`.

    Args:
        params: The ``params`` value from a JSON-RPC ``agent/event`` notification.

    Returns:
        A populated :class:`AgentEvent` instance.

    Raises:
        :class:`~pando.exceptions.PandoError`: If the params dict is missing the
            ``type`` field.
    """
    from pando.exceptions import PandoError

    event_type = params.get("type")
    if not event_type:
        raise PandoError(f"agent/event notification missing 'type' field: {params!r}")

    session_id = params.get("sessionId", "")

    # Collect known optional fields
    delta = params.get("delta")
    tool_call = params.get("toolCall")
    tool_result = params.get("toolResult")
    message = params.get("message")
    error = params.get("error")

    # Anything else goes into extra
    known_keys = {"type", "sessionId", "delta", "toolCall", "toolResult", "message", "error"}
    extra = {k: v for k, v in params.items() if k not in known_keys}

    return AgentEvent(
        type=event_type,
        session_id=session_id,
        delta=delta,
        tool_call=tool_call,
        tool_result=tool_result,
        message=message,
        error=error,
        extra=extra,
    )
