"""Shared type definitions for the Pando SDK."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RunResult:
    """Result of a single non-interactive pando run.

    Attributes:
        response: The text response from the assistant.
        session_id: The session ID used for this run (may be empty for ephemeral runs).
        raw: The raw parsed JSON dictionary returned by pando, if ``-f json`` was used.
    """

    response: str
    session_id: str = ""
    raw: dict = field(default_factory=dict)


@dataclass
class PermissionRequest:
    """A tool permission request that requires user approval.

    When pando asks for confirmation before executing a potentially destructive
    tool call, a ``PermissionRequest`` is emitted. The ``on_tool_permission``
    callback on :class:`~pando.agent.PandoAgent` receives this object and must
    return ``True`` to allow the action or ``False`` to deny it.

    Attributes:
        session_id: The session that issued this permission request.
        tool_name: Name of the tool requesting permission (e.g. ``"bash"``).
        description: Human-readable description of the action.
        action: The action type (e.g. ``"execute"``, ``"write"``, ``"delete"``).
        path: The file or resource path the tool wants to act on.
        params: Additional tool parameters.
    """

    session_id: str
    tool_name: str
    description: str
    action: str
    path: str
    params: dict = field(default_factory=dict)


@dataclass
class SessionInfo:
    """Metadata about a pando session.

    Attributes:
        session_id: Unique session identifier (UUID).
        title: Human-readable session title.
        updated_at: ISO-8601 timestamp of the last update (may be empty).
        cwd: Working directory associated with this session.
    """

    session_id: str
    title: str = ""
    updated_at: str = ""
    cwd: str = ""


@dataclass
class ModelInfo:
    """Information about an available model.

    Attributes:
        id: The model identifier (e.g. ``"claude-sonnet-4-6"``).
        name: Human-readable model name.
        provider: Provider name (e.g. ``"claude"``, ``"copilot"``).
    """

    id: str
    name: str = ""
    provider: str = ""


@dataclass
class MessageChunk:
    """A single SSE chunk returned from the HTTP REST streaming endpoint.

    Attributes:
        delta: Incremental text content, if any.
        type: Event type string (e.g. ``"content_delta"``, ``"response"``).
        raw: The full raw event data dictionary.
    """

    delta: str | None
    type: str
    raw: dict = field(default_factory=dict)
