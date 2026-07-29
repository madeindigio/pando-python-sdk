"""AG-UI protocol types as implemented by Pando's ``internal/agui`` adapter.

These mirror the Go structs one-to-one (camelCase JSON on the wire,
SCREAMING_SNAKE event types). They are declared here rather than pulled from a
third-party AG-UI package so this subpackage stays dependency-free: a caller
that only wants to stream events should not have to install anything.

The dataclasses are thin: every one of them round-trips through
:func:`to_wire` / ``from_wire`` and keeps unknown fields, so a newer adapter
never loses data on the way through the SDK.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "AGUI_EVENT_TYPES",
    "AguiEvent",
    "AguiMessage",
    "AguiToolCall",
    "AguiTool",
    "AguiContext",
    "RunAgentInput",
    "JsonPatchOperation",
    "PandoModelState",
    "PandoTokenUsageState",
    "PandoFileState",
    "PandoSubAgentState",
    "PandoState",
    "AguiCapabilities",
    "AguiAgentDescriptor",
    "AguiInfo",
    "to_wire",
]


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

#: Every event type the adapter can emit.
AGUI_EVENT_TYPES: tuple[str, ...] = (
    "RUN_STARTED",
    "RUN_FINISHED",
    "RUN_ERROR",
    "STEP_STARTED",
    "STEP_FINISHED",
    "TEXT_MESSAGE_START",
    "TEXT_MESSAGE_CONTENT",
    "TEXT_MESSAGE_END",
    "TEXT_MESSAGE_CHUNK",
    "TOOL_CALL_START",
    "TOOL_CALL_ARGS",
    "TOOL_CALL_END",
    "TOOL_CALL_RESULT",
    "STATE_SNAPSHOT",
    "STATE_DELTA",
    "MESSAGES_SNAPSHOT",
    "REASONING_START",
    "REASONING_MESSAGE_START",
    "REASONING_MESSAGE_CONTENT",
    "REASONING_MESSAGE_END",
    "REASONING_END",
    "ACTIVITY_SNAPSHOT",
    "ACTIVITY_DELTA",
    "CUSTOM",
    "RAW",
)

#: Outcomes of a run. ``interrupt`` means the agent called a frontend tool: the
#: run is suspended and the next request on the thread must carry the tool
#: result to resume it.
RUN_OUTCOMES: tuple[str, ...] = ("success", "interrupt")


def _camel(name: str) -> str:
    """Converts a snake_case attribute name to its camelCase wire name."""
    head, *rest = name.split("_")
    return head + "".join(part[:1].upper() + part[1:] for part in rest)


@dataclass(frozen=True)
class AguiEvent:
    """One protocol event.

    Rather than one class per event type, the payload is kept whole in
    :attr:`raw` and exposed through attribute access, so events added by a
    newer adapter are still readable::

        for event in client.run_sync(prompt="hi"):
            if event.type == "TEXT_MESSAGE_CONTENT":
                print(event.delta, end="")
            elif event.type == "TOOL_CALL_START":
                print(event.tool_call_name)

    Attribute lookup accepts both the wire name (``toolCallName``) and its
    snake_case spelling (``tool_call_name``). Missing fields raise
    :class:`AttributeError`; use :meth:`get` for an optional read.
    """

    type: str
    raw: Mapping[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        """Reads a field by wire or snake_case name, falling back to *default*."""
        if key in self.raw:
            return self.raw[key]
        return self.raw.get(_camel(key), default)

    def __getattr__(self, name: str) -> Any:
        # Only reached for attributes the dataclass itself does not define.
        if name.startswith("_"):
            raise AttributeError(name)
        raw = object.__getattribute__(self, "raw")
        if name in raw:
            return raw[name]
        camel = _camel(name)
        if camel in raw:
            return raw[camel]
        raise AttributeError(f"AG-UI event {self.type!r} has no field {name!r}")

    def __contains__(self, key: str) -> bool:
        return key in self.raw or _camel(key) in self.raw

    def __getitem__(self, key: str) -> Any:
        return self.__getattr__(key)

    @classmethod
    def from_wire(cls, data: Mapping[str, Any]) -> "AguiEvent":
        return cls(type=str(data.get("type", "")), raw=dict(data))


# ---------------------------------------------------------------------------
# Request payload
# ---------------------------------------------------------------------------

#: Roles a transcript message may carry.
AGUI_ROLES: tuple[str, ...] = (
    "developer",
    "system",
    "assistant",
    "user",
    "tool",
    "activity",
    "reasoning",
)


@dataclass
class AguiToolCall:
    """A tool call attached to an assistant message."""

    id: str
    name: str
    arguments: str = "{}"
    type: str = "function"

    def to_wire(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "function": {"name": self.name, "arguments": self.arguments},
        }


@dataclass
class AguiMessage:
    """One transcript message.

    AG-UI clients own the visible transcript and resend it in full on every
    turn, so this is both what you send and what you accumulate locally.
    """

    id: str
    role: str
    content: str | None = None
    name: str | None = None
    tool_calls: Sequence[AguiToolCall] | None = None
    tool_call_id: str | None = None
    error: str | None = None

    def to_wire(self) -> dict[str, Any]:
        out: dict[str, Any] = {"id": self.id, "role": self.role}
        if self.content is not None:
            out["content"] = self.content
        if self.name is not None:
            out["name"] = self.name
        if self.tool_calls:
            out["toolCalls"] = [to_wire(call) for call in self.tool_calls]
        if self.tool_call_id is not None:
            out["toolCallId"] = self.tool_call_id
        if self.error is not None:
            out["error"] = self.error
        return out


@dataclass
class AguiTool:
    """A tool the caller implements and the agent may call.

    The agent calling one of these interrupts the run: execute it and start a
    new run on the same thread with a ``tool`` message carrying the result.
    """

    name: str
    description: str | None = None
    #: JSON Schema for the tool's arguments.
    parameters: Any = None

    def to_wire(self) -> dict[str, Any]:
        out: dict[str, Any] = {"name": self.name}
        if self.description is not None:
            out["description"] = self.description
        if self.parameters is not None:
            out["parameters"] = self.parameters
        return out


@dataclass
class AguiContext:
    """Ambient context the caller attaches to the run."""

    description: str
    value: str

    def to_wire(self) -> dict[str, Any]:
        return {"description": self.description, "value": self.value}


@dataclass
class RunAgentInput:
    """The request body of a run."""

    thread_id: str
    run_id: str
    parent_run_id: str | None = None
    state: Any = None
    messages: Sequence[AguiMessage | Mapping[str, Any]] = field(default_factory=list)
    tools: Sequence[AguiTool | Mapping[str, Any]] | None = None
    context: Sequence[AguiContext | Mapping[str, Any]] | None = None
    forwarded_props: Any = None

    def to_wire(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "threadId": self.thread_id,
            "runId": self.run_id,
            "messages": [to_wire(message) for message in self.messages],
        }
        if self.parent_run_id is not None:
            out["parentRunId"] = self.parent_run_id
        if self.tools:
            out["tools"] = [to_wire(tool) for tool in self.tools]
        if self.context:
            out["context"] = [to_wire(item) for item in self.context]
        if self.state is not None:
            out["state"] = self.state
        if self.forwarded_props is not None:
            out["forwardedProps"] = self.forwarded_props
        return out


def to_wire(value: Any) -> Any:
    """Serialises a dataclass, mapping or list of them to plain JSON data."""
    if hasattr(value, "to_wire"):
        return value.to_wire()
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, (list, tuple)):
        return [to_wire(item) for item in value]
    return value


# ---------------------------------------------------------------------------
# Shared state document
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JsonPatchOperation:
    """A single RFC-6902 operation, as carried by ``STATE_DELTA``."""

    op: str
    path: str
    value: Any = None
    from_: str | None = None

    @classmethod
    def from_wire(cls, data: Mapping[str, Any]) -> "JsonPatchOperation":
        return cls(
            op=str(data.get("op", "")),
            path=str(data.get("path", "")),
            value=data.get("value"),
            from_=data.get("from"),
        )

    def to_wire(self) -> dict[str, Any]:
        out: dict[str, Any] = {"op": self.op, "path": self.path}
        if self.value is not None:
            out["value"] = self.value
        if self.from_ is not None:
            out["from"] = self.from_
        return out


@dataclass(frozen=True)
class PandoModelState:
    id: str = ""
    name: str | None = None
    provider: str | None = None
    context_window: int | None = None

    @classmethod
    def from_wire(cls, data: Mapping[str, Any]) -> "PandoModelState":
        return cls(
            id=str(data.get("id", "")),
            name=data.get("name"),
            provider=data.get("provider"),
            context_window=data.get("contextWindow"),
        )


@dataclass(frozen=True)
class PandoTokenUsageState:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    context_window: int = 0
    estimated: bool = False
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    reasoning_tokens: int | None = None
    cost: float | None = None

    @classmethod
    def from_wire(cls, data: Mapping[str, Any]) -> "PandoTokenUsageState":
        return cls(
            prompt_tokens=int(data.get("promptTokens", 0) or 0),
            completion_tokens=int(data.get("completionTokens", 0) or 0),
            context_window=int(data.get("contextWindow", 0) or 0),
            estimated=bool(data.get("estimated", False)),
            cache_read_tokens=data.get("cacheReadTokens"),
            cache_write_tokens=data.get("cacheWriteTokens"),
            reasoning_tokens=data.get("reasoningTokens"),
            cost=data.get("cost"),
        )


@dataclass(frozen=True)
class PandoFileState:
    path: str = ""
    name: str = ""
    #: ``read``, ``write``, ``edit``, ``patch`` — or whatever a newer tool reports.
    action: str = ""

    @classmethod
    def from_wire(cls, data: Mapping[str, Any]) -> "PandoFileState":
        return cls(
            path=str(data.get("path", "")),
            name=str(data.get("name", "")),
            action=str(data.get("action", "")),
        )


@dataclass(frozen=True)
class PandoSubAgentState:
    """One delegated mesnada task, as tracked by the adapter."""

    id: str = ""
    status: str = ""
    #: ``worker``, ``verifier``, ``synthesizer`` — or a newer role.
    role: str | None = None
    prompt: str | None = None
    engine: str | None = None
    model: str | None = None
    persona: str | None = None
    error: str | None = None
    exit_code: int | None = None
    #: The task's self-reported outcome: success | partial | failed | blocked.
    conclusion: str | None = None
    summary: str | None = None

    @classmethod
    def from_wire(cls, data: Mapping[str, Any]) -> "PandoSubAgentState":
        return cls(
            id=str(data.get("id", "")),
            status=str(data.get("status", "")),
            role=data.get("role"),
            prompt=data.get("prompt"),
            engine=data.get("engine"),
            model=data.get("model"),
            persona=data.get("persona"),
            error=data.get("error"),
            exit_code=data.get("exitCode"),
            conclusion=data.get("conclusion"),
            summary=data.get("summary"),
        )


@dataclass(frozen=True)
class PandoState:
    """The document published by ``STATE_SNAPSHOT`` and patched by ``STATE_DELTA``."""

    thread: str = ""
    session: str = ""
    agent: str = ""
    model: PandoModelState = field(default_factory=PandoModelState)
    todos: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    token_usage: PandoTokenUsageState | None = None
    files: Sequence[PandoFileState] = field(default_factory=tuple)
    sub_agents: Sequence[PandoSubAgentState] = field(default_factory=tuple)
    #: Echo of the state the caller pushed into the run.
    client: Any = None
    #: The snapshot exactly as received, for fields this class does not model.
    raw: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, data: Mapping[str, Any]) -> "PandoState":
        usage = data.get("tokenUsage")
        return cls(
            thread=str(data.get("thread", "")),
            session=str(data.get("session", "")),
            agent=str(data.get("agent", "")),
            model=PandoModelState.from_wire(data.get("model") or {}),
            todos=tuple(data.get("todos") or ()),
            token_usage=PandoTokenUsageState.from_wire(usage) if usage else None,
            files=tuple(PandoFileState.from_wire(f) for f in data.get("files") or ()),
            sub_agents=tuple(
                PandoSubAgentState.from_wire(a) for a in data.get("subAgents") or ()
            ),
            client=data.get("client"),
            raw=dict(data),
        )


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AguiCapabilities:
    """Which optional halves of the protocol a deployment implements."""

    frontend_tools: bool = False
    human_in_the_loop: bool = False
    shared_state: bool = False
    interrupts: bool = False

    @classmethod
    def from_wire(cls, data: Mapping[str, Any]) -> "AguiCapabilities":
        return cls(
            frontend_tools=bool(data.get("frontendTools", False)),
            human_in_the_loop=bool(data.get("humanInTheLoop", False)),
            shared_state=bool(data.get("sharedState", False)),
            interrupts=bool(data.get("interrupts", False)),
        )


@dataclass(frozen=True)
class AguiAgentDescriptor:
    name: str = ""
    description: str | None = None
    #: Absolute run endpoint for this agent.
    url: str = ""
    model: PandoModelState | None = None

    @classmethod
    def from_wire(cls, data: Mapping[str, Any]) -> "AguiAgentDescriptor":
        model = data.get("model")
        return cls(
            name=str(data.get("name", "")),
            description=data.get("description"),
            url=str(data.get("url", "")),
            model=PandoModelState.from_wire(model) if model else None,
        )


@dataclass(frozen=True)
class AguiInfo:
    """The ``GET {path}/info`` payload."""

    protocol: str = ""
    version: str | None = None
    path: str = ""
    agents: Sequence[AguiAgentDescriptor] = field(default_factory=tuple)
    capabilities: AguiCapabilities = field(default_factory=AguiCapabilities)
    raw: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, data: Mapping[str, Any]) -> "AguiInfo":
        return cls(
            protocol=str(data.get("protocol", "")),
            version=data.get("version"),
            path=str(data.get("path", "")),
            agents=tuple(
                AguiAgentDescriptor.from_wire(a) for a in data.get("agents") or ()
            ),
            capabilities=AguiCapabilities.from_wire(data.get("capabilities") or {}),
            raw=dict(data),
        )


def iter_state_deltas(events: Iterable[AguiEvent]) -> Iterable[JsonPatchOperation]:
    """Yields every JSON-Patch operation carried by ``STATE_DELTA`` events."""
    for event in events:
        if event.type == "STATE_DELTA":
            for op in event.get("delta") or ():
                yield JsonPatchOperation.from_wire(op)
