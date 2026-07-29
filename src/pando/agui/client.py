"""PandoAguiClient — dependency-free client for Pando's AG-UI endpoint.

It speaks the wire protocol directly (POST a ``RunAgentInput``, read back an SSE
stream of events), which is what CopilotKit and other Generative-UI frontends
speak to agent backends. Pando serves it from ``pando serve --agui-port`` or the
standalone ``pando agui-serve`` process.

The synchronous API uses only the standard library. The async API needs
``httpx``::

    pip install pando-sdk[agui]

Usage::

    from pando.agui import PandoAguiClient

    client = PandoAguiClient(base_url="http://localhost:8090", token=TOKEN)

    for event in client.run_sync(prompt="List the Go packages"):
        if event.type == "TEXT_MESSAGE_CONTENT":
            print(event.delta, end="", flush=True)

Async::

    async for event in client.run(prompt="List the Go packages"):
        ...
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Iterable, Iterator, Mapping, Sequence

from pando.agui.types import (
    AguiContext,
    AguiEvent,
    AguiInfo,
    AguiMessage,
    AguiTool,
    RunAgentInput,
)
from pando.exceptions import PandoConnectionError, PandoError

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_AGUI_PATH",
    "PERMISSION_TOOL_NAME",
    "PandoAguiClient",
    "PandoAguiError",
    "PandoPermissionAnswer",
    "PandoPermissionRequest",
    "parse_sse",
    "random_id",
]

#: The route prefix ``pando agui-serve`` uses out of the box.
DEFAULT_AGUI_PATH = "/api/v1/agui"

#: The synthetic tool a permission prompt arrives as.
PERMISSION_TOOL_NAME = "pando_permission_request"


class PandoAguiError(PandoError):
    """Raised when the adapter answers with a non-2xx status.

    :attr:`status` is the HTTP status code, or ``0`` for a ``RUN_ERROR`` event
    surfaced by :meth:`PandoAguiClient.run_text`.
    """

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class PandoPermissionRequest:
    """Arguments of a :data:`PERMISSION_TOOL_NAME` call."""

    tool_name: str
    action: str
    description: str | None = None
    path: str | None = None
    params: Any = None

    @classmethod
    def from_wire(cls, data: Mapping[str, Any]) -> "PandoPermissionRequest":
        return cls(
            tool_name=str(data.get("toolName", "")),
            action=str(data.get("action", "")),
            description=data.get("description"),
            path=data.get("path"),
            params=data.get("params"),
        )

    @classmethod
    def from_arguments(cls, arguments: str) -> "PandoPermissionRequest":
        """Parses the JSON ``arguments`` string of the tool call."""
        return cls.from_wire(json.loads(arguments or "{}"))


@dataclass(frozen=True)
class PandoPermissionAnswer:
    """Answer shape the adapter accepts for a permission prompt.

    Anything else — including no answer at all — is read as a denial.
    """

    approved: bool

    def to_wire(self) -> dict[str, Any]:
        return {"approved": self.approved}

    def to_json(self) -> str:
        return json.dumps(self.to_wire())


def random_id(prefix: str) -> str:
    """Builds an id that is unique enough for a thread, run or message."""
    return f"{prefix}-{uuid.uuid4()}"


class PandoAguiClient:
    """Client for Pando's AG-UI endpoint.

    Args:
        base_url: Origin of the adapter, e.g. ``http://localhost:8090``.
        path: Route prefix. Defaults to :data:`DEFAULT_AGUI_PATH`.
        agent: Agent to run. Defaults to ``coder``.
        token: Bearer token. Required unless the server was started with
            ``--no-token``: the adapter rejects unauthenticated requests with 401.
        headers: Extra headers merged into every request.
        timeout: Seconds to wait for the response headers. Default ``60``.
    """

    def __init__(
        self,
        base_url: str,
        *,
        path: str = DEFAULT_AGUI_PATH,
        agent: str = "coder",
        token: str | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.path = path.rstrip("/")
        self.agent = agent
        self.token = token
        self.headers = dict(headers or {})
        self.timeout = timeout

    # -- URLs and headers ---------------------------------------------------

    def agent_url(self, agent: str | None = None) -> str:
        """Absolute run endpoint of an agent."""
        return f"{self.base_url}{self.path}/{agent or self.agent}"

    @property
    def info_url(self) -> str:
        """Absolute discovery endpoint."""
        return f"{self.base_url}{self.path}/info"

    def _request_headers(self, extra: Mapping[str, str] | None = None) -> dict[str, str]:
        headers = dict(self.headers)
        if extra:
            headers.update(extra)
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    # -- Request building ---------------------------------------------------

    def build_input(
        self,
        *,
        prompt: str | None = None,
        messages: Sequence[AguiMessage | Mapping[str, Any]] | None = None,
        thread_id: str | None = None,
        run_id: str | None = None,
        tools: Sequence[AguiTool | Mapping[str, Any]] | None = None,
        context: Sequence[AguiContext | Mapping[str, Any]] | None = None,
        state: Any = None,
    ) -> RunAgentInput:
        """Builds the request body from the friendlier run arguments."""
        if messages is None:
            messages = (
                []
                if prompt is None
                else [AguiMessage(id=random_id("msg"), role="user", content=prompt)]
            )
        return RunAgentInput(
            thread_id=thread_id or random_id("thread"),
            run_id=run_id or random_id("run"),
            messages=list(messages),
            tools=list(tools) if tools else None,
            context=list(context) if context else None,
            state=state,
        )

    # -- Synchronous API ----------------------------------------------------

    def info_sync(self) -> AguiInfo:
        """Fetches the discovery document (standard library only).

        Tells which agents exist, their absolute URLs, the model behind each and
        which optional halves of the protocol this deployment implements.
        """
        request = urllib.request.Request(
            self.info_url, headers=self._request_headers(), method="GET"
        )
        with self._urlopen(request) as response:
            return AguiInfo.from_wire(json.loads(response.read().decode("utf-8")))

    def run_sync(
        self,
        *,
        prompt: str | None = None,
        messages: Sequence[AguiMessage | Mapping[str, Any]] | None = None,
        thread_id: str | None = None,
        run_id: str | None = None,
        tools: Sequence[AguiTool | Mapping[str, Any]] | None = None,
        context: Sequence[AguiContext | Mapping[str, Any]] | None = None,
        state: Any = None,
        agent: str | None = None,
    ) -> Iterator[AguiEvent]:
        """Runs the agent and yields protocol events as they arrive.

        The stream ends with ``RUN_FINISHED``. An outcome of ``interrupt`` means
        the agent called one of the caller's ``tools``: execute it and call
        :meth:`run_sync` again on the same thread with the transcript plus a
        ``tool`` message carrying the result, which resumes the suspended run.

        Uses only the standard library — no ``httpx`` needed.
        """
        payload = self.build_input(
            prompt=prompt,
            messages=messages,
            thread_id=thread_id,
            run_id=run_id,
            tools=tools,
            context=context,
            state=state,
        ).to_wire()

        request = urllib.request.Request(
            self.agent_url(agent),
            data=json.dumps(payload).encode("utf-8"),
            headers=self._request_headers(
                {"Content-Type": "application/json", "Accept": "text/event-stream"}
            ),
            method="POST",
        )
        with self._urlopen(request) as response:
            yield from parse_sse(iter(lambda: response.read(4096), b""))

    def run_text_sync(self, prompt: str, **kwargs: Any) -> str:
        """Runs a prompt and returns the assistant's text.

        Tool calls, state and reasoning are dropped: use :meth:`run_sync` when
        they matter.
        """
        return _collect_text(self.run_sync(prompt=prompt, **kwargs))

    def _urlopen(self, request: urllib.request.Request) -> Any:
        try:
            return urllib.request.urlopen(request, timeout=self.timeout)
        except urllib.error.HTTPError as exc:  # non-2xx: the body may explain why
            body = ""
            try:
                body = exc.read().decode("utf-8", "replace")
            except Exception:  # pragma: no cover - body already consumed
                pass
            raise PandoAguiError(exc.code, _error_message(body, exc.code, exc.reason)) from None
        except urllib.error.URLError as exc:
            raise PandoConnectionError(
                f"AG-UI request to {request.full_url} failed: {exc.reason}"
            ) from None

    # -- Asynchronous API ---------------------------------------------------

    async def info(self) -> AguiInfo:
        """Async variant of :meth:`info_sync`. Requires ``httpx``."""
        httpx = _require_httpx()
        async with httpx.AsyncClient(timeout=self.timeout) as http:
            try:
                response = await http.get(self.info_url, headers=self._request_headers())
            except httpx.HTTPError as exc:
                raise PandoConnectionError(
                    f"AG-UI request to {self.info_url} failed: {exc}"
                ) from None
            if response.status_code >= 400:
                raise PandoAguiError(
                    response.status_code,
                    _error_message(response.text, response.status_code, response.reason_phrase),
                )
            return AguiInfo.from_wire(response.json())

    async def run(
        self,
        *,
        prompt: str | None = None,
        messages: Sequence[AguiMessage | Mapping[str, Any]] | None = None,
        thread_id: str | None = None,
        run_id: str | None = None,
        tools: Sequence[AguiTool | Mapping[str, Any]] | None = None,
        context: Sequence[AguiContext | Mapping[str, Any]] | None = None,
        state: Any = None,
        agent: str | None = None,
    ) -> AsyncGenerator[AguiEvent, None]:
        """Async variant of :meth:`run_sync`. Requires ``httpx``."""
        httpx = _require_httpx()
        payload = self.build_input(
            prompt=prompt,
            messages=messages,
            thread_id=thread_id,
            run_id=run_id,
            tools=tools,
            context=context,
            state=state,
        ).to_wire()
        url = self.agent_url(agent)

        async with httpx.AsyncClient(timeout=self.timeout) as http:
            try:
                async with http.stream(
                    "POST",
                    url,
                    json=payload,
                    headers=self._request_headers(
                        {"Accept": "text/event-stream"}
                    ),
                ) as response:
                    if response.status_code >= 400:
                        body = (await response.aread()).decode("utf-8", "replace")
                        raise PandoAguiError(
                            response.status_code,
                            _error_message(
                                body, response.status_code, response.reason_phrase
                            ),
                        )
                    buffer = ""
                    async for chunk in response.aiter_text():
                        buffer += chunk
                        frames, buffer = _split_frames(buffer)
                        for frame in frames:
                            event = parse_frame(frame)
                            if event is not None:
                                yield event
                    tail = parse_frame(buffer)
                    if tail is not None:
                        yield tail
            except httpx.HTTPError as exc:
                raise PandoConnectionError(f"AG-UI request to {url} failed: {exc}") from None

    async def run_text(self, prompt: str, **kwargs: Any) -> str:
        """Async variant of :meth:`run_text_sync`. Requires ``httpx``."""
        text = ""
        async for event in self.run(prompt=prompt, **kwargs):
            if event.type == "TEXT_MESSAGE_CONTENT":
                text += event.get("delta") or ""
            elif event.type == "RUN_ERROR":
                raise PandoAguiError(0, event.get("message") or "run failed")
        return text


# ---------------------------------------------------------------------------
# SSE parsing
# ---------------------------------------------------------------------------


def parse_sse(chunks: Iterable[bytes | str]) -> Iterator[AguiEvent]:
    """Parses an SSE body into events.

    The adapter sends one JSON object per ``data:`` line and never splits an
    event across frames, but a chunk boundary can fall anywhere, so frames are
    reassembled from the raw stream.
    """
    buffer = ""
    for chunk in chunks:
        if not chunk:
            continue
        buffer += chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk
        frames, buffer = _split_frames(buffer)
        for frame in frames:
            event = parse_frame(frame)
            if event is not None:
                yield event
    # A stream cut without its final blank line still carries a whole event.
    tail = parse_frame(buffer)
    if tail is not None:
        yield tail


def _split_frames(buffer: str) -> tuple[list[str], str]:
    """Splits complete ``\\n\\n``-terminated frames off the front of *buffer*."""
    frames: list[str] = []
    while True:
        boundary = buffer.find("\n\n")
        if boundary == -1:
            return frames, buffer
        frames.append(buffer[:boundary])
        buffer = buffer[boundary + 2 :]


def parse_frame(frame: str) -> AguiEvent | None:
    """Extracts the JSON payload of one SSE frame, or ``None`` when it has none."""
    data = "\n".join(
        line[5:].lstrip() for line in frame.split("\n") if line.startswith("data:")
    )
    if not data or data == "[DONE]":
        return None
    try:
        payload = json.loads(data)
    except json.JSONDecodeError:
        # A malformed frame must not kill a run that is otherwise fine.
        logger.debug("Failed to parse AG-UI frame: %s", data[:200])
        return None
    if not isinstance(payload, Mapping):
        return None
    return AguiEvent.from_wire(payload)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collect_text(events: Iterable[AguiEvent]) -> str:
    text = ""
    for event in events:
        if event.type == "TEXT_MESSAGE_CONTENT":
            text += event.get("delta") or ""
        elif event.type == "RUN_ERROR":
            raise PandoAguiError(0, event.get("message") or "run failed")
    return text


def _error_message(body: str, status: int, reason: str | None) -> str:
    """Reads the adapter's JSON error body, falling back to the status text."""
    try:
        parsed = json.loads(body)
        if isinstance(parsed, Mapping):
            detail = parsed.get("error") or parsed.get("message")
            if detail:
                return str(detail)
    except (json.JSONDecodeError, TypeError):
        pass
    return f"{status} {reason or ''}".strip()


def _require_httpx() -> Any:
    try:
        import httpx
    except ImportError:
        raise ImportError(
            "The 'httpx' package is required for the async AG-UI API. "
            "Install it with: pip install pando-sdk[agui] "
            "(or use the run_sync/info_sync methods, which need no dependencies)."
        ) from None
    return httpx
