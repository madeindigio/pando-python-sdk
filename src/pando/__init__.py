"""Pando SDK — Python client for the pando AI coding assistant.

This package provides three client modes:

* :class:`PandoClient` — simple non-interactive subprocess mode
* :class:`PandoAgent` — ACP stdio JSON-RPC mode (recommended for interactive use)
* :class:`PandoHttpClient` — HTTP REST mode (requires ``pando-sdk[http]``)

Quick start::

    from pando import PandoClient

    client = PandoClient(cwd="/path/to/project")
    result = client.run("Fix all lint errors", allow_all_tools=True)
    print(result.response)

Streaming with async::

    from pando import PandoAgent

    async with PandoAgent(cwd="/path/to/project") as agent:
        session = await agent.create_session("My task")
        async for event in session.send("Refactor the database layer"):
            if event.type == "content_delta":
                print(event.delta, end="", flush=True)

Streaming with sync::

    from pando import PandoAgent

    with PandoAgent(cwd="/path/to/project") as agent:
        session = agent.create_session_sync("My task")
        for event in session.send_sync("Fix lint errors"):
            if event.type == "content_delta":
                print(event.delta, end="")
"""

from pando._version import version
from pando.agent import PandoAgent
from pando.client import PandoClient
from pando.events import AgentEvent, parse_agent_event
from pando.exceptions import (
    PandoBinaryNotFoundError,
    PandoConnectionError,
    PandoError,
    PandoRPCError,
    PandoSessionError,
    PandoTimeoutError,
)
from pando.session import PandoSession
from pando.types import MessageChunk, ModelInfo, PermissionRequest, RunResult, SessionInfo

try:
    from pando.http import PandoHttpClient

    _http_available = True
except ImportError:
    _http_available = False

__version__ = version
__all__ = [
    # Clients
    "PandoClient",
    "PandoAgent",
    "PandoSession",
    # HTTP client (optional)
    "PandoHttpClient",
    # Events
    "AgentEvent",
    "parse_agent_event",
    # Types
    "RunResult",
    "PermissionRequest",
    "SessionInfo",
    "ModelInfo",
    "MessageChunk",
    # Exceptions
    "PandoError",
    "PandoBinaryNotFoundError",
    "PandoConnectionError",
    "PandoSessionError",
    "PandoTimeoutError",
    "PandoRPCError",
    # Version
    "__version__",
]
