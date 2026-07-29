# pando-sdk

Python SDK for [pando](https://github.com/digiogithub/pando) — the AI coding assistant CLI.

## Installation

```bash
pip install pando-sdk
```

For HTTP REST client support (optional):

```bash
pip install pando-sdk[http]
```

## Requirements

- Python 3.10 or newer
- `pando` CLI installed and available on your PATH (or set `PANDO_PATH`)

## Quick Start

### Simple mode (one-shot subprocess)

```python
from pando import PandoClient

client = PandoClient(
    cwd="/path/to/project",   # working directory for pando
    model="copilot.gpt-5.4",  # optional model override
    timeout=300,              # seconds to wait for completion
)

result = client.run("Fix all lint errors", allow_all_tools=True)
print(result.response)
print(result.session_id)
```

### ACP session mode — async streaming

```python
import asyncio
from pando import PandoAgent

async def main():
    async with PandoAgent(
        cwd="/path/to/project",
        model="claude-sonnet-4-6",
        persona="software-engineer",
    ) as agent:
        session = await agent.create_session("Refactoring task")

        async for event in session.send("Refactor the database layer"):
            match event.type:
                case "content_delta":
                    print(event.delta, end="", flush=True)
                case "tool_call":
                    print(f"\n[Tool] {event.tool_call['name']}")
                case "tool_result":
                    print(f"[Result] {event.tool_result['content'][:80]}")
                case "response":
                    print("\n[Done]")
                case "error":
                    raise RuntimeError(event.error)

asyncio.run(main())
```

### ACP session mode — sync streaming

```python
from pando import PandoAgent

with PandoAgent(cwd="/path/to/project") as agent:
    session = agent.create_session_sync("My task")
    for event in session.send_sync("Fix all lint errors"):
        if event.type == "content_delta":
            print(event.delta, end="", flush=True)
    print()  # newline after streaming
```

### Ask for a full response (no streaming)

```python
import asyncio
from pando import PandoAgent

async def main():
    async with PandoAgent(cwd="/path/to/project") as agent:
        session = await agent.create_session()
        response = await session.ask("Explain the auth module")
        print(response)

asyncio.run(main())
```

### Manage multiple sessions

```python
import asyncio
from pando import PandoAgent

async def main():
    async with PandoAgent(cwd="/path/to/project") as agent:
        # List existing sessions
        sessions = await agent.list_sessions()
        for s in sessions:
            print(f"  {s.session_id}: {s.title}")

        # Create and work with a new session
        session = await agent.create_session("Feature implementation")
        await session.set_persona("software-engineer")

        async for event in session.send("Implement user authentication"):
            if event.type == "content_delta":
                print(event.delta, end="")

asyncio.run(main())
```

### HTTP REST client

Requires `pip install pando-sdk[http]` and a running `pando serve` process.

```python
import asyncio
from pando import PandoHttpClient

async def main():
    async with PandoHttpClient(
        base_url="http://localhost:8765",
        verify_ssl=False,  # for self-signed certs in development
    ) as client:
        # List sessions
        sessions = await client.sessions.list()

        # Create a session
        session = await client.sessions.create("My task")

        # Stream a message
        async for chunk in client.sessions.send_message(session.session_id, "Fix lint"):
            if chunk.delta:
                print(chunk.delta, end="", flush=True)

        # Models
        models = await client.models.list()
        await client.models.set_active("claude-sonnet-4-6")

        # Personas
        personas = await client.personas.list()
        await client.personas.set_active("qa")

asyncio.run(main())
```

### AG-UI / GenUI client (`pando.agui`)

[AG-UI](https://docs.ag-ui.com) is the protocol CopilotKit and other Generative-UI
frontends speak to agent backends. Pando serves it from `pando agui-serve --port 8090`
(or `pando serve --agui-port 8090`); it is **off by default** and requires a bearer
token and an origin allow-list, because it exposes a code-executing agent to a browser.

`pando.agui` is a separate subpackage: importing `pando` pulls in none of it. The
synchronous API uses only the standard library; the async one needs
`pip install pando-sdk[agui]`.

```python
import os

from pando.agui import PandoAguiClient, PandoState

client = PandoAguiClient(
    base_url="http://localhost:8090",
    token=os.environ["PANDO_TOKEN"],
    agent="coder",
)

# Discovery: which agents exist, their model, which capabilities are on
info = client.info_sync()

for event in client.run_sync(prompt="Summarise the repo"):
    if event.type == "TEXT_MESSAGE_CONTENT":
        print(event.delta, end="", flush=True)
    elif event.type == "STATE_SNAPSHOT":
        state = PandoState.from_wire(event.snapshot)
        print(state.todos, state.sub_agents)
    elif event.type == "RUN_FINISHED" and event.get("outcome") == "interrupt":
        # The agent called one of your `tools`: run it, then call run_sync again
        # on the same thread with a `tool` message carrying the result.
        ...
```

Async, with `httpx` installed:

```python
async for event in client.run(prompt="Summarise the repo"):
    ...

text = await client.run_text("What does cmd/root.go do?")
```

Frontend tools and human-in-the-loop:

```python
from pando.agui import AguiMessage, AguiTool, PandoPermissionRequest, PERMISSION_TOOL_NAME

tools = [AguiTool(name="show_chart", description="Renders a chart",
                  parameters={"type": "object"})]
history = [AguiMessage(id="m1", role="user", content="chart the commits")]

call_id = None
for event in client.run_sync(messages=history, tools=tools, thread_id="thread-1"):
    if event.type == "TOOL_CALL_START":
        call_id = event.tool_call_id          # PERMISSION_TOOL_NAME for approvals
    elif event.type == "TOOL_CALL_ARGS" and event.tool_call_id == call_id:
        arguments = event.delta               # accumulate, then parse

# Answer it and resume the suspended run on the same thread.
history.append(AguiMessage(id="m2", role="tool", tool_call_id=call_id, content="rendered"))
for event in client.run_sync(messages=history, tools=tools, thread_id="thread-1"):
    ...
```

| Export | Purpose |
|---|---|
| `PandoAguiClient` | Run/discovery client (`run`, `run_text`, `info` + `*_sync` variants) |
| `AguiEvent` | One protocol event; fields readable as `event.delta` or `event.get("delta")` |
| `PandoState` | The shared-state document (`STATE_SNAPSHOT`) |
| `JsonPatchOperation` | One `STATE_DELTA` operation |
| `AguiMessage` / `AguiTool` / `AguiContext` | Request payload types |
| `AguiInfo` | The `/info` discovery document |
| `parse_sse` | The event-stream parser, if you issue the request yourself |
| `PERMISSION_TOOL_NAME`, `PandoPermissionRequest` | Human-in-the-loop approvals |

## Reference

### PandoClient (simple subprocess mode)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cwd` | `str \| None` | `None` | Working directory for pando |
| `model` | `str \| None` | `None` | Model override |
| `pando_path` | `str \| None` | `None` | Explicit path to pando binary |
| `timeout` | `float` | `300.0` | Max seconds to wait |

Methods:
- `run(prompt, allow_all_tools=False) -> RunResult` — synchronous
- `run_async(prompt, allow_all_tools=False) -> RunResult` — async

### PandoAgent (ACP stdio mode)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cwd` | `str \| None` | `None` | Working directory |
| `model` | `str \| None` | `None` | Initial model |
| `persona` | `str \| None` | `None` | Initial persona |
| `pando_path` | `str \| None` | `None` | Explicit binary path |
| `on_tool_permission` | `Callable \| None` | `None` | Permission callback |
| `debug` | `bool` | `False` | Enable debug logging |

Async methods: `connect`, `disconnect`, `create_session`, `load_session`,
`list_sessions`, `set_model`, `list_models`, `set_persona`, `get_persona`,
`list_personas`, `list_tools`.

Each method has a `_sync` variant for use in synchronous context manager mode.

### PandoSession

| Method | Description |
|--------|-------------|
| `send(prompt)` | Async generator yielding `AgentEvent` |
| `ask(prompt)` | Collect full response text |
| `set_persona(name)` | Set persona for this session |
| `set_model(model_id)` | Set model for this session |
| `cancel()` | Cancel in-progress task |
| `close()` | Close the session |

Sync variants: `send_sync`, `ask_sync`, `set_persona_sync`, `cancel_sync`, `close_sync`.

### AgentEvent

| Field | Type | Populated for |
|-------|------|---------------|
| `type` | `str` | all events |
| `session_id` | `str` | all events |
| `delta` | `str \| None` | `content_delta`, `thinking_delta` |
| `tool_call` | `dict \| None` | `tool_call` |
| `tool_result` | `dict \| None` | `tool_result` |
| `message` | `dict \| None` | `response` |
| `error` | `str \| None` | `error` |

### Environment variables

| Variable | Description |
|----------|-------------|
| `PANDO_PATH` | Path to the pando binary |

### Exception hierarchy

```
PandoError (base)
├── PandoBinaryNotFoundError   # binary not found
├── PandoConnectionError       # subprocess/connection error
├── PandoSessionError          # session-level error
├── PandoTimeoutError          # timeout exceeded
└── PandoRPCError(code, msg)   # JSON-RPC error response
```

## Development

```bash
# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Type checking
mypy src/pando
```

## License

MIT
