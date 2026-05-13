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
