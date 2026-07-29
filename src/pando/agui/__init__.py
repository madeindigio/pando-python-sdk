"""``pando.agui`` — AG-UI protocol client for Pando.

AG-UI (https://docs.ag-ui.com) is the wire contract CopilotKit and other
Generative-UI frontends speak to agent backends. Pando serves it from
``pando serve --agui-port`` or the standalone ``pando agui-serve`` process; this
subpackage is the typed client for it.

The subpackage is separate so the main package stays untouched: nothing here is
imported unless you ask for it, and the synchronous API needs no third-party
dependency at all.

Stream a run::

    from pando.agui import PandoAguiClient

    client = PandoAguiClient(base_url="http://localhost:8090", token=TOKEN)
    for event in client.run_sync(prompt="Summarise the repo"):
        if event.type == "TEXT_MESSAGE_CONTENT":
            print(event.delta, end="", flush=True)

Answer a frontend tool call (an ``interrupt`` outcome)::

    from pando.agui import AguiMessage, AguiTool, random_id

    tools = [AguiTool(name="show_chart", parameters={"type": "object"})]
    history = [AguiMessage(id=random_id("msg"), role="user", content="chart it")]

    for event in client.run_sync(messages=history, tools=tools, thread_id=thread):
        if event.type == "TOOL_CALL_START":
            call_id, call_name = event.tool_call_id, event.tool_call_name

    history.append(AguiMessage(id=random_id("msg"), role="tool",
                               tool_call_id=call_id, content="rendered"))
    # Re-running on the same thread resumes the suspended run.
"""

from pando.agui.client import (
    DEFAULT_AGUI_PATH,
    PERMISSION_TOOL_NAME,
    PandoAguiClient,
    PandoAguiError,
    PandoPermissionAnswer,
    PandoPermissionRequest,
    parse_frame,
    parse_sse,
    random_id,
)
from pando.agui.types import (
    AGUI_EVENT_TYPES,
    AguiAgentDescriptor,
    AguiCapabilities,
    AguiContext,
    AguiEvent,
    AguiInfo,
    AguiMessage,
    AguiTool,
    AguiToolCall,
    JsonPatchOperation,
    PandoFileState,
    PandoModelState,
    PandoState,
    PandoSubAgentState,
    PandoTokenUsageState,
    RunAgentInput,
    iter_state_deltas,
)

__all__ = [
    # Client
    "PandoAguiClient",
    "PandoAguiError",
    "DEFAULT_AGUI_PATH",
    "PERMISSION_TOOL_NAME",
    "PandoPermissionRequest",
    "PandoPermissionAnswer",
    "parse_sse",
    "parse_frame",
    "random_id",
    # Protocol types
    "AGUI_EVENT_TYPES",
    "AguiEvent",
    "AguiMessage",
    "AguiToolCall",
    "AguiTool",
    "AguiContext",
    "RunAgentInput",
    "JsonPatchOperation",
    # State
    "PandoState",
    "PandoModelState",
    "PandoTokenUsageState",
    "PandoFileState",
    "PandoSubAgentState",
    "iter_state_deltas",
    # Discovery
    "AguiInfo",
    "AguiAgentDescriptor",
    "AguiCapabilities",
]
