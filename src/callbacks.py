"""
LangChain callback handler that captures LLM I/O and tool calls into the trace buffer.

Separated from logger.py so the pure logging layer has no LangChain dependency.

NOTE: Do NOT call any Streamlit rendering here. LangChain fires these callbacks
from a ThreadPoolExecutor thread which has no reliable ScriptRunContext.
Live rendering is handled by the main stream loop in app.py instead.
"""

from datetime import datetime
from typing import Any

from src.logger import _trace_buffer
from langchain_core.callbacks import BaseCallbackHandler


class TraceCallbackHandler(BaseCallbackHandler):
    """Captures every LLM prompt, completion, and tool call into _trace_buffer.

    ChatOllama (and all BaseChatModel subclasses) fires on_chat_model_start,
    NOT on_llm_start.  Both are implemented here so legacy text-completion
    models are also covered.
    """

    def on_chat_model_start(
        self, serialized: dict, messages: list[list], **kwargs: Any
    ):
        """Fires for all chat models (ChatOllama, ChatOpenAI, etc.)."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        for message_batch in messages:
            parts = []
            for msg in message_batch:
                role = getattr(msg, "type", type(msg).__name__).upper()
                content = getattr(msg, "content", str(msg))
                parts.append(f"[{role}]:\n{content[:600]}")
            _trace_buffer.append(
                {
                    "time": timestamp,
                    "direction": "input",
                    "content": "\n\n".join(parts)[:1500],
                }
            )

    def on_llm_start(self, serialized: dict, prompts: list[str], **kwargs: Any):
        """Fallback for legacy text-completion LLMs (not chat models)."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        for prompt in prompts:
            _trace_buffer.append(
                {
                    "time": timestamp,
                    "direction": "input",
                    "content": prompt[:1500],
                }
            )

    def on_llm_end(self, response: Any, **kwargs: Any):
        import json

        timestamp = datetime.now().strftime("%H:%M:%S")
        text = ""
        try:
            generation = response.generations[0][0]

            if hasattr(generation, "message") and hasattr(
                generation.message, "tool_calls"
            ):
                tool_calls = generation.message.tool_calls
                if tool_calls:
                    calls_str = [
                        f"🛠️ Tool Call: {call.get('name')}({json.dumps(call.get('args', {}))})"
                        for call in tool_calls
                    ]
                    text = "\n".join(calls_str)

            if hasattr(generation, "text") and generation.text:
                if text:
                    text += "\n\n"
                text += generation.text

        except Exception:
            text = str(response)

        if not text.strip():
            text = "*(Empty text response — check LangGraph state for hidden tool triggers)*"

        _trace_buffer.append(
            {
                "time": timestamp,
                "direction": "output",
                "content": text[:1500],
            }
        )

    def on_llm_error(self, error: BaseException, **kwargs: Any):
        timestamp = datetime.now().strftime("%H:%M:%S")
        _trace_buffer.append(
            {
                "time": timestamp,
                "direction": "error",
                "content": str(error)[:500],
            }
        )

    def on_tool_start(self, serialized: dict, input_str: str, **kwargs: Any):
        """Captures tool invocations so the Agent Log shows the full reasoning chain."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        tool_name = serialized.get("name", "unknown_tool")
        _trace_buffer.append(
            {
                "time": timestamp,
                "direction": "tool_call",
                "content": f"**{tool_name}**\n{input_str[:800]}",
            }
        )

    def on_tool_end(self, output: str, **kwargs: Any):
        """Captures tool results so the Agent Log shows what each tool returned."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        _trace_buffer.append(
            {
                "time": timestamp,
                "direction": "tool_result",
                "content": str(output)[:800],
            }
        )
