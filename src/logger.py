"""
Shared agent logger.

Two thread-safe buffers:
  - _log_buffer:   Step-level entries (search, extract, gate decisions)
  - _trace_buffer: LLM I/O pairs (actual prompts → completions)

Both are plain Python lists (no st.session_state) so they are safe to
call from LangGraph worker threads — avoids the ScriptRunContext warning.

Streamlit rendering is handled by src.ui.renderers.
LangChain callback capture is handled by src.callbacks.TraceCallbackHandler.
"""

import threading
from datetime import datetime

import streamlit as st

try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx, add_script_run_ctx
except ImportError:

    def get_script_run_ctx():
        return None

    def add_script_run_ctx(*args):
        return None

# ---------------------------------------------------------------------------
# Icons
# ---------------------------------------------------------------------------

_ICONS = {
    "info": "ℹ️",
    "success": "✅",
    "warning": "⚠️",
    "error": "❌",
    "tool": "🔧",
    "search": "🔍",
    "extract": "📦",
    "think": "🧠",
    "prune": "✂️",
}

# ---------------------------------------------------------------------------
# Thread-safe buffers and live containers
# ---------------------------------------------------------------------------

_log_buffer: list[dict] = []
_trace_buffer: list[dict] = []

_live_progress_empty = None
_live_trace_empty = None
_main_streamlit_ctx = None


def attach_live_containers(progress_empty=None, trace_empty=None):
    """Attach st.empty() placeholders so logs can re-render immediately as they happen."""
    global _live_progress_empty, _live_trace_empty, _main_streamlit_ctx
    _live_progress_empty = progress_empty
    _live_trace_empty = trace_empty
    if progress_empty is not None:
        _main_streamlit_ctx = get_script_run_ctx()
    else:
        _main_streamlit_ctx = None


def _ensure_streamlit_context():
    """Ensure the current thread has the Streamlit context to safely update the UI."""
    ctx = get_script_run_ctx()
    if ctx is None and _main_streamlit_ctx is not None:
        add_script_run_ctx(threading.current_thread(), _main_streamlit_ctx)


def get_log_entries() -> list[dict]:
    """Returns a snapshot of all step-level log entries."""
    if "progress_logs" in st.session_state:
        return st.session_state["progress_logs"]
    return list(_log_buffer)


def get_trace_entries() -> list[dict]:
    """Returns a snapshot of all LLM I/O trace entries."""
    if "trace_logs" in st.session_state:
        return st.session_state["trace_logs"]
    return list(_trace_buffer)


def clear_logs():
    """Wipes the log buffers for a fresh investigation."""
    _log_buffer.clear()
    _trace_buffer.clear()
    st.session_state.pop("progress_logs", None)
    st.session_state.pop("trace_logs", None)


# ---------------------------------------------------------------------------
# Step-level logging (called by agents)
# ---------------------------------------------------------------------------


def log_step(source: str, message: str, *, level: str = "info"):
    """Appends a timestamped entry to the step log. Safe from any thread."""
    icon = _ICONS.get(level, "ℹ️")
    timestamp = datetime.now().strftime("%H:%M:%S")
    _log_buffer.append(
        {
            "time": timestamp,
            "icon": icon,
            "source": source,
            "message": message,
        }
    )

    if _live_progress_empty:
        _ensure_streamlit_context()
        from src.ui.renderers import render_step_log

        with _live_progress_empty.container():
            render_step_log()
