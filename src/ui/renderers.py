"""
Streamlit renderers for the Progress Log and Agent Log tabs.

Separated from logger.py so the logging core has no Streamlit dependency
and can be used in tests or a future non-Streamlit backend.
"""

import streamlit as st

from src.logger import _log_buffer, _trace_buffer


def render_step_log(entries=None):
    """Renders the step-level log in the Progress tab, grouped by agent."""
    log = entries if entries is not None else _log_buffer

    st.caption(f"{len(log)} entries")

    # Group by source while preserving insertion order
    groups: dict[str, list[dict]] = {}
    group_order: list[str] = []
    for entry in log:
        source = entry["source"]
        if source not in groups:
            groups[source] = []
            group_order.append(source)
        groups[source].append(entry)

    for source in group_order:
        st.markdown(f"#### 🤖 {source}")
        for entry in groups[source]:
            st.markdown(f"`{entry['time']}` {entry['icon']} {entry['message']}")
        st.divider()


def render_llm_trace(entries=None):
    """Renders the LLM I/O trace in the Agent Log tab.

    Accepts an optional *entries* list so the live streaming loop in
    ``app.py`` can share the same renderer instead of duplicating logic.
    Falls back to the module-level ``_trace_buffer`` when called without
    arguments (post-execution static render).

    Direction types:
      input       — LLM prompt (from on_chat_model_start / on_llm_start)
      output      — LLM completion (from on_llm_end)
      error       — LLM error (from on_llm_error)
      tool_call   — Tool invocation (from on_tool_start)
      tool_result — Tool response (from on_tool_end)
    """
    trace = entries if entries is not None else _trace_buffer

    st.caption(f"{len(trace)} trace entries")

    llm_call_num = 0
    for entry in trace:
        direction = entry["direction"]

        if direction == "input":
            llm_call_num += 1
            st.markdown(f"📥 `{entry['time']}` — **LLM Input #{llm_call_num}**")
            st.code(entry["content"], language="text")

        elif direction == "output":
            st.markdown(f"📤 `{entry['time']}` — **LLM Output #{llm_call_num}**")
            st.markdown(entry["content"])

        elif direction == "tool_call":
            st.markdown(f"🛠️ `{entry['time']}` — **Tool Call**")
            st.code(entry["content"], language="text")

        elif direction == "tool_result":
            st.markdown(f"📋 `{entry['time']}` — **Tool Result**")
            with st.expander("Show tool output", expanded=False):
                st.code(entry["content"], language="text")

        else:  # error
            st.markdown(f"❌ `{entry['time']}` — **LLM Error**")
            st.error(entry["content"])

        st.divider()
