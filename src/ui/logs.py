"""
Logs UI component — renders live and static progress + trace log tabs.
"""

import streamlit as st

from src.ui.renderers import render_step_log, render_llm_trace


from src.logger import get_log_entries, get_trace_entries


def setup_log_tabs(tab_progress, tab_trace) -> tuple:
    """Creates placeholders inside the log tabs. Returns (progress_log_area, trace_log_area)."""
    with tab_progress:
        st.header("Investigation Progress")
        st.markdown("Step-by-step decisions made by each agent during the pipeline.")
        progress_status_area = st.empty()
        progress_log_area = st.empty()

    with tab_trace:
        st.header("Agent Log — LLM Input / Output")
        st.markdown("Actual prompts sent to the LLM and its raw responses.")
        trace_log_area = st.empty()

    return progress_status_area, progress_log_area, trace_log_area


def render_static_logs(progress_log_area, trace_log_area):
    """Fills the log tab placeholders with the current buffer contents (post-run)."""
    with progress_log_area.container():
        render_step_log(get_log_entries())
    with trace_log_area.container():
        render_llm_trace(get_trace_entries())
