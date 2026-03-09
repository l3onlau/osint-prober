"""
OSINT InvestigatAR — Streamlit application entry point.

Responsibilities (only):
  - Page config
  - Tab scaffold
  - Ingestion execution loop
  - Wiring of ui/ sub-components
"""

import os

import streamlit as st
from langchain_core.runnables.config import RunnableConfig

os.environ["USER_AGENT"] = "OSINT_Prober/1.0"

from src.graph import build_investigator_graph
from src.state import InvestigatorState
from src.callbacks import TraceCallbackHandler
from src.logger import attach_live_containers, get_log_entries
from src.ui.sidebar import render_sidebar
from src.ui.dashboard import render_dashboard
from src.ui.chat import render_chat
from src.ui.logs import setup_log_tabs, render_static_logs
from src.ui.renderers import render_llm_trace
from src.config import config as app_config

# ---------------------------------------------------------------------------
# 1. Page config
# ---------------------------------------------------------------------------

st.set_page_config(page_title="OSINT InvestigatAR", page_icon="🕵️", layout="wide")
st.title("OSINT Investigation Assistant")
st.markdown("Automated Open-Source Intelligence Gathering & GraphRAG Synthesis.")

if "graph_app" not in st.session_state:
    st.session_state.graph_app = build_investigator_graph()

if "is_investigating" not in st.session_state:
    st.session_state.is_investigating = False

# ---------------------------------------------------------------------------
# 2. Sidebar
# ---------------------------------------------------------------------------

render_sidebar()

# ---------------------------------------------------------------------------
# 3. Tab scaffold (must exist before the blocking loop so tabs are clickable)
# ---------------------------------------------------------------------------

tab_dash, tab_chat, tab_progress, tab_trace = st.tabs(
    ["Dashboard & Connections", "GraphRAG Q&A", "Progress Log", "Agent Log (LLM I/O)"]
)

progress_status_area, progress_log_area, trace_log_area = setup_log_tabs(
    tab_progress, tab_trace
)

# ---------------------------------------------------------------------------
# 4. Ingestion execution loop
# ---------------------------------------------------------------------------

if st.session_state.get("is_investigating", False):
    investigation_target = st.session_state.get("target_input", "")
    investigation_id = st.session_state.get("investigation_id", "global_intelligence")

    initial_state: InvestigatorState = {
        "investigation_id": investigation_id,
        "investigation_target": investigation_target,
        "target_name": None,
        "current_phase": "ingestion",
        "queries": [],
        "raw_context": [],
        "extracted_entities": [],
        "extracted_relationships": [],
        "chat_question": "",
        "synthesized_answer": "",
        "briefing": "",
        "iteration_count": 0,
        "max_iterations": st.session_state.get("max_iterations", 2),
        "max_search_results": st.session_state.get(
            "max_search_results", app_config.max_search_results
        ),
        "scrape_content_chars_max": st.session_state.get(
            "scrape_content_chars_max", app_config.scrape_content_chars_max
        ),
        "planner_query_count": st.session_state.get(
            "planner_query_count", app_config.planner_query_count
        ),
        "lateral_expansion_depth": st.session_state.get("lateral_expansion_depth", 1),
        "current_expansion_depth": 0,
        "max_entities_to_expand": st.session_state.get("max_entities_to_expand", 3),
    }

    trace_handler = TraceCallbackHandler()
    run_config = RunnableConfig(recursion_limit=25, callbacks=[trace_handler])
    final_state: dict = {}

    from src.logger import clear_logs, get_log_entries, get_trace_entries

    clear_logs()

    progress_status_area.info("🕵️ Investigation in progress...")

    try:
        attach_live_containers(progress_log_area, trace_log_area)
        for chunk in st.session_state.graph_app.stream(
            initial_state, config=run_config
        ):
            node_name = list(chunk.keys())[0]
            node_state = chunk.get(node_name)
            if node_state is not None:
                final_state.update(node_state)
            progress_status_area.info(f"🕵️ Running: **{node_name}**…")
            # Re-render Agent Log from the main thread after each node completes.
            # LangChain callbacks (on_llm_start/end) run in a ThreadPoolExecutor
            # and cannot safely hold a ScriptRunContext, so we push the render here.
            with trace_log_area.container():
                render_llm_trace()
    finally:
        attach_live_containers(None, None)

    progress_status_area.success("✅ Investigation complete!")

    st.session_state["briefing"] = final_state.get("briefing", "")
    st.session_state["resolved_target"] = final_state.get(
        "target_name", investigation_target
    )
    entities = final_state.get("extracted_entities", [])
    relationships = final_state.get("extracted_relationships", [])
    st.success(
        f"Ingestion complete! Found **{len(entities)}** entities and "
        f"**{len(relationships)}** relationships. "
        f"Investigation ID: `{investigation_id}`"
    )
    st.session_state["progress_logs"] = get_log_entries()
    st.session_state["trace_logs"] = get_trace_entries()
    st.session_state.is_investigating = False

# ---------------------------------------------------------------------------
# 5. Static renders (post-execution or empty state)
# ---------------------------------------------------------------------------

if not st.session_state.get("is_investigating", False):
    render_static_logs(progress_log_area, trace_log_area)

with tab_dash:
    render_dashboard()

with tab_chat:
    render_chat(st.session_state.graph_app)
