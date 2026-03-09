"""
Sidebar UI component — target input, ingestion trigger, and investigation settings.
"""

import streamlit as st


def render_sidebar() -> None:
    """Renders the sidebar with target input, investigation controls, and settings."""
    with st.sidebar:
        st.header("1. Intelligence Ingestion")
        st.markdown("Enter a precise name (e.g., `Jho Low`) OR a vague description.")

        st.text_area(
            "Target Profile or Description:",
            placeholder="e.g., 'The Malaysian financier from the 1MDB scandal'",
            disabled=st.session_state.get("is_investigating", False),
            key="target_input",
        )

        workspace_raw = st.text_input(
            "Investigation Workspace (ID):",
            value="global_intelligence",
            key="workspace_name",
            help="All data is saved here. Use the same name to expand an existing graph with a new search.",
            disabled=st.session_state.get("is_investigating", False),
        )

        import re

        workspace_clean = workspace_raw.strip()
        if not workspace_clean:
            workspace_clean = "global_intelligence"
        investigation_id = re.sub(r"[^a-z0-9A-Z_]+", "-", workspace_clean).strip("-")

        # Eagerly inject it so the dashboard can render immediately prior to clicking "Start Investigation"
        st.session_state["investigation_id"] = investigation_id

        st.divider()
        st.subheader("⚙️ Settings")

        st.slider(
            "Initial Plan Queries",
            min_value=3,
            max_value=12,
            value=st.session_state.get("planner_query_count", 5),
            step=1,
            key="planner_query_count",
            help="How many OSINT queries the planner generates. Higher = broader search, but takes longer.",
            disabled=st.session_state.get("is_investigating", False),
        )

        st.slider(
            "Search Depth (Results per Query)",
            min_value=2,
            max_value=15,
            value=st.session_state.get("max_search_results", 5),
            step=1,
            key="max_search_results",
            help="Number of URLs retrieved for each search query. Higher = more entities found.",
            disabled=st.session_state.get("is_investigating", False),
        )

        st.slider(
            "Scrape Text Limit (Chars)",
            min_value=1500,
            max_value=12000,
            value=st.session_state.get("scrape_content_chars_max", 3000),
            step=500,
            key="scrape_content_chars_max",
            help="Max characters extracted per scraped URL.",
            disabled=st.session_state.get("is_investigating", False),
        )

        st.slider(
            "Max Re-planning Iterations",
            min_value=0,
            max_value=5,
            value=st.session_state.get("max_iterations", 2),
            step=1,
            key="max_iterations",
            help="How many times the Quality Gate can route back to the Planner if data is thin. Higher = deeper but slower.",
            disabled=st.session_state.get("is_investigating", False),
        )

        st.divider()
        st.subheader("🕸️ Graph Expansion")

        st.slider(
            "Lateral Expansion Depth",
            min_value=0,
            max_value=2,
            value=st.session_state.get("lateral_expansion_depth", 1),
            step=1,
            key="lateral_expansion_depth",
            help="0 = Star topology. 1 = Find links between secondary entities. 2 = Deep recursion.",
            disabled=st.session_state.get("is_investigating", False),
        )

        st.slider(
            "Max Entities to Expand",
            min_value=1,
            max_value=5,
            value=st.session_state.get("max_entities_to_expand", 3),
            step=1,
            key="max_entities_to_expand",
            help="Limit how many secondary entities the planner researches to prevent infinite sprawl.",
            disabled=st.session_state.get("is_investigating", False),
        )

        st.divider()

        def _on_start():
            if st.session_state.get("target_input", "").strip():
                st.session_state.is_investigating = True

        st.button(
            "🚀 Start Investigation",
            type="primary",
            disabled=st.session_state.get("is_investigating", False),
            on_click=_on_start,
        )
