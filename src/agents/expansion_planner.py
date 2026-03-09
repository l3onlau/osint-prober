"""
Expansion Planner agent — generates lateral search queries based on existing graph entities.
"""

import json

import streamlit as st
from langchain_core.messages import HumanMessage, SystemMessage

from src.llm import get_planner_llm
from src.logger import log_step
from src.database.graph_writer import KnowledgeGraph
from src.prompts.templates import (
    EXPANSION_PLANNER_HUMAN_PROMPT,
    EXPANSION_PLANNER_SYSTEM_PROMPT,
)
from src.state import InvestigatorState
from src.utils import strip_think_tags


def expansion_planner_node(state: InvestigatorState) -> dict:
    """Uses LLM to evaluate the current graph and generate depth-2 lateral queries."""
    llm = get_planner_llm()
    investigation_target = state.get("investigation_target", "")
    investigation_id = state.get("investigation_id", "default")
    max_entities = state.get("max_entities_to_expand", 3)

    st.toast(
        "🕸️ Lateral Expansion initiated. Planner is looking for cross-connections...",
        icon="🤖",
    )

    log_step(
        "Expansion Planner",
        f"Evaluating graph for lateral expansion (Depth {state.get('current_expansion_depth', 0) + 1}/{state.get('lateral_expansion_depth', 1)})...",
        level="think",
    )

    # Load existing context so the planner knows what we already have
    try:
        graph = KnowledgeGraph(investigation_id)
        # We want the planner to focus on actual extracted entities, not just the root
        existing_entities = graph.get_all_entity_summaries()
        if len(existing_entities) > 3000:
            existing_entities = existing_entities[:3000] + "... [truncated]"
    except Exception:
        existing_entities = "No existing entities."

    try:
        messages = [
            SystemMessage(
                content=EXPANSION_PLANNER_SYSTEM_PROMPT.format(
                    investigation_target=investigation_target,
                    existing_entities=existing_entities,
                    max_entities=max_entities,
                )
            ),
            HumanMessage(
                content=EXPANSION_PLANNER_HUMAN_PROMPT.format(max_entities=max_entities)
            ),
        ]

        response = llm.invoke(messages)
        content = strip_think_tags(response.content)

        log_step(
            "Expansion Planner",
            f"Raw LLM output received ({len(content)} chars)",
            level="info",
        )

        # Extract JSON from output
        start_idx = content.find("{")
        end_idx = content.rfind("}") + 1

        if start_idx != -1 and end_idx > 0:
            data = json.loads(content[start_idx:end_idx])
            queries = data.get("queries", [])
            log_step(
                "Expansion Planner",
                f"Generated {len(queries)} lateral queries: {queries}",
                level="search",
            )
            return {
                "queries": queries,
                "current_expansion_depth": state.get("current_expansion_depth", 0) + 1,
            }
        else:
            log_step(
                "Expansion Planner",
                "JSON not found in LLM output — no expansion queries generated",
                level="warning",
            )
            return {
                "queries": [],
                "current_expansion_depth": state.get("current_expansion_depth", 0) + 1,
            }

    except Exception as e:
        log_step("Expansion Planner", f"Failed: {e}", level="error")
        return {
            "queries": [],
            "current_expansion_depth": state.get("current_expansion_depth", 0) + 1,
        }
