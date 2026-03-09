"""
Planner agent — identifies the root entity and generates OSINT search queries.
"""

import json

import streamlit as st
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage

from src.llm import get_planner_llm
from src.logger import log_step
from src.database.graph_writer import KnowledgeGraph
from src.prompts.templates import PLANNER_HUMAN_PROMPT, PLANNER_SYSTEM_PROMPT
from src.state import InvestigatorState
from src.tools.scrapers import get_duckduckgo_tool, get_wikipedia_tool
from src.utils import strip_think_tags


def query_planner_node(state: InvestigatorState) -> dict:
    """Uses an Agentic loop to identify the exact person from a vague description, then generates queries."""
    llm = get_planner_llm()
    investigation_target = state.get("investigation_target", "")

    agent = create_agent(llm, [get_duckduckgo_tool(), get_wikipedia_tool()])

    st.toast(
        "🧠 Agentic Ingestion initiated. Planner is identifying the Root Entity...",
        icon="🤖",
    )

    log_step(
        "Planner",
        f"Identifying root entity for: '{investigation_target}'...",
        level="think",
    )

    investigation_id = state.get("investigation_id", "default")

    # Load existing context so the planner knows what we already have
    try:
        graph = KnowledgeGraph(investigation_id)
        existing_entities = graph.get_all_entity_summaries()
        if len(existing_entities) > 3000:
            existing_entities = existing_entities[:3000] + "... [truncated]"
    except Exception:
        existing_entities = "No existing entities."

    try:
        messages = [
            SystemMessage(
                content=PLANNER_SYSTEM_PROMPT.format(
                    investigation_target=investigation_target,
                    query_count=state.get("planner_query_count", 5),
                    existing_entities=existing_entities,
                )
            ),
            HumanMessage(
                content=PLANNER_HUMAN_PROMPT.format(
                    investigation_target=investigation_target
                )
            ),
        ]

        response = agent.invoke({"messages": messages})
        content = strip_think_tags(response["messages"][-1].content)

        log_step(
            "Planner", f"Raw LLM output received ({len(content)} chars)", level="info"
        )

        # Extract JSON from output
        start_idx = content.find("{")
        end_idx = content.rfind("}") + 1

        if start_idx != -1 and end_idx > 0:
            data = json.loads(content[start_idx:end_idx])
            resolved_target = data.get("target_name", investigation_target)
            queries = data.get("queries", [])
            st.toast(f"✅ Root Entity Resolved: {resolved_target}", icon="🎯")
            log_step(
                "Planner", f"Resolved target → **{resolved_target}**", level="success"
            )
            log_step(
                "Planner",
                f"Generated {len(queries)} queries: {queries}",
                level="search",
            )
            return {
                "target_name": resolved_target,
                "queries": (
                    queries if len(queries) > 0 else [f"{resolved_target} background"]
                ),
            }
        else:
            log_step(
                "Planner",
                "JSON not found in LLM output — using raw target as fallback",
                level="warning",
            )
            return {
                "target_name": investigation_target,
                "queries": [f"{investigation_target} background"],
            }

    except Exception as e:
        log_step("Planner", f"Failed: {e}", level="error")
        return {
            "target_name": investigation_target,
            "queries": [f"{investigation_target} background"],
        }
