"""
Briefing agent — generates a proactive intelligence brief after ingestion.

Combines graph-structural analysis (degree centrality, timeline, entity types)
with an LLM narrative to produce a 3-paragraph executive brief.
"""

import streamlit as st
from langchain_core.messages import HumanMessage

from src.database.graph_writer import KnowledgeGraph
from src.llm import get_briefing_llm
from src.logger import log_step
from src.prompts.templates import BRIEFING_PROMPT
from src.state import InvestigatorState


def briefing_node(state: InvestigatorState) -> dict:
    """Generates a proactive intelligence brief after ingestion using graph analysis + LLM narrative."""
    investigation_id = state.get("investigation_id", "default")
    target_name = state.get("target_name", "Unknown")
    entities = state.get("extracted_entities", [])
    relationships = state.get("extracted_relationships", [])

    st.toast("📋 Auto-Briefing Agent generating intelligence summary...", icon="🧠")

    log_step(
        "Briefing",
        f"Generating intelligence brief for **{target_name}** "
        f"({len(entities)} entities, {len(relationships)} rels)",
        level="think",
    )

    # --- Phase 1: Graph structural analysis ---
    graph = KnowledgeGraph(investigation_id)

    pruned = graph.prune_orphans()
    if pruned:
        log_step(
            "Briefing", f"Pruned {pruned} orphaned nodes from graph", level="prune"
        )
        st.toast(f"🧹 Pruned {pruned} orphaned nodes from the graph.", icon="✂️")

    G = graph.graph
    analysis_parts = []

    # Key entities by degree centrality
    if G.number_of_nodes() > 0:
        degrees = dict(G.degree())
        top_entities = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:7]
        associates_list = [
            f"- **{G.nodes.get(nid, {}).get('name', nid)}** "
            f"({G.nodes.get(nid, {}).get('type', 'Unknown')}): {deg} connections"
            for nid, deg in top_entities
        ]
        analysis_parts.append(
            "### Key Entities by Connection Count\n" + "\n".join(associates_list)
        )

    # Chronological timeline of dated edges
    timeline = []
    for u, v, data in G.edges(data=True):
        date = data.get("date", "")
        if date and date.strip():
            u_name = G.nodes.get(u, {}).get("name", u)
            v_name = G.nodes.get(v, {}).get("name", v)
            desc = data.get("description", "related to")
            timeline.append(
                {"date": str(date), "text": f"**{u_name}** {desc} **{v_name}**"}
            )

    if timeline:
        timeline.sort(key=lambda x: x["date"])
        timeline_lines = [f"- *{ev['date']}*: {ev['text']}" for ev in timeline[:10]]
        analysis_parts.append(
            "### Chronological Timeline\n" + "\n".join(timeline_lines)
        )

    # Entity type breakdown
    type_counts: dict[str, int] = {}
    for _, data in G.nodes(data=True):
        etype = data.get("type", "Unknown")
        type_counts[etype] = type_counts.get(etype, 0) + 1
    if type_counts:
        breakdown = [
            f"- {t}: {c}"
            for t, c in sorted(type_counts.items(), key=lambda x: x[1], reverse=True)
        ]
        analysis_parts.append("### Entity Type Breakdown\n" + "\n".join(breakdown))

    structural_analysis = (
        "\n\n".join(analysis_parts)
        if analysis_parts
        else "No structural data available."
    )

    # --- Phase 2: LLM narrative brief ---
    llm = get_briefing_llm()
    prompt = BRIEFING_PROMPT.format(
        target_name=target_name,
        structural_analysis=structural_analysis,
        entity_count=len(entities),
        relationship_count=len(relationships),
    )

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        narrative = response.content
    except Exception as e:
        log_step("Briefing", f"LLM narrative failed: {e}", level="error")
        narrative = (
            "Auto-briefing LLM failed. Please review the structural analysis above."
        )

    full_brief = f"## 🕵️ Intelligence Brief: {target_name}\n\n{narrative}\n\n---\n\n{structural_analysis}"

    st.toast("✅ Intelligence Brief generated.", icon="📋")
    log_step(
        "Briefing",
        f"Brief generated ({len(entities)} entities, {len(relationships)} rels)",
        level="success",
    )

    return {"briefing": full_brief}
