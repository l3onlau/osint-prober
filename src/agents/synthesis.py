"""
Synthesis agent — agentic Q&A over the local knowledge base.

Uses search_vector_db, search_graph_db, and (as a last resort) live_web_search
to answer user questions with strict citations.
"""

import re

import streamlit as st
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from src.database.graph_writer import KnowledgeGraph, normalize_id
from src.database.vector_store import get_ensemble_retriever
from src.llm import get_synthesis_llm
from src.logger import log_step
from src.prompts.templates import SYNTHESIS_SYSTEM_PROMPT
from src.state import InvestigatorState


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


def synthesis_node(state: InvestigatorState) -> dict:
    """Uses a Tool-Calling ReAct agent to autonomously retrieve and synthesize data."""
    question = state.get("chat_question", "")
    target_name = state.get("target_name", "Unknown")
    investigation_id = state.get("investigation_id", "default")

    if not question:
        if state.get("current_phase") == "ingestion":
            return {
                "synthesized_answer": "Ingestion Phase Complete. Data stored in local Graph and Vector databases."
            }
        return {"synthesized_answer": "Please ask a question."}

    # ---------------------------------------------------------------------------
    # Tools (closures over investigation_id for namespaced retrieval)
    # ---------------------------------------------------------------------------

    @tool
    def search_vector_db(query: str) -> str:
        """Searches the unstructured web documents database for a semantic or keyword query.
        Use this to find specific paragraphs, quotes, dates, or financial amounts."""
        try:
            docs = get_ensemble_retriever(query, investigation_id=investigation_id, k=3)
            return (
                "\n\n".join([doc.page_content for doc in docs])
                if docs
                else "No unstructured documents found."
            )
        except Exception as e:
            return f"Database error: {e}"

    @tool
    def search_graph_db(entity_name: str) -> str:
        """Searches the structured Knowledge Graph for known connections (1-hop and 2-hop) to the specified entity.
        Use this to find who or what a business or person is connected to."""
        try:
            graph = KnowledgeGraph(investigation_id)
            target_slug = normalize_id(entity_name)
            return graph.get_summary_context(target_slug, depth=2)
        except Exception as e:
            return f"Graph database error: {e}"

    @tool
    def live_web_search(query: str) -> str:
        """Searches the live web for additional information not found in local databases.
        Use this ONLY after search_vector_db and search_graph_db have returned insufficient results.
        """
        try:
            log_step(
                "Synthesis",
                f"⚠️ Live web fallback triggered for: '{query}' — result is un-vetted",
                level="warning",
            )
            from src.tools.scrapers import get_duckduckgo_tool

            return get_duckduckgo_tool().invoke(query)
        except Exception as e:
            return f"Live search error: {e}"

    llm = get_synthesis_llm()
    agent = create_agent(llm, [search_vector_db, search_graph_db, live_web_search])

    st.toast(
        "🧠 Agentic Retrieval initiated. Synthesizer is querying tools autonomously...",
        icon="🤖",
    )
    log_step("Synthesis", f"Answering: '{question}'", level="think")

    try:
        messages = [
            SystemMessage(
                content=SYNTHESIS_SYSTEM_PROMPT.format(target_name=target_name)
            ),
            HumanMessage(content=question),
        ]

        response = agent.invoke({"messages": messages})
        final_answer = response["messages"][-1].content

        # Post-processing faithfulness check
        is_insufficient = "insufficient public information" in final_answer.lower()
        has_citation = re.search(r"\[\s*citation:\s*.*?\]", final_answer, re.IGNORECASE)

        if len(final_answer) > 50 and not is_insufficient and not has_citation:
            final_answer += (
                "\n\n⚠️ **WARNING: The Agentic Synthesizer failed to provide strict citations "
                "for this response. Consider this information unverified hallucination until "
                "manually checked.**"
            )
            log_step(
                "Synthesis",
                "Citation check FAILED — hallucination warning appended",
                level="warning",
            )
        else:
            log_step(
                "Synthesis",
                "Response generated with citations or acknowledged missing info",
                level="success",
            )

        return {"synthesized_answer": final_answer}

    except Exception as e:
        log_step("Synthesis", f"Agent error: {e}", level="error")
        return {
            "synthesized_answer": f"Failed to generate answer due to Agentic LLM error: {e}"
        }
