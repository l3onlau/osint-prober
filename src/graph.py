"""
LangGraph state machine — wires together all agent nodes with conditional routing.

Flow:
  START → route_phase ──┬── plan → gather → deduplicate → quality_gate ──┬── briefing → synthesize → END
                        │                                                 └── increment → plan (retry)
                        └── synthesize → END  (chat mode)
"""

from langgraph.graph import END, START, StateGraph

from src.agents.briefing import briefing_node
from src.agents.deduplicator import deduplicator_node
from src.agents.gatherer import gatherer_node
from src.agents.planner import query_planner_node
from src.agents.expansion_planner import expansion_planner_node
from src.agents.synthesis import synthesis_node
from src.state import InvestigatorState
from src.logger import log_step
from src.config import config

# ---------------------------------------------------------------------------
# Relevancy Validation (ModernBERT NLI)
# ---------------------------------------------------------------------------

_nli_pipeline = None


def _is_relevant_nli(extracted_text: str, target: str) -> bool:
    """Uses a local NLI cross-encoder to map semantic entailment (Relevancy check)."""
    global _nli_pipeline
    if _nli_pipeline is None:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from transformers import pipeline

            _nli_pipeline = pipeline(
                "text-classification", model=config.nli_model, device="cpu"
            )

    hypothesis = f"These entities provide meaningful intelligence regarding {target}."
    try:
        result = _nli_pipeline({"text": extracted_text, "text_pair": hypothesis})
        return result["label"] in ["entailment", "neutral"]
    except Exception as e:
        log_step("Quality Gate", f"NLI pipeline error: {e}", level="warning")
        return True  # Fail open if the model crashes


# ---------------------------------------------------------------------------
# Conditional edge functions
# ---------------------------------------------------------------------------


def route_phase(state: InvestigatorState) -> str:
    """Routes to ingestion pipeline or straight to Q&A based on `current_phase`."""
    return "synthesize" if state.get("current_phase") == "chat" else "plan"


def quality_gate(state: InvestigatorState) -> str:
    """Evaluates extraction quality — re-plans if data is too thin or irrelevant.

    Requires at least 3 entities AND 2 relationships to proceed.
    Also uses an LLM to check if the extracted entities actually relate to the target.
    """
    target = state.get("target_name", state.get("investigation_target", "Unknown"))
    entities = state.get("extracted_entities", [])
    iteration = state.get("iteration_count", 0)
    max_iter = state.get("max_iterations", 2)

    ent_count = len(entities)

    if iteration >= max_iter:
        log_step(
            "Quality Gate",
            f"Max iterations reached ({iteration}). Proceeding to briefing with what we have.",
            level="warning",
        )
        return "brief"

    if ent_count == 0:
        log_step(
            "Quality Gate",
            f"No entities extracted in iteration {iteration}. Re-planning...",
            level="warning",
        )
        return "replan"

    # Evaluate relevancy using NLI Cross-Encoder
    try:
        extracted_text = "Extracted data: " + ", ".join(
            [e.get("name", "") for e in entities]
        )

        is_relevant = _is_relevant_nli(extracted_text, target)

        if not is_relevant:
            log_step(
                "Quality Gate",
                "Data contradicts investigation relevance criteria (NLI Check Failed). Re-planning...",
                level="warning",
            )
            return "replan"

        log_step(
            "Quality Gate",
            f"Data passes relevancy entailment against target '{target}'. Proceeding.",
            level="success",
        )
    except Exception as e:
        log_step(
            "Quality Gate", f"Relevancy check failed: {e}. Proceeding.", level="warning"
        )

    depth = state.get("current_expansion_depth", 0)
    max_depth = state.get("lateral_expansion_depth", 1)

    if depth < max_depth:
        return "expand"
    return "brief"


def expansion_gate(state: InvestigatorState) -> str:
    """Decides if the expansion planner produced any queries."""
    queries = state.get("queries", [])

    if queries:
        return "gather"

    return "brief"


def increment_iteration(state: InvestigatorState) -> dict:
    """Bumps the iteration counter before cycling back to re-plan."""
    return {"iteration_count": state.get("iteration_count", 0) + 1}


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_investigator_graph():
    """Compiles and returns the LangGraph state machine."""
    builder = StateGraph(InvestigatorState)

    # Nodes
    builder.add_node("plan", query_planner_node)
    builder.add_node("gather", gatherer_node)
    builder.add_node("deduplicate", deduplicator_node)
    builder.add_node("expansion_planner", expansion_planner_node)
    builder.add_node("increment", increment_iteration)
    builder.add_node("briefing", briefing_node)
    builder.add_node("synthesize", synthesis_node)

    # Entry routing
    builder.add_conditional_edges(
        START,
        route_phase,
        {
            "plan": "plan",
            "synthesize": "synthesize",
        },
    )

    # Ingestion pipeline
    builder.add_edge("plan", "gather")

    # We loop gather -> deduplicate -> expand -> gather
    builder.add_edge("gather", "deduplicate")

    # Quality gate → expand OR re-plan
    builder.add_conditional_edges(
        "deduplicate",
        quality_gate,
        {
            "expand": "expansion_planner",
            "replan": "increment",
            "brief": "briefing",  # max iterations reached
        },
    )

    # Expansion gate -> gather OR brief
    builder.add_conditional_edges(
        "expansion_planner",
        expansion_gate,
        {
            "gather": "gather",
            "brief": "briefing",
        },
    )

    # Feedback loop
    builder.add_edge("increment", "plan")

    # Terminals
    builder.add_edge("briefing", "synthesize")
    builder.add_edge("synthesize", END)

    return builder.compile()
