"""
Gatherer agent — autonomous OSINT search, scrape, and extraction.

Uses a GathererContext dataclass (instead of module-level globals) to share
mutable state between the node function and the @tool closures in a
thread-safe, run-isolated way.

New in this version:
  - investigation_id namespaces all storage (graph + vectors)
  - scrape_page logged to Progress Log
  - assess_intelligence_coverage tool for self-evaluation
  - Conditional registration of NewsAPI, Wayback, and HIBP tools
"""

from dataclasses import dataclass, field

import html2text
import requests
import streamlit as st
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from src.database.graph_writer import KnowledgeGraph
from src.database.vector_store import add_texts_to_chroma, save_bm25_retriever
from src.llm import get_gatherer_llm
from src.logger import log_step
from src.prompts.templates import (
    EXTRACTION_PROMPT,
    GATHERER_HUMAN_PROMPT,
    GATHERER_SYSTEM_PROMPT,
)
from src.schemas.extraction import Entity, ExtractionResult, Relationship
from src.state import InvestigatorState
from src.config import config
from src.tools.scrapers import (
    get_duckduckgo_tool,
    get_hibp_tool,
    get_newsapi_tool,
    get_wayback_tool,
    get_wikipedia_tool,
    get_firecrawl_tool,
)


# ---------------------------------------------------------------------------
# Per-run context (replaces mutable module-level globals)
# ---------------------------------------------------------------------------


@dataclass
class GathererContext:
    """Holds all mutable state for a single gatherer invocation."""

    investigation_id: str
    target_name: str
    graph: KnowledgeGraph
    max_search_results: int
    scrape_content_chars_max: int
    raw_texts: list[str] = field(default_factory=list)
    entities: list[dict] = field(default_factory=list)
    relationships: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Faithfulness Validation (ModernBERT NLI)
# ---------------------------------------------------------------------------

_nli_pipeline = None


def _is_faithful_nli(premise: str, hypothesis: str) -> bool:
    """Uses a local NLI cross-encoder to map semantic entailment (Faithfulness check)."""
    global _nli_pipeline
    if _nli_pipeline is None:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from transformers import pipeline

            # Uses the model specified in config (defaults to tasksource/ModernBERT-base-nli)
            _nli_pipeline = pipeline(
                "text-classification", model=config.nli_model, device="cpu"
            )

    try:
        result = _nli_pipeline({"text": premise, "text_pair": hypothesis})
        return result["label"] == "entailment"
    except Exception as e:
        log_step("Gatherer", f"NLI pipeline error: {e}", level="warning")
        return True  # Fail open if the model crashes


# ---------------------------------------------------------------------------
# Tool factory — creates tools bound to a specific GathererContext
# ---------------------------------------------------------------------------


def _make_tools(ctx: GathererContext):
    """Returns LangChain tools pre-bound to the given GathererContext."""

    @tool
    def web_search(query: str) -> str:
        """Searches DuckDuckGo for OSINT intelligence. Returns snippets with titles and URLs.
        Use this for broad web searches about a person, organization, or event."""
        try:
            result = get_duckduckgo_tool(max_results=ctx.max_search_results).invoke(
                query
            )
            log_step(
                "Gatherer",
                f"Web search: '{query}' → {len(result)} chars",
                level="search",
            )
            return result if result else "No results found."
        except Exception as e:
            log_step("Gatherer", f"Web search failed: {e}", level="warning")
            return f"Search error: {e}"

    @tool
    def wiki_search(query: str) -> str:
        """Searches Wikipedia for background information on a person or topic.
        Use this to get structured biographical or organizational context."""
        try:
            result = get_wikipedia_tool().invoke(query)
            log_step(
                "Gatherer",
                f"Wiki search: '{query}' → {len(result) if result else 0} chars",
                level="search",
            )
            return result if result else "No Wikipedia article found."
        except Exception as e:
            log_step("Gatherer", f"Wiki search failed: {e}", level="warning")
            return f"Wikipedia error: {e}"

    @tool
    def scrape_page(url: str) -> str:
        """Fetches and cleans the full text content of a web page URL.
        Use this to get detailed information from a specific URL found in search results."""
        try:
            resp = requests.get(
                url, timeout=10, headers={"User-Agent": "OSINT_Prober/1.0"}
            )
            resp.raise_for_status()
            converter = html2text.HTML2Text()
            converter.ignore_links = True
            converter.ignore_images = True
            text = converter.handle(resp.text)
            truncated = text[: ctx.scrape_content_chars_max] if text else ""
            log_step(
                "Gatherer",
                f"Scraped `{url[:60]}…` → {len(truncated)} chars",
                level="extract",
            )
            return truncated or "Could not scrape page content."
        except Exception as e:
            log_step("Gatherer", f"Scrape failed `{url[:60]}`: {e}", level="warning")
            return f"Scraping error: {e}"

    @tool
    def extract_and_save(raw_text: str) -> str:
        """Extracts entities and relationships from raw text using LLM and saves them to the
        Knowledge Graph and Vector databases. Call this after gathering text from search or scrape.
        Returns a summary of what was extracted and how many items passed faithfulness validation.
        """
        if not raw_text or len(raw_text) < 30:
            return "Text too short to extract meaningful intelligence."

        # Persist raw text to vector stores
        ctx.raw_texts.append(raw_text)
        try:
            add_texts_to_chroma(
                [raw_text],
                [{"target": ctx.target_name, "source": "web_search"}],
                investigation_id=ctx.investigation_id,
            )
            save_bm25_retriever(ctx.raw_texts, investigation_id=ctx.investigation_id)
        except Exception as e:
            log_step(
                "Gatherer",
                f"Vector store caching failed (safe to ignore): {e}",
                level="warning",
            )

        # LLM Structured Extraction
        llm = get_gatherer_llm().with_structured_output(ExtractionResult)
        trimmed = raw_text[:2000]
        prompt = EXTRACTION_PROMPT.format(
            target_name=ctx.target_name,
            raw_context=trimmed,
            existing_entities=ctx.graph.get_all_entity_summaries(),
        )

        try:
            result: ExtractionResult = llm.invoke([HumanMessage(content=prompt)])
        except Exception as e:
            return f"Extraction LLM failed: {e}"

        # Process entities
        new_ents = 0
        if result.entities:
            for e in result.entities:
                ctx.entities.append(
                    {"id": e.id, "name": e.name, "type": e.type, "summary": e.summary}
                )
                new_ents += 1

        # Process relationships using ModernBERT NLI faithfulness validation
        new_rels = 0
        dropped = 0
        if result.relationships:
            for r in result.relationships:
                # Format hypothesis for the NLI cross-encoder
                hypothesis = f"{r.source_entity_id} {r.description} {r.target_entity_id}. {r.justifying_quote}"

                if _is_faithful_nli(trimmed, hypothesis):
                    ctx.relationships.append(
                        {
                            "source_entity_id": r.source_entity_id,
                            "target_entity_id": r.target_entity_id,
                            "description": r.description,
                            "date": r.date,
                            "justifying_quote": r.justifying_quote,
                        }
                    )
                    new_rels += 1
                else:
                    dropped += 1
                    log_step(
                        "Gatherer",
                        f"NLI rejected hallucinated relationship: {r.source_entity_id} -> {r.target_entity_id}",
                        level="warning",
                    )

        # Persist to GraphDB
        if new_ents > 0:
            ctx.graph.add_entities([Entity(**x) for x in ctx.entities[-new_ents:]])
        if new_rels > 0:
            ctx.graph.add_relationships(
                [Relationship(**x) for x in ctx.relationships[-new_rels:]]
            )

        log_step(
            "Gatherer",
            f"Extracted {new_ents} entities, {new_rels} rels ({dropped} dropped by NLI). "
            f"Totals: {len(ctx.entities)} entities, {len(ctx.relationships)} rels.",
            level="extract",
        )

        return (
            f"Extracted {new_ents} entities, {new_rels} relationships. "
            f"{dropped} relationships dropped by NLI check. "
            f"Total so far: {len(ctx.entities)} entities, {len(ctx.relationships)} relationships."
        )

    @tool
    def assess_intelligence_coverage() -> str:
        """Evaluates the current state of the intelligence gathered so far.
        Returns a structured report of entity count, relationship count, entity type
        breakdown, and suggestions for what topic areas may still be thin.
        Call this after processing several queries to decide whether to dig deeper or stop.
        """
        G = ctx.graph.graph
        node_count = G.number_of_nodes()
        edge_count = G.number_of_edges()

        # Entity type breakdown
        type_counts: dict[str, int] = {}
        for _, data in G.nodes(data=True):
            etype = data.get("type", "Unknown")
            type_counts[etype] = type_counts.get(etype, 0) + 1

        type_summary = ", ".join(f"{t}: {c}" for t, c in sorted(type_counts.items()))

        # Detect thin areas
        suggestions = []
        if node_count < 5:
            suggestions.append("entity count is low — consider broader searches")
        if edge_count < 3:
            suggestions.append(
                "few relationships found — try searching for connections between known entities"
            )
        if "Person" not in type_counts:
            suggestions.append("no Person entities — try searching for associates")
        if "Organization" not in type_counts:
            suggestions.append(
                "no Organization entities — try searching for companies or institutions"
            )

        report = (
            f"Coverage Report: {node_count} entities ({type_summary}), "
            f"{edge_count} relationships in Knowledge Graph. "
            f"Raw text chunks: {len(ctx.raw_texts)}."
        )
        if suggestions:
            report += " Weak areas: " + "; ".join(suggestions) + "."
        else:
            report += " Coverage looks reasonable — proceed to completion if all queries processed."

        log_step("Gatherer", report, level="think")
        return report

    # Build the base tool list (always active)

    # Check if we should use Firecrawl instead of requests scrape_page
    firecrawl = get_firecrawl_tool(
        scrape_content_chars_max=ctx.scrape_content_chars_max
    )
    if firecrawl:
        log_step(
            "Gatherer",
            "Firecrawl tool active (FIRECRAWL_API_KEY set). Replacing basic scraper.",
            level="info",
        )
        tools = [
            web_search,
            wiki_search,
            firecrawl,
            extract_and_save,
            assess_intelligence_coverage,
        ]
    else:
        tools = [
            web_search,
            wiki_search,
            scrape_page,
            extract_and_save,
            assess_intelligence_coverage,
        ]

    # Conditionally add optional tools
    wayback = get_wayback_tool()
    if wayback:
        tools.append(wayback)

    newsapi = get_newsapi_tool(max_results=ctx.max_search_results)
    if newsapi:
        log_step("Gatherer", "NewsAPI tool active (NEWSAPI_KEY set)", level="info")
        tools.append(newsapi)

    hibp = get_hibp_tool()
    if hibp:
        log_step("Gatherer", "HIBP tool active (HIBP_API_KEY set)", level="info")
        tools.append(hibp)

    return tools


# ---------------------------------------------------------------------------
# Main gatherer node
# ---------------------------------------------------------------------------


def gatherer_node(state: InvestigatorState) -> dict:
    """Agentic Gatherer: uses a ReAct agent to autonomously search, scrape, extract, and evaluate."""
    investigation_id = state.get("investigation_id", "default")
    ctx = GathererContext(
        investigation_id=investigation_id,
        target_name=state.get("target_name", "Unknown"),
        graph=KnowledgeGraph(investigation_id),
        max_search_results=state.get("max_search_results", 5),
        scrape_content_chars_max=state.get("scrape_content_chars_max", 3000),
    )

    queries = state.get("queries", [])
    queries_str = "\n".join([f"  {i + 1}. {q}" for i, q in enumerate(queries)])

    llm = get_gatherer_llm()
    tools = _make_tools(ctx)

    tool_names = [t.name for t in tools]
    log_step(
        "Gatherer",
        f"Deployed with {len(queries)} queries for '{ctx.target_name}' "
        f"| Tools: {', '.join(tool_names)}",
        level="think",
    )
    st.toast(
        "🕵️ Agentic Gatherer deployed. Autonomously searching and extracting...",
        icon="🤖",
    )

    agent = create_agent(llm, tools)

    try:
        messages = [
            SystemMessage(
                content=GATHERER_SYSTEM_PROMPT.format(
                    target_name=ctx.target_name, queries_str=queries_str
                )
            ),
            HumanMessage(
                content=GATHERER_HUMAN_PROMPT.format(
                    target_name=ctx.target_name, num_queries=len(queries)
                )
            ),
        ]
        agent.invoke({"messages": messages})
        log_step(
            "Gatherer",
            f"Complete: {len(ctx.entities)} entities, {len(ctx.relationships)} relationships",
            level="success",
        )
        st.toast(
            f"✅ Agentic Gatherer complete: {len(ctx.entities)} entities, {len(ctx.relationships)} relationships",
            icon="📊",
        )
    except Exception as e:
        log_step("Gatherer", f"Agent error: {e}", level="error")
        st.toast(f"⚠️ Gatherer encountered an error: {e}", icon="⚠️")

    return {
        "raw_context": ctx.raw_texts,
        "extracted_entities": ctx.entities,
        "extracted_relationships": ctx.relationships,
    }
