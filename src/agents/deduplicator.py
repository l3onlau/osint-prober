"""
Deduplicator agent — deterministic graph-level entity deduplication.

Runs after the Gatherer and before the quality gate to merge near-duplicate
nodes (same name or high slug overlap) without an LLM call.

Improvements in this version:
  - investigation_id threaded through to KnowledgeGraph
  - Type-compatibility guard: only merge nodes of the same entity type
  - Short-slug safety: minimum 6 chars for slug comparison (avoids
    false positives like 'ibm' ↔ 'ibi' or 'eu' ↔ 'ec')
"""

import itertools
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage

from src.database.graph_writer import KnowledgeGraph, normalize_id
from src.llm import get_llm
from src.logger import log_step
from src.state import InvestigatorState


def _levenshtein(a: str, b: str) -> int:
    """Computes Levenshtein edit distance between two strings."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(
                min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + (0 if ca == cb else 1))
            )
        prev = curr
    return prev[-1]


def _similarity(a: str, b: str) -> float:
    """Returns a [0, 1] name similarity score (1 = identical)."""
    max_len = max(len(a), len(b), 1)
    return 1.0 - _levenshtein(a, b) / max_len


_MERGE_THRESHOLD = 0.85  # names with similarity ≥ this are merged automatically
_LLM_MERGE_THRESHOLD = 0.40  # names with similarity between LLM_MERGE_THRESHOLD and MERGE_THRESHOLD trigger an LLM check
_MIN_SLUG_LENGTH = 6  # don't compare slugs shorter than this (prevents false positives)


class MergeDecision(BaseModel):
    should_merge: bool = Field(
        description="True if the two entities refer to the exact same real-world entity, False otherwise."
    )
    reason: str = Field(description="Brief reason for the decision.")


def _ask_llm_to_merge(a_name: str, a_context: str, b_name: str, b_context: str) -> bool:
    """Uses LLM to decide if two entities are actually the same."""
    llm = get_llm(temperature=0.0).with_structured_output(MergeDecision)
    prompt = f"""
Are the following two entities the exact same real-world entity?
You should merge if they are obvious aliases or variations (e.g., "Bill Gates" and "William H. Gates III", or "IBM" and "International Business Machines").
Do not merge if they are distinct entities with similar names, or if you are unsure.

Entity A: "{a_name}"
Summary/Context A: {a_context}

Entity B: "{b_name}"
Summary/Context B: {b_context}
"""
    try:
        messages = [
            SystemMessage(
                content="You are an expert intelligence analyst entity resolution AI."
            ),
            HumanMessage(content=prompt),
        ]
        decision = llm.invoke(messages)
        log_step(
            "Deduplicator",
            f"LLM decided merge={decision.should_merge} for '{a_name}' & '{b_name}'. Reason: {decision.reason}",
            level="think",
        )
        return decision.should_merge
    except Exception as e:
        log_step("Deduplicator", f"LLM merge check failed: {e}", level="warning")
        return False


def deduplicator_node(state: InvestigatorState) -> dict:
    """Merges near-duplicate graph nodes based on name similarity.

    Guards:
      - Slug length ≥ _MIN_SLUG_LENGTH before comparison
      - Entity types must match before merging (Person ≠ Organization)

    Returns an empty dict (state is unchanged; mutations happen
    in-place on disk via KnowledgeGraph._save).
    """
    investigation_id = state.get("investigation_id", "default")
    graph = KnowledgeGraph(investigation_id)
    G = graph.graph

    if G.number_of_nodes() < 2:
        log_step(
            "Deduplicator", "Fewer than 2 nodes — nothing to deduplicate.", level="info"
        )
        return {}

    merged_count = 0

    # Iterate over pairs; rebuild candidate list after each merge
    changed = True
    while changed:
        changed = False
        node_list = [
            (nid, G.nodes[nid].get("name", nid), G.nodes[nid].get("type", ""))
            for nid in list(G.nodes)
        ]
        for (aid, aname, atype), (bid, bname, btype) in itertools.combinations(
            node_list, 2
        ):
            if not G.has_node(aid) or not G.has_node(bid):
                continue

            a_slug = normalize_id(aname)
            b_slug = normalize_id(bname)

            # Guard 1: skip very short slugs to prevent false positives
            if len(a_slug) < _MIN_SLUG_LENGTH or len(b_slug) < _MIN_SLUG_LENGTH:
                continue

            # Guard 2: only merge nodes of the same entity type
            if atype and btype and atype.lower() != btype.lower():
                continue

            sim = _similarity(a_slug, b_slug)

            should_merge = False
            if aid != bid:
                if sim >= _MERGE_THRESHOLD:
                    should_merge = True
                elif sim >= _LLM_MERGE_THRESHOLD:
                    a_summary = G.nodes[aid].get("summary", "")
                    b_summary = G.nodes[bid].get("summary", "")
                    # Only ask LLM if we have some context on both
                    if a_summary and b_summary:
                        should_merge = _ask_llm_to_merge(
                            aname, a_summary, bname, b_summary
                        )

            if should_merge:
                # Keep the higher-degree node
                keep = aid if G.degree(aid) >= G.degree(bid) else bid
                drop = bid if keep == aid else aid
                drop_name = G.nodes[drop].get("name", drop)
                keep_name = G.nodes[keep].get("name", keep)

                # Re-point all edges that touched `drop` → `keep`
                for pred in list(G.predecessors(drop)):
                    if pred != keep and not G.has_edge(pred, keep):
                        edge_data = G.edges[pred, drop]
                        G.add_edge(pred, keep, **edge_data)
                for succ in list(G.successors(drop)):
                    if succ != keep and not G.has_edge(keep, succ):
                        edge_data = G.edges[drop, succ]
                        G.add_edge(keep, succ, **edge_data)

                G.remove_node(drop)
                merged_count += 1
                changed = True
                log_step(
                    "Deduplicator",
                    f"Merged '{drop_name}' ({drop}) → '{keep_name}' ({keep}) "
                    f"[sim={sim:.2f}, type={atype or 'unknown'}]",
                    level="prune",
                )
                break  # restart the pair scan

    if merged_count:
        graph._save()
        log_step(
            "Deduplicator", f"Merged {merged_count} duplicate node(s).", level="success"
        )
    else:
        log_step("Deduplicator", "No duplicates detected.", level="info")

    return {}
