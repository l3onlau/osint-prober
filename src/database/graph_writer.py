"""
Knowledge Graph layer — wraps NetworkX for entity/relationship persistence.

Provides:
  - Entity upsert with deduplication
  - Relationship upsert with fuzzy ID matching
  - BFS-based context retrieval for the Synthesis agent
  - Orphan pruning to clean disconnected nodes

Each investigation is namespaced under:
  {config.data_dir}/{investigation_id}/graph.graphml
"""

import os
import re
import difflib

import networkx as nx

from src.config import config
from src.schemas.extraction import Entity, Relationship
from src.logger import log_step

# ---------------------------------------------------------------------------
# ID normalization
# ---------------------------------------------------------------------------

_PREFIX_PATTERN = re.compile(
    r"^(person|organization|location|business|event|date|financial_amount):",
    re.IGNORECASE,
)


def normalize_id(entity_id: str) -> str:
    """Normalizes an entity ID to a URL-safe slug.

    Strips common LLM-inserted prefixes (e.g., 'person:'), lowercases,
    and replaces non-alphanumerics with hyphens.
    """
    if not entity_id:
        return "unknown"
    cid = _PREFIX_PATTERN.sub("", entity_id).lower()
    cid = re.sub(r"[^a-z0-9]+", "-", cid).strip("-")
    return cid


# ---------------------------------------------------------------------------
# KnowledgeGraph
# ---------------------------------------------------------------------------


class KnowledgeGraph:
    """Local NetworkX wrapper for the OSINT entity graph.

    Each investigation stores its graph in a separate directory:
      {config.data_dir}/{investigation_id}/graph.graphml
    """

    def __init__(self, investigation_id: str):
        self._investigation_id = investigation_id
        self._graph_path = os.path.join(
            config.data_dir, investigation_id, "graph.graphml"
        )
        self.graph = nx.DiGraph()
        self._path_ensured: bool = False
        self._load()

    # — Persistence —

    def _load(self):
        if os.path.exists(self._graph_path):
            self.graph = nx.read_graphml(self._graph_path)

    def _save(self):
        if not self._path_ensured:
            os.makedirs(os.path.dirname(self._graph_path), exist_ok=True)
            self._path_ensured = True
        nx.write_graphml(self.graph, self._graph_path)

    # — Entity operations —

    def add_entities(self, entities: list[Entity]):
        """Upserts entities as graph nodes, merging summaries on collision."""
        for e in entities:
            nid = normalize_id(e.id)
            if not self.graph.has_node(nid):
                self.graph.add_node(nid, name=e.name, type=e.type, summary=e.summary)
            else:
                nx.set_node_attributes(
                    self.graph, {nid: {"summary": e.summary, "type": e.type}}
                )
        self._save()

    # — Relationship operations —

    def _fuzzy_match_node(self, norm_id: str) -> str:
        """Attempts to snap a normalized ID to an existing node.

        Strategy: exact match → substring match (IDs > 4 chars) → name match.
        """
        if self.graph.has_node(norm_id):
            return norm_id

        # 2. Strict similarity match using difflib
        # Only match if they are very similar (e.g., > 0.85 ratio) to avoid merging short/distinct entities
        for existing in self.graph.nodes:
            if difflib.SequenceMatcher(None, norm_id, existing).ratio() > 0.85:
                log_step(
                    "Graph",
                    f"Fuzzy match (ID sim > 0.85): '{norm_id}' → '{existing}'",
                    level="info",
                )
                return existing

        # 3. Match by name similarity
        for existing, data in self.graph.nodes(data=True):
            existing_name = normalize_id(data.get("name", ""))
            if (
                existing_name
                and difflib.SequenceMatcher(None, norm_id, existing_name).ratio() > 0.85
            ):
                log_step(
                    "Graph",
                    f"Fuzzy match by name (sim > 0.85): '{norm_id}' → '{existing}'",
                    level="info",
                )
                return existing

        return norm_id

    def add_relationships(self, relationships: list[Relationship]):
        """Upserts directed edges with fuzzy ID resolution."""
        for r in relationships:
            src = self._fuzzy_match_node(normalize_id(r.source_entity_id))
            tgt = self._fuzzy_match_node(normalize_id(r.target_entity_id))

            if src == tgt:
                continue  # skip self-loops

            if not self.graph.has_edge(src, tgt):
                self.graph.add_edge(
                    src, tgt, description=r.description, date=r.date or ""
                )
            else:
                existing = self.graph.edges[src, tgt].get("description", "")
                if r.description not in existing:
                    merged = (
                        f"{existing}; {r.description}" if existing else r.description
                    )
                    nx.set_edge_attributes(
                        self.graph, {(src, tgt): {"description": merged}}
                    )
        self._save()

    # — Retrieval —

    def get_summary_context(self, target_id: str, depth: int = 1) -> str:
        """Returns a textual summary of n-hop connections for LLM context."""
        if not self.graph.has_node(target_id):
            return "No structured graph data found for this entity."

        lines = []
        for u, v in nx.bfs_edges(self.graph, target_id, depth_limit=depth):
            edge = self.graph.get_edge_data(u, v)
            v_data = self.graph.nodes[v]
            desc = edge.get("description", "connected to")
            lines.append(
                f"- {u} is {desc} {v_data.get('name', v)} ({v_data.get('type', 'Unknown')})"
            )

        return (
            "\n".join(lines)
            if lines
            else f"{target_id} exists but has no known connections."
        )

    def get_all_entity_summaries(self) -> str:
        """Returns a comma-separated list of existing entities for the extraction prompt."""
        if not self.graph.nodes:
            return "No existing entities."
        return ", ".join(
            f"Name: '{data.get('name', nid)}' (ID: {nid})"
            for nid, data in self.graph.nodes(data=True)
        )

    # — Maintenance —

    def prune_orphans(self) -> int:
        """Removes zero-degree nodes. Returns count of pruned nodes."""
        orphans = [n for n, deg in dict(self.graph.degree()).items() if deg == 0]
        for orphan in orphans:
            name = self.graph.nodes[orphan].get("name", orphan)
            log_step("Graph", f"Pruned orphan: {name} ({orphan})", level="prune")
            self.graph.remove_node(orphan)

        if orphans:
            self._save()
        return len(orphans)
