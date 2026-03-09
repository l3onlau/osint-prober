"""
Dashboard UI component — graph visualisation and entity/timeline stats.

Reads investigation_id from session_state to load the correct namespaced graph.
"""

import os

import networkx as nx
import streamlit as st
from streamlit_agraph import Config, Edge, Node, agraph

from src.database.graph_writer import KnowledgeGraph
from src.config import config as app_config


def render_dashboard():
    """Renders the briefing, interactive graph, and stat columns."""
    if st.session_state.get("briefing"):
        st.markdown(st.session_state["briefing"])
        st.markdown("---")

    investigation_id = st.session_state.get("investigation_id", "")
    if not investigation_id:
        st.info("No connections discovered yet. Run an ingestion phase.")
        return

    graph_path = os.path.join(app_config.data_dir, investigation_id, "graph.graphml")
    if not os.path.exists(graph_path):
        st.info("No connections discovered yet. Run an ingestion phase.")
        return

    try:
        G = KnowledgeGraph(investigation_id).graph
    except Exception as e:
        st.error(f"Could not read graph database: {e}")
        return

    try:
        centrality = nx.degree_centrality(G)
    except Exception:
        centrality = {n: 0.1 for n in G.nodes()}

    # Distinct colors for different entity types
    COLOR_MAP = {
        "Person": "#3498DB",  # Blue
        "Organization": "#2ECC71",  # Green
        "Location": "#E74C3C",  # Red
        "Event": "#9B59B6",  # Purple
        "Company": "#1ABC9C",  # Teal
        "Website": "#34495E",  # Navy
        "Role": "#F39C12",  # Orange
    }
    DEFAULT_COLOR = "#95A5A6"  # Grey

    nodes = []
    for n_id, data in G.nodes(data=True):
        raw_name = data.get("name", n_id)
        # Fix slug fallback: if the name is identical to the slug id, it probably wasn't
        # extracted with proper casing. Replace hyphens and use title case.
        display_name = (
            raw_name.replace("-", " ").title() if raw_name == n_id else raw_name
        )

        ent_type = data.get("type", "Unknown")
        node_color = COLOR_MAP.get(ent_type, DEFAULT_COLOR)

        # Scale size by centrality (base size: 15, max addition: 25)
        # This makes highly connected nodes visually larger
        cent_score = centrality.get(n_id, 0.0)
        node_size = 15 + int(cent_score * 25)

        nodes.append(
            Node(
                id=n_id,
                label=display_name,
                title=f"{ent_type}: {data.get('summary', '')}",
                color=node_color,
                # High contrast text (black text with white stroke for readability on any background)
                font={
                    "color": "black",
                    "size": 14,
                    "strokeWidth": 2,
                    "strokeColor": "white",
                },
                size=node_size,
            )
        )
    edges = [
        Edge(
            source=u,
            target=v,
            label="",
            title=data.get("description", "related"),
            color="#A6ACAF",  # Subtle grey for edges
        )
        for u, v, data in G.edges(data=True)
    ]

    graph_config = Config(
        width="100%",
        height=800,
        directed=True,
        physics=True,
        nodeHighlightBehavior=True,
        highlightColor="#F39C12",  # A nice amber highlight
        collapsible=False,
        hierarchical=False,
        node={"labelProperty": "label"},
        link={"labelProperty": "label", "renderLabel": True},
        # Spread the nodes out more so it's less cluttered
        layout={"hierarchical": False},
        interaction={"hover": True},
        physics_layout={
            "forceAtlas2Based": {
                "gravitationalConstant": -150,
                "centralGravity": 0.01,
                "springLength": 150,
                "springConstant": 0.08,
            },
            "minVelocity": 0.75,
            "solver": "forceAtlas2Based",
        },
    )

    st.subheader("Intelligence Connections")
    if nodes:
        with st.container(border=True):
            agraph(nodes=nodes, edges=edges, config=graph_config)
        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Top Associated Entities")
            degrees = dict(G.degree())
            for n, d in sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:5]:
                nd = G.nodes[n]
                st.write(
                    f"- **{nd.get('name', n)}** ({nd.get('type', 'Unknown')}): {d} connections"
                )

        with col2:
            st.markdown("### Known Timeline Events")
            events = [
                {
                    "date": data["date"],
                    "desc": (
                        f"**{G.nodes[u].get('name', u)}** "
                        f"{data.get('description', 'related to')} "
                        f"**{G.nodes[v].get('name', v)}**"
                    ),
                }
                for u, v, data in G.edges(data=True)
                if data.get("date")
            ]
            events.sort(key=lambda x: str(x["date"]))
            if events:
                for ev in events[:10]:
                    st.write(f"- *{ev['date']}*: {ev['desc']}")
            else:
                st.info("No dated events extracted.")
    else:
        st.info("Graph is currently empty.")
