"""
Chat UI component — GraphRAG Q&A interface.
"""

import streamlit as st
from langchain_core.runnables.config import RunnableConfig

from src.callbacks import TraceCallbackHandler
from src.state import InvestigatorState


def render_chat(graph_app):
    """Renders the Q&A chat interface using the compiled LangGraph app."""
    st.header("Interrogate the Intelligence")
    st.markdown(
        "Ask natural language questions. The AI will strictly cite the generated "
        "knowledge graph and scraped documents."
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input(
        "E.g., What shell companies is the target associated with?"
    ):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Agentic Retrievers searching databases..."):
                best_target_name = st.session_state.get(
                    "resolved_target", "Unknown Target"
                )
                investigation_id = st.session_state.get("investigation_id", "default")
                query_state: InvestigatorState = {
                    "investigation_id": investigation_id,
                    "investigation_target": "",
                    "target_name": best_target_name,
                    "current_phase": "chat",
                    "chat_question": prompt,
                    "queries": [],
                    "raw_context": [],
                    "extracted_entities": [],
                    "extracted_relationships": [],
                    "synthesized_answer": "",
                    "briefing": "",
                    "iteration_count": 0,
                    "max_iterations": 0,
                }
                chat_trace = TraceCallbackHandler()
                chat_config = RunnableConfig(callbacks=[chat_trace])
                final_chat_state = graph_app.invoke(query_state, config=chat_config)
                answer = final_chat_state.get(
                    "synthesized_answer", "Error retrieving answer."
                )

            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
