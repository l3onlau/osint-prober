"""
Vector and sparse retrieval layer.

Provides:
  - ChromaDB (dense semantic search)
  - BM25 (sparse keyword search)
  - FlashRank reranker (ensemble fusion)

All paths are namespaced per investigation:
  {config.data_dir}/{investigation_id}/chroma/
  {config.data_dir}/{investigation_id}/bm25.pkl
"""

import os
import pickle

import chromadb
import streamlit as st
from langchain_chroma import Chroma
from langchain_community.document_compressors import FlashrankRerank
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

from src.config import config

# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


def get_embeddings():
    """Returns the Ollama embedding model configured in settings."""
    return OllamaEmbeddings(model=config.embedding_model)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _chroma_path(investigation_id: str) -> str:
    return os.path.join(config.data_dir, investigation_id, "chroma")


def _bm25_path(investigation_id: str) -> str:
    return os.path.join(config.data_dir, investigation_id, "bm25.pkl")


# ---------------------------------------------------------------------------
# ChromaDB (dense vectors)
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def _get_persistent_client(path: str):
    """Caches the ChromaDB client across Streamlit reruns to prevent SQLite tenant lock errors."""
    return chromadb.PersistentClient(path=path)


def get_chroma_db(investigation_id: str):
    """Initializes or reconnects to the persistent Chroma collection for this investigation."""
    path = _chroma_path(investigation_id)
    os.makedirs(path, exist_ok=True)

    client = _get_persistent_client(path)

    return Chroma(
        client=client,
        collection_name="osint_documents",
        embedding_function=get_embeddings(),
    )


def add_texts_to_chroma(texts: list[str], metadatas: list[dict], investigation_id: str):
    """Embeds and persists new text chunks into the investigation's ChromaDB."""
    if not texts:
        return
    db = get_chroma_db(investigation_id)
    db.add_texts(texts=texts, metadatas=metadatas)


# ---------------------------------------------------------------------------
# BM25 (sparse keywords)
# ---------------------------------------------------------------------------


def save_bm25_retriever(texts: list[str], investigation_id: str):
    """Rebuilds and pickles the BM25 index from the given text corpus."""
    path = _bm25_path(investigation_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    bm25 = BM25Retriever.from_texts(texts)
    with open(path, "wb") as f:
        pickle.dump(bm25, f)


def load_bm25_retriever(investigation_id: str):
    """Loads the pickled BM25 retriever, or returns None if absent."""
    path = _bm25_path(investigation_id)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Ensemble retriever (Chroma + BM25 → FlashRank rerank)
# ---------------------------------------------------------------------------


def get_ensemble_retriever(
    query: str, investigation_id: str, k: int = 5
) -> list[Document]:
    """Searches both dense and sparse stores, deduplicates, then reranks."""
    chroma_docs = get_chroma_db(investigation_id).similarity_search(query, k=k)

    bm25 = load_bm25_retriever(investigation_id)
    bm25_docs = bm25.invoke(query) if bm25 else []

    # Deduplicate by content
    unique = {doc.page_content: doc for doc in chroma_docs + bm25_docs}
    docs = list(unique.values())

    if len(docs) < 2:
        return docs  # FlashrankRerank requires ≥2 docs for pairwise comparison

    compressor = FlashrankRerank(model=config.flashrank_model)
    return compressor.compress_documents(docs, query)
