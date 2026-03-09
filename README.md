# OSINT Investigation Assistant

An AI-powered Open-Source Intelligence (OSINT) tool that autonomously investigates a person of interest. Given a name or vague description, the system searches the web, extracts structured intelligence, builds a knowledge graph, and generates a proactive intelligence brief — all running locally.

## Features

- **Fully Agentic Pipeline**: Every stage (Planning, Gathering, Expansion, Briefing, Synthesis) is powered by autonomous LangGraph ReAct agents that reason, search, evaluate, and adapt.
- **Agentic Ingestion**: Accepts vague descriptions (e.g., *"The Malaysian financier from the 1MDB scandal"*) and autonomously identifies the correct target.
- **Lateral Graph Expansion**: Employs an Expansion Planner to autonomously cross-reference secondary entities, breaking out of "star topology" intelligence gathering to find deep, multi-hop links.
- **Proactive Intelligence Brief**: After ingestion, automatically surfaces key associates, timelines, and investigation gaps without the user needing to ask.
- **Continuous Workspaces**: Accumulate intelligence across multiple search targets into a single, cohesive namespace. Simply use the same **Workspace ID** for different targets, and the system mechanically links isolated graphs together, discovering hidden relationships between completely distinct investigations.
- **Hallucination Mitigation**: Uses a dedicated, high-context Natural Language Inference (NLI) Cross-Encoder (`ModernBERT-base-nli`) to rigorously verify semantic entailment between raw text and extracted relationships, completely replacing expensive LLM-as-a-judge patterns.
- **Relevancy Verification**: Automatically drops useless generic website boilerplate, advertisements, and completely unrelated news articles using the same NLI Cross-Encoder, guaranteeing all extracted data actually correlates to the investigation target.
- **GraphRAG Q&A**: Natural language chat grounded by Knowledge Graph traversal, semantic vector search, BM25 keyword matching, and FlashRank reranking.
- **Zero-Config Local Storage**: All data persists locally via NetworkX (GraphML), ChromaDB, and BM25 (Pickle). NLI and Rerankers are pulled automatically via HuggingFace.`
- **Observability**: Built-in Agent Log tab shows every reasoning step in real-time — no external tracing tools required.

## Requirements

- Python >= 3.12
- [`uv`](https://docs.astral.sh/uv/) for dependency management
- [Ollama](https://ollama.ai/) with the following models pulled:
  ```bash
  ollama pull qwen3:4b-instruct
  ollama pull qwen3-embedding:0.6b
  ```

## Setup & Run

1. Install dependencies:
   ```bash
   uv sync
   ```

2. Start the application:
   ```bash
   streamlit run app.py
   ```

3. Open `http://localhost:8501` in your browser.

## Configuration

All settings are centralized in `src/config.py` using Pydantic Settings. Override any value via environment variables or a `.env` file:

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_MODEL` | `qwen3:4b-instruct` | LLM model for reasoning |
| `EMBEDDING_MODEL` | `qwen3-embedding:0.6b` | Embedding model for vector search |
| `FLASHRANK_MODEL` | `ms-marco-MiniLM-L-12-v2` | Cross-encoder model for dense/sparse reranking |
| `NLI_MODEL` | `tasksource/ModernBERT-base-nli` | HuggingFace NLI model for faithfulness validation |
| `GRAPH_DB_PATH` | `./data/graph.graphml` | Knowledge Graph file path |
| `CHROMA_DB_PATH` | `./data/chroma` | ChromaDB persistence directory |
| `NEWSAPI_KEY` | `None` | (Optional) NewsAPI org key for news querying |
| `HIBP_API_KEY` | `None` | (Optional) HaveIBeenPwned API key for breach lookups |
| `FIRECRAWL_API_KEY` | `None` | (Optional) Firecrawl API key for JS-heavy web scraping |

## Project Structure

```
├── app.py                    # Streamlit UI entry point
├── ARCHITECTURE.md           # Architecture document (handoff guide)
├── src/
│   ├── config.py             # Centralized Pydantic Settings
│   ├── graph.py              # LangGraph state machine with feedback loop
│   ├── state.py              # InvestigatorState TypedDict
│   ├── agents/
│   │   ├── planner.py        # ReAct Agent — target resolution & query planning
│   │   ├── gatherer.py       # ReAct Agent — autonomous web search & extraction
│   │   ├── expansion_planner.py# ReAct Agent — lateral depth-2 query generation
│   │   ├── briefing.py       # Auto-Briefing Agent — proactive insights
│   │   └── synthesis.py      # ReAct Agent — GraphRAG Q&A with drill-down
│   ├── database/
│   │   ├── graph_writer.py   # NetworkX graph operations
│   │   └── vector_store.py   # Chroma, BM25 ensemble retrieval
│   ├── prompts/
│   │   └── templates.py      # Extraction prompt with faithfulness instructions
│   ├── schemas/
│   │   └── extraction.py     # Pydantic models for Entity, Relationship
│   └── tools/
│       └── scrapers.py       # DuckDuckGo, Wikipedia, HTML scraping tools
└── data/                     # Auto-generated local persistence (gitignored)
```

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed design decisions, extension points, and known limitations.

## Data Sources

| Source | Purpose | Justification |
|---|---|---|
| **DuckDuckGo** | General web search | No API key required, broad coverage |
| **Wikipedia** | Biographical/organizational context | Structured, high-quality factual data |
| **Full Page Scraping** | Deep content from specific URLs | Richer intelligence than search snippets |
| **Firecrawl (Optional)** | Robust SPA & JS Scraping | Handles modern anti-bot setups and renders markdown perfectly |