# Architecture Document: OSINT Investigation Assistant

> Written as a handoff to engineers extending this system.

## 1. What This System Does

Given a person's name (or even a vague description like *"The Malaysian financier from the 1MDB scandal"*), the system:

1. **Identifies** the exact person of interest using live web search
2. **Gathers** publicly available intelligence autonomously
3. **Extracts** structured entities and relationships into a Knowledge Graph
4. **Generates** a proactive Intelligence Brief (timeline, key associates, gaps)
5. **Allows** natural language Q&A against the gathered intelligence

All processing runs locally using Ollama. No data leaves the machine.

---

## 2. High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        Streamlit UI (app.py)                     │
│   ┌─────────────┐  ┌──────────────────┐  ┌───────────────────┐  │
│   │ Sidebar      │  │ Dashboard Tab    │  │ GraphRAG Q&A Tab  │  │
│   │ (Target      │  │ (Graph Viz +     │  │ (Chat Interface)  │  │
│   │  Input)      │  │  Auto-Brief)     │  │                   │  │
│   └──────┬───────┘  └────────▲─────────┘  └────────▲──────────┘  │
└──────────┼───────────────────┼─────────────────────┼─────────────┘
           │                   │                     │
           ▼                   │                     │
┌──────────────────────────────┼─────────────────────┼─────────────┐
│          LangGraph State Machine (src/graph.py)    │             │
│                              │                     │             │
│  ┌─────────┐   ┌──────────┐ │  ┌──────────────┐   │  ┌───────┐  │
│  │ Planner │──▶│ Gatherer │─┼─▶│   Briefing   │───┼─▶│Synth. │  │
│  │ (ReAct) │   │ (ReAct)  │ │  │   Agent      │   │  │(ReAct)│  │
│  └────▲────┘   └────┬─▲───┘ │  └──────────────┘   │  └───────┘  │
│       │             │ │     │                     │             │
│       │   ┌─────────▼─┴─┐   │                     │             │
│       │   │   Quality   │   │                     │             │
│       └───│    Gate     │   │                     │             │
│ (if thin) └──────┬──────┘   │                     │             │
│                  │ (if depth < max_depth)         │             │
│           ┌──────▼──────┐   │                     │             │
│           │  Expansion  │   │                     │             │
│           │   Planner   │   │                     │             │
│           └─────────────┘   │                     │             │
└──────────────────────────────┴─────────────────────┴─────────────┘
           │                                │
           ▼                                ▼
┌────────────────────────────────────────────────────────────────┐
│                    Data Layer (src/database/)                    │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────┐               │
│  │ NetworkX     │  │ ChromaDB     │  │ BM25      │               │
│  │ (GraphML)    │  │ (Dense Vec.) │  │ (Sparse)  │               │
│  │ ./data/graph │  │ ./data/chroma│  │ ./data/bm │               │
│  └──────────────┘  └──────────────┘  └───────────┘               │
└────────────────────────────────────────────────────────────────┘
```

---

## 3. Why These Decisions

### Why NetworkX over Neo4j?
- **Zero-config**: No database server to install or manage. The graph persists as a single `.graphml` file.
- **Embeddable**: Runs in-process with the Python application. Perfect for a locally-runnable prototype.
- **Trade-off**: Not suitable for production-scale graphs (millions of nodes). For this prototype's scope (hundreds of entities per investigation), it is more than sufficient.

### Why Chroma + BM25 Ensemble over a single vector DB?
- **Semantic coverage** (Chroma): Finds contextually relevant passages even when exact keywords differ. Uses `Qwen3-Embedding-0.6B` which provides strong multilingual MTEB performance.
- **Keyword coverage** (BM25): Catches exact names, dates, and financial figures that embeddings often miss.
- **FlashrankRerank**: Results from both retrievers are deduplicated and re-ranked using the ultra-lightweight and fast `cross-encoder/ms-marco-MiniLM-L-12-v2` cross-encoder for optimal relevance (Note: this model is English-only and not multilingual).
- **Trade-off**: Running two retrieval backends adds latency. For this prototype, the quality improvement justifies it.

### Why the Lateral Expansion Planner?
- **Problem**: Querying just the primary target creates a "Star Topology" graph (everything links to the center, nothing links laterally).
- **Solution**: A secondary ReAct planner acts upon the current graph, selecting high-value secondary entities and writing multi-hop queries linking them together. This iterates back into the gatherer.
- **Effect**: Creates deep, interconnected webs of intelligence automatically.

### Why ReAct Agents over LangChain Chains?
- **Autonomous decision-making**: Agents can evaluate their own results and decide whether to search more, scrape a page, or stop.
- **Tool-calling**: LangGraph's `create_agent` gives the LLM native function-calling over our custom tools, which is far more flexible than a fixed chain.
- **Trade-off**: ReAct loops with a 4B parameter model can sometimes loop excessively or produce malformed tool calls. We mitigate this with `recursion_limit` and structured output parsing.

### Why Qwen3-4B-Instruct-2507?
- **Instruction-following**: Strong structured output compliance for JSON extraction. High IFEval and MultiPL-E scores.
- **Tool-calling support**: Natively supports the function-calling format required by `create_agent` with strong BFCL-v3 results.
- **Size**: Small enough to run on consumer hardware via Ollama, yet capable enough for multi-step reasoning.

### Why NLI and Vector Math over LLM-as-a-Judge?
- **Problem**: Sub-7B models (like a 4B parameter model) severely struggle when acting as a secondary "judge" over long contexts, suffering from high false negatives, context saturation, and excessive token usage overhead.
- **Solution**: We use a high-context Natural Language Inference cross-encoder (`tasksource/ModernBERT-base-nli`) with an 8,192 token window to prove both **Faithfulness** (Does the scraped text entail the extracted relationship?) and **Relevancy** (Do the extracted entities provide meaningful intelligence regarding the target?).
- **Effect**: This completely offloads verification to a specialized, deterministic, and lightning-fast mathematical logic gate. The generative 4B model is freed to focus exclusively on JSON extraction logic, dropping the pipeline's failure rates dramatically. No brittle cosine/noise thresholds are needed.

---

## 4. Key Components

### Agents (`src/agents/`)

| Agent | Type | Purpose |
|---|---|---|
| **Planner** (`planner.py`) | ReAct Agent | Resolves vague target descriptions to exact names via web search, then generates initial queries. |
| **Gatherer** (`gatherer.py`) | ReAct Agent | Autonomously searches web/Wikipedia, scrapes pages, extracts entities, evaluates data. Uses Firecrawl if API key present. |
| **Expansion Planner** (`expansion_planner.py`) | ReAct Agent | Evaluates existing graph entities and generates depth-2 lateral queries to find inner connections. |
| **Briefing** (`briefing.py`) | Analysis Node | Runs graph centrality analysis + LLM narrative to generate a proactive Intelligence Brief. |
| **Synthesis** (`synthesis.py`) | ReAct Agent | Answers user questions by autonomously querying the graph DB, vector DB, and live web. |

### Data Stores (`src/database/`)

| Store | Library | File | Purpose |
|---|---|---|---|
| Knowledge Graph | NetworkX | `./data/graph.graphml` | Structured entity-relationship storage |
| Dense Vectors | ChromaDB | `./data/chroma/` | Semantic similarity search over scraped text |
| Sparse Index | BM25 (Pickle) | `./data/bm25.pkl` | Exact keyword matching |

### State Machine (`src/graph.py`)

The LangGraph flow includes a **feedback loop** and an **expansion loop**:

```text
plan → gather → deduplicate → quality_gate
                                 ├── (NLI Relevancy Fail) → increment → plan (retry)
                                 └── (Valid Intelligence) → expansion_gate
                                                      ├── (depth < max) → expansion_planner → gather (loop)
                                                      └── (depth == max) → briefing → synthesize → END
```

`quality_gate` checks relevancy using the NLI Entailment model. `expansion_gate` checks if the user's requested Lateral Expansion depth has been reached.

---

## 5. How to Extend

### Adding a New Data Source

To add a completely new data source (for example, searching Reddit via PRAW):

1. **Create a tool function** in `src/tools/scrapers.py`:
   ```python
   def get_reddit_tool():
       # Handle PRAW authentication and return a LangChain-compatible tool
       pass
   ```

2. **Register it in the Gatherer agent** (`src/agents/gatherer.py`):
   ```python
   @tool
   def reddit_search(query: str) -> str:
       """Searches Reddit subreddits for mentions of the target.
       Use this to find community discussions and unverified claims."""
       # ... PRAW logic ...
       pass
   
   # Add to the conditionally-loaded tools list:
   tools = [web_search, wiki_search, scrape_page, extract_and_save]
   if config.reddit_client_id:
       tools.append(reddit_search)
   ```

3. The Gatherer agent will automatically discover and use the new tool based on its docstring when the configuration criteria is met.

### Adding a New Entity Type

1. **Update the `Entity` schema** in `src/schemas/extraction.py`:
   ```python
   type: str = Field(..., description="Must be one of: 'Person', 'Organization', 'Location', 'Event', 'FinancialInstrument'")
   ```

2. **Update the extraction prompt** in `src/prompts/templates.py` to instruct the LLM about the new type.

3. The graph writer and dashboard will automatically handle the new type — no further changes needed since they read entity types dynamically.

### Observability (Agent Log Tab)

The built-in **Agent Log** tab displays the full reasoning trace of every agent decision:
- 🧠 Planning decisions and target resolution
- 🔍 Search queries and results
- 📦 Extraction counts and faithfulness validation drops
- ✂️ Orphan pruning activity
- ⚠️ Quality gate feedback loop triggers
- ✅ / ❌ Success and error outcomes

No external tracing tools are required. All entries are timestamped and displayed newest-first.

---

## 6. Known Limitations

- **Model Capacity**: Qwen3-4B-Instruct-2507 has a limited context window (~4K effective tokens). Large web pages are truncated to 2000 characters before extraction.
- **Entity Resolution**: The current approach uses lexical normalization (`normalize_id`) and LLM context-feeding. A production system would use embedding-based entity linking.
- **Multilingual Verification**: While the general extraction and vector stores (`Qwen3-Embedding`) are deeply multilingual, the Faithfulness NLI model (`ModernBERT-base-nli`) and the Q&A Reranking model (`ms-marco`) are both effectively English-only. Translingual investigations may suffer from arbitrarily dropped relationships at the NLI gate or poor reranking relevancy during Chat.
