# RFP Intelligence Platform

> **AI-powered RAG Search Engine & Multi-Agent System for RFP Documents**  
> Built for the Emplay AI Engineering Assignment

---

## 🎬 System Walkthrough & Demo Video
📺 **[Watch the Live System Demo & Walkthrough (Google Drive)](https://drive.google.com/file/d/1xszMSy_67kkY-JLspTxWJRhFQMdYek9o/view?usp=sharing)**

---

## 📋 Table of Contents
1. [Demo Video](#-system-walkthrough--demo-video)
2. [Architecture](#architecture)
3. [Quick Start](#quick-start)
4. [Setup](#setup)
5. [Usage](#usage)
6. [Design Decisions](#design-decisions)
7. [Evaluation Results](#evaluation-results)
8. [Agent Trace Example](#agent-trace-example)
9. [Project Structure](#project-structure)
10. [Assumptions & Limitations](#assumptions--limitations)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    RFP Intelligence Platform                      │
├─────────────────────────────────────────────────────────────────┤
│  INPUT: HTML / PDF files (any bid folder)                        │
│           │                                                      │
│           ▼                                                      │
│  ┌──────────────────────┐                                        │
│  │  Ingestion & Parsing  │  ← PyMuPDF + pdfplumber + BS4        │
│  │  (ingestion/)         │    trafilatura                        │
│  └──────────┬───────────┘                                        │
│             │ ParsedDocument (pages, tables, metadata)           │
│             ▼                                                    │
│  ┌──────────────────────┐                                        │
│  │  Chunking            │  ← Section-aware + Table-aware        │
│  │  (ingestion/chunker) │    chunk_size=512, overlap=64         │
│  └──────────┬───────────┘                                        │
│             │ DocumentChunk[]                                    │
│             ▼                                                    │
│  ┌──────────────────────┐   ┌────────────────────────────────┐   │
│  │  Vector Embeddings   │   │  BM25 Index (per-bid, on-disk) │   │
│  │  nomic-embed-text    │   │  rank_bm25 library             │   │
│  │  via Ollama          │   └────────────────────────────────┘   │
│  └──────────┬───────────┘             │                          │
│             │                         │                          │
│             ▼                         ▼                          │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              MongoDB Atlas Vector Store                   │   │
│  │  - $vectorSearch (ANN cosine similarity)                 │   │
│  │  - Metadata indexes: bid_id, doc_type, addendum_number   │   │
│  │  - Incremental upsert (no full re-index needed)          │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
├──────────── RAG SEARCH ENGINE ─────────────────────────────────-─┤
│                                                                  │
│  Query ──► Query Expansion ──► Vector Search ──┐                 │
│                             └► BM25 Search ────┤                 │
│                                                ▼                 │
│                                    RRF Fusion (alpha=0.6)        │
│                                                │                 │
│                                                ▼                 │
│                                    Cross-encoder Reranking       │
│                                    (ms-marco-MiniLM-L-6-v2)     │
│                                                │                 │
│                                                ▼                 │
│                                    Ranked Results + Citations    │
│                                                                  │
├──────────── MULTI-AGENT SYSTEM (LangGraph) ─────────────────────-┤
│                                                                  │
│  User Goal ──► [Orchestrator / LangGraph]                        │
│                       │                                          │
│         ┌─────────────┼──────────────┐                           │
│         ▼             ▼              ▼                           │
│  [Ingestion      [Retrieval     [Extraction                      │
│   Agent]          Agent]         Agent(s)]                       │
│         │             │              │                           │
│         └─────────────┼──────────────┘                           │
│                       ▼                                          │
│              [Addendum Reconciliation Agent]                     │
│                       │                                          │
│                       ▼                                          │
│              [Validator / Critic Agent]                          │
│                  │         │                                     │
│             PASS │         │ FAIL → retry (max 3)               │
│                  ▼         ▼                                     │
│              [Q&A / Report Agent]                                │
│                       │                                          │
│         ┌─────────────┴──────────────┐                           │
│         ▼                            ▼                           │
│  Structured JSON              Cited Natural                      │
│  (bid_extraction.json)        Language Answer                    │
└─────────────────────────────────────────────────────────────────┘
```

### API / UI Layer
```
FastAPI REST API (/index, /search, /ask, /extract)
      │
Streamlit Web UI (4 tabs: Search, Q&A Chat, Extract, Results)
```

---

## Quick Start

### ⚡ One-Command Start (Recommended for Evaluators)

Run **one command** to launch the entire platform:

```bash
python run.py
```
*(Or simply double-click `run.bat` on Windows, or run `./run.sh` on Linux/macOS)*

This command automatically:
1. Validates your Python environment (3.10+)
2. Initializes `.env` from `.env.example` if not already present
3. Verifies and installs missing dependencies
4. Launches the interactive Streamlit Web UI at **`http://localhost:8501`**

---

### 🛠️ Manual Step-by-Step Setup

```bash
# 1. Clone and install
git clone https://github.com/altamashforreal/Emplay-Assignment.git
cd Emplay-Assignment
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env with your MongoDB Atlas URI and Ollama settings

# 3. Pull Ollama models
ollama pull llama3.2
ollama pull nomic-embed-text

# 4. Create MongoDB Atlas Vector Search Index (see Setup section)

# 5. Index a bid folder
python main.py index --bid ./data/bids/Bid1 --bid-id Bid1

# 6. Run extraction
python main.py extract --bid ./data/bids/Bid1 --bid-id Bid1 --output ./output

# 7. Ask questions
python main.py ask --question "What is the submission deadline?" --bid-id Bid1

# 8. Start the web UI
streamlit run ui/app.py

# 9. Start the REST API
python main.py api
```

---

## Setup

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.ai) installed and running locally
- [MongoDB Atlas](https://cloud.mongodb.com) account (free tier works)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
```

Edit `.env`:
```env
# MongoDB Atlas connection string
MONGODB_URI=mongodb+srv://user:password@cluster.mongodb.net/

# Ollama (default localhost)
OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL=llama3.2
EMBEDDING_MODEL=nomic-embed-text
```

### 3. Pull Ollama Models

```bash
ollama pull llama3.2          # LLM for agents
ollama pull nomic-embed-text   # Embedding model
```

### 4. Create MongoDB Atlas Vector Search Index

In the Atlas UI → Search → Create Index → JSON Editor:

```json
{
  "fields": [
    {
      "type": "vector",
      "path": "embedding",
      "numDimensions": 768,
      "similarity": "cosine"
    },
    {
      "type": "filter",
      "path": "bid_id"
    },
    {
      "type": "filter",
      "path": "doc_type"
    },
    {
      "type": "filter",
      "path": "addendum_number"
    }
  ]
}
```

**Index name**: `vector_index` (must match `MONGODB_VECTOR_INDEX` in `.env`)

### 5. Prepare Bid Folders

```
data/
  bids/
    Bid1/
      ├── BidNetDirect_Bid1.html
      ├── JA-207652_FINAL.pdf
      ├── Addendum_1.pdf
      └── Addendum_2.pdf
    Bid2/
      ├── BidNetDirect_Bid2.html
      ├── PORFP_Dell_Laptop_Final.pdf
      ├── Dell_Laptop_Specs.pdf
      ├── Contract_Affidavit.pdf
      └── Mercury_Affidavit.pdf
```

### 6. Run Tests

```bash
pytest tests/ -v --cov=ingestion --cov=search
```

---

## Usage

### CLI Commands

```bash
# Index a new bid folder (incremental – won't re-index if already done)
python main.py index --bid ./data/bids/Bid1 --bid-id Bid1
python main.py index --bid ./data/bids/Bid2 --bid-id Bid2

# Force re-index (deletes existing data)
python main.py index --bid ./data/bids/Bid1 --bid-id Bid1 --force

# Full extraction (produces JSON output)
python main.py extract --bid ./data/bids/Bid1 --bid-id Bid1 --output ./output

# Search
python main.py search --query "bid bond requirements" --bid-id Bid1 --top-k 5
python main.py search --query "warranty specifications" --doc-type specs

# Q&A
python main.py ask --question "What is the submission deadline after all addendums?" --bid-id Bid1
python main.py ask --question "Compare the warranty requirements of both bids."

# Run evaluation
python main.py eval

# Start REST API
python main.py api --port 8000

# Start Web UI
streamlit run ui/app.py
```

### REST API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/index` | Index a bid folder |
| GET | `/search?q=...&bid_id=...&top_k=5` | Hybrid search |
| POST | `/ask` | Q&A with citations |
| POST | `/extract` | Full extraction pipeline |
| GET | `/bids` | List indexed bids |
| DELETE | `/bids/{bid_id}` | Remove bid index |

API docs available at `http://localhost:8000/docs`

### Output JSON Format

```json
{
  "bid_id": "Bid1",
  "fields": {
    "Due Date": {
      "value": "2024-02-15 14:00 EST",
      "sources": [{"file": "Addendum_2.pdf", "page": 1}],
      "confidence": 0.93,
      "notes": "Extended by Addendum 2 (original: 2024-01-30)"
    },
    "Bid Bond Requirement": {
      "value": null,
      "sources": [],
      "confidence": 0.80,
      "notes": "Not found in documents"
    }
  },
  "addendum_changes": [...],
  "validation": {"passed": 18, "failed": 0, "not_found": 2}
}
```

---

## Design Decisions

### Chunking Strategy
- **Size**: 512 characters (~128-150 tokens). Chosen as a balance between retrieving focused snippets and preserving enough context for the LLM. Smaller chunks improve precision; larger chunks improve recall.
- **Strategy**: Section/heading-aware first (split at numbered headings, ALL-CAPS headings), then size-based splitting within sections. This ensures section boundaries are not broken mid-topic.
- **Overlap**: 64 characters to prevent boundary information loss (e.g., a value that spans two chunks).
- **Tables**: Kept as single atomic chunks — never split across chunks — since table rows are interdependent.

### Embedding Model
- **nomic-embed-text** (via Ollama): Open-source, 768-dimensional, optimized for retrieval tasks. Runs locally (no API cost), competitive with OpenAI text-embedding-ada-002 on MTEB benchmarks.

### Vector Store
- **MongoDB Atlas**: Chosen for unified storage (metadata + vectors in one place), ACID transactions, and the ability to combine vector search with structured metadata filters in a single query via the `$vectorSearch` pipeline. Incremental indexing via `upsert` — new bids don't require re-indexing existing ones.

### Hybrid Search
- **Vector + BM25**: Vector search excels at semantic similarity; BM25 excels at exact keyword matches (model numbers, bid numbers, part numbers). Both are needed for RFP documents.
- **Reciprocal Rank Fusion**: No score normalization required (unlike linear combination), robust to different score scales.
- **alpha=0.6**: Slight preference for semantic search, tuned for the RFP domain.

### Re-ranking
- **cross-encoder/ms-marco-MiniLM-L-6-v2**: Lightweight cross-encoder that computes relevance scores for (query, passage) pairs. Applied to top-10 candidates to produce top-5 final results.

### Agent Framework
- **LangGraph**: Chosen for explicit stateful graph with conditional edges, making the feedback loop (Validator → Orchestrator → Retrieval/Extraction retry) easy to implement and reason about. Unlike CrewAI/AutoGen, LangGraph makes data flow explicit and testable.

### LLM
- **Ollama + llama3.2**: Free, local, no API costs. Can be swapped for GPT-4o or Claude by changing `.env` (the client is abstracted behind `OllamaLLM`).

---

## Evaluation Results

> Run `python main.py eval` to generate live results. Below is the expected format:

| Configuration | Recall@1 | Recall@5 | MRR | nDCG@5 |
|--------------|----------|----------|-----|--------|
| Vector Only | 0.XX | 0.XX | 0.XX | 0.XX |
| BM25 Only | 0.XX | 0.XX | 0.XX | 0.XX |
| Hybrid (RRF) | 0.XX | 0.XX | 0.XX | 0.XX |
| Hybrid + Rerank | 0.XX | 0.XX | 0.XX | 0.XX |

*Results will be populated after running `python main.py eval` with indexed bid data.*

The evaluation uses 20 curated question→expected-keyword pairs covering:
- 8 Bid1-specific questions
- 8 Bid2-specific questions  
- 4 cross-bid comparison questions

---

## Agent Trace Example

See `output/Bid1_trace.json` after running extraction. Format:
```json
[
  {
    "agent": "IngestionAgent",
    "input": {"bid_id": "Bid1", "bid_folder": "./data/bids/Bid1"},
    "output": {"chunks_indexed": 342, "files_parsed": 4},
    "latency": 45.2,
    "status": "success"
  },
  {
    "agent": "RetrievalAgent",
    "input": {"fields": ["Due Date", "Bid Bond Requirement", ...]},
    "output": {"Due Date": 5, "Bid Bond Requirement": 3, ...},
    "latency": 12.1,
    "status": "success"
  },
  ...
]
```

---

## Project Structure

```
rfp-intelligence/
├── main.py                  # CLI entry point
├── config.py                # Centralized settings (pydantic-settings)
├── models.py                # Shared Pydantic data models
├── requirements.txt
├── .env.example
│
├── ingestion/               # Part A – Document parsing
│   ├── pdf_parser.py        # PyMuPDF + pdfplumber
│   ├── html_parser.py       # BeautifulSoup + trafilatura
│   └── chunker.py           # Section-aware, table-aware chunking
│
├── search/                  # Part B – RAG Search Engine
│   ├── embedder.py          # Ollama nomic-embed-text
│   ├── vector_store.py      # MongoDB Atlas vector store
│   ├── bm25_index.py        # BM25 keyword index (per-bid)
│   ├── retrieval.py         # Hybrid search + RRF + reranking
│   └── indexing.py          # End-to-end indexing pipeline
│
├── agents/                  # Part C – Multi-Agent System
│   ├── orchestrator.py      # LangGraph graph + RFPOrchestrator
│   ├── llm_client.py        # Ollama LLM wrapper
│   ├── ingestion_agent.py   # File parsing + indexing
│   ├── retrieval_agent.py   # Evidence retrieval per field
│   ├── extraction_agent.py  # LLM-based field extraction
│   ├── addendum_agent.py    # Addendum reconciliation
│   ├── validator_agent.py   # Field validation + retry logic
│   └── qa_agent.py          # Q&A + final JSON assembly
│
├── api/                     # REST API (FastAPI)
│   └── app.py
│
├── ui/                      # Web UI (Streamlit)
│   └── app.py
│
├── eval/                    # Retrieval evaluation
│   ├── evaluator.py         # Recall@k, MRR, nDCG metrics
│   └── qa_log.py            # Sample Q&A log
│
├── tests/                   # Unit + integration tests
│   ├── conftest.py
│   ├── test_ingestion.py
│   └── test_search.py
│
├── data/                    # Bid folders go here
│   ├── bids/
│   │   ├── Bid1/
│   │   └── Bid2/
│   └── bm25_indices/        # BM25 index files (auto-created)
│
├── output/                  # Extraction JSON outputs
└── logs/                    # Log files
```

---

## Assumptions & Limitations

### Assumptions
1. Ollama is running locally (`http://localhost:11434`) with `llama3.2` and `nomic-embed-text` pulled.
2. MongoDB Atlas vector search index named `vector_index` is created with the schema in the Setup section.
3. Bid folders contain only HTML and PDF files (other types are skipped gracefully).
4. Addendum files are named with "addendum" in the filename for automatic type detection.

### Limitations
1. **Scanned PDFs**: No OCR support (scanned pages are flagged but not read). OCR via `pytesseract` is a planned bonus.
2. **Very large documents**: Documents >500 pages may be slow to index due to Ollama's embedding throughput.
3. **Hallucination**: The LLM may occasionally extract incorrect values from ambiguous text. The validator catches low-confidence extractions.
4. **MongoDB Atlas dependency**: The vector search requires Atlas; a `FAISS` fallback could be added for fully offline use.
5. **LLM quality**: Using `llama3.2` locally; GPT-4o would produce significantly better extraction accuracy.

### Known Improvements (Future Work)
- OCR integration for scanned PDFs
- Semantic caching of repeated queries
- Bid comparison agent (side-by-side report)
- Go/no-go recommendation agent
- Docker Compose deployment
- CI pipeline with GitHub Actions
