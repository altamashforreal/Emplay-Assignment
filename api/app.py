"""
FastAPI application – RFP Intelligence Platform REST API.

Endpoints:
  POST /index          – Index a bid folder
  GET  /search         – Hybrid search over indexed bids
  POST /ask            – Q&A mode
  POST /extract        – Full extraction pipeline
  GET  /bids           – List indexed bids
  DELETE /bids/{bid_id} – Remove a bid's index
  GET  /health         – Health check
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel, Field

from config import settings
from models import DocType, SearchQuery
from search.retrieval import HybridSearchEngine
from search.vector_store import MongoVectorStore

# ─── App Setup ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="RFP Intelligence Platform API",
    description="RAG-powered search engine and multi-agent system for RFP documents",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Dependency Injection ─────────────────────────────────────────────────────

_vector_store: Optional[MongoVectorStore] = None
_engine: Optional[HybridSearchEngine] = None


def get_vector_store() -> MongoVectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = MongoVectorStore()
    return _vector_store


def get_engine() -> HybridSearchEngine:
    global _engine
    if _engine is None:
        _engine = HybridSearchEngine(get_vector_store())
    return _engine


# ─── Request/Response Models ──────────────────────────────────────────────────

class IndexRequest(BaseModel):
    bid_folder: str = Field(..., description="Absolute path to the bid folder")
    bid_id: str = Field(..., description="Unique identifier for this bid")
    force_reindex: bool = Field(False, description="Re-index even if already indexed")


class IndexResponse(BaseModel):
    bid_id: str
    status: str
    chunks_indexed: int
    errors: List[str]


class SearchRequest(BaseModel):
    q: str = Field(..., description="Search query")
    bid_id: Optional[str] = None
    doc_types: Optional[List[str]] = None
    top_k: int = Field(5, ge=1, le=50)
    use_rerank: bool = True
    expand_query: bool = True


class AskRequest(BaseModel):
    question: str
    bid_id: Optional[str] = None


class ExtractRequest(BaseModel):
    bid_folder: str
    bid_id: str
    output_dir: Optional[str] = "./output"


# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    return {"status": "ok", "version": "1.0.0"}


@app.post("/index", response_model=IndexResponse)
async def index_bid(request: IndexRequest):
    """Index all documents in a bid folder."""
    from search.indexing import index_bid_folder

    folder = Path(request.bid_folder)
    if not folder.exists():
        raise HTTPException(status_code=404, detail=f"Folder not found: {request.bid_folder}")

    try:
        summary = index_bid_folder(
            folder_path=folder,
            bid_id=request.bid_id,
            vector_store=get_vector_store(),
            force_reindex=request.force_reindex,
        )
        return IndexResponse(
            bid_id=request.bid_id,
            status=summary.get("status", "indexed"),
            chunks_indexed=summary.get("chunks_indexed", 0),
            errors=summary.get("errors", []),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/search")
async def search(
    q: str = Query(..., description="Search query"),
    bid_id: Optional[str] = Query(None),
    doc_type: Optional[str] = Query(None),
    top_k: int = Query(5, ge=1, le=50),
    use_rerank: bool = Query(True),
):
    """Hybrid search over indexed bid documents."""
    try:
        doc_types = None
        if doc_type:
            try:
                doc_types = [DocType(doc_type)]
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid doc_type: {doc_type}")

        search_req = SearchQuery(
            query=q,
            bid_id=bid_id,
            doc_types=doc_types,
            top_k=top_k,
            use_rerank=use_rerank,
            expand_query=True,
        )

        response = get_engine().search(search_req)
        return response.model_dump()

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask")
async def ask_question(request: AskRequest):
    """Answer a natural-language question about bid documents."""
    from agents.qa_agent import answer_question

    try:
        answer = answer_question(
            question=request.question,
            bid_id=request.bid_id,
            engine=get_engine(),
        )
        return {"question": request.question, "answer": answer, "bid_id": request.bid_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/extract")
async def extract_bid(request: ExtractRequest, background_tasks: BackgroundTasks):
    """Run full extraction pipeline for a bid folder (async)."""
    folder = Path(request.bid_folder)
    if not folder.exists():
        raise HTTPException(status_code=404, detail=f"Folder not found: {request.bid_folder}")

    try:
        from agents.orchestrator import RFPOrchestrator
        orchestrator = RFPOrchestrator()
        result = orchestrator.run_extraction(
            bid_folder=request.bid_folder,
            bid_id=request.bid_id,
            output_dir=request.output_dir,
        )
        return JSONResponse(content=json.loads(json.dumps(result, default=str)))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/bids")
async def list_bids():
    """List all indexed bids."""
    vs = get_vector_store()
    pipeline = [
        {"$group": {"_id": "$bid_id", "chunk_count": {"$sum": 1}}},
        {"$sort": {"_id": 1}},
    ]
    bids = list(vs.chunks.aggregate(pipeline))
    return {"bids": [{"bid_id": b["_id"], "chunks": b["chunk_count"]} for b in bids]}


@app.delete("/bids/{bid_id}")
async def delete_bid(bid_id: str):
    """Remove all indexed data for a bid."""
    vs = get_vector_store()
    n = vs.delete_bid(bid_id)

    # Also remove BM25 index files
    from config import settings
    bm25_dir = Path(settings.bm25_index_dir)
    for f in [f"{bid_id}_bm25.pkl", f"{bid_id}_meta.json"]:
        fp = bm25_dir / f
        if fp.exists():
            fp.unlink()

    return {"bid_id": bid_id, "deleted_chunks": n}


if __name__ == "__main__":
    uvicorn.run(
        "api.app:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level=settings.log_level.lower(),
    )
