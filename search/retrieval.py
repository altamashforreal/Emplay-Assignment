"""
Hybrid Retrieval Engine – combines vector search + BM25 via Reciprocal Rank Fusion.
Includes query expansion, metadata filtering, and cross-encoder reranking.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

from loguru import logger

from config import settings
from models import DocType, SearchQuery, SearchResponse, SearchResult
from search.bm25_index import get_bm25_index
from search.embedder import get_embedder
from search.vector_store import MongoVectorStore


# ─── Query Expansion ─────────────────────────────────────────────────────────

_EXPANSION_MAP: Dict[str, List[str]] = {
    "deadline": ["due date", "submission deadline", "closing date", "bid opening"],
    "due date": ["deadline", "submission deadline", "closing date", "bid opening"],
    "bond": ["bid bond", "performance bond", "surety bond", "security deposit"],
    "warranty": ["guarantee", "service agreement", "extended warranty", "support"],
    "delivery": ["delivery date", "delivery window", "shipment", "lead time"],
    "affidavit": ["affidavit", "declaration", "certification", "sworn statement"],
    "contact": ["contact info", "procurement officer", "buyer", "email", "phone"],
    "term": ["contract term", "contract duration", "period of performance", "renewal"],
    "price": ["cost", "pricing", "unit price", "total cost", "amount"],
    "spec": ["specification", "technical spec", "requirement", "configuration"],
    "model": ["model number", "part number", "sku", "item number"],
    "payment": ["payment terms", "net 30", "invoice", "billing"],
    "addendum": ["amendment", "modification", "change", "update"],
    "pre-bid": ["pre bid meeting", "mandatory meeting", "pre-proposal", "conference"],
    "install": ["installation", "deployment", "imaging", "setup", "configuration"],
    "manufacturer": ["oem", "brand", "vendor registration", "authorized dealer"],
}


def expand_query(query: str) -> str:
    """
    Expand query with synonyms to improve recall.
    Returns original query + expanded terms.
    """
    lower = query.lower()
    expansions = []
    for key, alternatives in _EXPANSION_MAP.items():
        if key in lower:
            expansions.extend(alt for alt in alternatives if alt not in lower)

    if expansions:
        expanded = query + " " + " ".join(set(expansions))
        logger.debug(f"Query expanded: '{query}' → '{expanded}'")
        return expanded
    return query


# ─── Reciprocal Rank Fusion ──────────────────────────────────────────────────

def _reciprocal_rank_fusion(
    vector_results: List[SearchResult],
    bm25_results: List[SearchResult],
    alpha: float = 0.6,
    k: int = 60,
) -> List[SearchResult]:
    """
    Merge vector and BM25 results using Reciprocal Rank Fusion.
    alpha: weight towards vector search (0=BM25 only, 1=vector only)
    k: RRF constant (default 60 as per original paper)
    """
    scores: Dict[str, float] = {}
    chunk_map: Dict[str, SearchResult] = {}

    for rank, result in enumerate(vector_results):
        scores[result.chunk_id] = scores.get(result.chunk_id, 0) + alpha / (k + rank + 1)
        chunk_map[result.chunk_id] = result

    for rank, result in enumerate(bm25_results):
        scores[result.chunk_id] = scores.get(result.chunk_id, 0) + (1 - alpha) / (k + rank + 1)
        if result.chunk_id not in chunk_map:
            chunk_map[result.chunk_id] = result

    sorted_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    merged = []
    for cid in sorted_ids:
        result = chunk_map[cid]
        result.score = scores[cid]
        merged.append(result)

    return merged


# ─── Cross-encoder Reranker ──────────────────────────────────────────────────

_reranker = None


def _get_reranker():
    """Lazy-load cross-encoder reranker."""
    global _reranker
    if _reranker is None:
        try:
            from sentence_transformers import CrossEncoder
            _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            logger.info("Cross-encoder reranker loaded.")
        except Exception as e:
            logger.warning(f"Reranker not available: {e}. Skipping reranking.")
    return _reranker


def rerank(
    query: str, results: List[SearchResult], top_k: int
) -> List[SearchResult]:
    """Re-rank candidates using cross-encoder."""
    reranker = _get_reranker()
    if reranker is None or len(results) <= 1:
        return results[:top_k]

    try:
        pairs = [(query, r.text) for r in results]
        new_scores = reranker.predict(pairs)
        for r, s in zip(results, new_scores):
            r.score = float(s)
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]
    except Exception as e:
        logger.warning(f"Reranking failed: {e}")
        return results[:top_k]


# ─── Main Retrieval Engine ───────────────────────────────────────────────────

class HybridSearchEngine:
    """
    Main search engine combining:
    1. Query expansion
    2. Vector search (MongoDB Atlas)
    3. BM25 keyword search
    4. RRF fusion
    5. Cross-encoder reranking
    """

    def __init__(self, vector_store: MongoVectorStore):
        self.vector_store = vector_store
        self.bm25 = get_bm25_index()
        self.embedder = get_embedder()

    def search(self, query_req: SearchQuery) -> SearchResponse:
        """Execute hybrid search pipeline."""
        original_query = query_req.query
        rewritten_query = original_query

        # 1. Query expansion
        if query_req.expand_query:
            rewritten_query = expand_query(original_query)

        # 2. Vector search
        try:
            query_emb = self.embedder.embed_text(rewritten_query)
            vector_results = self.vector_store.vector_search(
                query_embedding=query_emb,
                top_k=query_req.top_k * 2,  # over-fetch for fusion
                bid_id=query_req.bid_id,
                doc_types=query_req.doc_types,
            )
        except Exception as e:
            logger.error(f"Vector search error: {e}")
            vector_results = []

        # 3. BM25 keyword search
        try:
            bm25_results = self.bm25.search_all(
                query=rewritten_query,
                top_k=query_req.top_k * 2,
                bid_id=query_req.bid_id,
                doc_types=query_req.doc_types,
            )
        except Exception as e:
            logger.error(f"BM25 search error: {e}")
            bm25_results = []

        # 4. Reciprocal Rank Fusion
        if vector_results and bm25_results:
            fused = _reciprocal_rank_fusion(
                vector_results, bm25_results, alpha=settings.hybrid_alpha
            )
        elif vector_results:
            fused = vector_results
        else:
            fused = bm25_results

        # 5. Reranking
        if query_req.use_rerank and fused:
            final = rerank(original_query, fused, query_req.top_k)
        else:
            final = fused[: query_req.top_k]

        logger.info(
            f"Search '{original_query[:50]}': "
            f"{len(vector_results)} vector, {len(bm25_results)} BM25 → "
            f"{len(final)} final"
        )

        return SearchResponse(
            query=original_query,
            rewritten_query=rewritten_query if rewritten_query != original_query else None,
            results=final,
            total_found=len(fused),
        )
