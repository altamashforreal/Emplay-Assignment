"""
BM25 Index – keyword search over document chunks.
Persisted to disk per bid_id for incremental updates.
Uses rank_bm25 library.
"""
from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger
from rank_bm25 import BM25Okapi

from config import settings
from models import DocumentChunk, DocType, SearchResult


def _tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenizer."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()


class BM25Index:
    """
    Per-bid BM25 index that can be incrementally rebuilt.
    Stored as pickle files in settings.bm25_index_dir.
    """

    def __init__(self, bid_id: str):
        self.bid_id = bid_id
        self._index_dir = Path(settings.bm25_index_dir)
        self._index_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self._index_dir / f"{bid_id}_bm25.pkl"
        self._meta_path = self._index_dir / f"{bid_id}_meta.json"

        self._bm25: Optional[BM25Okapi] = None
        self._chunk_metas: List[Dict] = []  # lightweight metadata (no embeddings)

        if self._index_path.exists():
            self._load()

    def build(self, chunks: List[DocumentChunk]) -> None:
        """Build (or rebuild) BM25 index from chunks."""
        corpus = [_tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(corpus)
        self._chunk_metas = [
            {
                "chunk_id": c.chunk_id,
                "bid_id": c.bid_id,
                "file_name": c.file_name,
                "doc_type": c.doc_type.value,
                "addendum_number": c.addendum_number,
                "page_number": c.page_number,
                "section_heading": c.section_heading,
                "text": c.text,
                "is_table": c.is_table,
            }
            for c in chunks
        ]
        self._save()
        logger.info(f"BM25 index built for [{self.bid_id}]: {len(chunks)} chunks")

    def search(
        self,
        query: str,
        top_k: int = 10,
        doc_types: Optional[List[DocType]] = None,
    ) -> List[SearchResult]:
        """BM25 keyword search with optional doc_type filtering."""
        if self._bm25 is None or not self._chunk_metas:
            logger.warning(f"BM25 index not built for {self.bid_id}")
            return []

        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)

        # Sort by score descending
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

        results: List[SearchResult] = []
        for idx, score in ranked:
            if score <= 0:
                break
            meta = self._chunk_metas[idx]
            dt = DocType(meta["doc_type"])
            if doc_types and dt not in doc_types:
                continue
            results.append(
                SearchResult(
                    chunk_id=meta["chunk_id"],
                    bid_id=meta["bid_id"],
                    file_name=meta["file_name"],
                    doc_type=dt,
                    page_number=meta["page_number"],
                    section_heading=meta.get("section_heading"),
                    text=meta["text"],
                    score=float(score),
                    addendum_number=meta.get("addendum_number"),
                    is_table=meta.get("is_table", False),
                )
            )
            if len(results) >= top_k:
                break

        return results

    def _save(self) -> None:
        with open(self._index_path, "wb") as f:
            pickle.dump(self._bm25, f)
        with open(self._meta_path, "w") as f:
            json.dump(self._chunk_metas, f, default=str)

    def _load(self) -> None:
        try:
            with open(self._index_path, "rb") as f:
                self._bm25 = pickle.load(f)
            with open(self._meta_path, "r") as f:
                self._chunk_metas = json.load(f)
            logger.info(f"BM25 index loaded for [{self.bid_id}]")
        except Exception as e:
            logger.warning(f"Failed to load BM25 index for {self.bid_id}: {e}")
            self._bm25 = None
            self._chunk_metas = []

    @property
    def is_built(self) -> bool:
        return self._bm25 is not None and len(self._chunk_metas) > 0


class GlobalBM25Index:
    """Manages per-bid BM25 indexes."""

    def __init__(self):
        self._indexes: Dict[str, BM25Index] = {}

    def get_or_create(self, bid_id: str) -> BM25Index:
        if bid_id not in self._indexes:
            self._indexes[bid_id] = BM25Index(bid_id)
        return self._indexes[bid_id]

    def search_all(
        self,
        query: str,
        top_k: int = 10,
        bid_id: Optional[str] = None,
        doc_types: Optional[List[DocType]] = None,
    ) -> List[SearchResult]:
        """Search across one or all bids."""
        all_results: List[SearchResult] = []
        targets = [bid_id] if bid_id else list(self._indexes.keys())
        for bid in targets:
            idx = self.get_or_create(bid)
            if idx.is_built:
                all_results.extend(idx.search(query, top_k, doc_types))
        # Re-sort combined results
        all_results.sort(key=lambda r: r.score, reverse=True)
        return all_results[:top_k]


_global_bm25 = GlobalBM25Index()


def get_bm25_index() -> GlobalBM25Index:
    return _global_bm25
