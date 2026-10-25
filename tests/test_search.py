"""
Tests for search components – BM25 index and hybrid retrieval.
"""
from __future__ import annotations

import pytest

from models import DocType, DocumentChunk, SearchResult
from search.bm25_index import BM25Index, _tokenize
from search.retrieval import expand_query, _reciprocal_rank_fusion


# ─── Unit: tokenizer ─────────────────────────────────────────────────────────

class TestTokenizer:
    def test_lowercases(self):
        tokens = _tokenize("Hello World")
        assert "hello" in tokens
        assert "world" in tokens

    def test_removes_punctuation(self):
        tokens = _tokenize("bid, number: JA-207652.")
        assert "207652" in tokens or "ja" in tokens

    def test_empty_string(self):
        tokens = _tokenize("")
        assert tokens == []


# ─── Unit: expand_query ──────────────────────────────────────────────────────

class TestExpandQuery:
    def test_expands_deadline(self):
        expanded = expand_query("What is the deadline?")
        assert "due date" in expanded.lower() or "closing date" in expanded.lower()

    def test_expands_bond(self):
        expanded = expand_query("Is a bond required?")
        assert "bid bond" in expanded.lower() or "surety" in expanded.lower()

    def test_no_expansion_for_unknown(self):
        query = "arbitrary unknown query xyz123"
        expanded = expand_query(query)
        assert expanded == query  # unchanged

    def test_does_not_duplicate(self):
        query = "submission deadline due date"
        expanded = expand_query(query)
        # Should not repeat the same word excessively
        assert expanded.count("deadline") <= 2


# ─── Unit: BM25Index ─────────────────────────────────────────────────────────

def _make_chunks(bid_id="TestBid"):
    texts = [
        "The submission deadline is January 15 2025 at 2PM EST",
        "A bid bond of 5 percent is required",
        "The contract term is two years with one renewal option",
        "Laptops must have 16GB RAM and 512GB SSD storage",
        "Contact the procurement officer at procurement@school.edu",
    ]
    return [
        DocumentChunk(
            chunk_id=f"chunk_{i}",
            bid_id=bid_id,
            file_name="test.pdf",
            doc_type=DocType.RFP,
            page_number=i + 1,
            text=text,
        )
        for i, text in enumerate(texts)
    ]


class TestBM25Index:
    def test_build_and_search(self, tmp_path, monkeypatch):
        monkeypatch.setattr("search.bm25_index.settings.bm25_index_dir", tmp_path)
        idx = BM25Index("TestBid")
        chunks = _make_chunks()
        idx.build(chunks)

        results = idx.search("submission deadline", top_k=3)
        assert len(results) > 0
        assert results[0].chunk_id == "chunk_0"  # most relevant

    def test_keyword_precision(self, tmp_path, monkeypatch):
        monkeypatch.setattr("search.bm25_index.settings.bm25_index_dir", tmp_path)
        idx = BM25Index("TestBid2")
        idx.build(_make_chunks("TestBid2"))

        results = idx.search("bid bond 5 percent", top_k=5)
        assert any("bond" in r.text.lower() for r in results)

    def test_empty_query_no_crash(self, tmp_path, monkeypatch):
        monkeypatch.setattr("search.bm25_index.settings.bm25_index_dir", tmp_path)
        idx = BM25Index("TestBid3")
        idx.build(_make_chunks("TestBid3"))
        results = idx.search("", top_k=5)
        assert isinstance(results, list)

    def test_is_built_flag(self, tmp_path, monkeypatch):
        monkeypatch.setattr("search.bm25_index.settings.bm25_index_dir", tmp_path)
        idx = BM25Index("NewBid")
        assert not idx.is_built
        idx.build(_make_chunks("NewBid"))
        assert idx.is_built


# ─── Unit: RRF Fusion ────────────────────────────────────────────────────────

def _make_result(chunk_id, score, bid_id="Bid1"):
    return SearchResult(
        chunk_id=chunk_id,
        bid_id=bid_id,
        file_name="test.pdf",
        doc_type=DocType.RFP,
        page_number=1,
        text=f"Content for {chunk_id}",
        score=score,
    )


class TestRRFFusion:
    def test_merges_results(self):
        vector = [_make_result("a", 0.9), _make_result("b", 0.8), _make_result("c", 0.7)]
        bm25 = [_make_result("b", 3.5), _make_result("d", 2.0), _make_result("a", 1.5)]
        merged = _reciprocal_rank_fusion(vector, bm25, alpha=0.5)

        ids = [r.chunk_id for r in merged]
        assert "a" in ids
        assert "b" in ids
        assert "c" in ids
        assert "d" in ids

    def test_alpha_1_favors_vector(self):
        vector = [_make_result("a", 0.95), _make_result("b", 0.5)]
        bm25 = [_make_result("c", 5.0), _make_result("a", 1.0)]
        merged = _reciprocal_rank_fusion(vector, bm25, alpha=1.0)
        # With alpha=1, only vector scores count, so 'a' should rank first
        assert merged[0].chunk_id == "a"

    def test_returns_all_unique(self):
        vector = [_make_result("x", 1.0), _make_result("y", 0.5)]
        bm25 = [_make_result("y", 2.0), _make_result("z", 1.5)]
        merged = _reciprocal_rank_fusion(vector, bm25, alpha=0.5)
        ids = [r.chunk_id for r in merged]
        assert len(ids) == len(set(ids))
