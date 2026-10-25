"""
Indexing Pipeline – orchestrates parsing → chunking → embedding → storing.
Entry point: index_bid_folder(folder_path, bid_id)
Supports incremental indexing (skip already-indexed bids or force re-index).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from loguru import logger
from tqdm import tqdm

from ingestion.chunker import chunk_document
from ingestion.html_parser import parse_html
from ingestion.pdf_parser import parse_pdf
from models import DocumentChunk, ParsedDocument
from search.bm25_index import get_bm25_index
from search.embedder import get_embedder
from search.vector_store import MongoVectorStore


def _classify_file(file_path: Path) -> Optional[str]:
    """Return 'pdf' | 'html' | None based on extension/content."""
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return "pdf"
    elif suffix in (".html", ".htm"):
        return "html"
    return None


def index_bid_folder(
    folder_path: str | Path,
    bid_id: str,
    vector_store: MongoVectorStore,
    force_reindex: bool = False,
) -> dict:
    """
    Parse all files in a bid folder, chunk, embed, and store.

    Args:
        folder_path: Path to the bid folder containing HTML/PDF files.
        bid_id:      Unique identifier for this bid (e.g. "Bid1").
        vector_store: MongoVectorStore instance.
        force_reindex: If True, delete existing chunks and re-index.

    Returns:
        Summary dict with counts and any errors.
    """
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Bid folder not found: {folder}")

    # Check for existing index
    if not force_reindex and vector_store.bid_exists(bid_id):
        logger.info(f"Bid [{bid_id}] already indexed. Use force_reindex=True to re-index.")
        return {"bid_id": bid_id, "status": "already_indexed", "chunks": 0, "errors": []}

    if force_reindex:
        deleted = vector_store.delete_bid(bid_id)
        logger.info(f"Deleted {deleted} existing chunks for [{bid_id}]")

    summary = {
        "bid_id": bid_id,
        "files_found": 0,
        "files_parsed": 0,
        "chunks_created": 0,
        "chunks_indexed": 0,
        "errors": [],
    }

    # ── Step 1: Parse all files ──────────────────────────────────────────────
    parsed_docs: List[ParsedDocument] = []
    files = list(folder.iterdir())
    summary["files_found"] = len(files)

    for file_path in files:
        ftype = _classify_file(file_path)
        if ftype is None:
            logger.debug(f"Skipping unsupported file: {file_path.name}")
            continue

        try:
            if ftype == "pdf":
                doc = parse_pdf(file_path, bid_id)
            else:
                doc = parse_html(file_path, bid_id)

            parsed_docs.append(doc)
            summary["files_parsed"] += 1
            if doc.parse_errors:
                summary["errors"].extend(
                    [f"{file_path.name}: {e}" for e in doc.parse_errors]
                )
        except Exception as e:
            error_msg = f"Failed to parse {file_path.name}: {e}"
            logger.error(error_msg)
            summary["errors"].append(error_msg)

    if not parsed_docs:
        logger.warning(f"No documents parsed for [{bid_id}]")
        return summary

    # ── Step 2: Chunk all documents ──────────────────────────────────────────
    all_chunks: List[DocumentChunk] = []
    for doc in parsed_docs:
        chunks = chunk_document(doc)
        all_chunks.extend(chunks)

    summary["chunks_created"] = len(all_chunks)
    logger.info(f"[{bid_id}] Created {len(all_chunks)} chunks from {len(parsed_docs)} docs")

    if not all_chunks:
        return summary

    # ── Step 3: Generate embeddings ──────────────────────────────────────────
    embedder = get_embedder()
    texts = [c.text for c in all_chunks]

    logger.info(f"[{bid_id}] Generating embeddings for {len(texts)} chunks...")
    embeddings = embedder.embed_batch(texts)

    for chunk, emb in zip(all_chunks, embeddings):
        chunk.embedding = emb

    # ── Step 4: Store in MongoDB ─────────────────────────────────────────────
    n_indexed = vector_store.upsert_chunks(all_chunks)
    summary["chunks_indexed"] = n_indexed

    # ── Step 5: Build / update BM25 index ────────────────────────────────────
    bm25_manager = get_bm25_index()
    bm25_idx = bm25_manager.get_or_create(bid_id)
    bm25_idx.build(all_chunks)

    logger.info(
        f"[{bid_id}] Indexing complete: {n_indexed} chunks stored, "
        f"BM25 built."
    )
    return summary
