"""
Chunker – section/heading-aware and table-aware chunking with overlap.

Strategy:
  1. Split document by section headings first (semantic boundaries).
  2. Within each section, split by sentences respecting chunk_size.
  3. Apply overlap between consecutive chunks.
  4. Tables are always kept as single chunks (never split).

Chunk size: 512 tokens (~2000 characters) balances context richness with
            precision. Overlap of 64 tokens prevents boundary information loss.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, Generator, List, Optional

from loguru import logger

from config import settings
from models import DocType, DocumentChunk, ParsedDocument, ParsedPage


# Heading patterns for section-aware splitting
_HEADING_PATTERNS = [
    re.compile(r"^\s{0,4}(\d+[\.\)]\s+[A-Z].{0,100})$", re.MULTILINE),  # 1. Section
    re.compile(r"^\s{0,4}([A-Z][A-Z\s]{3,60})$", re.MULTILINE),          # ALL CAPS
    re.compile(r"^\s{0,4}(SECTION\s+\w.{0,80})$", re.MULTILINE | re.I),  # SECTION X
]


def _make_chunk_id(bid_id: str, file_name: str, page: int, idx: int) -> str:
    raw = f"{bid_id}|{file_name}|{page}|{idx}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _split_by_headings(text: str) -> List[tuple[Optional[str], str]]:
    """Split text into (heading, content) pairs using heading heuristics."""
    # Build a combined split pattern
    split_points = []
    for pat in _HEADING_PATTERNS:
        for m in pat.finditer(text):
            split_points.append((m.start(), m.group(0).strip()))

    if not split_points:
        return [(None, text)]

    split_points.sort(key=lambda x: x[0])

    sections = []
    prev_end = 0
    prev_heading = None

    for pos, heading in split_points:
        if pos > prev_end:
            sections.append((prev_heading, text[prev_end:pos]))
        prev_heading = heading
        prev_end = pos + len(heading)

    # Last section
    sections.append((prev_heading, text[prev_end:]))
    return [(h, t.strip()) for h, t in sections if t.strip()]


def _chunk_text(
    text: str,
    chunk_size: int,
    overlap: int,
) -> Generator[str, None, None]:
    """Yield overlapping text chunks of approximately chunk_size characters."""
    if len(text) <= chunk_size:
        yield text
        return

    words = text.split()
    buf: List[str] = []
    buf_len = 0
    overlap_words: List[str] = []

    for word in words:
        buf.append(word)
        buf_len += len(word) + 1

        if buf_len >= chunk_size:
            chunk = " ".join(buf)
            yield chunk
            # Keep last `overlap` chars worth of words
            overlap_buf = []
            overlap_len = 0
            for w in reversed(buf):
                if overlap_len + len(w) + 1 > overlap:
                    break
                overlap_buf.insert(0, w)
                overlap_len += len(w) + 1
            buf = overlap_buf
            buf_len = overlap_len

    if buf:
        yield " ".join(buf)


def _table_to_text(table: Dict[str, Any]) -> str:
    """Convert table dict to readable text for embedding."""
    headers = table.get("headers", [])
    rows = table.get("rows", [])
    lines = []
    if headers:
        lines.append(" | ".join(headers))
        lines.append("-" * 40)
    for row in rows:
        if isinstance(row, dict):
            lines.append(" | ".join(str(v) for v in row.values()))
        elif isinstance(row, list):
            lines.append(" | ".join(str(c) for c in row))
    return "\n".join(lines)


def chunk_document(doc: ParsedDocument) -> List[DocumentChunk]:
    """Chunk a ParsedDocument into DocumentChunk objects ready for indexing."""
    chunks: List[DocumentChunk] = []
    chunk_size = settings.chunk_size
    overlap = settings.chunk_overlap
    global_idx = 0

    for page in doc.pages:
        # 1. Table chunks (one chunk per table)
        for tbl in page.tables:
            tbl_text = _table_to_text(tbl)
            if len(tbl_text.strip()) < 10:
                continue
            chunk_id = _make_chunk_id(doc.bid_id, doc.file_name, page.page_number, global_idx)
            chunks.append(DocumentChunk(
                chunk_id=chunk_id,
                bid_id=doc.bid_id,
                file_name=doc.file_name,
                doc_type=doc.doc_type,
                addendum_number=doc.addendum_number,
                page_number=page.page_number,
                section_heading=page.section_heading,
                text=f"[TABLE]\n{tbl_text}",
                document_date=doc.document_date,
                is_table=True,
            ))
            global_idx += 1

        # 2. Text chunks – split by headings, then by size
        sections = _split_by_headings(page.text)
        for heading, section_text in sections:
            effective_heading = heading or page.section_heading
            for text_chunk in _chunk_text(section_text, chunk_size, overlap):
                text_chunk = text_chunk.strip()
                if len(text_chunk) < 30:
                    continue
                chunk_id = _make_chunk_id(
                    doc.bid_id, doc.file_name, page.page_number, global_idx
                )
                chunks.append(DocumentChunk(
                    chunk_id=chunk_id,
                    bid_id=doc.bid_id,
                    file_name=doc.file_name,
                    doc_type=doc.doc_type,
                    addendum_number=doc.addendum_number,
                    page_number=page.page_number,
                    section_heading=effective_heading,
                    text=text_chunk,
                    document_date=doc.document_date,
                    is_table=False,
                ))
                global_idx += 1

    logger.info(
        f"Chunked [{doc.bid_id}] {doc.file_name}: {len(chunks)} chunks "
        f"({len(doc.pages)} pages)"
    )
    return chunks
