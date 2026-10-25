"""
Tests for ingestion – PDF/HTML parsing and chunking.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from ingestion.chunker import chunk_document, _split_by_headings, _chunk_text
from ingestion.html_parser import parse_html
from ingestion.pdf_parser import _clean_text, _detect_doc_type
from models import DocType, ParsedDocument, ParsedPage


# ─── Unit: _detect_doc_type ───────────────────────────────────────────────────

class TestDetectDocType:
    def test_addendum_with_number(self):
        dt, num = _detect_doc_type("Addendum 2 Final.pdf")
        assert dt == DocType.ADDENDUM
        assert num == 2

    def test_addendum_no_number(self):
        dt, num = _detect_doc_type("addendum.pdf")
        assert dt == DocType.ADDENDUM
        assert num == 1

    def test_rfp(self):
        dt, num = _detect_doc_type("JA-207652 FINAL RFP.pdf")
        assert dt == DocType.RFP
        assert num is None

    def test_specs(self):
        dt, num = _detect_doc_type("Dell Laptop Specs.pdf")
        assert dt == DocType.SPECS
        assert num is None

    def test_affidavit(self):
        dt, num = _detect_doc_type("Mercury Affidavit.pdf")
        assert dt == DocType.AFFIDAVIT
        assert num is None

    def test_contract(self):
        dt, num = _detect_doc_type("Contract Affidavit.pdf")
        assert dt in (DocType.CONTRACT, DocType.AFFIDAVIT)

    def test_unknown(self):
        dt, num = _detect_doc_type("README.txt")
        assert dt == DocType.OTHER


# ─── Unit: _clean_text ───────────────────────────────────────────────────────

class TestCleanText:
    def test_removes_repeated_lines(self):
        # Line "HEADER" repeated 3+ times should be removed
        text = "HEADER\nContent 1\nHEADER\nContent 2\nHEADER\nContent 3\nHEADER"
        cleaned = _clean_text(text)
        # repeated line should be gone
        assert cleaned.count("HEADER") == 0
        assert "Content 1" in cleaned

    def test_fixes_hyphenated_lines(self):
        text = "some-\nword continues here"
        cleaned = _clean_text(text)
        assert "some-word" in cleaned

    def test_normalizes_whitespace(self):
        text = "hello   world\n\n\n\nfoo"
        cleaned = _clean_text(text)
        assert "  " not in cleaned
        assert "\n\n\n" not in cleaned


# ─── Unit: _split_by_headings ─────────────────────────────────────────────────

class TestSplitByHeadings:
    def test_splits_on_numbered_heading(self):
        text = "Some intro\n1. First Section\nContent of first section\n2. Second Section\nContent of second"
        sections = _split_by_headings(text)
        assert len(sections) >= 2

    def test_handles_no_headings(self):
        text = "Just a regular paragraph with no headings at all."
        sections = _split_by_headings(text)
        assert len(sections) == 1
        assert sections[0][0] is None


# ─── Unit: _chunk_text ───────────────────────────────────────────────────────

class TestChunkText:
    def test_short_text_is_single_chunk(self):
        text = "Short text."
        chunks = list(_chunk_text(text, chunk_size=1000, overlap=100))
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_text_is_split(self):
        text = " ".join(["word"] * 300)  # 300 words
        chunks = list(_chunk_text(text, chunk_size=200, overlap=20))
        assert len(chunks) > 1

    def test_overlap_exists(self):
        words = [f"word{i}" for i in range(100)]
        text = " ".join(words)
        chunks = list(_chunk_text(text, chunk_size=100, overlap=30))
        if len(chunks) > 1:
            # Last words of chunk N should appear in start of chunk N+1
            last_words = chunks[0].split()[-3:]
            next_start = chunks[1].split()[:5]
            assert any(w in next_start for w in last_words)


# ─── Unit: chunk_document ────────────────────────────────────────────────────

class TestChunkDocument:
    def _make_doc(self, pages_text=None, with_table=False):
        pages_text = pages_text or ["This is a test page with some content about bid requirements."]
        pages = []
        for i, text in enumerate(pages_text):
            tables = []
            if with_table:
                tables = [{"headers": ["Model", "Price"], "rows": [{"Model": "XPS13", "Price": "$999"}]}]
            pages.append(ParsedPage(page_number=i + 1, text=text, tables=tables))

        return ParsedDocument(
            bid_id="TestBid",
            file_name="test.pdf",
            file_path="/tmp/test.pdf",
            doc_type=DocType.RFP,
            pages=pages,
        )

    def test_basic_chunking(self):
        doc = self._make_doc()
        chunks = chunk_document(doc)
        assert len(chunks) > 0
        assert all(c.bid_id == "TestBid" for c in chunks)
        assert all(c.file_name == "test.pdf" for c in chunks)

    def test_table_chunks_marked(self):
        doc = self._make_doc(with_table=True)
        chunks = chunk_document(doc)
        table_chunks = [c for c in chunks if c.is_table]
        assert len(table_chunks) > 0

    def test_chunk_ids_unique(self):
        doc = self._make_doc(pages_text=["Page one content." * 50, "Page two content." * 50])
        chunks = chunk_document(doc)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs must be unique"

    def test_metadata_preserved(self):
        doc = self._make_doc()
        doc.addendum_number = 2
        doc.doc_type = DocType.ADDENDUM
        chunks = chunk_document(doc)
        for c in chunks:
            assert c.addendum_number == 2
            assert c.doc_type == DocType.ADDENDUM


# ─── Integration: parse_html ─────────────────────────────────────────────────

class TestParseHTML:
    def test_parses_simple_html(self, tmp_path):
        html_file = tmp_path / "bid.html"
        html_file.write_text("""
        <html><body>
        <h1>Bid Title</h1>
        <p>Submission deadline: January 15, 2025</p>
        <table><tr><th>Item</th><th>Qty</th></tr>
        <tr><td>Laptop</td><td>100</td></tr></table>
        </body></html>
        """)
        doc = parse_html(str(html_file), "TestBid")
        assert doc.doc_type == DocType.BID_PAGE
        assert len(doc.pages) == 1
        assert "Bid Title" in doc.pages[0].text or "deadline" in doc.pages[0].text

    def test_graceful_on_missing_file(self):
        with pytest.raises(FileNotFoundError):
            parse_html("/nonexistent/file.html", "TestBid")
