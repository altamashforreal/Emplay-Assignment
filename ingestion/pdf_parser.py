"""
PDF Parser – uses PyMuPDF + pdfplumber for text and table extraction.
Handles scanned pages gracefully (logs them as errors).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from models import DocType, ParsedDocument, ParsedPage


def _detect_doc_type(file_name: str) -> tuple[DocType, Optional[int]]:
    """Infer doc type and addendum number from filename."""
    lower = file_name.lower()
    if "addendum" in lower:
        # Extract addendum number
        match = re.search(r"addendum\s*(\d+)", lower)
        num = int(match.group(1)) if match else 1
        return DocType.ADDENDUM, num
    elif "spec" in lower:
        return DocType.SPECS, None
    elif "affidavit" in lower:
        return DocType.AFFIDAVIT, None
    elif "contract" in lower:
        return DocType.CONTRACT, None
    elif any(k in lower for k in ["rfp", "porfp", "bid", "solicitation", "final"]):
        return DocType.RFP, None
    return DocType.OTHER, None


def _clean_text(text: str) -> str:
    """Remove repeated headers/footers, fix broken lines, normalize whitespace."""
    # Fix soft-hyphens and broken words
    text = text.replace("\u00ad", "")
    # Fix broken hyphenated lines (word-\n → word)
    text = re.sub(r"-\n(\w)", r"-\1", text)
    # Remove repeated running headers (lines that appear 3+ times)
    lines = text.split("\n")
    from collections import Counter
    line_counts = Counter(l.strip() for l in lines if len(l.strip()) > 5)
    repeated = {l for l, c in line_counts.items() if c >= 3}
    lines = [l for l in lines if l.strip() not in repeated]
    text = "\n".join(lines)
    # Normalize whitespace
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_pdf(
    file_path: str | Path,
    bid_id: str,
    doc_type: Optional[DocType] = None,
    addendum_number: Optional[int] = None,
) -> ParsedDocument:
    """Parse a PDF file and return a ParsedDocument."""
    try:
        try:
            import pymupdf as fitz  # PyMuPDF >= 1.24 new name
        except ImportError:
            import fitz  # older PyMuPDF (< 2.0) uses fitz
        import pdfplumber
    except ImportError as e:
        raise ImportError(f"Install PyMuPDF and pdfplumber: {e}") from e


    path = Path(file_path)
    fname = path.name

    if doc_type is None:
        doc_type, addendum_number = _detect_doc_type(fname)

    doc = ParsedDocument(
        bid_id=bid_id,
        file_name=fname,
        file_path=str(path.resolve()),
        doc_type=doc_type,
        addendum_number=addendum_number,
    )

    parse_errors: List[str] = []
    pages: List[ParsedPage] = []

    try:
        # Use PyMuPDF for text extraction
        mupdf_doc = fitz.open(str(path))
        doc.page_count = len(mupdf_doc)

        # Use pdfplumber for table extraction
        with pdfplumber.open(str(path)) as plumber_doc:
            for page_num in range(len(mupdf_doc)):
                try:
                    page = mupdf_doc[page_num]
                    raw_text = page.get_text("text")

                    # Detect scanned (no extractable text)
                    if len(raw_text.strip()) < 20:
                        parse_errors.append(
                            f"Page {page_num + 1}: Minimal text – may be scanned."
                        )
                        raw_text = f"[SCANNED PAGE {page_num + 1}]"

                    clean = _clean_text(raw_text)

                    # Extract tables via pdfplumber
                    page_tables: List[Dict[str, Any]] = []
                    if page_num < len(plumber_doc.pages):
                        plumber_page = plumber_doc.pages[page_num]
                        for tbl in plumber_page.extract_tables() or []:
                            if tbl:
                                # Convert to list of dicts using first row as headers
                                headers = [str(h or "").strip() for h in tbl[0]]
                                rows = []
                                for row in tbl[1:]:
                                    rows.append(
                                        {headers[i]: str(c or "").strip()
                                         for i, c in enumerate(row)}
                                    )
                                page_tables.append(
                                    {"headers": headers, "rows": rows}
                                )

                    # Detect section heading (first bold/large text line heuristic)
                    heading = _extract_heading(clean)

                    pages.append(ParsedPage(
                        page_number=page_num + 1,
                        text=clean,
                        tables=page_tables,
                        section_heading=heading,
                    ))

                except Exception as e:
                    error_msg = f"Page {page_num + 1}: {e}"
                    parse_errors.append(error_msg)
                    logger.warning(f"[{fname}] {error_msg}")

        mupdf_doc.close()

    except Exception as e:
        parse_errors.append(f"Fatal parse error: {e}")
        logger.error(f"Failed to parse {path}: {e}")

    doc.pages = pages
    doc.parse_errors = parse_errors
    logger.info(
        f"Parsed PDF [{bid_id}] {fname}: {len(pages)} pages, "
        f"{len(parse_errors)} errors"
    )
    return doc


def _extract_heading(text: str) -> Optional[str]:
    """Heuristic: first non-empty line that looks like a section heading."""
    for line in text.split("\n"):
        line = line.strip()
        if 5 < len(line) < 120 and not line.endswith(","):
            # Headings are often all-caps, title-case, or short lines
            if line.isupper() or line.istitle() or re.match(r"^\d+[\.\)]\s+\w", line):
                return line
    return None
