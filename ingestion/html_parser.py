"""
HTML Parser – extracts bid page content using BeautifulSoup + trafilatura.
Preserves tables and structured sections.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from models import DocType, ParsedDocument, ParsedPage


def _clean_html_text(text: str) -> str:
    """Clean extracted HTML text."""
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_html(
    file_path: str | Path,
    bid_id: str,
) -> ParsedDocument:
    """Parse an HTML bid page and return a ParsedDocument."""
    try:
        from bs4 import BeautifulSoup
    except ImportError as e:
        raise ImportError(f"Install beautifulsoup4: {e}") from e

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"HTML file not found: {path}")
    fname = path.name


    doc = ParsedDocument(
        bid_id=bid_id,
        file_name=fname,
        file_path=str(path.resolve()),
        doc_type=DocType.BID_PAGE,
        page_count=1,
    )

    parse_errors: List[str] = []

    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            html_content = f.read()

        soup = BeautifulSoup(html_content, "lxml")

        # Remove noise elements
        for tag in soup(["script", "style", "nav", "footer", "header", "iframe"]):
            tag.decompose()

        # Extract tables
        tables: List[Dict[str, Any]] = []
        for table in soup.find_all("table"):
            rows_data = []
            rows = table.find_all("tr")
            if not rows:
                continue
            headers: List[str] = []
            for i, row in enumerate(rows):
                cells = [td.get_text(strip=True) for td in row.find_all(["th", "td"])]
                if i == 0:
                    headers = cells
                else:
                    if headers:
                        rows_data.append(dict(zip(headers, cells)))
                    else:
                        rows_data.append({f"col_{j}": c for j, c in enumerate(cells)})
            if rows_data:
                tables.append({"headers": headers, "rows": rows_data})

        # Try trafilatura first for clean text, fall back to BS4
        full_text = ""
        try:
            import trafilatura
            full_text = trafilatura.extract(html_content, include_tables=True) or ""
        except Exception:
            pass

        if not full_text:
            # Fallback to BeautifulSoup text extraction
            full_text = soup.get_text(separator="\n")

        full_text = _clean_html_text(full_text)

        # Append table text for search
        for tbl in tables:
            tbl_text = "\n".join(
                " | ".join(f"{k}: {v}" for k, v in row.items())
                for row in tbl.get("rows", [])
            )
            full_text += f"\n\n[TABLE]\n{tbl_text}\n[/TABLE]"

        pages = [
            ParsedPage(
                page_number=1,
                text=full_text,
                tables=tables,
                section_heading="Bid Overview Page",
            )
        ]

        doc.pages = pages
        doc.tables = tables
        doc.parse_errors = parse_errors

    except Exception as e:
        parse_errors.append(f"Fatal HTML parse error: {e}")
        logger.error(f"Failed to parse HTML {path}: {e}")
        doc.parse_errors = parse_errors

    logger.info(f"Parsed HTML [{bid_id}] {fname}: {len(tables)} tables")
    return doc
