"""
Extraction Agent – fills structured fields using only retrieved evidence.
Uses LLM to extract values from passages with citations.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from loguru import logger

from agents.llm_client import get_llm
from models import ExtractedField, SearchResult, SharedState, SourceCitation

EXTRACTION_SYSTEM_PROMPT = """You are a precise document extraction assistant for RFP (Request for Proposal) documents.
Your task: extract specific field values from provided document passages.

Rules:
1. ONLY extract information explicitly present in the provided passages.
2. If the information is NOT in the passages, return null.
3. NEVER guess or invent information.
4. Always cite which passage the information came from.
5. Return ONLY valid JSON, no explanation.

Output format:
{
  "value": "extracted value or null",
  "confidence": 0.0-1.0,
  "notes": "any important context or 'Not found in documents'"
}"""


def _format_evidence(results: List[SearchResult]) -> str:
    """Format search results as evidence text for the LLM."""
    if not results:
        return "No relevant passages found."

    parts = []
    for i, r in enumerate(results, 1):
        parts.append(
            f"[Passage {i}] Source: {r.file_name}, Page {r.page_number}\n"
            f"{r.text}\n"
        )
    return "\n---\n".join(parts)


def extract_field(
    field_name: str,
    evidence: List[SearchResult],
    bid_id: str,
    field_description: str = "",
) -> ExtractedField:
    """Extract a single field using LLM over evidence passages."""
    llm = get_llm()

    evidence_text = _format_evidence(evidence)
    prompt = f"""Extract the following field from the RFP documents for bid [{bid_id}]:

Field: {field_name}
{f'Description: {field_description}' if field_description else ''}

Evidence passages:
{evidence_text}

Extract the {field_name} value. If not found, set value to null."""

    try:
        result = llm.chat_json(prompt, system=EXTRACTION_SYSTEM_PROMPT, temperature=0.0)

        value = result.get("value")
        confidence = float(result.get("confidence", 0.5))
        notes = result.get("notes", "")

        # Build citations from evidence
        sources: List[SourceCitation] = []
        if value is not None and evidence:
            for r in evidence[:3]:  # cite top 3 sources
                sources.append(SourceCitation(
                    file=r.file_name,
                    page=r.page_number,
                    chunk_id=r.chunk_id,
                    excerpt=r.text[:200] if r.text else None,
                ))

        return ExtractedField(
            value=value,
            sources=sources,
            confidence=confidence,
            notes=notes,
        )

    except Exception as e:
        logger.warning(f"Extraction failed for '{field_name}': {e}")
        return ExtractedField(
            value=None,
            sources=[],
            confidence=0.0,
            notes=f"Extraction error: {e}",
        )


# Field descriptions for the LLM
FIELD_DESCRIPTIONS = {
    "Bid Number": "Official solicitation / RFP identifier number",
    "Title": "Full title of the bid or RFP",
    "Due Date": "Final submission deadline with time and timezone",
    "Bid Submission Type": "How the bid is submitted (electronic portal, sealed paper, email, etc.)",
    "Term of Bid": "Contract duration and renewal options",
    "Pre Bid Meeting": "Pre-bid meeting date, time, location, and whether mandatory",
    "Installation": "Whether installation/deployment/imaging services are required",
    "Bid Bond Requirement": "Required bid bond or security amount/percentage, or Not required",
    "Delivery Date": "Required delivery date or delivery window after award",
    "Payment Terms": "Payment terms (e.g. Net 30), invoicing conditions",
    "Any Additional Documentation Required": "Required affidavits, forms, certificates, insurance, W-9, etc.",
    "MFG for Registration": "Manufacturer(s) the bidder must be registered/authorized with",
    "Contract or Cooperative to use": "Referenced cooperative/state contract vehicle",
    "Model_no": "Model number(s) of requested product(s)",
    "Part_no": "Part/SKU number(s) of requested product(s)",
    "Product": "Product name(s) and quantity",
    "contact_info": "Procurement contact: name, email, phone",
    "company_name": "Issuing organization/agency name",
    "Bid Summary": "Concise 3-6 sentence summary of the bid",
    "Product Specification": "Key technical specifications (CPU, RAM, storage, display, warranty, etc.)",
}


def run_extraction_agent(
    state: SharedState,
    fields: Optional[List[str]] = None,
    max_workers: int = 4,
) -> SharedState:
    """
    Extract all fields (or a subset) from retrieved evidence.
    Runs field groups in parallel for performance.
    """
    start = time.time()
    log_entry: Dict[str, Any] = {
        "agent": "ExtractionAgent",
        "input": {"bid_id": state.bid_id, "fields": fields},
        "start_time": start,
    }

    try:
        from agents.retrieval_agent import ALL_FIELDS
        target_fields = fields or ALL_FIELDS
        bid_id = state.bid_id

        logger.info(f"[ExtractionAgent] Extracting {len(target_fields)} fields for [{bid_id}]")

        def extract_one(field: str) -> tuple[str, ExtractedField]:
            evidence = state.evidence.get(field, [])
            desc = FIELD_DESCRIPTIONS.get(field, "")
            result = extract_field(field, evidence, bid_id, desc)
            return field, result

        # Parallel extraction
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(extract_one, f): f for f in target_fields}
            for future in as_completed(futures):
                field_name, extracted = future.result()
                state.draft_fields[field_name] = extracted
                logger.debug(
                    f"  {field_name}: {str(extracted.value)[:60]} "
                    f"(conf={extracted.confidence:.2f})"
                )

        state.current_step = "extraction_done"
        log_entry.update({
            "output": {f: v.value for f, v in state.draft_fields.items()},
            "latency": time.time() - start,
            "status": "success",
        })

    except Exception as e:
        error_msg = f"ExtractionAgent error: {e}"
        logger.error(error_msg)
        state.errors.append(error_msg)
        state.current_step = "extraction_failed"
        log_entry.update({"status": "error", "error": str(e), "latency": time.time() - start})

    state.agent_logs.append(log_entry)
    return state
