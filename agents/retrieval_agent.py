"""
Retrieval / Search Agent – rewrites queries, calls the hybrid search engine,
returns ranked evidence passages with citations.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from loguru import logger

from agents.llm_client import get_llm
from models import DocType, SearchQuery, SearchResult, SharedState
from search.retrieval import HybridSearchEngine


# Fields grouped for parallel extraction
FIELD_GROUPS = {
    "dates_logistics": [
        "Due Date", "Pre Bid Meeting", "Delivery Date", "Term of Bid"
    ],
    "commercial_legal": [
        "Bid Bond Requirement", "Payment Terms", "Any Additional Documentation Required",
        "Contract or Cooperative to use", "Bid Submission Type"
    ],
    "product_specs": [
        "Model_no", "Part_no", "Product", "Product Specification",
        "MFG for Registration", "Installation"
    ],
    "org_info": [
        "Bid Number", "Title", "contact_info", "company_name",
        "Bid Summary"
    ],
}

ALL_FIELDS = [f for group in FIELD_GROUPS.values() for f in group]

# Search queries per field
FIELD_QUERIES = {
    "Due Date": "submission deadline due date bid opening closing date time",
    "Pre Bid Meeting": "pre-bid meeting pre-proposal conference mandatory attendance date time location",
    "Delivery Date": "delivery date delivery window required delivery after award",
    "Term of Bid": "contract term duration period of performance renewal options",
    "Bid Bond Requirement": "bid bond requirement performance bond surety security deposit amount percentage",
    "Payment Terms": "payment terms net 30 invoice billing conditions",
    "Any Additional Documentation Required": "required documentation affidavit forms certificates insurance W-9",
    "Contract or Cooperative to use": "cooperative contract vehicle state contract purchasing cooperative",
    "Bid Submission Type": "bid submission method electronic portal sealed paper email",
    "Model_no": "model number model no item number product model",
    "Part_no": "part number part no SKU item number",
    "Product": "product name quantity items required",
    "Product Specification": "technical specifications CPU RAM storage display screen warranty configuration",
    "MFG for Registration": "manufacturer registration authorized dealer OEM brand",
    "Installation": "installation deployment imaging setup services required",
    "Bid Number": "bid number solicitation number RFP number reference number",
    "Title": "bid title solicitation title project name",
    "contact_info": "procurement contact officer buyer email phone number",
    "company_name": "issuing organization agency school district company name",
    "Bid Summary": "bid overview purpose scope services products requested",
}


def retrieve_evidence_for_field(
    field_name: str,
    bid_id: str,
    engine: HybridSearchEngine,
    top_k: int = 5,
) -> List[SearchResult]:
    """Retrieve evidence passages for a single field."""
    query = FIELD_QUERIES.get(field_name, field_name)
    search_req = SearchQuery(
        query=query,
        bid_id=bid_id,
        top_k=top_k,
        use_rerank=True,
        expand_query=True,
    )
    response = engine.search(search_req)
    return response.results


def run_retrieval_agent(
    state: SharedState,
    engine: HybridSearchEngine,
    fields: Optional[List[str]] = None,
) -> SharedState:
    """
    Retrieve evidence for all fields (or a subset for retry).
    Stores results in state.evidence[field_name].
    """
    start = time.time()
    log_entry: Dict[str, Any] = {
        "agent": "RetrievalAgent",
        "input": {"bid_id": state.bid_id, "fields": fields},
        "start_time": start,
    }

    try:
        target_fields = fields or ALL_FIELDS
        bid_id = state.bid_id

        logger.info(f"[RetrievalAgent] Retrieving evidence for {len(target_fields)} fields")

        for field in target_fields:
            try:
                results = retrieve_evidence_for_field(
                    field_name=field,
                    bid_id=bid_id,
                    engine=engine,
                    top_k=5,
                )
                state.evidence[field] = results
                logger.debug(f"  {field}: {len(results)} passages retrieved")
            except Exception as e:
                logger.warning(f"  Retrieval failed for '{field}': {e}")
                state.evidence[field] = []

        state.current_step = "retrieval_done"
        log_entry.update({
            "output": {f: len(r) for f, r in state.evidence.items()},
            "latency": time.time() - start,
            "status": "success",
        })

    except Exception as e:
        error_msg = f"RetrievalAgent error: {e}"
        logger.error(error_msg)
        state.errors.append(error_msg)
        state.current_step = "retrieval_failed"
        log_entry.update({"status": "error", "error": str(e), "latency": time.time() - start})

    state.agent_logs.append(log_entry)
    return state
