"""
Q&A / Report Agent – answers natural-language questions with citations
and produces the final structured JSON output.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from loguru import logger

from agents.llm_client import get_llm
from models import BidExtraction, DocType, ExtractedField, SearchQuery, SharedState, ValidationResult
from search.retrieval import HybridSearchEngine

QA_SYSTEM_PROMPT = """You are an expert RFP analyst assistant. Answer user questions about bid documents.

Rules:
1. ONLY use information from the provided evidence passages.
2. Always cite your sources: mention the exact document name and page number.
3. If the answer is not in the passages, state clearly what could not be found.
4. Be precise, thorough, and analytical.
5. CROSS-BID COMPARISONS: When comparing multiple bids:
   - Structure your response with a dedicated section for Bid 1 (Dallas ISD - Student & Staff Computing Devices), a dedicated section for Bid 2 (State of Maryland - Dell Laptops w/ Extended Warranty), followed by a clear Side-by-Side Comparison Breakdown/Table.
   - NEVER confuse documents within the same bid folder (e.g. portal HTML vs main RFP PDF) as separate bids. A bid consists of ALL its documents together.
   - Compare all key dimensions: duration/coverage (e.g. 1-year vs 3-year), start dates (in-service vs delivery), service terms (repair SLAs, turnaround time, hot-swap loaners, parts stock), and Addendum modifications (e.g. Dallas ISD Addendum 1 requiring OEM-only warranty with no Accidental Damage Protection).
6. ADDENDUM PRECEDENCE: Addendums (e.g. Addendum 1, Addendum 2) strictly supersede and override the original RFP. Always prioritize and highlight the addendum's updated terms as the authoritative source."""

REPORT_SYSTEM_PROMPT = """You are generating a final JSON report for an RFP extraction system.
Produce a clean, structured JSON following the exact schema provided.
Return ONLY valid JSON, no explanation or markdown."""


def _is_comparison_query(question: str, bid_id: Optional[str]) -> bool:
    """Check if the user is asking to compare across multiple bids."""
    if bid_id is not None:
        return False
    q = question.lower()
    comp_keywords = ["compare", "both bids", "each bid", "difference", "versus", " vs ", "across bids", "between the bids"]
    return any(k in q for k in comp_keywords)


def answer_question(
    question: str,
    bid_id: Optional[str],
    engine: HybridSearchEngine,
    top_k: int = 8,
) -> str:
    """Answer a natural-language question about bids using RAG with Addendum Precedence and Cross-Bid Comparison support."""
    llm = get_llm()
    q_lower = question.lower()

    # ── CASE 1: Cross-Bid Comparison Query ─────────────────────────────────────
    if _is_comparison_query(question, bid_id):
        # Discover all available bids
        known_bids = ["Bid1", "Bid2"]
        try:
            db_bids = engine.vector_store.chunks.distinct("bid_id")
            if db_bids:
                known_bids = sorted(list(set(db_bids)))
        except Exception:
            pass

        bid_evidence_blocks = []
        for b in known_bids:
            # 1. Search this bid specifically
            b_req = SearchQuery(
                query=question,
                bid_id=b,
                top_k=6,
                use_rerank=True,
                expand_query=True,
            )
            b_resp = engine.search(b_req)
            b_results = list(b_resp.results)

            # 2. Check for addenda in this bid
            if any(k in q_lower for k in ["warranty", "guarantee", "deadline", "due date", "spec", "addendum", "repair"]):
                try:
                    add_req = SearchQuery(
                        query=f"addendum warranty specifications {question}",
                        bid_id=b,
                        top_k=4,
                        doc_types=[DocType.ADDENDUM],
                        use_rerank=True,
                    )
                    add_resp = engine.search(add_req)
                    seen_ids = {r.chunk_id for r in b_results}
                    for ar in reversed(add_resp.results):
                        if ar.chunk_id not in seen_ids:
                            b_results.insert(0, ar)
                            seen_ids.add(ar.chunk_id)
                except Exception as e:
                    logger.warning(f"Addendum search error for [{b}]: {e}")

            if b_results:
                b_label = (
                    "BID 1: DALLAS ISD (JA-207652 Student and Staff Computing Devices)"
                    if "bid1" in b.lower()
                    else "BID 2: STATE OF MARYLAND (PORFP #E20P4600040 - Dell Laptops w/ Extended Warranty)"
                    if "bid2" in b.lower()
                    else f"BID: {b}"
                )
                b_text = "\n\n".join(
                    f"[Source: {r.file_name}, Page {r.page_number} | DocType: {r.doc_type.value}]\n{r.text}"
                    for r in b_results
                )
                bid_evidence_blocks.append(f"=== {b_label} ===\n{b_text}")

        if not bid_evidence_blocks:
            return "This information was not found in the documents."

        combined_evidence = "\n\n" + ("=" * 70) + "\n\n".join(bid_evidence_blocks)

        prompt = f"""Question: {question}

Evidence grouped by bid:
{combined_evidence}

Please provide a detailed, side-by-side comparison answering the question:
1. Summarize the requirements for Bid 1 (Dallas ISD), citing documents and noting any Addendum changes.
2. Summarize the requirements for Bid 2 (State of Maryland), citing documents.
3. Provide a clear Comparison highlighting differences in duration, start dates, repair SLAs (e.g., hot-swap/turnaround), Accidental Damage Protection, and OEM warranty rules.
4. Cite sources with filename and page number for every point."""

        return llm.chat(prompt, system=QA_SYSTEM_PROMPT, temperature=0.1)

    # ── CASE 2: Single-Bid or Specific Target Query ────────────────────────────
    search_req = SearchQuery(
        query=question,
        bid_id=bid_id,
        top_k=top_k,
        use_rerank=True,
        expand_query=True,
    )
    response = engine.search(search_req)
    results = list(response.results)

    # Ensure addendum overrides are never missed for date/specification queries
    if any(k in q_lower for k in ["deadline", "due date", "when", "date", "change", "addendum", "spec", "meeting"]):
        try:
            add_req = SearchQuery(
                query=f"addendum due date extend deadline {question}",
                bid_id=bid_id,
                top_k=5,
                doc_types=[DocType.ADDENDUM],
                use_rerank=True,
            )
            add_resp = engine.search(add_req)
            seen_ids = {r.chunk_id for r in results}
            for ar in reversed(add_resp.results):
                if ar.chunk_id not in seen_ids:
                    results.insert(0, ar)
                    seen_ids.add(ar.chunk_id)
        except Exception as e:
            logger.warning(f"Addendum retrieval warning: {e}")

    if not results:
        return "This information was not found in the documents."

    # Format evidence with addendum flags
    evidence_text = "\n\n".join(
        f"[Source: {r.file_name}, Page {r.page_number} | DocType: {r.doc_type.value}]\n{r.text}"
        for r in results
    )

    prompt = f"""Question: {question}

Evidence from bid documents:
{evidence_text}

Answer the question using the evidence above. If an Addendum overrides the base document, explicitly highlight the Addendum's updated value. Cite sources by filename and page number."""

    return llm.chat(prompt, system=QA_SYSTEM_PROMPT, temperature=0.1)


def _build_final_extraction(state: SharedState) -> BidExtraction:
    """Assemble the final BidExtraction from state."""
    validation_summary = {
        "passed": sum(1 for r in state.validation_results if r.passed),
        "failed": sum(1 for r in state.validation_results if not r.passed),
        "not_found": sum(
            1 for r in state.validation_results
            if not r.passed
            and state.draft_fields.get(r.field_name, ExtractedField()).value is None
        ),
    }

    return BidExtraction(
        bid_id=state.bid_id or "unknown",
        fields=state.draft_fields,
        addendum_changes=state.addendum_changes,
        validation=validation_summary,
    )


def run_qa_report_agent(
    state: SharedState,
    engine: Optional[HybridSearchEngine] = None,
) -> SharedState:
    """
    In 'extraction' mode: assembles final JSON.
    In 'qa' mode: answers the user question.
    """
    start = time.time()
    log_entry: Dict[str, Any] = {
        "agent": "QAReportAgent",
        "input": {"mode": state.mode, "bid_id": state.bid_id},
        "start_time": start,
    }

    try:
        if state.mode == "extraction":
            logger.info(f"[QAReportAgent] Building final extraction for [{state.bid_id}]")
            state.final_extraction = _build_final_extraction(state)
            state.current_step = "complete"
            log_entry.update({
                "output": f"{len(state.draft_fields)} fields extracted",
                "latency": time.time() - start,
                "status": "success",
            })

        elif state.mode == "qa" and engine:
            logger.info(f"[QAReportAgent] Answering: {state.user_question}")
            answer = answer_question(
                question=state.user_question or "",
                bid_id=state.bid_id,
                engine=engine,
            )
            state.final_answer = answer
            state.current_step = "complete"
            log_entry.update({
                "output": answer[:200],
                "latency": time.time() - start,
                "status": "success",
            })

    except Exception as e:
        error_msg = f"QAReportAgent error: {e}"
        logger.error(error_msg)
        state.errors.append(error_msg)
        state.current_step = "qa_failed"
        log_entry.update({"status": "error", "error": str(e), "latency": time.time() - start})

    state.agent_logs.append(log_entry)
    return state
