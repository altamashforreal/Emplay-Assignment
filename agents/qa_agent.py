"""
Q&A / Report Agent – answers natural-language questions with citations
and produces the final structured JSON output.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from loguru import logger

from agents.llm_client import get_llm
from models import BidExtraction, ExtractedField, SearchQuery, SharedState, ValidationResult
from search.retrieval import HybridSearchEngine

QA_SYSTEM_PROMPT = """You are an expert RFP analyst assistant. Answer user questions about bid documents.

Rules:
1. ONLY use information from the provided evidence passages.
2. Always cite your sources: mention the document name and page.
3. If the answer is not in the passages, say "This information was not found in the documents."
4. Be precise and concise.
5. For comparisons, organize clearly with headers."""

REPORT_SYSTEM_PROMPT = """You are generating a final JSON report for an RFP extraction system.
Produce a clean, structured JSON following the exact schema provided.
Return ONLY valid JSON, no explanation or markdown."""


def answer_question(
    question: str,
    bid_id: Optional[str],
    engine: HybridSearchEngine,
    top_k: int = 8,
) -> str:
    """Answer a natural-language question about bids using RAG."""
    llm = get_llm()

    # Search for relevant content
    search_req = SearchQuery(
        query=question,
        bid_id=bid_id,
        top_k=top_k,
        use_rerank=True,
        expand_query=True,
    )
    response = engine.search(search_req)

    if not response.results:
        return "This information was not found in the documents."

    # Format evidence
    evidence_text = "\n\n".join(
        f"[Source: {r.file_name}, Page {r.page_number}]\n{r.text}"
        for r in response.results
    )

    prompt = f"""Question: {question}

Evidence from bid documents:
{evidence_text}

Answer the question using only the evidence above. Cite sources by filename and page number."""

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
