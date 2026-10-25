"""
Addendum Reconciliation Agent – compares base RFP with addendums and applies
the latest valid changes to relevant fields.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from loguru import logger

from agents.llm_client import get_llm
from models import DocType, SearchQuery, SearchResult, SharedState, SourceCitation
from search.retrieval import HybridSearchEngine

RECONCILE_SYSTEM_PROMPT = """You are an expert RFP document analyst specializing in addendum reconciliation.
Your task: identify what changed between the original RFP and addendums.

Rules:
1. Addendums supersede the original document.
2. Later addendums supersede earlier ones.
3. Only report actual changes, not re-statements.
4. Return ONLY valid JSON.

Output format:
{
  "changes": [
    {
      "field": "field name affected",
      "original_value": "value in original RFP",
      "new_value": "value in addendum",
      "addendum_number": 1,
      "source_file": "filename",
      "source_page": 1,
      "reason": "brief explanation"
    }
  ]
}"""


def run_addendum_reconciliation_agent(
    state: SharedState,
    engine: HybridSearchEngine,
) -> SharedState:
    """
    Searches for addendum-specific changes and updates draft_fields accordingly.
    """
    start = time.time()
    log_entry: Dict[str, Any] = {
        "agent": "AddendumReconciliationAgent",
        "input": {"bid_id": state.bid_id},
        "start_time": start,
    }

    try:
        bid_id = state.bid_id
        llm = get_llm()

        logger.info(f"[AddendumAgent] Reconciling addendums for [{bid_id}]")

        # Search for addendum-specific content
        addendum_query = SearchQuery(
            query="addendum change amendment modification update revised extended",
            bid_id=bid_id,
            doc_types=[DocType.ADDENDUM],
            top_k=15,
            use_rerank=False,
            expand_query=True,
        )
        addendum_results = engine.search(addendum_query)

        if not addendum_results.results:
            logger.info(f"[AddendumAgent] No addendum content found for [{bid_id}]")
            state.current_step = "addendum_done"
            log_entry.update({"output": "no_addendums", "latency": time.time() - start, "status": "success"})
            state.agent_logs.append(log_entry)
            return state

        # Also get original RFP context for comparison
        rfp_query = SearchQuery(
            query="due date submission deadline original",
            bid_id=bid_id,
            doc_types=[DocType.RFP],
            top_k=5,
            use_rerank=False,
        )
        rfp_results = engine.search(rfp_query)

        # Build context for LLM
        addendum_text = "\n\n".join(
            f"[ADDENDUM {r.addendum_number or '?'}, File: {r.file_name}, Page {r.page_number}]\n{r.text}"
            for r in addendum_results.results[:10]
        )

        rfp_text = "\n\n".join(
            f"[ORIGINAL RFP, File: {r.file_name}, Page {r.page_number}]\n{r.text}"
            for r in rfp_results.results[:5]
        ) if rfp_results.results else "Original RFP content not retrieved."

        prompt = f"""Analyze the following addendum content for bid [{bid_id}] and identify what changed.

ORIGINAL RFP EXCERPTS:
{rfp_text}

ADDENDUM CONTENT:
{addendum_text}

Identify all field changes made by the addendum(s). Focus on: dates, deadlines, requirements, specifications."""

        result = llm.chat_json(prompt, system=RECONCILE_SYSTEM_PROMPT, temperature=0.0)
        changes = result.get("changes", [])

        # Apply changes to draft fields
        for change in changes:
            field = change.get("field", "")
            new_value = change.get("new_value")
            addendum_num = change.get("addendum_number")
            source_file = change.get("source_file", "")
            source_page = change.get("source_page", 1)

            if field in state.draft_fields and new_value:
                old_field = state.draft_fields[field]
                old_notes = old_field.notes or ""

                state.draft_fields[field].value = new_value
                state.draft_fields[field].notes = (
                    f"Updated by Addendum {addendum_num}. "
                    f"Original: {old_field.value}. {change.get('reason', '')}"
                )
                state.draft_fields[field].sources.append(
                    SourceCitation(file=source_file, page=source_page)
                )

            state.addendum_changes.append(change)

        logger.info(f"[AddendumAgent] Applied {len(changes)} addendum changes")
        state.current_step = "addendum_done"
        log_entry.update({
            "output": {"changes": len(changes)},
            "latency": time.time() - start,
            "status": "success",
        })

    except Exception as e:
        error_msg = f"AddendumAgent error: {e}"
        logger.error(error_msg)
        state.errors.append(error_msg)
        state.current_step = "addendum_failed"
        log_entry.update({"status": "error", "error": str(e), "latency": time.time() - start})

    state.agent_logs.append(log_entry)
    return state
