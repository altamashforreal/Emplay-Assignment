"""
Orchestrator – LangGraph-based multi-agent coordinator.

Graph structure (extraction mode):
  START → ingest → retrieve → extract → addendum → validate → [retry?] → report → END

Graph structure (QA mode):
  START → ingest (if needed) → qa → END

Design decisions:
  - LangGraph chosen for explicit stateful graph with conditional edges,
    making the retry/feedback loop easy to implement and audit.
  - SharedState Pydantic model serialized to dict for LangGraph compatibility.
  - Max retry loop: 3 attempts per failed field (configurable).
  - Agents run as pure functions (state in, state out) for testability.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from loguru import logger

from agents.addendum_agent import run_addendum_reconciliation_agent
from agents.extraction_agent import run_extraction_agent
from agents.ingestion_agent import run_ingestion_agent
from agents.qa_agent import run_qa_report_agent
from agents.retrieval_agent import run_retrieval_agent
from agents.validator_agent import run_validator_agent
from config import settings
from models import SharedState
from search.retrieval import HybridSearchEngine
from search.vector_store import MongoVectorStore


# ─── State serialization helpers ─────────────────────────────────────────────

def _to_dict(state: SharedState) -> dict:
    return json.loads(state.model_dump_json())


def _from_dict(d: dict) -> SharedState:
    return SharedState.model_validate(d)


# ─── Node wrappers ────────────────────────────────────────────────────────────

def make_ingest_node(vector_store: MongoVectorStore) -> Callable:
    def ingest_node(state_dict: dict) -> dict:
        state = _from_dict(state_dict)
        updated = run_ingestion_agent(state, vector_store)
        return _to_dict(updated)
    return ingest_node


def make_retrieve_node(engine: HybridSearchEngine) -> Callable:
    def retrieve_node(state_dict: dict) -> dict:
        state = _from_dict(state_dict)
        fields_to_retrieve = state.failed_fields if state.iteration > 0 else None
        updated = run_retrieval_agent(state, engine, fields=fields_to_retrieve)
        return _to_dict(updated)
    return retrieve_node


def make_extract_node() -> Callable:
    def extract_node(state_dict: dict) -> dict:
        state = _from_dict(state_dict)
        fields_to_extract = state.failed_fields if state.iteration > 0 else None
        updated = run_extraction_agent(state, fields=fields_to_extract)
        return _to_dict(updated)
    return extract_node


def make_addendum_node(engine: HybridSearchEngine) -> Callable:
    def addendum_node(state_dict: dict) -> dict:
        state = _from_dict(state_dict)
        updated = run_addendum_reconciliation_agent(state, engine)
        return _to_dict(updated)
    return addendum_node


def make_validate_node() -> Callable:
    def validate_node(state_dict: dict) -> dict:
        state = _from_dict(state_dict)
        updated = run_validator_agent(state)
        return _to_dict(updated)
    return validate_node


def make_report_node(engine: HybridSearchEngine) -> Callable:
    def report_node(state_dict: dict) -> dict:
        state = _from_dict(state_dict)
        updated = run_qa_report_agent(state, engine)
        return _to_dict(updated)
    return report_node


def make_qa_node(engine: HybridSearchEngine) -> Callable:
    def qa_node(state_dict: dict) -> dict:
        state = _from_dict(state_dict)
        updated = run_qa_report_agent(state, engine)
        return _to_dict(updated)
    return qa_node


# ─── Conditional edge functions ───────────────────────────────────────────────

def route_after_ingest(state_dict: dict) -> str:
    state = _from_dict(state_dict)
    if not state.indexed:
        logger.error("Ingestion failed, aborting.")
        return "end"
    if state.mode == "qa":
        return "qa"
    return "retrieve"


def route_after_validate(state_dict: dict) -> str:
    """Decide whether to retry failed fields or proceed to report."""
    state = _from_dict(state_dict)

    # Increment iteration counter – we do this in the routing function
    # so it affects the next cycle
    if not state.failed_fields:
        return "report"

    if state.iteration >= settings.max_retries:
        logger.warning(
            f"Max retries ({settings.max_retries}) reached. "
            f"Proceeding with {len(state.failed_fields)} unresolved fields."
        )
        return "report"

    logger.info(
        f"Retrying {len(state.failed_fields)} failed fields "
        f"(iteration {state.iteration}/{settings.max_retries})"
    )
    return "retrieve"


# ─── Graph builder ────────────────────────────────────────────────────────────

def build_graph(vector_store: MongoVectorStore, engine: HybridSearchEngine):
    """Build and compile the LangGraph workflow."""
    try:
        from langgraph.graph import StateGraph, END
    except ImportError:
        raise ImportError("Install langgraph: pip install langgraph>=0.2.0")

    graph = StateGraph(dict)

    # ── Nodes ──
    graph.add_node("ingest", make_ingest_node(vector_store))
    graph.add_node("retrieve", make_retrieve_node(engine))
    graph.add_node("extract", make_extract_node())
    graph.add_node("addendum", make_addendum_node(engine))
    graph.add_node("validate", make_validate_node())
    graph.add_node("report", make_report_node(engine))
    graph.add_node("qa", make_qa_node(engine))

    # ── Edges ──
    graph.set_entry_point("ingest")

    graph.add_conditional_edges(
        "ingest",
        route_after_ingest,
        {"retrieve": "retrieve", "qa": "qa", "end": END},
    )
    graph.add_edge("retrieve", "extract")
    graph.add_edge("extract", "addendum")
    graph.add_edge("addendum", "validate")
    graph.add_conditional_edges(
        "validate",
        route_after_validate,
        {"report": "report", "retrieve": "retrieve"},
    )
    graph.add_edge("report", END)
    graph.add_edge("qa", END)

    return graph.compile()


# ─── Main orchestrator class ──────────────────────────────────────────────────

class RFPOrchestrator:
    """
    High-level interface to the multi-agent system.
    Manages the vector store, search engine, and LangGraph workflow.
    """

    def __init__(self):
        logger.info("Initializing RFP Orchestrator...")
        self.vector_store = MongoVectorStore()
        self.engine = HybridSearchEngine(self.vector_store)
        self._graph = build_graph(self.vector_store, self.engine)
        logger.info("Orchestrator ready.")

    def run_extraction(
        self,
        bid_folder: str,
        bid_id: str,
        output_dir: Optional[str] = None,
    ) -> dict:
        """
        Run the full extraction pipeline for a bid folder.

        Args:
            bid_folder: Path to folder containing bid documents.
            bid_id:     Unique identifier (e.g. "Bid1").
            output_dir: If set, save JSON and trace to this directory.

        Returns:
            Structured extraction dict (BidExtraction schema).
        """
        initial_state = SharedState(
            mode="extraction",
            bid_id=bid_id,
            bid_folder=bid_folder,
        )

        logger.info(f"╔══ Starting extraction pipeline for [{bid_id}] ══╗")
        start = time.time()

        result_dict = self._graph.invoke(_to_dict(initial_state))
        final_state = _from_dict(result_dict)

        elapsed = time.time() - start
        logger.info(f"╚══ Extraction complete in {elapsed:.1f}s ══╝")

        # Assemble output
        output = {}
        if final_state.final_extraction:
            output = json.loads(final_state.final_extraction.model_dump_json())

        # Save outputs
        if output_dir:
            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

            json_path = out_dir / f"{bid_id}_extraction.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=2, default=str)
            logger.info(f"📄 Extraction saved: {json_path}")

            trace_path = out_dir / f"{bid_id}_trace.json"
            with open(trace_path, "w", encoding="utf-8") as f:
                json.dump(final_state.agent_logs, f, indent=2, default=str)
            logger.info(f"📊 Trace saved: {trace_path}")

        return output

    def run_qa(
        self,
        question: str,
        bid_id: Optional[str] = None,
        bid_folder: Optional[str] = None,
    ) -> str:
        """
        Answer a natural-language question about bid documents.

        Args:
            question:   The natural-language question.
            bid_id:     Optional – restrict to one bid.
            bid_folder: Optional – index this folder first before answering.

        Returns:
            Cited answer string.
        """
        # If a folder is provided and not yet indexed, index it first
        already_indexed = (
            bid_folder is None
            or (bid_id is not None and self.vector_store.bid_exists(bid_id))
        )

        initial_state = SharedState(
            mode="qa",
            bid_id=bid_id,
            bid_folder=bid_folder,
            user_question=question,
            indexed=already_indexed,
        )

        logger.info(f"╔══ Q&A: {question[:80]} ══╗")
        result_dict = self._graph.invoke(_to_dict(initial_state))
        final_state = _from_dict(result_dict)

        return final_state.final_answer or "Unable to answer the question."

    def index_bid(
        self,
        bid_folder: str,
        bid_id: str,
        force: bool = False,
    ) -> dict:
        """Index a bid folder without running extraction."""
        from search.indexing import index_bid_folder
        return index_bid_folder(bid_folder, bid_id, self.vector_store, force)

    def close(self):
        self.vector_store.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
