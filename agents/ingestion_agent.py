"""
Ingestion Agent – detects file types, parses documents, triggers indexing.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict

from loguru import logger

from models import SharedState
from search.indexing import index_bid_folder
from search.vector_store import MongoVectorStore


def run_ingestion_agent(state: SharedState, vector_store: MongoVectorStore) -> SharedState:
    """
    Parses all files in the bid folder, chunks, embeds, and indexes them.
    Updates state.indexed and state.parsed_documents.
    """
    start = time.time()
    log_entry: Dict[str, Any] = {
        "agent": "IngestionAgent",
        "input": {"bid_id": state.bid_id, "bid_folder": state.bid_folder},
        "start_time": start,
    }

    try:
        if not state.bid_folder or not state.bid_id:
            raise ValueError("bid_folder and bid_id must be set in state")

        folder = Path(state.bid_folder)
        if not folder.exists():
            raise FileNotFoundError(f"Bid folder not found: {folder}")

        logger.info(f"[IngestionAgent] Indexing {state.bid_id} from {state.bid_folder}")

        summary = index_bid_folder(
            folder_path=folder,
            bid_id=state.bid_id,
            vector_store=vector_store,
            force_reindex=False,
        )

        state.indexed = True
        state.current_step = "ingestion_done"

        log_entry.update({
            "output": summary,
            "latency": time.time() - start,
            "status": "success",
        })

    except Exception as e:
        error_msg = f"IngestionAgent error: {e}"
        logger.error(error_msg)
        state.errors.append(error_msg)
        state.current_step = "ingestion_failed"
        log_entry.update({"status": "error", "error": str(e), "latency": time.time() - start})

    state.agent_logs.append(log_entry)
    return state
