"""
Retrieval Evaluation Module.

Evaluates the search engine on a curated question set covering both bids.
Reports: Recall@k, MRR, nDCG for different configurations:
  1. Vector-only search
  2. BM25-only search
  3. Hybrid (RRF fusion)
  4. Hybrid + Re-ranking
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger
from rich.console import Console
from rich.table import Table

from models import DocType, SearchQuery

console = Console()

# ─── Evaluation Dataset ──────────────────────────────────────────────────────
# Format: {question, bid_id, expected_keywords (any keyword in correct chunk)}
EVAL_QUESTIONS = [
    # Bid1 questions
    {
        "id": "q01",
        "question": "What is the submission deadline for Bid1?",
        "bid_id": "Bid1",
        "expected_keywords": ["due date", "deadline", "submission", "closing"],
    },
    {
        "id": "q02",
        "question": "Is attendance at the pre-bid meeting mandatory for Bid1?",
        "bid_id": "Bid1",
        "expected_keywords": ["pre-bid", "pre bid", "mandatory", "meeting"],
    },
    {
        "id": "q03",
        "question": "What bid bond is required for Bid1?",
        "bid_id": "Bid1",
        "expected_keywords": ["bid bond", "bond", "surety", "security"],
    },
    {
        "id": "q04",
        "question": "What is the contract term for Bid1?",
        "bid_id": "Bid1",
        "expected_keywords": ["term", "duration", "period", "renewal"],
    },
    {
        "id": "q05",
        "question": "What addendum changes were made in Bid1 Addendum 2?",
        "bid_id": "Bid1",
        "expected_keywords": ["addendum 2", "amendment", "change", "revised"],
    },
    {
        "id": "q06",
        "question": "What model numbers are specified in Bid1?",
        "bid_id": "Bid1",
        "expected_keywords": ["model", "model number", "item", "specification"],
    },
    {
        "id": "q07",
        "question": "Who is the procurement contact for Bid1?",
        "bid_id": "Bid1",
        "expected_keywords": ["contact", "email", "phone", "buyer", "officer"],
    },
    {
        "id": "q08",
        "question": "What documents must be submitted with Bid1?",
        "bid_id": "Bid1",
        "expected_keywords": ["affidavit", "document", "certificate", "form", "required"],
    },
    # Bid2 questions
    {
        "id": "q09",
        "question": "What are the technical specifications for Dell laptops in Bid2?",
        "bid_id": "Bid2",
        "expected_keywords": ["specification", "cpu", "ram", "storage", "processor", "memory"],
    },
    {
        "id": "q10",
        "question": "What warranty is required for the Dell laptops?",
        "bid_id": "Bid2",
        "expected_keywords": ["warranty", "extended warranty", "guarantee", "support"],
    },
    {
        "id": "q11",
        "question": "What affidavits are required for Bid2?",
        "bid_id": "Bid2",
        "expected_keywords": ["affidavit", "mercury", "contract affidavit", "certification"],
    },
    {
        "id": "q12",
        "question": "What is the payment term for Bid2?",
        "bid_id": "Bid2",
        "expected_keywords": ["payment", "net 30", "invoice", "billing"],
    },
    {
        "id": "q13",
        "question": "What manufacturer registration is required for Bid2?",
        "bid_id": "Bid2",
        "expected_keywords": ["manufacturer", "dell", "authorized", "registration", "oem"],
    },
    {
        "id": "q14",
        "question": "What is the delivery requirement for Bid2?",
        "bid_id": "Bid2",
        "expected_keywords": ["delivery", "ship", "lead time", "after award"],
    },
    {
        "id": "q15",
        "question": "What is the bid number for Bid2?",
        "bid_id": "Bid2",
        "expected_keywords": ["bid number", "solicitation", "rfp", "reference"],
    },
    # Cross-bid questions
    {
        "id": "q16",
        "question": "Compare the warranty requirements of both bids",
        "bid_id": None,
        "expected_keywords": ["warranty", "guarantee", "support"],
    },
    {
        "id": "q17",
        "question": "What submission methods are accepted across all bids?",
        "bid_id": None,
        "expected_keywords": ["submission", "electronic", "portal", "sealed", "email"],
    },
    {
        "id": "q18",
        "question": "What is the installation requirement for computing devices?",
        "bid_id": "Bid1",
        "expected_keywords": ["installation", "deploy", "imaging", "setup"],
    },
    {
        "id": "q19",
        "question": "What cooperative contract should be referenced?",
        "bid_id": None,
        "expected_keywords": ["cooperative", "contract", "purchasing", "vehicle"],
    },
    {
        "id": "q20",
        "question": "What is the agency name issuing the Dell laptop bid?",
        "bid_id": "Bid2",
        "expected_keywords": ["agency", "organization", "district", "company", "institution"],
    },
]


def _result_is_relevant(result, expected_keywords: List[str]) -> bool:
    """Check if a search result is relevant based on keyword matching."""
    text_lower = result.text.lower()
    fname_lower = result.file_name.lower()
    combined = text_lower + " " + fname_lower
    return any(kw.lower() in combined for kw in expected_keywords)


def _recall_at_k(results, expected_keywords: List[str], k: int) -> float:
    """Recall@k: is at least one relevant result in top-k?"""
    top_k = results[:k]
    return 1.0 if any(_result_is_relevant(r, expected_keywords) for r in top_k) else 0.0


def _mrr(results, expected_keywords: List[str]) -> float:
    """Mean Reciprocal Rank of first relevant result."""
    for rank, result in enumerate(results, 1):
        if _result_is_relevant(result, expected_keywords):
            return 1.0 / rank
    return 0.0


def _ndcg_at_k(results, expected_keywords: List[str], k: int) -> float:
    """nDCG@k with binary relevance."""
    dcg = 0.0
    for rank, result in enumerate(results[:k], 1):
        if _result_is_relevant(result, expected_keywords):
            dcg += 1.0 / math.log2(rank + 1)
    ideal_dcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(k, 3) + 1))
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def _run_config(engine, config: dict, questions: List[dict]) -> dict:
    """Run evaluation for a single configuration."""
    recall_1 = recall_5 = mrr = ndcg_5 = 0.0
    n = len(questions)

    for q in questions:
        req = SearchQuery(
            query=q["question"],
            bid_id=q.get("bid_id"),
            top_k=10,
            use_rerank=config.get("use_rerank", False),
            expand_query=config.get("expand_query", True),
        )

        # Override search type
        search_type = config.get("type", "hybrid")
        if search_type == "vector":
            from search.bm25_index import get_bm25_index
            # Vector-only: bypass BM25
            from search.embedder import get_embedder
            emb = get_embedder().embed_text(req.query)
            results = engine.vector_store.vector_search(emb, 10, req.bid_id)
        elif search_type == "bm25":
            # BM25-only
            bm25 = get_bm25_index()
            results = bm25.search_all(req.query, 10, req.bid_id)
        else:
            # Full hybrid
            resp = engine.search(req)
            results = resp.results

        expected = q["expected_keywords"]
        recall_1 += _recall_at_k(results, expected, 1)
        recall_5 += _recall_at_k(results, expected, 5)
        mrr += _mrr(results, expected)
        ndcg_5 += _ndcg_at_k(results, expected, 5)

    return {
        "recall@1": round(recall_1 / n, 3),
        "recall@5": round(recall_5 / n, 3),
        "mrr": round(mrr / n, 3),
        "ndcg@5": round(ndcg_5 / n, 3),
        "n_questions": n,
    }


def run_evaluation(bid_id: Optional[str] = None):
    """Run full retrieval evaluation and print results."""
    from search.retrieval import HybridSearchEngine
    from search.vector_store import MongoVectorStore

    vs = MongoVectorStore()
    engine = HybridSearchEngine(vs)

    questions = EVAL_QUESTIONS
    if bid_id:
        questions = [q for q in questions if q.get("bid_id") == bid_id or q.get("bid_id") is None]

    configs = [
        {"name": "Vector Only", "type": "vector", "use_rerank": False, "expand_query": False},
        {"name": "BM25 Only", "type": "bm25", "use_rerank": False, "expand_query": False},
        {"name": "Hybrid (RRF)", "type": "hybrid", "use_rerank": False, "expand_query": True},
        {"name": "Hybrid + Rerank", "type": "hybrid", "use_rerank": True, "expand_query": True},
    ]

    table = Table(title=f"Retrieval Evaluation Results ({len(questions)} questions)")
    table.add_column("Configuration", style="cyan", no_wrap=True)
    table.add_column("Recall@1", style="yellow", justify="center")
    table.add_column("Recall@5", style="yellow", justify="center")
    table.add_column("MRR", style="green", justify="center")
    table.add_column("nDCG@5", style="green", justify="center")

    all_results = {}
    for config in configs:
        console.print(f"[dim]Running: {config['name']}...[/dim]")
        metrics = _run_config(engine, config, questions)
        all_results[config["name"]] = metrics
        table.add_row(
            config["name"],
            str(metrics["recall@1"]),
            str(metrics["recall@5"]),
            str(metrics["mrr"]),
            str(metrics["ndcg@5"]),
        )

    console.print(table)

    # Save results
    out_path = Path("./eval/evaluation_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"configs": all_results, "questions": EVAL_QUESTIONS}, f, indent=2)
    console.print(f"\n[green]Results saved to {out_path}[/green]")

    vs.close()
    return all_results
