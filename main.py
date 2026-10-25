"""
Main CLI entry point for the RFP Intelligence Platform.

Usage:
  python main.py index   --bid ./Bid1 --bid-id Bid1
  python main.py extract --bid ./Bid1 --bid-id Bid1 --output ./output
  python main.py ask     --question "What is the deadline?" --bid-id Bid1
  python main.py search  --query "bid bond requirements" --bid-id Bid1
  python main.py api     [--host 0.0.0.0] [--port 8000]
  python main.py eval    [--bid-id Bid1]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from loguru import logger
from rich.console import Console
from rich.json import JSON
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint

from config import settings

console = Console()


def setup_logging():
    logger.remove()
    logger.add(
        sys.stderr,
        level=settings.log_level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    )
    log_file = settings.log_dir / "rfp_platform.log"
    logger.add(log_file, rotation="10 MB", retention="7 days", level="DEBUG")


def cmd_index(args):
    """Index a bid folder."""
    from search.vector_store import MongoVectorStore
    from search.indexing import index_bid_folder

    console.print(Panel(f"[bold blue]Indexing bid: {args.bid_id}[/bold blue]"))

    vs = MongoVectorStore()
    summary = index_bid_folder(
        folder_path=args.bid,
        bid_id=args.bid_id,
        vector_store=vs,
        force_reindex=args.force,
    )
    vs.close()

    table = Table(title="Indexing Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    for k, v in summary.items():
        if k != "errors":
            table.add_row(k, str(v))
    console.print(table)

    if summary.get("errors"):
        console.print(f"[yellow]⚠ {len(summary['errors'])} warnings:[/yellow]")
        for err in summary["errors"][:5]:
            console.print(f"  {err}")


def cmd_extract(args):
    """Run full extraction pipeline."""
    from agents.orchestrator import RFPOrchestrator

    console.print(Panel(f"[bold magenta]Extracting bid: {args.bid_id}[/bold magenta]"))

    orch = RFPOrchestrator()
    result = orch.run_extraction(
        bid_folder=args.bid,
        bid_id=args.bid_id,
        output_dir=args.output,
    )
    orch.close()

    # Show summary
    fields = result.get("fields", {})
    validation = result.get("validation", {})
    console.print(f"\n✅ Extracted {len(fields)} fields")
    console.print(f"  Passed: {validation.get('passed', 0)} | "
                  f"Failed: {validation.get('failed', 0)} | "
                  f"Not found: {validation.get('not_found', 0)}")

    if args.output:
        out = Path(args.output) / f"{args.bid_id}_extraction.json"
        console.print(f"\n📄 Output saved to: {out}")

    # Print JSON
    if args.verbose:
        console.print(JSON(json.dumps(result, default=str)))


def cmd_ask(args):
    """Answer a question."""
    from agents.orchestrator import RFPOrchestrator

    console.print(Panel(f"[bold cyan]Q&A Mode[/bold cyan]\nQuestion: {args.question}"))

    orch = RFPOrchestrator()
    answer = orch.run_qa(
        question=args.question,
        bid_id=args.bid_id,
        bid_folder=args.bid,
    )
    orch.close()

    console.print(Panel(answer, title="[green]Answer[/green]", border_style="green"))


def cmd_search(args):
    """Search the index."""
    from models import DocType, SearchQuery
    from search.retrieval import HybridSearchEngine
    from search.vector_store import MongoVectorStore

    console.print(Panel(f"[bold yellow]Searching: {args.query}[/bold yellow]"))

    vs = MongoVectorStore()
    engine = HybridSearchEngine(vs)

    doc_types = None
    if args.doc_type:
        try:
            doc_types = [DocType(args.doc_type)]
        except ValueError:
            console.print(f"[red]Invalid doc_type: {args.doc_type}[/red]")
            return

    req = SearchQuery(
        query=args.query,
        bid_id=args.bid_id,
        doc_types=doc_types,
        top_k=args.top_k,
        use_rerank=not args.no_rerank,
        expand_query=True,
    )
    response = engine.search(req)
    vs.close()

    if response.rewritten_query:
        console.print(f"[dim]Query expanded: {response.rewritten_query}[/dim]\n")

    for i, r in enumerate(response.results, 1):
        console.print(Panel(
            f"[bold]{r.text[:400]}[/bold]\n\n"
            f"[dim]Source: {r.file_name} | Page: {r.page_number} | "
            f"Score: {r.score:.4f} | Doc: {r.doc_type}[/dim]",
            title=f"Result {i}",
            border_style="blue",
        ))


def cmd_api(args):
    """Start the FastAPI server."""
    import uvicorn
    console.print(Panel(f"[bold green]Starting API on {args.host}:{args.port}[/bold green]"))
    uvicorn.run(
        "api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


def cmd_eval(args):
    """Run retrieval evaluation."""
    from eval.evaluator import run_evaluation
    console.print(Panel("[bold]Running Retrieval Evaluation[/bold]"))
    run_evaluation(bid_id=args.bid_id)


def main():
    setup_logging()

    parser = argparse.ArgumentParser(
        description="RFP Intelligence Platform CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # index
    p_index = subparsers.add_parser("index", help="Index a bid folder")
    p_index.add_argument("--bid", required=True, help="Path to bid folder")
    p_index.add_argument("--bid-id", required=True, dest="bid_id")
    p_index.add_argument("--force", action="store_true", help="Force re-index")

    # extract
    p_extract = subparsers.add_parser("extract", help="Run full extraction")
    p_extract.add_argument("--bid", required=True)
    p_extract.add_argument("--bid-id", required=True, dest="bid_id")
    p_extract.add_argument("--output", default="./output")
    p_extract.add_argument("--verbose", "-v", action="store_true")

    # ask
    p_ask = subparsers.add_parser("ask", help="Q&A mode")
    p_ask.add_argument("--question", "-q", required=True)
    p_ask.add_argument("--bid-id", dest="bid_id", default=None)
    p_ask.add_argument("--bid", default=None, help="Bid folder to index first (optional)")

    # search
    p_search = subparsers.add_parser("search", help="Search the index")
    p_search.add_argument("--query", "-q", required=True)
    p_search.add_argument("--bid-id", dest="bid_id", default=None)
    p_search.add_argument("--doc-type", dest="doc_type", default=None)
    p_search.add_argument("--top-k", dest="top_k", type=int, default=5)
    p_search.add_argument("--no-rerank", action="store_true")

    # api
    p_api = subparsers.add_parser("api", help="Start REST API server")
    p_api.add_argument("--host", default=settings.api_host)
    p_api.add_argument("--port", type=int, default=settings.api_port)
    p_api.add_argument("--reload", action="store_true")

    # eval
    p_eval = subparsers.add_parser("eval", help="Run retrieval evaluation")
    p_eval.add_argument("--bid-id", dest="bid_id", default=None)

    args = parser.parse_args()

    commands = {
        "index": cmd_index,
        "extract": cmd_extract,
        "ask": cmd_ask,
        "search": cmd_search,
        "api": cmd_api,
        "eval": cmd_eval,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
