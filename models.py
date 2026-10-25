"""
Shared data models (Pydantic) for the RFP Intelligence Platform.
All agent-to-agent messages and stored documents use these schemas.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ─── Document Types ──────────────────────────────────────────────────────────

class DocType(str, Enum):
    BID_PAGE = "bid_page"
    RFP = "rfp"
    ADDENDUM = "addendum"
    SPECS = "specs"
    AFFIDAVIT = "affidavit"
    CONTRACT = "contract"
    OTHER = "other"


class ParsedDocument(BaseModel):
    """Output of the ingestion/parsing stage."""
    bid_id: str
    file_name: str
    file_path: str
    doc_type: DocType
    addendum_number: Optional[int] = None
    page_count: int = 0
    pages: List["ParsedPage"] = Field(default_factory=list)
    tables: List[Dict[str, Any]] = Field(default_factory=list)
    document_date: Optional[str] = None
    parse_errors: List[str] = Field(default_factory=list)


class ParsedPage(BaseModel):
    """Single page/section of a parsed document."""
    page_number: int
    text: str
    tables: List[Dict[str, Any]] = Field(default_factory=list)
    section_heading: Optional[str] = None


# ─── Chunks ──────────────────────────────────────────────────────────────────

class DocumentChunk(BaseModel):
    """A single retrievable chunk with full metadata."""
    chunk_id: str
    bid_id: str
    file_name: str
    doc_type: DocType
    addendum_number: Optional[int] = None
    page_number: int
    section_heading: Optional[str] = None
    text: str
    embedding: Optional[List[float]] = None
    document_date: Optional[str] = None
    char_start: int = 0
    char_end: int = 0
    is_table: bool = False


# ─── Search / Retrieval ──────────────────────────────────────────────────────

class SearchQuery(BaseModel):
    """A search request to the RAG engine."""
    query: str
    bid_id: Optional[str] = None
    doc_types: Optional[List[DocType]] = None
    top_k: int = 5
    use_rerank: bool = True
    expand_query: bool = True


class SearchResult(BaseModel):
    """A single retrieved chunk with its relevance score and citation."""
    chunk_id: str
    bid_id: str
    file_name: str
    doc_type: DocType
    page_number: int
    section_heading: Optional[str] = None
    text: str
    score: float
    addendum_number: Optional[int] = None
    is_table: bool = False


class SearchResponse(BaseModel):
    """Full response from the search engine."""
    query: str
    rewritten_query: Optional[str] = None
    results: List[SearchResult] = Field(default_factory=list)
    total_found: int = 0


# ─── Extraction ──────────────────────────────────────────────────────────────

class SourceCitation(BaseModel):
    file: str
    page: int
    chunk_id: Optional[str] = None
    excerpt: Optional[str] = None


class ExtractedField(BaseModel):
    """One extracted field with citation and confidence."""
    value: Optional[Any] = None
    sources: List[SourceCitation] = Field(default_factory=list)
    confidence: float = 0.0
    notes: Optional[str] = None


class ValidationResult(BaseModel):
    field_name: str
    passed: bool
    reason: Optional[str] = None
    confidence: float = 0.0


class BidExtraction(BaseModel):
    """Complete structured extraction output for one bid."""
    bid_id: str
    fields: Dict[str, ExtractedField] = Field(default_factory=dict)
    addendum_changes: List[Dict[str, Any]] = Field(default_factory=list)
    validation: Dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


# ─── Agent State ─────────────────────────────────────────────────────────────

class AgentTask(BaseModel):
    """A sub-task assigned by the orchestrator."""
    task_id: str
    task_type: str  # "extract_field", "answer_question", "reconcile_addendum", etc.
    payload: Dict[str, Any] = Field(default_factory=dict)
    status: str = "pending"  # pending | running | done | failed
    result: Optional[Any] = None
    error: Optional[str] = None
    retries: int = 0


class SharedState(BaseModel):
    """Shared memory object passed between all agents in LangGraph."""
    # User goal
    mode: str = "extraction"  # "extraction" | "qa"
    bid_id: Optional[str] = None
    bid_folder: Optional[str] = None
    user_question: Optional[str] = None

    # Parsed documents
    parsed_documents: List[ParsedDocument] = Field(default_factory=list)
    indexed: bool = False

    # Retrieval evidence
    evidence: Dict[str, List[SearchResult]] = Field(default_factory=dict)

    # Extraction draft
    draft_fields: Dict[str, ExtractedField] = Field(default_factory=dict)
    addendum_changes: List[Dict[str, Any]] = Field(default_factory=list)

    # Validation
    validation_results: List[ValidationResult] = Field(default_factory=list)
    failed_fields: List[str] = Field(default_factory=list)
    retry_counts: Dict[str, int] = Field(default_factory=dict)

    # Final output
    final_extraction: Optional[BidExtraction] = None
    final_answer: Optional[str] = None

    # Orchestration
    plan: List[AgentTask] = Field(default_factory=list)
    current_step: str = "start"
    iteration: int = 0
    errors: List[str] = Field(default_factory=list)

    # Observability
    agent_logs: List[Dict[str, Any]] = Field(default_factory=list)
