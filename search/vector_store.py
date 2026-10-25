"""
MongoDB Atlas Vector Store – stores document chunks with embeddings.
Uses MongoDB Atlas Vector Search for approximate nearest neighbor search.
Supports incremental indexing (add new bids without re-indexing everything).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from loguru import logger
from pymongo import MongoClient, UpdateOne
from pymongo.collection import Collection
from pymongo.errors import BulkWriteError

from config import settings
from models import DocType, DocumentChunk, SearchResult


class MongoVectorStore:
    """
    Manages document chunk storage and vector search via MongoDB Atlas.

    Required: Create a vector search index named 'vector_index' in Atlas UI
    with the following definition:
    {
      "fields": [{
        "type": "vector",
        "path": "embedding",
        "numDimensions": 768,
        "similarity": "cosine"
      }]
    }
    """

    def __init__(self):
        self.client = MongoClient(settings.mongodb_uri)
        self.db = self.client[settings.mongodb_db_name]
        self.chunks: Collection = self.db[settings.mongodb_collection_chunks]
        self._ensure_indexes()

    def _ensure_indexes(self) -> None:
        """Create metadata indexes for fast filtering."""
        try:
            self.chunks.create_index([("bid_id", 1)])
            self.chunks.create_index([("chunk_id", 1)], unique=True)
            self.chunks.create_index([("doc_type", 1)])
            self.chunks.create_index([("addendum_number", 1)])
            logger.info("MongoDB indexes ensured.")
        except Exception as e:
            logger.warning(f"Index creation warning: {e}")

    def upsert_chunks(self, chunks: List[DocumentChunk]) -> int:
        """Insert or update chunks (incremental indexing)."""
        if not chunks:
            return 0

        ops = []
        for chunk in chunks:
            doc = chunk.model_dump()
            doc["doc_type"] = chunk.doc_type.value
            ops.append(
                UpdateOne(
                    {"chunk_id": chunk.chunk_id},
                    {"$set": doc},
                    upsert=True,
                )
            )

        try:
            result = self.chunks.bulk_write(ops, ordered=False)
            n = result.upserted_count + result.modified_count
            logger.info(f"Upserted {n} chunks to MongoDB.")
            return n
        except BulkWriteError as bwe:
            logger.error(f"Bulk write error: {bwe.details}")
            return 0

    def vector_search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        bid_id: Optional[str] = None,
        doc_types: Optional[List[DocType]] = None,
        addendum_number: Optional[int] = None,
    ) -> List[SearchResult]:
        """
        Atlas Vector Search using $vectorSearch aggregation pipeline.
        """
        # Build pre-filter
        pre_filter: Dict[str, Any] = {}
        if bid_id:
            pre_filter["bid_id"] = {"$eq": bid_id}
        if doc_types:
            pre_filter["doc_type"] = {"$in": [dt.value for dt in doc_types]}
        if addendum_number is not None:
            pre_filter["addendum_number"] = {"$eq": addendum_number}

        pipeline: List[Dict[str, Any]] = [
            {
                "$vectorSearch": {
                    "index": settings.mongodb_vector_index,
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": top_k * 10,
                    "limit": top_k,
                    **({"filter": pre_filter} if pre_filter else {}),
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "chunk_id": 1,
                    "bid_id": 1,
                    "file_name": 1,
                    "doc_type": 1,
                    "addendum_number": 1,
                    "page_number": 1,
                    "section_heading": 1,
                    "text": 1,
                    "is_table": 1,
                    "score": {"$meta": "vectorSearchScore"},
                }
            },
        ]

        try:
            results = list(self.chunks.aggregate(pipeline))
            return [
                SearchResult(
                    chunk_id=r["chunk_id"],
                    bid_id=r["bid_id"],
                    file_name=r["file_name"],
                    doc_type=DocType(r["doc_type"]),
                    page_number=r["page_number"],
                    section_heading=r.get("section_heading"),
                    text=r["text"],
                    score=r.get("score", 0.0),
                    addendum_number=r.get("addendum_number"),
                    is_table=r.get("is_table", False),
                )
                for r in results
            ]
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    def get_chunks_by_bid(self, bid_id: str) -> List[DocumentChunk]:
        """Fetch all chunks for a bid (used for BM25 index building)."""
        docs = self.chunks.find({"bid_id": bid_id}, {"_id": 0, "embedding": 0})
        chunks = []
        for d in docs:
            d["doc_type"] = DocType(d["doc_type"])
            chunks.append(DocumentChunk(**d))
        return chunks

    def get_all_chunks(self) -> List[DocumentChunk]:
        """Fetch all chunks (without embeddings for memory efficiency)."""
        docs = self.chunks.find({}, {"_id": 0, "embedding": 0})
        chunks = []
        for d in docs:
            d["doc_type"] = DocType(d.get("doc_type", "other"))
            chunks.append(DocumentChunk(**d))
        return chunks

    def bid_exists(self, bid_id: str) -> bool:
        """Check if a bid has already been indexed."""
        return self.chunks.count_documents({"bid_id": bid_id}) > 0

    def delete_bid(self, bid_id: str) -> int:
        """Remove all chunks for a bid (for re-indexing)."""
        result = self.chunks.delete_many({"bid_id": bid_id})
        return result.deleted_count

    def close(self):
        self.client.close()
