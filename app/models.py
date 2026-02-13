"""Pydantic v2 models for documents, chunks, queries, and responses."""

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


class DocumentUpload(BaseModel):
    """Metadata for an uploaded resume or job description document."""

    file_id: str = Field(..., description="Unique identifier for the file (e.g., UUID)")
    file_name: str = Field(..., description="Original filename as uploaded")
    file_type: Literal["resume", "job_description"] = Field(
        ...,
        description="Type of document. Example: resume",
    )
    file_size: int = Field(..., ge=0, description="File size in bytes")
    uploaded_at: datetime = Field(
        ...,
        description="Timestamp when the file was uploaded",
    )
    status: Literal["pending", "processing", "completed", "failed"] = Field(
        ...,
        description="Processing status. Example: completed",
    )


class ChunkMetadata(BaseModel):
    """Metadata for a text chunk derived from a document."""

    chunk_id: str = Field(..., description="Unique identifier for the chunk")
    doc_id: str = Field(..., description="ID of the source document")
    doc_type: Literal["resume", "job_description"] = Field(
        ...,
        description="Type of the source document",
    )
    chunk_index: int = Field(..., ge=0, description="Zero-based index of chunk within document")
    page_number: Optional[int] = Field(
        default=None,
        ge=1,
        description="Page number in source document, if applicable. Example: 1",
    )
    section: Optional[str] = Field(
        default=None,
        description="Section or heading the chunk belongs to. Example: Experience",
    )
    char_start: int = Field(..., ge=0, description="Character offset where chunk starts")
    char_end: int = Field(..., ge=0, description="Character offset where chunk ends")


class QueryRequest(BaseModel):
    """Incoming query request for RAG retrieval and generation."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="User question. Example: What skills does this resume highlight?",
    )
    filter_doc_type: Optional[Literal["resume", "job_description"]] = Field(
        default=None,
        description="Optional filter to restrict results to one document type",
    )
    filter_doc_id: Optional[str] = Field(
        default=None,
        description="Optional filter to restrict results to a specific document ID",
    )
    max_results: Optional[int] = Field(
        default=5,
        ge=1,
        le=50,
        description="Maximum number of source chunks to retrieve. Example: 5",
    )


class SourceReference(BaseModel):
    """A single source chunk cited in a generated answer."""

    doc_id: str = Field(..., description="ID of the source document")
    doc_name: str = Field(..., description="Display name of the source document")
    doc_type: str = Field(..., description="Type of document (resume or job_description)")
    chunk_text: str = Field(..., description="The text excerpt used as source")
    relevance_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Similarity/relevance score of this chunk to the query",
    )


class QueryResponse(BaseModel):
    """Generated answer with source citations."""

    answer: str = Field(..., description="The generated answer to the user's query")
    sources: list[SourceReference] = Field(
        default_factory=list,
        description="List of source chunks referenced in the answer",
    )
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional confidence score of the answer",
    )
    generated_at: datetime = Field(
        ...,
        description="Timestamp when the answer was generated",
    )
