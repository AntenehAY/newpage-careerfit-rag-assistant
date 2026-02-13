"""Text chunking strategies for document ingestion."""

import logging
import re
from typing import Literal

from app.models import ChunkMetadata

logger = logging.getLogger(__name__)

# Section headers for resume detection (case-insensitive)
RESUME_SECTIONS = (
    "summary", "objective", "profile", "experience", "work experience",
    "employment", "education", "skills", "technical skills", "certifications",
    "projects", "achievements", "qualifications", "references",
)
# Section headers for job description detection
JD_SECTIONS = (
    "requirements", "responsibilities", "qualifications", "about the role",
    "about us", "what we offer", "benefits", "job description", "overview",
)


def _detect_section(text: str, doc_type: Literal["resume", "job_description"]) -> str | None:
    """Detect section name from chunk text based on document type."""
    sections = RESUME_SECTIONS if doc_type == "resume" else JD_SECTIONS
    text_lower = text.strip().lower()
    for section in sections:
        if re.search(rf"^{re.escape(section)}\s*[:-]?\s*$|^{re.escape(section)}\s*$", text_lower):
            return section.replace(" ", "_").title()
        if text_lower.startswith(section):
            return section.replace(" ", "_").title()
    return None


def chunk_document(
    text: str,
    metadata: dict,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> list[ChunkMetadata]:
    """Split document text into overlapping chunks with metadata.

    Uses RecursiveCharacterTextSplitter from LangChain. Detects section headers
    for resumes (Skills, Experience, Education) and JDs (Requirements, Responsibilities).
    Tags chunks with section name when detected.

    Args:
        text: Full document text to chunk.
        metadata: Must include 'doc_id' and 'doc_type' (resume|job_description).
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Overlap between consecutive chunks.

    Returns:
        List of ChunkMetadata with chunk_index, char_start, char_end, section.
    """
    doc_id = metadata.get("doc_id", "unknown")
    doc_type = metadata.get("doc_type", "resume")
    if doc_type not in ("resume", "job_description"):
        doc_type = "resume"

    text = text.strip()
    if not text:
        logger.warning("Empty or whitespace-only document; returning no chunks")
        return []

    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        try:
            from langchain.text_splitter import RecursiveCharacterTextSplitter
        except ImportError:
            raise ImportError(
                "RecursiveCharacterTextSplitter is required. "
                "Install with: pip install langchain-text-splitters"
            )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
        strip_whitespace=True,
        add_start_index=True,
    )
    docs = splitter.create_documents([text])

    result: list[ChunkMetadata] = []
    for i, doc in enumerate(docs):
        chunk_text = (doc.page_content or "").strip()
        if not chunk_text:
            continue

        start = doc.metadata.get("start_index", 0)
        if i > 0 and start == 0 and result:
            prev_end = result[-1].char_end
            start = min(prev_end, len(text) - len(chunk_text))
        end = start + len(chunk_text)
        if end > len(text):
            end = len(text)

        section = _detect_section(chunk_text[:100], doc_type)
        chunk_idx = len(result)
        chunk_id = f"{doc_id}_chunk_{chunk_idx}"
        result.append(
            ChunkMetadata(
                chunk_id=chunk_id,
                doc_id=doc_id,
                doc_type=doc_type,
                chunk_index=chunk_idx,
                page_number=None,
                section=section,
                char_start=start,
                char_end=end,
            )
        )

    if result:
        avg_size = sum(c.char_end - c.char_start for c in result) / len(result)
        logger.info(
            "Chunked document %s: %d chunks, avg size %.0f chars",
            doc_id, len(result), avg_size,
        )
    return result
