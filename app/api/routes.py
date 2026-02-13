"""FastAPI endpoints for Career Intelligence Assistant."""

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from app.api.dependencies import get_rag_chain, get_vector_store
from app.api.document_registry import DocumentRegistry
from app.ingestion.pipeline import ingest_document
from app.models import DocumentUpload, QueryRequest, QueryResponse
from app.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["career-intelligence"])

# Allowed file extensions for upload
ALLOWED_EXTENSIONS = {".pdf", ".docx"}
UPLOADS_DIR = Path("./data/uploads")
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

# Singleton document registry
_registry: DocumentRegistry | None = None


def get_registry() -> DocumentRegistry:
    """Get or create document registry instance. Used with Depends."""
    global _registry
    if _registry is None:
        _registry = DocumentRegistry()
    return _registry


@router.post("/upload", response_model=DocumentUpload)
async def upload_document(
    file: UploadFile = File(...),
    doc_type: str = Form(..., description="resume or job_description"),
    vector_store: VectorStore = Depends(get_vector_store),
    registry: DocumentRegistry = Depends(get_registry),
) -> DocumentUpload:
    """Accept file upload, validate type, save to data/uploads/, run ingestion, store in vector DB.

    Accepts .pdf and .docx files. Returns DocumentUpload with file_id and status.
    """
    if doc_type not in ("resume", "job_description"):
        logger.warning("Invalid doc_type: %s", doc_type)
        raise HTTPException(status_code=400, detail="doc_type must be 'resume' or 'job_description'")

    # Validate file extension
    original_filename = file.filename or "unknown"
    ext = Path(original_filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        logger.warning("Rejected file with unsupported extension: %s", ext)
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: .pdf, .docx. Got: {ext}",
        )

    file_id = str(uuid.uuid4())
    save_path = UPLOADS_DIR / f"{file_id}{ext}"
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

    content = b""
    try:
        content = await file.read()
        file_size = len(content)
        if file_size > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail="File too large. Maximum size: 10 MB")
        if file_size == 0:
            raise HTTPException(status_code=400, detail="File is empty")

        save_path.write_bytes(content)
        logger.info("Saved upload: %s (%d bytes)", save_path.name, file_size)

        # Register as processing
        doc_meta = DocumentUpload(
            file_id=file_id,
            file_name=original_filename,
            file_type=doc_type,
            file_size=file_size,
            uploaded_at=datetime.now(timezone.utc),
            status="processing",
        )
        registry.add(doc_meta)

        # Run ingestion pipeline with shared vector store
        chunks = ingest_document(
            file_path=str(save_path),
            doc_type=doc_type,
            doc_id=file_id,
            store_in_vector_db=True,
            vector_store=vector_store,
        )

        status = "completed" if chunks else "failed"
        doc_meta = DocumentUpload(
            file_id=file_id,
            file_name=original_filename,
            file_type=doc_type,
            file_size=file_size,
            uploaded_at=doc_meta.uploaded_at,
            status=status,
        )
        registry.add(doc_meta)

        logger.info("Upload complete: file_id=%s, status=%s, chunks=%d", file_id, status, len(chunks))
        return doc_meta

    except HTTPException:
        if save_path.exists():
            save_path.unlink(missing_ok=True)
        raise
    except Exception as e:
        logger.exception("Upload failed: %s", e)
        if save_path.exists():
            save_path.unlink(missing_ok=True)
        # Update registry with failed status (we added processing earlier)
        doc_meta = DocumentUpload(
            file_id=file_id,
            file_name=original_filename,
            file_type=doc_type,
            file_size=len(content),
            uploaded_at=datetime.now(timezone.utc),
            status="failed",
        )
        registry.add(doc_meta)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    request_obj: Request,
    rag_chain=Depends(get_rag_chain),
) -> QueryResponse:
    """Process query with guardrails, call RAG chain, return answer and sources."""
    user_id = request_obj.client.host if request_obj.client else "default"
    try:
        response = rag_chain.answer_query(
            query=request.query,
            filter_doc_type=request.filter_doc_type,
            filter_doc_id=request.filter_doc_id,
            user_id=user_id,
        )
        return response
    except RuntimeError as e:
        if "rate" in str(e).lower() or "busy" in str(e).lower():
            raise HTTPException(status_code=429, detail=str(e)) from e
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/documents", response_model=list[DocumentUpload])
async def list_documents(
    registry: DocumentRegistry = Depends(get_registry),
) -> list[DocumentUpload]:
    """List all uploaded documents."""
    return registry.list_all()


@router.delete("/documents/{doc_id}")
async def delete_document(
    doc_id: str,
    vector_store: VectorStore = Depends(get_vector_store),
    registry: DocumentRegistry = Depends(get_registry),
) -> dict:
    """Delete document from vector DB and remove file from uploads/."""
    doc = registry.get(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")

    # Delete from vector DB
    try:
        vector_store.delete_by_doc_id(doc_id)
    except Exception as e:
        logger.warning("Vector store delete failed for %s: %s", doc_id, e)
        # Continue to remove file and registry entry

    # Remove file from uploads
    for ext in ALLOWED_EXTENSIONS:
        upload_path = UPLOADS_DIR / f"{doc_id}{ext}"
        if upload_path.exists():
            upload_path.unlink()
            break

    registry.delete(doc_id)
    return {"success": True, "message": f"Document {doc_id} deleted"}


@router.get("/health")
async def health_check(
    vector_store: VectorStore = Depends(get_vector_store),
) -> dict:
    """Health check with component status."""
    try:
        stats = vector_store.get_collection_stats()
        return {
            "status": "ok",
            "components": {
                "vector_db": "ok" if stats.get("count", 0) >= 0 else "degraded",
                "collection": stats.get("collection_name", "careerfit_chunks"),
            },
        }
    except Exception as e:
        logger.warning("Health check failed: %s", e)
        return {
            "status": "degraded",
            "components": {"vector_db": "error", "error": str(e)},
        }


@router.get("/stats")
async def get_stats(
    vector_store: VectorStore = Depends(get_vector_store),
    registry: DocumentRegistry = Depends(get_registry),
) -> dict:
    """Return system stats: total documents, total chunks, vector DB info."""
    stats = vector_store.get_collection_stats()
    docs = registry.list_all()
    return {
        "total_documents": len(docs),
        "total_chunks": stats.get("count", 0),
        "vector_db": {
            "collection_name": stats.get("collection_name", "careerfit_chunks"),
            "persist_directory": stats.get("persist_directory", ""),
        },
        "documents_by_type": {
            "resume": sum(1 for d in docs if d.file_type == "resume"),
            "job_description": sum(1 for d in docs if d.file_type == "job_description"),
        },
    }
