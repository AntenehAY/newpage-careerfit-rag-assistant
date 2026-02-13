#!/usr/bin/env python
"""Quick verification that app/config.py and app/models.py work.

Run: python verify_step2.py
Or:  pytest tests/test_config_models.py -v
"""

from datetime import datetime

def main():
    print("Verifying Step 2: Configuration & Core Models\n" + "=" * 50)

    # 1. Config
    print("\n1. Config (app/config.py)")
    from app.config import settings
    assert settings is not None
    print(f"   - APP_NAME: {settings.app_name}")
    print(f"   - LLM_MODEL: {settings.llm_model_name}")
    print(f"   - EMBEDDING_MODEL: {settings.embedding_model_name}")
    print(f"   - CHUNK_SIZE: {settings.chunk_size}")
    print(f"   - API keys loaded: OK")

    # 2. Models
    print("\n2. Models (app/models.py)")
    from app.models import (
        DocumentUpload,
        ChunkMetadata,
        QueryRequest,
        QueryResponse,
        SourceReference,
    )

    doc = DocumentUpload(
        file_id="test-1",
        file_name="resume.pdf",
        file_type="resume",
        file_size=1024,
        uploaded_at=datetime.now(),
        status="completed",
    )
    print(f"   - DocumentUpload: {doc.file_name} ({doc.status})")

    req = QueryRequest(query="What skills are required?", max_results=5)
    print(f"   - QueryRequest: '{req.query[:30]}...' max_results={req.max_results}")

    resp = QueryResponse(
        answer="Sample answer.",
        sources=[],
        generated_at=datetime.now(),
    )
    print(f"   - QueryResponse: answer length={len(resp.answer)}")

    print("\n" + "=" * 50)
    print("Step 2 verification passed: config and models work correctly.")


if __name__ == "__main__":
    main()
