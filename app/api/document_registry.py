"""Persistent document metadata registry for uploaded files."""

import json
import logging
from pathlib import Path
from typing import Optional

from app.models import DocumentUpload

logger = logging.getLogger(__name__)


def _default_registry_path() -> Path:
    """Default path for document registry JSON file."""
    return Path("./data/document_registry.json")


class DocumentRegistry:
    """Persistent registry of uploaded document metadata."""

    def __init__(self, registry_path: Optional[Path] = None) -> None:
        self._path = Path(registry_path) if registry_path else _default_registry_path()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._docs: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        """Load registry from disk."""
        if self._path.exists():
            try:
                with open(self._path, encoding="utf-8") as f:
                    data = json.load(f)
                    self._docs = data.get("documents", {})
            except Exception as e:
                logger.warning("Failed to load document registry: %s", e)
                self._docs = {}

    def _save(self) -> None:
        """Persist registry to disk."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump({"documents": self._docs}, f, indent=2)
        except Exception as e:
            logger.error("Failed to save document registry: %s", e)
            raise

    def add(self, doc: DocumentUpload) -> None:
        """Add or update a document in the registry."""
        self._docs[doc.file_id] = doc.model_dump(mode="json")
        self._save()

    def get(self, file_id: str) -> Optional[DocumentUpload]:
        """Get document by file_id."""
        data = self._docs.get(file_id)
        if data is None:
            return None
        return DocumentUpload.model_validate(data)

    def list_all(self) -> list[DocumentUpload]:
        """List all documents in the registry."""
        result = []
        for data in self._docs.values():
            try:
                result.append(DocumentUpload.model_validate(data))
            except Exception as e:
                logger.warning("Skipping invalid registry entry: %s", e)
        return sorted(result, key=lambda d: d.uploaded_at, reverse=True)

    def delete(self, file_id: str) -> bool:
        """Remove document from registry. Returns True if it existed."""
        if file_id in self._docs:
            del self._docs[file_id]
            self._save()
            return True
        return False
