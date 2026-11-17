from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.services.document_registry import DocumentIndexInProgress, IndexClaim
from app.services.ingestion import IngestionService


def _file(tmp_path: Path) -> Path:
    file = tmp_path / "guide.txt"
    file.write_text("Document status should survive service restarts.", encoding="utf-8")
    return file


def test_new_document_claim_is_completed(tmp_path):
    registry = MagicMock()
    registry.acquire.return_value = IndexClaim(
        "abc", "new-doc", "claimed", "guide.txt"
    )
    store = MagicMock()
    store.find_document_by_hash.return_value = None
    embedding = MagicMock()
    embedding.embed_many.return_value = [[0.2, 0.3]]
    service = IngestionService(embedding, store, registry=registry)
    result = service.ingest(_file(tmp_path), "text/plain")
    assert result.document_id == "new-doc"
    assert result.chunks == 1
    registry.complete.assert_called_once_with(
        registry.acquire.return_value, "new-doc", "guide.txt", 1
    )
    registry.fail.assert_not_called()


def test_ready_document_claim_returns_duplicate_without_reembedding(tmp_path):
    registry = MagicMock()
    registry.acquire.return_value = IndexClaim(
        "abc", "existing", "ready", "original.txt"
    )
    store = MagicMock()
    embedding = MagicMock()
    result = IngestionService(embedding, store, registry=registry).ingest(
        _file(tmp_path), "text/plain"
    )
    assert result.already_indexed is True
    assert result.document_id == "existing"
    embedding.embed_many.assert_not_called()
    store.upsert.assert_not_called()


def test_concurrent_claim_refuses_second_upload(tmp_path):
    registry = MagicMock()
    registry.acquire.side_effect = DocumentIndexInProgress("Already running")
    store = MagicMock()
    with pytest.raises(DocumentIndexInProgress):
        IngestionService(MagicMock(), store, registry=registry).ingest(
            _file(tmp_path), "text/plain"
        )
    store.upsert.assert_not_called()


def test_embedding_failure_marks_registry_failed(tmp_path):
    registry = MagicMock()
    claim = IndexClaim("abc", "new-doc", "claimed", "guide.txt")
    registry.acquire.return_value = claim
    store = MagicMock()
    store.find_document_by_hash.return_value = None
    embedding = MagicMock()
    embedding.embed_many.side_effect = RuntimeError("embedding unavailable")
    with pytest.raises(RuntimeError, match="embedding unavailable"):
        IngestionService(embedding, store, registry=registry).ingest(
            _file(tmp_path), "text/plain"
        )
    registry.fail.assert_called_once_with(claim)
