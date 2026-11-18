"""Forced document reindex does not erase a working index on failure."""
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.services.document_registry import IndexClaim
from app.services.ingestion import IngestionService


def _fixture(tmp_path: Path):
    path = tmp_path / "Interview_Questions.txt"
    path.write_text(
        "Amazon Bedrock is a managed generative AI service. "
        "AWS IAM controls permissions, and EC2 runs compute instances.",
        encoding="utf-8",
    )
    registry = MagicMock()
    registry.acquire.return_value = IndexClaim(
        document_hash="old-hash",
        document_id="new-document",
        status="claimed",
        filename=path.name,
        previous_document_id="original-document",
        previous_chunks=3,
        restore_on_failure=True,
    )
    vectors = MagicMock()
    vectors.find_document_by_hash.return_value = {
        "document_id": "original-document", "filename": path.name,
    }
    embedding = MagicMock()
    embedding.embed_many.side_effect = lambda texts: [
        [0.1, 0.2, 0.3] for _ in texts
    ]
    return path, registry, vectors, IngestionService(
        embedding, vectors, registry=registry
    )


def test_force_reindex_upserts_first_and_retires_old_index_last(tmp_path):
    path, registry, vectors, service = _fixture(tmp_path)
    events = []
    vectors.upsert.side_effect = lambda points: events.append("new vectors written")
    registry.complete.side_effect = lambda *args: events.append("registry ready")
    vectors.delete_document.side_effect = lambda doc_id: events.append(f"delete {doc_id}")

    result = service.ingest(
        path, "text/plain", force_reindex=True
    )
    assert not result.already_indexed
    assert result.document_id == "new-document"
    assert result.chunks > 0
    registry.acquire.assert_called_once_with(
        service._calculate_hash(path), path.name, "text/plain", force=True
    )
    assert events == [
        "new vectors written", "registry ready", "delete original-document",
    ]


def test_force_reindex_failed_embedding_keeps_old_vectors(tmp_path):
    path, registry, vectors, service = _fixture(tmp_path)
    service.embedding_service.embed_many.side_effect = RuntimeError("embed offline")

    with pytest.raises(RuntimeError, match="embed offline"):
        service.ingest(path, "text/plain", force_reindex=True)
    vectors.delete_document.assert_not_called()
    registry.fail.assert_called_once()
    registry.complete.assert_not_called()
