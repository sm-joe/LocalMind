from types import SimpleNamespace
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
from qdrant_client.http import models

import app.routes.documents as documents
from app.main import app
from app.auth import require_api_key
from app.services.vector_store import VectorStore


def test_document_scoped_chunk_lookup_uses_both_filters():
    store = VectorStore.__new__(VectorStore)
    store.collection_name = "documents"
    store.client = MagicMock()
    store.client.scroll.return_value = ([SimpleNamespace(payload={
        "document_id": "doc-a", "chunk_id": 7, "text": "A source passage."
    })], None)
    payload = store.get_document_chunk("doc-a", 7)
    assert payload["text"] == "A source passage."
    kwargs = store.client.scroll.call_args.kwargs
    assert kwargs["limit"] == 1
    assert kwargs["with_vectors"] is False
    conditions = kwargs["scroll_filter"].must
    assert {(item.key, item.match.value) for item in conditions} == {
        ("document_id", "doc-a"), ("chunk_id", 7)
    }


def test_preview_returns_exact_document_chunk(monkeypatch):
    store = MagicMock()
    store.get_document_chunk.return_value = {
        "document_id": "abc", "chunk_id": 3, "filename": "guide.md",
        "text": "Only this cited passage.", "start_char": 42, "end_char": 66,
    }
    monkeypatch.setattr(documents, "create_vector_store", lambda: store)
    app.dependency_overrides[require_api_key] = lambda: None
    try:
        response = TestClient(app).get("/documents/abc/chunks/3")
    finally:
        app.dependency_overrides.pop(require_api_key, None)
    assert response.status_code == 200
    assert response.json()["text"] == "Only this cited passage."
    store.get_document_chunk.assert_called_once_with("abc", 3)


def test_preview_rejects_mismatched_or_missing_passage(monkeypatch):
    store = MagicMock()
    store.get_document_chunk.return_value = {
        "document_id": "other", "chunk_id": 3, "text": "private"
    }
    monkeypatch.setattr(documents, "create_vector_store", lambda: store)
    app.dependency_overrides[require_api_key] = lambda: None
    try:
        response = TestClient(app).get("/documents/abc/chunks/3")
    finally:
        app.dependency_overrides.pop(require_api_key, None)
    assert response.status_code == 404
