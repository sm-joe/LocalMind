"""Regression tests for server-side document filtering during RAG summaries."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from qdrant_client.http import models

from app.services.retrieval import RetrievalService
from app.services.vector_store import VectorStore


def _point(document_id: str, chunk_id: int, score: float = 0.85):
    text = f"Section {chunk_id} of {document_id}"
    return SimpleNamespace(
        score=score,
        payload={
            "document_id": document_id,
            "filename": "guide.pdf",
            "content_type": "application/pdf",
            "chunk_id": chunk_id,
            "text": text,
            "start_char": 0,
            "end_char": len(text),
        },
    )


def test_filtered_scroll_paginates_only_selected_document():
    store = VectorStore("qdrant", 6333, "documents")
    store.client = MagicMock()
    first = _point("chosen", 2)
    second = _point("chosen", 1)
    store.client.scroll.side_effect = [
        ([first], "cursor-1"),
        ([second], None),
    ]

    assert list(store.iter_points(limit=1, document_id="chosen")) == [first, second]
    assert store.client.scroll.call_count == 2
    first_call, second_call = store.client.scroll.call_args_list
    for call in (first_call, second_call):
        query_filter = call.kwargs["scroll_filter"]
        assert isinstance(query_filter, models.Filter)
        assert query_filter.must[0].key == "document_id"
        assert query_filter.must[0].match.value == "chosen"
        assert call.kwargs["with_vectors"] is False
    assert "offset" not in first_call.kwargs
    assert second_call.kwargs["offset"] == "cursor-1"


def test_filtered_scroll_rejects_blank_document_id():
    store = VectorStore("qdrant", 6333, "documents")
    store.client = MagicMock()
    with pytest.raises(ValueError, match="Document ID cannot be empty"):
        list(store.iter_points(document_id="  "))
    store.client.scroll.assert_not_called()


def test_document_summary_filters_by_id_and_orders_chunks():
    embeddings = MagicMock()
    embeddings.embed.return_value = [0.1, 0.2, 0.3]
    store = MagicMock()
    store.search.return_value = [
        _point("chosen", 2, 0.9),
        _point("another", 1, 0.6),
    ]
    store.iter_points.return_value = [
        _point("chosen", 3, 0.0),
        _point("chosen", 1, 0.0),
        _point("chosen", 2, 0.0),
    ]
    retrieval = RetrievalService(embeddings, store)
    response = retrieval.retrieve_document("Summarize guide.pdf")

    store.iter_points.assert_called_once_with(document_id="chosen")
    assert [chunk.chunk_id for chunk in response.chunks] == [1, 2, 3]
    assert all(chunk.document_id == "chosen" for chunk in response.chunks)
    assert response.diagnostics.returned_count == 3
