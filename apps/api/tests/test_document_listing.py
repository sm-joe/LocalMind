from types import SimpleNamespace
from unittest.mock import MagicMock

from app.routes import documents


def test_list_documents_aggregates_chunks_by_document(
    monkeypatch,
):
    vector_store = MagicMock()
    vector_store.client.get_collections.return_value.collections = [
        SimpleNamespace(name="localrag_documents"),
    ]

    vector_store.iter_points.return_value = iter(
        [
            SimpleNamespace(
                payload={
                    "document_id": "doc-1",
                    "filename": "report.pdf",
                    "content_type": "application/pdf",
                },
            ),
            SimpleNamespace(
                payload={
                    "document_id": "doc-1",
                    "filename": "report.pdf",
                    "content_type": "application/pdf",
                },
            ),
            SimpleNamespace(
                payload={
                    "document_id": "doc-2",
                    "filename": "notes.txt",
                    "content_type": "text/plain",
                },
            ),
        ]
    )

    monkeypatch.setattr(
        documents,
        "create_vector_store",
        lambda: vector_store,
    )
    monkeypatch.setattr(
        documents,
        "get_settings",
        lambda: SimpleNamespace(
            qdrant_collection="localrag_documents",
        ),
    )

    response = documents.list_documents()

    assert [
        item.model_dump()
        for item in response.documents
    ] == [
        {
            "document_id": "doc-1",
            "filename": "report.pdf",
            "content_type": "application/pdf",
            "chunks": 2,
        },
        {
            "document_id": "doc-2",
            "filename": "notes.txt",
            "content_type": "text/plain",
            "chunks": 1,
        },
    ]

    vector_store.iter_points.assert_called_once_with(
        limit=1000,
    )
    vector_store.list_points.assert_not_called()


def test_list_documents_returns_empty_when_collection_missing(
    monkeypatch,
):
    vector_store = MagicMock()
    vector_store.client.get_collections.return_value.collections = []

    monkeypatch.setattr(
        documents,
        "create_vector_store",
        lambda: vector_store,
    )
    monkeypatch.setattr(
        documents,
        "get_settings",
        lambda: SimpleNamespace(
            qdrant_collection="localrag_documents",
        ),
    )

    response = documents.list_documents()

    assert response.documents == []
    vector_store.iter_points.assert_not_called()