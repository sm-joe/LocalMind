import time
import uuid

import pytest
from fastapi.testclient import TestClient
from ollama import Client as OllamaClient
from qdrant_client import QdrantClient

from app.config import get_settings
from app.main import app
from app.routes import documents as documents_route


def wait_for_services(
    ollama: OllamaClient,
    qdrant: QdrantClient,
    timeout_seconds: int = 30,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None

    while time.monotonic() < deadline:
        try:
            ollama.list()
            qdrant.get_collections()
            return
        except Exception as exc:
            last_error = exc
            time.sleep(1)

    pytest.fail(
        "Ollama and Qdrant did not become ready: "
        f"{type(last_error).__name__ if last_error else 'unknown error'}"
    )


def test_document_lifecycle_upload_duplicate_list_delete(
    monkeypatch,
) -> None:
    settings = get_settings()

    # Isolate this test from the user's actual document collection.
    collection_name = (
        f"localrag_lifecycle_{uuid.uuid4().hex}"
    )
    test_settings = settings.model_copy(
        update={
            "qdrant_collection": collection_name,
        }
    )

    monkeypatch.setattr(
        documents_route,
        "get_settings",
        lambda: test_settings,
    )

    qdrant = QdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
    )
    ollama = OllamaClient(
        host=settings.ollama_base_url,
    )

    wait_for_services(
        ollama=ollama,
        qdrant=qdrant,
    )

    # A unique payload also isolates the PostgreSQL hash registry across runs.
    # Collection isolation alone is insufficient now that deduplication is durable.
    document_content = (
        f"Run {uuid.uuid4().hex}. "
        "LocalRAG document lifecycle integration test. "
        "This document verifies real text ingestion, embedding "
        "generation, Qdrant persistence, duplicate detection, "
        "document listing, and deletion. "
        "The same bytes uploaded under a different filename "
        "must not create a second indexed document."
    ).encode("utf-8")

    try:
        with TestClient(app) as client:
            # Compose configures a private API key; every business endpoint
            # must be exercised through its real authentication boundary.
            if settings.localrag_api_key:
                client.headers.update({"X-API-Key": settings.localrag_api_key})
            # 1. Upload the original document.
            upload_response = client.post(
                "/documents/upload",
                files={
                    "file": (
                        "lifecycle-test.txt",
                        document_content,
                        "text/plain",
                    )
                },
            )

            assert upload_response.status_code == 200, (
                f"Initial upload failed: {upload_response.text}"
            )

            uploaded = upload_response.json()

            assert uploaded["status"] == "indexed"
            assert uploaded["filename"] == "lifecycle-test.txt"
            assert uploaded["document_id"]
            assert uploaded["chunks"] > 0

            document_id = uploaded["document_id"]
            chunk_count = uploaded["chunks"]

            # 2. Upload the exact same bytes with a different filename.
            duplicate_response = client.post(
                "/documents/upload",
                files={
                    "file": (
                        "lifecycle-test-copy.txt",
                        document_content,
                        "text/plain",
                    )
                },
            )

            assert duplicate_response.status_code == 200, (
                f"Duplicate upload failed: "
                f"{duplicate_response.text}"
            )

            duplicate = duplicate_response.json()

            assert duplicate["status"] == "already_indexed"
            assert duplicate["document_id"] == document_id
            assert duplicate["filename"] == "lifecycle-test.txt"
            assert duplicate["chunks"] == 0

            # 3. Verify that listing contains one indexed document.
            listing_response = client.get("/documents")

            assert listing_response.status_code == 200, (
                f"Document listing failed: {listing_response.text}"
            )

            listed_documents = listing_response.json()["documents"]

            matching_documents = [
                document
                for document in listed_documents
                if document["document_id"] == document_id
            ]

            assert len(matching_documents) == 1
            assert matching_documents[0]["filename"] == (
                "lifecycle-test.txt"
            )
            assert matching_documents[0]["content_type"] == (
                "text/plain"
            )
            assert matching_documents[0]["chunks"] == chunk_count

            # 4. Delete the document using the real API endpoint.
            delete_response = client.delete(
                f"/documents/{document_id}",
            )

            assert delete_response.status_code == 200, (
                f"Document deletion failed: {delete_response.text}"
            )

            assert delete_response.json() == {
                "document_id": document_id,
                "status": "deleted",
            }

            # 5. Verify eventual disappearance from the listing.
            # Qdrant deletion may complete asynchronously.
            deadline = time.monotonic() + 15

            while time.monotonic() < deadline:
                listing_response = client.get("/documents")

                assert listing_response.status_code == 200, (
                    "Document listing failed after deletion: "
                    f"{listing_response.text}"
                )

                listed_documents = listing_response.json()["documents"]

                if not any(
                    document["document_id"] == document_id
                    for document in listed_documents
                ):
                    break

                time.sleep(0.25)
            else:
                pytest.fail(
                    "Deleted document remained in the document listing."
                )

    finally:
        # Clean up only this test's collection, including on failure.
        collections = qdrant.get_collections()

        if any(
            collection.name == collection_name
            for collection in collections.collections
        ):
            qdrant.delete_collection(
                collection_name=collection_name,
            )

        qdrant.close()