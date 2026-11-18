"""Opt-in real-service acceptance for the complete document -> cited answer path.

Run explicitly inside the Compose API container:
  RUN_LIVE_RAG_ACCEPTANCE=1 pytest -q tests/test_live_rag_acceptance.py

Requires PostgreSQL, Ollama embeddings + LLM, and Qdrant. This is not a
mock test; failing this test means LocalRAG is not yet ready for users.
"""
import os
import uuid

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from app.config import get_settings
from app.main import app
from app.routes import chat as chat_route
from app.routes import documents as documents_route

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_LIVE_RAG_ACCEPTANCE") != "1",
    reason="Run explicitly with RUN_LIVE_RAG_ACCEPTANCE=1 against running services.",
)


def test_real_upload_answer_and_citation(monkeypatch):
    settings = get_settings()
    isolated_collection = "localrag_acceptance_" + uuid.uuid4().hex
    test_settings = settings.model_copy(
        update={"qdrant_collection": isolated_collection}
    )
    monkeypatch.setattr(documents_route, "get_settings", lambda: test_settings)
    monkeypatch.setattr(chat_route, "get_settings", lambda: test_settings)

    qdrant = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    # PostgreSQL deduplicates by content hash across *all* Qdrant
    # collections. A unique filename/collection alone is not sufficient:
    # give every test run unique bytes so it cannot return already_indexed.
    run_id = uuid.uuid4().hex
    filename = f"localrag-acceptance-{run_id}.txt"
    content = (
        "LocalRAG acceptance project: Planet Cedar uses SaturnDB as its "
        "primary document metadata database. The service processes rainfall "
        "measurements every evening and archives reports on Fridays. "
        "SaturnDB stores the Cedar project metadata, not the model weights. "
        f"Unique acceptance run identifier: {run_id}."
    )
    document_id = None
    try:
        with TestClient(app) as client:
            if settings.localrag_api_key:
                client.headers.update({"X-API-Key": settings.localrag_api_key})
            upload = client.post(
                "/documents/upload",
                files={"file": (filename, content.encode("utf-8"), "text/plain")},
            )
            assert upload.status_code == 200, upload.text
            document_id = upload.json()["document_id"]
            assert upload.json()["status"] == "indexed"
            assert upload.json()["chunks"] >= 1

            answer_response = client.post(
                "/chat",
                json={
                    "question": "Which database stores the Planet Cedar project metadata?",
                    "top_k": 5,
                },
            )
            assert answer_response.status_code == 200, answer_response.text
            answer = answer_response.json()
            assert answer["retrieval"]["has_context"] is True
            assert answer["sources"], answer
            assert "SaturnDB" in answer["answer"], answer
            assert "[1]" in answer["answer"], answer
            assert "Relevant verbatim excerpts:" not in answer["answer"], (
                "The model could not generate a cited answer; it fell back to raw excerpts."
            )

            first = answer["sources"][0]
            assert first["document_id"] == document_id
            preview = client.get(
                f"/documents/{first['document_id']}/chunks/{first['chunk_id']}"
            )
            assert preview.status_code == 200, preview.text
            assert "SaturnDB" in preview.json()["text"]

            # A summary must use this document's indexed facts, not hallucinate.
            summary = client.post(
                "/chat",
                json={"question": f"Summarize {filename}", "top_k": 5},
            )
            assert summary.status_code == 200, summary.text
            data = summary.json()
            assert data["sources"], data
            assert data["retrieval"]["has_context"] is True
            assert "SaturnDB" in data["answer"], data
    finally:
        try:
            if any(
                item.name == isolated_collection
                for item in qdrant.get_collections().collections
            ):
                qdrant.delete_collection(collection_name=isolated_collection)
        finally:
            try:
                # Remove only the fixture's PostgreSQL registry row, never
                # any user document. Collection deletion alone leaves the
                # global hash registry claiming this file is still indexed.
                if document_id is not None:
                    documents_route.create_document_registry().forget(document_id)
            finally:
                qdrant.close()
