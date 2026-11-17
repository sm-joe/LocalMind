from fastapi.testclient import TestClient

from app.main import app
from app.auth import require_api_key
from app.routes import chat


client = TestClient(app)


def test_chat_does_not_expose_value_error_details(monkeypatch):
    class FailingRAGService:
        def answer(self, question, top_k=None):
            raise ValueError(
                "Sensitive internal path: /internal/private-data"
            )

    monkeypatch.setattr(
        chat,
        "create_rag_service",
        lambda: FailingRAGService(),
    )

    # Test error sanitization independently of authentication.
    app.dependency_overrides[require_api_key] = lambda: None
    try:
        response = client.post(
        "/chat",
        json={
            "question": "Summarize my documents",
            "top_k": 5,
        },
    )

    finally:
        app.dependency_overrides.pop(require_api_key, None)

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid chat request."
    assert "/internal/private-data" not in response.text