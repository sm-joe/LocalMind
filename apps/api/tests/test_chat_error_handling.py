from fastapi.testclient import TestClient

from app.main import app
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

    response = client.post(
        "/chat",
        json={
            "question": "Summarize my documents",
            "top_k": 5,
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid chat request."
    assert "/internal/private-data" not in response.text