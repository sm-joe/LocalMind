from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from sqlalchemy import create_engine

from app.db import get_db
from app.auth import require_api_key
from app.main import app
from app.models import Base, Conversation, Message, MessageSource
from app.routes import chat as chat_route


TEST_SESSION_FACTORY = None


@pytest.fixture
def client():
    engine = create_engine(
        "sqlite+pysqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(engine, "connect")
    def enable_sqlite_foreign_keys(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)
    TestingSession = sessionmaker(
        bind=engine,
        autoflush=False,
        expire_on_commit=False,
    )
    global TEST_SESSION_FACTORY
    TEST_SESSION_FACTORY = TestingSession

    def override_get_db():
        session = TestingSession()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = override_get_db
    # Unit tests isolate authorization; separate tests verify auth rejects bad keys.
    app.dependency_overrides[require_api_key] = lambda: None
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
    Base.metadata.drop_all(engine)
    engine.dispose()
    TEST_SESSION_FACTORY = None


def test_create_list_update_and_delete_conversation(client):
    created = client.post(
        "/conversations",
        json={"title": "  Security review  "},
    )
    assert created.status_code == 201
    conversation = created.json()
    assert conversation["title"] == "Security review"

    listed = client.get("/conversations")
    assert listed.status_code == 200
    assert listed.json()["items"][0]["id"] == conversation["id"]
    assert listed.json()["items"][0]["message_count"] == 0

    updated = client.patch(
        f"/conversations/{conversation['id']}",
        json={"title": "Incident response", "is_archived": True},
    )
    assert updated.status_code == 200
    assert updated.json()["title"] == "Incident response"
    assert updated.json()["is_archived"] is True

    assert client.get("/conversations").json()["items"] == []
    assert len(client.get("/conversations?include_archived=true").json()["items"]) == 1

    deleted = client.delete(f"/conversations/{conversation['id']}")
    assert deleted.status_code == 204
    assert client.get(f"/conversations/{conversation['id']}").status_code == 404


def test_conversation_rejects_blank_title(client):
    response = client.post("/conversations", json={"title": "   "})
    assert response.status_code == 422


def test_chat_persists_user_assistant_and_citations_and_replays_idempotently(
    client, monkeypatch
):
    conversation = client.post("/conversations", json={}).json()
    calls = {"count": 0}

    class FakeRAGService:
        def answer(self, question, top_k=None):
            calls["count"] += 1
            return SimpleNamespace(
                answer="The policy requires MFA [1].",
                sources=[
                    SimpleNamespace(
                        document_id="doc-123",
                        filename="security-policy.pdf",
                        chunk_id=4,
                        score=0.91,
                        citation="[1]",
                    )
                ],
                retrieval=SimpleNamespace(
                    requested_limit=5,
                    candidate_count=3,
                    returned_count=1,
                    filtered_count=2,
                    top_candidate_score=0.91,
                    bottom_candidate_score=0.91,
                    top_score=0.91,
                    bottom_score=0.91,
                    has_context=True,
                ),
                performance=SimpleNamespace(
                    embedding_ms=2.0,
                    retrieval_ms=3.0,
                    prompt_build_ms=1.0,
                    llm_ms=8.0,
                    total_ms=14.0,
                ),
            )

    monkeypatch.setattr(
        chat_route,
        "create_rag_service",
        lambda: FakeRAGService(),
    )

    request_body = {
        "question": "What does the policy require?",
        "top_k": 5,
        "conversation_id": conversation["id"],
    }
    first = client.post(
        "/chat",
        json=request_body,
        headers={"Idempotency-Key": "test-request-0001"},
    )
    assert first.status_code == 200
    result = first.json()
    assert result["conversation_id"] == conversation["id"]
    assert result["user_message_id"]
    assert result["assistant_message_id"]
    assert result["sources"][0]["filename"] == "security-policy.pdf"
    assert calls["count"] == 1

    history = client.get(
        f"/conversations/{conversation['id']}/messages"
    )
    assert history.status_code == 200
    messages = history.json()["items"]
    assert [item["role"] for item in messages] == ["user", "assistant"]
    assert messages[0]["content"] == request_body["question"]
    assert messages[1]["status"] == "complete"
    assert messages[1]["sources"][0]["document_id"] == "doc-123"

    replay = client.post(
        "/chat",
        json=request_body,
        headers={"Idempotency-Key": "test-request-0001"},
    )
    assert replay.status_code == 200
    assert replay.json()["assistant_message_id"] == result["assistant_message_id"]
    assert calls["count"] == 1


def test_chat_keeps_legacy_request_compatible(client, monkeypatch):
    class FakeRAGService:
        def answer(self, question, top_k=None):
            return SimpleNamespace(
                answer="No evidence.",
                sources=[],
                retrieval=SimpleNamespace(
                    requested_limit=5,
                    candidate_count=0,
                    returned_count=0,
                    filtered_count=0,
                    top_candidate_score=None,
                    bottom_candidate_score=None,
                    top_score=None,
                    bottom_score=None,
                    has_context=False,
                ),
                performance=SimpleNamespace(
                    embedding_ms=1.0,
                    retrieval_ms=1.0,
                    prompt_build_ms=0.0,
                    llm_ms=0.0,
                    total_ms=2.0,
                ),
            )

    monkeypatch.setattr(
        chat_route,
        "create_rag_service",
        lambda: FakeRAGService(),
    )
    response = client.post("/chat", json={"question": "Question"})
    assert response.status_code == 200
    assert response.json()["conversation_id"] is None


def test_deleting_conversation_cascades_messages_and_sources(client):
    created = client.post("/conversations", json={}).json()
    with TEST_SESSION_FACTORY() as session:
        conversation = session.get(Conversation, created["id"])
        user = Message(
            conversation_id=conversation.id,
            sequence_no=1,
            role="user",
            content="Question",
            status="complete",
        )
        assistant = Message(
            conversation_id=conversation.id,
            sequence_no=2,
            role="assistant",
            content="Answer [1]",
            status="complete",
        )
        session.add_all([user, assistant])
        session.flush()
        session.add(
            MessageSource(
                message_id=assistant.id,
                source_index=1,
                document_id="doc",
                filename="doc.txt",
                chunk_id=1,
                score=0.8,
                citation="[1]",
            )
        )
        session.commit()

    assert client.delete(f"/conversations/{created['id']}").status_code == 204
    assert client.get(f"/conversations/{created['id']}/messages").status_code == 404
