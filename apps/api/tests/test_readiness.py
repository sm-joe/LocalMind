from types import SimpleNamespace

from fastapi.testclient import TestClient

from app import main


class FakeQdrantClient:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def get_collections(self):
        return SimpleNamespace(collections=[])


class FakeOllamaClient:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def list(self):
        return SimpleNamespace(
            models=[
                SimpleNamespace(model="llama3.2:3b"),
                SimpleNamespace(model="nomic-embed-text:latest"),
            ]
        )


def configure_settings(monkeypatch):
    monkeypatch.setattr(
        main,
        "get_settings",
        lambda: SimpleNamespace(
            qdrant_host="qdrant",
            qdrant_port=6333,
            ollama_base_url="http://ollama:11434",
            llm_model="llama3.2:3b",
            embedding_model="nomic-embed-text",
        ),
    )


def configure_available_dependencies(monkeypatch):
    configure_settings(monkeypatch)
    monkeypatch.setattr(
        main,
        "QdrantClient",
        FakeQdrantClient,
    )
    monkeypatch.setattr(
        main,
        "OllamaClient",
        FakeOllamaClient,
    )


def test_health_does_not_depend_on_backends(monkeypatch):
    def unexpected_call(**kwargs):
        raise AssertionError(
            "Liveness must not contact dependencies."
        )

    monkeypatch.setattr(
        main,
        "QdrantClient",
        unexpected_call,
    )
    monkeypatch.setattr(
        main,
        "OllamaClient",
        unexpected_call,
    )

    response = TestClient(main.app).get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_when_dependencies_and_models_are_available(
    monkeypatch,
):
    configure_available_dependencies(monkeypatch)

    response = TestClient(main.app).get("/ready")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "dependencies": {
            "qdrant": "ok",
            "ollama": "ok",
            "llm_model": "ok",
            "embedding_model": "ok",
        },
    }


def test_ready_returns_503_when_qdrant_is_unavailable(
    monkeypatch,
):
    configure_settings(monkeypatch)

    class UnavailableQdrantClient:
        def __init__(self, **kwargs):
            pass

        def get_collections(self):
            raise ConnectionError("Qdrant unavailable")

    monkeypatch.setattr(
        main,
        "QdrantClient",
        UnavailableQdrantClient,
    )
    monkeypatch.setattr(
        main,
        "OllamaClient",
        FakeOllamaClient,
    )

    response = TestClient(main.app).get("/ready")

    assert response.status_code == 503
    assert response.json()["detail"] == {
        "status": "not_ready",
        "dependencies": {
            "qdrant": "unavailable",
            "ollama": "ok",
            "llm_model": "ok",
            "embedding_model": "ok",
        },
    }


def test_ready_returns_503_when_ollama_is_unavailable(
    monkeypatch,
):
    configure_settings(monkeypatch)

    class UnavailableOllamaClient:
        def __init__(self, **kwargs):
            pass

        def list(self):
            raise ConnectionError("Ollama unavailable")

    monkeypatch.setattr(
        main,
        "QdrantClient",
        FakeQdrantClient,
    )
    monkeypatch.setattr(
        main,
        "OllamaClient",
        UnavailableOllamaClient,
    )

    response = TestClient(main.app).get("/ready")

    assert response.status_code == 503
    assert response.json()["detail"] == {
        "status": "not_ready",
        "dependencies": {
            "qdrant": "ok",
            "ollama": "unavailable",
            "llm_model": "unavailable",
            "embedding_model": "unavailable",
        },
    }


def test_ready_returns_503_when_llm_model_is_missing(
    monkeypatch,
):
    configure_settings(monkeypatch)

    class MissingLLMModelClient(FakeOllamaClient):
        def list(self):
            return SimpleNamespace(
                models=[
                    SimpleNamespace(
                        model="nomic-embed-text:latest"
                    ),
                ]
            )

    monkeypatch.setattr(
        main,
        "QdrantClient",
        FakeQdrantClient,
    )
    monkeypatch.setattr(
        main,
        "OllamaClient",
        MissingLLMModelClient,
    )

    response = TestClient(main.app).get("/ready")

    assert response.status_code == 503
    assert response.json()["detail"] == {
        "status": "not_ready",
        "dependencies": {
            "qdrant": "ok",
            "ollama": "ok",
            "llm_model": "missing",
            "embedding_model": "ok",
        },
    }


def test_ready_returns_503_when_embedding_model_is_missing(
    monkeypatch,
):
    configure_settings(monkeypatch)

    class MissingEmbeddingModelClient(FakeOllamaClient):
        def list(self):
            return SimpleNamespace(
                models=[
                    SimpleNamespace(model="llama3.2:3b"),
                ]
            )

    monkeypatch.setattr(
        main,
        "QdrantClient",
        FakeQdrantClient,
    )
    monkeypatch.setattr(
        main,
        "OllamaClient",
        MissingEmbeddingModelClient,
    )

    response = TestClient(main.app).get("/ready")

    assert response.status_code == 503
    assert response.json()["detail"] == {
        "status": "not_ready",
        "dependencies": {
            "qdrant": "ok",
            "ollama": "ok",
            "llm_model": "ok",
            "embedding_model": "missing",
        },
    }


def test_ready_matches_untagged_model_to_latest():
    assert main._model_is_available(
        "nomic-embed-text",
        {"nomic-embed-text:latest"},
    )


def test_ready_does_not_match_different_model_tags():
    assert not main._model_is_available(
        "llama3.2:3b",
        {"llama3.2:1b"},
    )