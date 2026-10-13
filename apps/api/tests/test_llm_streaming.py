from types import SimpleNamespace

from app.services.llm import LLMService


class FakeOllamaClient:
    def __init__(self, parts):
        self.parts = parts
        self.kwargs = None

    def chat(self, **kwargs):
        self.kwargs = kwargs
        if kwargs.get("stream"):
            return iter(self.parts)
        return SimpleNamespace(message=SimpleNamespace(content="complete answer"))


def test_generate_stream_yields_ollama_deltas_in_order():
    client = FakeOllamaClient([
        SimpleNamespace(message=SimpleNamespace(content="hello")),
        SimpleNamespace(message=SimpleNamespace(content=" world")),
    ])
    service = LLMService.__new__(LLMService)
    service.client = client
    service.model = "test-model"

    assert list(service.generate_stream("prompt")) == ["hello", " world"]
    assert client.kwargs["stream"] is True
    assert client.kwargs["options"]["num_predict"] == 320


def test_generate_stream_rejects_empty_output():
    client = FakeOllamaClient([
        SimpleNamespace(message=SimpleNamespace(content="")),
    ])
    service = LLMService.__new__(LLMService)
    service.client = client
    service.model = "test-model"

    try:
        list(service.generate_stream("prompt"))
    except RuntimeError as exc:
        assert "empty streaming response" in str(exc)
    else:
        raise AssertionError("Expected an empty-stream error")
