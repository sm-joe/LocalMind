"""Ensure retrieved evidence fits the configured local-model context."""
from types import SimpleNamespace

from app.services.llm import LLMService


def test_ollama_chat_uses_extended_context_for_cited_answers(monkeypatch):
    service = LLMService.__new__(LLMService)
    service.model = "llama3.2:3b"
    service.num_ctx = 6144

    class FakeClient:
        def __init__(self):
            self.options = None
        def chat(self, **kwargs):
            self.options = kwargs["options"]
            return SimpleNamespace(message=SimpleNamespace(content="Qdrant stores vectors [1]."))

    service.client = FakeClient()
    result = service.generate("Answer with retrieved evidence and citations.")
    assert result == "Qdrant stores vectors [1]."
    assert service.client.options["num_ctx"] == 6144


def test_qwen3_disables_thinking_for_interactive_answers():
    service = LLMService.__new__(LLMService)
    service.model = "qwen3:8b"
    service.num_ctx = 6144

    class FakeClient:
        def chat(self, **kwargs):
            assert kwargs["think"] is False
            assert kwargs["options"]["num_predict"] <= 512
            return SimpleNamespace(
                message=SimpleNamespace(content="SaturnDB stores metadata [1].")
            )

    service.client = FakeClient()
    assert service.generate("Which database is used?") == "SaturnDB stores metadata [1]."
