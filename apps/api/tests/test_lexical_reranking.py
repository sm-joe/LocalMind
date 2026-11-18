from types import SimpleNamespace
from unittest.mock import MagicMock

from app.services.retrieval import RetrievalService


def _candidate(chunk_id: int, score: float, text: str):
    return SimpleNamespace(
        score=score,
        payload={
            "document_id": "doc-1",
            "filename": "policy.md",
            "content_type": "text/markdown",
            "chunk_id": chunk_id,
            "text": text,
            "start_char": 0,
            "end_char": len(text),
        },
    )


def test_lexical_overlap_can_rescue_relevant_candidate():
    embedding = MagicMock()
    embedding.embed.return_value = [0.1, 0.3]
    store = MagicMock()
    store.search.return_value = [
        _candidate(1, 0.83, "General weather forecast and office events."),
        _candidate(2, 0.76, "Customer log retention policy keeps audit logs."),
    ]
    retrieval = RetrievalService(
        embedding, store, top_k=2, score_threshold=0.4, lexical_weight=0.2
    )
    result = retrieval.retrieve("What is the customer log retention policy?")
    assert [chunk.chunk_id for chunk in result.chunks] == [2, 1]
    # Citation scores remain genuine vector similarity, not blended ranking scores.
    assert result.chunks[0].score == 0.76
    assert result.diagnostics.candidate_count == 2


def test_lexical_reranking_is_opt_in_and_default_preserves_vector_order():
    embedding = MagicMock()
    embedding.embed.return_value = [0.1, 0.3]
    store = MagicMock()
    store.search.return_value = [
        _candidate(1, 0.83, "Weather forecasts."),
        _candidate(2, 0.76, "Customer log retention policy."),
    ]
    result = RetrievalService(embedding, store, top_k=2).retrieve(
        "customer log retention policy"
    )
    assert [chunk.chunk_id for chunk in result.chunks] == [1, 2]
