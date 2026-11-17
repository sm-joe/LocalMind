"""Product hardening regression tests: partial ingestion and bounded RAG."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.services.ingestion import IngestionService
from app.services.rag import RAGService
from app.services.retrieval import (
    RetrievalDiagnostics,
    RetrievalPerformance,
    RetrievalResult,
    RetrievedChunk,
)


def test_index_failure_removes_partially_written_document(tmp_path: Path):
    path = tmp_path / "input.txt"
    path.write_text("A document about reliable search indexing.", encoding="utf-8")
    embedding = MagicMock()
    embedding.embed_many.return_value = [[0.2, 0.3, 0.5]]
    store = MagicMock()
    store.find_document_by_hash.return_value = None
    store.upsert.side_effect = RuntimeError("Qdrant request interrupted")

    with pytest.raises(RuntimeError, match="interrupted"):
        IngestionService(embedding, store).ingest(path, "text/plain")

    document_id = store.upsert.call_args.args[0][0].payload["document_id"]
    store.delete_document.assert_called_once_with(document_id)


def test_index_cleanup_failure_does_not_hide_original_error(tmp_path: Path):
    path = tmp_path / "input.txt"
    path.write_text("A document about interrupted uploads.", encoding="utf-8")
    embedding = MagicMock()
    embedding.embed_many.return_value = [[0.2, 0.3, 0.5]]
    store = MagicMock()
    store.find_document_by_hash.return_value = None
    store.upsert.side_effect = RuntimeError("upsert rejected")
    store.delete_document.side_effect = RuntimeError("cleanup rejected")

    with pytest.raises(RuntimeError, match="upsert rejected"):
        IngestionService(embedding, store).ingest(path, "text/plain")

    store.delete_document.assert_called_once()


def _chunk(number: int, chars: int = 1800) -> RetrievedChunk:
    return RetrievedChunk(
        document_id="doc-1",
        filename="manual.pdf",
        content_type="application/pdf",
        chunk_id=number,
        text=f"Section {number}: " + ("A" * chars),
        start_char=number * chars,
        end_char=(number + 1) * chars,
        score=0.8,
    )


def test_long_summary_has_bounded_context_and_source_coverage():
    chunks = [_chunk(i) for i in range(100)]
    selected = RAGService._select_context_chunks(chunks, summary_mode=True)
    assert len(selected) == RAGService.MAX_CONTEXT_CHUNKS
    assert selected[0].chunk_id == 0
    assert selected[-1].chunk_id == 99
    assert sum(len(chunk.text) for chunk in selected) <= RAGService.MAX_CONTEXT_CHARS
    assert all(selected[i].chunk_id < selected[i + 1].chunk_id for i in range(len(selected) - 1))
    assert chunks[-1].text != selected[-1].text  # source data is not mutated


def test_normal_question_preserves_relevance_order_and_budget():
    chunks = [_chunk(i) for i in range(30)]
    selected = RAGService._select_context_chunks(chunks, summary_mode=False)
    assert [chunk.chunk_id for chunk in selected] == list(range(20))
    assert sum(len(chunk.text) for chunk in selected) <= RAGService.MAX_CONTEXT_CHARS


def test_answer_citations_only_reference_context_sent_to_model():
    chunks = [_chunk(i) for i in range(100)]
    retrieval = MagicMock()
    retrieval.retrieve_document.return_value = RetrievalResult(
        chunks=chunks,
        diagnostics=RetrievalDiagnostics(
            requested_limit=100, candidate_count=100, returned_count=100,
            filtered_count=0, top_candidate_score=0.9, bottom_candidate_score=0.5,
            top_score=0.9, bottom_score=0.5, has_context=True,
        ),
        performance=RetrievalPerformance(embedding_ms=1.0, search_ms=1.0, total_ms=2.0),
    )
    llm = MagicMock()
    llm.generate.return_value = "The last section is included [20]."
    result = RAGService(retrieval, llm).answer("Summarize manual.pdf")
    assert len(result.sources) == RAGService.MAX_CONTEXT_CHUNKS
    assert result.sources[-1].chunk_id == 99
    assert result.sources[-1].citation == "[20]"
    assert result.answer == "The last section is included [20]."
    assert len(llm.generate.call_args.args[0]) < 32_000
