"""Regression coverage for upload-first document Q&A and summaries."""
from types import SimpleNamespace
from unittest.mock import MagicMock

from app.services.retrieval import RetrievalService, RetrievalDiagnostics, RetrievalPerformance, RetrievalResult, RetrievedChunk
from app.services.rag import RAGService


def point(document_id: str, filename: str, number: int, text: str):
    return SimpleNamespace(payload={
        "document_id": document_id,
        "filename": filename,
        "content_type": "text/plain",
        "chunk_id": number,
        "text": text,
        "start_char": 0,
        "end_char": len(text),
    })


def test_generic_summary_reads_indexed_documents_without_embedding():
    embedding = MagicMock()
    vector_store = MagicMock()
    vector_store.iter_points.return_value = iter([
        point("one", "architecture.txt", 0, "Architecture uses FastAPI."),
        point("one", "architecture.txt", 1, "Qdrant stores vectors."),
        point("two", "operations.md", 0, "Operations uses containers."),
        point("two", "operations.md", 1, "Health checks test readiness."),
    ])
    retrieval = RetrievalService(embedding, vector_store)
    result = retrieval.retrieve_document("Please summarize the uploaded documents.")
    assert [c.filename for c in result.chunks] == [
        "architecture.txt", "operations.md", "architecture.txt", "operations.md",
    ]
    assert result.diagnostics.has_context is True
    embedding.embed.assert_not_called()
    vector_store.search.assert_not_called()


def test_generic_summary_has_no_sources_when_collection_empty():
    vector_store = MagicMock()
    vector_store.iter_points.return_value = iter([])
    result = RetrievalService(MagicMock(), vector_store).retrieve_document(
        "Summarize the uploaded document."
    )
    assert result.chunks == []
    assert result.diagnostics.has_context is False


def test_summary_without_model_citations_returns_real_cited_excerpts():
    chunks = [
        RetrievedChunk(
            document_id="one", filename="notes.txt",
            content_type="text/plain", chunk_id=0,
            text="Qdrant stores vector embeddings.", start_char=0,
            end_char=32, score=0.8,
        ),
    ]
    result = RetrievalResult(
        chunks=chunks,
        diagnostics=RetrievalDiagnostics(
            requested_limit=1, candidate_count=1, returned_count=1,
            filtered_count=0, top_candidate_score=0.8,
            bottom_candidate_score=0.8, top_score=0.8,
            bottom_score=0.8, has_context=True,
        ),
        performance=RetrievalPerformance(embedding_ms=0, search_ms=1, total_ms=1),
    )
    retrieval = MagicMock()
    retrieval.retrieve_document.return_value = result
    llm = MagicMock()
    llm.generate.return_value = "The answer is an unsupported summary without citations."
    response = RAGService(retrieval, llm).answer(
        "Please summarize the uploaded document."
    )
    assert "Relevant verbatim excerpts:" in response.answer
    assert "Qdrant stores vector embeddings." in response.answer
    assert "[1]" in response.answer
    assert response.sources[0].document_id == "one"
    assert llm.generate.call_count == 1


def test_twelve_chunk_docx_summary_uses_one_small_prompt():
    chunks = [
        RetrievedChunk(
            document_id="interviews",
            filename="Interview_Questions.docx",
            content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            chunk_id=index,
            text=f"Question {index}: Describe the engineering decision. " + ("Example and tradeoffs. " * 65),
            start_char=index * 1100,
            end_char=(index + 1) * 1100,
            score=0.8,
        )
        for index in range(12)
    ]
    retrieval = MagicMock()
    retrieval.retrieve_document.return_value = RetrievalResult(
        chunks=chunks,
        diagnostics=RetrievalDiagnostics(
            requested_limit=12, candidate_count=12, returned_count=12,
            filtered_count=0, top_candidate_score=0.8, bottom_candidate_score=0.8,
            top_score=0.8, bottom_score=0.8, has_context=True,
        ),
        performance=RetrievalPerformance(embedding_ms=0, search_ms=1, total_ms=1),
    )
    llm = MagicMock()
    llm.generate.return_value = "The document covers engineering interview decisions [1]."
    result = RAGService(retrieval, llm).answer("Summarize the document.")
    assert llm.generate.call_count == 1
    assert len(result.sources) == 12
    assert result.sources[-1].chunk_id == 11
    assert result.answer.endswith("[1].")
    prompt = llm.generate.call_args.args[0]
    assert len(prompt) < 11000
    assert 'source="12"' in prompt
