from types import SimpleNamespace

from app.services.rag import RAGService
from app.services.retrieval import (
    RetrievedChunk, RetrievalDiagnostics, RetrievalPerformance, RetrievalResult
)


def _chunk(i: int) -> RetrievedChunk:
    return RetrievedChunk(
        document_id="doc", filename="AWS.txt", content_type="text/plain",
        chunk_id=i, text=f"Document excerpt {i}: AWS service evidence.",
        start_char=0, end_char=40, score=0.8,
    )


def _result(chunks):
    return RetrievalResult(
        chunks=chunks,
        diagnostics=RetrievalDiagnostics(
            requested_limit=5, candidate_count=len(chunks),
            returned_count=len(chunks), filtered_count=0,
            top_candidate_score=0.8, bottom_candidate_score=0.8,
            top_score=0.8, bottom_score=0.8,
            has_context=bool(chunks),
        ),
        performance=RetrievalPerformance(
            embedding_ms=1.0, search_ms=1.0, total_ms=2.0,
        ),
    )


def test_hybrid_adapter_supports_normal_question_and_five_chunk_budget():
    class Hybrid:
        def retrieve_hybrid(self, question, limit=None):
            return _result([_chunk(i) for i in range(6)])

    prompts = []
    class LLM:
        def generate(self, prompt):
            prompts.append(prompt)
            return "Document excerpt about AWS [1]."

    response = RAGService(Hybrid(), LLM()).answer("What does AWS provide?")
    assert len(response.sources) == 5
    assert 'source="6"' not in prompts[0]


def test_requested_numbered_list_reads_document_and_falls_back_to_cited_evidence():
    calls = []
    class Retrieval:
        def retrieve_document(self, question, limit=None):
            calls.append("document")
            return _result([_chunk(0), _chunk(1)])

        def retrieve(self, question, limit=None):
            calls.append("normal")
            return _result([])

    class LLM:
        def generate(self, prompt):
            return "The response has no relation to the indexed content."

    response = RAGService(Retrieval(), LLM()).answer(
        "List top 10 services mentioned in the document."
    )
    assert calls == ["document"]
    assert "Document excerpt 0" in response.answer
    assert "Document excerpt 1" in response.answer
    assert "[1]" in response.answer and "[2]" in response.answer
