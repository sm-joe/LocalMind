from dataclasses import dataclass

from app.services.rag import RAGService
from app.services.retrieval import (
    RetrievalDiagnostics,
    RetrievalPerformance,
    RetrievalResult,
    RetrievedChunk,
)


def make_chunk(chunk_id: int, text: str) -> RetrievedChunk:
    return RetrievedChunk(
        document_id="doc-fast-path",
        filename="guide.txt",
        content_type="text/plain",
        chunk_id=chunk_id,
        text=text,
        start_char=0,
        end_char=len(text),
        score=0.9 - (chunk_id * 0.001),
    )


def make_result(chunks: list[RetrievedChunk]) -> RetrievalResult:
    count = len(chunks)
    return RetrievalResult(
        chunks=chunks,
        diagnostics=RetrievalDiagnostics(
            requested_limit=count,
            candidate_count=count,
            returned_count=count,
            filtered_count=0,
            top_candidate_score=0.9 if count else None,
            bottom_candidate_score=0.8 if count else None,
            top_score=0.9 if count else None,
            bottom_score=0.8 if count else None,
            has_context=bool(chunks),
        ),
        performance=RetrievalPerformance(
            embedding_ms=3.0,
            search_ms=2.0,
            total_ms=5.0,
        ),
    )


@dataclass
class FakeRetrieval:
    result: RetrievalResult
    document_calls: int = 0
    semantic_calls: int = 0

    def retrieve_document(self, query: str, limit: int | None = None) -> RetrievalResult:
        self.document_calls += 1
        return self.result

    def retrieve(self, query: str, limit: int | None = None) -> RetrievalResult:
        self.semantic_calls += 1
        return self.result


@dataclass
class FakeLLM:
    calls: list[str]

    def generate(self, prompt: str) -> str:
        self.calls.append(prompt)
        return "The guide describes the LocalRAG system [1]."


def test_document_overview_uses_extracts_without_calling_llm() -> None:
    chunks = [
        make_chunk(1, "LocalRAG indexes uploaded documents for retrieval. It uses chunking to prepare content."),
        make_chunk(2, "Qdrant stores the document embeddings and chunk metadata."),
        make_chunk(3, "Ollama generates answers from relevant retrieved passages."),
    ]
    retrieval = FakeRetrieval(make_result(chunks))
    llm = FakeLLM(calls=[])

    response = RAGService(retrieval, llm).answer("Give me an overview of the uploaded document")

    assert retrieval.document_calls == 1
    assert retrieval.semantic_calls == 0
    assert llm.calls == []
    assert response.performance.llm_ms == 0.0
    assert len(response.sources) == 3
    assert "LocalRAG indexes uploaded documents" in response.answer
    assert "Qdrant stores the document embeddings" in response.answer
    assert "Ollama generates answers" in response.answer
    assert "[1]" in response.answer and "[3]" in response.answer


def test_overview_samples_at_most_eight_chunks_from_across_document() -> None:
    chunks = [make_chunk(i, f"Section {i} explains a distinct important concept in this guide.") for i in range(1, 21)]
    retrieval = FakeRetrieval(make_result(chunks))
    llm = FakeLLM(calls=[])

    response = RAGService(retrieval, llm).answer("Summarize the document")

    assert len(response.sources) == 8
    assert llm.calls == []
    assert "Section 1" in response.answer
    assert "Section 20" in response.answer


def test_context_format_truncates_long_chunks_but_keeps_citation_boundary() -> None:
    chunk = make_chunk(1, "Useful evidence " + ("detail " * 1000))

    context = RAGService._format_context([chunk])

    assert len(context) < 1800
    assert "Useful evidence" in context
    assert "[Passage truncated to keep the prompt concise.]" in context
    assert '<retrieved_document source="1">' in context
    assert "⟦SRC:1⟧" in context
    assert "</retrieved_document>" in context


def test_normal_question_limits_context_to_five_chunks() -> None:
    chunks = [make_chunk(i, f"Chunk {i} contains relevant evidence for a normal question.") for i in range(1, 8)]
    retrieval = FakeRetrieval(make_result(chunks))
    llm = FakeLLM(calls=[])

    response = RAGService(retrieval, llm).answer("What does the guide say about evidence?")

    assert len(response.sources) == 5
    assert len(llm.calls) == 1
    assert '<retrieved_document source="5">' in llm.calls[0]
    assert '<retrieved_document source="6">' not in llm.calls[0]


def test_streaming_document_overview_skips_generation_and_token_events() -> None:
    chunks = [
        make_chunk(1, "LocalRAG prepares uploaded files for semantic retrieval."),
        make_chunk(2, "Qdrant stores vectors alongside chunk metadata."),
    ]
    retrieval = FakeRetrieval(make_result(chunks))
    llm = FakeLLM(calls=[])
    service = RAGService(retrieval, llm)

    events = list(service.stream_answer("Give me an overview of the uploaded document"))

    assert [event["type"] for event in events].count("token") == 0
    assert llm.calls == []
    assert events[-1]["type"] == "complete"
    assert events[-1]["result"].performance.llm_ms == 0.0
    assert "LocalRAG prepares uploaded files" in events[-1]["result"].answer


def test_streaming_normal_question_uses_bounded_context() -> None:
    chunks = [
        make_chunk(i, f"Chunk {i} contains relevant evidence for a normal question.")
        for i in range(1, 8)
    ]
    retrieval = FakeRetrieval(make_result(chunks))
    llm = FakeLLM(calls=[])
    service = RAGService(retrieval, llm)

    events = list(service.stream_answer("What does the guide say about evidence?"))
    result = events[-1]["result"]

    assert len(result.sources) == 5
    assert len(llm.calls) == 1
    assert '<retrieved_document source="5">' in llm.calls[0]
    assert '<retrieved_document source="6">' not in llm.calls[0]
