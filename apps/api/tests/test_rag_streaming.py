"""Reference-style RAG streaming: small retrieval, immediate tokens, no retry."""
from dataclasses import dataclass

from app.services.rag import RAGService
from app.services.retrieval import (
    RetrievedChunk, RetrievalDiagnostics, RetrievalPerformance, RetrievalResult,
)


def corpus(count=12):
    chunks = [
        RetrievedChunk(
            document_id="interview-document",
            filename="Interview_Questions.docx",
            content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            chunk_id=i,
            text=f"Interview topic {i}: Angular and TypeScript answer guidance.",
            start_char=i * 65,
            end_char=(i + 1) * 65,
            score=0.9,
        )
        for i in range(count)
    ]
    return RetrievalResult(
        chunks=chunks,
        diagnostics=RetrievalDiagnostics(
            requested_limit=count, candidate_count=count, returned_count=count,
            filtered_count=0, top_candidate_score=.9, bottom_candidate_score=.9,
            top_score=.9, bottom_score=.9, has_context=bool(chunks),
        ),
        performance=RetrievalPerformance(
            embedding_ms=0, search_ms=1, total_ms=1,
        ),
    )


def test_summary_streams_tokens_once_and_preserves_citations():
    class Retrieval:
        def retrieve_document(self, query, limit=None):
            assert "Summarize" in query
            return corpus()

    class LLM:
        calls = 0
        prompts = []
        def stream(self, prompt):
            self.calls += 1
            self.prompts.append(prompt)
            yield "This document covers "
            yield "Angular and TypeScript [1]."

    llm = LLM()
    parts = list(RAGService(Retrieval(), llm).stream_answer(
        "Summarize the document."
    ))
    assert [kind for kind, _ in parts] == ["chunk", "chunk", "done"]
    assert parts[0][1] == "This document covers "
    assert llm.calls == 1
    assert len(llm.prompts[0]) < 5000
    response = parts[-1][1]
    assert response.answer == "This document covers Angular and TypeScript [1]."
    assert len(response.sources) == 8
    assert response.sources[0].chunk_id == 0
    assert response.sources[-1].chunk_id == 11


def test_question_streams_top_three_relevant_chunks():
    class Retrieval:
        def retrieve(self, query, limit=None):
            assert limit == 3
            return corpus()

    class LLM:
        def stream(self, prompt):
            assert prompt.count("<source id=") == 3
            yield "Angular and TypeScript [1]."

    output = list(RAGService(Retrieval(), LLM()).stream_answer("What are the topics?"))
    assert len(output[-1][1].sources) == 3
    assert output[-1][1].answer == "Angular and TypeScript [1]."


def test_uncited_model_output_never_relabels_unverified_claims_as_cited():
    class Retrieval:
        def retrieve(self, query, limit=None):
            return corpus(1)

    class LLM:
        def stream(self, prompt):
            yield "Unsupported claim."

    events = list(RAGService(Retrieval(), LLM()).stream_answer("What is covered?"))
    assert events[0] == ("chunk", "Unsupported claim.")
    assert events[-1][0] == "done"
    assert events[-1][1].answer.startswith("Relevant verbatim excerpts:")
    assert "Interview topic 0" in events[-1][1].answer
    assert "[1]" in events[-1][1].answer


def test_missing_corpus_streams_a_clear_answer_without_model_call():
    class Retrieval:
        def retrieve(self, query, limit=None):
            return corpus(0)

    class LLM:
        def stream(self, prompt):
            raise AssertionError("Do not call Ollama when Qdrant returned no chunks.")

    events = list(RAGService(Retrieval(), LLM()).stream_answer("Anything?"))
    assert len(events) == 1
    assert events[0][0] == "done"
    assert events[0][1].sources == []
