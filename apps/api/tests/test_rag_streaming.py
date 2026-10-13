from app.services.rag import RAGService
from app.services.retrieval import (
    RetrievedChunk,
    RetrievalDiagnostics,
    RetrievalPerformance,
    RetrievalResult,
)


class FakeRetrievalService:
    def retrieve_hybrid(self, query, limit=None):
        chunk = RetrievedChunk(
            document_id="doc-1",
            filename="Interview_Questions.docx",
            content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            chunk_id=2,
            text="IAM Roles provide temporary credentials and avoid storing long-term AWS access keys on EC2 instances, improving security.",
            start_char=0,
            end_char=120,
            score=0.91,
        )
        return RetrievalResult(
            chunks=[chunk],
            diagnostics=RetrievalDiagnostics(
                requested_limit=5,
                candidate_count=1,
                returned_count=1,
                filtered_count=0,
                top_candidate_score=0.91,
                bottom_candidate_score=0.91,
                top_score=0.91,
                bottom_score=0.91,
                has_context=True,
            ),
            performance=RetrievalPerformance(
                embedding_ms=5.0,
                search_ms=3.0,
                total_ms=8.0,
            ),
        )


class FakeLLMService:
    def generate_stream(self, prompt):
        yield "IAM Roles provide temporary credentials "
        yield "and avoid storing long-term AWS access keys on EC2 instances, improving security."


def test_stream_answer_emits_retrieval_sources_tokens_and_canonical_final_answer():
    service = RAGService(FakeRetrievalService(), FakeLLMService())

    events = list(service.stream_answer("Why prefer IAM Roles for EC2?", top_k=5))
    event_types = [event["type"] for event in events]

    assert event_types[0] == "status"
    assert "retrieval_complete" in [event.get("stage") for event in events]
    assert "sources" in event_types
    assert event_types.index("sources") < event_types.index("token")
    assert event_types.count("token") == 2
    final = events[-1]
    assert final["type"] == "complete"
    assert final["result"].sources[0].citation == "[1]"
    assert "[1]" in final["result"].answer
    assert final["result"].retrieval.has_context is True


def test_stream_answer_rejects_blank_question_before_retrieval():
    service = RAGService(FakeRetrievalService(), FakeLLMService())
    stream = service.stream_answer("   ")
    try:
        next(stream)
    except ValueError as exc:
        assert "Question cannot be empty" in str(exc)
    else:
        raise AssertionError("Expected blank question to be rejected")
