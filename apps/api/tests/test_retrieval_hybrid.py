from types import SimpleNamespace

from app.services.retrieval import (
    RetrievalDiagnostics,
    RetrievalPerformance,
    RetrievalResult,
    RetrievalService,
    RetrievedChunk,
)


class FakeEmbedding:
    def embed(self, text: str) -> list[float]:
        return [1.0, 0.0]


class FakeVectorStore:
    def __init__(self):
        self.points = [
            SimpleNamespace(payload={
                "document_id": "doc-a", "filename": "aws.txt",
                "content_type": "text/plain", "chunk_id": 0,
                "text": "Amazon S3 provides object storage for AWS applications.",
                "start_char": 0, "end_char": 55,
            }),
            SimpleNamespace(payload={
                "document_id": "doc-b", "filename": "terraform.txt",
                "content_type": "text/plain", "chunk_id": 0,
                "text": "Terraform manages infrastructure as code.",
                "start_char": 0, "end_char": 41,
            }),
        ]

    def search(self, vector, limit=5, filename=None):
        return []

    def iter_points(self):
        yield from self.points


class TestableRetrievalService(RetrievalService):
    def retrieve(self, query: str, limit: int | None = None) -> RetrievalResult:
        requested = self.top_k if limit is None else limit
        return RetrievalResult(
            chunks=[],
            diagnostics=RetrievalDiagnostics(
                requested_limit=requested, candidate_count=0, returned_count=0,
                filtered_count=0, top_candidate_score=None, bottom_candidate_score=None,
                top_score=None, bottom_score=None, has_context=False,
            ),
            performance=RetrievalPerformance(embedding_ms=1, search_ms=1, total_ms=2),
        )


def test_hybrid_retrieval_finds_lexical_match_without_dense_hits():
    service = TestableRetrievalService(FakeEmbedding(), FakeVectorStore(), top_k=5, score_threshold=0.4)
    result = service.retrieve_hybrid("Amazon S3 object storage")
    assert result.diagnostics.has_context is True
    assert result.chunks[0].filename == "aws.txt"
    assert result.chunks[0].score >= 0.4


def test_hybrid_retrieval_returns_no_chunks_for_unmatched_query():
    service = TestableRetrievalService(FakeEmbedding(), FakeVectorStore(), top_k=5, score_threshold=0.4)
    result = service.retrieve_hybrid("photos of dolphins")
    assert result.chunks == []
    assert result.diagnostics.has_context is False
