"""Regression: compare two concepts, not two subtypes of one concept."""
from types import SimpleNamespace

from app.services.retrieval import RetrievalService
from app.services.rag import RAGService


def point(chunk_id: int, text: str):
    return SimpleNamespace(payload={
        "document_id": "interviews",
        "filename": "Interview_Questions.docx",
        "content_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "chunk_id": chunk_id,
        "text": text,
        "start_char": 0,
        "end_char": len(text),
    })


def create_retrieval():
    class Embedding:
        def embed(self, query):
            raise AssertionError("A named comparison should use exact-term source coverage")

    class Store:
        def iter_points(self):
            yield point(0, "Amazon EC2 is a compute service for virtual machines.")
            yield point(1, "AWS IAM manages identities and access permissions.")
            yield point(2, "IAM users have long-term identities. IAM roles use temporary credentials.")
            yield point(3, "EC2 instances can assume IAM roles.")

    return RetrievalService(Embedding(), Store())


def test_ec2_versus_iam_retrieves_both_concepts():
    result = create_retrieval().retrieve_comparison(
        "difference between ec2 and iam", "ec2", "iam"
    )
    assert result.diagnostics.has_context
    assert any("compute service" in chunk.text for chunk in result.chunks)
    assert any("access permissions" in chunk.text for chunk in result.chunks)
    assert result.performance.embedding_ms == 0


def test_ec2_versus_iam_rejects_iam_user_role_answer():
    class WrongModel:
        def stream(self, prompt):
            assert "EC2 is a compute service" in prompt
            assert "IAM manages identities" in prompt
            yield "An IAM Role uses temporary credentials, unlike an IAM User [2]."

    events = list(RAGService(create_retrieval(), WrongModel()).stream_answer(
        "difference between ec2 and iam"
    ))
    answer = events[-1][1].answer
    assert "From the indexed document:" in answer
    assert "EC2 is a compute service" in answer
    assert "IAM manages identities" in answer
    assert "[1]" in answer and "[2]" in answer
    assert "IAM Role uses temporary" not in answer


def test_ec2_versus_iam_accepts_direct_answer():
    class Model:
        def stream(self, prompt):
            yield "EC2 runs virtual servers [1], while IAM manages AWS permissions [2]."

    events = list(RAGService(create_retrieval(), Model()).stream_answer(
        "difference between ec2 and iam"
    ))
    answer = events[-1][1].answer
    assert "EC2 runs virtual servers" in answer
    assert "IAM manages AWS permissions" in answer


def test_comparison_does_not_guess_when_one_subject_missing():
    class Embedding:
        def embed(self, query):
            raise AssertionError("Should not embed")
    class Store:
        def iter_points(self):
            yield point(0, "IAM users and IAM roles control access.")

    result = RetrievalService(Embedding(), Store()).retrieve_comparison(
        "difference between ec2 and iam", "ec2", "iam"
    )
    assert result.chunks == []
