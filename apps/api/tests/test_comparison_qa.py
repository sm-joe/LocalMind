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


def test_real_audit_style_ec2_iam_prioritizes_definition_over_role_example():
    """Reproduce indexed DOCX chunk ordering reported by the operator."""
    class Embedding:
        def embed(self, query):
            raise AssertionError("Comparison evidence comes from indexed chunks.")

    class Store:
        def iter_points(self):
            yield point(0, "Shared responsibility includes IAM permissions.")
            yield point(1, "2. IAM and Security 6. What is AWS IAM? Answer: IAM (Identity and Access Management) controls who can access AWS resources and which actions they can perform.")
            yield point(2, "Why should IAM Roles be preferred over access keys for EC2? Answer: IAM Roles provide temporary credentials and avoid storing AWS access keys on EC2.")
            yield point(5, "An Application Load Balancer distributes traffic to EC2 instances.")
            yield point(6, "An Auto Scaling Group launches EC2 instances according to demand.")
            yield point(9, "Architecture: Route 53, an EC2 Auto Scaling Group and an RDS database.")

    result = RetrievalService(Embedding(), Store()).retrieve_comparison(
        "difference between ec2 and iam", "ec2", "iam"
    )
    assert result.diagnostics.has_context
    # The genuine IAM definition, not IAM Role-vs-User details, must be
    # the first IAM source given to the model.
    assert result.chunks[1].chunk_id == 1
    assert result.chunks[0].chunk_id != 2
