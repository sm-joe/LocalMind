"""Named-service RAG must search the complete indexed text and cite matching evidence."""
from types import SimpleNamespace

from app.services.rag import RAGService
from app.services.retrieval import RetrievalService


def point(index: int, text: str):
    return SimpleNamespace(payload={
        "document_id": "interview-guide",
        "filename": "Interview_Questions.docx",
        "content_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "chunk_id": index,
        "text": text,
        "start_char": index * 1000,
        "end_char": (index + 1) * 1000,
    })


def retrieval(points):
    class Embeddings:
        def embed(self, text):
            raise AssertionError("Named-service lookup must not rely on top-three embeddings")

    class Store:
        def iter_points(self):
            return iter(points)
        def search(self, **kwargs):
            raise AssertionError("Named-service query must scan indexed text")

    return RetrievalService(Embeddings(), Store())


def test_bedrock_found_in_late_indexed_chunk_and_answered():
    points = [
        point(i, f"Unrelated AWS topic {i}.") for i in range(11)
    ] + [point(11, "Amazon Bedrock is a managed service for building generative AI applications.")]
    class Model:
        def stream(self, prompt):
            assert "Amazon Bedrock is a managed service" in prompt
            yield "Amazon Bedrock is a managed generative AI service [1]."

    result = list(RAGService(retrieval(points), Model()).stream_answer(
        "what is Amazon Bedrock service mentioned in the document?"
    ))[-1][1]
    assert "managed generative AI service" in result.answer
    assert result.sources[0].chunk_id == 11
    assert result.performance.embedding_ms == 0


def test_bedrock_mention_in_middle_of_long_chunk_survives_prompt_trimming():
    text = ("Earlier unrelated details. " * 150)
    text += "Amazon Bedrock is used for foundation model access. "
    text += "Trailing details. " * 200

    class Model:
        def stream(self, prompt):
            assert "Amazon Bedrock is used for foundation model access." in prompt
            yield "Amazon Bedrock provides foundation model access [1]."

    result = list(RAGService(retrieval([point(2, text)]), Model()).stream_answer(
        "What is Amazon Bedrock service?"
    ))[-1][1]
    assert "foundation model access" in result.answer


def test_named_subject_not_in_index_reports_limit_without_guessing():
    class Model:
        def stream(self, prompt):
            raise AssertionError("Do not fabricate an answer when named entity is absent.")

    result = list(RAGService(retrieval([
        point(0, "AWS EC2 provides virtual machines."),
        point(1, "AWS IAM manages permissions."),
    ]), Model()).stream_answer(
        "what is Amazon Bedrock service mentioned in the document?"
    ))[-1][1]
    assert result.sources == []
    assert "couldn't find an exact mention" in result.answer
    assert "indexed document text" in result.answer


def test_comparison_requires_both_sides_cited_on_their_own_line():
    from app.services.retrieval import RetrievedChunk

    def chunk(i, value):
        return RetrievedChunk(
            document_id="guide", filename="Interview_Questions.docx",
            content_type="text/plain", chunk_id=i, text=value,
            start_char=0, end_char=len(value), score=0.7,
        )
    chunks = [
        chunk(0, "EC2 provides virtual servers."),
        chunk(1, "IAM provides identity and access management."),
        chunk(2, "EC2 instances may assume IAM roles."),
    ]
    bad_answer = (
        "EC2 provides virtual machines [1].\n"
        "IAM helps manage access through policies and roles.\n"
    )
    assert not RAGService._comparison_claims_supported(
        bad_answer, chunks, "ec2", "iam"
    )
    assert RAGService._comparison_claims_supported(
        "EC2 provides virtual servers [1], while IAM manages access [2].",
        chunks, "ec2", "iam"
    )
