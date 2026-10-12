from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.services.retrieval import RetrievalService


def create_payload(
    *,
    filename: str = "Behavioural_questions.docx",
    chunk_id: int = 1,
    text: str = "Example document content.",
) -> dict[str, object]:
    return {
        "document_id": "doc-123",
        "filename": filename,
        "content_type": (
            "application/vnd.openxmlformats-officedocument."
            "wordprocessingml.document"
        ),
        "chunk_id": chunk_id,
        "text": text,
        "start_char": 0,
        "end_char": len(text),
    }


def create_point(
    score: float,
    *,
    filename: str = "Behavioural_questions.docx",
    chunk_id: int = 1,
    text: str = "Example document content.",
) -> SimpleNamespace:
    return SimpleNamespace(
        score=score,
        payload=create_payload(
            filename=filename,
            chunk_id=chunk_id,
            text=text,
        ),
    )


def create_service(
    *,
    top_k: int = 5,
    score_threshold: float = 0.0,
) -> tuple[RetrievalService, MagicMock, MagicMock]:
    embedding_service = MagicMock()
    embedding_service.embed.return_value = [0.1, 0.2, 0.3]

    vector_store = MagicMock()

    service = RetrievalService(
        embedding_service=embedding_service,
        vector_store=vector_store,
        top_k=top_k,
        score_threshold=score_threshold,
    )

    return service, embedding_service, vector_store


def test_extract_filename_from_query() -> None:
    assert (
        RetrievalService._extract_filename(
            "What is in Behavioural_questions.docx?"
        )
        == "Behavioural_questions.docx"
    )


def test_extract_quoted_filename_from_query() -> None:
    assert (
        RetrievalService._extract_filename(
            'Summarize "Behavioural_questions.docx"'
        )
        == "Behavioural_questions.docx"
    )


def test_extract_filename_returns_none_for_normal_query() -> None:
    assert (
        RetrievalService._extract_filename(
            "What is retrieval augmented generation?"
        )
        is None
    )


def test_filename_search_uses_question_embedding_and_exact_filter() -> None:
    service, embedding_service, vector_store = create_service()
    question = (
        "What interview questions are covered in "
        "Behavioural_questions.docx?"
    )
    vector_store.search.return_value = [
        create_point(
            0.87,
            chunk_id=8,
            text="Leadership and behavioral interview questions.",
        ),
        create_point(
            0.71,
            chunk_id=2,
            text="Introduction and interview preparation.",
        ),
    ]

    result = service.retrieve(question)

    embedding_service.embed.assert_called_once_with(question)
    vector_store.search.assert_called_once_with(
        vector=[0.1, 0.2, 0.3],
        limit=5,
        filename="Behavioural_questions.docx",
    )

    assert [chunk.score for chunk in result.chunks] == [0.87, 0.71]
    assert [chunk.chunk_id for chunk in result.chunks] == [8, 2]
    assert result.diagnostics.has_context is True


def test_filename_search_ranks_chunks_by_actual_score() -> None:
    service, _, vector_store = create_service()
    vector_store.search.return_value = [
        create_point(0.72, chunk_id=1, text="Lower relevance."),
        create_point(0.93, chunk_id=9, text="Higher relevance."),
        create_point(0.81, chunk_id=3, text="Medium relevance."),
    ]

    result = service.retrieve(
        "Explain the topic in Behavioural_questions.docx"
    )

    assert [chunk.score for chunk in result.chunks] == [
        0.93,
        0.81,
        0.72,
    ]
    assert [chunk.chunk_id for chunk in result.chunks] == [9, 3, 1]
    assert result.diagnostics.top_candidate_score == 0.93
    assert result.diagnostics.bottom_candidate_score == 0.72
    assert result.diagnostics.top_score == 0.93
    assert result.diagnostics.bottom_score == 0.72


def test_filename_search_applies_configured_score_threshold() -> None:
    service, _, vector_store = create_service(
        score_threshold=0.45,
    )
    vector_store.search.return_value = [
        create_point(0.80, chunk_id=2),
        create_point(0.44, chunk_id=3),
        create_point(0.45, chunk_id=4),
    ]

    result = service.retrieve(
        "Summarize Behavioural_questions.docx"
    )

    assert [chunk.score for chunk in result.chunks] == [0.80, 0.45]
    assert result.diagnostics.candidate_count == 3
    assert result.diagnostics.returned_count == 2
    assert result.diagnostics.filtered_count == 1
    assert result.diagnostics.has_context is True


def test_missing_filename_does_not_fall_back_to_global_search() -> None:
    service, _, vector_store = create_service()
    vector_store.search.return_value = []

    result = service.retrieve(
        "Summarize missing_document.docx"
    )

    vector_store.search.assert_called_once_with(
        vector=[0.1, 0.2, 0.3],
        limit=5,
        filename="missing_document.docx",
    )
    assert result.chunks == []
    assert result.diagnostics.candidate_count == 0
    assert result.diagnostics.returned_count == 0
    assert result.diagnostics.has_context is False


def test_filename_search_returns_no_context_below_threshold() -> None:
    service, _, vector_store = create_service(
        score_threshold=0.45,
    )
    vector_store.search.return_value = [
        create_point(0.44, chunk_id=1),
        create_point(0.31, chunk_id=2),
    ]

    result = service.retrieve(
        "Find details in Behavioural_questions.docx"
    )

    assert result.chunks == []
    assert result.diagnostics.candidate_count == 2
    assert result.diagnostics.returned_count == 0
    assert result.diagnostics.filtered_count == 2
    assert result.diagnostics.top_candidate_score == 0.44
    assert result.diagnostics.bottom_candidate_score == 0.31
    assert result.diagnostics.top_score is None
    assert result.diagnostics.bottom_score is None
    assert result.diagnostics.has_context is False


def test_normal_query_uses_unfiltered_semantic_search() -> None:
    service, embedding_service, vector_store = create_service()
    vector_store.search.return_value = [
        create_point(
            0.91,
            filename="guide.pdf",
            text="Retrieval augmented generation guide.",
        )
    ]
    question = "What is retrieval augmented generation?"

    result = service.retrieve(question)

    embedding_service.embed.assert_called_once_with(question)
    vector_store.search.assert_called_once_with(
        vector=[0.1, 0.2, 0.3],
        limit=20,
    )
    assert len(result.chunks) == 1
    assert result.chunks[0].filename == "guide.pdf"

def test_semantic_search_overfetches_and_limits_final_chunks() -> None:
    service, embedding_service, vector_store = create_service(
        top_k=5,
        score_threshold=0.45,
    )
    vector_store.search.return_value = [
        create_point(
            0.91 - index * 0.05,
            chunk_id=index,
            text=f"Relevant semantic passage {index}.",
        )
        for index in range(8)
    ]

    result = service.retrieve(
        "What security recommendations does the document contain?"
    )

    embedding_service.embed.assert_called_once_with(
        "What security recommendations does the document contain?"
    )
    vector_store.search.assert_called_once_with(
        vector=[0.1, 0.2, 0.3],
        limit=20,
    )

    assert result.diagnostics.candidate_count == 8
    assert result.diagnostics.returned_count == 5
    assert len(result.chunks) == 5
    assert [chunk.score for chunk in result.chunks] == sorted(
        (chunk.score for chunk in result.chunks),
        reverse=True,
    )


def test_filename_search_rejects_invalid_limit() -> None:
    service, _, vector_store = create_service()

    with pytest.raises(ValueError, match="greater than zero"):
        service.retrieve(
            "Summarize Behavioural_questions.docx",
            limit=0,
        )

    vector_store.search.assert_not_called()