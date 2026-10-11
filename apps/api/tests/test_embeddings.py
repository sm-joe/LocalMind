
from unittest.mock import MagicMock

import pytest

from app.services.embeddings import EmbeddingService


def create_service() -> EmbeddingService:
    return EmbeddingService(
        base_url="http://ollama:11434",
        model="nomic-embed-text",
    )


def test_embed_returns_vector() -> None:
    service = create_service()
    service.client = MagicMock()
    service.client.embed.return_value.embeddings = [
        [0.1, 0.2, 0.3]
    ]

    result = service.embed("Hello LocalRAG")

    assert result == [0.1, 0.2, 0.3]
    service.client.embed.assert_called_once_with(
        model="nomic-embed-text",
        input="Hello LocalRAG",
    )


def test_embed_rejects_empty_text() -> None:
    service = create_service()

    with pytest.raises(
        ValueError,
        match="empty text",
    ):
        service.embed("   ")


def test_embed_rejects_empty_response() -> None:
    service = create_service()
    service.client = MagicMock()
    service.client.embed.return_value.embeddings = []

    with pytest.raises(
        RuntimeError,
        match="no embeddings",
    ):
        service.embed("Hello LocalRAG")


def test_embed_many_batches_requests() -> None:
    service = create_service()
    service.client = MagicMock()

    service.client.embed.side_effect = [
        MagicMock(
            embeddings=[
                [0.1, 0.2],
                [0.3, 0.4],
            ]
        ),
        MagicMock(
            embeddings=[
                [0.5, 0.6],
            ]
        ),
    ]

    result = service.embed_many(
        ["first", "second", "third"],
        batch_size=2,
    )

    assert result == [
        [0.1, 0.2],
        [0.3, 0.4],
        [0.5, 0.6],
    ]

    assert service.client.embed.call_count == 2

    assert service.client.embed.call_args_list[0].kwargs == {
        "model": "nomic-embed-text",
        "input": ["first", "second"],
    }

    assert service.client.embed.call_args_list[1].kwargs == {
        "model": "nomic-embed-text",
        "input": ["third"],
    }


def test_embed_many_returns_empty_list_for_empty_input() -> None:
    service = create_service()
    service.client = MagicMock()

    assert service.embed_many([]) == []
    service.client.embed.assert_not_called()


def test_embed_many_rejects_blank_text() -> None:
    service = create_service()
    service.client = MagicMock()

    with pytest.raises(
        ValueError,
        match="empty text",
    ):
        service.embed_many(["first", "  "])

    service.client.embed.assert_not_called()


def test_embed_many_rejects_invalid_batch_size() -> None:
    service = create_service()
    service.client = MagicMock()

    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        service.embed_many(["first"], batch_size=0)

    service.client.embed.assert_not_called()


def test_embed_many_rejects_incomplete_batch_response() -> None:
    service = create_service()
    service.client = MagicMock()
    service.client.embed.return_value.embeddings = [
        [0.1, 0.2]
    ]

    with pytest.raises(
        RuntimeError,
        match="unexpected number",
    ):
        service.embed_many(
            ["first", "second"],
            batch_size=2,
        )
