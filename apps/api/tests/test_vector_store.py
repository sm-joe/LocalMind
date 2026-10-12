from unittest.mock import MagicMock, call

import pytest
from qdrant_client.http import models

from app.services.vector_store import VectorStore


def create_store() -> VectorStore:
    store = VectorStore(
        host="qdrant",
        port=6333,
        collection_name="test_collection",
    )

    store.client = MagicMock()

    # Default behavior: the collection exists.
    existing_collection = MagicMock()
    existing_collection.name = "test_collection"

    store.client.get_collections.return_value.collections = [
        existing_collection,
    ]

    # Default schema: three-dimensional cosine vectors.
    store.client.get_collection.return_value.config.params.vectors = (
        models.VectorParams(
            size=3,
            distance=models.Distance.COSINE,
        )
    )

    return store


def test_ensure_collection_creates_missing_collection() -> None:
    store = create_store()
    store.client.get_collections.return_value.collections = []

    store.ensure_collection(vector_size=3)

    store.client.create_collection.assert_called_once()

    kwargs = store.client.create_collection.call_args.kwargs

    assert kwargs["collection_name"] == "test_collection"

    vector_config = kwargs["vectors_config"]

    assert vector_config.size == 3
    assert vector_config.distance == models.Distance.COSINE


def test_ensure_collection_rejects_invalid_vector_size() -> None:
    store = create_store()

    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        store.ensure_collection(vector_size=0)

    store.client.get_collections.assert_not_called()
    store.client.create_collection.assert_not_called()


def test_ensure_collection_validates_existing_collection() -> None:
    store = create_store()
    existing = MagicMock()
    existing.name = "test_collection"

    store.client.get_collections.return_value.collections = [
        existing,
    ]

    store.ensure_collection(vector_size=3)

    store.client.get_collection.assert_called_once_with(
        collection_name="test_collection",
    )
    store.client.create_collection.assert_not_called()


def test_ensure_collection_rejects_dimension_mismatch() -> None:
    store = create_store()
    existing = MagicMock()
    existing.name = "test_collection"

    store.client.get_collections.return_value.collections = [
        existing,
    ]
    store.client.get_collection.return_value.config.params.vectors = (
        models.VectorParams(
            size=768,
            distance=models.Distance.COSINE,
        )
    )

    with pytest.raises(
        RuntimeError,
        match="vector dimension mismatch",
    ):
        store.ensure_collection(vector_size=3)

    store.client.create_collection.assert_not_called()


def test_ensure_collection_rejects_distance_mismatch() -> None:
    store = create_store()
    existing = MagicMock()
    existing.name = "test_collection"

    store.client.get_collections.return_value.collections = [
        existing,
    ]
    store.client.get_collection.return_value.config.params.vectors = (
        models.VectorParams(
            size=3,
            distance=models.Distance.DOT,
        )
    )

    with pytest.raises(
        RuntimeError,
        match="distance mismatch",
    ):
        store.ensure_collection(vector_size=3)

    store.client.create_collection.assert_not_called()


def test_ensure_collection_rejects_named_vector_configuration() -> None:
    store = create_store()
    existing = MagicMock()
    existing.name = "test_collection"

    store.client.get_collections.return_value.collections = [
        existing,
    ]
    store.client.get_collection.return_value.config.params.vectors = {
        "dense": models.VectorParams(
            size=3,
            distance=models.Distance.COSINE,
        )
    }

    with pytest.raises(
        RuntimeError,
        match="named or unsupported vector configuration",
    ):
        store.ensure_collection(vector_size=3)

    store.client.create_collection.assert_not_called()


def test_upsert_points() -> None:
    store = create_store()
    point = models.PointStruct(
        id="chunk-1",
        vector=[0.1, 0.2, 0.3],
        payload={
            "text": "Hello LocalRAG",
        },
    )

    store.upsert([point])

    store.client.upsert.assert_called_once_with(
        collection_name="test_collection",
        points=[point],
    )


def test_upsert_empty_points_does_nothing() -> None:
    store = create_store()

    store.upsert([])

    store.client.upsert.assert_not_called()


def test_search_rejects_empty_vector() -> None:
    store = create_store()

    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        store.search([])

    store.client.get_collection.assert_not_called()
    store.client.query_points.assert_not_called()


def test_search_rejects_invalid_limit() -> None:
    store = create_store()

    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        store.search(
            [0.1, 0.2, 0.3],
            limit=0,
        )

    store.client.get_collection.assert_not_called()
    store.client.query_points.assert_not_called()


def test_search_returns_points() -> None:
    store = create_store()
    points = [
        MagicMock(),
        MagicMock(),
    ]

    store.client.query_points.return_value.points = points

    result = store.search(
        [0.1, 0.2, 0.3],
        limit=5,
    )

    assert result == points

    store.client.get_collection.assert_called_once_with(
        collection_name="test_collection",
    )
    store.client.query_points.assert_called_once_with(
        collection_name="test_collection",
        query=[0.1, 0.2, 0.3],
        limit=5,
    )


def test_search_rejects_dimension_mismatch() -> None:
    store = create_store()
    store.client.get_collection.return_value.config.params.vectors = (
        models.VectorParams(
            size=768,
            distance=models.Distance.COSINE,
        )
    )

    with pytest.raises(
        RuntimeError,
        match="vector dimension mismatch",
    ):
        store.search([0.1, 0.2, 0.3])

    store.client.query_points.assert_not_called()


def test_search_rejects_distance_mismatch() -> None:
    store = create_store()
    store.client.get_collection.return_value.config.params.vectors = (
        models.VectorParams(
            size=3,
            distance=models.Distance.DOT,
        )
    )

    with pytest.raises(
        RuntimeError,
        match="distance mismatch",
    ):
        store.search([0.1, 0.2, 0.3])

    store.client.query_points.assert_not_called()


def test_search_filters_by_filename() -> None:
    store = create_store()
    points = [MagicMock()]
    store.client.query_points.return_value.points = points

    result = store.search(
        vector=[0.1, 0.2, 0.3],
        limit=5,
        filename=" report.pdf ",
    )

    assert result == points

    kwargs = store.client.query_points.call_args.kwargs

    assert kwargs["collection_name"] == "test_collection"
    assert kwargs["query"] == [0.1, 0.2, 0.3]
    assert kwargs["limit"] == 5

    query_filter = kwargs["query_filter"]

    assert len(query_filter.must) == 1
    assert query_filter.must[0].key == "filename"
    assert query_filter.must[0].match.value == "report.pdf"


def test_search_rejects_blank_filename() -> None:
    store = create_store()

    with pytest.raises(
        ValueError,
        match="Filename cannot be empty",
    ):
        store.search(
            vector=[0.1, 0.2, 0.3],
            filename="   ",
        )

    store.client.query_points.assert_not_called()


def test_list_points_returns_records_from_multiple_pages() -> None:
    store = create_store()
    first_page = [
        MagicMock(id="point-1"),
        MagicMock(id="point-2"),
    ]
    second_page = [
        MagicMock(id="point-3"),
    ]

    store.client.scroll.side_effect = [
        (first_page, "next-page"),
        (second_page, None),
    ]

    result = store.list_points(limit=2)

    assert result == first_page + second_page
    assert [point.id for point in result] == [
        "point-1",
        "point-2",
        "point-3",
    ]

    assert store.client.scroll.call_args_list == [
        call(
            collection_name="test_collection",
            limit=2,
            with_payload=True,
            with_vectors=False,
        ),
        call(
            collection_name="test_collection",
            limit=2,
            with_payload=True,
            with_vectors=False,
            offset="next-page",
        ),
    ]

def test_iter_points_yields_records_page_by_page() -> None:
    store = create_store()

    first_page = [
        MagicMock(id="point-1"),
        MagicMock(id="point-2"),
    ]
    second_page = [
        MagicMock(id="point-3"),
    ]

    store.client.scroll.side_effect = [
        (first_page, "next-page"),
        (second_page, None),
    ]

    points = store.iter_points(limit=2)

    assert next(points) is first_page[0]

    # The first page has been fetched, but the second has not.
    assert store.client.scroll.call_count == 1

    assert next(points) is first_page[1]
    assert store.client.scroll.call_count == 1

    assert next(points) is second_page[0]
    assert store.client.scroll.call_count == 2

    with pytest.raises(StopIteration):
        next(points)

    assert store.client.scroll.call_count == 2


def test_list_points_stops_when_no_next_page_exists() -> None:
    store = create_store()
    points = [MagicMock(id="point-1")]

    store.client.scroll.return_value = (points, None)

    result = store.list_points(limit=100)

    assert result == points

    store.client.scroll.assert_called_once_with(
        collection_name="test_collection",
        limit=100,
        with_payload=True,
        with_vectors=False,
    )


def test_list_points_rejects_invalid_limit() -> None:
    store = create_store()

    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        store.list_points(limit=0)

    store.client.scroll.assert_not_called()


def test_list_points_rejects_non_advancing_cursor() -> None:
    store = create_store()
    points = [MagicMock(id="point-1")]

    store.client.scroll.side_effect = [
        (points, "next-page"),
        (points, "next-page"),
    ]

    with pytest.raises(
        RuntimeError,
        match="pagination did not advance",
    ):
        store.list_points(limit=1)


def test_find_document_by_hash_rejects_empty_hash() -> None:
    store = create_store()

    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        store.find_document_by_hash("   ")

    store.client.scroll.assert_not_called()


def test_find_points_by_filename_rejects_empty_filename() -> None:
    store = create_store()

    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        store.find_points_by_filename("   ")

    store.client.scroll.assert_not_called()


def test_delete_document_rejects_empty_document_id() -> None:
    store = create_store()

    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        store.delete_document("   ")

    store.client.delete.assert_not_called()

def test_search_returns_empty_when_collection_does_not_exist() -> None:
    store = create_store()
    store.client.get_collections.return_value.collections = []

    result = store.search(
        vector=[0.1, 0.2, 0.3],
    )

    assert result == []
    store.client.get_collection.assert_not_called()
    store.client.query_points.assert_not_called()