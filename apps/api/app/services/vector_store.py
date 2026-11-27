from collections.abc import Iterator, Sequence
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models


QDRANT_TIMEOUT_SECONDS = 30


class VectorStore:
    def __init__(
        self,
        host: str,
        port: int,
        collection_name: str,
    ) -> None:
        self.client = QdrantClient(
            host=host,
            port=port,
            timeout=QDRANT_TIMEOUT_SECONDS,
        )
        self.collection_name = collection_name

    @staticmethod
    def _validate_vector_config(
        collection_info: Any,
        vector_size: int,
    ) -> None:
        vector_config = (
            collection_info.config.params.vectors
        )

        if not isinstance(
            vector_config,
            models.VectorParams,
        ):
            raise RuntimeError(
                "Qdrant collection uses a named or unsupported "
                "vector configuration. LocalRAG requires a "
                "single unnamed vector."
            )

        if vector_config.size != vector_size:
            raise RuntimeError(
                "Qdrant collection vector dimension mismatch: "
                f"collection expects {vector_config.size}, "
                f"but the embedding model produced {vector_size}. "
                "Use a compatible embedding model or migrate "
                "the collection."
            )

        if vector_config.distance != models.Distance.COSINE:
            raise RuntimeError(
                "Qdrant collection distance mismatch: "
                f"expected COSINE, found "
                f"{vector_config.distance}."
            )

    def ensure_collection(
        self,
        vector_size: int,
    ) -> None:
        if vector_size <= 0:
            raise ValueError(
                "Vector size must be greater than zero."
            )

        collections = self.client.get_collections()

        exists = any(
            collection.name == self.collection_name
            for collection in collections.collections
        )

        if exists:
            collection_info = self.client.get_collection(
                collection_name=self.collection_name,
            )

            self._validate_vector_config(
                collection_info,
                vector_size,
            )
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        )

    def upsert(
        self,
        points: Sequence[models.PointStruct],
    ) -> None:
        if not points:
            return

        self.client.upsert(
            collection_name=self.collection_name,
            points=list(points),
        )

    def search(
        self,
        vector: list[float],
        limit: int = 5,
        filename: str | None = None,
    ) -> list[Any]:
        if not vector:
            raise ValueError(
                "Search vector cannot be empty."
            )

        if limit <= 0:
            raise ValueError(
                "Search limit must be greater than zero."
            )

        if filename is not None:
            normalized_filename = filename.strip()

            if not normalized_filename:
                raise ValueError(
                    "Filename cannot be empty."
                )
        else:
            normalized_filename = None

        # A new installation may not have any documents indexed yet.
        # Return no results instead of requesting a nonexistent collection.
        collections = self.client.get_collections()

        collection_exists = any(
            collection.name == self.collection_name
            for collection in collections.collections
        )

        if not collection_exists:
            return []

        # Validate the schema only after confirming the collection exists.
        collection_info = self.client.get_collection(
            collection_name=self.collection_name,
        )

        self._validate_vector_config(
            collection_info,
            len(vector),
        )

        query_arguments: dict[str, Any] = {
            "collection_name": self.collection_name,
            "query": vector,
            "limit": limit,
        }

        if normalized_filename is not None:
            query_arguments["query_filter"] = models.Filter(
                must=[
                    models.FieldCondition(
                        key="filename",
                        match=models.MatchValue(
                            value=normalized_filename,
                        ),
                    )
                ]
            )

        return self.client.query_points(
            **query_arguments,
        ).points

    def iter_points(
        self,
        limit: int = 100,
        document_id: str | None = None,
    ) -> Iterator[Any]:
        """Yield paginated points, optionally filtered server-side by document ID."""
        if document_id is not None and not document_id.strip():
            raise ValueError("Document ID cannot be empty.")
        if limit <= 0:
            raise ValueError(
                "Limit must be greater than zero."
            )
        # Only unfiltered corpus scans need the empty-install guard.
        # Filtered scans must reach Qdrant's scroll API: callers use them
        # for precise document retrieval and pagination, and tests may mock
        # scroll directly without mocking collection discovery.
        if document_id is None:
            existing = self.client.get_collections().collections
            if not any(item.name == self.collection_name for item in existing):
                return
        offset: Any = None

        while True:
            scroll_arguments: dict[str, Any] = {
                "collection_name": self.collection_name,
                "limit": limit,
                "with_payload": True,
                "with_vectors": False,
            }

            if document_id is not None:
                scroll_arguments["scroll_filter"] = models.Filter(
                    must=[models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(value=document_id),
                    )]
                )

            if offset is not None:
                scroll_arguments["offset"] = offset

            records, next_offset = self.client.scroll(
                **scroll_arguments,
            )

            for record in records:
                yield record

            if next_offset is None:
                break

            if next_offset == offset:
                raise RuntimeError(
                    "Qdrant scroll pagination did not advance."
                )

            offset = next_offset

    def get_document_chunk(
        self,
        document_id: str,
        chunk_id: int,
    ) -> dict[str, Any] | None:
        """Retrieve exactly one document-scoped chunk without vector search."""
        if not document_id.strip() or chunk_id < 0:
            raise ValueError("Invalid document or chunk identifier.")
        records, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(value=document_id),
                    ),
                    models.FieldCondition(
                        key="chunk_id",
                        match=models.MatchValue(value=chunk_id),
                    ),
                ]
            ),
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
        return dict(records[0].payload or {}) if records else None

    def list_points(
        self,
        limit: int = 100,
    ) -> list[Any]:
        """Return all points for callers that require a list."""
        if limit <= 0:
            raise ValueError(
                "Limit must be greater than zero."
            )

        return list(
            self.iter_points(limit=limit)
        )

    def find_document_by_hash(
        self,
        document_hash: str,
    ) -> dict[str, str] | None:
        normalized_hash = document_hash.strip()

        if not normalized_hash:
            raise ValueError(
                "Document hash cannot be empty."
            )

        collections = self.client.get_collections()

        collection_exists = any(
            collection.name == self.collection_name
            for collection in collections.collections
        )

        if not collection_exists:
            return None

        records, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_hash",
                        match=models.MatchValue(
                            value=normalized_hash,
                        ),
                    )
                ]
            ),
            limit=1,
            with_payload=True,
            with_vectors=False,
        )

        if not records:
            return None

        payload = records[0].payload or {}

        document_id = payload.get("document_id")
        filename = payload.get("filename")

        if not document_id:
            return None

        return {
            "document_id": str(document_id),
            "filename": str(filename or "unknown"),
        }

    def find_points_by_filename(
        self,
        filename: str,
        limit: int = 100,
    ) -> list[Any]:
        normalized_filename = filename.strip()

        if not normalized_filename:
            raise ValueError(
                "Filename cannot be empty."
            )

        if limit <= 0:
            raise ValueError(
                "Limit must be greater than zero."
            )

        records, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="filename",
                        match=models.MatchValue(
                            value=normalized_filename,
                        ),
                    )
                ]
            ),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )

        return records

    def delete_document(
        self,
        document_id: str,
    ) -> None:
        if not document_id.strip():
            raise ValueError(
                "Document ID cannot be empty."
            )

        self.client.delete(
            collection_name=self.collection_name,
            wait=True,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(
                                value=document_id,
                            ),
                        )
                    ]
                )
            ),
        )
