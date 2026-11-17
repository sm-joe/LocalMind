import hashlib
import logging
import uuid
from pathlib import Path

from qdrant_client.http import models

from app.services.chunker import chunk_text
from app.services.embeddings import EmbeddingService
from app.services.document_registry import DocumentIndexRegistry, DocumentIndexInProgress
from app.services.parser import parse_document
from app.services.vector_store import VectorStore

logger = logging.getLogger(__name__)


class IngestionResult:
    def __init__(
        self,
        document_id: str,
        filename: str,
        chunks: int,
        already_indexed: bool = False,
    ) -> None:
        self.document_id = document_id
        self.filename = filename
        self.chunks = chunks
        self.already_indexed = already_indexed


class IngestionService:
    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
        registry: DocumentIndexRegistry | None = None,
    ) -> None:
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.registry = registry

    def ingest(
        self,
        file_path: str | Path,
        content_type: str,
        original_filename: str | None = None,
    ) -> IngestionResult:
        path = Path(file_path)
        if self.registry is None:
            return self._ingest_document(path, content_type, original_filename)

        if not path.is_file():
            raise ValueError("Document file does not exist.")
        document_hash = self._calculate_hash(path)
        filename = original_filename or path.name
        claim = self.registry.acquire(document_hash, filename, content_type)
        if claim.status == "ready":
            return IngestionResult(
                document_id=claim.document_id,
                filename=claim.filename,
                chunks=0,
                already_indexed=True,
            )
        try:
            if claim.previous_document_id is not None:
                # Stale/failed attempt may have left partial Qdrant chunks.
                self.vector_store.delete_document(claim.previous_document_id)
            result = self._ingest_document(
                path, content_type, original_filename,
                document_id=claim.document_id,
            )
            self.registry.complete(
                claim, result.document_id, result.filename, result.chunks
            )
            return result
        except Exception:
            try:
                self.registry.fail(claim)
            except Exception:
                logger.exception(
                    "Could not persist failed document indexing for hash=%s",
                    document_hash,
                )
            raise

    def _ingest_document(
        self,
        file_path: str | Path,
        content_type: str,
        original_filename: str | None = None,
        document_id: str | None = None,
    ) -> IngestionResult:
        path = Path(file_path)

        if not path.is_file():
            raise ValueError(
                "Document file does not exist."
            )

        document_hash = self._calculate_hash(path)

        existing_document = (
            self.vector_store.find_document_by_hash(
                document_hash
            )
        )

        filename = (
            original_filename
            if original_filename
            else path.name
        )

        if self._is_existing_document(existing_document):
            return IngestionResult(
                document_id=existing_document["document_id"],
                filename=existing_document["filename"],
                chunks=0,
                already_indexed=True,
            )

        parsed = parse_document(
            path,
            content_type,
        )

        chunks = chunk_text(
            parsed.text,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        if not chunks:
            raise ValueError(
                "Document produced no chunks."
            )

        chunk_texts = [
            chunk.text
            for chunk in chunks
        ]

        # Prefer batch embedding when supported.
        # Fall back to individual embeddings for compatible
        # implementations and existing test doubles.
        embed_many = getattr(
            self.embedding_service,
            "embed_many",
            None,
        )

        if callable(embed_many):
            embeddings = embed_many(chunk_texts)
        else:
            embeddings = [
                self.embedding_service.embed(text)
                for text in chunk_texts
            ]

        if len(embeddings) != len(chunks):
            raise RuntimeError(
                "Embedding count does not match chunk count."
            )

        vector_size = len(embeddings[0])

        if vector_size == 0:
            raise RuntimeError(
                "Embedding vectors must not be empty."
            )

        if any(
            len(embedding) != vector_size
            for embedding in embeddings
        ):
            raise RuntimeError(
                "Embedding vectors have inconsistent dimensions."
            )

        self.vector_store.ensure_collection(
            vector_size=vector_size
        )

        document_id = document_id or str(uuid.uuid4())

        points: list[models.PointStruct] = []

        for chunk, embedding in zip(
            chunks,
            embeddings,
            strict=True,
        ):
            points.append(
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={
                        "document_id": document_id,
                        "document_hash": document_hash,
                        "filename": filename,
                        "content_type": parsed.content_type,
                        "chunk_id": chunk.chunk_id,
                        "text": chunk.text,
                        "start_char": chunk.start_char,
                        "end_char": chunk.end_char,
                    },
                )
            )

        # A transport/server error may occur after Qdrant has persisted some
        # points. Never leave a partially indexed document that could be
        # mistaken for a successfully completed duplicate on the next upload.
        try:
            self.vector_store.upsert(points)
        except Exception:
            try:
                self.vector_store.delete_document(document_id)
            except Exception:
                logger.exception(
                    "Could not roll back partial indexing for document_id=%s",
                    document_id,
                )
            raise

        return IngestionResult(
            document_id=document_id,
            filename=filename,
            chunks=len(chunks),
            already_indexed=False,
        )

    @staticmethod
    def _is_existing_document(
        document: object,
    ) -> bool:
        if not isinstance(document, dict):
            return False

        document_id = document.get("document_id")
        filename = document.get("filename")

        return bool(
            isinstance(document_id, str)
            and document_id.strip()
            and isinstance(filename, str)
            and filename.strip()
        )

    @staticmethod
    def _calculate_hash(
        path: Path,
    ) -> str:
        digest = hashlib.sha256()

        with path.open("rb") as document:
            for block in iter(
                lambda: document.read(1024 * 1024),
                b"",
            ):
                digest.update(block)

        return digest.hexdigest()