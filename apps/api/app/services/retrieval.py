import re
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from app.services.embeddings import EmbeddingService
from app.services.vector_store import VectorStore


@dataclass(frozen=True)
class RetrievedChunk:
    document_id: str
    filename: str
    content_type: str
    chunk_id: int
    text: str
    start_char: int
    end_char: int
    score: float


@dataclass(frozen=True)
class RetrievalDiagnostics:
    requested_limit: int
    candidate_count: int
    returned_count: int
    filtered_count: int
    top_candidate_score: float | None
    bottom_candidate_score: float | None
    top_score: float | None
    bottom_score: float | None
    has_context: bool


@dataclass(frozen=True)
class RetrievalPerformance:
    embedding_ms: float
    search_ms: float
    total_ms: float


@dataclass(frozen=True)
class RetrievalResult:
    chunks: list[RetrievedChunk]
    diagnostics: RetrievalDiagnostics
    performance: RetrievalPerformance


class RetrievalService:
    SUPPORTED_FILENAME_EXTENSIONS = (
        ".pdf",
        ".docx",
        ".txt",
        ".md",
    )

    FILENAME_PATTERN = re.compile(
        r"""
        (?:
            ["']([^"']+\.(?:pdf|docx|txt|md))
            ["']
        )
        |
        (?:
            ([^\s"'<>]+\.(?:pdf|docx|txt|md))
        )
        """,
        re.IGNORECASE | re.VERBOSE,
    )

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> None:
        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero"
            )

        if not 0.0 <= score_threshold <= 1.0:
            raise ValueError(
                "score_threshold must be between "
                "0.0 and 1.0"
            )

        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.top_k = top_k
        self.score_threshold = score_threshold

    def retrieve(
        self,
        query: str,
        limit: int | None = None,
    ) -> RetrievalResult:
        normalized_query = query.strip()

        if not normalized_query:
            raise ValueError(
                "Query cannot be empty"
            )

        search_limit = (
            self.top_k
            if limit is None
            else limit
        )

        if search_limit <= 0:
            raise ValueError(
                "Search limit must be greater than zero"
            )

        total_start = perf_counter()

        explicit_filename = self._extract_filename(
            normalized_query
        )

        if explicit_filename:
            return self._retrieve_by_filename(
                query=normalized_query,
                filename=explicit_filename,
                limit=search_limit,
                total_start=total_start,
            )

        embedding_start = perf_counter()

        query_embedding = (
            self.embedding_service.embed(
                normalized_query
            )
        )

        embedding_ms = (
            perf_counter() - embedding_start
        ) * 1000

        search_start = perf_counter()

        candidate_limit = max(search_limit * 4, search_limit)

        results = self.vector_store.search(
            vector=query_embedding,
            limit=candidate_limit,
        )

        search_ms = (
            perf_counter() - search_start
        ) * 1000

        return self._build_result(
            results=results,
            limit=search_limit,
            embedding_ms=embedding_ms,
            search_ms=search_ms,
            total_start=total_start,
        )

    def retrieve_document(
        self,
        query: str,
        limit: int | None = None,
    ) -> RetrievalResult:
        """Retrieve every indexed chunk for the best-matching document.

        This path is intended for whole-document tasks such as summaries. It
        uses semantic search only to identify the target document, then reads
        that document's chunks in source order. The normal similarity threshold
        is deliberately not applied to individual chunks: summary coverage is
        not the same as per-chunk semantic relevance.
        """
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("Query cannot be empty")

        search_limit = self.top_k if limit is None else limit
        if search_limit <= 0:
            raise ValueError("Search limit must be greater than zero")

        total_start = perf_counter()
        embedding_start = perf_counter()
        query_embedding = self.embedding_service.embed(normalized_query)
        embedding_ms = (perf_counter() - embedding_start) * 1000

        search_start = perf_counter()
        explicit_filename = self._extract_filename(normalized_query)
        search_arguments: dict[str, Any] = {
            "vector": query_embedding,
            "limit": max(search_limit * 4, 20),
        }
        if explicit_filename:
            search_arguments["filename"] = explicit_filename
        candidates = self.vector_store.search(**search_arguments)
        search_ms = (perf_counter() - search_start) * 1000

        candidate_scores = [float(point.score) for point in candidates]
        target_payload = next(
            (
                point.payload or {}
                for point in candidates
                if self._build_chunk(point.payload or {}, float(point.score))
                is not None
            ),
            None,
        )

        document_chunks: list[RetrievedChunk] = []
        if target_payload is not None:
            target_document_id = str(target_payload["document_id"])
            candidate_chunk_scores = {
                (str(point.payload.get("document_id")), int(point.payload.get("chunk_id"))): float(point.score)
                for point in candidates
                if point.payload
                and point.payload.get("document_id") is not None
                and point.payload.get("chunk_id") is not None
            }
            seen_chunk_ids: set[int] = set()
            for point in self.vector_store.iter_points(document_id=target_document_id):
                payload = point.payload or {}
                if str(payload.get("document_id", "")) != target_document_id:
                    continue
                chunk_id_value = payload.get("chunk_id")
                if chunk_id_value is None:
                    continue
                chunk_id = int(chunk_id_value)
                if chunk_id in seen_chunk_ids:
                    continue
                seen_chunk_ids.add(chunk_id)
                score = candidate_chunk_scores.get(
                    (target_document_id, chunk_id), 0.0
                )
                chunk = self._build_chunk(payload=payload, score=score)
                if chunk is not None:
                    document_chunks.append(chunk)

        document_chunks.sort(key=lambda chunk: chunk.chunk_id)
        candidate_scores_for_document = [
            score
            for (document_id, _), score in (
                {
                    (str(point.payload.get("document_id")), int(point.payload.get("chunk_id"))): float(point.score)
                    for point in candidates
                    if point.payload
                    and point.payload.get("document_id") is not None
                    and point.payload.get("chunk_id") is not None
                }
            ).items()
            if document_chunks and document_id == document_chunks[0].document_id
        ]
        diagnostics = RetrievalDiagnostics(
            requested_limit=len(document_chunks),
            candidate_count=len(candidates),
            returned_count=len(document_chunks),
            filtered_count=max(0, len(candidates) - len(candidate_scores_for_document)),
            top_candidate_score=max(candidate_scores) if candidate_scores else None,
            bottom_candidate_score=min(candidate_scores) if candidate_scores else None,
            top_score=max(candidate_scores_for_document) if candidate_scores_for_document else None,
            bottom_score=min(candidate_scores_for_document) if candidate_scores_for_document else None,
            has_context=bool(document_chunks),
        )
        performance = RetrievalPerformance(
            embedding_ms=round(embedding_ms, 2),
            search_ms=round(search_ms, 2),
            total_ms=round((perf_counter() - total_start) * 1000, 2),
        )
        return RetrievalResult(
            chunks=document_chunks,
            diagnostics=diagnostics,
            performance=performance,
        )

    def _retrieve_by_filename(
        self,
        query: str,
        filename: str,
        limit: int,
        total_start: float,
    ) -> RetrievalResult:
        embedding_start = perf_counter()

        query_embedding = (
            self.embedding_service.embed(query)
        )

        embedding_ms = (
            perf_counter() - embedding_start
        ) * 1000

        search_start = perf_counter()

        results = self.vector_store.search(
            vector=query_embedding,
            limit=limit,
            filename=filename,
        )

        search_ms = (
            perf_counter() - search_start
        ) * 1000

        return self._build_result(
            results=results,
            limit=limit,
            embedding_ms=embedding_ms,
            search_ms=search_ms,
            total_start=total_start,
        )

    def _build_result(
        self,
        results: list[Any],
        limit: int,
        embedding_ms: float,
        search_ms: float,
        total_start: float,
    ) -> RetrievalResult:
        candidate_scores = [
            float(result.score)
            for result in results
        ]

        retrieved_chunks: list[RetrievedChunk] = []
        filtered_count = 0

        for result in results:
            score = float(result.score)

            if score < self.score_threshold:
                filtered_count += 1
                continue

            payload = result.payload or {}

            chunk = self._build_chunk(
                payload=payload,
                score=score,
            )

            if chunk is None:
                filtered_count += 1
                continue

            retrieved_chunks.append(chunk)

        retrieved_chunks.sort(
            key=lambda chunk: chunk.score,
            reverse=True,
        )

        retrieved_chunks = retrieved_chunks[:limit]

        scores = [
            chunk.score
            for chunk in retrieved_chunks
        ]

        diagnostics = RetrievalDiagnostics(
            requested_limit=limit,
            candidate_count=len(results),
            returned_count=len(retrieved_chunks),
            filtered_count=filtered_count,
            top_candidate_score=(
                max(candidate_scores)
                if candidate_scores
                else None
            ),
            bottom_candidate_score=(
                min(candidate_scores)
                if candidate_scores
                else None
            ),
            top_score=(
                max(scores)
                if scores
                else None
            ),
            bottom_score=(
                min(scores)
                if scores
                else None
            ),
            has_context=bool(retrieved_chunks),
        )

        performance = RetrievalPerformance(
            embedding_ms=round(embedding_ms, 2),
            search_ms=round(search_ms, 2),
            total_ms=round(
                (perf_counter() - total_start) * 1000,
                2,
            ),
        )

        return RetrievalResult(
            chunks=retrieved_chunks,
            diagnostics=diagnostics,
            performance=performance,
        )

    @classmethod
    def _extract_filename(
        cls,
        query: str,
    ) -> str | None:
        matches = cls.FILENAME_PATTERN.findall(query)

        if not matches:
            return None

        for quoted, unquoted in matches:
            filename = (
                quoted.strip()
                if quoted
                else unquoted.strip()
            )

            if cls._is_supported_filename(filename):
                return filename

        return None

    @classmethod
    def _is_supported_filename(
        cls,
        filename: str,
    ) -> bool:
        return filename.lower().endswith(
            cls.SUPPORTED_FILENAME_EXTENSIONS
        )

    @staticmethod
    def _build_chunk(
        payload: dict[str, Any],
        score: float,
    ) -> RetrievedChunk | None:
        required_fields = {
            "document_id",
            "filename",
            "content_type",
            "chunk_id",
            "text",
            "start_char",
            "end_char",
        }

        if not required_fields.issubset(payload):
            return None

        return RetrievedChunk(
            document_id=str(payload["document_id"]),
            filename=str(payload["filename"]),
            content_type=str(payload["content_type"]),
            chunk_id=int(payload["chunk_id"]),
            text=str(payload["text"]),
            start_char=int(payload["start_char"]),
            end_char=int(payload["end_char"]),
            score=score,
        )