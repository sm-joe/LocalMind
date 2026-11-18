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
    STOPWORDS = frozenset({
        "the", "and", "for", "with", "from", "this", "that", "what",
        "when", "where", "which", "does", "are", "was", "how",
        "can", "into", "about", "please", "your",
    })

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
        lexical_weight: float = 0.0,
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
        if not 0.0 <= lexical_weight <= 1.0:
            raise ValueError("lexical_weight must be between 0 and 1.")
        self.lexical_weight = lexical_weight

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
            query=normalized_query,
            limit=search_limit,
            embedding_ms=embedding_ms,
            search_ms=search_ms,
            total_start=total_start,
        )

    def retrieve_named_entity(
        self, query: str, entity: str, limit: int = 4
    ) -> RetrievalResult:
        """Look for an explicitly named subject in all indexed document text.

        Semantic top-k may miss a rare subject entirely. Literal matches
        are a stronger first step for 'What is Amazon Bedrock?' and prevent
        claiming a source is absent based on only three vector hits.
        """
        if not entity.strip() or limit <= 0:
            raise ValueError("Entity and positive limit are required.")
        start = perf_counter()
        needle = re.compile(
            rf"(?<![a-z0-9]){re.escape(entity.strip())}(?![a-z0-9])",
            re.IGNORECASE,
        )
        matches: list[RetrievedChunk] = []
        scanned = 0
        seen: set[tuple[str, int]] = set()
        for point in self.vector_store.iter_points():
            if scanned >= 10000:
                break
            scanned += 1
            chunk = self._build_chunk(point.payload or {}, 0.0)
            if chunk is None or (chunk.document_id, chunk.chunk_id) in seen:
                continue
            seen.add((chunk.document_id, chunk.chunk_id))
            if needle.search(chunk.text):
                matches.append(chunk)
        # Prefer definitional sentences over incidental mentions. If the
        # term appears late in a chunk, RAG retains that nearby text.
        def rank(chunk: RetrievedChunk) -> tuple[int, int]:
            match = needle.search(chunk.text)
            text_after = chunk.text[match.start():match.end() + 120] if match else ""
            definition = bool(re.search(
                r"\b(?:is|provides|offers|enables|allows|service|platform)\b",
                text_after, re.IGNORECASE
            ))
            return (int(definition), -chunk.chunk_id)
        matches.sort(key=rank, reverse=True)
        matches = matches[:limit]
        elapsed = round((perf_counter() - start) * 1000, 2)
        return RetrievalResult(
            chunks=matches,
            diagnostics=RetrievalDiagnostics(
                requested_limit=limit, candidate_count=scanned,
                returned_count=len(matches),
                filtered_count=max(0, scanned-len(matches)),
                top_candidate_score=None, bottom_candidate_score=None,
                top_score=None, bottom_score=None,
                has_context=bool(matches),
            ),
            performance=RetrievalPerformance(
                embedding_ms=0.0, search_ms=elapsed, total_ms=elapsed
            ),
        )

    def retrieve_comparison(
        self, query: str, left: str, right: str, limit: int = 4
    ) -> RetrievalResult:
        """Find evidence for BOTH subjects of an explicit comparison.

        Small document libraries can scatter definitions across distant chunks.
        A single embedding of the whole comparison often retrieves only one.
        Scan indexed text for both requested terms, retaining only real chunks.
        """
        if limit <= 0:
            raise ValueError("Search limit must be greater than zero.")
        start = perf_counter()
        def contains(term: str, text: str) -> bool:
            return bool(re.search(
                rf"(?<![a-z0-9]){re.escape(term.casefold())}(?![a-z0-9])",
                text.casefold(),
            ))
        both: list[RetrievedChunk] = []
        left_only: list[RetrievedChunk] = []
        right_only: list[RetrievedChunk] = []
        seen: set[tuple[str, int]] = set()
        examined = 0
        for point in self.vector_store.iter_points():
            if examined >= 10000:
                break
            examined += 1
            chunk = self._build_chunk(point.payload or {}, 0.0)
            if chunk is None or (chunk.document_id, chunk.chunk_id) in seen:
                continue
            seen.add((chunk.document_id, chunk.chunk_id))
            has_left = contains(left, chunk.text)
            has_right = contains(right, chunk.text)
            if has_left and has_right:
                both.append(chunk)
            elif has_left:
                left_only.append(chunk)
            elif has_right:
                right_only.append(chunk)
        # Preserve evidence for BOTH subjects rather than allowing the
        # more frequent name to crowd out the other.
        selected: list[RetrievedChunk] = []
        # Lead with a definition for each side (when available), so a
        # comparison prompt and its citation mapping are naturally balanced.
        for group in (left_only, right_only):
            if group and len(selected) < limit:
                selected.append(group[0])
        for chunk in both:
            if len(selected) >= limit:
                break
            selected.append(chunk)
        for group in (left_only[1:], right_only[1:]):
            for chunk in group:
                if len(selected) >= limit:
                    break
                selected.append(chunk)
        # An answer would be deceptive if the document never mentions one
        # of the requested concepts. Do not ask Ollama to guess that half.
        if not any(contains(left, c.text) for c in selected) or not any(
            contains(right, c.text) for c in selected
        ):
            selected = []
        elapsed = round((perf_counter() - start) * 1000, 2)
        return RetrievalResult(
            chunks=selected,
            diagnostics=RetrievalDiagnostics(
                requested_limit=limit,
                candidate_count=examined,
                returned_count=len(selected),
                filtered_count=max(0, examined - len(selected)),
                top_candidate_score=None,
                bottom_candidate_score=None,
                top_score=None,
                bottom_score=None,
                has_context=bool(selected),
            ),
            performance=RetrievalPerformance(
                embedding_ms=0.0, search_ms=elapsed, total_ms=elapsed
            ),
        )

    def retrieve_document_by_id(
        self,
        document_id: str,
        query: str,
    ) -> RetrievalResult:
        """Read an explicitly selected document without semantic guessing.

        Chunk IDs are stable citation references; every result is verified
        to belong to the selected document before reaching the model.
        """
        if not document_id.strip():
            raise ValueError("Document ID cannot be empty.")
        start = perf_counter()
        chunks: list[RetrievedChunk] = []
        seen: set[int] = set()
        for point in self.vector_store.iter_points(document_id=document_id):
            payload = point.payload or {}
            if str(payload.get("document_id", "")) != document_id:
                continue
            chunk = self._build_chunk(payload, 1.0)
            if chunk is None or chunk.chunk_id in seen:
                continue
            seen.add(chunk.chunk_id)
            chunks.append(chunk)
        chunks.sort(key=lambda item: item.chunk_id)
        diagnostics = RetrievalDiagnostics(
            requested_limit=len(chunks),
            candidate_count=len(chunks),
            returned_count=len(chunks),
            filtered_count=0,
            top_candidate_score=1.0 if chunks else None,
            bottom_candidate_score=1.0 if chunks else None,
            top_score=1.0 if chunks else None,
            bottom_score=1.0 if chunks else None,
            has_context=bool(chunks),
        )
        return RetrievalResult(
            chunks=chunks,
            diagnostics=diagnostics,
            performance=RetrievalPerformance(
                embedding_ms=0.0,
                search_ms=round((perf_counter() - start) * 1000, 2),
                total_ms=round((perf_counter() - start) * 1000, 2),
            ),
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

        # A generic summary has no semantic target. Scoring the sentence
        # "summarize the uploaded document" against chunks is unreliable:
        # index the actual corpus first, then cover all available files.
        # Filename-specific requests retain their normal targeted retrieval.
        explicit_filename = self._extract_filename(normalized_query)
        if explicit_filename is None:
            return self._retrieve_indexed_corpus()

        total_start = perf_counter()
        embedding_start = perf_counter()
        query_embedding = self.embedding_service.embed(normalized_query)
        embedding_ms = (perf_counter() - embedding_start) * 1000

        search_start = perf_counter()
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

    def _retrieve_indexed_corpus(self) -> RetrievalResult:
        """Cover indexed files for generic summary/extraction requests.

        Read Qdrant payloads rather than guessing the file from a generic
        question embedding. Round-robin sources so a long first PDF cannot
        crowd out other uploaded documents. The RAG layer still enforces
        the final prompt budget.
        """
        start = perf_counter()
        by_document: dict[str, dict[int, RetrievedChunk]] = {}
        # The store returns payloads only (no vectors) via paginated scroll.
        # Avoid an unbounded in-memory corpus scan for very large installs.
        for index, point in enumerate(self.vector_store.iter_points()):
            if index >= 10000:
                break
            payload = point.payload or {}
            chunk = self._build_chunk(payload, 0.0)
            if chunk is not None:
                by_document.setdefault(chunk.document_id, {}).setdefault(
                    chunk.chunk_id, chunk
                )
        groups = [
            sorted(parts.values(), key=lambda chunk: chunk.chunk_id)
            for _, parts in sorted(by_document.items())
        ]
        # Interleave chunks across documents; preserves each document's
        # internal chunk order and gives every file a chance in the prompt.
        chunks: list[RetrievedChunk] = []
        for i in range(max((len(group) for group in groups), default=0)):
            for group in groups:
                if i < len(group):
                    chunks.append(group[i])
        elapsed = round((perf_counter() - start) * 1000, 2)
        return RetrievalResult(
            chunks=chunks,
            diagnostics=RetrievalDiagnostics(
                requested_limit=len(chunks),
                candidate_count=len(chunks),
                returned_count=len(chunks),
                filtered_count=0,
                top_candidate_score=None,
                bottom_candidate_score=None,
                top_score=None,
                bottom_score=None,
                has_context=bool(chunks),
            ),
            performance=RetrievalPerformance(
                embedding_ms=0.0,
                search_ms=elapsed,
                total_ms=elapsed,
            ),
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
        query: str | None = None,
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

        if self.lexical_weight and query:
            terms = self._query_terms(query)
            retrieved_chunks.sort(
                key=lambda chunk: (
                    chunk.score + self.lexical_weight
                    * self._lexical_overlap(terms, chunk.text),
                    chunk.score,
                ),
                reverse=True,
            )
        else:
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
    def _query_terms(cls, query: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[a-z0-9]{3,}", query.lower())
            if token not in cls.STOPWORDS
        }

    @staticmethod
    def _lexical_overlap(query_terms: set[str], text: str) -> float:
        if not query_terms:
            return 0.0
        text_terms = set(re.findall(r"[a-z0-9]{3,}", text.lower()))
        return len(query_terms & text_terms) / len(query_terms)

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