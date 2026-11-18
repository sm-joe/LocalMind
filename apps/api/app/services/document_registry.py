"""Transactional document state and concurrent upload coordination.

PostgreSQL is authoritative for indexing state. Qdrant stores chunks.
Uploaded bytes remain request-scoped: a stale task can be retried only by
uploading the file again, not by restarting a lost in-memory worker.
"""
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import uuid

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from app.db import get_session_factory
from app.models import DocumentIndex


STALE_AFTER = timedelta(minutes=15)


class DocumentIndexInProgress(Exception):
    pass


@dataclass(frozen=True)
class IndexClaim:
    document_hash: str
    document_id: str
    status: str
    filename: str
    previous_document_id: str | None = None
    previous_chunks: int = 0
    previous_filename: str | None = None
    previous_content_type: str | None = None
    restore_on_failure: bool = False


class DocumentIndexRegistry:
    def __init__(self, session_factory=None):
        self.session_factory = session_factory or get_session_factory()

    def acquire(
        self, document_hash: str, filename: str, content_type: str,
        force: bool = False,
    ) -> IndexClaim:
        now = datetime.now(timezone.utc)
        proposed_id = str(uuid.uuid4())
        with self.session_factory() as db:
            with db.begin():
                db.execute(
                    insert(DocumentIndex).values(
                        document_hash=document_hash,
                        document_id=proposed_id,
                        filename=filename,
                        content_type=content_type,
                        status="indexing",
                        chunks=0,
                        updated_at=now,
                    ).on_conflict_do_nothing(index_elements=["document_hash"])
                )
                record = db.scalar(
                    select(DocumentIndex)
                    .where(DocumentIndex.document_hash == document_hash)
                    .with_for_update()
                )
                if record is None:
                    raise RuntimeError("Could not claim document index entry.")
                if record.status == "ready" and not force:
                    return IndexClaim(
                        document_hash, record.document_id, "ready",
                        record.filename,
                    )
                if record.status == "indexing" and record.document_id != proposed_id:
                    updated_at = record.updated_at
                    if updated_at.tzinfo is None:
                        updated_at = updated_at.replace(tzinfo=timezone.utc)
                    if now - updated_at < STALE_AFTER:
                        raise DocumentIndexInProgress(
                            "The same document is being indexed. Retry later."
                        )
                prior_id = record.document_id if record.document_id != proposed_id else None
                restore_previous = force and record.status == "ready"
                prior_chunks = record.chunks if restore_previous else 0
                prior_filename = record.filename if restore_previous else None
                prior_content_type = record.content_type if restore_previous else None
                record.document_id = proposed_id
                record.status = "indexing"
                record.filename = filename
                record.content_type = content_type
                record.updated_at = now
                record.chunks = 0
                return IndexClaim(
                    document_hash, proposed_id, "claimed",
                    filename, previous_document_id=prior_id,
                    previous_chunks=prior_chunks,
                    previous_filename=prior_filename,
                    previous_content_type=prior_content_type,
                    restore_on_failure=restore_previous,
                )

    def complete(
        self, claim: IndexClaim, document_id: str, filename: str, chunks: int
    ) -> None:
        with self.session_factory() as db:
            with db.begin():
                record = db.scalar(
                    select(DocumentIndex).where(
                        DocumentIndex.document_hash == claim.document_hash
                    ).with_for_update()
                )
                if record is None or record.document_id != claim.document_id:
                    raise RuntimeError("Document indexing claim was superseded.")
                record.document_id = document_id
                record.filename = filename
                record.chunks = chunks
                record.status = "ready"
                record.updated_at = datetime.now(timezone.utc)

    def fail(self, claim: IndexClaim) -> None:
        with self.session_factory() as db:
            with db.begin():
                record = db.scalar(
                    select(DocumentIndex).where(
                        DocumentIndex.document_hash == claim.document_hash
                    ).with_for_update()
                )
                if record and record.document_id == claim.document_id:
                    if claim.restore_on_failure and claim.previous_document_id:
                        # Forced reindex failed: restore the prior registry
                        # record. Its Qdrant chunks have never been deleted.
                        record.document_id = claim.previous_document_id
                        record.status = "ready"
                        record.chunks = claim.previous_chunks
                        record.filename = claim.previous_filename or record.filename
                        record.content_type = (
                            claim.previous_content_type or record.content_type
                        )
                    else:
                        record.status = "failed"
                    record.updated_at = datetime.now(timezone.utc)

    def get_status(self, document_id: str) -> DocumentIndex | None:
        with self.session_factory() as db:
            return db.scalar(
                select(DocumentIndex).where(DocumentIndex.document_id == document_id)
            )

    def forget(self, document_id: str) -> None:
        with self.session_factory() as db:
            with db.begin():
                db.execute(
                    delete(DocumentIndex).where(
                        DocumentIndex.document_id == document_id
                    )
                )
