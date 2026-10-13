import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from app.config import get_settings
from app.db import get_db
from app.models import Conversation, Message, MessageSource
from app.services.embeddings import EmbeddingService
from app.services.llm import LLMService
from app.services.rag import RAGService
from app.services.retrieval import RetrievalService
from app.services.vector_store import VectorStore

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/chat",
    tags=["chat"],
)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=20)
    # Optional for backward compatibility. New history-aware clients should
    # create a conversation and send its ID with each chat request.
    conversation_id: str | None = Field(default=None, min_length=36, max_length=36)


class ChatSource(BaseModel):
    document_id: str
    filename: str
    chunk_id: int
    score: float
    citation: str


class ChatRetrievalDiagnostics(BaseModel):
    requested_limit: int
    candidate_count: int
    returned_count: int
    filtered_count: int
    top_candidate_score: float | None
    bottom_candidate_score: float | None
    top_score: float | None
    bottom_score: float | None
    has_context: bool


class ChatPerformance(BaseModel):
    embedding_ms: float
    retrieval_ms: float
    prompt_build_ms: float
    llm_ms: float
    total_ms: float


class ChatResponse(BaseModel):
    answer: str
    sources: list[ChatSource]
    retrieval: ChatRetrievalDiagnostics
    performance: ChatPerformance
    conversation_id: str | None = None
    user_message_id: str | None = None
    assistant_message_id: str | None = None


def create_rag_service() -> RAGService:
    settings = get_settings()

    embedding_service = EmbeddingService(
        base_url=settings.ollama_base_url,
        model=settings.embedding_model,
    )

    vector_store = VectorStore(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        collection_name=settings.qdrant_collection,
    )

    retrieval_service = RetrievalService(
        embedding_service=embedding_service,
        vector_store=vector_store,
        top_k=5,
        score_threshold=0.40,
    )

    llm_service = LLMService(
        base_url=settings.ollama_base_url,
        model=settings.llm_model,
    )

    return RAGService(
        retrieval_service=retrieval_service,
        llm_service=llm_service,
    )


def _persist_start(
    db: Session,
    conversation_id: str,
    question: str,
    idempotency_key: str | None,
) -> tuple[Message, Message, ChatResponse | None]:
    conversation = db.scalar(
        select(Conversation)
        .where(
            Conversation.id == conversation_id,
            Conversation.owner_id == "local",
        )
        .with_for_update()
    )
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")

    if idempotency_key:
        prior_user_message = db.scalar(
            select(Message).where(
                Message.conversation_id == conversation_id,
                Message.role == "user",
                Message.idempotency_key == idempotency_key,
            )
        )
        if prior_user_message is not None:
            prior_assistant = db.scalar(
                select(Message)
                .where(
                    Message.conversation_id == conversation_id,
                    Message.sequence_no == prior_user_message.sequence_no + 1,
                    Message.role == "assistant",
                )
                .options(selectinload(Message.sources))
            )
            if prior_assistant is None:
                raise HTTPException(
                    status_code=409,
                    detail="A request with this idempotency key is still being processed.",
                )
            if prior_assistant.status != "complete":
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "A previous request with this idempotency key did not complete. "
                        "Retry with a new idempotency key."
                    ),
                )
            metadata = prior_assistant.metadata_json or {}
            try:
                replay = ChatResponse(
                    answer=prior_assistant.content,
                    sources=[
                        ChatSource(
                            document_id=source.document_id,
                            filename=source.filename,
                            chunk_id=source.chunk_id,
                            score=source.score,
                            citation=source.citation,
                        )
                        for source in prior_assistant.sources
                    ],
                    retrieval=ChatRetrievalDiagnostics(**metadata["retrieval"]),
                    performance=ChatPerformance(**metadata["performance"]),
                    conversation_id=conversation_id,
                    user_message_id=prior_user_message.id,
                    assistant_message_id=prior_assistant.id,
                )
            except (KeyError, TypeError, ValueError) as exc:
                logger.error("Stored chat response metadata is invalid.")
                raise HTTPException(
                    status_code=500,
                    detail="Stored chat response could not be restored.",
                ) from exc
            db.rollback()
            return prior_user_message, prior_assistant, replay

    current_max = db.scalar(
        select(func.max(Message.sequence_no)).where(
            Message.conversation_id == conversation_id
        )
    ) or 0
    now = datetime.now(timezone.utc)
    user_message = Message(
        conversation_id=conversation_id,
        sequence_no=current_max + 1,
        role="user",
        content=question,
        status="complete",
        idempotency_key=idempotency_key,
        created_at=now,
    )
    assistant_message = Message(
        conversation_id=conversation_id,
        sequence_no=current_max + 2,
        role="assistant",
        content="",
        status="pending",
        created_at=now,
    )
    db.add_all([user_message, assistant_message])

    if conversation.title == "New conversation":
        compact_title = " ".join(question.split())
        conversation.title = (
            compact_title[:197] + "..."
            if len(compact_title) > 200
            else compact_title
        )
    conversation.updated_at = now
    db.commit()
    db.refresh(user_message)
    db.refresh(assistant_message)
    return user_message, assistant_message, None


def _mark_failed(db: Session, assistant_message_id: str, exc: Exception) -> None:
    """Mark a pending assistant message as failed when it still exists.

    The message may have been deleted while the LLM was running. Use the
    captured primary key rather than an expired ORM instance, and recover the
    session before attempting this best-effort update.
    """
    try:
        db.rollback()
        message = db.scalar(
            select(Message)
            .where(Message.id == assistant_message_id)
            .with_for_update()
        )
        if message is None:
            return
        message.status = "failed"
        message.metadata_json = {"error_type": type(exc).__name__}
        db.commit()
    except Exception:
        db.rollback()
        logger.exception("Could not persist failed chat generation state.")


@router.post("", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    idempotency_key: str | None = Header(
        default=None,
        alias="Idempotency-Key",
        max_length=128,
    ),
    db: Session = Depends(get_db),
) -> ChatResponse:
    if idempotency_key and not request.conversation_id:
        raise HTTPException(
            status_code=400,
            detail="Idempotency-Key requires a conversation_id.",
        )

    user_message: Message | None = None
    assistant_message: Message | None = None
    user_message_id: str | None = None
    assistant_message_id: str | None = None

    if request.conversation_id:
        try:
            user_message, assistant_message, replay = _persist_start(
                db=db,
                conversation_id=request.conversation_id,
                question=request.question,
                idempotency_key=idempotency_key,
            )
            if replay is not None:
                return replay

            # Capture primary keys while the ORM instances are still attached.
            # Do not retain a database transaction while embeddings/LLM work runs.
            user_message_id = user_message.id
            assistant_message_id = assistant_message.id
            db.rollback()
        except HTTPException:
            raise
        except Exception as exc:
            db.rollback()
            logger.exception("Could not persist chat request.")
            raise HTTPException(
                status_code=500,
                detail="Could not persist chat request.",
            ) from exc

    try:
        service = create_rag_service()
        result = service.answer(
            question=request.question,
            top_k=request.top_k,
        )

        response = ChatResponse(
            answer=result.answer,
            sources=[
                ChatSource(
                    document_id=source.document_id,
                    filename=source.filename,
                    chunk_id=source.chunk_id,
                    score=source.score,
                    citation=source.citation,
                )
                for source in result.sources
            ],
            retrieval=ChatRetrievalDiagnostics(
                requested_limit=result.retrieval.requested_limit,
                candidate_count=result.retrieval.candidate_count,
                returned_count=result.retrieval.returned_count,
                filtered_count=result.retrieval.filtered_count,
                top_candidate_score=result.retrieval.top_candidate_score,
                bottom_candidate_score=result.retrieval.bottom_candidate_score,
                top_score=result.retrieval.top_score,
                bottom_score=result.retrieval.bottom_score,
                has_context=result.retrieval.has_context,
            ),
            performance=ChatPerformance(
                embedding_ms=result.performance.embedding_ms,
                retrieval_ms=result.performance.retrieval_ms,
                prompt_build_ms=result.performance.prompt_build_ms,
                llm_ms=result.performance.llm_ms,
                total_ms=result.performance.total_ms,
            ),
            conversation_id=request.conversation_id,
            user_message_id=user_message_id,
            assistant_message_id=assistant_message_id,
        )

        if assistant_message_id is not None and request.conversation_id is not None:
            # The LLM can take a long time. Re-fetch and lock the conversation
            # and pending message in a fresh transaction before persisting.
            # This handles a conversation/message deleted while generation ran.
            conversation = db.scalar(
                select(Conversation)
                .where(
                    Conversation.id == request.conversation_id,
                    Conversation.owner_id == "local",
                )
                .with_for_update()
            )
            if conversation is None:
                db.rollback()
                raise HTTPException(
                    status_code=409,
                    detail="Conversation was deleted while the response was being generated.",
                )

            persisted_assistant = db.scalar(
                select(Message)
                .where(
                    Message.id == assistant_message_id,
                    Message.conversation_id == request.conversation_id,
                    Message.role == "assistant",
                )
                .with_for_update()
            )
            if persisted_assistant is None:
                db.rollback()
                raise HTTPException(
                    status_code=409,
                    detail="Chat message was deleted while the response was being generated.",
                )

            persisted_assistant.content = response.answer
            persisted_assistant.status = "complete"
            persisted_assistant.metadata_json = {
                "retrieval": response.retrieval.model_dump(mode="json"),
                "performance": response.performance.model_dump(mode="json"),
            }
            for index, source in enumerate(response.sources, start=1):
                db.add(
                    MessageSource(
                        message_id=assistant_message_id,
                        source_index=index,
                        document_id=source.document_id,
                        filename=source.filename,
                        chunk_id=source.chunk_id,
                        score=source.score,
                        citation=source.citation,
                    )
                )
            conversation.updated_at = datetime.now(timezone.utc)
            db.commit()

        return response

    except HTTPException:
        db.rollback()
        raise

    except ValueError as exc:
        if assistant_message_id is not None:
            _mark_failed(db, assistant_message_id, exc)
        logger.warning(
            "RAG chat rejected input; exception_type=%s",
            type(exc).__name__,
        )
        raise HTTPException(
            status_code=400,
            detail="Invalid chat request.",
        ) from exc

    except Exception as exc:
        if assistant_message_id is not None:
            _mark_failed(db, assistant_message_id, exc)
        logger.exception(
            "RAG chat failed; exception_type=%s",
            type(exc).__name__,
        )
        raise HTTPException(
            status_code=500,
            detail="RAG chat failed.",
        ) from exc
