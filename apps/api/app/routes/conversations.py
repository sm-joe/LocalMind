import base64
import binascii
import json
import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field, model_validator
from sqlalchemy import and_, func, or_, select
from sqlalchemy.orm import Session, selectinload

from app.db import get_db
from app.models import Conversation, Message, MessageSource

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/conversations", tags=["conversations"])
LOCAL_OWNER_ID = "local"


class ConversationCreate(BaseModel):
    title: str = Field(default="New conversation", min_length=1, max_length=200)


class ConversationUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=200)
    is_archived: bool | None = None

    @model_validator(mode="after")
    def require_update_field(self):
        if self.title is None and self.is_archived is None:
            raise ValueError("At least one field must be provided.")
        return self


class ConversationSummary(BaseModel):
    id: str
    title: str
    is_archived: bool
    created_at: datetime
    updated_at: datetime
    message_count: int = 0


class ConversationListResponse(BaseModel):
    items: list[ConversationSummary]
    next_cursor: str | None


class ConversationDetail(BaseModel):
    id: str
    title: str
    is_archived: bool
    created_at: datetime
    updated_at: datetime


class MessageSourceResponse(BaseModel):
    document_id: str
    filename: str
    chunk_id: int
    score: float
    citation: str


class MessageResponse(BaseModel):
    id: str
    sequence_no: int
    role: str
    content: str
    status: str
    created_at: datetime
    sources: list[MessageSourceResponse]
    metadata: dict[str, Any] | None = None


class MessageListResponse(BaseModel):
    items: list[MessageResponse]
    next_cursor: int | None


def _encode_cursor(timestamp: datetime, item_id: str) -> str:
    raw = json.dumps([timestamp.isoformat(), item_id], separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _decode_cursor(cursor: str) -> tuple[datetime, str]:
    try:
        padded = cursor + ("=" * (-len(cursor) % 4))
        timestamp, item_id = json.loads(base64.urlsafe_b64decode(padded.encode()).decode())
        parsed = datetime.fromisoformat(timestamp)
        if not item_id or len(item_id) > 36:
            raise ValueError
        return parsed, item_id
    except (
        ValueError,
        TypeError,
        json.JSONDecodeError,
        UnicodeDecodeError,
        binascii.Error,
    ) as exc:
        raise HTTPException(status_code=400, detail="Invalid pagination cursor.") from exc


def _get_conversation(db: Session, conversation_id: str) -> Conversation:
    conversation = db.scalar(
        select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.owner_id == LOCAL_OWNER_ID,
        )
    )
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return conversation


@router.post("", response_model=ConversationDetail, status_code=201)
def create_conversation(
    request: ConversationCreate,
    db: Session = Depends(get_db),
) -> ConversationDetail:
    title = request.title.strip()
    if not title:
        raise HTTPException(status_code=422, detail="Title cannot be blank.")
    conversation = Conversation(owner_id=LOCAL_OWNER_ID, title=title)
    db.add(conversation)
    db.commit()
    db.refresh(conversation)
    return ConversationDetail(
        id=conversation.id,
        title=conversation.title,
        is_archived=conversation.is_archived,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
    )


@router.get("", response_model=ConversationListResponse)
def list_conversations(
    limit: int = Query(default=50, ge=1, le=100),
    cursor: str | None = Query(default=None, max_length=512),
    include_archived: bool = False,
    db: Session = Depends(get_db),
) -> ConversationListResponse:
    message_count = (
        select(func.count(Message.id))
        .where(Message.conversation_id == Conversation.id)
        .scalar_subquery()
    )
    statement = select(
        Conversation,
        message_count.label("message_count"),
    ).where(Conversation.owner_id == LOCAL_OWNER_ID)
    if not include_archived:
        statement = statement.where(Conversation.is_archived.is_(False))
    if cursor:
        timestamp, item_id = _decode_cursor(cursor)
        statement = statement.where(
            or_(
                Conversation.updated_at < timestamp,
                and_(
                    Conversation.updated_at == timestamp,
                    Conversation.id < item_id,
                ),
            )
        )
    statement = statement.order_by(
        Conversation.updated_at.desc(), Conversation.id.desc()
    ).limit(limit + 1)
    rows = list(db.execute(statement).all())
    has_more = len(rows) > limit
    rows = rows[:limit]
    next_cursor = (
        _encode_cursor(rows[-1][0].updated_at, rows[-1][0].id)
        if has_more and rows else None
    )
    return ConversationListResponse(
        items=[
            ConversationSummary(
                id=conversation.id,
                title=conversation.title,
                is_archived=conversation.is_archived,
                created_at=conversation.created_at,
                updated_at=conversation.updated_at,
                message_count=message_count_value,
            )
            for conversation, message_count_value in rows
        ],
        next_cursor=next_cursor,
    )


@router.get("/{conversation_id}", response_model=ConversationDetail)
def get_conversation(
    conversation_id: str,
    db: Session = Depends(get_db),
) -> ConversationDetail:
    conversation = _get_conversation(db, conversation_id)
    return ConversationDetail(
        id=conversation.id,
        title=conversation.title,
        is_archived=conversation.is_archived,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
    )


@router.get("/{conversation_id}/messages", response_model=MessageListResponse)
def list_messages(
    conversation_id: str,
    limit: int = Query(default=50, ge=1, le=100),
    before_sequence: int | None = Query(default=None, ge=1),
    db: Session = Depends(get_db),
) -> MessageListResponse:
    _get_conversation(db, conversation_id)
    statement = (
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .options(selectinload(Message.sources))
    )
    if before_sequence is not None:
        statement = statement.where(Message.sequence_no < before_sequence)
    statement = statement.order_by(Message.sequence_no.desc()).limit(limit + 1)
    messages = list(db.scalars(statement).all())
    has_more = len(messages) > limit
    messages = messages[:limit]
    messages.reverse()
    next_cursor = messages[0].sequence_no if has_more and messages else None
    return MessageListResponse(
        items=[
            MessageResponse(
                id=message.id,
                sequence_no=message.sequence_no,
                role=message.role,
                content=message.content,
                status=message.status,
                created_at=message.created_at,
                sources=[
                    MessageSourceResponse(
                        document_id=source.document_id,
                        filename=source.filename,
                        chunk_id=source.chunk_id,
                        score=source.score,
                        citation=source.citation,
                    )
                    for source in message.sources
                ],
                metadata=message.metadata_json,
            )
            for message in messages
        ],
        next_cursor=next_cursor,
    )


@router.patch("/{conversation_id}", response_model=ConversationDetail)
def update_conversation(
    conversation_id: str,
    request: ConversationUpdate,
    db: Session = Depends(get_db),
) -> ConversationDetail:
    conversation = _get_conversation(db, conversation_id)
    changes = request.model_dump(exclude_unset=True)
    if "title" in changes:
        title = (changes["title"] or "").strip()
        if not title:
            raise HTTPException(status_code=422, detail="Title cannot be blank.")
        conversation.title = title
    if "is_archived" in changes:
        conversation.is_archived = changes["is_archived"]
    conversation.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(conversation)
    return ConversationDetail(
        id=conversation.id,
        title=conversation.title,
        is_archived=conversation.is_archived,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
    )


@router.delete("/{conversation_id}", status_code=204)
def delete_conversation(
    conversation_id: str,
    db: Session = Depends(get_db),
) -> None:
    conversation = _get_conversation(db, conversation_id)
    db.delete(conversation)
    db.commit()
