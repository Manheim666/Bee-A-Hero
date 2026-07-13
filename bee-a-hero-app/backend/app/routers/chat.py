from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..auth import get_current_user
from ..db import get_db
from ..models import Conversation, Message, MessageRole, User
from ..schemas import (
    ConversationDetail,
    ConversationOut,
    MessageCreate,
    MessageOut,
)
from ..services import stats as stats_service
from ..services.llm import available_providers, chat as llm_chat

router = APIRouter(prefix="/api/conversations", tags=["chat"])


@router.get("/providers")
def list_providers(user: User = Depends(get_current_user)):
    """LLM providers the Assistant tab can pick from (auto / gemini / hugging face / …)."""
    return available_providers()


def _owned_conversation(db: Session, conv_id: int, user: User) -> Conversation:
    conv = db.get(Conversation, conv_id)
    if conv is None or conv.user_id != user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found"
        )
    return conv


def _user_context(db: Session, user: User) -> str:
    ov = stats_service.overview(db, user.id)
    if ov.videos_processed == 0:
        return "This user has not processed any videos yet."
    top = f"#{ov.top_flower}" if ov.top_flower is not None else "n/a"
    return (
        f"{ov.videos_processed} processed videos, {ov.total_visits} total insect "
        f"visits, {ov.pollinator_pct}% pollinator, {ov.avg_visits_per_flower} "
        f"average visits per flower, top flower is {top}."
    )


@router.get("", response_model=list[ConversationOut])
def list_conversations(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    return db.scalars(
        select(Conversation)
        .where(Conversation.user_id == user.id)
        .order_by(Conversation.updated_at.desc())
    ).all()


@router.post("", response_model=ConversationDetail)
def create_conversation(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    conv = Conversation(user_id=user.id, title="New chat")
    db.add(conv)
    db.commit()
    db.refresh(conv)
    return conv


@router.get("/{conv_id}", response_model=ConversationDetail)
def get_conversation(
    conv_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    return _owned_conversation(db, conv_id, user)


@router.post("/{conv_id}/messages", response_model=MessageOut)
def post_message(
    conv_id: int,
    payload: MessageCreate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    conv = _owned_conversation(db, conv_id, user)

    user_msg = Message(
        conversation_id=conv.id,
        role=MessageRole.user,
        content=payload.content,
    )
    db.add(user_msg)

    # Auto-title from the first user message.
    if conv.title == "New chat":
        conv.title = payload.content[:60] + ("…" if len(payload.content) > 60 else "")

    db.commit()

    history = [
        {"role": m.role.value, "content": m.content} for m in conv.messages
    ]
    reply = llm_chat(history, _user_context(db, user), payload.provider)

    assistant_msg = Message(
        conversation_id=conv.id,
        role=MessageRole.assistant,
        content=reply,
    )
    db.add(assistant_msg)
    db.commit()
    db.refresh(assistant_msg)
    return assistant_msg


@router.delete("/{conv_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_conversation(
    conv_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    conv = _owned_conversation(db, conv_id, user)
    db.delete(conv)
    db.commit()
