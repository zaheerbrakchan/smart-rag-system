"""
Support query routes for students and admins.
"""

import asyncio
import json
from datetime import datetime
from typing import Optional, List, Dict

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import select, func, exists
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from sqlalchemy import inspect as sa_inspect
from sqlalchemy.orm.attributes import NO_VALUE

from database.connection import get_db
from dependencies.auth import get_current_user, get_current_admin
from models.user import User
from models.support_query import (
    SupportQuery,
    SupportQueryReply,
    UserNotification,
    SupportQueryStatus,
)
from models.activity_log import ActionType
from models.system_settings import SystemSettings, SettingsKeys
from repositories.activity_log_repository import ActivityLogRepository
from repositories.user_repository import UserRepository
from services.auth_service import AuthService
from services.support_notification_service import SupportNotificationService

router = APIRouter(tags=["Support"])


class _SupportNotificationHub:
    """
    In-process pub/sub for per-user support notification events.
    """

    def __init__(self) -> None:
        self._subs: Dict[int, List[asyncio.Queue]] = {}
        self._lock = asyncio.Lock()

    async def subscribe(self, user_id: int) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        async with self._lock:
            self._subs.setdefault(user_id, []).append(q)
        return q

    async def unsubscribe(self, user_id: int, queue: asyncio.Queue) -> None:
        async with self._lock:
            queues = self._subs.get(user_id, [])
            if queue in queues:
                queues.remove(queue)
            if not queues and user_id in self._subs:
                self._subs.pop(user_id, None)

    async def publish(self, user_id: int, event: dict) -> None:
        async with self._lock:
            queues = list(self._subs.get(user_id, []))
        for q in queues:
            try:
                q.put_nowait(event)
            except Exception:
                continue


notification_hub = _SupportNotificationHub()


class SupportQueryCreateRequest(BaseModel):
    student_name: Optional[str] = Field(default=None, max_length=120)
    phone: Optional[str] = Field(default=None, max_length=20)
    email: Optional[str] = Field(default=None, max_length=255)
    subject: Optional[str] = Field(default=None, max_length=255)
    message: str = Field(min_length=1, max_length=5000)


class SupportQueryReplyRequest(BaseModel):
    reply_text: str = Field(min_length=2, max_length=5000)


class SupportQueryStatusRequest(BaseModel):
    status: SupportQueryStatus
    assigned_admin_id: Optional[int] = None


class SupportReplyDTO(BaseModel):
    id: int
    responder_admin_id: Optional[int]
    reply_text: str
    sent_email: bool
    sent_sms: bool
    created_at: datetime


class SupportQueryDTO(BaseModel):
    id: int
    user_id: int
    student_name: str
    phone: str
    email: Optional[str]
    subject: str
    message: str
    status: SupportQueryStatus
    assigned_admin_id: Optional[int]
    answered_at: Optional[datetime]
    created_at: datetime
    updated_at: datetime
    replies: List[SupportReplyDTO] = []


class SupportQueryListResponse(BaseModel):
    queries: List[SupportQueryDTO]
    total: int
    page: int
    page_size: int
    total_pages: int


class NotificationDTO(BaseModel):
    id: int
    type: str
    title: str
    body: str
    related_query_id: Optional[int]
    is_read: bool
    created_at: datetime


def _to_query_dto(item: SupportQuery) -> SupportQueryDTO:
    replies_data = []
    rel_state = sa_inspect(item).attrs.replies
    if rel_state.loaded_value is not NO_VALUE:
        replies_data = list(item.replies or [])
    replies = [
        SupportReplyDTO(
            id=r.id,
            responder_admin_id=r.responder_admin_id,
            reply_text=r.reply_text,
            sent_email=r.sent_email,
            sent_sms=r.sent_sms,
            created_at=r.created_at,
        )
        for r in replies_data
    ]
    return SupportQueryDTO(
        id=item.id,
        user_id=item.user_id,
        student_name=item.student_name,
        phone=item.phone,
        email=item.email,
        subject=item.subject,
        message=item.message,
        status=item.status,
        assigned_admin_id=item.assigned_admin_id,
        answered_at=item.answered_at,
        created_at=item.created_at,
        updated_at=item.updated_at,
        replies=replies,
    )


async def _get_bool_setting(db: AsyncSession, key: str, default: bool) -> bool:
    setting = await db.get(SystemSettings, key)
    if not setting:
        return default
    return str(setting.value).strip().lower() in {"1", "true", "yes", "on"}


async def _get_str_setting(db: AsyncSession, key: str, default: str = "") -> str:
    setting = await db.get(SystemSettings, key)
    if not setting:
        return default
    return str(setting.value or "").strip() or default


@router.post("/support/queries", response_model=SupportQueryDTO)
async def create_support_query(
    http_request: Request,
    request: SupportQueryCreateRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    student_name = (request.student_name or current_user.full_name or "").strip()
    phone = (request.phone or current_user.phone or "").strip()
    if not student_name:
        raise HTTPException(status_code=400, detail="Student name is required.")
    if not phone:
        raise HTTPException(status_code=400, detail="Phone number is required.")
    if len(phone) < 10 or len(phone) > 20:
        raise HTTPException(status_code=400, detail="Phone number format is invalid.")
    email = (request.email or "").strip() or None
    if email and ("@" not in email or "." not in email.split("@")[-1]):
        raise HTTPException(status_code=400, detail="Email format is invalid.")

    subject = (request.subject or "").strip()
    if not subject:
        subject = "Support query from student"

    item = SupportQuery(
        user_id=current_user.id,
        student_name=student_name,
        phone=phone,
        email=email,
        subject=subject,
        message=request.message.strip(),
        status=SupportQueryStatus.PENDING,
    )
    db.add(item)
    await db.commit()
    await db.refresh(item)

    # Fire best-effort notifications after DB commit.
    email_enabled = await _get_bool_setting(db, SettingsKeys.SUPPORT_EMAIL_ENABLED, True)
    sms_enabled = await _get_bool_setting(db, SettingsKeys.SUPPORT_SMS_ENABLED, True)
    inbox_override = await _get_str_setting(db, SettingsKeys.SUPPORT_INBOX_EMAIL, "")

    if email_enabled:
        inbox_ok, inbox_err = SupportNotificationService.notify_support_inbox_new_query(item, inbox_override or None)
    else:
        inbox_ok, inbox_err = False, "support email disabled by setting"

    # Per requirement: do NOT send SMS on query submission, only on admin replies.
    sms_ok, sms_err = False, "submit ack sms disabled by product rule"
    if not inbox_ok:
        print(f"[SUPPORT] inbox email failed for Q#{item.id}: {inbox_err}")
    if not sms_ok:
        print(f"[SUPPORT] ack sms failed for Q#{item.id}: {sms_err}")
    await ActivityLogRepository(db).log_action(
        action_type=ActionType.CHAT_MESSAGE,
        description="Support query submitted by student",
        user_id=current_user.id,
        target_type="support_query",
        target_id=str(item.id),
        ip_address=http_request.client.host if http_request.client else None,
        user_agent=http_request.headers.get("user-agent"),
        request_data={"subject": item.subject},
        response_data={
            "inbox_email_sent": inbox_ok,
            "ack_sms_sent": False,
        },
        success=True,
        error_details={
            "inbox_error": inbox_err,
            "sms_error": sms_err,
        },
    )

    await db.refresh(item, attribute_names=["replies"])
    return _to_query_dto(item)


@router.get("/support/queries/me", response_model=List[SupportQueryDTO])
async def list_my_support_queries(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(
        select(SupportQuery)
        .where(SupportQuery.user_id == current_user.id)
        .options(selectinload(SupportQuery.replies))
        .order_by(SupportQuery.created_at.desc())
    )
    rows = result.scalars().all()
    return [_to_query_dto(r) for r in rows]


@router.get("/support/notifications/me", response_model=List[NotificationDTO])
async def list_my_notifications(
    limit: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(
        select(UserNotification)
        .where(UserNotification.user_id == current_user.id)
        .order_by(UserNotification.created_at.desc())
        .limit(limit)
    )
    rows = result.scalars().all()
    return [
        NotificationDTO(
            id=n.id,
            type=n.type,
            title=n.title,
            body=n.body,
            related_query_id=n.related_query_id,
            is_read=n.is_read,
            created_at=n.created_at,
        )
        for n in rows
    ]


async def _get_user_from_access_token(db: AsyncSession, token: str) -> User:
    payload = AuthService.verify_token(token, token_type="access")
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token payload")
    user = await UserRepository(db).get_by_id(int(user_id))
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="User account is deactivated")
    return user


@router.get("/support/notifications/stream")
async def stream_my_notifications(
    token: str = Query(..., min_length=10),
    db: AsyncSession = Depends(get_db),
):
    current_user = await _get_user_from_access_token(db, token)
    queue = await notification_hub.subscribe(current_user.id)

    async def event_gen():
        try:
            # Handshake event so frontend knows stream is connected.
            yield f"data: {json.dumps({'type': 'connected'})}\n\n"
            while True:
                try:
                    evt = await asyncio.wait_for(queue.get(), timeout=30)
                    yield f"data: {json.dumps(evt)}\n\n"
                except asyncio.TimeoutError:
                    # Keep connection alive without extra DB/API calls.
                    yield ": keepalive\n\n"
        except asyncio.CancelledError:
            return
        finally:
            await notification_hub.unsubscribe(current_user.id, queue)

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Content-Type": "text/event-stream; charset=utf-8",
            "Connection": "keep-alive",
            "Pragma": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.patch("/support/notifications/{notification_id}/read")
async def mark_notification_read(
    notification_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    item = await db.get(UserNotification, notification_id)
    if not item or item.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Notification not found.")
    item.is_read = True
    await db.commit()
    return {"success": True}


@router.get("/admin/support/queries", response_model=SupportQueryListResponse)
async def list_support_queries_admin(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status: Optional[SupportQueryStatus] = None,
    replied: Optional[bool] = None,
    user_id: Optional[int] = None,
    search: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
    current_admin: User = Depends(get_current_admin),
):
    query = select(SupportQuery).options(selectinload(SupportQuery.replies))
    count_query = select(func.count(SupportQuery.id))

    if status:
        query = query.where(SupportQuery.status == status)
        count_query = count_query.where(SupportQuery.status == status)
    if replied is not None:
        has_reply = exists(
            select(SupportQueryReply.id).where(SupportQueryReply.query_id == SupportQuery.id)
        )
        if replied:
            query = query.where(has_reply)
            count_query = count_query.where(has_reply)
        else:
            query = query.where(~has_reply)
            count_query = count_query.where(~has_reply)
    if user_id:
        query = query.where(SupportQuery.user_id == user_id)
        count_query = count_query.where(SupportQuery.user_id == user_id)
    if search:
        like = f"%{search.strip()}%"
        query = query.where(
            (SupportQuery.student_name.ilike(like))
            | (SupportQuery.subject.ilike(like))
            | (SupportQuery.message.ilike(like))
            | (SupportQuery.phone.ilike(like))
        )
        count_query = count_query.where(
            (SupportQuery.student_name.ilike(like))
            | (SupportQuery.subject.ilike(like))
            | (SupportQuery.message.ilike(like))
            | (SupportQuery.phone.ilike(like))
        )

    total = await db.scalar(count_query) or 0
    offset = (page - 1) * page_size
    result = await db.execute(
        query.order_by(SupportQuery.created_at.desc()).offset(offset).limit(page_size)
    )
    rows = result.scalars().all()
    return SupportQueryListResponse(
        queries=[_to_query_dto(r) for r in rows],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=(total + page_size - 1) // page_size,
    )


@router.patch("/admin/support/queries/{query_id}", response_model=SupportQueryDTO)
async def update_support_query_admin(
    http_request: Request,
    query_id: int,
    request: SupportQueryStatusRequest,
    db: AsyncSession = Depends(get_db),
    current_admin: User = Depends(get_current_admin),
):
    item = await db.get(SupportQuery, query_id)
    if not item:
        raise HTTPException(status_code=404, detail="Support query not found.")

    item.status = request.status
    if request.assigned_admin_id is not None:
        assignee = await db.get(User, request.assigned_admin_id)
        if not assignee or not assignee.is_admin:
            raise HTTPException(status_code=400, detail="Assigned admin is invalid.")
        item.assigned_admin_id = assignee.id

    if request.status in {SupportQueryStatus.ANSWERED, SupportQueryStatus.CLOSED}:
        item.answered_at = datetime.utcnow()

    await db.commit()
    await db.refresh(item)
    await db.refresh(item, attribute_names=["replies"])
    await ActivityLogRepository(db).log_action(
        action_type=ActionType.CHAT_MESSAGE,
        description=f"Admin updated support query status to {request.status.value}",
        user_id=current_admin.id,
        target_type="support_query",
        target_id=str(item.id),
        ip_address=http_request.client.host if http_request.client else None,
        user_agent=http_request.headers.get("user-agent"),
        request_data={"assigned_admin_id": request.assigned_admin_id},
        success=True,
    )
    return _to_query_dto(item)


@router.post("/admin/support/queries/{query_id}/reply", response_model=SupportQueryDTO)
async def reply_support_query_admin(
    http_request: Request,
    query_id: int,
    request: SupportQueryReplyRequest,
    db: AsyncSession = Depends(get_db),
    current_admin: User = Depends(get_current_admin),
):
    item = await db.get(SupportQuery, query_id)
    if not item:
        raise HTTPException(status_code=404, detail="Support query not found.")
    existing_replies = await db.scalar(
        select(func.count(SupportQueryReply.id)).where(SupportQueryReply.query_id == item.id)
    )
    if (existing_replies or 0) > 0:
        raise HTTPException(
            status_code=409,
            detail="This query is already answered. Multiple replies are not allowed.",
        )

    reply = SupportQueryReply(
        query_id=item.id,
        responder_admin_id=current_admin.id,
        reply_text=request.reply_text.strip(),
        sent_email=False,
        sent_sms=False,
    )
    db.add(reply)
    item.status = SupportQueryStatus.ANSWERED
    item.answered_at = datetime.utcnow()
    item.assigned_admin_id = current_admin.id

    notif = UserNotification(
        user_id=item.user_id,
        type="support_update",
        title="Your support query has a new reply",
        body=request.reply_text.strip()[:500],
        related_query_id=item.id,
        is_read=False,
    )
    db.add(notif)

    email_enabled = await _get_bool_setting(db, SettingsKeys.SUPPORT_EMAIL_ENABLED, True)
    sms_enabled = await _get_bool_setting(db, SettingsKeys.SUPPORT_SMS_ENABLED, True)

    if email_enabled:
        email_ok, _email_err = SupportNotificationService.notify_student_reply_email(item, request.reply_text.strip())
    else:
        email_ok, _email_err = False, "support email disabled by setting"
    if sms_enabled:
        sms_ok, _sms_err = SupportNotificationService.notify_student_reply_sms(item, request.reply_text.strip())
    else:
        sms_ok, _sms_err = False, "support sms disabled by setting"
    reply.sent_email = email_ok
    reply.sent_sms = sms_ok
    if not email_ok:
        print(f"[SUPPORT] reply email failed for Q#{item.id}: {_email_err}")
    if not sms_ok:
        print(f"[SUPPORT] reply sms failed for Q#{item.id}: {_sms_err}")

    await db.commit()
    await db.refresh(item)
    await db.refresh(item, attribute_names=["replies"])
    await db.refresh(notif)
    await notification_hub.publish(
        item.user_id,
        {
            "type": "support_notification_created",
            "query_id": item.id,
            "notification_id": notif.id,
        },
    )
    await ActivityLogRepository(db).log_action(
        action_type=ActionType.CHAT_MESSAGE,
        description="Admin replied to support query",
        user_id=current_admin.id,
        target_type="support_query",
        target_id=str(item.id),
        ip_address=http_request.client.host if http_request.client else None,
        user_agent=http_request.headers.get("user-agent"),
        response_data={"email_sent": email_ok, "sms_sent": sms_ok},
        success=True,
        error_details={"email_error": _email_err, "sms_error": _sms_err},
    )
    return _to_query_dto(item)
