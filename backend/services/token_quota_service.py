"""
Daily OpenAI token budget per student (UTC day). Admins are exempt.
Settings: system_settings keys + env fallback DAILY_TOKEN_LIMIT_PER_USER.
"""

from __future__ import annotations

import os
from datetime import date, datetime, timezone
from typing import Any, Dict, Optional

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from models.system_settings import SystemSettings, SettingsKeys
from models.user import User, UserRole
from models.user_daily_llm_usage import UserDailyLlmUsage


def utc_today() -> date:
    return datetime.now(timezone.utc).date()


def accum_add_openai_completion(accumulator: Optional[Dict[str, int]], response: Any) -> None:
    """Add OpenAI usage.total_tokens from a non-streaming completion response."""
    if not accumulator or response is None:
        return
    usage = getattr(response, "usage", None)
    if not usage:
        return
    total = getattr(usage, "total_tokens", None)
    if total is None:
        return
    try:
        n = int(total)
    except (TypeError, ValueError):
        return
    if n > 0:
        accumulator["total"] = int(accumulator.get("total") or 0) + n


def accum_add_tokens(accumulator: Optional[Dict[str, int]], n: int) -> None:
    if not accumulator or n <= 0:
        return
    accumulator["total"] = int(accumulator.get("total") or 0) + int(n)


def _env_default_limit() -> int:
    raw = os.getenv("DAILY_TOKEN_LIMIT_PER_USER", "200000")
    try:
        return max(1000, int(raw))
    except (TypeError, ValueError):
        return 200000


async def get_daily_token_limit_enabled(db: AsyncSession) -> bool:
    row = await db.get(SystemSettings, SettingsKeys.DAILY_TOKEN_LIMIT_ENABLED)
    if row is None:
        return os.getenv("DAILY_TOKEN_LIMIT_ENABLED", "false").lower() in ("1", "true", "yes")
    return str(row.value).lower() in ("1", "true", "yes")


async def get_daily_token_limit_per_user(db: AsyncSession) -> int:
    row = await db.get(SystemSettings, SettingsKeys.DAILY_TOKEN_LIMIT_PER_USER)
    if row is None:
        return _env_default_limit()
    try:
        return max(1000, int(row.value))
    except (TypeError, ValueError):
        return _env_default_limit()


async def _user_is_quota_exempt(db: AsyncSession, user_id: int) -> bool:
    user = await db.get(User, user_id)
    if not user:
        return True
    return user.role in (UserRole.ADMIN, UserRole.SUPER_ADMIN)


async def get_usage_for_day(
    db: AsyncSession, user_id: int, day: date
) -> int:
    row = await db.execute(
        select(UserDailyLlmUsage.total_tokens).where(
            UserDailyLlmUsage.user_id == user_id,
            UserDailyLlmUsage.usage_date == day,
        )
    )
    val = row.scalar_one_or_none()
    return int(val or 0)


async def increment_user_daily_tokens(
    db: AsyncSession,
    user_id: int,
    delta: int,
) -> None:
    if delta <= 0:
        return
    if await _user_is_quota_exempt(db, user_id):
        return
    day = utc_today()
    await db.execute(
        text(
            """
            INSERT INTO user_daily_llm_usage (user_id, usage_date, total_tokens)
            VALUES (:uid, :day, :delta)
            ON CONFLICT ON CONSTRAINT uq_user_daily_llm_usage_user_date
            DO UPDATE SET
                total_tokens = user_daily_llm_usage.total_tokens + EXCLUDED.total_tokens,
                updated_at = now()
            """
        ),
        {"uid": user_id, "day": day, "delta": delta},
    )
    await db.commit()


async def get_quota_status_for_user(
    db: AsyncSession,
    user_id: int,
) -> Dict[str, Any]:
    """
    enabled: master toggle from settings
    exempt: admins / missing user — no enforcement
    limit, used_today, remaining, blocked
    """
    if await _user_is_quota_exempt(db, user_id):
        return {
            "exempt": True,
            "enabled": False,
            "limit": 0,
            "used_today": 0,
            "remaining": 0,
            "blocked": False,
        }
    enabled = await get_daily_token_limit_enabled(db)
    limit = await get_daily_token_limit_per_user(db)
    used = await get_usage_for_day(db, user_id, utc_today())
    remaining = max(0, limit - used) if enabled else max(0, limit - used)
    blocked = bool(enabled and used >= limit)
    return {
        "exempt": False,
        "enabled": enabled,
        "limit": limit,
        "used_today": used,
        "remaining": remaining,
        "blocked": blocked,
    }
