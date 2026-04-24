"""
WhatsApp OTP service via webhook.
"""

from __future__ import annotations

import os
import random
import string

import httpx


def generate_otp() -> str:
    """Generate a random 6-digit OTP."""
    return "".join(random.choices(string.digits, k=6))


def normalize_phone(phone: str) -> str:
    """
    Normalize phone number to digits-only format required by webhook.
    - Remove +, spaces, dashes, parentheses
    - If 10 digits, prepend India country code 91
    """
    raw = (phone or "").strip()
    digits = "".join(ch for ch in raw if ch.isdigit())
    if len(digits) == 10:
        return f"91{digits}"
    return digits


async def send_whatsapp_otp(phone_number: str, otp: str) -> bool:
    """
    Send OTP via WhatsApp webhook.
    Returns True only when webhook responds with {"accepted": true}.
    """
    webhook_url = (os.getenv("WHATSAPP_OTP_WEBHOOK") or "").strip()
    if not webhook_url:
        return False

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(
                webhook_url,
                params={"number": phone_number, "otp": otp},
            )
            if response.status_code != 200:
                return False
            payload = response.json()
            return bool(payload.get("accepted") is True)
    except Exception:
        return False
