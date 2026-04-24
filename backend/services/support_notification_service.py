"""
Support notification delivery service (email + SMS).
"""

import os
import smtplib
from email.message import EmailMessage
from typing import Tuple, Optional

from dotenv import load_dotenv

from models.support_query import SupportQuery

load_dotenv()


class SupportNotificationService:
    @staticmethod
    def _smtp_settings() -> Tuple[str, int, str, str, bool]:
        host = os.getenv("SMTP_HOST", "").strip()
        port = int(os.getenv("SMTP_PORT", "587"))
        username = os.getenv("SMTP_USERNAME", "").strip()
        password = os.getenv("SMTP_PASSWORD", "").strip()
        use_tls = os.getenv("SMTP_USE_TLS", "true").strip().lower() in {"1", "true", "yes"}
        return host, port, username, password, use_tls

    @staticmethod
    def _from_email() -> str:
        return (
            os.getenv("SUPPORT_SENDER_EMAIL")
            or os.getenv("SMTP_FROM_EMAIL")
            or "support@getmyuniversity.com"
        ).strip()

    @staticmethod
    def support_inbox_email() -> str:
        return (
            os.getenv("SUPPORT_INBOX_EMAIL")
            or os.getenv("SMTP_FROM_EMAIL")
            or "support@getmyuniversity.com"
        ).strip()

    @staticmethod
    def _send_email(to_email: str, subject: str, body: str) -> Tuple[bool, Optional[str]]:
        to_email = (to_email or "").strip()
        if not to_email:
            return False, "missing recipient"
        host, port, username, password, use_tls = SupportNotificationService._smtp_settings()
        if not host:
            return False, "smtp not configured"

        msg = EmailMessage()
        msg["From"] = SupportNotificationService._from_email()
        msg["To"] = to_email
        msg["Subject"] = subject
        msg.set_content(body)

        try:
            with smtplib.SMTP(host, port, timeout=30) as server:
                if use_tls:
                    server.starttls()
                if username and password:
                    server.login(username, password)
                server.send_message(msg)
            return True, None
        except Exception as e:
            return False, str(e)

    @staticmethod
    def _send_sms(phone: str, body: str) -> Tuple[bool, Optional[str]]:
        phone = (phone or "").strip()
        if not phone:
            return False, "missing phone"

        debug_mode = os.getenv("DEBUG", "true").lower() == "true"
        if debug_mode:
            print(f"[SUPPORT_SMS][DEV] to={phone} body={body}")
            return True, None

        # Twilio SMS integration removed from this codebase.
        # Keep graceful behavior when SMS is enabled in settings but no provider is configured.
        return False, "sms provider not configured"

    @staticmethod
    def notify_support_inbox_new_query(
        item: SupportQuery,
        inbox_override: Optional[str] = None,
    ) -> Tuple[bool, Optional[str]]:
        to_email = (inbox_override or "").strip() or SupportNotificationService.support_inbox_email()
        subject = f"Support query from student | {item.student_name} | {item.phone} | Q#{item.id}"
        body = (
            "New support query received.\n\n"
            f"Query ID: {item.id}\n"
            f"Student Name: {item.student_name}\n"
            f"Phone: {item.phone}\n"
            f"Email: {item.email or '-'}\n"
            f"User ID: {item.user_id}\n"
            f"Status: {item.status.value}\n"
            f"Subject: {item.subject}\n\n"
            "Student Message:\n"
            f"{item.message}\n"
        )
        return SupportNotificationService._send_email(to_email, subject, body)

    @staticmethod
    def send_student_ack_sms(item: SupportQuery) -> Tuple[bool, Optional[str]]:
        body = (
            f"Get My University support: we received your query (Q#{item.id}). "
            "Our team will respond soon."
        )
        return SupportNotificationService._send_sms(item.phone, body)

    @staticmethod
    def notify_student_reply_email(item: SupportQuery, reply_text: str) -> Tuple[bool, Optional[str]]:
        if not item.email:
            return False, "student email missing"
        subject = f"Response to your support query | Q#{item.id}"
        body = (
            f"Hello {item.student_name},\n\n"
            "Our support team has replied to your query.\n\n"
            f"Query ID: {item.id}\n"
            f"Original subject: {item.subject}\n\n"
            "Support Reply:\n"
            f"{reply_text}\n\n"
            "Regards,\nGet My University Support Team"
        )
        return SupportNotificationService._send_email(item.email, subject, body)

    @staticmethod
    def notify_student_reply_sms(item: SupportQuery, reply_text: str) -> Tuple[bool, Optional[str]]:
        snippet = reply_text.strip().replace("\n", " ")
        if len(snippet) > 120:
            snippet = snippet[:117] + "..."
        body = f"Get My University support replied to Q#{item.id}: {snippet}"
        return SupportNotificationService._send_sms(item.phone, body)
