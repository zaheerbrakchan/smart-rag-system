"""Normalize DATABASE_URL for sync (psycopg2) vs async (asyncpg) drivers."""

import os
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse


def get_database_url() -> str:
    return os.getenv(
        "DATABASE_URL",
        "postgresql+asyncpg://postgres:postgres@localhost:5432/neet_assistant",
    )


def async_url_for_neon(url: str) -> str:
    """Ensure asyncpg driver; strip params asyncpg does not accept on connect()."""
    u = url
    if u.startswith("postgresql://") and "+asyncpg" not in u:
        u = u.replace("postgresql://", "postgresql+asyncpg://", 1)
    parsed = urlparse(u)
    q = parse_qs(parsed.query)
    # asyncpg.connect() does not accept sslmode / channel_binding — causes:
    # "got an unexpected keyword argument 'sslmode'"
    for bad in ("channel_binding", "sslmode"):
        if bad in q:
            del q[bad]
    new_query = urlencode({k: v[0] for k, v in q.items()}, doseq=True)
    parsed = parsed._replace(query=new_query)
    return urlunparse(parsed)


def is_local_postgres_host(database_url: str) -> bool:
    """True if host looks like local dev (no TLS required)."""
    u = database_url.replace("postgresql+asyncpg://", "postgresql://", 1)
    u = u.replace("postgresql+psycopg2://", "postgresql://", 1)
    parsed = urlparse(u)
    host = (parsed.hostname or "").lower()
    return host in ("localhost", "127.0.0.1", "::1")


def sync_database_url(url: str | None = None) -> str:
    """SQLAlchemy sync URL for PGVectorStore (psycopg2)."""
    raw = url or get_database_url()
    u = raw.replace("postgresql+asyncpg://", "postgresql+psycopg2://", 1)
    if u.startswith("postgresql://") and "+psycopg2" not in u and "+asyncpg" not in u:
        u = u.replace("postgresql://", "postgresql+psycopg2://", 1)
    parsed = urlparse(u)
    q = parse_qs(parsed.query)
    if "channel_binding" in q:
        del q["channel_binding"]
        new_query = urlencode({k: v[0] for k, v in q.items()}, doseq=True)
        parsed = parsed._replace(query=new_query)
        u = urlunparse(parsed)
    return u
