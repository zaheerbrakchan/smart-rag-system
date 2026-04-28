"""
Singleton PGVectorStore (PostgreSQL + pgvector).
Uses LlamaIndex PGVectorStore; physical table name is data_<PGVECTOR_TABLE_NAME>.
"""

import os
from typing import Any, Optional

from services.db_url import normalize_async_database_url, get_database_url, sync_database_url

_vector_store: Optional[Any] = None


def get_embedding_dim() -> int:
    return int(os.getenv("EMBEDDING_DIM", "1536"))


def get_pgvector_table_name() -> str:
    # LlamaIndex creates public.data_<table_name>
    return os.getenv("PGVECTOR_TABLE_NAME", "neet_assistant").lower()


def get_vector_store():
    """Return cached PGVectorStore (LlamaIndex)."""
    global _vector_store
    if _vector_store is None:
        from llama_index.vector_stores.postgres import PGVectorStore

        async_url = normalize_async_database_url(get_database_url())
        sync_url = sync_database_url()

        # Filterable metadata (keep small). doc_topic = admin "sub-category" at upload; chunk_* = per-chunk AI labels.
        indexed_keys = {
            ("document_type", "text"),
            ("state", "text"),
            ("file_id", "text"),
            ("is_faq", "boolean"),
            ("doc_topic", "text"),
        }

        _vector_store = PGVectorStore.from_params(
            database=None,
            host=None,
            port=None,
            user=None,
            password=None,
            connection_string=sync_url,
            async_connection_string=async_url,
            table_name=get_pgvector_table_name(),
            schema_name="public",
            embed_dim=get_embedding_dim(),
            use_jsonb=True,
            perform_setup=True,
            indexed_metadata_keys=indexed_keys,
        )
    return _vector_store


def reset_vector_store_cache() -> None:
    global _vector_store
    _vector_store = None


def count_vectors_sync() -> int:
    """Row count in LlamaIndex PGVector data table (sync)."""
    vs = get_vector_store()
    vs._initialize()
    from sqlalchemy import text

    tbl = vs._table_class.__tablename__
    schema = vs.schema_name
    with vs._session() as session:
        r = session.execute(
            text(f'SELECT COUNT(*) FROM "{schema}"."{tbl}"')
        ).scalar()
        return int(r or 0)
