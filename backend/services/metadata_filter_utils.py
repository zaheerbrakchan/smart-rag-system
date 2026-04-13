"""
Convert legacy Pinecone-style metadata filters to LlamaIndex MetadataFilters (for PGVectorStore).
"""

from __future__ import annotations

from typing import Any, Dict, List

from llama_index.core.vector_stores.types import (
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)


def pinecone_filter_to_metadata_filters(pc: Dict[str, Any]) -> MetadataFilters:
    """
    Pinecone examples:
      {"document_type": {"$eq": "nta_bulletin"}}
      {"$and": [{"document_type": {"$eq": "state_counseling"}}, {"state": {"$eq": "Karnataka"}}]}
    """
    if not pc:
        return MetadataFilters(filters=[])

    if "$and" in pc:
        parts: List[MetadataFilters] = []
        for sub in pc["$and"]:
            parts.append(pinecone_filter_to_metadata_filters(sub))
        flat: List[MetadataFilter] = []
        for p in parts:
            flat.extend(p.filters)
        return MetadataFilters(filters=flat, condition=FilterCondition.AND)

    filters: List[MetadataFilter] = []
    for key, val in pc.items():
        if isinstance(val, dict) and "$eq" in val:
            filters.append(
                MetadataFilter(
                    key=key,
                    value=val["$eq"],
                    operator=FilterOperator.EQ,
                )
            )
        else:
            raise ValueError(f"Unsupported Pinecone filter fragment: {key}={val}")

    return MetadataFilters(filters=filters)
