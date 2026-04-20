"""
Knowledge Search Tool for NEET Counselling Assistant

This module provides a unified search interface that:
- Performs semantic search on the vector database
- Supports optional metadata filter by state only (document_type / doc_topic are not used for retrieval)
- Returns formatted chunks for LLM consumption
"""

import os
import logging
from typing import Optional, List, Dict, Any, Tuple, Union, Sequence
from dataclasses import dataclass

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.postgres import PGVectorStore

logger = logging.getLogger(__name__)

def log(msg: str):
    """Simple logging helper."""
    print(msg)
    logger.info(msg)


@dataclass
class SearchResult:
    """A single search result with text and metadata."""
    text: str
    score: float
    state: Optional[str]
    document_type: Optional[str]
    doc_topic: Optional[str]
    chunk_category: Optional[str]
    file_name: Optional[str]
    page_label: Optional[str]


@dataclass 
class SearchResponse:
    """Response from knowledge base search."""
    results: List[SearchResult]
    query: str
    filters_applied: Dict[str, str]
    total_results: int


# Singleton for vector store index
_vector_index: Optional[VectorStoreIndex] = None


def get_vector_index() -> VectorStoreIndex:
    """Get or create the vector store index."""
    global _vector_index
    
    if _vector_index is None:
        log("[TOOL] Initializing vector store index...")
        
        # Connection parameters
        db_url = os.getenv("DATABASE_URL", "")
        # Convert asyncpg URL to psycopg2 for sync operations
        sync_url = db_url.replace("postgresql+asyncpg://", "postgresql://")
        
        # Parse connection string
        import urllib.parse
        parsed = urllib.parse.urlparse(sync_url)
        
        vector_store = PGVectorStore.from_params(
            host=parsed.hostname,
            port=parsed.port or 5432,
            database=parsed.path.lstrip("/").split("?")[0],
            user=parsed.username,
            password=parsed.password,
            table_name=os.getenv("PGVECTOR_TABLE_NAME", "neet_assistant"),
            embed_dim=int(os.getenv("EMBEDDING_DIM", "1536")),
        )
        
        embed_model = OpenAIEmbedding(
            model="text-embedding-3-small",
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        _vector_index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=embed_model
        )
        log("[TOOL] Vector store index initialized")
    
    return _vector_index


def warm_knowledge_tool():
    """
    Pre-initialize the vector store index at startup.
    This eliminates the cold-start delay on the first user query.
    """
    try:
        log("[WARM] Pre-initializing knowledge tool vector index...")
        get_vector_index()
        log("[WARM] Knowledge tool ready")
        return True
    except Exception as e:
        log(f"[WARM] Failed to initialize knowledge tool: {e}")
        return False


def _normalize_states_argument(
    state: Optional[str],
    states: Optional[Union[str, Sequence[str]]],
) -> List[str]:
    """
    Build a non-empty list of state/UT filter strings from tool args.
    Empty list means: no state filter (search all).
    """
    out: List[str] = []
    if states is not None:
        if isinstance(states, str):
            parts = [s.strip() for s in states.split(",") if s.strip()]
            out.extend(parts)
        else:
            for s in states:
                t = str(s).strip()
                if t:
                    out.append(t)
    if not out and state and str(state).strip():
        out.append(str(state).strip())
    return out


def build_metadata_filters(state: Optional[str] = None) -> Optional[Any]:
    """
    Build LlamaIndex metadata filters (state only).

    Args:
        state: Filter by state/UT name (e.g. "Maharashtra", "All-India"). Omit to search all states.

    Returns:
        MetadataFilters object or None if no filters
    """
    from llama_index.core.vector_stores import (
        MetadataFilters,
        MetadataFilter,
        FilterOperator,
    )

    if not state:
        return None

    return MetadataFilters(
        filters=[
            MetadataFilter(
                key="state",
                value=state,
                operator=FilterOperator.EQ,
            )
        ],
    )


def _search_knowledge_base_single_state(
    query: str,
    state: Optional[str],
    top_k: int,
) -> SearchResponse:
    """One vector retrieval with optional single-state metadata filter."""
    filters = build_metadata_filters(state)
    filters_applied: Dict[str, str] = {}
    if state:
        filters_applied["state"] = state

    index = get_vector_index()
    retriever = index.as_retriever(similarity_top_k=top_k, filters=filters)

    try:
        nodes: List[NodeWithScore] = retriever.retrieve(query)
        log(f"[TOOL] Retrieved {len(nodes)} results (state={state or 'ALL'})")
    except Exception as e:
        log(f"[TOOL] Search error: {e}")
        if filters:
            log("[TOOL] Retrying without filters...")
            retriever = index.as_retriever(similarity_top_k=top_k)
            nodes = retriever.retrieve(query)
            filters_applied = {}
        else:
            nodes = []

    results: List[SearchResult] = []
    for node in nodes:
        metadata = node.node.metadata if hasattr(node.node, "metadata") else {}
        text = node.node.text if hasattr(node.node, "text") else str(node.node)
        results.append(
            SearchResult(
                text=text,
                score=node.score if hasattr(node, "score") else 0.0,
                state=metadata.get("state"),
                document_type=metadata.get("document_type"),
                doc_topic=metadata.get("doc_topic"),
                chunk_category=metadata.get("chunk_category") or metadata.get("category"),
                file_name=metadata.get("file_name"),
                page_label=metadata.get("page_label"),
            )
        )

    return SearchResponse(
        results=results,
        query=query,
        filters_applied=filters_applied,
        total_results=len(results),
    )


def search_knowledge_base(
    query: str,
    state: Optional[str] = None,
    states: Optional[Union[str, Sequence[str]]] = None,
    top_k: int = 8,
) -> SearchResponse:
    """
    Search the NEET counselling knowledge base.

    Args:
        query: Semantic search query (required)
        state: Optional single state/UT filter (legacy)
        states: Optional list of states/UTs — for comparisons, retrieval runs per state
                and results are merged (deduped, top scores kept). Prefer this over a
                single `state` when the user needs chunks from multiple states.
        top_k: Max results after merge (default: 8)

    Returns:
        SearchResponse with results and metadata
    """
    state_list = _normalize_states_argument(state, states)

    log(f"[TOOL] search_knowledge_base called:")
    log(f"       Query: {query}")
    log(f"       State filter: {state_list if state_list else 'None (all states)'}")

    if len(state_list) <= 1:
        only = state_list[0] if state_list else None
        return _search_knowledge_base_single_state(query, only, top_k)

    # Multi-state: one retrieval per state, merge + dedupe, keep best scores
    per_state_k = max(3, min(top_k, (top_k * 2) // len(state_list)))
    merged: List[SearchResult] = []
    seen: set = set()
    for st in state_list:
        part = _search_knowledge_base_single_state(query, st, per_state_k)
        for r in part.results:
            key = (r.file_name or "", r.page_label or "", (r.text or "")[:200])
            if key in seen:
                continue
            seen.add(key)
            merged.append(r)

    merged.sort(key=lambda x: x.score, reverse=True)
    merged = merged[:top_k]
    return SearchResponse(
        results=merged,
        query=query,
        filters_applied={"states_or": ", ".join(state_list)},
        total_results=len(merged),
    )


def format_search_results_for_llm(response: SearchResponse) -> str:
    """
    Format search results as a string for LLM consumption.
    
    Args:
        response: SearchResponse from search_knowledge_base
    
    Returns:
        Formatted string with all relevant information
    """
    if not response.results:
        return "No relevant information found in the knowledge base."
    
    parts = []
    parts.append(f"Search Query: {response.query}")
    
    if response.filters_applied:
        filter_str = ", ".join(f"{k}={v}" for k, v in response.filters_applied.items())
        parts.append(f"Filters Applied: {filter_str}")
    
    parts.append(f"Results Found: {response.total_results}")
    parts.append("\n--- RETRIEVED CONTENT ---\n")
    
    for i, result in enumerate(response.results, 1):
        # Build metadata line
        meta_parts = []
        if result.state:
            meta_parts.append(f"State: {result.state}")
        if result.document_type:
            meta_parts.append(f"Type: {result.document_type}")
        if result.doc_topic:
            meta_parts.append(f"Topic: {result.doc_topic}")
        if result.file_name:
            meta_parts.append(f"Source: {result.file_name}")
        if result.page_label:
            meta_parts.append(f"Page: {result.page_label}")
        
        meta_line = " | ".join(meta_parts) if meta_parts else "No metadata"
        
        parts.append(f"[{i}] {meta_line}")
        parts.append(f"Score: {result.score:.3f}")
        parts.append(result.text)
        parts.append("")  # Blank line between results
    
    return "\n".join(parts)


def execute_tool_call(
    tool_name: str,
    arguments: Dict[str, Any]
) -> Tuple[str, bool]:
    """
    Execute a tool call from the LLM.
    
    Args:
        tool_name: Name of the tool to execute
        arguments: Arguments for the tool
    
    Returns:
        Tuple of (result_string, success_bool)
    """
    try:
        if tool_name == "search_knowledge_base":
            raw_states = arguments.get("states")
            response = search_knowledge_base(
                query=arguments.get("query", ""),
                state=arguments.get("state"),
                states=raw_states,
            )
            formatted = format_search_results_for_llm(response)
            return formatted, True

        if tool_name == "search_web":
            from services.web_search_tool import web_search_neet, format_web_results_for_llm

            query = arguments.get("query", "")
            results = web_search_neet(query=query, max_results=5)
            formatted = format_web_results_for_llm(query, results)
            return formatted, True

        return f"Unknown tool: {tool_name}", False

    except Exception as e:
        log(f"[TOOL] Execution error ({tool_name}): {e}")
        return f"Error executing {tool_name}: {str(e)}", False


# Convenience function for direct search (bypassing LLM tool call)
def quick_search(
    query: str,
    state: Optional[str] = None,
    top_k: int = 5
) -> str:
    """
    Quick search for direct use without full tool call flow.
    
    Args:
        query: Search query
        state: Optional state filter
        top_k: Number of results
    
    Returns:
        Formatted results string
    """
    response = search_knowledge_base(query, state=state, top_k=top_k)
    return format_search_results_for_llm(response)
