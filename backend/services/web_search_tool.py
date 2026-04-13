"""
Web Search fallback service for unanswered in-domain queries.

This is only used when:
- Query is in NEET counselling domain
- RAG returns no relevant chunks
- Admin has explicitly enabled the fallback setting
"""

from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


def _web_log(message: str) -> None:
    """Ensure web logs appear in uvicorn terminal output."""
    print(message)
    logger.info(message)


def web_search_neet(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """
    Run a constrained web search for NEET counselling context.

    Returns a list of dictionaries:
    - title
    - url
    - snippet
    """
    from ddgs import DDGS

    _web_log("[WEB] search_web called")
    _web_log(f"[WEB] Original query: {query}")

    # Use LLM-provided query as-is (no auto suffix/appending).
    scoped_query = (query or "").strip()
    _web_log(f"[WEB] Scoped query sent: {scoped_query}")
    results: List[Dict[str, str]] = []

    with DDGS(timeout=12) as ddgs:
        for item in ddgs.text(scoped_query, max_results=max_results):
            title = (item.get("title") or "").strip()
            url = (item.get("href") or "").strip()
            snippet = (item.get("body") or "").strip()
            if not title or not url:
                continue
            results.append(
                {
                    "title": title,
                    "url": url,
                    "snippet": snippet,
                }
            )

    _web_log(f"[WEB] Results received: {len(results)}")
    for i, row in enumerate(results[:5], 1):
        snippet = (row.get("snippet", "") or "").replace("\n", " ").strip()
        if len(snippet) > 220:
            snippet = snippet[:220] + "..."
        _web_log(f"[WEB] Result {i}: {row.get('title', 'N/A')}")
        _web_log(f"[WEB] URL {i}: {row.get('url', 'N/A')}")
        _web_log(f"[WEB] Snippet {i}: {snippet}")

    return results


def format_web_results_for_llm(query: str, results: List[Dict[str, str]]) -> str:
    """Format web search results as model-readable context."""
    if not results:
        return (
            "WEB_SEARCH_RESULT: No reliable web results found.\n"
            "Use the standard fallback response and ask user to check official NTA/state authority."
        )

    lines = [
        f"WEB_SEARCH_QUERY: {query}",
        f"WEB_RESULTS_FOUND: {len(results)}",
        "",
        "--- WEB SEARCH CONTENT (UNVERIFIED, USE CAUTIOUSLY) ---",
    ]

    for i, row in enumerate(results, 1):
        lines.append(f"[{i}] Title: {row.get('title', 'N/A')}")
        lines.append(f"URL: {row.get('url', 'N/A')}")
        lines.append(f"Snippet: {row.get('snippet', '')}")
        lines.append("")

    lines.append(
        "INSTRUCTION: Use these results only if relevant to the user's question, "
        "prefer official sources, and cite URLs in the answer."
    )
    return "\n".join(lines)

