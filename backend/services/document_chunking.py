"""
Page list preparation before LlamaIndex: drop blank pages and merge multi-page
fee tables for college_info + fees uploads.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

# Ignore pages with almost no extractable text (scanned blank, divider, etc.)
MIN_PAGE_TEXT_CHARS = 40


def filter_blank_pages(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for p in pages:
        t = (p.get("text") or "").strip()
        if len(t) >= MIN_PAGE_TEXT_CHARS:
            np = dict(p)
            np["text"] = t
            out.append(np)
    return out


def page_starts_new_college_fee_block(text: str) -> bool:
    """
    Heuristic: a NEW college / fee section usually repeats a header like COLLEGE NAME:
    Continuation pages (next page of same table) typically do not.
    """
    head = (text or "")[:1800]
    hl = head.lower()
    # Strong header lines near the top of the page
    if "college name" in hl[:700] or "institute name" in hl[:700] or "institution name" in hl[:700]:
        return True
    # New numbered institute entry (common in PDFs)
    if re.search(r"^\s*\d+[\.)]\s+[A-Za-z]", head[:500], re.MULTILINE):
        return True
    return False


def merge_college_fee_pages(
    pages: List[Dict[str, Any]], document_type: str, doc_topic: str
) -> List[Dict[str, Any]]:
    """
    For state college/fee PDFs tagged as fees: merge consecutive pages that belong to the
    same fee block (e.g. page 3 + 4 for one unified Maharashtra fee table).
    When a page looks like a new college/header block, start a new merged document.
    """
    if document_type != "college_info" or doc_topic != "fees" or len(pages) <= 1:
        single: List[Dict[str, Any]] = []
        for p in pages:
            d = dict(p)
            d["page_end"] = p["page_num"]
            single.append(d)
        return single

    merged: List[Dict[str, Any]] = []
    buf = pages[0]["text"]
    p_start = pages[0]["page_num"]
    p_end = pages[0]["page_num"]

    for i in range(1, len(pages)):
        t = pages[i]["text"]
        pn = pages[i]["page_num"]
        if page_starts_new_college_fee_block(t):
            merged.append(
                {"text": buf, "page_num": p_start, "page_end": p_end}
            )
            buf = t
            p_start = pn
            p_end = pn
        else:
            buf = buf + "\n\n" + t
            p_end = pn

    merged.append({"text": buf, "page_num": p_start, "page_end": p_end})
    return merged


def prepare_pages_for_indexing(
    pages: List[Dict[str, Any]], document_type: str, doc_topic: str
) -> List[Dict[str, Any]]:
    """Filter empties, then optional merge for college fee PDFs."""
    pages = filter_blank_pages(pages)
    return merge_college_fee_pages(pages, document_type, doc_topic)


def format_page_label(page_data: Dict[str, Any]) -> str:
    start = page_data["page_num"]
    end = page_data.get("page_end", start)
    if end is None or end == start:
        return str(start)
    return f"{start}-{end}"


def get_chunk_settings_for_document(document_type: str, doc_topic: str) -> Tuple[int, int]:
    """Larger windows for merged college fee text so tables stay in fewer chunks."""
    if document_type == "college_info" and doc_topic == "fees":
        return 4096, 220
    return 1024, 160
