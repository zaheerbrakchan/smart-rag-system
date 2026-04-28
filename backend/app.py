"""
RAG Chatbot - FastAPI Backend (OpenAI + PostgreSQL pgvector)
Production RAG with Admin Document Management & Smart Query Routing
"""

import os
import re
import time
import json
import shutil
import uuid
import asyncio
import tempfile
import traceback
import sys
import logging
import threading
from pathlib import Path
from datetime import datetime
from datetime import timedelta
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Tuple, Any, AsyncGenerator
from dotenv import load_dotenv

# Use uvicorn's logger (same one that shows "INFO: Uvicorn running on...")
uvicorn_logger = logging.getLogger("uvicorn.error")

def log(msg):
    """Log using uvicorn's logger"""
    uvicorn_logger.info(msg)


def _v2_timing_log_enabled() -> bool:
    """Per-request phase timings for /chat/v2/stream (disable with V2_TIMING_LOG=false)."""
    return os.getenv("V2_TIMING_LOG", "true").lower() in ("1", "true", "yes")


def _elapsed_ms(since: float) -> float:
    return (time.perf_counter() - since) * 1000.0


V2_STREAM_TOKEN_DELAY_SEC = float(os.getenv("V2_STREAM_TOKEN_DELAY_SEC", "0"))
V2_TOOL_CONTEXT_CHAR_LIMIT = int(os.getenv("V2_TOOL_CONTEXT_CHAR_LIMIT", "12000"))
V2_FINAL_MAX_TOKENS = int(os.getenv("V2_FINAL_MAX_TOKENS", "900"))
V2_TOOL_DEBUG_LOG = os.getenv("V2_TOOL_DEBUG_LOG", "true").lower() in ("1", "true", "yes")


def _trim_tool_result_for_model(raw: str, limit: int = V2_TOOL_CONTEXT_CHAR_LIMIT) -> str:
    text = str(raw or "")
    if len(text) <= limit:
        return text
    return text[:limit] + "\n\n[Tool output truncated for model context]"


def _log_v2_tool_debug(msg: str) -> None:
    if V2_TOOL_DEBUG_LOG:
        log(msg)


def _summarize_messages_for_debug(messages: List[Dict[str, Any]], tail: int = 6) -> str:
    """
    Compact, safe debug summary of messages passed to LLM in tool loop.
    Shows role and truncated content/tool marker only.
    """
    out: List[str] = []
    for m in messages[-tail:]:
        role = str(m.get("role", "?"))
        if m.get("tool_calls"):
            out.append(f"{role}:<tool_calls>")
            continue
        content = str(m.get("content") or "").replace("\n", " ").strip()
        if len(content) > 120:
            content = content[:120] + "..."
        out.append(f"{role}:{content}")
    return " | ".join(out)


def _log_final_llm_messages_snapshot(messages: List[Dict[str, Any]], *, label: str = "final") -> None:
    """
    Verbose snapshot of messages passed to the final answer LLM call.
    Trim each message to keep logs readable while preserving structure/context.
    """
    lines: List[str] = []
    for i, m in enumerate(messages, 1):
        role = str(m.get("role", "?"))
        if m.get("tool_calls"):
            calls = m.get("tool_calls") or []
            preview = []
            for c in calls[:5]:
                fn = (((c or {}).get("function") or {}).get("name")) or "unknown_tool"
                args = (((c or {}).get("function") or {}).get("arguments")) or "{}"
                args = str(args).replace("\n", " ")
                if len(args) > 220:
                    args = args[:220] + "..."
                preview.append(f"{fn}({args})")
            call_text = "; ".join(preview)
            lines.append(f"[{i}] role={role} tool_calls={call_text}")
            continue

        content = str(m.get("content") or "").strip().replace("\n", "\\n")
        if len(content) > 900:
            content = content[:900] + "...[truncated]"
        lines.append(f"[{i}] role={role} content={content}")

    _log_v2_tool_debug(
        f"[V2][DBG] {label}_llm_messages_start\n" + "\n".join(lines) + f"\n[V2][DBG] {label}_llm_messages_end"
    )


def _build_sufficiency_context(messages: List[Dict[str, Any]], current_question: str, max_turns: int = 4) -> str:
    """
    Build compact conversational context for sufficiency checks so short follow-ups
    are interpreted with the correct topic from recent turns.
    """
    turns: List[str] = []
    for m in messages:
        role = str(m.get("role", ""))
        if role not in {"user", "assistant"}:
            continue
        if m.get("tool_calls"):
            continue
        content = str(m.get("content") or "").strip()
        if not content:
            continue
        if len(content) > 300:
            content = content[:300] + "..."
        turns.append(f"{role}: {content}")
    if max_turns > 0:
        turns = turns[-max_turns:]
    return (
        "Conversation context (recent):\n"
        + ("\n".join(turns) if turns else "(none)")
        + f"\n\nCurrent user message:\n{current_question}"
    )


def build_web_fallback_query_with_llm(
    client,
    *,
    user_question: str,
    conversation_context: str,
    kb_tool_result: str,
) -> str:
    """
    Rewrite contextual/vague user text into a web-search-ready query.
    Keeps logic generic and prompt-driven (no entity hardcoding).
    """
    base_query = (user_question or "").strip()
    if not base_query:
        return base_query
    try:
        kb_compact = _trim_tool_result_for_model(kb_tool_result, limit=2500)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Rewrite the user's latest question into ONE concise web search query.\n"
                        "Rules:\n"
                        "- Resolve references like above/these/those/same using conversation context.\n"
                        "- Keep the exact topic and intent from latest user turn.\n"
                        "- Include explicit entities when inferable from context.\n"
                        "- Prefer NEET UG India counselling/admission wording and official source intent.\n"
                        "- Do NOT invent entities not present in context.\n"
                        "- Return JSON only: {\"query\": \"...\"}."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"LATEST_USER_QUESTION:\n{base_query}\n\n"
                        f"CONVERSATION_CONTEXT:\n{conversation_context}\n\n"
                        f"KB_RETRIEVAL_SUMMARY:\n{kb_compact}\n\n"
                        "Return JSON only."
                    ),
                },
            ],
            temperature=0,
            max_tokens=120,
        )
        raw = (response.choices[0].message.content or "").strip()
        parsed = json.loads(raw)
        query = str(parsed.get("query", "")).strip()
        if query:
            return query
    except Exception as err:
        log(f"[V2] ⚠️ Web fallback query rewrite failed; using raw question: {err}")
    return base_query


def plan_targeted_web_gap_fill_with_llm(
    client,
    *,
    user_question: str,
    conversation_context: str,
    kb_tool_result: str,
) -> Dict[str, Any]:
    """
    Ask LLM to identify missing entities (if any) and produce targeted web queries.
    Returns JSON-like dict:
    {
      "requested_entities": [...],
      "covered_entities": [...],
      "missing_entities": [...],
      "web_queries": [...]
    }
    """
    try:
        kb_compact = _trim_tool_result_for_model(kb_tool_result, limit=3500)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You plan retrieval gap-fill for NEET counselling.\n"
                        "Given user question + conversation context + KB retrieval text:\n"
                        "1) identify requested entities for this turn,\n"
                        "2) identify entities sufficiently covered by KB text,\n"
                        "3) list missing entities,\n"
                        "4) create focused web queries ONLY for missing entities.\n"
                        "Rules:\n"
                        "- Do not invent entities.\n"
                        "- Keep one query per missing entity when possible.\n"
                        "- Include topic words from latest turn (fee structure/cutoff/etc.).\n"
                        "- If nothing is missing, return empty web_queries.\n"
                        "Return JSON only with keys:\n"
                        "requested_entities (array), covered_entities (array), missing_entities (array), web_queries (array)."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"LATEST_USER_QUESTION:\n{user_question}\n\n"
                        f"CONVERSATION_CONTEXT:\n{conversation_context}\n\n"
                        f"KB_RETRIEVAL_TEXT:\n{kb_compact}\n\n"
                        "Return JSON only."
                    ),
                },
            ],
            temperature=0,
            max_tokens=260,
        )
        raw = (response.choices[0].message.content or "").strip()
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except Exception as err:
        log(f"[V2] ⚠️ Targeted web gap-fill planning failed: {err}")
        return {}


def _normalize_entity_text(text: str) -> str:
    s = (text or "").lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _extract_target_entity_from_kb_query(query: str) -> str:
    q = _normalize_entity_text(query)
    # Remove common intent words so entity tokens dominate.
    noise = {
        "fee", "fees", "structure", "mbbs", "bds", "ug", "neet", "college",
        "medical", "state", "quota", "admission", "for", "of", "in", "and",
        "uttar", "pradesh", "general", "obc", "sc", "st", "ews"
    }
    tokens = [t for t in q.split() if t not in noise and len(t) > 2]
    return " ".join(tokens[:6]).strip()


def _kb_result_mentions_entity(entity_hint: str, kb_result: str) -> bool:
    """
    Conservative entity check:
    - Require at least 2 meaningful tokens from entity_hint to appear in KB result.
    - Helps reject nearest-neighbor chunks for different colleges.
    """
    hint = _normalize_entity_text(entity_hint)
    hay = _normalize_entity_text(kb_result)
    if not hint or not hay:
        return False
    tokens = [t for t in hint.split() if len(t) > 2]
    if not tokens:
        return False
    matched = sum(1 for t in set(tokens) if re.search(rf"\b{re.escape(t)}\b", hay))
    return matched >= min(2, len(set(tokens)))


async def _stream_chat_completion_text(
    client,
    *,
    model: str,
    messages: List[Dict[str, Any]],
    temperature: float = 0.3,
    max_tokens: int = V2_FINAL_MAX_TOKENS,
):
    """
    Yield text deltas from OpenAI chat completion stream.
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    done = object()
    error_holder: List[Exception] = []

    def _producer() -> None:
        try:
            stream = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
            for chunk in stream:
                try:
                    delta = chunk.choices[0].delta.content
                except Exception:
                    delta = None
                if delta:
                    loop.call_soon_threadsafe(queue.put_nowait, delta)
        except Exception as e:
            error_holder.append(e)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, done)

    thread = threading.Thread(target=_producer, daemon=True)
    thread.start()

    while True:
        item = await queue.get()
        if item is done:
            break
        yield item

    if error_holder:
        raise error_holder[0]


SUPPORTED_LANGUAGES = {"en", "hi", "mr"}


def _v2_async_cutoff_router_enabled() -> bool:
    # Backward compatible with older env name.
    raw = os.getenv("V2_ASYNC_CUTOFF_ROUTER", os.getenv("V2_PARALLEL_CLASSIFIERS", "false"))
    return str(raw).lower() in ("1", "true", "yes")


def _v2_direct_non_english_enabled() -> bool:
    return os.getenv("V2_DIRECT_NON_ENGLISH", "false").lower() in ("1", "true", "yes")


def _normalize_language_code(raw: Optional[str]) -> str:
    code = str(raw or "en").strip().lower()
    if code in SUPPORTED_LANGUAGES:
        return code
    return "en"


def _detect_user_language_sync(text: str) -> str:
    """
    Detect user language among en/hi/mr.
    """
    if not text or not text.strip():
        return "en"
    # English greeting "hi" clashes with ISO code `hi` for Hindi — treat known greetings as English.
    if _is_greeting_only(text):
        return "en"
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Detect the natural language of the USER TEXT. Return JSON only: {\"language\": \"en|hi|mr\"}.\n"
                        "- Use **en** for English (including the standalone greeting words hi, hey, hello).\n"
                        "- Use **hi** only when the text is actually **Hindi** (Devanagari script or clear Hindi words).\n"
                        "- Use **mr** only for **Marathi** (Devanagari typical Marathi forms).\n"
                        "Important: the token \"hi\" as an English hello is **en**, not Hindi. "
                        "The value \"hi\" in JSON must mean Hindi **content**, not the greeting \"hi\"."
                    ),
                },
                {"role": "user", "content": text[:800]},
            ],
            temperature=0,
            max_tokens=40,
        )
        raw = (response.choices[0].message.content or "").strip()
        parsed = json.loads(raw)
        return _normalize_language_code(str(parsed.get("language", "en")))
    except Exception as e:
        log(f"[V2] ⚠️ Language detection failed, defaulting en: {e}")
        return "en"


def _is_markdown_table_separator_line(line: str) -> bool:
    """True for GFM-style header separator rows like |---|:---:|."""
    s = line.strip()
    if not (s.startswith("|") and s.endswith("|")):
        return False
    cells = [c.strip() for c in s.strip("|").split("|")]
    if not cells or not any(cells):
        return False
    for c in cells:
        if not c:
            continue
        if not re.fullmatch(r":?-{1,}:?", c):
            return False
    return True


def _document_contains_pipe_table(text: str) -> bool:
    """Heuristic: at least two pipe-delimited table rows (avoids one-line false positives)."""
    run = 0
    for raw in text.split("\n"):
        st = raw.strip()
        if st.startswith("|") and st.endswith("|") and st.count("|") >= 2:
            run += 1
            if run >= 2:
                return True
        else:
            run = 0
    return False


def _translate_plain_chunk_sync(text: str, source_lang: str, target_lang: str) -> str:
    """Translate prose / non-table markdown in one shot."""
    if not text or source_lang == target_lang:
        return text
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        max_out = min(8192, max(900, len(text) // 2 + 1200))
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a precise translator for NEET counselling chat.\n"
                        "Preserve meaning, numbers, bullet/heading markdown, and blank lines.\n"
                        "Translate only the natural-language content.\n"
                        "Return translated text only."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Source language: {source_lang}\n"
                        f"Target language: {target_lang}\n\n"
                        f"Text:\n{text}"
                    ),
                },
            ],
            temperature=0,
            max_tokens=max_out,
        )
        translated = (response.choices[0].message.content or "").strip()
        return translated or text
    except Exception as e:
        log(f"[V2] ⚠️ Translation failed ({source_lang}->{target_lang}): {e}")
        return text


def _translate_markdown_table_block_lines_sync(lines: List[str], target_lang: str) -> List[str]:
    """
    Translate a contiguous Markdown pipe table without merging rows.
    Separator rows are copied verbatim so column alignment stays valid.
    """
    if not lines:
        return []
    code = _normalize_language_code(target_lang)
    if code == "en":
        return list(lines)
    lang_name = {"hi": "Hindi (Devanagari)", "mr": "Marathi (Devanagari)"}.get(code, target_lang)
    pipe_counts = [ln.count("|") for ln in lines]
    locked = [_is_markdown_table_separator_line(ln) for ln in lines]
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        n = len(lines)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"You translate Markdown **pipe table rows** to {lang_name}.\n"
                        f"Return JSON only: {{\"lines\": [string, ...]}} with exactly {n} strings.\n"
                        "Rules:\n"
                        "- lines[i] is the translation of input line i (same index).\n"
                        "- NEVER merge two input lines into one string. One input row → one output string.\n"
                        "- Each output line must contain the **same number of pipe '|' characters** as input line i.\n"
                        "- For markdown separator rows (only |, -, :, spaces in cells), copy the input line **exactly**.\n"
                        "- Translate words inside cells; keep numbers, AIR, round codes like R1, state codes like BIHAR unchanged.\n"
                        "- Do not add text before the first | or after the last | on a row.\n"
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps({"lines": lines}, ensure_ascii=False),
                },
            ],
            temperature=0.1,
            max_tokens=8192,
        )
        raw = (resp.choices[0].message.content or "").strip()
        parsed = json.loads(raw)
        outs = parsed.get("lines")
        if not isinstance(outs, list) or len(outs) != n:
            log("[V2] ⚠️ Table translation JSON shape invalid; keeping English table")
            return list(lines)
        merged: List[str] = []
        for i, orig in enumerate(lines):
            if locked[i]:
                merged.append(orig)
                continue
            cand = str(outs[i] if i < len(outs) else "").strip()
            if cand.count("|") != pipe_counts[i]:
                merged.append(orig)
            else:
                merged.append(cand)
        return merged
    except Exception as e:
        log(f"[V2] ⚠️ Table-row translation failed: {e}")
        return list(lines)


def _translate_text_preserving_pipe_tables_sync(text: str, source_lang: str, target_lang: str) -> str:
    """Walk the document; translate pipe-table row blocks line-wise, other chunks with plain translator."""
    if not text or source_lang == target_lang:
        return text
    lines = text.split("\n")
    out: List[str] = []
    i = 0
    n = len(lines)
    while i < n:
        st = lines[i].strip()
        if st.startswith("|") and st.endswith("|") and st.count("|") >= 2:
            j = i
            while j < n:
                sj = lines[j].strip()
                if sj.startswith("|") and sj.endswith("|") and sj.count("|") >= 2:
                    j += 1
                else:
                    break
            out.extend(_translate_markdown_table_block_lines_sync(lines[i:j], target_lang))
            i = j
        else:
            j = i
            while j < n:
                s2 = lines[j].strip()
                if s2.startswith("|") and s2.endswith("|") and s2.count("|") >= 2:
                    break
                j += 1
            chunk = "\n".join(lines[i:j])
            if chunk.strip():
                out.extend(_translate_plain_chunk_sync(chunk, source_lang, target_lang).split("\n"))
            else:
                out.extend(lines[i:j])
            i = j
    return "\n".join(out)


def _translate_text_sync(text: str, source_lang: str, target_lang: str) -> str:
    if not text or source_lang == target_lang:
        return text
    if _normalize_language_code(target_lang) != "en" and _document_contains_pipe_table(text):
        return _translate_text_preserving_pipe_tables_sync(text, source_lang, target_lang)
    return _translate_plain_chunk_sync(text, source_lang, target_lang)


async def v2_background_save_conversation_turn(
    conversation_id: int,
    question: str,
    assistant_content: str,
    response_time_ms: int,
    *,
    sources: Optional[List[Dict]] = None,
    was_faq_match: bool = False,
    faq_confidence: Optional[float] = None,
) -> None:
    """
    Persist user + assistant messages after the client has received `done` (SSE).
    Does not block stream completion; errors are logged only.
    """
    from database.connection import async_session_maker
    from services.conversation_memory import save_message_to_db

    t_save = time.perf_counter()
    try:
        async with async_session_maker() as conv_db:
            await save_message_to_db(conv_db, conversation_id, "user", question)
            await save_message_to_db(
                conv_db,
                conversation_id,
                "assistant",
                assistant_content,
                sources=sources,
                was_faq_match=was_faq_match,
                faq_confidence=faq_confidence,
                response_time_ms=response_time_ms,
            )
        log(
            f"[V2] 💾 Background save completed conv={conversation_id} "
            f"({_elapsed_ms(t_save):.0f}ms)"
        )
    except Exception as e:
        log(f"[V2] ⚠️ Background save failed conv={conversation_id}: {e}")


async def v2_background_update_conversation_context(
    conversation_id: int,
    medbuddy_context: Dict[str, object],
) -> None:
    """Persist Med Buddy lightweight conversation orchestration state."""
    from database.connection import async_session_maker

    try:
        async with async_session_maker() as db:
            convo = await db.get(Conversation, conversation_id)
            if not convo:
                return
            context_data = dict(convo.context_data or {})
            context_data["medbuddy"] = medbuddy_context
            convo.context_data = context_data
            await db.commit()
    except Exception as err:
        log(f"[V2] ⚠️ Context update failed conv={conversation_id}: {err}")


async def _v2_conversation_needs_title(conversation_id: int) -> bool:
    """Return True when title is empty or still a generic fallback (e.g., Chat 147)."""
    from database.connection import async_session_maker
    try:
        async with async_session_maker() as db:
            convo = await db.get(Conversation, conversation_id)
            if not convo:
                return False
            title = (convo.title or "").strip()
            if not title:
                return True
            # Regenerate if title is still a placeholder from UI fallback naming.
            return bool(re.fullmatch(r"(?i)\s*chat\s*#?\s*\d+\s*", title))
    except Exception as err:
        log(f"[V2] ⚠️ Could not check title state conv={conversation_id}: {err}")
        return False


async def v2_background_generate_conversation_title(
    conversation_id: int,
    question: str,
    *,
    log_label: str = "default",
) -> None:
    """
    Generate/persist conversation title after stream completion without blocking UX.
    """
    try:
        should_generate_title = (
            conversation_id
            and not _is_greeting_only(question)
            and await _v2_conversation_needs_title(conversation_id)
        )
        if not should_generate_title:
            return

        from database.connection import async_session_maker
        from services.conversation_memory import generate_conversation_title, update_conversation_title

        generated_title = await generate_conversation_title(question)
        async with async_session_maker() as title_db:
            await update_conversation_title(title_db, conversation_id, generated_title)
        log(f"[V2] 🏷️ Generated title ({log_label}): {generated_title}")
    except Exception as title_err:
        log(f"[V2] ⚠️ Title generation error ({log_label}): {title_err}")


def sse_tokens_preserving_formatting(text: str):
    """
    Yield text chunks for SSE without destroying newlines or markdown structure.
    Using str.split() drops newlines and merges lines, so ### headings and lists
    end up on one line and the UI cannot render Markdown properly.
    """
    if not text:
        return
    for part in re.split(r"(\s+)", text):
        if part:
            yield part


def clarification_followup_message(user_state: Optional[str]) -> str:
    """Conversational copy when the router needs central vs state scope (no UI buttons)."""
    text = (
        "To give you the most accurate answer, could you tell me whether you're asking about **All India (AIQ / MCC)** "
        "counselling, or about **a specific state's** rules?\n\n"
        "Just reply in your own words—for example *All India quota*, *MCC*, or a state name like *Karnataka* or *Tamil Nadu*."
    )
    if user_state:
        text += (
            f"\n\nIf you want information for the state on your profile (**{user_state}**), you can say that in your message too."
        )
    return text


MEDBUDDY_DEFAULT_REPLIES = [
    "NEET exam guidance",
    "Counselling process",
    "College shortlist",
    "College fee structure",
]


def _medbuddy_default_replies_for_language(lang: str) -> List[str]:
    code = _normalize_language_code(lang)
    if code == "hi":
        return [
            "नीट परीक्षा मार्गदर्शन",
            "काउंसलिंग प्रक्रिया",
            "कॉलेज शॉर्टलिस्ट",
            "कॉलेज फीस संरचना",
        ]
    if code == "mr":
        return [
            "नीट परीक्षा मार्गदर्शन",
            "कौन्सेलिंग प्रक्रिया",
            "महाविद्यालय शॉर्टलिस्ट",
            "फीस संरचना",
        ]
    return list(MEDBUDDY_DEFAULT_REPLIES)


def _session_close_suggested_replies(lang: str) -> List[str]:
    code = _normalize_language_code(lang)
    if code == "hi":
        return [
            "नया प्रश्न शुरू करें",
            "फीस की तुलना",
            "काउंसलिंग प्रक्रिया",
        ]
    if code == "mr":
        return [
            "नवा प्रश्न सुरू करा",
            "फी तुलना",
            "कौन्सेलिंग प्रक्रिया",
        ]
    return ["Start new query", "Compare fees", "Counselling process"]


def _chip_generation_language_rules(output_language: str) -> str:
    """System-prompt fragment so quick-reply chips match the user's UI language."""
    code = _normalize_language_code(output_language)
    if code == "hi":
        return (
            "LANGUAGE (mandatory): Write **every** chip in **Hindi** using **Devanagari** script only. "
            "Do not use English in chip strings.\n"
            "- Keep each chip a very short user question (one brief line; about as compact as 5–8 English words).\n"
            "- RETRIEVED_EVIDENCE may be English; chips must still be grounded Hindi questions about the same facts.\n"
        )
    if code == "mr":
        return (
            "LANGUAGE (mandatory): Write **every** chip in **Marathi** using **Devanagari** script only. "
            "Do not use English in chip strings.\n"
            "- Keep each chip a very short user question (one brief line; about as compact as 5–8 English words).\n"
            "- RETRIEVED_EVIDENCE may be English; chips must still be grounded Marathi questions about the same facts.\n"
        )
    return (
        "LANGUAGE: Write every chip in **English**.\n"
        "- Each reply at most 6 words.\n"
    )


def _extract_college_hint_for_chips(*texts: str) -> Optional[str]:
    """Best-effort college name hint for deterministic chip fallback."""
    pattern = re.compile(r"\b(?:AIIMS|GMC)\s+[A-Za-z&\-\s]{2,40}", re.IGNORECASE)
    for text in texts:
        if not text:
            continue
        match = pattern.search(text)
        if not match:
            continue
        name = re.sub(r"\s+", " ", match.group(0)).strip(" .,:;")
        if name:
            # Normalize presentation while preserving recognizable acronym
            parts = name.split(" ", 1)
            if len(parts) == 2:
                return f"{parts[0].upper()} {parts[1].strip()}"
            return name.upper()
    return None


def _fallback_contextual_chips(
    user_question: str,
    assistant_response: str,
    output_language: str,
) -> List[str]:
    """Deterministic fallback when model/filters produce no chips."""
    code = _normalize_language_code(output_language)
    college = _extract_college_hint_for_chips(user_question, assistant_response)
    if code != "en":
        # Keep non-English fallback predictable and safe.
        return _medbuddy_default_replies_for_language(output_language)[:2]
    if college:
        return [
            f"What is total fee at {college}?",
            f"Compare {college} with another college",
        ]
    return [
        "Want fee breakup details?",
        "Compare with another college?",
    ]



def _combine_retrieval_for_suggestion_chips(
    kb_text: Optional[str],
    web_text: Optional[str],
) -> Optional[str]:
    """Merge KB + web tool outputs for chip grounding (truncated for context limits)."""
    kb_text = (kb_text or "").strip()
    web_text = (web_text or "").strip()
    if not kb_text and not web_text:
        return None
    parts: List[str] = []
    if kb_text:
        parts.append("=== KNOWLEDGE BASE (retrieved) ===\n" + kb_text[:8000])
    if web_text:
        parts.append("=== WEB SEARCH (retrieved) ===\n" + web_text[:6000])
    return "\n\n".join(parts)[:14000]


def _filter_chips_not_supported_by_evidence(replies: List[str], evidence: Optional[str]) -> List[str]:
    """Drop follow-up chips whose topic is not substantively present in retrieval text."""
    if not evidence or not replies:
        return replies
    ev = _normalize_text(evidence)
    out: List[str] = []
    for r in replies:
        q = _normalize_text(r)
        skip = False
        if any(phrase in q for phrase in (
            "other colleges", "other college", "another college", "another state",
            "different college", "different colleges", "more colleges", "more college",
        )):
            skip = True
        if not skip and "placement" in q and "placement" not in ev:
            skip = True
        if not skip and "internship" in q and "internship" not in ev:
            skip = True
        asks_payment_modality = (
            ("payment" in q and ("option" in q or "method" in q or "mode" in q))
            or "how to pay" in q or "pay online" in q or "pay fees" in q
        )
        if asks_payment_modality:
            if not any(k in ev for k in (
                "demand draft", "neft", "rtgs", "upi", "online payment", "bank",
                "cheque", "cash payment", "installment", "emi", "mode of payment",
                "payment mode", "payable at", "pay through",
            )):
                skip = True
        if not skip and any(k in q for k in ("scholarship", "fee waiver", "financial aid")):
            if not any(k in ev for k in ("scholarship", "waiver", "freeship", "fee concession")):
                skip = True
        if not skip and "refund" in q:
            if "refund" not in ev and "forfeit" not in ev:
                skip = True
        if not skip:
            out.append(r)
    return out

async def _generate_contextual_suggested_replies(
    user_question: str,
    assistant_response: str,
    med_ctx: Optional[Dict[str, object]] = None,
    retrieval_evidence: Optional[str] = None,
    output_language: str = "en",
) -> List[str]:
    """Generate optional quick-reply chips grounded in retrieval evidence."""
    response_text = str(assistant_response or "")
    question_text = str(user_question or "")
    response_norm = _normalize_text(response_text)
    evidence_text = (retrieval_evidence or "").strip()

    # Skip chips when collecting profile data
    collection_signals = [
        "share either your neet score", "share your neet category",
        "tell me your home state", "please tell me your home state",
        "please share your neet rank", "neet air rank or",
        "neet score/marks", "home state (domicile", "which state(s) should i",
    ]
    if any(sig in response_norm for sig in collection_signals):
        return []

    if _is_greeting_only(question_text):
        return _medbuddy_default_replies_for_language(output_language)

    max_words = 6 if _normalize_language_code(output_language) == "en" else 14

    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        topic = str((med_ctx or {}).get("last_topic") or "")

        grounding_rules = (
            "GROUNDING: Chips must be answerable from RETRIEVED_EVIDENCE only. "
            "Never suggest 'other colleges', 'another state' unless those are in the evidence. "
            "Stay on the same college/entity as the response."
            if evidence_text else
            "Stay in the same topic lane. Prefer 0-3 chips."
        )

        user_payload: Dict[str, object] = {
            "current_user_question": question_text,
            "assistant_response": response_text[:2000],
            "last_topic": topic,
            "ui_language": _normalize_language_code(output_language),
        }
        if evidence_text:
            user_payload["RETRIEVED_EVIDENCE"] = evidence_text[:12000]

        lang_rules = _chip_generation_language_rules(output_language)

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Generate concise suggested reply chips for a NEET counselling chat UI. "
                        "Return JSON only: {\"replies\": [\"...\", ...]}. "
                        "At most 3 chips. Each chip is a short natural follow-up the user might click. "
                        "Direct action style: 'Check cutoff for AIIMS Bhopal', 'Show UP government only'. "
                        "Never repeat what the bot already answered. "
                        "If bot asked for input (rank/state/category), return empty list. "
                        + lang_rules
                        + grounding_rules
                    ),
                },
                {"role": "user", "content": json.dumps(user_payload)},
            ],
            temperature=0.2,
            max_tokens=200,
        )
        raw = (resp.choices[0].message.content or "").strip()
        parsed = json.loads(raw)
        replies = parsed.get("replies", [])
        if not isinstance(replies, list):
            return []

        cleaned: List[str] = []
        seen: set = set()
        for r in replies:
            text = str(r or "").strip()
            key = text.casefold()
            if not text or key in seen:
                continue
            if len(text.split()) > max_words:
                continue
            cleaned.append(text)
            seen.add(key)
            if len(cleaned) >= 3:
                break

        if evidence_text and cleaned:
            filtered = _filter_chips_not_supported_by_evidence(cleaned, evidence_text)
            cleaned = filtered if filtered else cleaned[:2]

        if not cleaned:
            cleaned = _fallback_contextual_chips(question_text, response_text, output_language)
        return cleaned
    except Exception as e:
        log(f"[V2] ⚠️ Suggested replies generation failed: {e}")
        return []



MEDBUDDY_CAPS = {
    "cutoff": True,
    "mbbs_abroad": os.getenv("MEDBUDDY_ENABLE_MBBS_ABROAD", "false").lower() in ("1", "true", "yes"),
}


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip().lower())


def _contains_any(text: str, phrases: List[str]) -> bool:
    return any(p in text for p in phrases)


def _detect_greeting_only(question: str) -> Optional[str]:
    q = _normalize_text(question)
    greeting_words = {
        "hi": "hi",
        "hello": "hello",
        "hey": "hey",
        "hii": "hi",
        "good morning": "good morning",
        "good afternoon": "good afternoon",
        "good evening": "good evening",
    }
    return greeting_words.get(q)


def _is_greeting_only(question: str) -> bool:
    return _detect_greeting_only(question) is not None


def _is_session_close_intent(question: str) -> bool:
    q = _normalize_text(question)
    return _contains_any(
        q,
        ["bye", "goodbye", "thanks i am done", "i am done", "that's all", "thats all", "see you"],
    )


def _ensure_medbuddy_context(raw_context_data: Optional[Dict[str, object]]) -> Dict[str, object]:
    data = dict(raw_context_data or {})
    med = dict(data.get("medbuddy") or {})
    med.setdefault("stage", "normal_qa")
    med.setdefault("onboarding", {})
    med.setdefault("last_topic", None)
    med.setdefault("last_state", None)
    med.setdefault("last_activity_at", datetime.utcnow().isoformat())
    return med


def _extract_onboarding_updates(question: str) -> Dict[str, str]:
    q = _normalize_text(question)
    updates: Dict[str, str] = {}

    if re.search(r"\b\d{2,7}\b", q) or any(k in q for k in ["rank", "score", "marks", "expected rank"]):
        updates["rank_or_score"] = "provided"

    if any(x in q for x in ["general", "obc", "sc", "st", "ews", "pwd", "pwbd"]):
        if "obc" in q:
            updates["category"] = "OBC"
        elif "sc" in q:
            updates["category"] = "SC"
        elif "st" in q:
            updates["category"] = "ST"
        elif "ews" in q:
            updates["category"] = "EWS"
        elif "pwd" in q or "pwbd" in q:
            updates["category"] = "PwD"
        else:
            updates["category"] = "General"

    if any(x in q for x in ["all india", "aiq", "mcc"]):
        updates["preference"] = "all_india"
    elif "home state" in q or "my state" in q:
        updates["preference"] = "home_state_only"
    elif "government" in q:
        updates["preference"] = "government_only"
    elif "all types" in q or "private" in q or "deemed" in q:
        updates["preference"] = "all_types"

    if any(x in q for x in ["mbbs abroad", "abroad", "outside india", "yes show abroad"]):
        updates["abroad_interest"] = "yes"
    elif any(x in q for x in ["no india only", "india only", "no abroad"]):
        updates["abroad_interest"] = "no"

    # Simple state detection using CUTOFF_DB_STATES list
    q_lower = question.lower()
    if any(s.lower() in q_lower for s in CUTOFF_DB_STATES if s != "MCC"):
        updates["state_scope"] = "provided"
    return updates


def _next_onboarding_prompt(med_ctx: Dict[str, object], question: str) -> Optional[Dict[str, object]]:
    """
    Legacy onboarding prompt hook kept as a no-op.
    Guided onboarding is handled by the cutoff profile collection flow in V2.
    """
    return None


def _extract_first_name(full_name: Optional[str]) -> Optional[str]:
    if not full_name:
        return None
    cleaned = str(full_name).strip()
    if not cleaned:
        return None
    return cleaned.split()[0]


def _first_visit_welcome_message(first_name: Optional[str] = None, greeting_word: Optional[str] = None) -> str:
    gw = (greeting_word or "hi").strip()
    if gw not in {"hi", "hello", "hey", "good morning", "good afternoon", "good evening"}:
        gw = "hi"
    opening = gw.title()
    greeting = (
        f"{opening} {first_name}, hope you are doing well.\n\n"
        if first_name else
        f"{opening}, hope you are doing well.\n\n"
    )
    return (
        f"{greeting}"
        "I am **Med Buddy**, India’s first AI counselling companion built exclusively for NEET UG aspirants, "
        "proudly powered by **Get My University**.\n\n"
        "Whether you are trying to figure out which medical/dental colleges match your score, understand "
        "last year’s cutoffs, decode fee structures, plan counselling process, or check NEET exam details, "
        "you have support at every step.\n\n"
        "Tell me what you want to explore first."
    )


def _return_visit_welcome_message(first_name: Optional[str] = None, greeting_word: Optional[str] = None) -> str:
    gw = (greeting_word or "hi").strip()
    if gw not in {"hi", "hello", "hey", "good morning", "good afternoon", "good evening"}:
        gw = "hi"
    opening = gw.title()
    greeting = (
        f"{opening} {first_name}, hope you are doing well.\n\n"
        if first_name else
        f"{opening}, hope you are doing well.\n\n"
    )
    return (
        f"{greeting}"
        "Welcome back! Great to see you again.\n\n"
        "Ready to continue your NEET UG counselling journey? "
        "Pick a direction below or ask your next question."
    )


def _build_session_close_message(med_ctx: Dict[str, object]) -> str:
    topic = med_ctx.get("last_topic") or "your NEET counselling queries"
    state = med_ctx.get("last_state")
    state_line = f"\n- Focus state/scope: **{state}**" if state else ""
    return (
        "This was a solid session.\n\n"
        "Here is a quick recap:\n"
        f"- Last focus area: **{topic}**"
        f"{state_line}\n\n"
        "Your progress is saved. Come back anytime and we can continue from here.\n\n"
        "_Med Buddy, powered by Get My University_"
    )


def _infer_topic_label(question: str) -> str:
    q = _normalize_text(question)
    if any(k in q for k in ["fee", "fees", "tuition", "cost"]):
        return "Fee structure"
    if any(k in q for k in ["cutoff", "rank", "score"]):
        return "Cutoff analysis"
    if any(k in q for k in ["counselling", "counseling", "round", "mcc"]):
        return "Counselling process"
    if any(k in q for k in ["college", "shortlist", "which college"]):
        return "College shortlist"
    return "NEET counselling guidance"


def _apply_response_policy(
    answer: str,
    question: str,
    skip_compare_cta: bool = False,
    allow_factual_addons: bool = True,
) -> str:
    text = (answer or "").strip()
    if not text:
        return text
    if not allow_factual_addons:
        return text
    q = _normalize_text(question)
    factual = any(k in q for k in ["fee", "cutoff", "rank", "date", "reservation", "seat"])
    tlow = text.lower()
    has_disclaimer = (
        "always verify" in tlow
        or "official website" in tlow
        or "note — disclaimer" in tlow
        or "note - disclaimer" in tlow
        or ("note" in tlow and "disclaimer" in tlow)
    )
    if factual and not has_disclaimer:
        text += (
            "\n\n"
            "> *Note — Disclaimer: Information is based on available counselling documents. "
            "Always verify the latest updates on official MCC/state counselling websites before taking admission decisions.*"
        )
    if factual and not skip_compare_cta and "would you like" not in tlow:
        text += "\n\nWould you like me to also compare this with another state/college for you?"
    return text


def _kb_results_count(tool_result: str) -> int:
    m = re.search(r"Results Found:\s*(\d+)", str(tool_result or ""), flags=re.IGNORECASE)
    if not m:
        return 0
    try:
        return int(m.group(1))
    except Exception:
        return 0


def _should_skip_kb_sufficiency_llm(question: str, kb_tool_result: str) -> bool:
    """
    Latency fast-path for straightforward date/timeline questions where KB
    already returned concrete hits.
    """
    q = _normalize_text(question)
    is_date_like = any(
        k in q
        for k in [
            "exam date",
            "date",
            "deadline",
            "last date",
            "schedule",
            "timeline",
            "application date",
            "result date",
            "admit card",
        ]
    )
    if not is_date_like:
        return False
    return _kb_results_count(kb_tool_result) > 0


def _is_compact_factual_query(question: str) -> bool:
    q = _normalize_text(question)
    return any(
        k in q
        for k in [
            "exam date",
            "date",
            "deadline",
            "last date",
            "schedule",
            "timeline",
            "application date",
            "result date",
            "admit card",
            "timing",
            "duration",
        ]
    )


def _final_answer_temperature(question: str) -> float:
    # Keep factual/date outputs deterministic to reduce repeated drafts.
    return 0.1 if _is_compact_factual_query(question) else 0.3


def _final_answer_max_tokens(question: str) -> int:
    # Smaller budget for compact factual answers improves latency and focus.
    if _is_compact_factual_query(question):
        return min(V2_FINAL_MAX_TOKENS, 320)
    return V2_FINAL_MAX_TOKENS


def _should_run_kb_sufficiency_check(question: str) -> bool:
    q = _normalize_text(question)
    return any(
        k in q
        for k in [
            "fee",
            "fees",
            "tuition",
            "cutoff",
            "rank",
            "score",
            "seat",
            "reservation",
            "quota",
            "date",
            "deadline",
            "eligibility",
        ]
    )


def _looks_like_neet_factual_query(question: str) -> bool:
    q = _normalize_text(question)
    return any(
        k in q
        for k in [
            "neet",
            "syllabus",
            "exam pattern",
            "eligibility",
            "admit card",
            "result",
            "counselling",
            "college",
            "fee",
            "cutoff",
            "rank",
            "reservation",
            "seat matrix",
            "documents",
        ]
    )


_VAGUE_SCOPE_EXPANSION_PATTERNS: Tuple[str, ...] = (
    "what about other",
    "other colleges",
    "other college",
    "another colleges",
    "another college",
    "any other college",
    "some other college",
    "different colleges",
    "different college",
    "more colleges",
    "more college",
    "rest of colleges",
    "rest of college",
    "compare with another",
    "compare to another",
    "another state",
    "other state",
    "other states",
)


def _is_explicit_college_shortlist_trigger(question: str) -> bool:
    q = _normalize_text(question)
    normalized = re.sub(r"[^a-z0-9\s]", " ", q)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    phrases = {
        "college shortlist",
        "shortlist colleges",
        "shortlist college",
        "college shortlisting",
        "start college shortlist",
    }
    return normalized in phrases


def _extract_json_object(raw_text: str) -> Optional[Dict[str, object]]:
    raw = (raw_text or "").strip()
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        m = re.search(r"\{[\s\S]*\}", raw)
        if not m:
            return None
        try:
            parsed = json.loads(m.group(0))
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None



def _canonicalize_state_name(raw: Optional[str]) -> Optional[str]:
    """Map a raw state string to the exact DB value from CUTOFF_DB_STATES."""
    if not raw:
        return None
    s = str(raw).strip().lower()
    for db_state in CUTOFF_DB_STATES:
        if db_state.lower() == s:
            return db_state
    # Partial match fallback
    for db_state in CUTOFF_DB_STATES:
        if s in db_state.lower() or db_state.lower() in s:
            return db_state
    return str(raw).strip() or None


def _question_mentions_category_filters(question: str) -> bool:
    """Detect explicit category/sub-category refinement in current user turn."""
    q = _normalize_text(question or "")
    if not q:
        return False
    keywords = [
        "category",
        "sub category",
        "subcategory",
        "general",
        "obc",
        "sc",
        "st",
        "ews",
        "pwd",
        "pwbd",
        "ur",
    ]
    return any(k in q for k in keywords)



# ═══════════════════════════════════════════════════════════════════════════
# CUTOFF MODULE — CLEAN REDESIGN
# ═══════════════════════════════════════════════════════════════════════════
#
# ARCHITECTURE:
#   One LLM call (_cutoff_resolve_intent) does everything:
#     - Detects scope: central (MCC) or state
#     - Extracts rank/score, category, states, college_type, course
#     - Knows what fields are still missing
#     - Returns a clean structured JSON
#
#   One SQL builder (_cutoff_build_and_run_sql) takes that JSON and runs the query.
#   No hardcoded if-else chains. No multiple LLM calls per turn.
#
# SCOPES:
#   CENTRAL: state='MCC', no category filter, no domicile filter.
#            Triggered by: AIIMS, Deemed, JIPMER, AMU, BHU, Jamia Milia, MCC, AIQ
#            Optional: college_type filter (only when user specifically asks)
#
#   STATE:   state=<real state>, category filter, domicile filter.
#            home_state == target_state → DOMICILE or OPEN
#            home_state != target_state → NON DOMICILE or OPEN
#            category always applied for state scope
#
# ═══════════════════════════════════════════════════════════════════════════

# ── DB constants ────────────────────────────────────────────────────────────

CUTOFF_DB_STATES = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
    "Delhi", "Gujarat", "Haryana", "Himachal Pradesh", "Jammu & Kashmir",
    "Jharkhand", "Karnataka", "Kerala", "MCC", "Madhya Pradesh", "Maharashtra",
    "Manipur", "Nagaland", "Odisha", "Puducherry", "Punjab", "Rajasthan",
    "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand",
    "West Bengal",
]

CUTOFF_CENTRAL_COLLEGE_TYPES = {"AIIMS", "Deemed", "JIPMER", "AMU", "BHU", "Jamia Milia"}
CUTOFF_ALL_COLLEGE_TYPES = {"AIIMS", "Deemed", "JIPMER", "AMU", "BHU", "Jamia Milia", "Government", "Private"}
CUTOFF_COURSES = {"MBBS", "BDS", "B.Sc. Nursing"}
CUTOFF_CATEGORIES = {"GENERAL", "OBC", "SC", "ST", "EWS", "PWD"}


# ── Profile helpers ─────────────────────────────────────────────────────────

def _normalize_cutoff_category(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    raw = str(value).strip().upper().replace("-", "").replace(" ", "")
    mapping = {
        "GENERAL": "GENERAL", "GEN": "GENERAL", "UR": "GENERAL",
        "OBC": "OBC", "SC": "SC", "ST": "ST", "EWS": "EWS",
        "PWD": "PWD", "PWBD": "PWD",
    }
    return mapping.get(raw) or str(value).strip().upper()


async def _load_user_cutoff_profile(user_id: Optional[int]) -> Dict[str, object]:
    if not user_id:
        return {}
    try:
        from database.connection import async_session_maker
        async with async_session_maker() as db:
            user = await db.get(User, user_id)
            if not user:
                return {}
            profile_data = dict(getattr(user, "profile_data", {}) or {})
            cp = dict(profile_data.get("cutoff_profile") or {})
            home_state = _canonicalize_state_name(cp.get("home_state"))
            category = _normalize_cutoff_category(cp.get("category"))
            sub_cat = str(cp.get("sub_category") or "").strip().upper() or None
            return {
                "home_state": home_state,
                "category": category,
                "sub_category": sub_cat,
                "preferences_set": bool(home_state and category),
            }
    except Exception as e:
        log(f"[CUTOFF] ⚠️ Could not load user cutoff profile: {e}")
        return {}


async def _save_user_cutoff_profile(user_id: Optional[int], home_state: str, category: str, sub_category: Optional[str]) -> None:
    if not user_id or not home_state or not category:
        return
    try:
        from database.connection import async_session_maker
        async with async_session_maker() as db:
            user = await db.get(User, user_id)
            if not user:
                return
            profile_data = dict(getattr(user, "profile_data", {}) or {})
            cp = dict(profile_data.get("cutoff_profile") or {})
            cp["home_state"] = home_state
            cp["category"] = category
            if sub_category:
                cp["sub_category"] = sub_category
            cp["preferences_set"] = True
            cp["updated_at"] = datetime.utcnow().isoformat()
            profile_data["cutoff_profile"] = cp
            user.profile_data = profile_data
            await db.commit()
    except Exception as e:
        log(f"[CUTOFF] ⚠️ Could not save user cutoff profile: {e}")


async def _get_cutoff_category_options(state: Optional[str]) -> List[str]:
    if not state:
        return []
    try:
        from database.connection import async_session_maker
        from sqlalchemy import text
        async with async_session_maker() as db:
            result = await db.execute(
                text("""
                    SELECT DISTINCT TRIM(UPPER(category))
                    FROM neet_ug_2025_cutoffs
                    WHERE state ILIKE :state AND category IS NOT NULL AND TRIM(category) <> ''
                    ORDER BY 1
                """),
                {"state": state},
            )
            return [str(r[0]) for r in result.fetchall() if r[0]]
    except Exception as e:
        log(f"[CUTOFF] ⚠️ category options failed: {e}")
        return []


async def _get_cutoff_subcategory_options(state: Optional[str], category: Optional[str]) -> List[str]:
    if not state or not category:
        return []
    try:
        from database.connection import async_session_maker
        from sqlalchemy import text
        async with async_session_maker() as db:
            result = await db.execute(
                text("""
                    SELECT DISTINCT TRIM(UPPER(sub_category))
                    FROM neet_ug_2025_cutoffs
                    WHERE state ILIKE :state AND category ILIKE :cat
                      AND sub_category IS NOT NULL AND TRIM(sub_category) <> ''
                    ORDER BY 1
                """),
                {"state": state, "cat": f"%{category}%"},
            )
            return [str(r[0]) for r in result.fetchall() if r[0]]
    except Exception as e:
        log(f"[CUTOFF] ⚠️ sub_category options failed: {e}")
        return []


# ── Intent resolver (THE single LLM call per turn) ─────────────────────────

_CUTOFF_INTENT_SYSTEM_PROMPT = """You are a NEET UG counselling assistant that resolves college cutoff search intent.

Given the conversation history and the latest user message, return a JSON object that fully describes what SQL query to run.

## DATABASE FACTS (use exact values)
States in DB: Andhra Pradesh, Arunachal Pradesh, Assam, Bihar, Chhattisgarh, Delhi, Gujarat,
  Haryana, Himachal Pradesh, Jammu & Kashmir, Jharkhand, Karnataka, Kerala, MCC,
  Madhya Pradesh, Maharashtra, Manipur, Nagaland, Odisha, Puducherry, Punjab, Rajasthan,
  Tamil Nadu, Telangana, Tripura, Uttar Pradesh, Uttarakhand, West Bengal

College types: AIIMS, Deemed, JIPMER, AMU, BHU, Jamia Milia, Government, Private

Central college types (MCC only): AIIMS, Deemed, JIPMER, AMU, BHU, Jamia Milia

Courses: MBBS, BDS, B.Sc. Nursing

Categories: GENERAL, OBC, SC, ST, EWS, PWD

## TWO SCOPES — NEVER MIX

### CENTRAL scope (state = 'MCC')
Use when user asks about: MCC, AIQ, all india quota, central counselling,
  AIIMS, JIPMER, Deemed universities, AMU, BHU, Jamia Millia
Rules:
- target_states = ["MCC"]
- Do not assume filters by scope. Any filter is optional and should be applied only when user explicitly asks.
- Only need: rank or score
- Set college_type ONLY if user specifically named a type (AIIMS → "AIIMS", deemed → "Deemed")
- "MCC colleges" or "central" without specific type → college_type = null
- Carry forward college_type from previous turn if user hasn't changed it
- Optional refinement filters (set when user explicitly asks):
  - category: GENERAL | OBC | SC | ST | EWS | PWD
  - quota: user says "NRI quota", "management quota", "open quota" etc. → set quota_keywords
  - course: MBBS | BDS | B.Sc. Nursing
  - domicile, sub_category, seat_type are also optional if user explicitly requests

### STATE scope (state = real Indian state)
Use when user explicitly names a state or says "my state", "home state", "in [state name]".
Rules:
- target_states = [exactly what the user named — never assume]
- Do not auto-apply extra filters by default. Apply filter columns only when user explicitly asks or already active in context.
- Need: rank/score + home_state + category + target_states

## CRITICAL RULE — NEVER ASSUME target_states
target_states must ONLY be set when the user has EXPLICITLY named a state or said "my state"/"home state".
NEVER default target_states to the user's home_state from their profile.
If the user has not told you where to search → target_states = [] → ask them.

Example:
- User says "college shortlist" → scope=need_more_info, ask: "Are you looking for colleges in a specific state, or MCC/All India colleges?"
- User says "colleges in Bihar" → scope=state, target_states=["Bihar"]
- User says "AIIMS colleges" → scope=central, target_states=["MCC"]
- User says "my rank is 2300" (no location) → ask where they want to search

## CONTEXT RULES
- Carry forward rank/score from previous turns — never ask again
- Carry forward home_state/category from profile or previous turns — never ask again
- Carry forward target_states from previous turns — never ask again once set
- If user switches to central scope → set target_states=["MCC"], drop state scope fields
- If user switches to a new state → update target_states to new state

## WHAT TO RETURN

Return ONLY this JSON (no extra text):
{
  "scope": "central" | "state" | "need_more_info",
  "target_states": ["MCC"] or ["Bihar", "Uttar Pradesh"] or [],
  "college_type": null or "AIIMS" | "Deemed" | "JIPMER" | "AMU" | "BHU" | "Jamia Milia" | "Government" | "Private",
  "course": null or "MBBS" | "BDS" | "B.Sc. Nursing",
  "metric_type": "rank" | "score" | null,
  "metric_value": integer or null,
  "home_state": null or exact state name from DB list,
  "category": null or "GENERAL" | "OBC" | "SC" | "ST" | "EWS" | "PWD",
  "sub_category": null or string,
  "quota_keywords": null or ["NRI", "Management", "Open"] (short keywords for SQL ILIKE filter),
  "missing_fields": [],
  "follow_up_message": null or "natural conversational question"
}

## REFINEMENT RULES (follow-up filters — apply on top of existing results)
When user sends a follow-up that adds a filter, carry all previous fields forward and ADD the new filter:
- "show ST category" / "only for OBC" → set category=ST/OBC (works for both central and state scope)
- "only MBBS" / "BDS only" → set course=MBBS/BDS
- "NRI quota" / "management quota" → set quota_keywords=["NRI"] or ["Management"]
- "government colleges only" / "private only" → set college_type=Government/Private
- "only AIIMS" / "show AIIMS" → set college_type=AIIMS
- "sub-category XYZ only" / "remove sub-category" → set/clear sub_category
- "seat type management/open/state quota" → set seat_type_keywords
- "remove category filter" / "all categories" → set category=null

## MISSING FIELDS LOGIC

### CENTRAL scope:
- metric missing → missing_fields=["metric"], ask for rank/score
- category is OPTIONAL for central — only set when user explicitly asks, never required
- everything else ready → missing_fields=[], run query

### STATE scope — ask in this order (one group at a time):
1. If scope/location unknown (user hasn't said where to search):
   → scope=need_more_info, missing_fields=["scope"]
   → follow_up_message: "Are you looking for colleges in a specific state, or MCC/All India colleges like AIIMS, Deemed universities?"

2. If target_states is empty (user hasn't named a state yet):
   → missing_fields=["target_states"]
   → follow_up_message: "Which state(s) would you like me to search in?"
   → If user_profile has home_state, hint: "Would you like to search in [home_state], or a different state?"

3. If metric missing:
   → missing_fields=["metric"], ask for rank/score

4. All present → run query

## METRIC DISAMBIGUATION
- Numbers ≤ 720 → score; Numbers > 720 → rank
- "rank"/"AIR" → rank; "score"/"marks" → score

## PROFILE DATA
user_profile contains saved home_state and category — use directly, never ask again.
But NEVER use home_state as target_states unless user explicitly says "my state" or "home state".
"""


async def _cutoff_resolve_intent(
    question: str,
    conversation_history: List[Dict],
    cutoff_ctx: Dict[str, object],
    user_profile: Dict[str, object],
) -> Dict[str, object]:
    """
    Single LLM call that resolves everything needed to build the SQL query.
    Returns a clean dict with scope, filters, missing_fields, follow_up_message.
    """
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Build a compact conversation context (last 6 messages)
        history_lines = []
        for msg in conversation_history[-6:]:
            role = str(getattr(msg, "role", "")).lower()
            content = str(getattr(msg, "content", "") or "").strip()
            if not content:
                continue
            if "user" in role:
                history_lines.append(f"User: {content[:300]}")
            elif "assistant" in role:
                history_lines.append(f"Assistant: {content[:300]}")
        history_text = "\n".join(history_lines) if history_lines else "(none)"

        # Compact current cutoff context
        ctx_summary = {
            "scope": cutoff_ctx.get("scope"),
            "metric_type": cutoff_ctx.get("metric_type"),
            "metric_value": cutoff_ctx.get("metric_value"),
            "target_states": cutoff_ctx.get("target_states"),
            "college_type": cutoff_ctx.get("college_type_filter"),
            "course": cutoff_ctx.get("course_filter"),
            "category": cutoff_ctx.get("category"),
            "home_state": cutoff_ctx.get("home_state"),
        }

        user_message = f"""CONVERSATION HISTORY:
{history_text}

CURRENT CUTOFF CONTEXT (already known from this conversation):
{json.dumps(ctx_summary, ensure_ascii=False)}

USER PROFILE (saved preferences — use directly, never ask again):
home_state: {user_profile.get('home_state') or 'not set'}
category: {user_profile.get('category') or 'not set'}
sub_category: {user_profile.get('sub_category') or 'not set'}

LATEST USER MESSAGE:
{question}

Return JSON only."""

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _CUTOFF_INTENT_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0,
            max_tokens=400,
        )

        raw = (resp.choices[0].message.content or "").strip()
        parsed = json.loads(raw)

        if not isinstance(parsed, dict):
            return {"scope": "need_more_info", "missing_fields": ["metric"], "follow_up_message": "Please share your NEET rank or score."}

        # Validate and clean the response
        out: Dict[str, object] = {}

        # Scope
        scope = str(parsed.get("scope") or "need_more_info").lower()
        if scope not in {"central", "state", "need_more_info"}:
            scope = "need_more_info"
        out["scope"] = scope

        # Target states
        raw_states = parsed.get("target_states") or []
        valid_states = set(CUTOFF_DB_STATES)
        target_states = []
        for s in (raw_states if isinstance(raw_states, list) else []):
            s = str(s).strip()
            if s in valid_states:
                target_states.append(s)
        out["target_states"] = target_states

        # College type — only valid DB values
        ct = str(parsed.get("college_type") or "").strip()
        out["college_type_filter"] = ct if ct in CUTOFF_ALL_COLLEGE_TYPES else None

        # Course
        course = str(parsed.get("course") or "").strip()
        out["course_filter"] = course if course in CUTOFF_COURSES else None

        # Metric
        mt = str(parsed.get("metric_type") or "").lower()
        out["metric_type"] = mt if mt in {"rank", "score"} else None
        mv = parsed.get("metric_value")
        out["metric_value"] = int(mv) if isinstance(mv, (int, float)) and int(mv) > 0 else None

        # State scope fields
        hs = str(parsed.get("home_state") or "").strip()
        out["home_state"] = hs if hs in valid_states else None

        cat = str(parsed.get("category") or "").strip().upper()
        out["category"] = cat if cat in CUTOFF_CATEGORIES else None

        sub = str(parsed.get("sub_category") or "").strip().upper()
        out["sub_category"] = sub if sub else None

        # Quota keywords (refinement filter — fuzzy ILIKE)
        qk = parsed.get("quota_keywords")
        if isinstance(qk, list) and qk:
            out["quota_keywords"] = [str(k).strip() for k in qk if str(k).strip()][:5]
        else:
            out["quota_keywords"] = None

        # Missing fields and follow-up
        out["missing_fields"] = list(parsed.get("missing_fields") or [])
        # Guard against inconsistent LLM output:
        # if target_states is already resolved, never keep "target_states" as missing.
        if out["target_states"] and "target_states" in out["missing_fields"]:
            out["missing_fields"] = [m for m in out["missing_fields"] if m != "target_states"]
        out["follow_up_message"] = str(parsed.get("follow_up_message") or "").strip() or None

        log(
            f"[CUTOFF] 🧠 Intent resolved | scope={out['scope']} "
            f"states={out['target_states']} metric={out['metric_type']}={out['metric_value']} "
            f"college_type={out['college_type_filter']} course={out['course_filter']} "
            f"category={out['category']} home_state={out['home_state']} "
            f"missing={out['missing_fields']}"
        )
        return out

    except Exception as e:
        log(f"[CUTOFF] ⚠️ Intent resolver failed: {e}")
        return {
            "scope": "need_more_info",
            "missing_fields": ["metric"],
            "follow_up_message": "Please share your NEET rank or score so I can find matching colleges.",
        }


# ── SQL builder + runner ────────────────────────────────────────────────────

async def _cutoff_run_sql(
    intent: Dict[str, object],
    total_limit: int = 10,
) -> List[Dict]:
    """
    Pure Python SQL builder. Takes the resolved intent dict, builds query, runs it.
    No LLM involved here.
    """
    from database.connection import async_session_maker
    from sqlalchemy import text

    scope = str(intent.get("scope") or "")
    target_states = list(intent.get("target_states") or [])
    metric_type = str(intent.get("metric_type") or "")
    metric_value = intent.get("metric_value")
    college_type = str(intent.get("college_type_filter") or "").strip()
    course = str(intent.get("course_filter") or "").strip()
    category = str(intent.get("category") or "").strip().upper()
    home_state = str(intent.get("home_state") or "").strip()
    sub_category = str(intent.get("sub_category") or "").strip().upper()
    quota_keywords = list(intent.get("quota_keywords") or []) or None

    if not target_states or not metric_value or not metric_type:
        return []

    is_central = (scope == "central") or (len(target_states) == 1 and target_states[0].upper() == "MCC")

    # Distribute limit across states
    n = len(target_states)
    base = max(1, total_limit // n)
    rem = max(0, total_limit - base * n)
    per_state = {s: base + (1 if i < rem else 0) for i, s in enumerate(target_states)}

    all_rows: List[Dict] = []

    async with async_session_maker() as db:
        for state in target_states:
            params: Dict[str, object] = {
                "state_val": state,
                "limit_val": per_state.get(state, base),
            }

            # Metric condition
            if metric_type == "score":
                metric_cond = "AND score IS NOT NULL AND score <= :metric_val"
                order_col = "score DESC"
            else:
                metric_cond = "AND air_rank IS NOT NULL AND air_rank >= :metric_val"
                order_col = "air_rank ASC"
            params["metric_val"] = metric_value

            # Category filter (applied when present in SQL intent).
            cat_cond = ""
            if category:
                cat_cond = "AND TRIM(UPPER(COALESCE(category, ''))) = :category_val"
                params["category_val"] = category

            # Domicile filter — optional across scopes (applies when home_state is available)
            dom_cond = ""
            if home_state:
                same = home_state.strip().lower() == state.strip().lower()
                if same:
                    dom_cond = "AND REPLACE(TRIM(UPPER(COALESCE(domicile, ''))), '-', ' ') IN ('DOMICILE', 'OPEN')"
                else:
                    dom_cond = "AND REPLACE(TRIM(UPPER(COALESCE(domicile, ''))), '-', ' ') IN ('NON DOMICILE', 'OPEN')"

            # College type filter
            ct_cond = ""
            if college_type and college_type in CUTOFF_ALL_COLLEGE_TYPES:
                ct_cond = "AND TRIM(college_type) = :college_type_val"
                params["college_type_val"] = college_type

            # Course filter
            course_cond = ""
            if course and course in CUTOFF_COURSES:
                course_cond = "AND TRIM(UPPER(COALESCE(course, ''))) = :course_val"
                params["course_val"] = course.upper()

            # Sub-category filter (applied when present in SQL intent).
            sub_cond = ""
            if sub_category:
                sub_cond = "AND TRIM(UPPER(COALESCE(sub_category, ''))) = :sub_cat_val"
                params["sub_cat_val"] = sub_category

            # Quota filter — fuzzy ILIKE, works for both central and state scope
            quota_cond = ""
            if quota_keywords:
                quota_clauses = " OR ".join(
                    f"quota ILIKE :quota_kw_{i}"
                    for i in range(len(quota_keywords))
                )
                quota_cond = f"AND ({quota_clauses})"
                for i, kw in enumerate(quota_keywords):
                    params[f"quota_kw_{i}"] = f"%{kw}%"

            sql = f"""
                SELECT DISTINCT ON (COALESCE(institution_name, college_name))
                  state,
                  COALESCE(institution_name, college_name) AS institution_name,
                  college_name, college_type, course, category, sub_category,
                  seat_type, quota, domicile, eligibility, score, air_rank, round
                FROM neet_ug_2025_cutoffs
                WHERE state ILIKE :state_val
                  {metric_cond}
                  {cat_cond}
                  {dom_cond}
                  {ct_cond}
                  {course_cond}
                  {sub_cond}
                  {quota_cond}
                ORDER BY COALESCE(institution_name, college_name), {order_col}
                LIMIT :limit_val
            """

            log(f"[CUTOFF_SQL] state={state} scope={scope} is_central={is_central}")
            log(f"[CUTOFF_SQL] params={params}")

            result = await db.execute(text(sql), params)
            rows = [dict(r) for r in result.mappings().all()]
            log(f"[CUTOFF_SQL] rows fetched={len(rows)}")
            all_rows.extend(rows)

    # Sort by closest to metric value
    if metric_type == "score":
        all_rows.sort(key=lambda r: abs(metric_value - float(r.get("score") or 0)))
    else:
        all_rows.sort(key=lambda r: abs(int(r.get("air_rank") or 0) - metric_value))

    # Deduplicate by institution
    seen: set = set()
    deduped = []
    for row in all_rows:
        key = (row.get("state"), row.get("institution_name") or row.get("college_name"))
        if key not in seen:
            seen.add(key)
            deduped.append(row)
        if len(deduped) >= total_limit:
            break

    return deduped


# ── Markdown formatter ──────────────────────────────────────────────────────

def _cutoff_format_markdown(
    rows: List[Dict],
    intent: Dict[str, object],
    display_limit: int = 10,
) -> str:
    scope = str(intent.get("scope") or "")
    is_central = scope == "central"
    metric_type = str(intent.get("metric_type") or "rank")
    metric_value = intent.get("metric_value") or 0
    metric_label = "Score" if metric_type == "score" else "AIR Rank"
    target_states = list(intent.get("target_states") or [])
    states_label = ", ".join(target_states)
    category = str(intent.get("category") or "")
    sub_category = str(intent.get("sub_category") or "")
    home_state = str(intent.get("home_state") or "")
    college_type = str(intent.get("college_type_filter") or "")
    course = str(intent.get("course_filter") or "")
    quota_keywords = [str(k).strip() for k in (intent.get("quota_keywords") or []) if str(k).strip()]

    domicile_note = ""
    if target_states and home_state:
        domicile_note = "DOMICILE + OPEN" if home_state.lower() in [s.lower() for s in target_states] else "NON DOMICILE + OPEN"

    applied_filters = [f"- **{metric_label}:** {metric_value}"]
    applied_filters.append("- **Scope:** All India / MCC" if is_central else "- **Scope:** State")
    if states_label:
        applied_filters.append(f"- **Target state(s):** {states_label}")
    if home_state:
        applied_filters.append(f"- **Home state:** {home_state}")
    if domicile_note:
        applied_filters.append(f"- **Domicile rows shown:** {domicile_note}")
    if college_type:
        applied_filters.append(f"- **College type:** {college_type}")
    if course:
        applied_filters.append(f"- **Course:** {course}")
    if category:
        applied_filters.append(f"- **Category:** {category}")
    if sub_category:
        applied_filters.append(f"- **Sub-category:** {sub_category}")
    if quota_keywords:
        applied_filters.append(f"- **Quota keywords:** {', '.join(quota_keywords)}")

    if not rows:
        return (
            f"I searched the 2025 cutoff data but found no matches for your profile.\n\n"
            + "### Applied Filters\n\n"
            + "\n".join(applied_filters)
            + f"\n\nTry relaxing filters — for example, remove college type or try nearby states."
        )

    shown = rows[:display_limit]
    lines = [
        "### Best-Match Colleges (2025 Cutoff Data)",
        "",
        "### Applied Filters",
        "",
    ]
    lines.extend(applied_filters)

    lines += [
        f"- **Showing:** {len(shown)} of {len(rows)} matched options",
        "",
        "| # | Institution | State | Category | Quota | Domicile | AIR | Score | Round |",
        "|---|---|---|---|---|---|---:|---:|---|",
    ]

    for idx, row in enumerate(shown, 1):
        inst = (row.get("institution_name") or row.get("college_name") or "-").replace("|", " ")
        course_str = str(row.get("course") or "-").replace("|", " ")
        lines.append(
            f"| {idx} | {inst} ({course_str}) | {row.get('state') or '-'} | "
            f"{row.get('category') or '-'} | "
            f"{row.get('quota') or '-'} | {row.get('domicile') or '-'} | "
            f"{row.get('air_rank') if row.get('air_rank') is not None else '-'} | "
            f"{int(float(row.get('score'))) if row.get('score') is not None else '-'} | "
            f"{row.get('round') or '-'} |"
        )

    return "\n".join(lines)


# ── Quick interpretation (LLM summary of results) ──────────────────────────

async def _cutoff_quick_interpretation(
    rows: List[Dict],
    intent: Dict[str, object],
    user_question: str,
) -> str:
    if not rows:
        return ""
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        sample = [
            {
                "institution": r.get("institution_name") or r.get("college_name"),
                "state": r.get("state"),
                "college_type": r.get("college_type"),
                "course": r.get("course"),
                "category": r.get("category"),
                "quota": r.get("quota"),
                "domicile": r.get("domicile"),
                "air_rank": r.get("air_rank"),
                "score": r.get("score"),
                "round": r.get("round"),
            }
            for r in rows[:8]
        ]
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Summarize these NEET cutoff results in 4-6 bullet points.\n"
                        "Anchor the summary to the user's latest question intent (central/state, college type, category, quota).\n"
                        "Be practical: highlight best options, patterns, rank/score range, rounds, and directly answer what user asked.\n"
                        "End with ONE natural follow-up question.\n"
                        "Use only the data provided. No markdown tables. Bullets only."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "latest_user_question": user_question,
                            "profile": intent,
                            "results": sample,
                        }
                    ),
                },
            ],
            temperature=0.2,
            max_tokens=300,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        log(f"[CUTOFF] ⚠️ Quick interpretation failed: {e}")
        return ""


# ── Routing helper: is this a cutoff query? ─────────────────────────────────

def _is_cutoff_query(question: str, med_ctx: Dict[str, object]) -> bool:
    """Fast pre-check before calling the router LLM."""
    q = question.strip().lower()
    cutoff_keywords = [
        "college", "shortlist", "rank", "score", "cutoff", "cut off", "mbbs",
        "bds", "aiims", "jipmer", "deemed", "mcc", "aiq", "amu", "bhu",
        "admission", "seat", "which college", "can i get", "eligible",
        "nursing", "government medical", "private medical", "state quota",
    ]
    if any(k in q for k in cutoff_keywords):
        return True
    # Active cutoff conversation continuation handling:
    # keep cutoff route for short replies and natural language state-switch asks.
    cutoff_ctx = dict(med_ctx.get("cutoff") or {})
    if cutoff_ctx:
        if len(question.strip().split()) <= 8:
            return True
        continuation_phrases = [
            "what options",
            "which options",
            "can i get",
            "if i look in",
            "look in ",
            "in this state",
            "other state",
            "different state",
            "what can i get",
        ]
        if any(p in q for p in continuation_phrases):
            return True
    return False


def _should_skip_cutoff_router(question: str, med_ctx: Dict[str, object]) -> bool:
    """
    Returns True when we can skip the LLM router entirely and go straight to cutoff handler.
    Handles pure continuations: number reply, single category word, single state name, short refinement.
    """
    cutoff_ctx = dict(med_ctx.get("cutoff") or {})
    if not cutoff_ctx:
        return False
    q = question.strip()
    # Pure number (rank/score reply like "2300")
    if re.match(r"^\d{2,7}$", q):
        return True
    # Single category word
    if q.lower() in {"general", "obc", "sc", "st", "ews", "pwd", "pwbd"}:
        return True
    # Single state name from DB
    if q in CUTOFF_DB_STATES:
        return True
    # Very short message in active scoped conversation
    if cutoff_ctx.get("scope") and len(q.split()) <= 4:
        return True
    # Longer natural-language continuation in an active cutoff thread
    # e.g., "if i look in maharastra what options i can get?"
    ql = q.lower()
    continuation_phrases = [
        "what options",
        "can i get",
        "if i look in",
        "look in ",
        "in this state",
        "other state",
        "different state",
    ]
    if cutoff_ctx.get("metric_value") and any(p in ql for p in continuation_phrases):
        return True
    return False


async def _get_registered_user_name(user_id: Optional[int]) -> Optional[str]:
    """Fetch the logged-in student's name for response personalization."""
    if not user_id:
        return None
    try:
        from database.connection import async_session_maker
        async with async_session_maker() as db:
            user = await db.get(User, user_id)
            if not user:
                return None
            full_name = str(getattr(user, "full_name", "") or "").strip()
            return full_name or None
    except Exception as e:
        log(f"[V2] ⚠️ Could not load registered user name: {e}")
        return None


async def _get_registered_home_state(user_id: Optional[int]) -> Optional[str]:
    """Fetch the user's registered home state."""
    if not user_id:
        return None
    try:
        from database.connection import async_session_maker
        async with async_session_maker() as db:
            user = await db.get(User, user_id)
            if not user:
                return None
            profile_data = dict(getattr(user, "profile_data", {}) or {})
            state_raw = (
                profile_data.get("state_or_ut")
                or profile_data.get("state")
                or (dict(getattr(user, "preferences", {}) or {}).get("preferred_state"))
            )
            return _canonicalize_state_name(state_raw)
    except Exception as e:
        log(f"[V2] ⚠️ Could not load registered home state: {e}")
        return None



async def _should_route_to_cutoff(question: str, med_ctx: Dict[str, object]) -> bool:
    """LLM router: is this question about college cutoff shortlisting?"""
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        cutoff_ctx = dict(med_ctx.get("cutoff") or {})
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": "You are a router. Return JSON only: {\"route\": true/false, \"reason\": \"brief\"}",
                },
                {
                    "role": "user",
                    "content": (
                        "Is this message asking about college shortlisting, cutoffs, or which colleges "
                        "a student can get based on their NEET rank/score?\n\n"
                        "Route TRUE for: college shortlist, which college can I get, AIIMS/Deemed/MCC colleges, "
                        "colleges in a state, rank/score + college query, short follow-ups in active cutoff conversation "
                        "(supplying rank, state, category, college type).\n"
                        "Also route TRUE for state-switch continuation wording such as "
                        "'if I look in Maharashtra what options can I get', even if state spelling is imperfect.\n"
                        "Route FALSE for: fee structure, hostel, counselling process steps, documents, "
                        "eligibility rules, exam dates, 'how does X work' questions.\n\n"
                        f"Active cutoff conversation: {bool(cutoff_ctx.get('metric_value'))}\n"
                        f"Last topic: {med_ctx.get('last_topic') or 'none'}\n"
                        f"User message: \"{question}\"\n\n"
                        "Return JSON only."
                    ),
                },
            ],
            temperature=0,
            max_tokens=60,
        )
        data = json.loads(resp.choices[0].message.content or "{}")
        result = bool(data.get("route", False))
        log(f"[CUTOFF] 🧭 Route={result} | {data.get('reason', '')}")
        return result
    except Exception as e:
        log(f"[CUTOFF] ⚠️ Router failed: {e}")
        return False


# ── Profile collection UI ───────────────────────────────────────────────────

async def _cutoff_needs_first_time_profile(user_id: Optional[int]) -> bool:
    """Returns True if user has never set their cutoff profile (home_state + category)."""
    profile = await _load_user_cutoff_profile(user_id)
    return not bool(profile.get("preferences_set"))


def _parse_cutoff_profile_submission_text(question: str) -> Dict[str, Optional[str]]:
    """
    Parse structured hidden profile form text sent by frontend:
      Home state: X
      Category: Y
      Sub-category: Z
    """
    text = str(question or "")
    out: Dict[str, Optional[str]] = {"home_state": None, "category": None, "sub_category": None}
    if not text:
        return out

    m_state = re.search(r"^\s*Home state:\s*(.+)\s*$", text, flags=re.IGNORECASE | re.MULTILINE)
    m_cat = re.search(r"^\s*Category:\s*(.+)\s*$", text, flags=re.IGNORECASE | re.MULTILINE)
    m_sub = re.search(r"^\s*Sub-category:\s*(.+)\s*$", text, flags=re.IGNORECASE | re.MULTILINE)

    if m_state:
        raw = m_state.group(1).strip()
        if raw and raw.lower() not in {"not sure", "not_sure", "none"}:
            out["home_state"] = _canonicalize_state_name(raw)
    if m_cat:
        raw = m_cat.group(1).strip()
        if raw and raw.lower() not in {"not sure", "not_sure", "none"}:
            out["category"] = _normalize_cutoff_category(raw)
    if m_sub:
        raw = m_sub.group(1).strip()
        if raw and raw.lower() not in {"not sure", "not_sure", "none", "skip"}:
            out["sub_category"] = raw.upper()
    return out


# ── Main stage handler ──────────────────────────────────────────────────────

async def _v2_handle_cutoff_stage(
    *,
    request: "ChatRequest",
    conversation_id: Optional[int],
    med_ctx: Dict[str, object],
    conversation_memory,
    preferred_language: str,
    localize_output,
    start_time: "datetime",
    state: Dict[str, object],
) -> "AsyncGenerator[str, None]":
    """
    Clean cutoff handler. Called when routing decides this is a cutoff query.
    
    Flow:
    1. Load user profile from DB
    2. Call intent resolver (single LLM)
    3. If missing fields → ask user naturally
    4. If all fields present → run SQL → format → stream
    """
    from services.cutoff_service import format_cutoff_markdown
    state["handled"] = False
    t_start = time.perf_counter()

    # Load saved user profile
    user_profile = await _load_user_cutoff_profile(request.user_id)

    # Get existing cutoff context from conversation
    cutoff_ctx = dict(med_ctx.get("cutoff") or {})

    # Merge saved profile into context (only if not already set in context)
    if user_profile.get("home_state") and not cutoff_ctx.get("home_state"):
        cutoff_ctx["home_state"] = user_profile["home_state"]

    # Get conversation history for context
    chat_history = []
    if conversation_memory:
        chat_history = conversation_memory.get_chat_history() or []

    # ── STEP 1: Resolve intent ──
    t_intent = time.perf_counter()
    intent = await _cutoff_resolve_intent(
        question=request.question,
        conversation_history=chat_history,
        cutoff_ctx=cutoff_ctx,
        user_profile=user_profile,
    )
    intent_ms = _elapsed_ms(t_intent)

    scope = str(intent.get("scope") or "need_more_info")
    missing = list(intent.get("missing_fields") or [])
    follow_up = str(intent.get("follow_up_message") or "").strip()

    # Deterministic self-home-state override:
    # If user asks for "my home state" and profile already has it, do not ask again.
    q_norm = _normalize_text(request.question or "")
    profile_home_state = _canonicalize_state_name(user_profile.get("home_state"))
    asks_my_home_state = (
        "my home state" in q_norm
        or "home state" in q_norm
        or "my own state" in q_norm
        or "in my state" in q_norm
        or "my hometown" in q_norm
    )
    if asks_my_home_state and profile_home_state:
        intent["scope"] = "state"
        intent["target_states"] = [profile_home_state]
        intent["home_state"] = profile_home_state
        scope = "state"
        missing = [m for m in missing if m not in {"scope", "target_states"}]
        if not follow_up:
            follow_up = ""

    # Deterministic override for profile-form submission text (no LLM dependency).
    submitted_profile = _parse_cutoff_profile_submission_text(request.question)

    # ── STEP 2: Update cutoff context with resolved values ──
    category_explicit_this_turn = _question_mentions_category_filters(request.question)
    if intent.get("metric_type"):
        cutoff_ctx["metric_type"] = intent["metric_type"]
    if intent.get("metric_value"):
        cutoff_ctx["metric_value"] = int(intent["metric_value"])
    if intent.get("target_states"):
        cutoff_ctx["target_states"] = intent["target_states"]
    if intent.get("college_type_filter"):
        cutoff_ctx["college_type_filter"] = intent["college_type_filter"]
    else:
        # Only clear stale college_type when scope genuinely changed (not just a refinement turn)
        if intent.get("scope") in {"central", "state"}:
            cutoff_ctx.pop("college_type_filter", None)
    if intent.get("course_filter"):
        cutoff_ctx["course_filter"] = intent["course_filter"]
    if intent.get("home_state"):
        cutoff_ctx["home_state"] = intent["home_state"]
    # Category: set when explicitly provided; set to None if intent says null (user cleared it)
    # "category" key presence in intent means LLM made a decision about it
    if "category" in intent:
        if intent["category"]:
            # Never auto-apply profile-derived category; require explicit category intent in this turn,
            # or keep/refine an already active conversation filter.
            if category_explicit_this_turn or cutoff_ctx.get("category"):
                cutoff_ctx["category"] = intent["category"]
        else:
            # LLM returned null — only clear if it was a deliberate "remove filter" action
            # Keep existing category if this is central scope (LLM returns null by default for central)
            if scope != "central" and category_explicit_this_turn:
                cutoff_ctx.pop("category", None)
    if intent.get("sub_category"):
        if category_explicit_this_turn or cutoff_ctx.get("sub_category"):
            cutoff_ctx["sub_category"] = intent["sub_category"]
    if submitted_profile.get("home_state"):
        cutoff_ctx["home_state"] = submitted_profile["home_state"]
        if "home_state" in missing:
            missing = [m for m in missing if m != "home_state"]
    if submitted_profile.get("category"):
        cutoff_ctx["category"] = submitted_profile["category"]
        if "category" in missing:
            missing = [m for m in missing if m != "category"]
    # Explicitly clear sub_category when not provided in submitted form.
    if "Sub-category:" in str(request.question):
        if submitted_profile.get("sub_category"):
            cutoff_ctx["sub_category"] = submitted_profile["sub_category"]
        else:
            cutoff_ctx.pop("sub_category", None)
    # Quota keywords: update when provided by LLM, preserve existing when not mentioned
    if intent.get("quota_keywords") is not None:
        if intent["quota_keywords"]:
            cutoff_ctx["quota_keywords"] = intent["quota_keywords"]
        else:
            cutoff_ctx.pop("quota_keywords", None)
    cutoff_ctx["scope"] = scope
    # Guard against inconsistent missing_fields from LLM:
    # when state scope already has resolved target_states, don't ask for state again.
    if scope == "state" and list(cutoff_ctx.get("target_states") or []):
        if "target_states" in missing:
            missing = [m for m in missing if m != "target_states"]
            if follow_up and "state" in follow_up.lower():
                follow_up = ""
    cutoff_ctx["last_turn_at"] = datetime.utcnow().isoformat()

    # Persist profile immediately once available so first-time form does not reappear
    # on the next turn due to async race.
    if request.user_id and cutoff_ctx.get("home_state") and cutoff_ctx.get("category"):
        await _save_user_cutoff_profile(
            request.user_id,
            str(cutoff_ctx.get("home_state") or ""),
            str(cutoff_ctx.get("category") or ""),
            str(cutoff_ctx.get("sub_category") or "") or None,
        )

    med_ctx["cutoff"] = cutoff_ctx

    # ── STEP 2.5: First-time profile form (restored) ──
    needs_profile_form = bool(
        request.user_id
        and await _cutoff_needs_first_time_profile(request.user_id)
        and (not cutoff_ctx.get("home_state") or not cutoff_ctx.get("category"))
    )
    if needs_profile_form:
        selected_state = str(cutoff_ctx.get("home_state") or "").strip()
        selected_category = str(cutoff_ctx.get("category") or "").strip()
        selected_sub_category = str(cutoff_ctx.get("sub_category") or "").strip()

        category_options = await _get_cutoff_category_options(selected_state or None)
        sub_category_options: List[str] = []
        if selected_state and selected_category:
            sub_category_options = await _get_cutoff_subcategory_options(
                selected_state,
                selected_category,
            )

        form_message = localize_output(
            "Great - I can shortlist colleges accurately once you confirm your profile details below."
        )

        yield (
            "data: "
            + json.dumps(
                {
                    "type": "cutoff_profile_form",
                    "message": form_message,
                    "state": selected_state,
                    "category": selected_category,
                    "sub_category": selected_sub_category,
                    "states": [s for s in CUTOFF_DB_STATES if str(s).upper() != "MCC"],
                    "categories": category_options,
                    "sub_categories": sub_category_options,
                }
            )
            + "\n\n"
        )
        for token in sse_tokens_preserving_formatting(form_message):
            yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"
            if V2_STREAM_TOKEN_DELAY_SEC > 0:
                await asyncio.sleep(V2_STREAM_TOKEN_DELAY_SEC)
        yield f"data: {json.dumps({'type': 'done', 'conversation_id': conversation_id})}\n\n"
        if request.user_id and conversation_id:
            asyncio.create_task(v2_background_save_conversation_turn(
                conversation_id,
                request.question,
                form_message,
                int((datetime.now() - start_time).total_seconds() * 1000),
            ))
            asyncio.create_task(v2_background_update_conversation_context(conversation_id, med_ctx))
        log(f"[CUTOFF] ⏱ total={_elapsed_ms(t_start):.0f}ms intent={intent_ms:.0f}ms path=profile_form")
        state["handled"] = True
        return

    # ── STEP 3: If missing fields → ask naturally ──
    # Hard guard: never run SQL if target_states is empty for state scope
    target_states_resolved = list(intent.get("target_states") or [])
    if scope == "state" and not target_states_resolved and "target_states" not in missing:
        missing = ["target_states"]
        home_hint = (
            f" Would you like to search in **{user_profile.get('home_state')}**, or a different state?"
            if user_profile.get("home_state") else ""
        )
        follow_up = f"Which state(s) would you like me to search in?{home_hint}"

    if missing or scope == "need_more_info":
        if not follow_up:
            follow_up = "Could you share a bit more? I need your NEET rank or score to find matching colleges."
        follow_up = localize_output(follow_up)
        log(f"[CUTOFF] 📝 Missing fields: {missing} → asking: {follow_up[:80]}")
        for token in sse_tokens_preserving_formatting(follow_up):
            yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"
            if V2_STREAM_TOKEN_DELAY_SEC > 0:
                await asyncio.sleep(V2_STREAM_TOKEN_DELAY_SEC)
        yield f"data: {json.dumps({'type': 'done', 'conversation_id': conversation_id})}\n\n"
        if request.user_id and conversation_id:
            asyncio.create_task(v2_background_save_conversation_turn(
                conversation_id, request.question, follow_up,
                int((datetime.now() - start_time).total_seconds() * 1000),
            ))
            asyncio.create_task(v2_background_update_conversation_context(conversation_id, med_ctx))
        log(f"[CUTOFF] ⏱ total={_elapsed_ms(t_start):.0f}ms intent={intent_ms:.0f}ms path=missing_fields")
        state["handled"] = True
        return

    # ── STEP 4: Run SQL ──
    # Build final SQL intent by merging resolved intent with persisted cutoff_ctx.
    # This ensures refinement filters (category, quota_keywords) from previous turns
    # are preserved even when not mentioned in the current turn.
    sql_scope = str(cutoff_ctx.get("scope") or scope)
    is_central_sql_scope = sql_scope == "central" or (
        len(list(cutoff_ctx.get("target_states") or [])) == 1
        and str(list(cutoff_ctx.get("target_states") or [""])[0]).upper() == "MCC"
    )

    # Central-scope guard:
    # do not auto-carry profile category/sub-category/home-state unless explicitly requested this turn.
    sql_category = cutoff_ctx.get("category")
    sql_sub_category = cutoff_ctx.get("sub_category")
    sql_home_state = cutoff_ctx.get("home_state")
    if is_central_sql_scope:
        if not intent.get("category"):
            sql_category = None
        if not intent.get("sub_category"):
            sql_sub_category = None
        # Domicile/home-state filters should never be auto-applied for generic MCC central queries.
        sql_home_state = None
    else:
        # Non-home-state guard (strict):
        # apply category/sub-category only for home-state queries.
        # For other states, never apply these filters.
        target_states = list(cutoff_ctx.get("target_states") or [])
        target_state = target_states[0] if target_states else None
        is_home_state_query = (
            bool(sql_home_state)
            and bool(target_state)
            and str(sql_home_state).strip().lower() == str(target_state).strip().lower()
        )
        if not is_home_state_query:
            sql_category = None
            sql_sub_category = None

    sql_intent = {
        "scope": sql_scope,
        "target_states": cutoff_ctx.get("target_states") or [],
        "metric_type": cutoff_ctx.get("metric_type"),
        "metric_value": cutoff_ctx.get("metric_value"),
        "college_type_filter": cutoff_ctx.get("college_type_filter"),
        "course_filter": cutoff_ctx.get("course_filter"),
        "category": sql_category,
        "home_state": sql_home_state,
        "sub_category": sql_sub_category,
        "quota_keywords": cutoff_ctx.get("quota_keywords"),
    }
    cutoff_result_limit = await get_cutoff_result_limit()
    t_sql = time.perf_counter()
    rows = await _cutoff_run_sql(sql_intent, total_limit=cutoff_result_limit)
    sql_ms = _elapsed_ms(t_sql)
    cutoff_ctx["last_result_count"] = len(rows)
    med_ctx["cutoff"] = cutoff_ctx

    log(f"[CUTOFF] ✅ SQL done | rows={len(rows)} sql={sql_ms:.0f}ms")

    # Source tag
    source = {
        "file_name": "neet_ug_2025_cutoffs",
        "document_type": "sql_cutoff_table",
        "state": ", ".join(intent.get("target_states") or []),
        "text_snippet": f"Matched rows: {len(rows)}",
    }
    yield f"data: {json.dumps({'type': 'sources', 'sources': [source]})}\n\n"

    # ── STEP 5: Format table ──
    table_md = localize_output(_cutoff_format_markdown(rows, sql_intent, display_limit=cutoff_result_limit))

    # Stream table immediately
    for token in sse_tokens_preserving_formatting(table_md):
        yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"
        if V2_STREAM_TOKEN_DELAY_SEC > 0:
            await asyncio.sleep(V2_STREAM_TOKEN_DELAY_SEC)

    # ── STEP 6: Quick interpretation (concurrent) ──
    yield f"data: {json.dumps({'type': 'meta', 'cutoff_interpretation_loading': True})}\n\n"
    t_explain = time.perf_counter()
    explain_task = asyncio.create_task(_cutoff_quick_interpretation(rows, intent, request.question))

    # ── STEP 7: Suggested chips (concurrent) ──
    chips_task = asyncio.create_task(
        _generate_contextual_suggested_replies(
            request.question,
            table_md,
            med_ctx,
            retrieval_evidence=table_md,
            output_language=preferred_language,
        )
    )

    explanation = await explain_task
    explain_ms = _elapsed_ms(t_explain)

    disclaimer = (
        "\n\n> *Note — Disclaimer: Cutoffs vary year to year and by round/quota/sub-category. "
        "Always verify on official MCC/state counselling portals.*"
    )

    full_answer = table_md
    if explanation:
        tail = "\n\n### Quick Interpretation\n\n" + localize_output(explanation) + disclaimer
        full_answer = table_md + tail
        yield f"data: {json.dumps({'type': 'meta', 'cutoff_interpretation_loading': False})}\n\n"
        for token in sse_tokens_preserving_formatting(tail):
            yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"
            if V2_STREAM_TOKEN_DELAY_SEC > 0:
                await asyncio.sleep(V2_STREAM_TOKEN_DELAY_SEC)
    else:
        tail = disclaimer
        full_answer = table_md + tail
        yield f"data: {json.dumps({'type': 'meta', 'cutoff_interpretation_loading': False})}\n\n"
        for token in sse_tokens_preserving_formatting(tail):
            yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"

    # Chips
    try:
        chips = await chips_task
        if chips:
            yield f"data: {json.dumps({'type': 'suggested_replies', 'replies': chips})}\n\n"
    except Exception:
        pass

    # Update context
    med_ctx["stage"] = "normal_qa"
    med_ctx["last_topic"] = "Cutoff analysis"
    med_ctx["last_state"] = ", ".join(intent.get("target_states") or [])
    med_ctx["last_activity_at"] = datetime.utcnow().isoformat()
    med_ctx["cutoff"] = cutoff_ctx

    yield f"data: {json.dumps({'type': 'done', 'conversation_id': conversation_id})}\n\n"

    total_ms = _elapsed_ms(t_start)
    log(
        f"[CUTOFF] ⏱ total={total_ms:.0f}ms | intent={intent_ms:.0f}ms "
        f"sql={sql_ms:.0f}ms explain={explain_ms:.0f}ms path=final_answer"
    )

    if request.user_id and conversation_id:
        asyncio.create_task(v2_background_save_conversation_turn(
            conversation_id, request.question, full_answer,
            int((datetime.now() - start_time).total_seconds() * 1000),
            sources=[source],
        ))
        asyncio.create_task(v2_background_update_conversation_context(conversation_id, med_ctx))
        asyncio.create_task(v2_background_generate_conversation_title(
            conversation_id, request.question, log_label="cutoff"
        ))

    state["handled"] = True


from llama_index.core import (
    VectorStoreIndex,
    Settings,
    StorageContext,
    Document
)
from llama_index.core.prompts import PromptTemplate
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter, FilterOperator
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from pypdf import PdfReader
from openai import OpenAI  # For domain classification

# Database imports
from database.connection import engine, Base, init_db, close_db
from models import User, Conversation, Message, PendingQA, ActivityLog

# Smart routing services
from services.query_router import (
    route_query,
    build_vector_filters,
    QueryIntent,
    format_mixed_response_prompt,
    expand_query,
)
from services.chunk_classifier import classify_chunk
from services.vector_store_factory import get_vector_store, count_vectors_sync
from services.metadata_filter_utils import vector_filter_to_metadata_filters
from services.pdf_extraction import extract_text_from_pdf
from services.document_chunking import (
    prepare_pages_for_indexing,
    format_page_label,
    get_chunk_settings_for_document,
)

# Load environment variables from script directory
ENV_PATH = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

# Configuration
DATA_DIR = Path(__file__).parent / "data"
CUTOFF_SQL_STATES: List[str] = [
    "Andhra Pradesh",
    "Arunachal Pradesh",
    "Assam",
    "Bihar",
    "Chhattisgarh",
    "Delhi",
    "Gujarat",
    "Haryana",
    "Himachal Pradesh",
    "Jammu & Kashmir",
    "Jharkhand",
    "Karnataka",
    "Kerala",
    "MCC",
    "Madhya Pradesh",
    "Maharashtra",
    "Manipur",
    "Nagaland",
    "Odisha",
    "Puducherry",
    "Punjab",
    "Rajasthan",
    "Tamilnadu",
    "Telangana",
    "Tripura",
    "Uttar Pradesh",
    "Uttarakhand",
    "West Bengal",
]


def _rag_text_from_node(node) -> tuple:
    """Return (text, metadata dict) for a retrieved LlamaIndex node (PGVector / legacy shape)."""
    import json as _json

    md = dict(node.metadata) if getattr(node, "metadata", None) else {}
    text = ""
    nc = md.get("_node_content", "")
    if nc:
        try:
            parsed = _json.loads(nc)
            text = parsed.get("text", "") or ""
        except Exception:
            pass
    if not text:
        text = md.get("text", "") or ""
    if not text and hasattr(node, "get_content"):
        text = node.get_content() or ""
    return text, md


def _interleave_chunks_by_filter(
    per_filter: List[List[Tuple[str, Dict]]],
    max_chunks: int = 12,
) -> Tuple[List[str], List[Dict]]:
    """
    Round-robin merge chunks from each PGVector query (brochure vs college_info vs cutoffs, etc.).
    Without this, the prompt's first N chunks are only from document_type filter #1 and starve
    college/fee PDFs even when separate searches were run.
    """
    texts_out: List[str] = []
    sources_out: List[Dict] = []
    round_idx = 0
    while len(texts_out) < max_chunks:
        added_round = False
        for flist in per_filter:
            if round_idx < len(flist):
                text, src = flist[round_idx]
                texts_out.append(text)
                sources_out.append(src)
                added_round = True
                if len(texts_out) >= max_chunks:
                    return texts_out, sources_out
        if not added_round:
            break
        round_idx += 1
    return texts_out, sources_out


async def _get_cutoff_db_states_cached() -> List[str]:
    """
    Return fixed cutoff states list provided from DB distinct values.
    No runtime DB call is made for state loading.
    """
    return CUTOFF_SQL_STATES


# ============== LIFESPAN (Startup/Shutdown) ==============
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.
    - On startup: Create database tables if they don't exist
    - On shutdown: Close database connections
    """
    print("🚀 Starting application...")
    
    # Create all tables if they don't exist (auto-migration for dev)
    try:
        await init_db()
        print("✅ Database tables created/verified")
    except Exception as e:
        print(f"⚠️ Database init skipped (may not be configured): {e}")
    
    async def warm_all_services():
        """Warm all services in the background to reduce first-query latency."""
        # Warm the legacy vector store (for old endpoints)
        try:
            count_vectors_sync()
            print("✅ pgvector store warmed")
        except Exception as e:
            print(f"⚠️ Vector store warm failed: {e}")
        
        # Warm the knowledge tool (for V2 endpoint)
        try:
            from services.knowledge_tool import warm_knowledge_tool
            warm_knowledge_tool()
            print("✅ Knowledge tool warmed")
        except Exception as e:
            print(f"⚠️ Knowledge tool warm failed: {e}")
        
        # Warm OpenAI connection (reduces first-query latency)
        try:
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            # A minimal embedding call to establish connection
            client.embeddings.create(input="warm", model="text-embedding-3-small")
            print("✅ OpenAI connection warmed")
        except Exception as e:
            print(f"⚠️ OpenAI warm failed: {e}")

        # Warm and cache cutoff DB states once.
        try:
            states = await _get_cutoff_db_states_cached()
            print(f"✅ Cutoff states cached ({len(states)})")
        except Exception as e:
            print(f"⚠️ Cutoff states warm failed: {e}")

    asyncio.create_task(warm_all_services())
    print("📡 Warming services in background...")
    
    yield  # App runs here
    
    # Shutdown: close connections
    print("🛑 Shutting down...")
    try:
        await close_db()
        print("✅ Database connections closed")
    except Exception as e:
        print(f"⚠️ Database close error: {e}")


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="NEET Assistant RAG API",
    description="RAG Chatbot with Admin Document Management",
    version="2.0.0",
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware removed - BaseHTTPMiddleware causes hanging issues with streaming

# Include auth routes
from routes.auth import router as auth_router
app.include_router(auth_router)

# Include admin routes
from routes.admin import router as admin_router
app.include_router(admin_router)

# Include FAQ routes
from routes.faq import router as faq_router, search_faq
app.include_router(faq_router)

# Include conversation routes
from routes.conversations import router as conversations_router
app.include_router(conversations_router)

# Include support query routes
from routes.support import router as support_router
app.include_router(support_router)

# ============== MODELS ==============

class UserPreferences(BaseModel):
    """User preferences for smart query routing"""
    preferred_state: Optional[str] = None
    category: Optional[str] = None

class ChatRequest(BaseModel):
    question: str
    model: str = "openai"
    state_filter: Optional[str] = None  # Optional manual state filter
    conversation_id: Optional[int] = None  # For continuing a conversation
    user_id: Optional[int] = None  # For authenticated users' history
    user_preferences: Optional[UserPreferences] = None  # User preferences for smart routing
    clarified_scope: Optional[str] = None  # User's clarification: "central", "preference", or state name
    preferred_language: Optional[str] = None  # optional override: en | hi | mr

class Source(BaseModel):
    file_name: str
    page: Optional[str] = None
    text_snippet: str
    state: Optional[str] = None
    document_type: Optional[str] = None
    doc_topic: Optional[str] = None
    chunk_category: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    sources: Optional[List[Source]] = None
    model_used: str
    filters_applied: Optional[Dict] = None

class DocumentMetadata(BaseModel):
    state: str
    document_type: str
    category: str
    year: str = "2026"
    description: Optional[str] = None

class IndexStats(BaseModel):
    total_vectors: int
    namespaces: Dict
    index_name: str

# ============== GLOBALS ==============

index: Optional[VectorStoreIndex] = None
vector_store = None  # legacy name: PGVectorStore instance

# State mapping for query routing
STATES = {
    "karnataka": "Karnataka",
    "tamil nadu": "Tamil Nadu",
    "tamilnadu": "Tamil Nadu",
    "maharashtra": "Maharashtra",
    "andhra pradesh": "Andhra Pradesh",
    "andhra": "Andhra Pradesh",
    "telangana": "Telangana",
    "kerala": "Kerala",
    "gujarat": "Gujarat",
    "rajasthan": "Rajasthan",
    "uttar pradesh": "Uttar Pradesh",
    "up": "Uttar Pradesh",
    "madhya pradesh": "Madhya Pradesh",
    "mp": "Madhya Pradesh",
    "west bengal": "West Bengal",
    "bengal": "West Bengal",
    "bihar": "Bihar",
    "odisha": "Odisha",
    "punjab": "Punjab",
    "haryana": "Haryana",
    "delhi": "Delhi",
    "jammu and kashmir": "Jammu & Kashmir",
    "jammu & kashmir": "Jammu & Kashmir",
    "j&k": "Jammu & Kashmir",
    "jk": "Jammu & Kashmir",
    "assam": "Assam",
    "jharkhand": "Jharkhand",
    "chhattisgarh": "Chhattisgarh",
    "uttarakhand": "Uttarakhand",
    "himachal": "Himachal Pradesh",
    "goa": "Goa",
    "chandigarh": "Chandigarh",
    "ladakh": "Ladakh",
    "lakshadweep": "Lakshadweep",
    "puducherry": "Puducherry",
    "pondicherry": "Puducherry",
    "andaman and nicobar": "Andaman and Nicobar Islands",
    "dadra and nagar haveli and daman and diu": "Dadra and Nagar Haveli and Daman and Diu",
    "daman and diu": "Dadra and Nagar Haveli and Daman and Diu",
    "arunachal pradesh": "Arunachal Pradesh",
    "arunachal": "Arunachal Pradesh",
    "manipur": "Manipur",
    "meghalaya": "Meghalaya",
    "mizoram": "Mizoram",
    "nagaland": "Nagaland",
    "sikkim": "Sikkim",
    "tripura": "Tripura",
    "aiq": "All-India",
    "all india": "All-India",
    "national": "All-India",
    "nta": "All-India",
}

CATEGORIES = {
    "eligibility": ["eligible", "eligibility", "qualify", "qualification", "criteria", "requirement", "age limit", "who can"],
    "dates": ["date", "deadline", "when", "schedule", "last date", "exam date", "registration date"],
    "fees": ["fee", "fees", "cost", "payment", "amount", "price", "charges"],
    "colleges": ["college", "colleges", "seat", "seats", "admission", "institute", "university", "medical college"],
    "cutoff": ["cutoff", "cut off", "cut-off", "rank", "score", "marks", "percentile"],
    "process": ["process", "procedure", "how to", "steps", "apply", "registration", "counselling", "counseling"],
    "documents": ["document", "documents", "certificate", "certificates", "required documents", "papers"],
}

DOCUMENT_TYPES = {
    "nta_bulletin": ["nta", "bulletin", "information bulletin", "neet ug"],
    "mcc_counseling": ["mcc", "all india counselling", "all india counseling", "aiq", "15% quota"],
    "state_counseling": ["counselling", "counseling", "state counseling", "state counselling"],
    "college_info": ["college", "institute", "university", "medical college"],
    "cutoffs": ["cutoff", "cut off", "previous year", "rank"],
}


# ============== HELPER FUNCTIONS ==============

# Domain-SPECIFIC keywords (must have at least one of these)
# Quick rejection keywords - obviously off-topic (saves LLM cost)
OBVIOUSLY_OFF_TOPIC = [
    "movie", "song", "music", "game", "cricket", "football", "weather",
    "stock", "bitcoin", "crypto", "recipe", "cooking", "amazon", "flipkart",
    "politics", "election", "celebrity", "dating", "marriage", "relationship",
    "visa", "passport", "flight", "hotel", "restaurant"
]

OUT_OF_DOMAIN_RESPONSE = """Hi! Sorry, I can't help with that topic.

I am here to help you with **NEET UG counselling and medical admissions in India**:
- Eligibility and important dates
- Counselling process (MCC/AIQ + state)
- College options, fees, cutoffs, and documents
- Reservation and quota-related guidance

Share your NEET-related question, and I will help right away."""


async def is_web_search_fallback_enabled() -> bool:
    """Read runtime setting for web-search fallback."""
    try:
        from database.connection import async_session_maker
        from models.system_settings import SystemSettings, SettingsKeys

        async with async_session_maker() as db:
            setting = await db.get(SystemSettings, SettingsKeys.WEB_SEARCH_FALLBACK_ENABLED)
            if setting is None:
                return False  # Safe default
            return setting.value.lower() == "true"
    except Exception as err:
        log(f"[V2] ⚠️ Could not read web fallback setting: {err}")
        return False


async def is_chat_references_enabled() -> bool:
    """Read runtime setting for chat reference visibility."""
    try:
        from database.connection import async_session_maker
        from models.system_settings import SystemSettings, SettingsKeys

        async with async_session_maker() as db:
            setting = await db.get(SystemSettings, SettingsKeys.CHAT_REFERENCES_ENABLED)
            if setting is None:
                return True  # Default: enabled
            return setting.value.lower() == "true"
    except Exception as err:
        log(f"[V2] ⚠️ Could not read chat references setting: {err}")
        return True


async def is_faq_lookup_enabled() -> bool:
    """
    Read runtime FAQ toggle from admin-managed settings.
    We currently reuse AUTO_LEARNING_ENABLED because this is the FAQ switch
    available in the admin dashboard.
    """
    try:
        from database.connection import async_session_maker
        from models.system_settings import SystemSettings, SettingsKeys

        async with async_session_maker() as db:
            setting = await db.get(SystemSettings, SettingsKeys.AUTO_LEARNING_ENABLED)
            if setting is None:
                return True  # safe default for existing installs
            return setting.value.lower() == "true"
    except Exception as err:
        log(f"[V2] ⚠️ Could not read FAQ setting: {err}")
        return True


async def get_cutoff_result_limit() -> int:
    """Read runtime setting for cutoff SQL result limit."""
    try:
        from database.connection import async_session_maker
        from models.system_settings import SystemSettings, SettingsKeys
        async with async_session_maker() as db:
            setting = await db.get(SystemSettings, SettingsKeys.CUTOFF_COLLEGE_RESULT_LIMIT)
            if setting is None:
                return 10
            try:
                value = int(setting.value)
            except (TypeError, ValueError):
                return 10
            return max(1, min(200, value))
    except Exception as err:
        log(f"[V2] ⚠️ Could not read cutoff result limit setting: {err}")
        return 10


def assess_kb_sufficiency_with_llm(
    client,
    user_question: str,
    kb_tool_result: str,
    conversation_context: Optional[str] = None,
) -> tuple[bool, str, List[str]]:
    """
    LLM-based generic sufficiency check.
    Returns (is_sufficient, reason, web_queries).
    """
    try:
        kb_compact = _trim_tool_result_for_model(kb_tool_result, limit=6000)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a retrieval sufficiency evaluator for NEET counselling queries.\n"
                        "Given the user question and retrieved knowledge-base text, decide if the KB alone "
                        "is enough to answer accurately without inventing facts.\n"
                        "Judge against the user's intent, not naive substring matching.\n"
                        "Same-institution matching: treat as the same college when the KB clearly refers to the institution "
                        "the user meant—common abbreviations vs full official names, minor spelling variants of place names.\n"
                        "Entity-specific questions (critical): If the user names a **specific** college or location "
                        "(any state, any type—government/private/AIIMS/etc.), the KB must contain information that **applies "
                        "to that named institution** for the topic asked (fees, cutoff, seats, etc.). "
                        "Do **not** treat data for a **different** college as sufficient just because it is nearby, "
                        "in the same state, or the same broad category (e.g. another government college). "
                        "Only mark sufficient if the evidence clearly covers the asked institution—e.g. same row in a table, "
                        "explicit naming of that college, or text that states one shared rule/fees for a defined group that "
                        "unambiguously includes the one the user asked about.\n"
                        "Do NOT mark insufficient just because broader background is missing "
                        "(e.g. history when they only asked fee/cutoff).\n"
                        "Use provided conversation context to interpret short follow-ups. "
                        "If current message is short (e.g., 'also for GMC Srinagar'), infer topic from recent context, "
                        "but still require exact entity match for sufficiency.\n"
                        "For relative-reference follow-ups (e.g., 'above top 5 colleges', 'these colleges'), "
                        "first resolve target entities from conversation context. Mark is_sufficient=true only if KB evidence "
                        "covers the requested topic for the resolved entities (or clearly states a shared rule that applies to them). "
                        "If KB covers only a subset, mark is_sufficient=false so missing entities can be fetched via web fallback.\n"
                        "STRICT ENTITY COVERAGE RULE (must enforce):\n"
                        "- If the request resolves to N specific colleges/entities, sufficiency is true ONLY when all N are explicitly covered.\n"
                        "- Do not treat another college from the same state/category/prefix (e.g., another ASMC) as a match.\n"
                        "- If one requested entity is missing exact evidence, mark is_sufficient=false.\n"
                        "- Treat morphological variants as match only when clearly canonical (punctuation/abbreviation/order), not different place names.\n"
                        "- Example: 'Autonomous State Medical College, Kaushambi' is NOT covered by chunks for Hardoi/Ghazipur/Fatehpur.\n"
                        "If retrieved text is for a different college/state than asked, mark is_sufficient=false.\n"
                        "Mark is_sufficient=true when the KB contains the requested data for the correct scope/entity; "
                        "mark false when the requested college/state/round/detail is absent, wrong, or ambiguous.\n"
                        "If is_sufficient=false, also provide targeted web queries ONLY for missing entities/details "
                        "(1 query per missing entity where possible). Do not include already covered entities.\n"
                        "If is_sufficient=true, return empty web_queries.\n"
                        "Return ONLY valid JSON with keys: is_sufficient (boolean), reason (string), web_queries (array of strings)."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"USER_QUESTION:\n{user_question}\n\n"
                        f"CONVERSATION_CONTEXT:\n{conversation_context or '(none)'}\n\n"
                        f"KB_RESULT:\n{kb_compact}\n\n"
                        "Return JSON only."
                    ),
                },
            ],
            temperature=0,
            max_tokens=70,
        )
        raw = (response.choices[0].message.content or "").strip()
        try:
            parsed = json.loads(raw)
            is_sufficient = bool(parsed.get("is_sufficient", False))
            reason = str(parsed.get("reason", "")).strip() or "No reason provided"
            web_queries = [
                str(q).strip()
                for q in (parsed.get("web_queries") or [])
                if str(q).strip()
            ]
            return is_sufficient, reason, web_queries
        except Exception:
            lowered = raw.lower()
            m = re.search(r'"is_sufficient"\s*:\s*(true|false)', lowered)
            if m:
                is_sufficient = m.group(1) == "true"
                reason_match = re.search(r'"reason"\s*:\s*"([^"]*)"', raw, re.IGNORECASE)
                reason = reason_match.group(1).strip() if reason_match else "Parsed from non-JSON sufficiency output"
                log(f"[V2] ⚠️ KB sufficiency non-JSON output parsed via fallback: {raw[:300]!r}")
                return is_sufficient, reason, []
            log(f"[V2] ⚠️ KB sufficiency output unparseable: {raw[:300]!r}")
            return False, "Sufficiency output parse failed", []
    except Exception as err:
        log(f"[V2] ⚠️ KB sufficiency check failed, defaulting to insufficient: {err}")
        return False, "Sufficiency check failed", []


def _looks_like_contextual_in_domain_followup(
    question: str,
    conversation_context: Optional[Dict[str, object]] = None,
) -> bool:
    """
    Detect short/contextual follow-ups that should stay in NEET domain flow.
    Example: after counselling discussion, user asks "when is security money forfeited?"
    """
    if not conversation_context:
        return False

    q = (question or "").strip().lower()
    if not q:
        return False

    detected_topic = str(conversation_context.get("detected_topic") or "").lower()
    detected_state = str(conversation_context.get("detected_state") or "").strip()
    last_user_question = str(conversation_context.get("last_user_question") or "").lower()
    is_followup = bool(conversation_context.get("is_followup"))

    if not (is_followup or detected_topic or detected_state or last_user_question):
        return False

    followup_markers = [
        "what about",
        "in which case",
        "which case",
        "for this",
        "for that",
        "same for",
        "and for",
        "and what",
    ]
    domain_terms = [
        "neet",
        "counselling",
        "admission",
        "round",
        "seat",
        "quota",
        "reservation",
        "category",
        "fee",
        "security",
        "security deposit",
        "forfeit",
        "forfeited",
        "refund",
        "upgradation",
        "resignation",
        "document",
        "eligibility",
        "mcc",
        "aiq",
        "state counselling",
    ]

    short_query = len(q.split()) <= 12
    has_followup_marker = any(marker in q for marker in followup_markers)
    has_domain_term = any(term in q for term in domain_terms)

    prior_topic_is_domain = detected_topic in {
        "fee",
        "eligibility",
        "dates",
        "colleges",
        "cutoff",
        "reservation",
        "process",
        "documents",
    }
    prior_question_is_domain = any(term in last_user_question for term in domain_terms)

    return (
        (short_query or has_followup_marker or has_domain_term)
        and (prior_topic_is_domain or bool(detected_state) or prior_question_is_domain)
    )


def is_query_in_domain(
    question: str,
    conversation_context: Optional[Dict[str, object]] = None,
) -> bool:
    """
    Smart domain check using LLM to understand context.
    Quick rejection for obviously off-topic queries to save LLM cost.
    """
    question_lower = question.lower()
    
    # Context-aware continuation override:
    # if the user is clearly continuing a NEET counselling thread, keep it in-domain.
    if _looks_like_contextual_in_domain_followup(question, conversation_context):
        log("[INFO] ✅ Domain override: contextual in-domain follow-up detected")
        return True

    # Quick rejection for obviously off-topic
    for keyword in OBVIOUSLY_OFF_TOPIC:
        if keyword in question_lower:
            log(f"[INFO] ❌ Quick rejection: '{keyword}' found in query")
            return False
    
    # Use LLM for context-aware domain classification
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """You are a domain classifier for an Indian education counselling chatbot.

The chatbot helps with:
- NEET UG (medical entrance exam) - dates, syllabus, eligibility, application, results
- JEE (engineering entrance exam) - dates, syllabus, eligibility, application, results  
- Medical/Engineering college admissions and counselling
- State and central (AIQ/MCC) counselling processes
- Reservation policies (OBC/SC/ST/EWS/General/PwD)
- Seat matrix, cutoffs, college lists, fees
- Documents required, eligibility criteria, domicile rules
- AIIMS, JIPMER, deemed universities, state medical colleges

Respond with ONLY "YES" if the query is related to any of the above topics.
Respond with ONLY "NO" if the query is completely unrelated (like movies, sports, cooking, stocks, general knowledge, etc.)

Be liberal - if there's ANY reasonable connection to education/admissions/exams, say YES."""
                },
                {
                    "role": "user",
                    "content": (
                        "Is this query related to the education counselling domain?\n\n"
                        f"Conversation context (if available): {json.dumps(conversation_context or {})}\n\n"
                        f"Query: {question}\n\n"
                        "Important: If the query appears to be a follow-up to prior NEET/counselling context, respond YES."
                    )
                }
            ],
            max_tokens=5,
            temperature=0
        )
        
        result = response.choices[0].message.content.strip().upper()
        is_in_domain = result == "YES"
        
        if not is_in_domain:
            log(f"[INFO] ❌ LLM classified as out-of-domain: {question[:50]}...")
        
        return is_in_domain
        
    except Exception as e:
        log(f"[WARN] Domain check LLM error: {e}, defaulting to in-domain")
        # On error, assume in-domain (better to answer than reject)
        return True


def get_llm():
    """Get OpenAI LLM instance (LlamaIndex wrapper)"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set in environment")
    # Default 60s is too low for large RAG prompts + streaming; httpx raises "read operation timed out"
    llm_timeout = float(os.getenv("OPENAI_LLM_TIMEOUT", "300"))
    return LlamaOpenAI(
        model="gpt-4o-mini",
        api_key=api_key,
        temperature=0.1,
        timeout=llm_timeout,
    )


def get_pg_vector_store():
    """LlamaIndex PGVectorStore (PostgreSQL + pgvector)."""
    global vector_store
    vector_store = get_vector_store()
    return vector_store


def detect_query_context(question: str, user_preferences: Optional[Dict] = None) -> Dict:
    """
    Smart query router - detect state, category, and document type from question.
    Uses user preferences as fallback ONLY for state-specific topics.
    
    Priority:
    1. All-India topics (NTA, exam dates, syllabus) → NO state filter
    2. Explicit state mention in query → Use that state
    3. State-specific topics without explicit state → Use user's preference
    4. Everything else → No state filter (All-India search)
    """
    question_lower = question.lower()
    context = {
        "state": None,
        "category": None,
        "document_type": None,
        "preference_used": False,
        "is_all_india_topic": False,
    }
    
    # ========== STEP 1: Check for All-India topics FIRST ==========
    # These topics are UNIVERSAL and should NEVER use state preferences
    all_india_keywords = [
        # NTA / Exam related
        "nta", "neet ug", "neet-ug", "neet 2026", "neet exam",
        "exam date", "exam dates", "important date", "important dates",
        "exam pattern", "pattern", "syllabus", "marking scheme",
        "exam center", "exam centres", "admit card", "hall ticket",
        "result", "results", "scorecard", "answer key",
        # Application related
        "application form", "apply online", "registration fee",
        "neet application", "nta application", "correction window",
        # Eligibility (general)
        "age limit", "attempt limit", "number of attempts",
        "qualifying marks", "passing marks",
        # All India Quota
        "aiq", "all india quota", "all india", "deemed university",
        "central university", "esic", "afmc", "aiims", "jipmer",
    ]
    
    # Check if this is an All-India topic
    if any(kw in question_lower for kw in all_india_keywords):
        context["is_all_india_topic"] = True
        context["state"] = None  # Explicitly no state filter for All-India
        # Don't return yet - continue to detect category/doc type
    
    # ========== STEP 2: Detect explicit state mention ==========
    if not context["is_all_india_topic"]:
        explicit_state = None
        for key, value in STATES.items():
            if key in question_lower:
                explicit_state = value
                break
        
        if explicit_state:
            context["state"] = explicit_state
        elif user_preferences and user_preferences.get("preferred_state"):
            # ========== STEP 3: State-specific topics → Use preference ==========
            # ONLY use preference for truly state-specific topics
            state_specific_keywords = [
                "state quota", "state counselling", "state counseling",
                "state seat", "state cutoff", "state cut-off",
                "state merit", "state rank", "state college",
                "private college", "government college", "govt college",
                "medical college", "mbbs seat", "bds seat",
                "counselling date", "counseling date", "choice filling",
                "reporting", "document verification", "fee structure",
                "tuition fee", "hostel fee", "bond", "stipend",
            ]
            if any(kw in question_lower for kw in state_specific_keywords):
                context["state"] = user_preferences["preferred_state"]
                context["preference_used"] = True
    
    # ========== Detect category ==========
    for category, keywords in CATEGORIES.items():
        if any(keyword in question_lower for keyword in keywords):
            context["category"] = category
            break
    
    # ========== Detect document type ==========
    for doc_type, keywords in DOCUMENT_TYPES.items():
        if any(keyword in question_lower for keyword in keywords):
            context["document_type"] = doc_type
            break
    
    return context


def build_metadata_filters(context: Dict, manual_state: Optional[str] = None) -> Optional[MetadataFilters]:
    """Build metadata filters based on detected context"""
    filters = []
    
    # Use manual state filter if provided, otherwise use detected (including preference-based)
    state = manual_state or context.get("state")
    
    if state and state != "All-India":
        filters.append(
            MetadataFilter(key="state", value=state, operator=FilterOperator.EQ)
        )
    
    if not filters:
        return None
    
    return MetadataFilters(filters=filters)


def build_context_enhanced_prompt(question: str, context: Dict, user_preferences: Optional[Dict] = None) -> str:
    """
    Build an enhanced query prompt that incorporates user preferences for better retrieval.
    This helps the LLM understand the user's context without hard-filtering.
    """
    prompt_additions = []
    
    if context.get("preference_used"):
        if context.get("state"):
            prompt_additions.append(f"(User's preferred state: {context['state']})")
        if context.get("reservation_category"):
            prompt_additions.append(f"(User's category: {context['reservation_category']})")
    
    if prompt_additions:
        # Add context hint to help LLM prioritize relevant info
        return f"{question} {' '.join(prompt_additions)}"
    
    return question


def load_index() -> VectorStoreIndex:
    """Load existing index from PGVectorStore"""
    global index

    if index is not None:
        return index

    llm = get_llm()
    Settings.llm = llm

    vs = get_pg_vector_store()
    index = VectorStoreIndex.from_vector_store(vs)
    print("Index loaded from PostgreSQL pgvector!")

    return index


# ============== API ENDPOINTS ==============

@app.get("/")
async def root():
    """Health check"""
    return {"status": "healthy", "message": "NEET Assistant API is running"}

@app.get("/test-log")
async def test_log():
    """Test if logging works"""
    log("=" * 60)
    log("[TEST] This is a test log message!")
    log("[TEST] If you see this, logging works!")
    log("=" * 60)
    return {"message": "Check your terminal for logs"}


@app.get("/health")
async def health_check():
    """Detailed health check"""
    try:
        n = count_vectors_sync()
        return {
            "status": "healthy",
            "index_loaded": index is not None,
            "vector_store_connected": True,
            "total_vectors": n,
            "vector_store": "pgvector",
        }
    except Exception as e:
        return {
            "status": "degraded",
            "error": str(e)
        }


@app.get("/cutoff/profile/options")
async def get_cutoff_profile_options(
    state: Optional[str] = Query(default=None),
    category: Optional[str] = Query(default=None),
):
    """Public options endpoint for cutoff profile form dropdowns."""
    states = [s for s in await _get_cutoff_db_states_cached() if str(s).upper() != "MCC"]
    selected_state = _canonicalize_state_name(state) if state else None
    normalized_category = _normalize_cutoff_category(category) if category else None
    categories = await _get_cutoff_category_options(selected_state) if selected_state else []
    sub_categories = (
        await _get_cutoff_subcategory_options(selected_state, normalized_category)
        if selected_state and normalized_category
        else []
    )
    return {
        "states": states,
        "categories": categories,
        "sub_categories": sub_categories,
        "selected_state": selected_state,
        "selected_category": normalized_category,
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat endpoint with smart query routing and user preferences"""
    try:
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        # Load index
        idx = load_index()
        
        # Extract user preferences if provided
        user_prefs = None
        if request.user_preferences:
            user_prefs = {
                "preferred_state": request.user_preferences.preferred_state,
                "category": request.user_preferences.category,
            }
        
        # Detect query context (smart routing with preferences)
        context = detect_query_context(request.question, user_prefs)
        print(f"DEBUG: Query context: {context}, preferences_used: {context.get('preference_used')}")
        
        # Build metadata filters for targeted retrieval
        filters = build_metadata_filters(context, request.state_filter)
        
        # Create query engine with filters
        query_engine = idx.as_query_engine(
            similarity_top_k=5,
            response_mode="compact",
            filters=filters  # Apply state/category filters
        )
        
        # Get appropriate context for responses
        if context.get('is_all_india_topic'):
            state_context = "NTA NEET UG"
        elif context.get('state'):
            state_context = context['state']
        else:
            state_context = "NTA NEET UG"
        
        # Custom RAG prompt with professional fallback
        qa_template = PromptTemplate(
            f"""You are a helpful NEET UG 2026 AI assistant. Answer questions based ONLY on the provided context.

RULES:
1. Use ONLY the context below — never invent fees, dates, seat numbers, or college-specific figures.
2. If the context has RELATED information (e.g. registration/counselling fees, fee headings, categories) but not every detail asked (e.g. a named college's full tuition), summarize what IS stated and clearly say what is NOT in these excerpts.
3. Say "I'm sorry, this information is not available at the moment" ONLY when the context has nothing relevant to the question — not when you can partially answer from the context.
4. Be concise, accurate, and professional. Name the document or state when helpful.

Context:
{{context_str}}

Question: {{query_str}}

Answer:"""
        )
        
        query_engine.update_prompts({"response_synthesizer:text_qa_template": qa_template})
        
        # Execute query
        response = query_engine.query(request.question)
        
        # Extract sources with metadata
        sources = []
        if hasattr(response, 'source_nodes'):
            for node in response.source_nodes[:5]:
                metadata = node.node.metadata if hasattr(node.node, 'metadata') else {}
                text = node.node.text if hasattr(node.node, 'text') else str(node.node)
                
                source = Source(
                    file_name=metadata.get("file_name", "Unknown"),
                    page=metadata.get("page_label"),
                    text_snippet=text[:200] + "..." if len(text) > 200 else text,
                    state=metadata.get("state"),
                    document_type=metadata.get("document_type"),
                    doc_topic=metadata.get("doc_topic"),
                    chunk_category=metadata.get("chunk_category")
                    or metadata.get("category"),
                )
                sources.append(source)
        
        return ChatResponse(
            answer=str(response),
            sources=sources if sources else None,
            model_used="openai",
            filters_applied=context if any(context.values()) else None
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Streaming chat endpoint with smart query routing and conversation memory"""
    
    # Log immediately when endpoint is hit (before generator starts)
    log(f"\n{'='*60}")
    log(f"[INFO] 📥 NEW QUERY: {request.question[:100]}...")
    if request.conversation_id:
        log(f"[INFO] 💬 CONVERSATION ID: {request.conversation_id}")
    
    async def generate_stream() -> AsyncGenerator[str, None]:
        # Import memory service here to avoid circular imports
        from services.conversation_memory import (
            ConversationMemory, 
            get_or_create_conversation,
            save_message_to_db,
            build_prompt_with_memory
        )
        from database.connection import async_session_maker
        
        conversation_memory = None
        conversation_id = request.conversation_id
        start_time = datetime.now()
        
        try:
            # Emit an immediate frame to encourage early proxy/client flush for SSE.
            yield f"data: {json.dumps({'type': 'stream_started'})}\n\n"
            references_enabled = await is_chat_references_enabled()
            yield f"data: {json.dumps({'type': 'meta', 'chat_references_enabled': references_enabled})}\n\n"
            if not request.question.strip():
                yield f"data: {json.dumps({'error': 'Question cannot be empty'})}\n\n"
                return
            
            # ========== CREATE OR LOAD CONVERSATION ==========
            if request.user_id:
                try:
                    async with async_session_maker() as conv_db:
                        if request.conversation_id:
                            # Load existing conversation
                            conversation_memory = ConversationMemory(
                                conversation_id=request.conversation_id,
                                user_id=request.user_id
                            )
                            await conversation_memory.load_from_db(conv_db)
                            msg_count = len(conversation_memory.get_chat_history())
                            if msg_count > 0:
                                log(f"[INFO] 🧠 MEMORY LOADED: {msg_count} messages from conversation {request.conversation_id}")
                        else:
                            # Create new conversation for this chat session
                            conversation = await get_or_create_conversation(
                                db=conv_db,
                                user_id=request.user_id,
                                conversation_id=None
                            )
                            conversation_id = conversation.id
                            conversation_memory = ConversationMemory(
                                conversation_id=conversation_id,
                                user_id=request.user_id
                            )
                            log(f"[INFO] 📝 NEW CONVERSATION CREATED: id={conversation_id}")
                except Exception as mem_err:
                    log(f"[WARN] Conversation error: {mem_err}")
                    conversation_memory = None
            
            # Extract user's registered state from preferences (needed for FAQ search)
            user_state = None
            if request.user_preferences and request.user_preferences.preferred_state:
                user_state = request.user_preferences.preferred_state

            onboarding_prompt = _next_onboarding_prompt(med_ctx, request.question)
            if onboarding_prompt:
                med_ctx["stage"] = onboarding_prompt.get("stage", "guided_onboarding")
                yield f"data: {json.dumps({'type': 'suggested_replies', 'replies': onboarding_prompt.get('replies', [])})}\n\n"
                for token in sse_tokens_preserving_formatting(str(onboarding_prompt.get("message", ""))):
                    yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"
                    if V2_STREAM_TOKEN_DELAY_SEC > 0:
                        await asyncio.sleep(V2_STREAM_TOKEN_DELAY_SEC)
                yield f"data: {json.dumps({'type': 'done', 'conversation_id': conversation_id})}\n\n"
                if request.user_id and conversation_id:
                    asyncio.create_task(v2_background_update_conversation_context(conversation_id, med_ctx))
                return
            
            # ========== FAQ CHECK FIRST (BEFORE EVERYTHING) ==========
            # If FAQ matches with high confidence, skip domain check, routing, RAG.
            # Default 0.85: 0.95 was too strict—near-duplicate wording often scores ~0.78–0.92.
            faq_lookup_enabled = await is_faq_lookup_enabled()
            FAQ_SCORE_THRESHOLD = float(os.getenv("FAQ_SCORE_THRESHOLD", "0.85"))
            try:
                if not faq_lookup_enabled:
                    log("[INFO] ⏭️ FAQ lookup skipped (disabled in admin settings)")
                    faq_matches = []
                else:
                    log("[INFO] 🔍 Checking FAQs FIRST...")
                    faq_matches = await search_faq(request.question, state_filter=user_state, top_k=1)
                
                if faq_lookup_enabled and faq_matches and faq_matches[0]["score"] >= FAQ_SCORE_THRESHOLD:
                    faq_match = faq_matches[0]
                    log(f"[INFO] ✅ FAQ MATCH! Score: {faq_match['score']:.3f} (≥{FAQ_SCORE_THRESHOLD})")
                    log(f"[INFO]    Question: {faq_match['question'][:50]}...")
                    
                    # Return FAQ answer directly - skip all other checks!
                    faq_source = {
                        "file_name": "FAQ Database",
                        "page": None,
                        "text_snippet": faq_match["question"],
                        "state": faq_match.get("state"),
                        "document_type": "faq",
                        "category": faq_match.get("category"),
                        "score": round(faq_match["score"], 3)
                    }
                    yield f"data: {json.dumps({'type': 'sources', 'sources': [faq_source]})}\n\n"
                    
                    # Stream the FAQ answer
                    faq_answer = faq_match["answer"]
                    for token in sse_tokens_preserving_formatting(faq_answer):
                        yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"
                        if V2_STREAM_TOKEN_DELAY_SEC > 0:
                            await asyncio.sleep(V2_STREAM_TOKEN_DELAY_SEC)
                    
                    # Save FAQ response to conversation history
                    if request.user_id and conversation_id:
                        try:
                            response_time = int((datetime.now() - start_time).total_seconds() * 1000)
                            async with async_session_maker() as conv_db:
                                await save_message_to_db(
                                    db=conv_db,
                                    conversation_id=conversation_id,
                                    role="user",
                                    content=request.question
                                )
                                await save_message_to_db(
                                    db=conv_db,
                                    conversation_id=conversation_id,
                                    role="assistant",
                                    content=faq_answer,
                                    sources=[faq_source],
                                    was_faq_match=True,
                                    faq_confidence=faq_match["score"],
                                    response_time_ms=response_time
                                )
                                log(f"[INFO]    💾 FAQ response saved to conversation {conversation_id}")
                        except Exception as faq_conv_err:
                            log(f"[WARN]    FAQ conversation save error: {faq_conv_err}")
                    
                    yield f"data: {json.dumps({'type': 'done', 'from_faq': True, 'conversation_id': conversation_id})}\n\n"
                    log("[INFO] ✅ Response served from FAQ (skipped domain check, routing, RAG)")
                    log(f"{'='*60}")
                    return
                else:
                    if faq_matches:
                        log(f"[INFO] ⚠️ FAQ score too low ({faq_matches[0]['score']:.3f} < {FAQ_SCORE_THRESHOLD})")
                    else:
                        log("[INFO] ℹ️ No FAQ matches found")
            except Exception as faq_error:
                log(f"[WARN] FAQ search error: {faq_error}")
            
            # ========== EXTRACT CONVERSATION CONTEXT ==========
            conversation_context = None
            if conversation_memory:
                conversation_context = conversation_memory.extract_conversation_context()
                if conversation_context.get("detected_state"):
                    log(f"[INFO] 📝 Conversation context: state={conversation_context.get('detected_state')}, topic={conversation_context.get('detected_topic')}")

            # ========== GUARDRAILS: Domain restriction ==========
            # Use conversation context so short follow-ups are not wrongly blocked.
            if not is_query_in_domain(request.question, conversation_context):
                log("[WARN] ❌ OUT OF DOMAIN - Query rejected")
                yield f"data: {json.dumps({'type': 'sources', 'sources': []})}\n\n"
                for token in sse_tokens_preserving_formatting(OUT_OF_DOMAIN_RESPONSE):
                    yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"
                    await asyncio.sleep(0.02)
                yield f"data: {json.dumps({'type': 'done', 'out_of_domain': True})}\n\n"
                return
            
            log("[INFO] ✅ DOMAIN CHECK: Query is in domain")
            log(f"[INFO] 👤 USER STATE: {user_state or 'Not set'}")
            
            # ========== SMART QUERY ROUTING ==========
            # FAQ already checked above - if we're here, no FAQ match or score too low
            # Pass conversation context so follow-up questions use the right state
            routing = route_query(
                request.question, 
                user_state, 
                request.clarified_scope,
                conversation_context=conversation_context
            )
            log(f"[INFO] 🎯 ROUTING:")
            log(f"[INFO]    Intent: {routing.intent.value}")
            log(f"[INFO]    Detected State: {routing.detected_state or 'None'}")
            log(f"[INFO]    Use User Preference: {routing.use_user_preference}")
            log(f"[INFO]    Confidence: {routing.confidence:.2f}")
            
            # ========== HANDLE CLARIFICATION NEEDED ==========
            if routing.needs_clarification:
                log("[INFO] ❓ CLARIFICATION NEEDED - asking user to specify scope")
                yield f"data: {json.dumps({'type': 'clarification_needed', 'options': routing.clarification_options or [], 'message': clarification_followup_message(user_state)})}\n\n"
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                return
            
            # Build vector metadata filters
            vector_filters = build_vector_filters(routing, user_state)
            log(f"[INFO] 🔍 VECTOR FILTERS: {vector_filters}")

            # ========== PGVECTOR RAG RETRIEVAL ==========
            try:
                vs = get_pg_vector_store()

                # ========== REFRAME FOLLOW-UP QUESTIONS FOR BETTER RETRIEVAL ==========
                # For vague follow-ups like "what about ST category?", reframe to include context
                original_question = request.question
                search_query = request.question
                
                if conversation_memory:
                    reframed_query = conversation_memory.reframe_query_with_context(request.question)
                    if reframed_query != request.question:
                        search_query = reframed_query
                        log(f"[INFO] 🔄 QUERY REFRAMED for retrieval:")
                        log(f"[INFO]    Original: '{original_question}'")
                        log(f"[INFO]    Reframed: '{search_query}'")
                
                expanded_query = expand_query(search_query)

                log("[INFO] 🧮 Generating query embedding...")
                from llama_index.embeddings.openai import OpenAIEmbedding
                embed_model = OpenAIEmbedding(model="text-embedding-3-small")
                query_embedding = embed_model.get_text_embedding(expanded_query)
                log(f"[INFO] ✅ Embedding generated (dim={len(query_embedding)})")

                all_sources: List[Dict] = []
                all_context_texts: List[str] = []
                central_context = ""
                state_context = ""
                state_name = routing.detected_state or user_state or "your state"
                per_filter_chunks: List[List[Tuple[str, Dict]]] = []

                for filter_idx, pc_filter in enumerate(vector_filters):
                    log(f"[INFO] 🔎 PGVECTOR QUERY {filter_idx + 1}: filter={pc_filter}")

                    mf = vector_filter_to_metadata_filters(pc_filter)
                    vq = VectorStoreQuery(
                        query_embedding=query_embedding,
                        similarity_top_k=6,
                        filters=mf,
                        mode=VectorStoreQueryMode.DEFAULT,
                    )
                    qresult = await vs.aquery(vq)
                    log(f"[INFO]    ↳ Found {len(qresult.nodes)} matches")

                    query_texts: List[str] = []
                    filter_pairs: List[Tuple[str, Dict]] = []
                    for match_idx, node in enumerate(qresult.nodes):
                        score = (
                            qresult.similarities[match_idx]
                            if match_idx < len(qresult.similarities)
                            else 0
                        )
                        text, metadata = _rag_text_from_node(node)

                        if match_idx == 0:
                            log(f"[DEBUG]    Text length: {len(text)}, Preview: {text[:100]}...")

                        if text and len(text) > 50:
                            ascii_ratio = sum(1 for c in text if ord(c) < 128) / max(len(text), 1)
                            if ascii_ratio >= 0.3:
                                query_texts.append(text)
                                src = {
                                    "file_name": metadata.get("file_name", "Unknown"),
                                    "page": metadata.get("page_label"),
                                    "text_snippet": text[:250] + "..." if len(text) > 250 else text,
                                    "state": metadata.get("state"),
                                    "document_type": metadata.get("document_type"),
                                    "doc_topic": metadata.get("doc_topic"),
                                    "chunk_category": metadata.get("chunk_category")
                                    or metadata.get("category"),
                                    "score": round(float(score), 3),
                                }
                                filter_pairs.append((text, src))
                                if match_idx < 3:
                                    ch = metadata.get("chunk_category") or metadata.get("category")
                                    log(
                                        f"[INFO]    📄 Match {match_idx + 1}: score={score:.3f} | "
                                        f"{metadata.get('file_name', 'Unknown')} | page {metadata.get('page_label')} "
                                        f"| doc_topic={metadata.get('doc_topic')} | chunk={ch}"
                                    )
                    per_filter_chunks.append(filter_pairs)

                    # For MIXED intent, separate central and state context (accumulate all state doc types)
                    if routing.intent == QueryIntent.MIXED:
                        if filter_idx == 0:
                            central_context = "\n\n".join(query_texts[:3])
                        else:
                            block = "\n\n".join(query_texts[:3])
                            if block:
                                state_context = (
                                    f"{state_context}\n\n{block}".strip()
                                    if state_context
                                    else block
                                )

                if routing.intent == QueryIntent.STATE_COUNSELLING:
                    if len(per_filter_chunks) > 1:
                        all_context_texts, all_sources = _interleave_chunks_by_filter(
                            per_filter_chunks, max_chunks=12
                        )
                        log(
                            f"[INFO] 📚 STATE_COUNSELLING: interleaved {len(per_filter_chunks)} "
                            f"doc-type searches → {len(all_context_texts)} chunks for the prompt"
                        )
                    elif per_filter_chunks:
                        all_context_texts = [t for t, _ in per_filter_chunks[0]]
                        all_sources = [s for _, s in per_filter_chunks[0]]
                else:
                    for fp in per_filter_chunks:
                        for text, src in fp:
                            all_context_texts.append(text)
                            all_sources.append(src)

                log(f"[INFO] 📚 TOTAL CONTEXT: {len(all_context_texts)} chunks collected")
                
                # Send sources
                yield f"data: {json.dumps({'type': 'sources', 'sources': all_sources[:5]})}\n\n"
                
                # Handle no results
                if not all_context_texts:
                    log("[WARN] ⚠️ NO RESULTS - No relevant chunks found")
                    if routing.intent == QueryIntent.STATE_COUNSELLING:
                        no_result_msg = f"I'm sorry, I don't have {state_name} counselling information available at the moment. Please check the official {state_name} state medical counselling website for accurate details."
                    else:
                        no_result_msg = "I'm sorry, I couldn't find the information you're looking for. Please check the official NTA NEET UG bulletin for accurate details."
                    
                    for token in sse_tokens_preserving_formatting(no_result_msg):
                        yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"
                        await asyncio.sleep(0.02)
                    yield f"data: {json.dumps({'type': 'done', 'intent': routing.intent.value, 'no_data': True})}\n\n"
                    return
                
                # Build prompt based on intent
                log("[INFO] 🤖 LLM GENERATION:")
                registered_user_name = await _get_registered_user_name(request.user_id)
                
                # Get conversation history if available
                conversation_history = ""
                if conversation_memory:
                    history = conversation_memory.get_formatted_history(max_messages=5)
                    if history:
                        conversation_history = f"""
CONVERSATION HISTORY (for context continuity):
{history}

---

"""
                        log(f"[INFO]    📝 Including {len(conversation_memory.get_chat_history())} messages of history")
                
                if routing.intent == QueryIntent.MIXED and central_context and state_context:
                    # Special prompt for MIXED intent
                    log("[INFO]    Using MIXED prompt (central + state)")
                    prompt = format_mixed_response_prompt(
                        central_context, 
                        state_context, 
                        state_name, 
                        request.question,
                        conversation_history=conversation_history
                    )
                else:
                    # Standard prompt (state queries: more + interleaved chunks so college_info isn't starved)
                    if routing.intent == QueryIntent.STATE_COUNSELLING:
                        context_str = "\n\n---\n\n".join(all_context_texts[:10])
                    else:
                        context_str = "\n\n---\n\n".join(all_context_texts[:5])
                    
                    if routing.intent == QueryIntent.EXAM_INFO:
                        source_label = "NTA NEET UG Bulletin"
                    elif routing.intent == QueryIntent.CENTRAL_COUNSELLING:
                        source_label = "NTA NEET UG Bulletin (Central Counselling)"
                    elif routing.intent == QueryIntent.STATE_COUNSELLING:
                        source_label = f"{state_name} state counselling materials (brochure, college/fee documents, and related PDFs)"
                    else:
                        source_label = "official NEET documents"
                    
                    log(f"[INFO]    Source: {source_label}")
                    log(f"[INFO]    Context length: {len(context_str)} chars")
                    
                    # Build comprehensive system message + user prompt
                    personalized_name_guidance = (
                        f'\nPERSONALIZATION:\n'
                        f'- The logged-in student name is "{registered_user_name}".\n'
                        f"- Use the student's first name naturally in key moments (opening line, reassurance, action-oriented guidance), "
                        f"but do not overuse it in every sentence.\n"
                        if registered_user_name else ""
                    )

                    system_message = f"""You are an expert NEET UG 2026 counselling assistant for Indian medical college admissions. You help students understand:
- NEET exam details (syllabus, dates, eligibility, application, results)
- State and All-India counselling processes (MCC, AIQ, state quotas)
- College information, fees, cutoffs, and seat matrix
- Reservation policies (OBC/SC/ST/EWS/General/PwD)
- Required documents and admission procedures

CRITICAL RULES:
1. ONLY use information from the PROVIDED CONTEXT below. Never invent fees, dates, ranks, or percentages.
2. If context has RELATED information (even partial), share it and clarify what's missing.
3. Say "information not available" ONLY when context has NOTHING relevant.
4. Be professional, accurate, and cite the brochure/bulletin when relevant.
5. Use a human counsellor tone with light appreciation/validation when suitable (e.g., "Great question", "Thanks for sharing that").
   Keep it brief and natural (max one short validation line), and avoid repetitive praise.

CONVERSATION CONTINUITY:
- The user may ask follow-up questions that refer to previous context.
- If conversation history is provided, understand the ONGOING topic and state/region being discussed.
- "What about ST category?" after discussing J&K fees means the user wants J&K ST category info, NOT a different state.
- Always maintain context from previous messages when answering follow-ups.
{personalized_name_guidance}

Current Source: {source_label}"""

                    # Build the user message with history and context
                    user_message_parts = []
                    
                    if conversation_history.strip():
                        user_message_parts.append(conversation_history.strip())
                    
                    user_message_parts.append(f"""RETRIEVED CONTEXT:
{context_str}

CURRENT QUESTION: {request.question}

Provide a helpful, accurate answer based on the context above. If this is a follow-up question, use the conversation history to understand what the user is referring to.""")
                    
                    prompt = f"{system_message}\n\n{chr(10).join(user_message_parts)}"
                
                # Stream LLM response
                log("[INFO]    ⏳ Streaming response...")
                llm = get_llm()
                response_stream = llm.stream_complete(prompt)
                full_response = ""
                
                for chunk in response_stream:
                    text_chunk = ""
                    if hasattr(chunk, 'delta') and chunk.delta:
                        text_chunk = chunk.delta
                    elif hasattr(chunk, 'text') and chunk.text:
                        if chunk.text.startswith(full_response):
                            text_chunk = chunk.text[len(full_response):]
                    
                    if text_chunk:
                        full_response += text_chunk
                        yield f"data: {json.dumps({'type': 'token', 'token': text_chunk})}\n\n"
                
                log(f"[INFO] ✅ RESPONSE COMPLETE: {len(full_response)} chars")
                log(f"[INFO]    Preview: {full_response[:100]}...")
                
                # ========== AUTO-LEARNING: Save good Q&A for admin review ==========
                # Only save genuine answers, not "info not available" responses
                response_lower = full_response.lower()
                
                # Phrases that indicate no real answer was found
                skip_phrases = [
                    "couldn't find", "could not find", "not available", "not found",
                    "don't have", "i'm sorry", "i apologize", "unable to find",
                    "no information", "not mentioned", "not specified", "not provided",
                    "cannot provide", "cannot answer", "i cannot", "i don't know",
                    "please check the official", "refer to the official",
                    "out of scope", "outside my knowledge", "beyond my", "not related to neet"
                ]
                
                # Skip only if: contains skip phrases (no length check - short answers are valid!)
                is_skip_response = any(phrase in response_lower for phrase in skip_phrases)
                
                if is_skip_response:
                    log("[INFO]    ⏭️ Auto-learn skipped: Response indicates no info found")
                else:
                    try:
                        from database.connection import async_session_maker
                        from models.pending_qa import PendingQA, QAStatus
                        from models.system_settings import SystemSettings, SettingsKeys
                        from sqlalchemy import select, func
                        
                        # Check if auto-learning is enabled
                        async with async_session_maker() as settings_db:
                            setting = await settings_db.get(SystemSettings, SettingsKeys.AUTO_LEARNING_ENABLED)
                            auto_learning_enabled = setting is None or setting.value.lower() == "true"
                        
                        if not auto_learning_enabled:
                            log("[INFO]    ⏸️ Auto-learn PAUSED by admin setting")
                        else:
                            # User question: store verbatim (only remove null bytes — never append state, rephrase, or trim).
                            verbatim_question = request.question.replace('\x00', '')
                            if not verbatim_question.strip():
                                log("[INFO]    ⏭️ Auto-learn skipped: empty question")
                            else:
                                clean_answer = full_response.replace('\x00', '').strip()
                                
                                # Truncate answer only if too long
                                if len(clean_answer) > 5000:
                                    clean_answer = clean_answer[:5000] + "..."
                            
                                async with async_session_maker() as db:
                                    # Check if similar question already exists
                                    # Use first 100 chars for matching to catch variations
                                    question_pattern = verbatim_question[:100].lower()
                                    existing_query = select(PendingQA).where(
                                        func.lower(PendingQA.question).contains(question_pattern[:50])
                                    ).limit(1)
                                    
                                    result = await db.execute(existing_query)
                                    existing = result.scalar_one_or_none()
                                    
                                    if existing:
                                        # Increment occurrence count for similar question
                                        existing.occurrence_count += 1
                                        await db.commit()
                                        log(f"[INFO]    📊 Similar Q&A exists (id={existing.id}), occurrence count: {existing.occurrence_count}")
                                    else:
                                        # Create new pending Q&A
                                        pending_qa = PendingQA(
                                            question=verbatim_question,
                                            original_answer=clean_answer,
                                            detected_state=routing.detected_state or user_state,
                                            detected_exam="NEET",
                                            detected_category=(
                                                (
                                                    all_sources[0].get("chunk_category")
                                                    or all_sources[0].get("category")
                                                )
                                                if all_sources
                                                else None
                                            ),
                                            source_documents=[{"file": s.get("file_name"), "page": s.get("page")} for s in all_sources[:3]],
                                            original_confidence=all_sources[0].get("score") if all_sources else None,
                                            status=QAStatus.PENDING,
                                            occurrence_count=1
                                        )
                                        db.add(pending_qa)
                                        await db.commit()
                                        log(f"[INFO]    📝 New Q&A saved for admin review (id={pending_qa.id})")
                                
                    except Exception as auto_learn_err:
                        log(f"[WARN]    Auto-learn error: {auto_learn_err}")
                
                # ========== SAVE TO CONVERSATION HISTORY ==========
                if request.user_id and conversation_id:
                    try:
                        response_time = int((datetime.now() - start_time).total_seconds() * 1000)
                        async with async_session_maker() as conv_db:
                            # Save user message
                            await save_message_to_db(
                                db=conv_db,
                                conversation_id=conversation_id,
                                role="user",
                                content=request.question,
                                filters_applied={"intent": routing.intent.value, "state": routing.detected_state}
                            )
                            # Save assistant response
                            await save_message_to_db(
                                db=conv_db,
                                conversation_id=conversation_id,
                                role="assistant",
                                content=full_response,
                                sources=[s for s in all_sources[:5]],
                                model_used="gpt-4o-mini",
                                was_faq_match=False,
                                response_time_ms=response_time
                            )
                            log(f"[INFO]    💾 Messages saved to conversation {conversation_id}")
                    except Exception as conv_err:
                        log(f"[WARN]    Conversation save error: {conv_err}")
                
                log("=" * 60)
                yield f"data: {json.dumps({'type': 'done', 'intent': routing.intent.value, 'source': 'rag', 'conversation_id': conversation_id})}\n\n"
                
            except Exception as rag_err:
                import traceback
                log(f"[ERROR] ❌ RAG ERROR: {rag_err}")
                traceback.print_exc()
                yield f"data: {json.dumps({'type': 'sources', 'sources': []})}\n\n"
                yield f"data: {json.dumps({'type': 'token', 'token': f'Sorry, there was an error processing your query: {str(rag_err)}'})}\n\n"
                yield f"data: {json.dumps({'type': 'done', 'error': True})}\n\n"
            
        except Exception as e:
            import traceback
            log(f"[ERROR] ❌ STREAM ERROR: {e}")
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Content-Type": "text/event-stream; charset=utf-8",
            "Connection": "keep-alive",
            "Pragma": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )


# ============== ADMIN ENDPOINTS ==============

@app.get("/admin/stats")
async def get_index_stats():
    """Get vector index statistics (pgvector)"""
    try:
        n = count_vectors_sync()
        return {
            "total_vectors": n,
            "namespaces": {"_default": {"vector_count": n}},
            "index_name": os.getenv("PGVECTOR_TABLE_NAME", "neet_assistant"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/admin/upload")
async def upload_document(
    file: UploadFile = File(...),
    state: str = Form(...),
    document_type: str = Form(...),
    category: str = Form(...),
    year: str = Form("2026"),
    description: str = Form("")
):
    """Upload and index a new document with metadata (PDF stored in R2)"""
    global index
    
    temp_file_path = None
    storage_path = None
    storage_url = None
    
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        # Generate file ID
        file_id = str(uuid.uuid4())[:8]
        safe_filename = re.sub(r'[^a-zA-Z0-9._-]', '_', file.filename)
        
        # Read file content
        file_content = await file.read()
        file_size_kb = len(file_content) / 1024
        
        # Save to temp file for PDF extraction
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            tmp.write(file_content)
            temp_file_path = Path(tmp.name)
        
        print(f"Temp file saved: {temp_file_path}")
        
        # Extract text, drop blank pages, merge multi-page fee tables when applicable
        pages_raw = extract_text_from_pdf(temp_file_path)
        try:
            total_pdf_pages = len(PdfReader(str(temp_file_path)).pages)
        except Exception:
            total_pdf_pages = max((p["page_num"] for p in pages_raw), default=0) if pages_raw else 0
        pages = prepare_pages_for_indexing(pages_raw, document_type, category)

        if not pages:
            raise HTTPException(status_code=400, detail="Could not extract text from PDF (all pages empty or too short)")

        print(
            f"Prepared {len(pages)} indexing unit(s) from {total_pdf_pages} PDF page(s) "
            f"(document_type={document_type}, doc_topic={category})"
        )

        # Create documents with metadata (text added separately after chunking)
        documents = []
        for page_data in pages:
            page_text = page_data["text"]
            doc = Document(
                text=page_text,
                metadata={
                    "file_name": file.filename,
                    "file_id": file_id,
                    "page_label": format_page_label(page_data),
                    "state": state,
                    "document_type": document_type,
                    # Whole-document scope chosen in admin (sub-category: fees, eligibility, …)
                    "doc_topic": category,
                    "year": year,
                    "description": description or "",
                    "uploaded_at": datetime.now().isoformat(),
                },
            )
            documents.append(doc)
        
        print(f"Created {len(documents)} document chunks")
        
        vs = get_pg_vector_store()
        
        # Create storage context
        storage_context = StorageContext.from_defaults(vector_store=vs)
        
        # Configure LLM and embed model
        Settings.llm = get_llm()
        
        # Index documents in batches to avoid API rate limits
        BATCH_SIZE = 10  # Process 10 documents at a time
        total_indexed = 0
        
        # Set embedding model and chunk size explicitly
        from llama_index.embeddings.openai import OpenAIEmbedding
        from llama_index.core.node_parser import SentenceSplitter
        
        Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
        chunk_size, chunk_overlap = get_chunk_settings_for_document(document_type, category)
        Settings.chunk_size = chunk_size
        Settings.chunk_overlap = chunk_overlap

        # Create node parser that adds text to metadata for vector retrieval
        node_parser = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            include_metadata=True,
            include_prev_next_rel=False,
        )
        
        # Parse documents into nodes and add text to metadata
        all_nodes = node_parser.get_nodes_from_documents(documents, show_progress=True)
        
        # Classify each node and add enhanced metadata
        print(f"Classifying {len(all_nodes)} chunks...")
        for i, node in enumerate(all_nodes):
            chunk_text = node.get_content()
            # Do not duplicate chunk text in metadata — node content is already stored for embedding/RAG.

            # Use chunk classifier to get proper category
            classification = classify_chunk(
                text=chunk_text,
                document_type=document_type,
                state=state
            )
            
            # Per-chunk AI labels (do not reuse doc_topic — that is admin upload scope for the whole file)
            node.metadata["chunk_category"] = classification["category"]
            node.metadata["chunk_section"] = classification["section"]
            node.metadata["chunk_importance"] = classification["importance"]
            
            if (i + 1) % 20 == 0:
                print(f"Classified {i + 1}/{len(all_nodes)} chunks...")
        
        print(f"Created and classified {len(all_nodes)} nodes from {len(documents)} documents")
        
        try:
            # Index nodes in batches
            for i in range(0, len(all_nodes), BATCH_SIZE):
                batch = all_nodes[i:i + BATCH_SIZE]
                batch_num = (i // BATCH_SIZE) + 1
                total_batches = (len(all_nodes) + BATCH_SIZE - 1) // BATCH_SIZE
                print(f"Indexing batch {batch_num}/{total_batches} ({len(batch)} nodes)...")
                
                if index is not None:
                    # Add to existing index
                    index.insert_nodes(batch)
                else:
                    # Create new index with first batch
                    index = VectorStoreIndex(
                        nodes=batch,
                        storage_context=storage_context,
                        show_progress=True
                    )
                
                total_indexed += len(batch)
                print(f"Batch {batch_num} completed. Total indexed: {total_indexed}")
                
                # Small delay between batches to respect rate limits
                if i + BATCH_SIZE < len(all_nodes):
                    import time
                    time.sleep(0.5)  # 0.5 second delay between batches
            
            print(f"Successfully indexed {total_indexed} nodes")
            
            # IMPORTANT: Refresh the global index to use new vectors
            index = VectorStoreIndex.from_vector_store(vs)
            print("Index cache refreshed with new vectors")
            
        except Exception as embed_err:
            import traceback
            print(f"INDEXING ERROR: {embed_err}")
            traceback.print_exc()
            
            error_msg = str(embed_err)
            if "rate_limit" in error_msg.lower() or "429" in error_msg:
                raise HTTPException(
                    status_code=429, 
                    detail=f"OpenAI rate limit exceeded. Indexed {total_indexed}/{len(documents)} pages. Please wait a minute and try again with a smaller document."
                )
            elif "quota" in error_msg.lower() or "insufficient_quota" in error_msg.lower():
                raise HTTPException(
                    status_code=402, 
                    detail="OpenAI API quota exceeded. Please check your billing settings at platform.openai.com"
                )
            else:
                raise HTTPException(
                    status_code=500, 
                    detail=f"Embedding error after {total_indexed} pages: {error_msg}"
                )
        
        total_vectors = count_vectors_sync()

        try:
            from services.r2_storage import upload_pdf_to_r2
            storage_path, storage_url = await upload_pdf_to_r2(
                file_content=file_content,
                file_id=file_id,
                original_filename=file.filename,
                state=state,
                document_type=document_type,
            )
            print(f"✅ Uploaded to R2: {storage_path}")
        except Exception as storage_err:
            print(f"⚠️ R2 upload failed (continuing without storage): {storage_err}")
            storage_path = None
            storage_url = None
        
        # Save to database with versioning
        new_version = 1
        deactivated_docs = []
        try:
            from database.connection import async_session_maker
            from models.indexed_document import IndexedDocument
            from sqlalchemy import select, and_
            
            async with async_session_maker() as db:
                # Check for existing documents with same (state, document_type, category, year)
                existing_query = select(IndexedDocument).where(
                    and_(
                        IndexedDocument.state == state,
                        IndexedDocument.document_type == document_type,
                        IndexedDocument.category == category,
                        IndexedDocument.year == year,
                        IndexedDocument.is_active == True
                    )
                )
                existing_result = await db.execute(existing_query)
                existing_docs = existing_result.scalars().all()
                
                if existing_docs:
                    # Find highest version number
                    max_version = max(doc.version for doc in existing_docs)
                    new_version = max_version + 1
                    
                    # Deactivate old versions and delete their vectors from pgvector
                    for old_doc in existing_docs:
                        old_doc.is_active = False
                        old_doc.index_status = "superseded"
                        deactivated_docs.append(old_doc.file_id)

                        try:
                            mf_del = MetadataFilters(
                                filters=[
                                    MetadataFilter(
                                        key="file_id",
                                        value=old_doc.file_id,
                                        operator=FilterOperator.EQ,
                                    )
                                ]
                            )
                            vs.delete_nodes(filters=mf_del)
                            print(f"Deleted vectors for old version: {old_doc.file_id}")
                        except Exception as vec_err:
                            print(f"Warning: Could not delete old vectors: {vec_err}")
                    
                    print(f"Deactivated {len(existing_docs)} old version(s), new version: {new_version}")
                
                # Create new document record with version
                indexed_doc = IndexedDocument(
                    file_id=file_id,
                    filename=f"{file_id}_{safe_filename}",
                    original_filename=file.filename,
                    state=state,
                    document_type=document_type,
                    category=category,
                    year=year,
                    description=description,
                    version=new_version,
                    total_pages=total_pdf_pages,
                    total_vectors=total_indexed,  # Use actual indexed count
                    file_size_kb=round(file_size_kb, 2),
                    storage_path=storage_path,
                    storage_url=storage_url,
                    is_active=True,
                    index_status="indexed"
                )
                db.add(indexed_doc)
                await db.commit()
                print(f"Document tracked in database: {file_id} (v{new_version})")
        except Exception as db_err:
            print(f"Warning: Could not save to database: {db_err}")
        
        # Clean up temp file
        if temp_file_path and temp_file_path.exists():
            try:
                temp_file_path.unlink()
                print(f"Cleaned up temp file: {temp_file_path}")
            except Exception as del_err:
                print(f"Warning: Could not delete temp file {temp_file_path}: {del_err}")
        
        return {
            "success": True,
            "message": f"Successfully indexed {total_indexed} chunks from {file.filename} ({total_pdf_pages} PDF pages)",
            "file_id": file_id,
            "version": new_version,
            "pages_indexed": total_pdf_pages,
            "chunks_indexed": total_indexed,
            "metadata": {
                "state": state,
                "document_type": document_type,
                "category": category,
                "doc_topic": category,
                "year": year,
            },
            "total_vectors": total_vectors,
            "storage": {
                "path": storage_path,
                "url": storage_url
            } if storage_path else None,
            "deactivated_versions": deactivated_docs if deactivated_docs else None
        }
        
    except HTTPException:
        # Clean up temp file on error
        if temp_file_path and temp_file_path.exists():
            try:
                temp_file_path.unlink()
            except:
                pass
        raise
    except Exception as e:
        # Clean up temp file on error
        if temp_file_path and temp_file_path.exists():
            try:
                temp_file_path.unlink()
            except:
                pass
        raise HTTPException(status_code=500, detail=f"Error indexing document: {str(e)}")


# NOTE: /admin/documents endpoints are handled by routes/admin.py with database support


@app.get("/admin/metadata-options")
async def get_metadata_options():
    """Get available metadata options for document upload"""
    return {
        "states": [
            "All-India",
            "Andaman and Nicobar Islands",
            "Andhra Pradesh",
            "Arunachal Pradesh",
            "Assam",
            "Bihar",
            "Chandigarh",
            "Chhattisgarh",
            "Dadra and Nagar Haveli and Daman and Diu",
            "Delhi",
            "Goa",
            "Gujarat",
            "Haryana",
            "Himachal Pradesh",
            "Jammu & Kashmir",
            "Jharkhand",
            "Karnataka",
            "Kerala",
            "Ladakh",
            "Lakshadweep",
            "Madhya Pradesh",
            "Maharashtra",
            "Manipur",
            "Meghalaya",
            "Mizoram",
            "Nagaland",
            "Odisha",
            "Puducherry",
            "Punjab",
            "Rajasthan",
            "Sikkim",
            "Tamil Nadu",
            "Telangana",
            "Tripura",
            "Uttar Pradesh",
            "Uttarakhand",
            "West Bengal",
        ],
        "document_types": [
            {"value": "nta_bulletin", "label": "NTA Official Bulletin"},
            {"value": "mcc_counseling", "label": "MCC Counseling Guide"},
            {"value": "state_counseling", "label": "State Counseling Guide"},
            {"value": "college_info", "label": "College/Institute Info"},
            {"value": "cutoffs", "label": "Previous Year Cutoffs"},
            {"value": "faq", "label": "FAQ Document"},
            {"value": "other", "label": "Other"}
        ],
        "categories": [
            {"value": "general", "label": "Comprehensive (All Topics)"},
            {"value": "eligibility", "label": "Eligibility Criteria"},
            {"value": "dates", "label": "Important Dates"},
            {"value": "fees", "label": "Fees & Payments"},
            {"value": "colleges", "label": "Colleges & Seats"},
            {"value": "cutoff", "label": "Cutoffs & Ranks"},
            {"value": "process", "label": "Process & Procedure"},
            {"value": "documents", "label": "Required Documents"}
        ],
        "years": ["2024", "2025", "2026", "2027"]
    }


@app.get("/models")
async def get_available_models():
    """Get available models"""
    return {
        "models": [
            {
                "id": "openai",
                "name": "GPT-4o-mini",
                "provider": "OpenAI",
                "description": "OpenAI's efficient GPT-4 variant"
            }
        ]
    }


@app.delete("/admin/vectors/clear")
async def clear_all_vectors():
    """Clear ALL vectors from pgvector table (use with caution!)"""
    global index
    try:
        vs = get_pg_vector_store()
        total_before = count_vectors_sync()
        await vs.aclear()
        index = None

        return {
            "success": True,
            "message": f"Deleted {total_before} vectors from pgvector",
            "action": "Please re-upload your documents to rebuild the knowledge base",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear vectors: {str(e)}")


@app.get("/admin/vectors/sample")
async def get_sample_vectors():
    """Get sample vectors to check metadata structure"""
    try:
        vs = get_pg_vector_store()
        total = count_vectors_sync()

        from llama_index.embeddings.openai import OpenAIEmbedding
        embed_model = OpenAIEmbedding(model="text-embedding-3-small")
        sample_query = embed_model.get_text_embedding("NEET eligibility")

        vq = VectorStoreQuery(
            query_embedding=sample_query,
            similarity_top_k=3,
            mode=VectorStoreQueryMode.DEFAULT,
        )
        qresult = await vs.aquery(vq)

        samples = []
        for i, node in enumerate(qresult.nodes):
            score = qresult.similarities[i] if i < len(qresult.similarities) else 0
            metadata = dict(node.metadata) if node.metadata else {}
            vec_id = getattr(node, "node_id", None) or metadata.get("doc_id", "unknown")
            text_preview = ""
            if hasattr(node, "get_content"):
                text_preview = (node.get_content() or "")[:100]
            samples.append({
                "id": vec_id,
                "score": score,
                "metadata_keys": list(metadata.keys()),
                "text_preview": text_preview + "..." if len(text_preview) >= 100 else text_preview,
                "file_name": metadata.get("file_name"),
                "state": metadata.get("state"),
                "is_faq": metadata.get("is_faq", False),
            })

        return {
            "total_vectors": total,
            "samples": samples,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============== UNIFIED CHAT V2 (TOOL-BASED ARCHITECTURE) ==============

async def _v2_run_rag_pipeline(
    *,
    request: ChatRequest,
    client,
    messages: List[Dict[str, object]],
    available_tools: List[Dict[str, object]],
    web_fallback_enabled: bool,
    references_enabled: bool,
    med_ctx: Dict[str, object],
    preferred_language: str,
    localize_output,
    request_started_at: float,
    round_stats: List[Dict[str, object]],
    state: Dict[str, object],
) -> AsyncGenerator[str, None]:
    """
    Stage function for V2 RAG pipeline:
    - tool loop (KB-first, optional web fallback)
    - final answer generation and token streaming
    - optional suggested replies
    """
    from services.knowledge_tool import execute_tool_call

    used_web_fallback = False
    max_tool_rounds = 3
    assistant_message = None
    full_response = ""
    kb_attempted = False
    kb_sufficient_for_final = False
    kb_insufficient_and_web_disabled = False
    forced_fallback_response = None
    loop_final_assistant_content: Optional[str] = None
    last_kb_retrieval: Optional[str] = None
    last_web_retrieval: Optional[str] = None
    streamed_final_in_loop = False
    direct_web_fallback_done = False
    # Track stream-time origin so we can upgrade KB -> WEB if fallback is used.
    stream_source_origin: Optional[str] = None
    final_ttft_ms = 0.0
    final_stream_ms = 0.0
    user_visible_ttft_ms = 0.0
    final_temperature = _final_answer_temperature(request.question)
    final_max_tokens = _final_answer_max_tokens(request.question)

    for round_idx in range(max_tool_rounds):
        log(f"[V2] 🤖 Tool round {round_idx + 1}/{max_tool_rounds}")
        _log_v2_tool_debug(
            f"[V2][DBG] round={round_idx + 1} user_question={request.question!r} "
            f"messages_tail={_summarize_messages_for_debug(messages)}"
        )
        if preferred_language == "en" and round_idx > 0 and kb_attempted and kb_sufficient_for_final:
            t_llm = time.perf_counter()
            _first_out = True
            streamed_raw = ""
            async for delta in _stream_chat_completion_text(
                client,
                model="gpt-4o-mini",
                messages=messages,
                temperature=final_temperature,
                max_tokens=final_max_tokens,
            ):
                streamed_raw += delta
                if _first_out:
                    final_ttft_ms = _elapsed_ms(t_llm)
                    if user_visible_ttft_ms <= 0:
                        user_visible_ttft_ms = _elapsed_ms(request_started_at)
                    _first_out = False
                yield f"data: {json.dumps({'type': 'token', 'token': delta})}\n\n"

            llm_round_ms = _elapsed_ms(t_llm)
            full_response = _apply_response_policy(streamed_raw, request.question)
            if full_response.startswith(streamed_raw):
                tail = full_response[len(streamed_raw):]
                if tail:
                    for token in sse_tokens_preserving_formatting(tail):
                        if user_visible_ttft_ms <= 0:
                            user_visible_ttft_ms = _elapsed_ms(request_started_at)
                        yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"
                        if V2_STREAM_TOKEN_DELAY_SEC > 0:
                            await asyncio.sleep(V2_STREAM_TOKEN_DELAY_SEC)
            else:
                full_response = streamed_raw

            final_stream_ms = _elapsed_ms(t_llm)
            streamed_final_in_loop = True
            round_stats.append({"i": round_idx + 1, "llm": llm_round_ms, "tool": None, "tool_exec": 0.0, "suff": 0.0})
            break

        round_tools = available_tools
        if round_idx == 0:
            round_tools = [t for t in available_tools if t.get("function", {}).get("name") == "search_knowledge_base"]
        t_llm = time.perf_counter()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=round_tools,
            tool_choice="auto",
            temperature=0.3,
            max_tokens=V2_FINAL_MAX_TOKENS
        )
        llm_round_ms = _elapsed_ms(t_llm)
        assistant_message = response.choices[0].message
        _log_v2_tool_debug(
            f"[V2][DBG] round={round_idx + 1} assistant_tool_calls_count="
            f"{len(assistant_message.tool_calls or [])}"
        )
        if not assistant_message.tool_calls:
            loop_final_assistant_content = (assistant_message.content or "").strip()
            _log_v2_tool_debug(
                f"[V2][DBG] round={round_idx + 1} no tool call; assistant_content_preview="
                f"{(loop_final_assistant_content[:160] + '...') if len(loop_final_assistant_content) > 160 else loop_final_assistant_content!r}"
            )
            round_stats.append({"i": round_idx + 1, "llm": llm_round_ms, "tool": None, "tool_exec": 0.0, "suff": 0.0})
            break

        tool_calls = assistant_message.tool_calls or []
        kb_results_this_round: List[str] = []
        kb_call_records: List[Dict[str, Any]] = []
        round_tool_names: List[str] = []
        round_tool_exec_ms = 0.0
        pending_kb_tool_groups: List[Dict[str, Any]] = []
        pending_other_tool_messages: List[Dict[str, Any]] = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments or "{}")
            _log_v2_tool_debug(
                f"[V2][DBG] round={round_idx + 1} tool_selected name={tool_name} args={json.dumps(tool_args, ensure_ascii=False)}"
            )
            round_tool_names.append(tool_name)
            t_tool = time.perf_counter()
            tool_result, success = execute_tool_call(tool_name, tool_args)
            tool_exec_ms = _elapsed_ms(t_tool)
            round_tool_exec_ms += tool_exec_ms
            _log_v2_tool_debug(
                f"[V2][DBG] round={round_idx + 1} tool_result name={tool_name} "
                f"success={success} exec_ms={tool_exec_ms:.0f} preview="
                f"{((tool_result or '')[:160].replace(chr(10), ' ') + '...') if len(tool_result or '') > 160 else (tool_result or '').replace(chr(10), ' ')!r}"
            )
            if success:
                if tool_name == "search_knowledge_base":
                    kb_results_this_round.append(tool_result)
                    kb_call_records.append(
                        {
                            "query": str(tool_args.get("query", "")),
                            "state": str(tool_args.get("state", "")),
                            "result": tool_result,
                        }
                    )
                    # Emit KB only when no origin is set yet.
                    if stream_source_origin is None:
                        yield f"data: {json.dumps({'type': 'meta', 'source_origin': 'kb'})}\n\n"
                        stream_source_origin = "kb"
                elif tool_name == "search_web":
                    last_web_retrieval = tool_result
                    # If web fallback is used, always upgrade origin to WEB.
                    if stream_source_origin != "web":
                        yield f"data: {json.dumps({'type': 'meta', 'source_origin': 'web'})}\n\n"
                        stream_source_origin = "web"
                    used_web_fallback = True

            sources = []
            if tool_name == "search_web":
                for line in tool_result.split("\n"):
                    if line.startswith("[") and "Title:" in line:
                        title = line.split("Title:", 1)[1].strip()
                        sources.append({"file_name": title, "document_type": "web_search"})
            elif "State:" in tool_result or "Type:" in tool_result:
                for line in tool_result.split("\n"):
                    if line.startswith("[") and "] State:" in line:
                        parts = line.split(" | ")
                        source_info = {}
                        for part in parts:
                            if "State:" in part:
                                source_info["state"] = part.split("State:", 1)[1].strip()
                            elif "Type:" in part:
                                source_info["document_type"] = part.split("Type:", 1)[1].strip()
                            elif "Source:" in part:
                                source_info["file_name"] = part.split("Source:", 1)[1].strip()
                            elif "Page:" in part:
                                source_info["page"] = part.split("Page:", 1)[1].strip()
                        if source_info:
                            sources.append(source_info)
            if sources and references_enabled:
                yield f"data: {json.dumps({'type': 'sources', 'sources': sources[:5]})}\n\n"

            tool_pair = [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": tool_call.id,
                            "type": "function",
                            "function": {"name": tool_name, "arguments": tool_call.function.arguments},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": tool_call.id, "content": _trim_tool_result_for_model(tool_result)},
            ]
            if tool_name == "search_knowledge_base":
                pending_kb_tool_groups.append(
                    {
                        "query": str(tool_args.get("query", "")),
                        "tool_pair": tool_pair,
                    }
                )
            else:
                pending_other_tool_messages.extend(tool_pair)

        suff_ms = 0.0
        if kb_results_this_round:
            kb_attempted = True
            last_kb_retrieval = "\n\n".join(kb_results_this_round)
            t_suff = time.perf_counter()
            # Per-tool-call sufficiency (strict): evaluate each KB call independently
            # using the same query that was sent to that tool call.
            suff_web_queries: List[str] = []
            per_call_reasons: List[str] = []
            per_call_sufficiency_map: Dict[str, bool] = {}
            all_calls_sufficient = True

            for rec in kb_call_records:
                kb_query = str(rec.get("query", "")).strip()
                kb_result = str(rec.get("result", ""))
                if not kb_query:
                    continue
                call_sufficient, call_reason, _call_web_queries = assess_kb_sufficiency_with_llm(
                    client=client,
                    user_question=kb_query,
                    kb_tool_result=kb_result,
                    conversation_context=_build_sufficiency_context(messages, kb_query),
                )
                _log_v2_tool_debug(
                    f"[V2][DBG] per_call_sufficiency query={kb_query!r} "
                    f"is_sufficient={bool(call_sufficient)} reason={call_reason!r}"
                )
                per_call_sufficiency_map[kb_query] = bool(call_sufficient)
                if not call_sufficient:
                    all_calls_sufficient = False
                    per_call_reasons.append(f"{kb_query}: {call_reason}")
                    suff_web_queries.append(kb_query)

            is_sufficient = all_calls_sufficient
            if is_sufficient:
                _reason = "All per-call KB sufficiency checks passed."
            else:
                _reason = "; ".join(per_call_reasons[:3]) or "One or more per-call KB sufficiency checks failed."
            suff_ms = _elapsed_ms(t_suff)
            _log_v2_tool_debug(
                f"[V2][DBG] sufficiency round={round_idx + 1} "
                f"is_sufficient={bool(is_sufficient)} reason={_reason!r}"
            )
            kb_sufficient_for_final = bool(is_sufficient)
            if not is_sufficient:
                if web_fallback_enabled:
                    # Keep ONLY sufficient KB evidence in final context.
                    if pending_kb_tool_groups:
                        kept_kb_groups = 0
                        dropped_kb_groups = 0
                        for grp in pending_kb_tool_groups:
                            q = str(grp.get("query", "")).strip()
                            keep = per_call_sufficiency_map.get(q, False)
                            if keep:
                                messages.extend(grp.get("tool_pair") or [])
                                kept_kb_groups += 1
                            else:
                                dropped_kb_groups += 1
                        _log_v2_tool_debug(
                            f"[V2][DBG] kb_groups_kept={kept_kb_groups} kb_groups_dropped={dropped_kb_groups}"
                        )
                        pending_kb_tool_groups = []

                    web_queries = [
                        str(q).strip()
                        for q in (suff_web_queries or [])
                        if str(q).strip()
                    ]
                    _log_v2_tool_debug(
                        f"[V2][DBG] sufficiency_web_queries={web_queries}"
                    )

                    if not web_queries:
                        kb_insufficient_and_web_disabled = True
                        web_queries = []

                    merged_web_payloads: List[str] = []
                    all_web_sources: List[Dict[str, str]] = []
                    any_web_success = False

                    for fallback_query in web_queries[:5]:
                        t_web = time.perf_counter()
                        web_result, web_success = execute_tool_call("search_web", {"query": fallback_query})
                        web_exec_ms = _elapsed_ms(t_web)
                        round_tool_exec_ms += web_exec_ms
                        round_tool_names.append("search_web(direct)")
                        _log_v2_tool_debug(
                            f"[V2][DBG] direct_web_fallback query={fallback_query!r} "
                            f"success={web_success} exec_ms={web_exec_ms:.0f} preview="
                            f"{((web_result or '')[:160].replace(chr(10), ' ') + '...') if len(web_result or '') > 160 else (web_result or '').replace(chr(10), ' ')!r}"
                        )
                        _log_v2_tool_debug(
                            "[V2][DBG] direct_web_fallback full_result_start=\n"
                            + _trim_tool_result_for_model(web_result or "", limit=2500)
                        )
                        if not web_success:
                            continue

                        any_web_success = True
                        merged_web_payloads.append(web_result)
                        for line in web_result.split("\n"):
                            if line.startswith("[") and "Title:" in line:
                                title = line.split("Title:", 1)[1].strip()
                                all_web_sources.append({"file_name": title, "document_type": "web_search"})

                        direct_tool_id = f"direct_web_{uuid.uuid4().hex[:8]}"
                        messages.append({
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [{
                                "id": direct_tool_id,
                                "type": "function",
                                "function": {"name": "search_web", "arguments": json.dumps({"query": fallback_query})}
                            }]
                        })
                        messages.append({
                            "role": "tool",
                            "tool_call_id": direct_tool_id,
                            "content": _trim_tool_result_for_model(web_result),
                        })

                    if any_web_success:
                        last_web_retrieval = "\n\n".join(merged_web_payloads)
                        _log_v2_tool_debug(
                            "[V2][DBG] merged_web_retrieval_for_final=\n"
                            + _trim_tool_result_for_model(last_web_retrieval or "", limit=4000)
                        )
                        used_web_fallback = True
                        if stream_source_origin != "web":
                            yield f"data: {json.dumps({'type': 'meta', 'source_origin': 'web'})}\n\n"
                            stream_source_origin = "web"
                        if all_web_sources and references_enabled:
                            yield f"data: {json.dumps({'type': 'sources', 'sources': all_web_sources[:5]})}\n\n"

                        messages.append({
                            "role": "system",
                            "content": (
                                "SOURCE MERGE RULES (highest priority):\n"
                                "- Use KB evidence for entities covered by KB.\n"
                                "- Use web evidence only for entities missing in KB.\n"
                                "- Never replace KB-backed values with web estimates.\n"
                                "- If an entity is missing in both KB and web, provide one polite professional fallback note instead of listing multiple missing fields.\n"
                                "- Ignore chunks that are about different entities.\n"
                                "- For a web-only entity, output numeric fee/cutoff values ONLY if that entity's evidence text explicitly contains those numbers.\n"
                                "- If web evidence for an entity has no explicit numeric figure, do NOT fabricate per-field values; mark that entity as unavailable in the same polite fallback note.\n"
                                "- Never borrow numbers from similar colleges/states for a missing entity.\n"
                                "- Preferred fallback wording: 'Sorry, I am not able to provide reliable details for this right now. Please verify directly from the official college website or authorised counselling sources.'\n"
                                "- Keep the fallback concise, user-friendly, and at most once in the response."
                            ),
                        })
                        direct_web_fallback_done = True
                    else:
                        kb_insufficient_and_web_disabled = True
                else:
                    kb_insufficient_and_web_disabled = True
            if is_sufficient:
                # Keep KB retrieval in final context only when sufficiency is true.
                if pending_kb_tool_groups:
                    for grp in pending_kb_tool_groups:
                        messages.extend(grp.get("tool_pair") or [])
                if pending_other_tool_messages:
                    messages.extend(pending_other_tool_messages)
        elif kb_results_this_round:
            kb_attempted = True
            last_kb_retrieval = "\n\n".join(kb_results_this_round)
            kb_sufficient_for_final = True
            if pending_kb_tool_groups:
                for grp in pending_kb_tool_groups:
                    messages.extend(grp.get("tool_pair") or [])
            if pending_other_tool_messages:
                messages.extend(pending_other_tool_messages)
        else:
            if pending_kb_tool_groups:
                for grp in pending_kb_tool_groups:
                    messages.extend(grp.get("tool_pair") or [])
            if pending_other_tool_messages:
                messages.extend(pending_other_tool_messages)

        round_stats.append({"i": round_idx + 1, "llm": llm_round_ms, "tool": ", ".join(round_tool_names) if round_tool_names else None, "tool_exec": round_tool_exec_ms, "suff": suff_ms})
        if direct_web_fallback_done:
            break
        if kb_insufficient_and_web_disabled:
            forced_fallback_response = (
                "I want to make sure you get accurate details.\n\n"
                "Right now, I do not have this exact information in the current knowledge base.\n\n"
                "> *Note — Disclaimer: Please verify from official MCC/state counselling websites for the latest confirmed values.*"
            )
            break

    # Follow-up chips must align with the evidence used for the final answer.
    # If we fell back to web (KB insufficient), ground chips in WEB only.
    if used_web_fallback:
        chip_evidence = _combine_retrieval_for_suggestion_chips(None, last_web_retrieval)
    else:
        chip_evidence = _combine_retrieval_for_suggestion_chips(last_kb_retrieval, None)
    policy_evidence = _combine_retrieval_for_suggestion_chips(last_kb_retrieval, last_web_retrieval)
    has_retrieval_evidence = bool((policy_evidence or "").strip())
    if streamed_final_in_loop:
        # Final answer already streamed inside the tool-loop fast path.
        # Do not run another final-generation branch, or content may duplicate.
        pass
    elif forced_fallback_response:
        full_response = localize_output(_apply_response_policy(forced_fallback_response, request.question, allow_factual_addons=has_retrieval_evidence))
        t_out = time.perf_counter()
        _first_out = True
        for token in sse_tokens_preserving_formatting(full_response):
            if _first_out:
                final_ttft_ms = _elapsed_ms(t_out)
                if user_visible_ttft_ms <= 0:
                    user_visible_ttft_ms = _elapsed_ms(request_started_at)
                _first_out = False
            yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"
            if V2_STREAM_TOKEN_DELAY_SEC > 0:
                await asyncio.sleep(V2_STREAM_TOKEN_DELAY_SEC)
        final_stream_ms = _elapsed_ms(t_out)
    elif loop_final_assistant_content:
        # For web-fallback paths, prefer true model streaming so frontend receives
        # gradual token updates (instead of bursty local re-chunking).
        if used_web_fallback and preferred_language == "en":
            t_out = time.perf_counter()
            _first_out = True
            streamed_raw = ""
            async for delta in _stream_chat_completion_text(
                client,
                model="gpt-4o-mini",
                messages=messages,
                temperature=final_temperature,
                max_tokens=final_max_tokens,
            ):
                streamed_raw += delta
                if _first_out:
                    final_ttft_ms = _elapsed_ms(t_out)
                    if user_visible_ttft_ms <= 0:
                        user_visible_ttft_ms = _elapsed_ms(request_started_at)
                    _first_out = False
                yield f"data: {json.dumps({'type': 'token', 'token': delta})}\n\n"
            full_response = _apply_response_policy(
                streamed_raw,
                request.question,
                allow_factual_addons=has_retrieval_evidence,
            )
            if full_response.startswith(streamed_raw):
                tail = full_response[len(streamed_raw):]
                if tail:
                    for token in sse_tokens_preserving_formatting(tail):
                        if user_visible_ttft_ms <= 0:
                            user_visible_ttft_ms = _elapsed_ms(request_started_at)
                        yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"
                        if V2_STREAM_TOKEN_DELAY_SEC > 0:
                            await asyncio.sleep(V2_STREAM_TOKEN_DELAY_SEC)
            else:
                full_response = streamed_raw
            final_stream_ms = _elapsed_ms(t_out)
        else:
            # Optimization: avoid an extra final LLM call when the tool loop already produced
            # an assistant answer (no further tool call needed).
            full_response = localize_output(
                _apply_response_policy(
                    loop_final_assistant_content,
                    request.question,
                    allow_factual_addons=has_retrieval_evidence,
                )
            )
            t_out = time.perf_counter()
            _first_out = True
            for token in sse_tokens_preserving_formatting(full_response):
                if _first_out:
                    final_ttft_ms = _elapsed_ms(t_out)
                    if user_visible_ttft_ms <= 0:
                        user_visible_ttft_ms = _elapsed_ms(request_started_at)
                    _first_out = False
                yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"
                if V2_STREAM_TOKEN_DELAY_SEC > 0:
                    await asyncio.sleep(V2_STREAM_TOKEN_DELAY_SEC)
            final_stream_ms = _elapsed_ms(t_out)
    else:
        t_out = time.perf_counter()
        if preferred_language == "en":
            _log_final_llm_messages_snapshot(messages, label="final_answer")
            _first_out = True
            streamed_raw = ""
            async for delta in _stream_chat_completion_text(
                client,
                model="gpt-4o-mini",
                messages=messages,
                temperature=final_temperature,
                max_tokens=final_max_tokens,
            ):
                streamed_raw += delta
                if _first_out:
                    final_ttft_ms = _elapsed_ms(t_out)
                    if user_visible_ttft_ms <= 0:
                        user_visible_ttft_ms = _elapsed_ms(request_started_at)
                    _first_out = False
                yield f"data: {json.dumps({'type': 'token', 'token': delta})}\n\n"
            full_response = _apply_response_policy(streamed_raw, request.question, allow_factual_addons=has_retrieval_evidence)
            if full_response.startswith(streamed_raw):
                tail = full_response[len(streamed_raw):]
                if tail:
                    for token in sse_tokens_preserving_formatting(tail):
                        if user_visible_ttft_ms <= 0:
                            user_visible_ttft_ms = _elapsed_ms(request_started_at)
                        yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"
                        if V2_STREAM_TOKEN_DELAY_SEC > 0:
                            await asyncio.sleep(V2_STREAM_TOKEN_DELAY_SEC)
            else:
                full_response = streamed_raw
            final_stream_ms = _elapsed_ms(t_out)
        else:
            _log_final_llm_messages_snapshot(messages, label="final_answer")
            final_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=final_temperature,
                max_tokens=final_max_tokens,
            )
            full_response = localize_output(_apply_response_policy(final_response.choices[0].message.content or "", request.question, allow_factual_addons=has_retrieval_evidence))
            _first_out = True
            for token in sse_tokens_preserving_formatting(full_response):
                if _first_out:
                    final_ttft_ms = _elapsed_ms(t_out)
                    if user_visible_ttft_ms <= 0:
                        user_visible_ttft_ms = _elapsed_ms(request_started_at)
                    _first_out = False
                yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"
                if V2_STREAM_TOKEN_DELAY_SEC > 0:
                    await asyncio.sleep(V2_STREAM_TOKEN_DELAY_SEC)
            final_stream_ms = _elapsed_ms(t_out)

    if has_retrieval_evidence:
        replies = await _generate_contextual_suggested_replies(
            request.question,
            full_response,
            med_ctx,
            retrieval_evidence=chip_evidence,
            output_language=preferred_language,
        )
        if replies:
            yield f"data: {json.dumps({'type': 'suggested_replies', 'replies': replies})}\n\n"

    state["used_web_fallback"] = used_web_fallback
    state["last_kb_retrieval"] = last_kb_retrieval
    state["full_response"] = full_response
    state["final_ttft_ms"] = final_ttft_ms
    state["final_stream_ms"] = final_stream_ms
    state["user_visible_ttft_ms"] = user_visible_ttft_ms


async def _v2_try_fast_path_response(
    *,
    request: ChatRequest,
    conversation_id: Optional[int],
    med_ctx: Dict[str, object],
    is_first_visit: bool,
    preferred_language: str,
    localize_output,
    start_time: datetime,
    state: Dict[str, object],
) -> AsyncGenerator[str, None]:
    """
    Stage function for greeting/session-close fast paths.
    Emits events and sets state['handled'].
    """
    state["handled"] = False
    if _is_session_close_intent(request.question):
        med_ctx["stage"] = "closing"
        med_ctx["last_activity_at"] = datetime.utcnow().isoformat()
        close_msg = localize_output(_build_session_close_message(med_ctx))
        yield f"data: {json.dumps({'type': 'suggested_replies', 'replies': _session_close_suggested_replies(preferred_language)})}\n\n"
        for token in sse_tokens_preserving_formatting(close_msg):
            yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"
            if V2_STREAM_TOKEN_DELAY_SEC > 0:
                await asyncio.sleep(V2_STREAM_TOKEN_DELAY_SEC)
        yield f"data: {json.dumps({'type': 'done', 'conversation_id': conversation_id})}\n\n"
        if request.user_id and conversation_id:
            asyncio.create_task(v2_background_update_conversation_context(conversation_id, med_ctx))
        state["handled"] = True
        return

    greeting_word = _detect_greeting_only(request.question)
    if greeting_word:
        registered_user_name = await _get_registered_user_name(request.user_id)
        first_name = _extract_first_name(registered_user_name)
        welcome = (
            _first_visit_welcome_message(first_name, greeting_word)
            if is_first_visit
            else _return_visit_welcome_message(first_name, greeting_word)
        )
        welcome = localize_output(welcome)
        med_ctx["stage"] = "first_visit" if is_first_visit else "returning"
        med_ctx["last_activity_at"] = datetime.utcnow().isoformat()
        yield f"data: {json.dumps({'type': 'suggested_replies', 'replies': _medbuddy_default_replies_for_language(preferred_language)})}\n\n"
        for token in sse_tokens_preserving_formatting(welcome):
            yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"
            if V2_STREAM_TOKEN_DELAY_SEC > 0:
                await asyncio.sleep(V2_STREAM_TOKEN_DELAY_SEC)
        yield f"data: {json.dumps({'type': 'done', 'conversation_id': conversation_id})}\n\n"
        if request.user_id and conversation_id:
            asyncio.create_task(
                v2_background_save_conversation_turn(
                    conversation_id,
                    request.question,
                    welcome,
                    int((datetime.now() - start_time).total_seconds() * 1000),
                )
            )
            asyncio.create_task(v2_background_update_conversation_context(conversation_id, med_ctx))
        state["handled"] = True
        return


@app.post("/chat/v2/stream")
async def chat_v2_stream(request: ChatRequest):
    """
    Unified chat endpoint with tool-based architecture.
    
    Uses a single master prompt that:
    - Handles intent classification internally
    - Decides when to search the knowledge base
    - Asks for clarification when truly needed
    - Generates accurate, concise responses
    
    The LLM has access to a search_knowledge_base tool with optional `state` or multi-state `states` filters.
    """
    from openai import OpenAI as OpenAIClient
    from services.unified_prompt import get_system_prompt, get_tools
    from services.knowledge_tool import execute_tool_call, format_search_results_for_llm
    from services.conversation_memory import (
        ConversationMemory,
        get_or_create_conversation,
    )
    from database.connection import async_session_maker
    
    preferred_language = (
        _normalize_language_code(request.preferred_language)
        if request.preferred_language
        else _detect_user_language_sync(request.question)
    )
    direct_non_english = _v2_direct_non_english_enabled()
    original_user_question = request.question
    if preferred_language != "en" and not direct_non_english:
        request.question = _translate_text_sync(request.question, preferred_language, "en")
        if request.clarified_scope:
            request.clarified_scope = _translate_text_sync(request.clarified_scope, preferred_language, "en")

    log(f"\n{'='*60}")
    log(f"[V2] 📥 NEW QUERY: {request.question[:100]}... | lang={preferred_language}")
    if request.conversation_id:
        log(f"[V2] 💬 CONVERSATION ID: {request.conversation_id}")
    
    async def generate_stream() -> AsyncGenerator[str, None]:
        conversation_memory = None
        conversation_id = request.conversation_id
        med_ctx: Dict[str, object] = _ensure_medbuddy_context(None)
        is_first_visit = False
        start_time = datetime.now()
        t_wall = time.perf_counter()
        conv_ms = 0.0
        faq_ms = 0.0
        settings_ms = 0.0
        round_stats: List[Dict] = []
        final_ttft_ms = 0.0
        final_stream_ms = 0.0
        
        def localize_output(text: str) -> str:
            if preferred_language == "en":
                return text
            if direct_non_english:
                return text
            return _translate_text_sync(text, "en", preferred_language)
        
        try:
            # Emit an immediate frame to encourage early proxy/client flush for SSE.
            yield f"data: {json.dumps({'type': 'stream_started'})}\n\n"
            references_enabled = await is_chat_references_enabled()
            yield f"data: {json.dumps({'type': 'meta', 'chat_references_enabled': references_enabled})}\n\n"
            if not request.question.strip():
                yield f"data: {json.dumps({'error': 'Question cannot be empty'})}\n\n"
                return
            
            # ========== LOAD/CREATE CONVERSATION ==========
            if request.user_id:
                t_conv = time.perf_counter()
                try:
                    async with async_session_maker() as conv_db:
                        if request.conversation_id:
                            conversation = await conv_db.get(Conversation, request.conversation_id)
                            if conversation and conversation.user_id == request.user_id:
                                med_ctx = _ensure_medbuddy_context(conversation.context_data)
                            conversation_memory = ConversationMemory(
                                conversation_id=request.conversation_id,
                                user_id=request.user_id
                            )
                            await conversation_memory.load_from_db(conv_db)
                            msg_count = len(conversation_memory.get_chat_history())
                            if msg_count > 0:
                                log(f"[V2] 🧠 MEMORY LOADED: {msg_count} messages")
                            is_first_visit = msg_count == 0
                        else:
                            conversation = await get_or_create_conversation(
                                db=conv_db,
                                user_id=request.user_id,
                                conversation_id=None
                            )
                            conversation_id = conversation.id
                            conversation_memory = ConversationMemory(
                                conversation_id=conversation_id,
                                user_id=request.user_id
                            )
                            med_ctx = _ensure_medbuddy_context(conversation.context_data)
                            is_first_visit = True
                            log(f"[V2] 📝 NEW CONVERSATION: id={conversation_id}")
                except Exception as mem_err:
                    log(f"[V2] ⚠️ Conversation error: {mem_err}")
                    conversation_memory = None
                finally:
                    conv_ms = _elapsed_ms(t_conv)

            # ========== SESSION CLOSE OR GREETING-WELCOME FAST PATH ==========
            fast_path_state: Dict[str, object] = {}
            async for event in _v2_try_fast_path_response(
                request=request,
                conversation_id=conversation_id,
                med_ctx=med_ctx,
                is_first_visit=is_first_visit,
                preferred_language=preferred_language,
                localize_output=localize_output,
                start_time=start_time,
                state=fast_path_state,
            ):
                yield event
            if fast_path_state.get("handled"):
                return

            # Apply lightweight onboarding updates when user answers guided prompts.
            onboarding = dict(med_ctx.get("onboarding") or {})
            onboarding.update(_extract_onboarding_updates(request.question))
            med_ctx["onboarding"] = onboarding
            med_ctx["last_activity_at"] = datetime.utcnow().isoformat()

            # ========== CUTOFF PATH ==========
            should_try_cutoff = _is_cutoff_query(request.question, med_ctx)
            if should_try_cutoff:
                # Skip LLM router for obvious continuations (pure number, state name, category word)
                if _should_skip_cutoff_router(request.question, med_ctx):
                    cutoff_confirmed = True
                    log(f"[CUTOFF] ⚡ Skipping router — clear continuation: {request.question!r}")
                else:
                    cutoff_confirmed = await _should_route_to_cutoff(request.question, med_ctx)
                if cutoff_confirmed:
                    cutoff_stage_state: Dict[str, object] = {}
                    async for event in _v2_handle_cutoff_stage(
                        request=request,
                        conversation_id=conversation_id,
                        med_ctx=med_ctx,
                        conversation_memory=conversation_memory,
                        preferred_language=preferred_language,
                        localize_output=localize_output,
                        start_time=start_time,
                        state=cutoff_stage_state,
                    ):
                        yield event
                    if cutoff_stage_state.get("handled"):
                        return
            
            # ========== FAQ CHECK (FAST PATH) ==========
            user_state = None
            if request.user_preferences and request.user_preferences.preferred_state:
                user_state = request.user_preferences.preferred_state
            
            faq_lookup_enabled = await is_faq_lookup_enabled()
            FAQ_SCORE_THRESHOLD = float(os.getenv("FAQ_SCORE_THRESHOLD", "0.85"))
            try:
                t_faq = time.perf_counter()
                if not faq_lookup_enabled:
                    log("[V2] ⏭️ FAQ lookup skipped (disabled in admin settings)")
                    faq_matches = []
                    faq_ms = _elapsed_ms(t_faq)
                else:
                    log("[V2] 🔍 Checking FAQs...")
                    try:
                        faq_matches = await search_faq(request.question, state_filter=user_state, top_k=1)
                    finally:
                        faq_ms = _elapsed_ms(t_faq)
                
                if faq_lookup_enabled and faq_matches and faq_matches[0]["score"] >= FAQ_SCORE_THRESHOLD:
                    faq_match = faq_matches[0]
                    log(f"[V2] ✅ FAQ MATCH! Score: {faq_match['score']:.3f}")
                    faq_answer = _apply_response_policy(faq_match["answer"], request.question)
                    faq_answer = localize_output(faq_answer)
                    
                    faq_source = {
                        "file_name": "FAQ Database",
                        "page": None,
                        "text_snippet": faq_match["question"],
                        "state": faq_match.get("state"),
                        "document_type": "faq",
                        "category": faq_match.get("category"),
                        "score": round(faq_match["score"], 3)
                    }
                    yield f"data: {json.dumps({'type': 'sources', 'sources': [faq_source]})}\n\n"
                    
                    faq_replies = await _generate_contextual_suggested_replies(
                        request.question,
                        faq_answer,
                        med_ctx,
                        retrieval_evidence=str(faq_match.get("answer") or ""),
                    )
                    if faq_replies:
                        yield f"data: {json.dumps({'type': 'suggested_replies', 'replies': faq_replies})}\n\n"
                    for token in sse_tokens_preserving_formatting(faq_answer):
                        yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"
                        if V2_STREAM_TOKEN_DELAY_SEC > 0:
                            await asyncio.sleep(V2_STREAM_TOKEN_DELAY_SEC)
                    
                    is_new_conversation = not request.conversation_id and conversation_id
                    faq_response_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)

                    # Done first so the client is not blocked on DB writes
                    yield f"data: {json.dumps({'type': 'done', 'from_faq': True, 'conversation_id': conversation_id})}\n\n"

                    if request.user_id and conversation_id:
                        asyncio.create_task(
                            v2_background_save_conversation_turn(
                                conversation_id,
                                request.question,
                                faq_answer,
                                faq_response_time_ms,
                                sources=[faq_source],
                                was_faq_match=True,
                                faq_confidence=faq_match["score"],
                            )
                        )
                        med_ctx["stage"] = "normal_qa"
                        med_ctx["last_topic"] = _infer_topic_label(request.question)
                        med_ctx["last_state"] = user_state
                        asyncio.create_task(v2_background_update_conversation_context(conversation_id, med_ctx))
                    log("[V2] ✅ Response from FAQ")
                    if _v2_timing_log_enabled():
                        log(
                            f"[V2] ⏱ TIMING (FAQ path) wall≈{_elapsed_ms(t_wall):.0f}ms | "
                            f"conversation_db={conv_ms:.0f} | faq_lookup={faq_ms:.0f}"
                        )
                    
                    if request.user_id and conversation_id:
                        asyncio.create_task(
                            v2_background_generate_conversation_title(
                                conversation_id,
                                request.question,
                                log_label="faq",
                            )
                        )
                    
                    return
                else:
                    if faq_matches:
                        log(f"[V2] ℹ️ FAQ score too low: {faq_matches[0]['score']:.3f}")
            except Exception as faq_err:
                log(f"[V2] ⚠️ FAQ error: {faq_err}")
            
            # ========== BUILD MESSAGES FOR LLM ==========
            client = OpenAIClient(api_key=os.getenv("OPENAI_API_KEY"))
            t_settings = time.perf_counter()
            web_fallback_enabled = await is_web_search_fallback_enabled()
            settings_ms = _elapsed_ms(t_settings)
            available_tools = get_tools()
            registered_user_name = await _get_registered_user_name(request.user_id)
            if not web_fallback_enabled:
                available_tools = [
                    t for t in available_tools
                    if t.get("function", {}).get("name") != "search_web"
                ]
            
            messages = [{"role": "system", "content": get_system_prompt()}]
            if direct_non_english and preferred_language != "en":
                target_label = "Hindi" if preferred_language == "hi" else "Marathi"
                messages.append({
                    "role": "system",
                    "content": (
                        f"Output language override: Respond in {target_label} for all assistant text. "
                        "Keep numbers, proper nouns, and official terms unchanged when needed."
                    )
                })
            messages.append({
                "role": "system",
                "content": (
                    "Runtime tool availability: "
                    + ("`search_web` is ENABLED." if web_fallback_enabled else "`search_web` is DISABLED.")
                    + " For factual queries, use `search_knowledge_base` first. "
                    + "If KB is insufficient and web tool is enabled, call `search_web`."
                )
            })
            if registered_user_name:
                messages.append({
                    "role": "system",
                    "content": (
                        f'The logged-in student name is "{registered_user_name}". '
                        "Personalize naturally by using the student's first name occasionally "
                        "(opening line, reassurance, next-step guidance). Do not overuse the name."
                    )
                })
            
            # Add conversation history
            if conversation_memory:
                # Use structured chat history directly.
                # Parsing newline-formatted history drops multi-line assistant tables
                # (e.g., shortlist rows), which breaks follow-up grounding like
                # "above top 3 colleges".
                for chat_msg in conversation_memory.get_chat_history()[-10:]:
                    role_raw = str(getattr(chat_msg, "role", "")).lower()
                    content = str(getattr(chat_msg, "content", "") or "").strip()
                    if not content:
                        continue
                    if role_raw.endswith("user"):
                        messages.append({"role": "user", "content": content})
                    elif role_raw.endswith("assistant"):
                        messages.append({"role": "assistant", "content": content})
            
            # Add current question
            messages.append({"role": "user", "content": request.question})
            # Turn-level tool-scope guard (instruction-only, no backend hard filtering):
            # prioritize current message scope over older entities unless user explicitly asks comparison/list.
            messages.append({
                "role": "system",
                "content": (
                    "TURN-SPECIFIC RETRIEVAL POLICY (highest priority for this turn):\n"
                    "- Scope retrieval to the LATEST user message.\n"
                    "- If latest message implies a single target entity/college, do not issue retrieval calls for previously discussed entities.\n"
                    "- Only retrieve multiple entities when the latest message explicitly asks comparison, versus, list, or multi-target output.\n"
                    "- If latest message uses relative references (e.g., 'above', 'these', 'those', 'top 5', 'same colleges'), resolve those entities from the most recent assistant shortlist/result in history before tool calls.\n"
                    "- For relative 'top N' asks, resolve entities from the most recent ranked shortlist/table message (not from later derived summaries that may be partial).\n"
                    "- For resolved multi-entity asks, keep actual resolved names in search queries (avoid generic replacements like 'top colleges in India').\n"
                    "- For resolved relative 'top N' multi-entity asks, ensure retrieval covers all resolved entities in this turn (prefer one focused KB call per entity).\n"
                    "- It is valid to run multiple KB calls (including one per resolved entity) and then use web search only for entities still missing after KB.\n"
                    "- You may infer missing topic words from immediate context (e.g., fee structure), "
                    "but must keep entity scope anchored to the latest message."
                )
            })
            messages.append({
                "role": "system",
                "content": (
                    "OUTPUT QUALITY RULES (highest priority for final answer formatting):\n"
                    "- Do not repeat sections, bullets, or paragraphs.\n"
                    "- If you restructure the answer, keep each fact only once.\n"
                    "- Produce only one final draft; never restart the answer with a second Overview/Key section.\n"
                    "- Keep concise, high-signal formatting (single pass).\n"
                    "- Include at most one short disclaimer and at most one closing follow-up question."
                )
            })

            log(f"[V2] 🤖 Calling LLM with {len(messages)} messages...")

            # Clarification-first is now fully prompt-driven by the LLM.
            # Keep this branch disabled to avoid backend hardcoded gating.
            if False:
                log(
                    "[V2] 🔔 Clarification-first: skipped KB/web — question broadens scope "
                    "without state/college anchors"
                )
                messages.append({
                    "role": "system",
                    "content": (
                        "The user's latest message is too broad to search the knowledge base or the web "
                        "(for example 'other colleges' or 'another state' without naming which college(s) or state/UT). "
                        "Do NOT use any tools. Reply in plain text only: ask ONE short, friendly clarification "
                        "so you know the exact college name(s) or state/UT to look up. "
                        "Briefly say that searching without that scope would return unreliable or off-topic results."
                    ),
                })
                t_clar = time.perf_counter()
                clar_resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    temperature=0.25,
                    max_tokens=500,
                )
                clar_llm_ms = _elapsed_ms(t_clar)
                full_response = (clar_resp.choices[0].message.content or "").strip()
                if not full_response:
                    full_response = (
                        "To share accurate fee details, please tell me which state or UT you mean, "
                        "and ideally the exact college name(s)."
                    )
                full_response = _apply_response_policy(
                    full_response,
                    request.question,
                    skip_compare_cta=True,
                )
                full_response = localize_output(full_response)
                t_out = time.perf_counter()
                _first_out = True
                for token in sse_tokens_preserving_formatting(full_response):
                    if _first_out:
                        final_ttft_ms = _elapsed_ms(t_out)
                        _first_out = False
                    yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"
                    if V2_STREAM_TOKEN_DELAY_SEC > 0:
                        await asyncio.sleep(V2_STREAM_TOKEN_DELAY_SEC)
                final_stream_ms = _elapsed_ms(t_out)
                if _v2_timing_log_enabled():
                    log(
                        f"[V2] ⏱ TIMING (clarification-first) wall≈{_elapsed_ms(t_wall):.0f}ms | "
                        f"clarification_llm≈{clar_llm_ms:.0f}ms | stream≈{final_stream_ms:.0f}ms"
                    )
                log(f"[V2] ✅ RESPONSE COMPLETE: {len(full_response)} chars (clarification-first)")
                med_ctx["stage"] = "normal_qa"
                med_ctx["last_topic"] = _infer_topic_label(request.question)
                if request.user_preferences and request.user_preferences.preferred_state:
                    med_ctx["last_state"] = request.user_preferences.preferred_state
                med_ctx["last_activity_at"] = datetime.utcnow().isoformat()
                yield f"data: {json.dumps({'type': 'done', 'conversation_id': conversation_id})}\n\n"
                if request.user_id and conversation_id:
                    asyncio.create_task(
                        v2_background_save_conversation_turn(
                            conversation_id,
                            request.question,
                            full_response,
                            int((datetime.now() - start_time).total_seconds() * 1000),
                        )
                    )
                    asyncio.create_task(v2_background_update_conversation_context(conversation_id, med_ctx))
                if request.user_id and conversation_id:
                    asyncio.create_task(
                        v2_background_generate_conversation_title(
                            conversation_id,
                            request.question,
                            log_label="clarification-first",
                        )
                    )
                return
            
            rag_state: Dict[str, object] = {}
            async for event in _v2_run_rag_pipeline(
                request=request,
                client=client,
                messages=messages,
                available_tools=available_tools,
                web_fallback_enabled=web_fallback_enabled,
                references_enabled=references_enabled,
                med_ctx=med_ctx,
                preferred_language=preferred_language,
                localize_output=localize_output,
                request_started_at=t_wall,
                round_stats=round_stats,
                state=rag_state,
            ):
                yield event

            used_web_fallback = bool(rag_state.get("used_web_fallback"))
            last_kb_retrieval = rag_state.get("last_kb_retrieval")
            full_response = str(rag_state.get("full_response") or "")
            final_ttft_ms = float(rag_state.get("final_ttft_ms") or 0.0)
            final_stream_ms = float(rag_state.get("final_stream_ms") or 0.0)
            user_visible_ttft_ms = float(rag_state.get("user_visible_ttft_ms") or 0.0)

            if used_web_fallback:
                yield f"data: {json.dumps({'type': 'meta', 'web_fallback_used': True, 'source_origin': 'web'})}\n\n"
                log("[V2] 🌐 Final response generated with web fallback context")
            else:
                source_origin = "kb" if last_kb_retrieval else "none"
                yield f"data: {json.dumps({'type': 'meta', 'source_origin': source_origin})}\n\n"
                log("[V2] 🧠 Final response generated with RAG tool context")
                
            log(f"[V2] ✅ RESPONSE COMPLETE: {len(full_response)} chars")
            med_ctx["stage"] = "normal_qa"
            med_ctx["last_topic"] = _infer_topic_label(request.question)
            if request.user_preferences and request.user_preferences.preferred_state:
                med_ctx["last_state"] = request.user_preferences.preferred_state
            med_ctx["last_activity_at"] = datetime.utcnow().isoformat()
            
            is_new_conversation = not request.conversation_id and conversation_id
            response_time = int((datetime.now() - start_time).total_seconds() * 1000)

            # Done before DB persist so the client is not blocked on DB writes
            yield f"data: {json.dumps({'type': 'done', 'conversation_id': conversation_id})}\n\n"

            if request.user_id and conversation_id:
                asyncio.create_task(
                    v2_background_save_conversation_turn(
                        conversation_id,
                        request.question,
                        full_response,
                        response_time,
                    )
                )
                asyncio.create_task(v2_background_update_conversation_context(conversation_id, med_ctx))

            if _v2_timing_log_enabled():
                wall_ms = _elapsed_ms(t_wall)
                parts = [
                    f"wall_total≈{wall_ms:.0f}ms",
                    f"conversation_db={conv_ms:.0f}",
                    f"faq_lookup={faq_ms:.0f}",
                    f"settings_read={settings_ms:.0f}",
                ]
                for r in round_stats:
                    tool = r.get("tool") or "—"
                    parts.append(
                        f"round{r['i']}:llm={r['llm']:.0f}"
                        f"|tool={tool}"
                        f"|exec={r['tool_exec']:.0f}"
                        f"|suff={r['suff']:.0f}"
                    )
                parts.append(f"answer_ttft={final_ttft_ms:.0f}ms")
                parts.append(f"first_token_to_user={user_visible_ttft_ms:.0f}ms")
                parts.append(f"answer_stream_total={final_stream_ms:.0f}ms")
                parts.append("save_db=deferred(background)")
                log(
                    "[V2] ⏱ TIMING SUMMARY (V2_TIMING_LOG=false to hide) → "
                    + " ".join(parts)
                )
            log(f"{'='*60}")
            
            if request.user_id and conversation_id:
                asyncio.create_task(
                    v2_background_generate_conversation_title(
                        conversation_id,
                        request.question,
                        log_label="rag",
                    )
                )
            
        except Exception as e:
            import traceback
            log(f"[V2] ❌ ERROR: {e}")
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Content-Type": "text/event-stream; charset=utf-8",
            "Connection": "keep-alive",
            "Pragma": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)