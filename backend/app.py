"""
RAG Chatbot - FastAPI Backend (OpenAI + Neon pgvector)
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


MEDBUDDY_CAPS = {
    "cutoff": True,
    "mbbs_abroad": os.getenv("MEDBUDDY_ENABLE_MBBS_ABROAD", "false").lower() in ("1", "true", "yes"),
}


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip().lower())


def _contains_any(text: str, phrases: List[str]) -> bool:
    return any(p in text for p in phrases)


def _is_greeting_only(question: str) -> bool:
    q = _normalize_text(question)
    greeting_words = {"hi", "hello", "hey", "good morning", "good afternoon", "good evening", "hii"}
    return q in greeting_words


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

    if _extract_states_from_text(question, cutoff_db_states=None, use_llm_fallback=False):
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


def _first_visit_welcome_message(first_name: Optional[str] = None) -> str:
    greeting = (
        f"Hi {first_name}, hope you are doing well.\n\n"
        if first_name else
        "Hi, hope you are doing well.\n\n"
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


def _return_visit_welcome_message(first_name: Optional[str] = None) -> str:
    greeting = (
        f"Hi {first_name}, hope you are doing well.\n\n"
        if first_name else
        "Hi, hope you are doing well.\n\n"
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


def _llm_should_route_cutoff_sql(question: str, med_ctx: Dict[str, object]) -> Optional[bool]:
    """
    LLM-first routing decision for cutoff SQL path.
    Returns True/False when model gives a valid decision, otherwise None.
    """
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        cutoff_ctx = dict(med_ctx.get("cutoff") or {})
        prompt = f"""You are a routing classifier for a NEET UG assistant.

Decide if this user message should be routed to SQL cutoff prediction table (`neet_ug_2025_cutoffs`) instead of document RAG.

Route **TRUE** to SQL cutoff when the user wants **college finding, shortlist, prediction, cutoffs tied to their chances**,
or follow-ups that supply rank/score, category, or state — **even if they have not given a full profile yet**
(the server will ask for missing rank/score, home state, target state(s), and category **before** running the query).

Route **TRUE** for vague asks like "help me find a good college", "which college can I get", "shortlist", "cutoff for my rank".

Route **FALSE** for general counselling process, exam-only guidance, fee structure, documents, dates, reservation policy
**without** college prediction, or pure definitions.

College comparison rule (important):
- If user asks to compare two colleges (fees, facilities, courses, brochure details, documents, eligibility, etc.)
  and does NOT explicitly ask prediction/chances/cutoff-for-rank, route **FALSE** (RAG path, not SQL cutoff).
- Do NOT route to SQL cutoff just because two colleges are named.
- Route **TRUE** for comparisons only when user explicitly asks cutoff/chances/prediction language
  (for example: "compare cutoffs", "which one can I get at my rank", "chance comparison by rank/score").
- If the assistant previously asked a comparison clarification like "which college to compare with X?"
  and user now replies with just a college name (e.g., "AIIMS Delhi"), route **FALSE** and continue comparison in RAG.
- A bare college-name follow-up after comparison clarification is NOT a cutoff profile input.

Important disambiguation:
- The phrase "state counselling" alone does **not** mean cutoff prediction.
- Route **FALSE** for asks like "state counselling details", "counselling process", "how counselling works", "registration steps", unless the user clearly asks prediction/chances/shortlist.
- Route **TRUE** only when intent is explicitly about "which college can I get", "college prediction", "cutoff for my rank/score", "shortlist colleges".

Conversation last topic: {med_ctx.get("last_topic") or "unknown"}
Cutoff context already present: {bool(cutoff_ctx)}
Cutoff context keys: {list(cutoff_ctx.keys())}
User message: "{question}"

Critical continuation rule:
- If recent conversation is already in cutoff shortlisting flow and current user message is a short follow-up
  (state name, category, rank/score value, yes/no, refinement detail), keep routing to SQL cutoff.
- If current message itself asks counselling process/details (not prediction), route away from SQL cutoff even if previous turn was cutoff.
- Only continue SQL cutoff when the follow-up is clearly profile/refinement data for prediction.

Respond ONLY JSON:
{{
  "route_to_cutoff_sql": true or false,
  "reason": "short reason"
}}
"""
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=80,
        )
        text = (resp.choices[0].message.content or "").strip()
        data = _extract_json_object(text)
        if not data or "route_to_cutoff_sql" not in data:
            log(f"[V2] ⚠️ Cutoff route classifier parse failed: {text[:200]}")
            return None
        decision = bool(data.get("route_to_cutoff_sql"))
        log(f"[V2] 🧭 Cutoff route classifier: {decision} | reason={data.get('reason', '')}")
        return decision
    except Exception as e:
        log(f"[V2] ⚠️ Cutoff route classifier error: {e}")
        return None


async def _llm_should_route_cutoff_sql_async(question: str, med_ctx: Dict[str, object]) -> Optional[bool]:
    return await asyncio.to_thread(_llm_should_route_cutoff_sql, question, med_ctx)


def _should_continue_cutoff_from_context(question: str, cutoff_ctx: Dict[str, object]) -> bool:
    if not cutoff_ctx:
        return False
    if cutoff_ctx.get("awaiting_refinement_choice") or cutoff_ctx.get("awaiting_refinement_details"):
        return True
    if _missing_cutoff_fields(cutoff_ctx):
        return True

    q = _normalize_text(question)
    metric_type, metric_value = _extract_metric_from_text(question)
    short_ack = q in {"yes", "yes please", "yup", "sure", "ok", "okay"}
    # Very short inputs in an active cutoff thread are usually continuation tokens.
    if short_ack or (metric_type is not None and metric_value is not None):
        return True
    return False


def _is_recent_iso_timestamp(ts: Optional[str], within_minutes: int) -> bool:
    if not ts:
        return False
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        if dt.tzinfo is not None:
            now_dt = datetime.now(dt.tzinfo)
        else:
            now_dt = datetime.utcnow()
        return (now_dt - dt) <= timedelta(minutes=within_minutes)
    except Exception:
        return False


def _expire_stale_cutoff_run_state(cutoff_ctx: Dict[str, object], freshness_minutes: int = 30) -> None:
    """
    Keep profile preferences (home_state, category, sub_category, preferences_set),
    but expire run-specific fields after inactivity.
    """
    last_turn_at = str(cutoff_ctx.get("last_turn_at") or "")
    if _is_recent_iso_timestamp(last_turn_at, freshness_minutes):
        return
    for key in [
        "metric_type",
        "metric_value",
        "target_states",
        "target_states_confirmed",
        "awaiting_refinement_choice",
        "awaiting_refinement_details",
        "college_type_filter",
        "course_filter",
        "quota_keywords",
        "seat_type_keywords",
        "last_result_count",
    ]:
        cutoff_ctx.pop(key, None)


def _looks_like_cutoff_profile_payload(question: str) -> bool:
    q = _normalize_text(question)
    metric_type, metric_value = _extract_metric_from_text(question)
    has_metric = metric_type is not None and metric_value is not None
    has_state_hint = any(k in q for k in ["preferred state", "preffered state", "home state", "domicile"])
    has_state = len(_extract_states_from_text(question)) > 0 or has_state_hint
    return has_metric and has_state


def _extract_metric_from_text(question: str) -> Tuple[Optional[str], Optional[int]]:
    q = _normalize_text(question)
    num_match = re.search(r"\b(\d{2,7})\b", q)
    if not num_match:
        return None, None
    value = int(num_match.group(1))

    if any(k in q for k in ["score", "marks", "mark"]):
        return "score", value
    if any(k in q for k in ["air", "rank", "all india rank"]):
        return "rank", value

    # Fallback inference by range
    if value <= 720:
        return "score", value
    return "rank", value


def _normalize_state_key(raw: str) -> str:
    s = _normalize_text(raw)
    s = s.replace("&", " and ")
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def _map_canonical_to_cutoff_db_state(
    canonical_state: str,
    cutoff_db_states: Optional[List[str]] = None,
) -> str:
    if not cutoff_db_states:
        return canonical_state
    target_key = _normalize_state_key(canonical_state)
    for db_state in cutoff_db_states:
        if _normalize_state_key(db_state) == target_key:
            return db_state
    return canonical_state


def _llm_map_states_to_db_values(question: str, cutoff_db_states: List[str]) -> List[str]:
    """
    LLM fallback for fuzzy state mentions (e.g., 'up' -> 'Uttar Pradesh'),
    constrained to DB-provided distinct state names.
    """
    if not question or not cutoff_db_states:
        return []
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You map user-entered Indian state mentions to canonical values from an allowed list.\n"
                        "Return ONLY JSON with key `states` as an array of exact strings from allowed_states.\n"
                        "Rules:\n"
                        "- Use only values that exist in allowed_states.\n"
                        "- Resolve abbreviations like UP, MP, AP, J&K when clear.\n"
                        "- Do not invent states.\n"
                        "- If no state is present, return {\"states\": []}."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "query": question,
                            "allowed_states": cutoff_db_states,
                        }
                    ),
                },
            ],
            temperature=0,
            max_tokens=180,
        )
        raw = (response.choices[0].message.content or "").strip()
        parsed = json.loads(raw)
        states = parsed.get("states", [])
        if not isinstance(states, list):
            return []
        allowed = {s for s in cutoff_db_states}
        deduped: List[str] = []
        seen = set()
        for st in states:
            if isinstance(st, str) and st in allowed and st not in seen:
                deduped.append(st)
                seen.add(st)
        return deduped
    except Exception as e:
        log(f"[V2] ⚠️ LLM state-mapping fallback failed: {e}")
        return []


def _extract_states_from_text(
    question: str,
    cutoff_db_states: Optional[List[str]] = None,
    use_llm_fallback: bool = False,
) -> List[str]:
    if cutoff_db_states:
        return _llm_map_states_to_db_values(question, cutoff_db_states)
    # LLM-first policy: without DB-provided state values, do not run regex extraction.
    return []


def _extract_home_state(question: str) -> Optional[str]:
    q = _normalize_text(question)
    patterns = [
        r"(?:i am from|i'm from|my home state is|my state is|i belong to|from)\s+([a-z\s&]+)",
        r"([a-z\s&]+)\s+(?:home state|domicile)\b",
    ]
    for pattern in patterns:
        m = re.search(pattern, q)
        if not m:
            continue
        phrase = m.group(1).strip()
        for alias, canonical in STATES.items():
            if canonical == "All-India":
                continue
            if re.search(rf"\b{re.escape(alias)}\b", phrase):
                return canonical
    return None


def _canonicalize_state_name(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    s = _normalize_text(str(raw))
    if s in STATES:
        val = STATES[s]
        return None if val == "All-India" else val
    for canonical in sorted(set(STATES.values()), key=len, reverse=True):
        if canonical.lower() == s:
            return None if canonical == "All-India" else canonical
    # Fallback for cutoff DB exact states (handles values like "Jammu & Kashmir", "Tamilnadu").
    raw_text = str(raw).strip()
    if raw_text:
        raw_key = _normalize_state_key(raw_text)
        for db_state in CUTOFF_SQL_STATES:
            if _normalize_state_key(db_state) == raw_key:
                return db_state
    return None


# _is_friend_or_general_profile_mode removed — profile mode is now detected
# purely by the LLM inside _llm_extract_cutoff_profile_turn via profile_mode field.


async def _get_registered_home_state(user_id: Optional[int]) -> Optional[str]:
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


def _normalize_cutoff_category(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    raw = str(value).strip().upper().replace("-", "").replace(" ", "")
    mapping = {
        "GENERAL": "GENERAL",
        "GEN": "GENERAL",
        "UR": "GENERAL",
        "OBC": "OBC",
        "SC": "SC",
        "ST": "ST",
        "EWS": "EWS",
        "PWD": "PWD",
        "PWBD": "PWD",
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
            cutoff_profile = dict(profile_data.get("cutoff_profile") or {})
            home_state = _canonicalize_state_name(cutoff_profile.get("home_state"))
            category = _normalize_cutoff_category(cutoff_profile.get("category"))
            sub_category_raw = str(cutoff_profile.get("sub_category") or "").strip()
            preferences_set = bool(home_state and category)
            out: Dict[str, object] = {}
            if home_state:
                out["home_state"] = home_state
            if category:
                out["category"] = category
            if sub_category_raw:
                out["sub_category"] = sub_category_raw.upper()
            out["preferences_set"] = preferences_set
            return out
    except Exception as e:
        log(f"[V2] ⚠️ Could not load user cutoff profile: {e}")
        return {}


async def _save_user_cutoff_profile(user_id: Optional[int], cutoff_ctx: Dict[str, object]) -> None:
    if not user_id:
        return
    try:
        from database.connection import async_session_maker

        async with async_session_maker() as db:
            user = await db.get(User, user_id)
            if not user:
                return
            profile_data = dict(getattr(user, "profile_data", {}) or {})
            cutoff_profile = dict(profile_data.get("cutoff_profile") or {})

            home_state = _canonicalize_state_name(cutoff_ctx.get("home_state"))
            category = _normalize_cutoff_category(cutoff_ctx.get("category"))
            sub_category = str(cutoff_ctx.get("sub_category") or "").strip().upper() or None

            if home_state:
                cutoff_profile["home_state"] = home_state
            if category:
                cutoff_profile["category"] = category
            if sub_category:
                cutoff_profile["sub_category"] = sub_category
            cutoff_profile["preferences_set"] = bool(home_state and category)
            cutoff_profile["updated_at"] = datetime.utcnow().isoformat()

            profile_data["cutoff_profile"] = cutoff_profile
            user.profile_data = profile_data
            await db.commit()
    except Exception as e:
        log(f"[V2] ⚠️ Could not save user cutoff profile: {e}")


async def _get_cutoff_category_options(state: Optional[str]) -> List[str]:
    if not state:
        return []
    try:
        from database.connection import async_session_maker
        from sqlalchemy import text

        async with async_session_maker() as db:
            result = await db.execute(
                text(
                    """
                    SELECT DISTINCT TRIM(UPPER(category)) AS category
                    FROM neet_ug_2025_cutoffs
                    WHERE TRIM(state) ILIKE :state_like
                      AND category IS NOT NULL
                      AND TRIM(category) <> ''
                    ORDER BY 1
                    """
                ),
                {"state_like": state},
            )
            return [str(row[0]).strip().upper() for row in result.fetchall() if row[0]]
    except Exception as e:
        log(f"[V2] ⚠️ Could not load cutoff category options: {e}")
        return []


async def _get_cutoff_subcategory_options(state: Optional[str], category: Optional[str]) -> List[str]:
    if not state or not category:
        return []
    try:
        from database.connection import async_session_maker
        from sqlalchemy import text

        async with async_session_maker() as db:
            result = await db.execute(
                text(
                    """
                    SELECT DISTINCT TRIM(UPPER(sub_category)) AS sub_category
                    FROM neet_ug_2025_cutoffs
                    WHERE TRIM(state) ILIKE :state_like
                      AND category ILIKE :category_like
                      AND sub_category IS NOT NULL
                      AND TRIM(sub_category) <> ''
                    ORDER BY 1
                    """
                ),
                {"state_like": state, "category_like": f"%{category}%"},
            )
            return [str(row[0]).strip().upper() for row in result.fetchall() if row[0]]
    except Exception as e:
        log(f"[V2] ⚠️ Could not load cutoff sub-category options: {e}")
        return []


def _missing_cutoff_fields(cutoff_ctx: Dict[str, object]) -> List[str]:
    """
    Two-phase field collection — mode-aware, no hardcoded detection:

    Phase 1 — profile fields (home_state + category):
      - 'self' mode: collected once at first cutoff query, saved to DB as preferences.
        preferences_set=True means phase 1 is already done; skip it.
      - 'friend' or 'general' mode: always collected fresh every query,
        never loaded from saved profile, never saved to profile.

    Phase 2 — per-query fields (metric + target_states):
      Always required regardless of mode. Nothing else ever asked proactively —
      course, college_type, seat_type, quota are applied ONLY when user mentions them.
    """
    missing: List[str] = []
    profile_mode = str(cutoff_ctx.get("profile_mode") or "self")

    if profile_mode == "self":
        # Phase 1 only needed once — skip entirely if preferences already saved
        if not cutoff_ctx.get("preferences_set"):
            if not cutoff_ctx.get("home_state"):
                missing.append("home_state")
            if not cutoff_ctx.get("category"):
                missing.append("category")
            if missing:
                return missing
    else:
        # friend/general: always need these fresh in current context
        if not cutoff_ctx.get("home_state"):
            missing.append("home_state")
        if not cutoff_ctx.get("category"):
            missing.append("category")
        if missing:
            return missing

    # Phase 2: per-query essentials — same for all modes
    if not cutoff_ctx.get("metric_type") or not cutoff_ctx.get("metric_value"):
        missing.append("metric")
    if not cutoff_ctx.get("target_states") or not bool(cutoff_ctx.get("target_states_confirmed")):
        missing.append("target_states")
    return missing


def _build_cutoff_followup_prompt(
    missing_fields: List[str],
    cutoff_ctx: Optional[Dict[str, object]] = None,
    category_options: Optional[List[str]] = None,
    sub_category_options: Optional[List[str]] = None,
) -> str:
    ctx = dict(cutoff_ctx or {})
    profile_mode = str(ctx.get("profile_mode") or "self")

    # Subject phrasing driven entirely by profile_mode set by LLM
    if profile_mode == "friend":
        subject_home = "your friend's home state (domicile)"
        subject_category = "your friend's NEET category"
        subject_metric = "your friend's NEET AIR rank or NEET score"
        subject_states = "which state(s) your friend wants to explore"
        already_have_home = f"Friend's home state: **{ctx.get('home_state')}**."
        one_time_note = ""
    elif profile_mode == "general":
        subject_home = "the candidate's home state (domicile)"
        subject_category = "the candidate's NEET category"
        subject_metric = "the candidate's NEET AIR rank or NEET score"
        subject_states = "which state(s) to explore"
        already_have_home = f"Home state: **{ctx.get('home_state')}**."
        one_time_note = ""
    else:
        subject_home = "your home state (domicile)"
        subject_category = "your NEET category"
        subject_metric = "your NEET AIR rank or NEET score"
        subject_states = "which state(s) you want to explore"
        already_have_home = f"Home state: **{ctx.get('home_state')}**."
        one_time_note = "\n\nThis is a one-time setup — I'll remember it for future searches."

    if "metric" in missing_fields and "target_states" in missing_fields:
        return (
            f"To find matching colleges, please share:\n"
            f"1. {subject_metric.capitalize()}\n"
            f"2. And {subject_states}\n\n"
            f"Example: *Rank 45000, looking in Bihar and UP*"
        )

    if "metric" in missing_fields:
        return (
            f"Please share {subject_metric} "
            f"so I can match colleges from last year's cutoff data."
        )

    if "home_state" in missing_fields:
        return (
            f"Please share {subject_home}. "
            f"I need this to apply domicile vs non-domicile cutoff rows correctly.{one_time_note}"
        )

    if "target_states" in missing_fields:
        hs = str(ctx.get("home_state") or "").strip()
        prefix = f"{already_have_home}\n\n" if hs else ""
        home_hint = (
            f"You can pick the home state (**{hs}**) or any other state(s).\n\n"
            if hs else ""
        )
        return (
            f"{prefix}Which state(s) should I look for colleges in?\n\n"
            f"{home_hint}"
            f"You can name one or multiple states."
        )

    if "category" in missing_fields:
        prefix = f"{already_have_home}\n\n" if ctx.get("home_state") else ""
        if category_options:
            opts = " / ".join(category_options[:10])
            return (
                f"{prefix}Which is {subject_category}?\n\n"
                f"Available options: {opts}"
            )
        return (
            f"{prefix}Which is {subject_category}?\n\n"
            "Options: General / OBC / SC / ST / EWS / PwD"
        )

    if "sub_category" in missing_fields:
        if sub_category_options:
            opts = " / ".join(sub_category_options[:12])
            return (
                f"Does any sub-category apply for **{ctx.get('category') or 'selected category'}**?\n\n"
                f"Available options in cutoff data: {opts}\n\n"
                "If none apply, just say **none** or **skip**."
            )
        return (
            "Does any sub-category apply? If not, just say **none** or **skip**."
        )

    return "Please share a bit more detail so I can run cutoff analysis."


def _is_affirmative_reply(question: str) -> bool:
    q = _normalize_text(question)
    return q in {"yes", "yes please", "yeah", "yup", "sure", "okay", "ok", "go ahead"}


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
    """
    Drop follow-up chips whose topic is not substantively present in retrieval text.
    Fee tables often list amounts only — do not offer payment/scholarship/refund chips unless evidence mentions them.
    """
    if not evidence or not replies:
        return replies
    ev = _normalize_text(evidence)
    out: List[str] = []
    for r in replies:
        q = _normalize_text(r)
        skip = False
        # Chips that broaden to unnamed colleges/states are not grounded in a single retrieval.
        if any(
            phrase in q
            for phrase in (
                "other colleges",
                "other college",
                "another college",
                "another state",
                "different college",
                "different colleges",
                "more colleges",
                "more college",
            )
        ):
            skip = True
        if not skip and "placement" in q and "placement" not in ev:
            skip = True
        if not skip and "internship" in q and "internship" not in ev:
            skip = True
        # Payment *methods* / how to pay (amounts named "fees" are not enough)
        asks_payment_modality = (
            ("payment" in q and ("option" in q or "method" in q or "mode" in q))
            or "how to pay" in q
            or "pay online" in q
            or "pay fees" in q
        )
        if asks_payment_modality:
            if not any(
                k in ev
                for k in (
                    "demand draft",
                    "neft",
                    "rtgs",
                    "upi",
                    "online payment",
                    "bank",
                    "cheque",
                    "cash payment",
                    "installment",
                    "emi",
                    "mode of payment",
                    "payment mode",
                    "payable at",
                    "pay through",
                )
            ):
                skip = True
        if not skip and any(k in q for k in ("scholarship", "fee waiver", "financial aid")):
            if not any(k in ev for k in ("scholarship", "waiver", "freeship", "fee concession", "economic weaker")):
                skip = True
        if not skip and "refund" in q:
            if "refund" not in ev and "forfeit" not in ev:
                skip = True
        if not skip and "stipend" in q:
            if "stipend" not in ev:
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
    """
    Generate optional quick-reply chips.
    Uses deterministic shortcuts for known guided prompts, then LLM fallback.
    When retrieval_evidence is set, chips must stay on-topic and be answerable from that evidence.
    """
    response_text = str(assistant_response or "")
    question_text = str(user_question or "")
    response_norm = _normalize_text(response_text)
    evidence_text = (retrieval_evidence or "").strip()

    # If we are explicitly collecting rank/score/category/home state, avoid chips.
    collection_signals = [
        "share either your neet score",
        "share your neet category",
        "tell me your home state",
        "please tell me your home state",
        "please share your neet rank",
        "neet air rank or",
        "neet score/marks",
        "home state (domicile",
        "candidate's home state",
        "state(s) or ut(s) should i pull",
        "which **neet category**",
        "which state(s) should i pull",
    ]
    if any(sig in response_norm for sig in collection_signals):
        return []

    # Keep initial starter choices stable.
    if _is_greeting_only(question_text):
        return _medbuddy_default_replies_for_language(output_language)

    lang_rules = _chip_generation_language_rules(output_language)
    max_words_per_chip = 6 if _normalize_language_code(output_language) == "en" else 14

    if evidence_text:
        grounding_rules = (
            "CRITICAL — Answerability from evidence only:\n"
            "- Read RETRIEVED_EVIDENCE carefully. Propose a chip ONLY if a correct answer could be written using "
            "**only** sentences/facts from RETRIEVED_EVIDENCE (plus trivial math like summing line items already shown).\n"
            "- NEVER propose chips that broaden to unnamed targets: no 'other colleges', 'another college', "
            "'another state', 'more colleges', or similar unless RETRIEVED_EVIDENCE already lists those extra colleges "
            "with comparable facts for the same question.\n"
            "- NEVER propose placement, internship, or non-admission topics unless they appear in RETRIEVED_EVIDENCE.\n"
            "- If the evidence is mainly a **fee table** (amounts, components, year-wise totals) and does **not** discuss "
            "how fees are **paid** (bank, DD, UPI, instalments, portal), do **not** ask about payment methods, "
            "payment options, or online payment — the user would get 'not in documents'.\n"
            "- Do **not** ask about scholarships, refund policy, admission steps, hostel **facilities** (beyond a fee line), "
            "or curriculum unless those topics are **explicitly** covered in RETRIEVED_EVIDENCE.\n"
            "- Good fee-only chips (examples): ask about a **named component** in the table, year-to-year comparison, "
            "security deposit, or total — only if that detail appears in the evidence.\n"
            "- Stay on the same named college(s)/state as in the evidence; do not invent new institutes.\n"
            "- If nothing sensible remains answerable from the evidence, return {\"replies\": []}.\n"
        )
    else:
        grounding_rules = (
            "No retrieval blob provided — use only the user question and assistant reply:\n"
            "- Suggest follow-ups only if the assistant answer actually contains enough detail to support a further answer; "
            "do not suggest payment methods, scholarships, or refunds unless the assistant explicitly gave information about them.\n"
            "- Stay in the SAME topic lane (fee → fee-related; cutoff → cutoff-related).\n"
            "- Prefer 0–3 chips; return empty if the reply was already exhaustive or purely disclaimer.\n"
        )

    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        topic = str((med_ctx or {}).get("last_topic") or "")
        user_payload: Dict[str, object] = {
            "current_user_question": question_text,
            "assistant_response": response_text[:2000],
            "last_topic": topic,
            "ui_language": _normalize_language_code(output_language),
        }
        if evidence_text:
            user_payload["RETRIEVED_EVIDENCE"] = evidence_text[:14000]

        llm_resp = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Generate concise suggested reply chips for a NEET counselling chat UI.\n"
                        "Return JSON only: {\"replies\": [\"...\", \"...\", ...]}.\n\n"
                        "## CHIP GENERATION STRATEGY\n"
                        "Generate at most 3 chips, each serving a different purpose:\n\n"
                        "1. DEPTH chip (always try to include):\n"
                        "   - Digs deeper into the SAME topic and SAME college/entity the bot just answered about.\n"
                        "   - Example: Bot answered hostel availability at AIIMS Bhopal -> "
                        "'Want to know the hostel fee, room type, and mess charges at AIIMS Bhopal?'\n"
                        "   - Example: Bot answered MBBS fee at GMC Kathua -> "
                        "'What is the total fee including hostel and mess charges at GMC Kathua?'\n\n"
                        "2. BREADTH chip (include when relevant):\n"
                        "   - Explores a RELATED but DIFFERENT topic for the SAME college/entity.\n"
                        "   - Example: Bot answered hostel at AIIMS Bhopal -> "
                        "'Should I also show the NEET cutoff rank needed to get into AIIMS Bhopal?'\n"
                        "   - Example: Bot answered fee at GMC Kathua -> "
                        "'What documents are needed for admission at GMC Kathua?'\n\n"
                        "3. COMPARISON chip (include when relevant):\n"
                        "   - Same topic but DIFFERENT colleges/entities for comparison.\n"
                        "   - Example: Bot answered hostel at AIIMS Bhopal -> "
                        "'Want hostel details at other AIIMS campuses like Nagpur, Raipur, or Jodhpur?'\n"
                        "   - Example: Bot answered fee at GMC Kathua -> "
                        "'How does GMC Kathua fee compare with other government colleges in J&K?'\n\n"
                        "## RULES\n"
                        "- Return 0 to 3 chips only - one per pattern above.\n"
                        "- Each chip must be a short natural question the user might actually send next.\n"
                        "- Always ground chips in the EXACT college/entity/topic from the bot's last answer.\n"
                        "- Never generate generic chips like 'What is the total fee?' without naming the college.\n"
                        "- Never repeat the same information the bot already answered.\n"
                        "- If the bot asked the user for input (rank, state, category), return empty list [].\n"
                        "- If the answer was about cutoff shortlist results, chips should be about refining "
                        "(state, college type, course) not repeating the same search.\n"
                        + lang_rules
                        + "- Do not include 'Thanks' or acknowledgement chips.\n"
                        "- Do not repeat the same meaning across chips.\n"
                        + grounding_rules
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(user_payload),
                },
            ],
            temperature=0.2,
            max_tokens=220,
        )
        raw = (llm_resp.choices[0].message.content or "").strip()
        parsed = json.loads(raw)
        replies = parsed.get("replies", [])
        if not isinstance(replies, list):
            return []
        cleaned: List[str] = []
        seen = set()
        for r in replies:
            text = str(r or "").strip()
            key = text.casefold()
            if not text or key in seen:
                continue
            if len(text.split()) > max_words_per_chip:
                continue
            cleaned.append(text)
            seen.add(key)
            if len(cleaned) >= 5:
                break
        if evidence_text and cleaned:
            filtered = _filter_chips_not_supported_by_evidence(cleaned, evidence_text)
            # If strict grounding filter becomes too aggressive and removes all chips,
            # keep up to 2 model-generated chips as a graceful fallback for UX continuity.
            cleaned = filtered if filtered else cleaned[:2]
        if not cleaned:
            cleaned = _fallback_contextual_chips(question_text, response_text, output_language)
        return cleaned
    except Exception as e:
        log(f"[V2] ⚠️ Suggested replies generation failed: {e}")
        return []


async def _llm_explain_cutoff_results(
    *,
    user_question: str,
    metric_type: str,
    metric_value: int,
    home_state: str,
    target_states: List[str],
    category: str,
    sub_category: Optional[str],
    rows: List[Dict[str, object]],
) -> str:
    """Generate a short round-2 explanation for cutoff SQL results."""
    if not rows:
        return ""
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        sample_rows = []
        for row in rows[:8]:
            sample_rows.append(
                {
                    "institution": row.get("institution_name") or row.get("college_name"),
                    "state": row.get("state"),
                    "course": row.get("course"),
                    "quota": row.get("quota"),
                    "domicile": row.get("domicile"),
                    "air_rank": row.get("air_rank"),
                    "score": row.get("score"),
                    "round": row.get("round"),
                }
            )

        payload = {
            "question": user_question,
            "profile": {
                "metric_type": metric_type,
                "metric_value": metric_value,
                "home_state": home_state,
                "target_states": target_states,
                "category": category,
                "sub_category": sub_category,
            },
            "matched_count": len(rows),
            "sample_rows": sample_rows,
        }
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You summarize NEET cutoff matches for students.\n"
                        "Return concise markdown only with heading + bullets.\n"
                        "Do not use markdown tables.\n"
                        "Use only provided rows; do not invent facts.\n"
                        "Include 4-8 bullets with practical pattern insights and one final next-step question.\n"
                    ),
                },
                {"role": "user", "content": json.dumps(payload)},
            ],
            temperature=0.2,
            max_tokens=320,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        log(f"[V2] ⚠️ Cutoff round-2 explanation skipped: {e}")
        return ""


def _fallback_cutoff_quick_interpretation(
    *,
    rows: List[Dict[str, object]],
    metric_type: str,
    metric_value: int,
    target_states: List[str],
) -> str:
    """Deterministic backup text when cutoff round-2 LLM is unavailable."""
    if not rows:
        return ""
    top = rows[0]
    metric_label = "AIR rank" if metric_type == "rank" else "score"
    top_metric_value = top.get("air_rank") if metric_type == "rank" else top.get("score")
    top_metric_text = str(top_metric_value) if top_metric_value is not None else "N/A"
    top_inst = str(top.get("institution_name") or top.get("college_name") or "Top matched college")
    states_text = ", ".join(target_states) if target_states else "selected states"
    return (
        f"- Closest match in **{states_text}** is **{top_inst}** "
        f"(cutoff {metric_label}: **{top_metric_text}**).\n"
        f"- Your input {metric_label} is **{metric_value}**; options shown are the nearest available rows.\n"
        "- You can refine further by course, quota, domicile mode, or adding/removing states."
    )


def _extract_cutoff_refinements(
    question: str,
    cutoff_db_states: Optional[List[str]] = None,
) -> Dict[str, object]:
    updates: Dict[str, object] = {}
    states = _extract_states_from_text(
        question,
        cutoff_db_states=cutoff_db_states,
        use_llm_fallback=True,
    )
    if states:
        updates["target_states"] = states
    # Keep this helper minimal; quota/type relaxation is interpreted by LLM path.
    return updates


async def _llm_interpret_cutoff_refinement(
    question: str,
    cutoff_ctx: Dict[str, object],
    allowed_states: List[str],
    current_turn_signals: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """
    LLM interprets follow-up refinement intent after results are shown.
    Handles: state changes, college_type, course, seat_type, quota.
    All mapping is done by the LLM — no hardcoded keyword lists.
    Called only when user sends a follow-up after seeing results.
    """
    if not question:
        return {"apply": False}
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        payload = {
            "question": question,
            "current_context": {
                "metric_type": cutoff_ctx.get("metric_type"),
                "metric_value": cutoff_ctx.get("metric_value"),
                "category": cutoff_ctx.get("category"),
                "home_state": cutoff_ctx.get("home_state"),
                "target_states": list(cutoff_ctx.get("target_states") or []),
                "active_college_type": cutoff_ctx.get("college_type_filter"),
                "active_course": cutoff_ctx.get("course_filter"),
                "active_quota_keywords": cutoff_ctx.get("quota_keywords"),
                "active_seat_type_keywords": cutoff_ctx.get("seat_type_keywords"),
                "last_result_count": cutoff_ctx.get("last_result_count"),
            },
            "current_turn_signals": dict(current_turn_signals or {}),
            "allowed_states": allowed_states,
        }
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You interpret NEET UG cutoff search refinement intent from a user follow-up message.\n"
                        "The user has already seen a college shortlist and is now refining or expanding it.\n"
                        "Output structured JSON only — no extra text.\n\n"

                        "## RETURN KEYS\n"
                        "apply: bool — true if any change should be applied\n"
                        "target_states: array of exact strings from allowed_states (empty = no change)\n"
                        "college_type_filter: exact string | null — one of: "
                        "Government, Private, Deemed, AIIMS, JIPMER, AMU, BHU, Jamia Milia\n"
                        "clear_college_type: bool — true if user wants to remove college type filter\n"
                        "course_filter: exact string | null — one of: MBBS, BDS, B.Sc. Nursing\n"
                        "clear_course: bool — true if user wants all courses again\n"
                        "quota_keywords: array of short keyword strings for ILIKE search | null\n"
                        "clear_quota: bool — true if user wants to remove quota filter\n"
                        "seat_type_keywords: array of short keyword strings for ILIKE search | null\n"
                        "clear_seat_type: bool — true if user wants to remove seat type filter\n\n"

                        "## STATE RULES\n"
                        "- Only change target_states if user explicitly names a state or says switch/home state only/nearby.\n"
                        "- If current_turn_signals has explicit_states_count > 0, metric_detected, or category_detected, "
                        "treat as profile input not refinement — return apply=false.\n"
                        "- Use only values from allowed_states.\n"
                        "- 'home state only' → target_states = [home_state from current_context].\n"
                        "- 'nearby states' / 'expand' → intelligently expand around current target_states.\n\n"

                        "## COLLEGE TYPE RULES\n"
                        "Map user language to exact DB values:\n"
                        "- government / govt / sarkari / public college → 'Government'\n"
                        "- private college → 'Private'\n"
                        "- deemed / deemed university → 'Deemed'\n"
                        "- aiims → 'AIIMS'\n"
                        "- jipmer → 'JIPMER'\n"
                        "- amu / aligarh muslim → 'AMU'\n"
                        "- bhu / banaras hindu → 'BHU'\n"
                        "- jamia / jamia millia → 'Jamia Milia'\n"
                        "- 'show all types' / 'any college type' / 'remove type filter' → clear_college_type=true\n\n"

                        "## COURSE RULES\n"
                        "Map user language to exact DB values:\n"
                        "- mbbs / medical / mbbs seat → 'MBBS'\n"
                        "- bds / dental / dentistry → 'BDS'\n"
                        "- nursing / bsc nursing / b.sc nursing → 'B.Sc. Nursing'\n"
                        "- 'all courses' / 'any course' / 'remove course filter' → clear_course=true\n"
                        "Only set course_filter when user explicitly restricts to a course.\n\n"

                        "## QUOTA RULES (fuzzy ILIKE keywords — DB values are inconsistent)\n"
                        "Extract short keywords the user intends. Examples:\n"
                        "- all india quota / aiq → ['AIQ', 'All India']\n"
                        "- state quota → ['State Quota']\n"
                        "- management quota / management seats → ['Management']\n"
                        "- nri quota → ['NRI']\n"
                        "- open quota → ['Open']\n"
                        "- defence quota / army quota → ['Defence', 'Defense']\n"
                        "- 'remove quota filter' → clear_quota=true\n"
                        "Keep keywords short — they are used in SQL ILIKE '%keyword%'.\n\n"

                        "## SEAT TYPE RULES (fuzzy ILIKE keywords)\n"
                        "Only set if user explicitly mentions seat type. Examples:\n"
                        "- government seat → ['Government']\n"
                        "- nri seat → ['NRI']\n"
                        "- management seat → ['Management', 'MGT']\n"
                        "- state quota seat → ['State Quota']\n"
                        "- 'remove seat type filter' → clear_seat_type=true\n\n"

                        "## CRITICAL RULE\n"
                        "Only set apply=true when there is a clear, actionable refinement in the current message.\n"
                        "If user is just acknowledging, asking a general question, or continuing — return apply=false.\n"
                        "Never invent filters the user did not mention."
                    ),
                },
                {"role": "user", "content": json.dumps(payload)},
            ],
            temperature=0,
            max_tokens=300,
        )
        raw = (resp.choices[0].message.content or "").strip()
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            return {"apply": False}

        out: Dict[str, object] = {
            "apply": bool(parsed.get("apply", False)),
            "target_states": [],
            "college_type_filter": None,
            "clear_college_type": bool(parsed.get("clear_college_type", False)),
            "course_filter": None,
            "clear_course": bool(parsed.get("clear_course", False)),
            "quota_keywords": None,
            "clear_quota": bool(parsed.get("clear_quota", False)),
            "seat_type_keywords": None,
            "clear_seat_type": bool(parsed.get("clear_seat_type", False)),
        }

        # Validate target states
        allowed = {s for s in allowed_states}
        states = parsed.get("target_states", [])
        if isinstance(states, list):
            deduped: List[str] = []
            seen: set = set()
            for st in states:
                if isinstance(st, str) and st in allowed and st not in seen:
                    deduped.append(st)
                    seen.add(st)
            out["target_states"] = deduped[:6]

        # Validate college_type_filter
        valid_college_types = {
            "Government", "Private", "Deemed", "AIIMS", "JIPMER", "AMU", "BHU", "Jamia Milia"
        }
        ct = parsed.get("college_type_filter")
        if isinstance(ct, str) and ct.strip() in valid_college_types:
            out["college_type_filter"] = ct.strip()

        # Validate course_filter
        valid_courses = {"MBBS", "BDS", "B.Sc. Nursing"}
        cf = parsed.get("course_filter")
        if isinstance(cf, str) and cf.strip() in valid_courses:
            out["course_filter"] = cf.strip()

        # Validate quota_keywords (short strings for ILIKE)
        qk = parsed.get("quota_keywords")
        if isinstance(qk, list) and qk:
            out["quota_keywords"] = [str(k).strip() for k in qk if str(k).strip()][:5]

        # Validate seat_type_keywords
        stk = parsed.get("seat_type_keywords")
        if isinstance(stk, list) and stk:
            out["seat_type_keywords"] = [str(k).strip() for k in stk if str(k).strip()][:5]

        # Set apply=true if any change is present
        if (
            out["target_states"]
            or out["college_type_filter"]
            or out["clear_college_type"]
            or out["course_filter"]
            or out["clear_course"]
            or out["quota_keywords"]
            or out["clear_quota"]
            or out["seat_type_keywords"]
            or out["clear_seat_type"]
        ):
            out["apply"] = True

        return out
    except Exception as e:
        log(f"[V2] ⚠️ Cutoff refinement interpreter failed: {e}")
        return {"apply": False}


async def _llm_extract_cutoff_profile_turn(
    question: str,
    cutoff_ctx: Dict[str, object],
    allowed_states: List[str],
    pending_fields: Optional[List[str]] = None,
    allowed_sub_categories: Optional[List[str]] = None,
) -> Dict[str, object]:
    """
    LLM-only extraction for cutoff profile fields.
    Determines who the query is for (self/friend/general) and extracts
    all relevant fields. No hardcoded keyword detection anywhere.
    """
    if not question:
        return {}
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        payload = {
            "question": question,
            "current_context": {
                "profile_mode": cutoff_ctx.get("profile_mode"),
                "metric_type": cutoff_ctx.get("metric_type"),
                "metric_value": cutoff_ctx.get("metric_value"),
                "category": cutoff_ctx.get("category"),
                "home_state": cutoff_ctx.get("home_state"),
                "target_states": list(cutoff_ctx.get("target_states") or []),
            },
            "allowed_states": allowed_states,
            "pending_fields": list(pending_fields or []),
            "allowed_sub_categories": list(allowed_sub_categories or []),
        }
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You extract NEET UG cutoff search profile fields from a user message "
                        "in a counselling chat.\n\n"

                        "## WHO IS THIS QUERY FOR? (profile_mode)\n"
                        "Determine the subject of the query:\n"
                        "- 'self': user is asking for themselves (default when no other signal)\n"
                        "- 'friend': user explicitly mentions asking for a friend, cousin, sibling, "
                        "relative, child, son, daughter, or any person they refer to as someone else\n"
                        "- 'general': user is asking hypothetically or about an unnamed/unrelated person "
                        "('if someone has rank X', 'a student with score Y', 'in general', "
                        "'suppose a candidate')\n\n"
                        "If current_context already has profile_mode set, preserve it UNLESS the current "
                        "message clearly signals a switch (e.g. was 'self', now user says 'for my friend').\n\n"

                        "## RETURN JSON WITH EXACTLY THESE KEYS\n"
                        "profile_mode: 'self' | 'friend' | 'general'\n"
                        "metric_type: 'rank' | 'score' | null\n"
                        "metric_value: integer | null\n"
                        "home_state: exact string from allowed_states | null\n"
                        "target_states: array of exact strings from allowed_states\n"
                        "category: GENERAL | OBC | SC | ST | EWS | PWD | null\n"
                        "sub_category: exact string from allowed_sub_categories | null\n"
                        "signals: object with booleans: metric_detected, category_detected, "
                        "explicit_states_detected, home_state_detected\n\n"

                        "## EXTRACTION RULES\n\n"
                        "States:\n"
                        "- Use ONLY values from allowed_states.\n"
                        "- Resolve common abbreviations: UP→Uttar Pradesh, MP→Madhya Pradesh, "
                        "J&K→Jammu & Kashmir, AP→Andhra Pradesh, TN→Tamilnadu, HP→Himachal Pradesh, "
                        "UK→Uttarakhand, MH→Maharashtra, KA/KL→Karnataka/Kerala.\n\n"

                        "Metric (rank vs score):\n"
                        "- Numbers ≤ 720 are almost certainly NEET scores (max score is 720).\n"
                        "- Numbers > 720 are AIR ranks.\n"
                        "- Keywords 'rank'/'AIR'/'all india rank' → metric_type=rank.\n"
                        "- Keywords 'score'/'marks'/'percentile' → metric_type=score.\n"
                        "- When ambiguous, infer from range above.\n\n"

                        "Home state:\n"
                        "- Extract when user says 'from X', 'home state is X', 'domicile X', "
                        "'belongs to X', 'native of X', or in friend mode 'friend is from X'.\n"
                        "- If pending_fields includes 'home_state' and user gives a single state name, "
                        "treat as home_state AND include in target_states if no other target given.\n\n"

                        "Target states:\n"
                        "- States where user wants to explore colleges. May differ from home_state.\n"
                        "- 'home state only' → target_states = [home_state from current_context].\n"
                        "- 'check in X' / 'switch to X' / 'look in X' → target_states = [X].\n"
                        "- If user gives rank/score + one state in natural text (e.g., 'rank 2300 jharkhand', "
                        "'AIR 23k in UP', Hindi/Marathi equivalents), treat that state as target_states.\n"
                        "- Do not ask target state again when the message already contains one clear state for search.\n"
                        "- Compact replies are valid profile input. Example: '23000 Delhi' means "
                        "metric_value=23000 (rank by range) and target_states=['Delhi'].\n"
                        "- If pending_fields includes target_states and user provides one clear state token "
                        "(e.g., 'Delhi'), set target_states to that state and mark explicit_states_detected=true.\n"
                        "- For rank/score-only replies, do NOT fabricate target_states.\n\n"

                        "Category:\n"
                        "- Map: general/UR/open/unreserved → GENERAL, obc → OBC, sc → SC, "
                        "st → ST, ews → EWS, pwd/pwbd/handicapped/disabled/physically handicapped → PWD.\n"
                        "- If pending_fields includes 'category' and user reply is a short category word, "
                        "map it with high confidence.\n\n"

                        "Sub-category:\n"
                        "- Only extract if allowed_sub_categories is non-empty and user mentions one.\n"
                        "- If user says 'none' or 'skip', return sub_category=null.\n\n"

                        "General:\n"
                        "- Do not fabricate values. If not clearly in the message, return null.\n"
                        "- For friend/general mode: extracted fields belong to the friend/candidate.\n"
                    ),
                },
                {"role": "user", "content": json.dumps(payload)},
            ],
            temperature=0,
            max_tokens=300,
        )
        raw = (resp.choices[0].message.content or "").strip()
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            return {}

        allowed = {s for s in allowed_states}
        out: Dict[str, object] = {}

        # Profile mode
        pm = parsed.get("profile_mode")
        if pm in {"self", "friend", "general"}:
            out["profile_mode"] = pm

        # Metric
        mt = parsed.get("metric_type")
        if mt in {"rank", "score"}:
            out["metric_type"] = mt
        mv = parsed.get("metric_value")
        if isinstance(mv, (int, float)) and int(mv) > 0:
            out["metric_value"] = int(mv)

        # Category
        cat = parsed.get("category")
        if isinstance(cat, str) and cat.strip():
            out["category"] = cat.strip().upper()

        # Sub-category
        sub_cat = str(parsed.get("sub_category") or "").strip().upper()
        allowed_sub = {
            str(s).strip().upper()
            for s in (allowed_sub_categories or [])
            if str(s).strip()
        }
        if sub_cat and (not allowed_sub or sub_cat in allowed_sub):
            out["sub_category"] = sub_cat

        # Home state
        hs = parsed.get("home_state")
        if isinstance(hs, str) and hs in allowed:
            out["home_state"] = hs

        # Target states
        ts = parsed.get("target_states", [])
        if isinstance(ts, list):
            deduped: List[str] = []
            seen: set = set()
            for st in ts:
                if isinstance(st, str) and st in allowed and st not in seen:
                    deduped.append(st)
                    seen.add(st)
            if deduped:
                out["target_states"] = deduped[:6]

        # Signals
        signals = parsed.get("signals")
        if isinstance(signals, dict):
            out["signals"] = {
                "metric_detected": bool(signals.get("metric_detected", False)),
                "category_detected": bool(signals.get("category_detected", False)),
                "explicit_states_detected": bool(signals.get("explicit_states_detected", False)),
                "home_state_detected": bool(signals.get("home_state_detected", False)),
            }

        # LLM-assisted recovery: when target_states is pending, metric is detected, and a single
        # state was extracted into home_state from compact natural text (e.g. "rank 2300, jharkhand"),
        # treat it as target state to avoid redundant "which state?" follow-up.
        if (
            "target_states" in (pending_fields or [])
            and not out.get("target_states")
            and isinstance(out.get("home_state"), str)
        ):
            sig = dict(out.get("signals") or {})
            if bool(sig.get("metric_detected")) and (
                bool(sig.get("explicit_states_detected")) or bool(sig.get("home_state_detected"))
            ):
                out["target_states"] = [str(out["home_state"])]
                sig["explicit_states_detected"] = True
                out["signals"] = sig

        # Category retry for terse single-word replies when we're waiting for category
        if "category" in (pending_fields or []) and not out.get("category"):
            retry = client.chat.completions.create(
                model="gpt-4o-mini",
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Extract NEET category from user text.\n"
                            "Return JSON only: {\"category\": <GENERAL|OBC|SC|ST|EWS|PWD|null>}.\n"
                            "Map: general/UR/open/unreserved → GENERAL, obc → OBC, sc → SC, "
                            "st → ST, ews → EWS, pwd/pwbd/handicapped/disabled → PWD.\n"
                        ),
                    },
                    {"role": "user", "content": json.dumps({"question": question})},
                ],
                temperature=0,
                max_tokens=80,
            )
            retry_raw = (retry.choices[0].message.content or "").strip()
            retry_parsed = json.loads(retry_raw)
            retry_cat = str(retry_parsed.get("category") or "").strip().upper()
            if retry_cat in {"GENERAL", "OBC", "SC", "ST", "EWS", "PWD"}:
                out["category"] = retry_cat
                signals_obj = dict(out.get("signals") or {})
                signals_obj["category_detected"] = True
                out["signals"] = signals_obj

        return out
    except Exception as e:
        log(f"[V2] ⚠️ Cutoff profile extractor failed: {e}")
        return {}

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
    build_pinecone_filters,
    QueryIntent,
    format_mixed_response_prompt,
    expand_query,
)
from services.chunk_classifier import classify_chunk
from services.vector_store_factory import get_vector_store, count_vectors_sync
from services.metadata_filter_utils import pinecone_filter_to_metadata_filters
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
) -> tuple[bool, str]:
    """
    LLM-based generic sufficiency check.
    Returns (is_sufficient, reason).
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
                        "If retrieved text is for a different college/state than asked, mark is_sufficient=false.\n"
                        "Mark is_sufficient=true when the KB contains the requested data for the correct scope/entity; "
                        "mark false when the requested college/state/round/detail is absent, wrong, or ambiguous."
                        "\nReturn ONLY valid JSON with keys: is_sufficient (boolean), reason (string)."
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
            return is_sufficient, reason
        except Exception:
            lowered = raw.lower()
            m = re.search(r'"is_sufficient"\s*:\s*(true|false)', lowered)
            if m:
                is_sufficient = m.group(1) == "true"
                reason_match = re.search(r'"reason"\s*:\s*"([^"]*)"', raw, re.IGNORECASE)
                reason = reason_match.group(1).strip() if reason_match else "Parsed from non-JSON sufficiency output"
                log(f"[V2] ⚠️ KB sufficiency non-JSON output parsed via fallback: {raw[:300]!r}")
                return is_sufficient, reason
            log(f"[V2] ⚠️ KB sufficiency output unparseable: {raw[:300]!r}")
            return False, "Sufficiency output parse failed"
    except Exception as err:
        log(f"[V2] ⚠️ KB sufficiency check failed, defaulting to insufficient: {err}")
        return False, "Sufficiency check failed"


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
    """LlamaIndex PGVectorStore (Neon + pgvector)."""
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
    print("Index loaded from Neon pgvector!")

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
            "pinecone_connected": True,
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
            
            # Build vector metadata filters (same shapes as legacy Pinecone filters)
            pinecone_filters = build_pinecone_filters(routing, user_state)
            log(f"[INFO] 🔍 VECTOR FILTERS: {pinecone_filters}")

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

                for filter_idx, pc_filter in enumerate(pinecone_filters):
                    log(f"[INFO] 🔎 PGVECTOR QUERY {filter_idx + 1}: filter={pc_filter}")

                    mf = pinecone_filter_to_metadata_filters(pc_filter)
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
                temperature=0.3,
                max_tokens=V2_FINAL_MAX_TOKENS,
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
        round_tool_names: List[str] = []
        round_tool_exec_ms = 0.0
        pending_kb_tool_messages: List[Dict[str, Any]] = []
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
                pending_kb_tool_messages.extend(tool_pair)
            else:
                pending_other_tool_messages.extend(tool_pair)

        suff_ms = 0.0
        if kb_results_this_round:
            kb_attempted = True
            last_kb_retrieval = "\n\n".join(kb_results_this_round)
            t_suff = time.perf_counter()
            is_sufficient, _reason = assess_kb_sufficiency_with_llm(
                client=client,
                user_question=request.question,
                kb_tool_result=last_kb_retrieval,
                conversation_context=_build_sufficiency_context(messages, request.question),
            )
            suff_ms = _elapsed_ms(t_suff)
            _log_v2_tool_debug(
                f"[V2][DBG] sufficiency round={round_idx + 1} "
                f"is_sufficient={bool(is_sufficient)} reason={_reason!r}"
            )
            kb_sufficient_for_final = bool(is_sufficient)
            if not is_sufficient:
                if web_fallback_enabled:
                    # Latency optimization: avoid an extra planner LLM round.
                    # Once sufficiency is false, trigger web search directly.
                    fallback_query = (request.question or "").strip()
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

                    if web_success:
                        last_web_retrieval = web_result
                        used_web_fallback = True
                        if stream_source_origin != "web":
                            yield f"data: {json.dumps({'type': 'meta', 'source_origin': 'web'})}\n\n"
                            stream_source_origin = "web"

                        web_sources = []
                        for line in web_result.split("\n"):
                            if line.startswith("[") and "Title:" in line:
                                title = line.split("Title:", 1)[1].strip()
                                web_sources.append({"file_name": title, "document_type": "web_search"})
                        if web_sources and references_enabled:
                            yield f"data: {json.dumps({'type': 'sources', 'sources': web_sources[:5]})}\n\n"

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
                        direct_web_fallback_done = True
                    else:
                        kb_insufficient_and_web_disabled = True
                else:
                    kb_insufficient_and_web_disabled = True
            else:
                # Keep KB retrieval in final context only when sufficiency is true.
                if pending_kb_tool_messages:
                    messages.extend(pending_kb_tool_messages)
                if pending_other_tool_messages:
                    messages.extend(pending_other_tool_messages)
        elif kb_results_this_round:
            kb_attempted = True
            last_kb_retrieval = "\n\n".join(kb_results_this_round)
            kb_sufficient_for_final = True
            if pending_kb_tool_messages:
                messages.extend(pending_kb_tool_messages)
            if pending_other_tool_messages:
                messages.extend(pending_other_tool_messages)
        else:
            if pending_kb_tool_messages:
                messages.extend(pending_kb_tool_messages)
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
    if forced_fallback_response:
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
                temperature=0.3,
                max_tokens=V2_FINAL_MAX_TOKENS,
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
            _first_out = True
            streamed_raw = ""
            async for delta in _stream_chat_completion_text(client, model="gpt-4o-mini", messages=messages, temperature=0.3, max_tokens=V2_FINAL_MAX_TOKENS):
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
            final_response = client.chat.completions.create(model="gpt-4o-mini", messages=messages, temperature=0.3, max_tokens=V2_FINAL_MAX_TOKENS)
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

    if _is_greeting_only(request.question):
        registered_user_name = await _get_registered_user_name(request.user_id)
        first_name = _extract_first_name(registered_user_name)
        welcome = _first_visit_welcome_message(first_name) if is_first_visit else _return_visit_welcome_message(first_name)
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


async def _v2_route_cutoff_stage(
    *,
    question: str,
    med_ctx: Dict[str, object],
) -> Tuple[bool, bool, bool]:
    """
    Stage function for deciding cutoff SQL routing.
    Returns: (cutoff_triggered, explicit_shortlist_start, continuing_cutoff_flow)
    """
    cutoff_ctx_peek = dict(med_ctx.get("cutoff") or {})
    cutoff_triggered = False
    explicit_shortlist_start = False
    continuing_cutoff_flow = False
    if not MEDBUDDY_CAPS["cutoff"]:
        return False, False, False

    if _is_explicit_college_shortlist_trigger(question):
        cutoff_triggered = True
        explicit_shortlist_start = True
        log("[V2] 🧭 Explicit college shortlist trigger detected; forcing cutoff SQL profile flow")
    elif _should_continue_cutoff_from_context(question, cutoff_ctx_peek):
        cutoff_triggered = True
        continuing_cutoff_flow = True
        log("[V2] 🧭 Continuing cutoff refinement flow from conversation context")

    if cutoff_triggered:
        llm_route_decision = True
    else:
        if _v2_async_cutoff_router_enabled():
            route_task = asyncio.create_task(_llm_should_route_cutoff_sql_async(question, med_ctx))
            llm_route_decision = await route_task
        else:
            llm_route_decision = await _llm_should_route_cutoff_sql_async(question, med_ctx)
    if llm_route_decision is None:
        cutoff_triggered = False
        log("[V2] 🧭 Cutoff route fallback decision: False (LLM classifier unavailable)")
    else:
        cutoff_triggered = llm_route_decision
    return cutoff_triggered, explicit_shortlist_start, continuing_cutoff_flow


async def _v2_handle_cutoff_sql_stage(
    *,
    request: ChatRequest,
    conversation_id: Optional[int],
    med_ctx: Dict[str, object],
    preferred_language: str,
    localize_output,
    start_time: datetime,
    explicit_shortlist_start: bool,
    continuing_cutoff_flow: bool,
    state: Dict[str, object],
) -> AsyncGenerator[str, None]:
    """
    Stage function for cutoff SQL flow.
    Emits events and sets state['handled'] when cutoff stage handles the request.
    """
    from services.cutoff_service import (
        fetch_cutoff_recommendations,
        format_cutoff_markdown,
    )

    state["handled"] = False
    log("[V2] 🎯 Routing to CUTOFF SQL path")
    cutoff_db_states = await _get_cutoff_db_states_cached()

    cutoff_ctx = dict(med_ctx.get("cutoff") or {})
    _expire_stale_cutoff_run_state(cutoff_ctx, freshness_minutes=30)

    # Determine current profile mode before deciding what to clear
    current_profile_mode = str(cutoff_ctx.get("profile_mode") or "self")

    if explicit_shortlist_start or not continuing_cutoff_flow:
        keys_to_clear = [
            "metric_type",
            "metric_value",
            "target_states",
            "target_states_confirmed",
            "awaiting_refinement_choice",
            "awaiting_refinement_details",
            "college_type_filter",
            "course_filter",
            "quota_keywords",
            "seat_type_keywords",
            "last_result_count",
        ]
        # friend/general: also reset profile fields so we collect fresh each time
        if current_profile_mode in {"friend", "general"}:
            keys_to_clear += [
                "home_state", "category", "sub_category",
                "preferences_set", "profile_bootstrapped",
            ]
        for key in keys_to_clear:
            cutoff_ctx.pop(key, None)
    profile_mode = str(cutoff_ctx.get("profile_mode") or "self")
    # profile_mode is set by _llm_extract_cutoff_profile_turn later in this turn.
    # Bootstrap from saved DB profile ONLY for self mode, ONLY once per conversation.
    if profile_mode == "self" and request.user_id and not cutoff_ctx.get("profile_bootstrapped"):
        stored_profile = await _load_user_cutoff_profile(request.user_id)
        if stored_profile.get("home_state") and not cutoff_ctx.get("home_state"):
            cutoff_ctx["home_state"] = stored_profile["home_state"]
        if stored_profile.get("category") and not cutoff_ctx.get("category"):
            cutoff_ctx["category"] = stored_profile["category"]
        if stored_profile.get("sub_category") and not cutoff_ctx.get("sub_category"):
            cutoff_ctx["sub_category"] = stored_profile["sub_category"]
        cutoff_ctx["preferences_set"] = bool(stored_profile.get("preferences_set", False))
        cutoff_ctx["profile_bootstrapped"] = True
        log(
            f"[V2] 👤 Profile bootstrapped from DB | "
            f"home_state={cutoff_ctx.get('home_state')} "
            f"category={cutoff_ctx.get('category')} "
            f"preferences_set={cutoff_ctx.get('preferences_set')}"
        )

    category_state_for_options = cutoff_ctx.get("home_state")
    if not category_state_for_options and cutoff_ctx.get("target_states"):
        category_state_for_options = list(cutoff_ctx.get("target_states") or [None])[0]
    category_options = await _get_cutoff_category_options(
        str(category_state_for_options) if category_state_for_options else None
    )
    sub_category_options: List[str] = []
    if cutoff_ctx.get("category"):
        sub_category_options = await _get_cutoff_subcategory_options(
            str(category_state_for_options) if category_state_for_options else None,
            str(cutoff_ctx.get("category")),
        )
    cutoff_ctx["needs_sub_category"] = bool(sub_category_options)
    missing_before_turn = _missing_cutoff_fields(cutoff_ctx)

    if cutoff_ctx.get("awaiting_refinement_choice") and _is_affirmative_reply(request.question):
        # Build dynamic prompt showing active filters
        active_filters = []
        if cutoff_ctx.get("college_type_filter"):
            active_filters.append(f"College type: **{cutoff_ctx['college_type_filter']}**")
        if cutoff_ctx.get("course_filter"):
            active_filters.append(f"Course: **{cutoff_ctx['course_filter']}**")
        if cutoff_ctx.get("quota_keywords"):
            active_filters.append(f"Quota: **{', '.join(cutoff_ctx['quota_keywords'])}**")
        if cutoff_ctx.get("seat_type_keywords"):
            active_filters.append(f"Seat type: **{', '.join(cutoff_ctx['seat_type_keywords'])}**")

        active_str = (
            f"\n\nCurrently active filters: {', '.join(active_filters)}"
            if active_filters else ""
        )
        followup = (
            f"Sure — happy to refine.{active_str}\n\n"
            "What would you like to change?\n\n"
            "- **College type:** Government / Private / Deemed / AIIMS / JIPMER\n"
            "- **Course:** MBBS / BDS / B.Sc. Nursing\n"
            "- **Quota:** AIQ / State Quota / Management Quota / NRI / Open\n"
            "- **State(s):** Switch or add states\n\n"
            "Example: `Only government MBBS colleges in Bihar with State quota`"
        )
        followup = localize_output(followup)
        cutoff_ctx["awaiting_refinement_choice"] = False
        cutoff_ctx["awaiting_refinement_details"] = True
        med_ctx["cutoff"] = cutoff_ctx
        for token in sse_tokens_preserving_formatting(followup):
            yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"
            if V2_STREAM_TOKEN_DELAY_SEC > 0:
                await asyncio.sleep(V2_STREAM_TOKEN_DELAY_SEC)
        yield f"data: {json.dumps({'type': 'done', 'conversation_id': conversation_id})}\n\n"
        if request.user_id and conversation_id:
            asyncio.create_task(
                v2_background_save_conversation_turn(
                    conversation_id,
                    request.question,
                    followup,
                    int((datetime.now() - start_time).total_seconds() * 1000),
                )
            )
            asyncio.create_task(v2_background_update_conversation_context(conversation_id, med_ctx))
        state["handled"] = True
        return

    profile_turn: Dict[str, object] = {}
    if explicit_shortlist_start and _is_explicit_college_shortlist_trigger(request.question):
        # Fast path: plain "College Shortlist" starter does not carry extractable profile fields.
        # Skip extractor LLM call to reduce first-response latency.
        log("[V2] ⚡ Skipping cutoff extractor LLM for plain shortlist starter")
    else:
        profile_turn = await _llm_extract_cutoff_profile_turn(
            request.question,
            cutoff_ctx,
            cutoff_db_states,
            pending_fields=missing_before_turn,
            allowed_sub_categories=sub_category_options,
        )
    # Apply profile_mode from LLM extraction — trust LLM, no keyword detection
    extracted_mode = profile_turn.get("profile_mode")
    prev_mode = str(cutoff_ctx.get("profile_mode") or "self")
    if extracted_mode in {"self", "friend", "general"}:
        # If switching from self to friend/general, clear stale self-profile data
        # so it doesn't bleed into the friend/general query
        if prev_mode == "self" and extracted_mode in {"friend", "general"}:
            for key in ["home_state", "category", "sub_category",
                        "preferences_set", "profile_bootstrapped"]:
                cutoff_ctx.pop(key, None)
            log(f"[V2] 🔄 Mode switch self→{extracted_mode}, cleared self-profile data")
        cutoff_ctx["profile_mode"] = extracted_mode
        profile_mode = extracted_mode
    else:
        # LLM didn't return a mode — keep existing
        profile_mode = prev_mode
        cutoff_ctx["profile_mode"] = profile_mode

    metric_type = profile_turn.get("metric_type")
    metric_value = profile_turn.get("metric_value")
    detected_states = list(profile_turn.get("target_states") or [])
    category = profile_turn.get("category") or cutoff_ctx.get("category")
    home_state = profile_turn.get("home_state") or cutoff_ctx.get("home_state")

    explicit_states_in_turn = bool(profile_turn.get("signals", {}).get("explicit_states_detected"))
    target_state_was_missing = "target_states" in missing_before_turn
    # Accept detected target state either when extractor marks it explicit,
    # or when target state is currently missing and user provides a compact reply like "23000 Delhi".
    accept_detected_states = bool(detected_states) and (explicit_states_in_turn or target_state_was_missing)
    target_states = detected_states if accept_detected_states else list(cutoff_ctx.get("target_states") or [])

    if metric_type and metric_value:
        cutoff_ctx["metric_type"] = metric_type
        cutoff_ctx["metric_value"] = int(metric_value)
    if category:
        cutoff_ctx["category"] = str(category).upper()
    if home_state:
        cutoff_ctx["home_state"] = str(home_state)
    if profile_turn.get("sub_category"):
        cutoff_ctx["sub_category"] = str(profile_turn.get("sub_category"))
    if target_states:
        cutoff_ctx["target_states"] = target_states
        if explicit_states_in_turn or target_state_was_missing:
            cutoff_ctx["target_states_confirmed"] = True

    category_state_for_options = cutoff_ctx.get("home_state")
    if not category_state_for_options and cutoff_ctx.get("target_states"):
        category_state_for_options = list(cutoff_ctx.get("target_states") or [None])[0]
    sub_category_options = []
    if cutoff_ctx.get("category"):
        sub_category_options = await _get_cutoff_subcategory_options(
            str(category_state_for_options) if category_state_for_options else None,
            str(cutoff_ctx.get("category")),
        )
    cutoff_ctx["needs_sub_category"] = bool(sub_category_options)
    if not cutoff_ctx.get("needs_sub_category"):
        cutoff_ctx.pop("sub_category", None)
    # Save to DB profile ONLY for self mode — never for friend/general
    if profile_mode == "self" and request.user_id:
        await _save_user_cutoff_profile(request.user_id, cutoff_ctx)
        cutoff_ctx["preferences_set"] = bool(
            cutoff_ctx.get("home_state") and cutoff_ctx.get("category")
        )
    # friend/general: never save, never set preferences_set

    # Refinement: run LLM interpreter when user has already seen results
    # or when awaiting refinement details. No hardcoded extraction.
    should_run_refinement_llm = bool(
        cutoff_ctx.get("awaiting_refinement_details")
        or cutoff_ctx.get("awaiting_refinement_choice")
        or (
            cutoff_ctx.get("last_result_count") is not None
            and not missing_before_turn
            and not profile_turn.get("signals", {}).get("metric_detected")
            and not profile_turn.get("signals", {}).get("home_state_detected")
        )
    )
    if cutoff_ctx.get("awaiting_refinement_details"):
        cutoff_ctx["awaiting_refinement_details"] = False

    refinement_plan = {"apply": False}
    if should_run_refinement_llm:
        refinement_plan = await _llm_interpret_cutoff_refinement(
            request.question,
            cutoff_ctx,
            cutoff_db_states,
            current_turn_signals={
                "explicit_states_count": len(detected_states) if profile_turn.get("signals", {}).get("explicit_states_detected") else 0,
                "metric_detected": bool(profile_turn.get("signals", {}).get("metric_detected", False)),
                "category_detected": bool(profile_turn.get("signals", {}).get("category_detected", False)),
            },
        )

    if refinement_plan.get("apply"):
        planned_states = list(refinement_plan.get("target_states") or [])
        if planned_states:
            cutoff_ctx["target_states"] = planned_states
            cutoff_ctx["target_states_confirmed"] = True

        # College type — exact match (clear or set)
        if refinement_plan.get("clear_college_type"):
            cutoff_ctx.pop("college_type_filter", None)
        elif refinement_plan.get("college_type_filter"):
            cutoff_ctx["college_type_filter"] = refinement_plan["college_type_filter"]

        # Course — exact match (clear or set)
        if refinement_plan.get("clear_course"):
            cutoff_ctx.pop("course_filter", None)
        elif refinement_plan.get("course_filter"):
            cutoff_ctx["course_filter"] = refinement_plan["course_filter"]

        # Quota — fuzzy keywords (clear or set)
        if refinement_plan.get("clear_quota"):
            cutoff_ctx.pop("quota_keywords", None)
        elif refinement_plan.get("quota_keywords"):
            cutoff_ctx["quota_keywords"] = list(refinement_plan["quota_keywords"])

        # Seat type — fuzzy keywords (clear or set)
        if refinement_plan.get("clear_seat_type"):
            cutoff_ctx.pop("seat_type_keywords", None)
        elif refinement_plan.get("seat_type_keywords"):
            cutoff_ctx["seat_type_keywords"] = list(refinement_plan["seat_type_keywords"])

        log(
            f"[V2] 🧩 Refinement applied | "
            f"states={cutoff_ctx.get('target_states')} "
            f"college_type={cutoff_ctx.get('college_type_filter')} "
            f"course={cutoff_ctx.get('course_filter')} "
            f"quota_kw={cutoff_ctx.get('quota_keywords')} "
            f"seat_kw={cutoff_ctx.get('seat_type_keywords')}"
        )

    cutoff_ctx["last_turn_at"] = datetime.utcnow().isoformat()
    med_ctx["cutoff"] = cutoff_ctx
    missing = _missing_cutoff_fields(cutoff_ctx)
    if missing:
        # Profile form (frontend UI) only for self mode on first-time setup
        needs_profile_form = (
            profile_mode == "self"
            and not bool(cutoff_ctx.get("preferences_set"))
            and ("home_state" in missing or "category" in missing)
        )
        if needs_profile_form:
            intro = localize_output(
                "Great - I can help with college shortlist. Please fill the details below so I can suggest the best matches."
            )
            category_state = cutoff_ctx.get("home_state")
            if not category_state and cutoff_ctx.get("target_states"):
                category_state = list(cutoff_ctx.get("target_states") or [None])[0]
            profile_categories = await _get_cutoff_category_options(
                str(category_state) if category_state else None
            )
            profile_sub_categories = []
            if category_state and cutoff_ctx.get("category"):
                profile_sub_categories = await _get_cutoff_subcategory_options(
                    str(category_state),
                    str(cutoff_ctx.get("category")),
                )
            yield (
                "data: "
                + json.dumps(
                    {
                        "type": "cutoff_profile_form",
                        "message": intro,
                        "state": cutoff_ctx.get("home_state"),
                        "category": cutoff_ctx.get("category"),
                        "sub_category": cutoff_ctx.get("sub_category"),
                        "states": cutoff_db_states,
                        "categories": profile_categories,
                        "sub_categories": profile_sub_categories,
                    }
                )
                + "\n\n"
            )
            for token in sse_tokens_preserving_formatting(intro):
                yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"
                if V2_STREAM_TOKEN_DELAY_SEC > 0:
                    await asyncio.sleep(V2_STREAM_TOKEN_DELAY_SEC)
            yield f"data: {json.dumps({'type': 'done', 'conversation_id': conversation_id})}\n\n"
            if request.user_id and conversation_id:
                asyncio.create_task(
                    v2_background_save_conversation_turn(
                        conversation_id,
                        request.question,
                        intro,
                        int((datetime.now() - start_time).total_seconds() * 1000),
                    )
                )
                asyncio.create_task(v2_background_update_conversation_context(conversation_id, med_ctx))
            state["handled"] = True
            return

        followup = localize_output(
            _build_cutoff_followup_prompt(
                missing,
                cutoff_ctx,
                category_options=category_options,
                sub_category_options=sub_category_options,
            )
        )
        for token in sse_tokens_preserving_formatting(followup):
            yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"
            if V2_STREAM_TOKEN_DELAY_SEC > 0:
                await asyncio.sleep(V2_STREAM_TOKEN_DELAY_SEC)
        yield f"data: {json.dumps({'type': 'done', 'conversation_id': conversation_id})}\n\n"
        if request.user_id and conversation_id:
            asyncio.create_task(
                v2_background_save_conversation_turn(
                    conversation_id,
                    request.question,
                    followup,
                    int((datetime.now() - start_time).total_seconds() * 1000),
                )
            )
            asyncio.create_task(v2_background_update_conversation_context(conversation_id, med_ctx))
        state["handled"] = True
        return

    metric_type = str(cutoff_ctx.get("metric_type"))
    metric_value = int(cutoff_ctx.get("metric_value"))
    home_state = str(cutoff_ctx.get("home_state"))
    category = str(cutoff_ctx.get("category"))
    sub_category = str(cutoff_ctx.get("sub_category")) if cutoff_ctx.get("sub_category") else None
    target_states = [str(s) for s in list(cutoff_ctx.get("target_states") or [])]
    cutoff_result_limit = await get_cutoff_result_limit()
    rows = await fetch_cutoff_recommendations(
        metric_type=metric_type,
        metric_value=metric_value,
        home_state=home_state,
        target_states=target_states,
        category=category,
        sub_category=sub_category,
        college_type_filter=cutoff_ctx.get("college_type_filter") or None,
        course_filter=cutoff_ctx.get("course_filter") or None,
        quota_keywords=cutoff_ctx.get("quota_keywords") if isinstance(cutoff_ctx.get("quota_keywords"), list) else None,
        seat_type_keywords=cutoff_ctx.get("seat_type_keywords") if isinstance(cutoff_ctx.get("seat_type_keywords"), list) else None,
        total_limit=cutoff_result_limit,
    )
    cutoff_ctx["last_result_count"] = len(rows)
    cutoff_answer = localize_output(
        format_cutoff_markdown(
            rows=rows,
            metric_type=metric_type,
            metric_value=metric_value,
            category=category,
            sub_category=sub_category,
            home_state=home_state,
            target_states=target_states,
            display_limit=cutoff_result_limit,
        )
    )
    cutoff_explain = await _llm_explain_cutoff_results(
        user_question=request.question,
        metric_type=metric_type,
        metric_value=metric_value,
        home_state=home_state,
        target_states=target_states,
        category=category,
        sub_category=sub_category,
        rows=rows,
    )
    if cutoff_explain:
        cutoff_answer = (
            cutoff_answer
            + "\n\n### Quick Interpretation\n\n"
            + localize_output(cutoff_explain)
        )
    elif rows:
        cutoff_answer = (
            cutoff_answer
            + "\n\n### Quick Interpretation\n\n"
            + localize_output(
                _fallback_cutoff_quick_interpretation(
                    rows=rows,
                    metric_type=metric_type,
                    metric_value=metric_value,
                    target_states=target_states,
                )
            )
        )
    cutoff_answer += (
        "\n\n"
        + localize_output(
            "> *Note — Disclaimer: Cutoffs vary year to year and by round/quota/sub-category. "
            "Please verify final options on official MCC/state counselling portals.*"
        )
    )
    source = {
        "file_name": "neet_ug_2025_cutoffs",
        "document_type": "sql_cutoff_table",
        "state": ", ".join(target_states),
        "text_snippet": f"Cutoff rows matched: {len(rows)}",
    }
    yield f"data: {json.dumps({'type': 'sources', 'sources': [source]})}\n\n"
    cutoff_replies = await _generate_contextual_suggested_replies(
        request.question,
        cutoff_answer,
        med_ctx,
        retrieval_evidence=cutoff_answer,
        output_language=preferred_language,
    )
    if cutoff_replies:
        yield f"data: {json.dumps({'type': 'suggested_replies', 'replies': cutoff_replies})}\n\n"
    for token in sse_tokens_preserving_formatting(cutoff_answer):
        yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"
        if V2_STREAM_TOKEN_DELAY_SEC > 0:
            await asyncio.sleep(V2_STREAM_TOKEN_DELAY_SEC)
    med_ctx["stage"] = "normal_qa"
    med_ctx["last_topic"] = "Cutoff analysis"
    med_ctx["last_state"] = ", ".join(target_states)
    med_ctx["last_activity_at"] = datetime.utcnow().isoformat()
    cutoff_ctx["awaiting_refinement_choice"] = True
    med_ctx["cutoff"] = cutoff_ctx
    yield f"data: {json.dumps({'type': 'done', 'conversation_id': conversation_id})}\n\n"
    if request.user_id and conversation_id:
        asyncio.create_task(
            v2_background_save_conversation_turn(
                conversation_id,
                request.question,
                cutoff_answer,
                int((datetime.now() - start_time).total_seconds() * 1000),
                sources=[source],
            )
        )
        asyncio.create_task(v2_background_update_conversation_context(conversation_id, med_ctx))
    state["handled"] = True


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

            # ========== CUTOFF SQL PATH (TABLE-BASED, NO VECTOR SEARCH) ==========
            cutoff_triggered, explicit_shortlist_start, continuing_cutoff_flow = await _v2_route_cutoff_stage(
                question=request.question,
                med_ctx=med_ctx,
            )
            if cutoff_triggered:
                cutoff_stage_state: Dict[str, object] = {}
                async for event in _v2_handle_cutoff_sql_stage(
                    request=request,
                    conversation_id=conversation_id,
                    med_ctx=med_ctx,
                    preferred_language=preferred_language,
                    localize_output=localize_output,
                    start_time=start_time,
                    explicit_shortlist_start=explicit_shortlist_start,
                    continuing_cutoff_flow=continuing_cutoff_flow,
                    state=cutoff_stage_state,
                ):
                    yield event
                if cutoff_stage_state.get("handled"):
                    if request.user_id and conversation_id:
                        asyncio.create_task(
                            v2_background_generate_conversation_title(
                                conversation_id,
                                request.question,
                                log_label="cutoff",
                            )
                        )
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
                history = conversation_memory.get_formatted_history()
                if history.strip():
                    # Parse history and add as messages
                    for line in history.strip().split("\n"):
                        if line.startswith("User: "):
                            messages.append({"role": "user", "content": line[6:]})
                        elif line.startswith("Assistant: "):
                            messages.append({"role": "assistant", "content": line[11:]})
            
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
                    "- You may infer missing topic words from immediate context (e.g., fee structure), "
                    "but must keep entity scope anchored to the latest message."
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

            # Done before DB persist so the client is not blocked on Neon writes
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