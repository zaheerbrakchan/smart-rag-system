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
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Tuple, AsyncGenerator
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
    """Return True when conversation title is empty and should be generated."""
    from database.connection import async_session_maker
    try:
        async with async_session_maker() as db:
            convo = await db.get(Conversation, conversation_id)
            if not convo:
                return False
            title = (convo.title or "").strip()
            return not title
    except Exception as err:
        log(f"[V2] ⚠️ Could not check title state conv={conversation_id}: {err}")
        return False


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


def _is_broad_discovery_query(question: str) -> bool:
    q = _normalize_text(question)
    return _contains_any(
        q,
        [
            "help me",
            "start",
            "where should i apply",
            "which college",
            "college list",
            "shortlist",
            "suggest",
            "what should i do",
            "guide me",
        ],
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
    return updates


def _next_onboarding_prompt(med_ctx: Dict[str, object], question: str) -> Optional[Dict[str, object]]:
    onboarding = dict(med_ctx.get("onboarding") or {})
    q = _normalize_text(question)

    # Collect rank/category only when user is explicitly asking for
    # shortlist/cutoff-style guidance.
    needs_rank_context = any(
        k in q for k in ["shortlist", "cutoff", "cut off", "college list", "which college", "compare colleges"]
    )

    if not (_is_broad_discovery_query(question) or needs_rank_context):
        return None

    if needs_rank_context and not onboarding.get("rank_or_score"):
        return {
            "message": (
                "To guide you accurately for shortlist/cutoff planning, please share your "
                "NEET rank (or expected rank/score)."
            ),
            "replies": [],
            "stage": "guided_onboarding",
        }

    if not onboarding.get("category"):
        return {
            "message": (
                "Great to start with! To guide you better, which NEET category should I use?\n\n"
                "You can pick one option below."
            ),
            "replies": ["General", "OBC", "SC", "ST", "EWS", "PwD"],
            "stage": "guided_onboarding",
        }

    if not onboarding.get("preference"):
        return {
            "message": (
                "Got it. Do you want me to explore colleges across All India, "
                "focus on your home state, or only government colleges?"
            ),
            "replies": ["All India", "Home state only", "Government only", "All types"],
            "stage": "guided_onboarding",
        }

    if MEDBUDDY_CAPS["mbbs_abroad"] and not onboarding.get("abroad_interest"):
        return {
            "message": "Would you also like me to include MBBS abroad options?",
            "replies": ["Yes, show abroad options", "No, India only for now"],
            "stage": "guided_onboarding",
        }
    return None


def _first_visit_welcome_message() -> str:
    return (
        "I am **Med Buddy**, India’s first AI counselling companion built exclusively for NEET UG aspirants, "
        "proudly powered by **Get My University**.\n\n"
        "Whether you are trying to figure out which medical/dental colleges match your score, understand "
        "last year’s cutoffs, decode fee structures, plan counselling process, or check NEET exam details, "
        "you have support at every step.\n\n"
        "Tell me what you want to explore first."
    )


def _return_visit_welcome_message() -> str:
    return (
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


def _extract_entity_hint(question: str) -> Optional[str]:
    """
    Try to extract a specific college/entity phrase from the user question.
    Used to avoid false 'sufficient' decisions for entity-specific fee queries.
    """
    q = _normalize_text(question)
    patterns = [
        r"(gmc\s+[a-z&\-\s]{2,40})",
        r"(government medical college\s+[a-z&\-\s]{2,40})",
        r"(aiims\s+[a-z&\-\s]{2,40})",
        r"(amu\s+[a-z&\-\s]{2,40})",
        r"(jipmer\s+[a-z&\-\s]{2,40})",
        r"(kgmu\s+[a-z&\-\s]{2,40})",
    ]
    for pat in patterns:
        m = re.search(pat, q)
        if m:
            return m.group(1).strip()
    return None


def _apply_response_policy(answer: str, question: str) -> str:
    text = (answer or "").strip()
    if not text:
        return text
    q = _normalize_text(question)
    factual = any(k in q for k in ["fee", "cutoff", "rank", "date", "reservation", "seat"])
    has_disclaimer = "always verify" in text.lower() or "official website" in text.lower()
    if factual and not has_disclaimer:
        text += (
            "\n\n> Disclaimer: Information is based on available counselling documents. "
            "Always verify the latest updates on official MCC/state counselling websites before taking admission decisions."
        )
    if factual and "would you like" not in text.lower():
        text += "\n\nWould you like me to also compare this with another state/college for you?"
    return text


def _has_structured_data_signals(text: str) -> bool:
    t = _normalize_text(text)
    return any(
        k in t
        for k in [
            "fee", "tuition", "cost", "₹",
            "cutoff", "cut off", "rank", "score", "percentile",
            "seat", "quota", "reservation",
            "round", "counselling", "date", "schedule",
        ]
    )


def _is_entity_specific_query(question: str) -> bool:
    q = _normalize_text(question)
    if _extract_entity_hint(question):
        return True
    # Fallback pattern for "<college/entity> <metric>" type asks
    return bool(
        re.search(
            r"\b([a-z]{3,}\s+[a-z]{2,}(?:\s+[a-z]{2,}){0,4})\b.*\b(fee|cutoff|rank|seat|reservation|date|schedule)\b",
            q,
        )
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


def _is_cutoff_intent(question: str) -> bool:
    q = _normalize_text(question)
    return any(
        k in q
        for k in [
            "cutoff",
            "cut off",
            "cut-off",
            "which college can i get",
            "which colleges can i get",
            "college can i get",
            "expected college",
            "expected colleges",
            "based on my rank",
            "based on my score",
            "college prediction",
            "shortlist",
        ]
    )


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

Route to SQL cutoff when intent is college prediction/shortlisting based on rank or score, including follow-up profile payloads that provide score/rank/category/state.
Do NOT route to SQL for general counselling process, exam guidance, fee structure, documents, dates, or reservation policy info.

Conversation last topic: {med_ctx.get("last_topic") or "unknown"}
Cutoff context already present: {bool(cutoff_ctx)}
Cutoff context keys: {list(cutoff_ctx.keys())}
User message: "{question}"

Critical continuation rule:
- If recent conversation is already in cutoff shortlisting flow and current user message is a short follow-up
  (state name, category, rank/score value, yes/no, refinement detail), keep routing to SQL cutoff.
- Only route away from SQL cutoff if user clearly changes topic to fees/process/docs/exam/general info.

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


def _looks_like_non_cutoff_topic_shift(question: str) -> bool:
    q = _normalize_text(question)
    return any(
        k in q
        for k in [
            "fee",
            "fees",
            "tuition",
            "counselling process",
            "counseling process",
            "documents",
            "eligibility",
            "exam date",
            "syllabus",
            "reservation policy",
        ]
    )


def _should_continue_cutoff_from_context(question: str, cutoff_ctx: Dict[str, object]) -> bool:
    if not cutoff_ctx:
        return False
    if _looks_like_non_cutoff_topic_shift(question):
        return False
    if cutoff_ctx.get("awaiting_refinement_choice") or cutoff_ctx.get("awaiting_refinement_details"):
        return True
    if _missing_cutoff_fields(cutoff_ctx):
        return True

    q = _normalize_text(question)
    metric_type, metric_value = _extract_metric_from_text(question)
    has_state = len(_extract_states_from_text(question)) > 0
    has_category = _extract_category(question) is not None
    short_ack = q in {"yes", "yes please", "yup", "sure", "ok", "okay"}
    # Very short inputs in an active cutoff thread are usually continuation tokens.
    if short_ack or has_state or has_category or (metric_type is not None and metric_value is not None):
        return True
    return False


def _looks_like_cutoff_profile_payload(question: str) -> bool:
    q = _normalize_text(question)
    metric_type, metric_value = _extract_metric_from_text(question)
    has_metric = metric_type is not None and metric_value is not None
    has_category = _extract_category(question) is not None
    has_state = len(_extract_states_from_text(question)) > 0 or any(
        k in q for k in ["preferred state", "preffered state", "home state", "domicile"]
    )
    return has_metric and has_category and has_state


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


def _extract_states_from_text(question: str) -> List[str]:
    q = _normalize_text(question)
    # Avoid false state extraction from phrases like "follow-up".
    q = q.replace("follow-up", "followup").replace("follow up", "followup")
    states: List[str] = []
    seen = set()
    for alias, canonical in STATES.items():
        if canonical == "All-India":
            continue
        # Ignore 2-letter aliases here; they create too many false positives
        # in free text ("up" in "follow-up", etc.). Full names still work.
        if len(alias) <= 2:
            continue
        if re.search(rf"\b{re.escape(alias)}\b", q):
            if canonical not in seen:
                seen.add(canonical)
                states.append(canonical)
    return states


def _extract_home_state(question: str) -> Optional[str]:
    q = _normalize_text(question)
    m = re.search(r"(?:i am from|i'm from|my home state is|from)\s+([a-z\s&]+)", q)
    if m:
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
    return None


def _is_friend_or_general_profile_mode(question: str) -> bool:
    q = _normalize_text(question)
    return any(
        k in q
        for k in [
            "for my friend",
            "for friend",
            "for my cousin",
            "for someone",
            "not for me",
            "general query",
            "in general",
            "generic query",
        ]
    )


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


def _extract_category(question: str) -> Optional[str]:
    q = _normalize_text(question)
    if "obc" in q:
        return "OBC"
    if re.search(r"\bsc\b", q):
        return "SC"
    if re.search(r"\bst\b", q):
        return "ST"
    if "ews" in q:
        return "EWS"
    if "pwd" in q or "pwbd" in q:
        return "PwD"
    if "general" in q or "ur" in q:
        return "GENERAL"
    return None


def _missing_cutoff_fields(cutoff_ctx: Dict[str, object]) -> List[str]:
    missing: List[str] = []
    if not cutoff_ctx.get("metric_type") or not cutoff_ctx.get("metric_value"):
        missing.append("metric")
    if not cutoff_ctx.get("category"):
        missing.append("category")
    if not cutoff_ctx.get("home_state"):
        missing.append("home_state")
    if not cutoff_ctx.get("target_states"):
        missing.append("target_states")
    return missing


def _build_cutoff_followup_prompt(missing_fields: List[str]) -> str:
    if "metric" in missing_fields:
        return (
            "For cutoff prediction, please share **either your NEET score/marks or AIR rank**.\n\n"
            "Example: `Score 540` or `AIR 5400`."
        )
    if "category" in missing_fields:
        return "Please share your NEET category (General/OBC/SC/ST/EWS/PwD) so I can fetch accurate cutoff rows."
    if "home_state" in missing_fields:
        return (
            "Please tell me your **home state** (domicile state). "
            "I use this to apply domicile/non-domicile cutoff rules correctly."
        )
    if "target_states" in missing_fields:
        return (
            "Which state(s) do you want to check for colleges?\n\n"
            "You can type one or multiple states, e.g. `Bihar` or `Delhi, Haryana, Punjab`."
        )
    return "Please share a bit more detail so I can run cutoff analysis."


def _is_affirmative_reply(question: str) -> bool:
    q = _normalize_text(question)
    return q in {"yes", "yes please", "yeah", "yup", "sure", "okay", "ok", "go ahead"}


def _extract_cutoff_refinements(question: str) -> Dict[str, object]:
    q = _normalize_text(question)
    updates: Dict[str, object] = {}
    states = _extract_states_from_text(question)
    if states:
        updates["target_states"] = states

    college_type_patterns: List[str] = []
    if any(k in q for k in ["government", "govt"]):
        college_type_patterns.append("%GOV%")
    if "private" in q:
        college_type_patterns.append("%PRIVATE%")
    if "deemed" in q:
        college_type_patterns.append("%DEEMED%")
    if college_type_patterns:
        updates["college_type_patterns"] = college_type_patterns

    quota_patterns: List[str] = []
    if "aiq" in q or "all india quota" in q:
        quota_patterns.append("%AIQ%")
    if "state quota" in q:
        quota_patterns.append("%STATE%")
    if "management" in q:
        quota_patterns.append("%MANAGEMENT%")
    if "nri" in q:
        quota_patterns.append("%NRI%")
    if "open" in q:
        quota_patterns.append("%OPEN%")
    if quota_patterns:
        updates["quota_patterns"] = quota_patterns

    return updates

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


def assess_kb_sufficiency_with_llm(client, user_question: str, kb_tool_result: str) -> tuple[bool, str]:
    """
    LLM-based generic sufficiency check.
    Returns (is_sufficient, reason).
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict retrieval sufficiency evaluator for NEET counselling queries.\n"
                        "Given a user question and retrieved knowledge-base content, decide if the KB content "
                        "is enough to answer accurately without assumptions.\n"
                        "Judge ONLY against the exact user ask.\n"
                        "Do NOT mark insufficient just because broader background is missing "
                        "(for example geography/history when the user asked only fee/cutoff/process).\n"
                        "If retrieved text contains the requested data-type and correct entity/scope, mark sufficient.\n"
                        "Return ONLY valid JSON with keys: is_sufficient (boolean), reason (string).\n"
                        "Mark is_sufficient=false if exact requested entity/detail is missing, ambiguous, "
                        "or only similar entities are present."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"USER_QUESTION:\n{user_question}\n\n"
                        f"KB_RESULT:\n{kb_tool_result}\n\n"
                        "Return JSON only."
                    ),
                },
            ],
            temperature=0,
            max_tokens=120,
        )
        raw = (response.choices[0].message.content or "").strip()
        parsed = json.loads(raw)
        is_sufficient = bool(parsed.get("is_sufficient", False))
        reason = str(parsed.get("reason", "")).strip() or "No reason provided"
        return is_sufficient, reason
    except Exception as err:
        log(f"[V2] ⚠️ KB sufficiency check failed, defaulting to insufficient: {err}")
        return False, "Sufficiency check failed"


def is_query_in_domain(question: str) -> bool:
    """
    Smart domain check using LLM to understand context.
    Quick rejection for obviously off-topic queries to save LLM cost.
    """
    question_lower = question.lower()
    
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
                    "content": f"Is this query related to the education counselling domain?\n\nQuery: {question}"
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
                    await asyncio.sleep(0.01)
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
                        await asyncio.sleep(0.01)
                    
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
            
            # ========== GUARDRAILS: Domain restriction ==========
            # Only check domain if FAQ didn't match
            if not is_query_in_domain(request.question):
                log("[WARN] ❌ OUT OF DOMAIN - Query rejected")
                yield f"data: {json.dumps({'type': 'sources', 'sources': []})}\n\n"
                for token in sse_tokens_preserving_formatting(OUT_OF_DOMAIN_RESPONSE):
                    yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"
                    await asyncio.sleep(0.02)
                yield f"data: {json.dumps({'type': 'done', 'out_of_domain': True})}\n\n"
                return
            
            log("[INFO] ✅ DOMAIN CHECK: Query is in domain")
            log(f"[INFO] 👤 USER STATE: {user_state or 'Not set'}")
            
            # ========== EXTRACT CONVERSATION CONTEXT FOR ROUTING ==========
            conversation_context = None
            if conversation_memory:
                conversation_context = conversation_memory.extract_conversation_context()
                if conversation_context.get("detected_state"):
                    log(f"[INFO] 📝 Conversation context: state={conversation_context.get('detected_state')}, topic={conversation_context.get('detected_topic')}")
            
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

CONVERSATION CONTINUITY:
- The user may ask follow-up questions that refer to previous context.
- If conversation history is provided, understand the ONGOING topic and state/region being discussed.
- "What about ST category?" after discussing J&K fees means the user wants J&K ST category info, NOT a different state.
- Always maintain context from previous messages when answering follow-ups.

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
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
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

@app.post("/chat/v2/stream")
async def chat_v2_stream(request: ChatRequest):
    """
    Unified chat endpoint with tool-based architecture.
    
    Uses a single master prompt that:
    - Handles intent classification internally
    - Decides when to search the knowledge base
    - Asks for clarification when truly needed
    - Generates accurate, concise responses
    
    The LLM has access to a search_knowledge_base tool with optional state filter only.
    """
    from openai import OpenAI as OpenAIClient
    from services.unified_prompt import get_system_prompt, get_tools
    from services.knowledge_tool import execute_tool_call, format_search_results_for_llm
    from services.conversation_memory import (
        ConversationMemory,
        get_or_create_conversation,
    )
    from database.connection import async_session_maker
    
    log(f"\n{'='*60}")
    log(f"[V2] 📥 NEW QUERY: {request.question[:100]}...")
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
        
        try:
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
            qnorm = _normalize_text(request.question)
            if _is_session_close_intent(request.question):
                med_ctx["stage"] = "closing"
                med_ctx["last_activity_at"] = datetime.utcnow().isoformat()
                close_msg = _build_session_close_message(med_ctx)
                yield f"data: {json.dumps({'type': 'suggested_replies', 'replies': ['Start new query', 'Compare fees', 'Counselling process']})}\n\n"
                for token in sse_tokens_preserving_formatting(close_msg):
                    yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"
                    await asyncio.sleep(0.01)
                yield f"data: {json.dumps({'type': 'done', 'conversation_id': conversation_id})}\n\n"
                if request.user_id and conversation_id:
                    asyncio.create_task(v2_background_update_conversation_context(conversation_id, med_ctx))
                return

            if _is_greeting_only(request.question):
                welcome = _first_visit_welcome_message() if is_first_visit else _return_visit_welcome_message()
                med_ctx["stage"] = "first_visit" if is_first_visit else "returning"
                med_ctx["last_activity_at"] = datetime.utcnow().isoformat()
                yield f"data: {json.dumps({'type': 'suggested_replies', 'replies': MEDBUDDY_DEFAULT_REPLIES})}\n\n"
                for token in sse_tokens_preserving_formatting(welcome):
                    yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"
                    await asyncio.sleep(0.01)
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
                return

            # Apply lightweight onboarding updates when user answers guided prompts.
            onboarding = dict(med_ctx.get("onboarding") or {})
            onboarding.update(_extract_onboarding_updates(request.question))
            med_ctx["onboarding"] = onboarding
            med_ctx["last_activity_at"] = datetime.utcnow().isoformat()

            # ========== CUTOFF SQL PATH (TABLE-BASED, NO VECTOR SEARCH) ==========
            cutoff_ctx_peek = dict(med_ctx.get("cutoff") or {})
            cutoff_triggered = False
            if MEDBUDDY_CAPS["cutoff"]:
                if _should_continue_cutoff_from_context(request.question, cutoff_ctx_peek):
                    cutoff_triggered = True
                    log("[V2] 🧭 Continuing cutoff refinement flow from conversation context")
                if cutoff_triggered:
                    llm_route_decision = True
                else:
                    llm_route_decision = _llm_should_route_cutoff_sql(request.question, med_ctx)
                if llm_route_decision is None:
                    # Safety fallback only if LLM classifier fails.
                    in_shortlist_context = str(med_ctx.get("last_topic") or "").lower() in {
                        "cutoff analysis",
                        "college shortlist",
                    }
                    metric_hint, _metric_value_hint = _extract_metric_from_text(request.question)
                    cutoff_triggered = (
                        _is_cutoff_intent(request.question)
                        or _looks_like_cutoff_profile_payload(request.question)
                        or (in_shortlist_context and metric_hint is not None)
                    )
                    log(f"[V2] 🧭 Cutoff route fallback decision: {cutoff_triggered}")
                else:
                    cutoff_triggered = llm_route_decision
            if cutoff_triggered:
                from services.cutoff_service import fetch_cutoff_recommendations, format_cutoff_markdown
                log("[V2] 🎯 Routing to CUTOFF SQL path")

                cutoff_ctx = dict(med_ctx.get("cutoff") or {})
                profile_mode = str(cutoff_ctx.get("profile_mode") or "self")
                if _is_friend_or_general_profile_mode(request.question):
                    profile_mode = "friend_or_general"
                cutoff_ctx["profile_mode"] = profile_mode
                if cutoff_ctx.get("awaiting_refinement_choice") and _is_affirmative_reply(request.question):
                    followup = (
                        "Sure — happy to refine this.\n\n"
                        "Tell me what you want to refine:\n"
                        "- College type (Government / Private / Deemed)\n"
                        "- Quota (AIQ / State / Management / NRI / Open)\n"
                        "- Specific state(s)\n\n"
                        "Example: `Government colleges in Bihar with State quota`."
                    )
                    cutoff_ctx["awaiting_refinement_choice"] = False
                    cutoff_ctx["awaiting_refinement_details"] = True
                    med_ctx["cutoff"] = cutoff_ctx
                    for token in sse_tokens_preserving_formatting(followup):
                        yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"
                        await asyncio.sleep(0.01)
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
                    return

                detected_states = _extract_states_from_text(request.question)
                metric_type, metric_value = _extract_metric_from_text(request.question)
                category = _extract_category(request.question) or cutoff_ctx.get("category") or onboarding.get("category")
                explicit_home_state = _extract_home_state(request.question)
                home_state = explicit_home_state or cutoff_ctx.get("home_state")

                # Default to registered user home state only for self-mode.
                if not home_state and profile_mode != "friend_or_general":
                    registered_home_state = await _get_registered_home_state(request.user_id)
                    if registered_home_state:
                        home_state = registered_home_state
                    elif request.user_preferences and request.user_preferences.preferred_state:
                        home_state = request.user_preferences.preferred_state

                # Target states must be explicit from user message or previously provided in this
                # cutoff context. Never auto-inject home/profile state into target_states.
                if detected_states:
                    target_states = detected_states
                else:
                    target_states = list(cutoff_ctx.get("target_states") or [])

                if not category and request.user_preferences and request.user_preferences.category:
                    category = request.user_preferences.category

                if metric_type and metric_value:
                    cutoff_ctx["metric_type"] = metric_type
                    cutoff_ctx["metric_value"] = int(metric_value)
                if category:
                    cutoff_ctx["category"] = str(category).upper()
                if home_state:
                    cutoff_ctx["home_state"] = str(home_state)
                if target_states:
                    cutoff_ctx["target_states"] = target_states

                # Apply optional refinement filters from user's follow-up details.
                refinement_updates = _extract_cutoff_refinements(request.question)
                if refinement_updates.get("target_states"):
                    cutoff_ctx["target_states"] = refinement_updates["target_states"]
                if refinement_updates.get("college_type_patterns"):
                    cutoff_ctx["college_type_patterns"] = refinement_updates["college_type_patterns"]
                if refinement_updates.get("quota_patterns"):
                    cutoff_ctx["quota_patterns"] = refinement_updates["quota_patterns"]
                if cutoff_ctx.get("awaiting_refinement_details"):
                    cutoff_ctx["awaiting_refinement_details"] = False

                med_ctx["cutoff"] = cutoff_ctx
                missing = _missing_cutoff_fields(cutoff_ctx)

                if missing:
                    followup = _build_cutoff_followup_prompt(missing)
                    for token in sse_tokens_preserving_formatting(followup):
                        yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"
                        await asyncio.sleep(0.01)
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
                    return

                metric_type = str(cutoff_ctx.get("metric_type"))
                metric_value = int(cutoff_ctx.get("metric_value"))
                home_state = str(cutoff_ctx.get("home_state"))
                category = str(cutoff_ctx.get("category"))
                target_states = [str(s) for s in list(cutoff_ctx.get("target_states") or [])]
                college_type_patterns = cutoff_ctx.get("college_type_patterns")
                quota_patterns = cutoff_ctx.get("quota_patterns")

                cutoff_result_limit = await get_cutoff_result_limit()
                rows = await fetch_cutoff_recommendations(
                    metric_type=metric_type,
                    metric_value=metric_value,
                    home_state=home_state,
                    target_states=target_states,
                    category=category,
                    college_type_patterns=college_type_patterns if isinstance(college_type_patterns, list) else None,
                    quota_patterns=quota_patterns if isinstance(quota_patterns, list) else None,
                    total_limit=cutoff_result_limit,
                )
                cutoff_answer = format_cutoff_markdown(
                    rows=rows,
                    metric_type=metric_type,
                    metric_value=metric_value,
                    category=category,
                    home_state=home_state,
                    target_states=target_states,
                    display_limit=cutoff_result_limit,
                )
                source = {
                    "file_name": "neet_ug_2025_cutoffs",
                    "document_type": "sql_cutoff_table",
                    "state": ", ".join(target_states),
                    "text_snippet": f"Cutoff rows matched: {len(rows)}",
                }
                yield f"data: {json.dumps({'type': 'sources', 'sources': [source]})}\n\n"
                for token in sse_tokens_preserving_formatting(cutoff_answer):
                    yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"
                    await asyncio.sleep(0.01)

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

                should_generate_title = (
                    request.user_id
                    and conversation_id
                    and not _is_greeting_only(request.question)
                    and await _v2_conversation_needs_title(conversation_id)
                )
                if should_generate_title:
                    from services.conversation_memory import generate_conversation_title, update_conversation_title
                    try:
                        generated_title = await generate_conversation_title(request.question)
                        async with async_session_maker() as title_db:
                            await update_conversation_title(title_db, conversation_id, generated_title)
                        yield f"data: {json.dumps({'type': 'title', 'title': generated_title, 'conversation_id': conversation_id})}\n\n"
                    except Exception as title_err:
                        log(f"[V2] ⚠️ Title generation error (cutoff path): {title_err}")
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
                    
                    yield f"data: {json.dumps({'type': 'suggested_replies', 'replies': MEDBUDDY_DEFAULT_REPLIES})}\n\n"
                    for token in sse_tokens_preserving_formatting(faq_answer):
                        yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"
                        await asyncio.sleep(0.01)
                    
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
                    
                    # Generate title AFTER done event (non-blocking for user).
                    # Also handle conversations that started with greeting-only turns.
                    should_generate_title = (
                        request.user_id
                        and conversation_id
                        and not _is_greeting_only(request.question)
                        and await _v2_conversation_needs_title(conversation_id)
                    )
                    if should_generate_title:
                        from services.conversation_memory import generate_conversation_title, update_conversation_title
                        try:
                            generated_title = await generate_conversation_title(request.question)
                            async with async_session_maker() as title_db:
                                await update_conversation_title(title_db, conversation_id, generated_title)
                            log(f"[V2] 🏷️ Generated title: {generated_title}")
                            # Send title as separate event
                            yield f"data: {json.dumps({'type': 'title', 'title': generated_title, 'conversation_id': conversation_id})}\n\n"
                        except Exception as title_err:
                            log(f"[V2] ⚠️ Title generation error: {title_err}")
                    
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
            if not web_fallback_enabled:
                available_tools = [
                    t for t in available_tools
                    if t.get("function", {}).get("name") != "search_web"
                ]
            
            messages = [{"role": "system", "content": get_system_prompt()}]
            messages.append({
                "role": "system",
                "content": (
                    "Runtime tool availability: "
                    + ("`search_web` is ENABLED." if web_fallback_enabled else "`search_web` is DISABLED.")
                    + " For factual queries, use `search_knowledge_base` first. "
                    + "If KB is insufficient and web tool is enabled, call `search_web`."
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
            
            log(f"[V2] 🤖 Calling LLM with {len(messages)} messages...")
            
            # ========== TOOL LOOP (LLM decides if/when to call KB then web) ==========
            used_web_fallback = False
            max_tool_rounds = 3
            assistant_message = None
            kb_attempted = False
            force_web_search_next_round = False
            kb_insufficient_and_web_disabled = False
            kb_marked_insufficient = False
            web_only_messages = None
            forced_fallback_response = None

            for round_idx in range(max_tool_rounds):
                log(f"[V2] 🤖 Tool round {round_idx + 1}/{max_tool_rounds}")
                # Enforce KB-first policy at runtime: round-1 only allows KB tool.
                round_tools = available_tools
                tool_choice_mode = "auto"
                if force_web_search_next_round and web_fallback_enabled:
                    round_tools = [
                        t for t in available_tools
                        if t.get("function", {}).get("name") == "search_web"
                    ]
                    # If KB is marked insufficient and web is enabled, force a web lookup.
                    tool_choice_mode = "required"
                elif round_idx == 0:
                    round_tools = [
                        t for t in available_tools
                        if t.get("function", {}).get("name") == "search_knowledge_base"
                    ]
                t_llm = time.perf_counter()
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    tools=round_tools,
                    tool_choice=tool_choice_mode,
                    temperature=0.3,
                    max_tokens=1500
                )
                llm_round_ms = _elapsed_ms(t_llm)
                assistant_message = response.choices[0].message

                if not assistant_message.tool_calls:
                    # Runtime safety net:
                    # If the model skips tools on a NEET factual query, force a KB lookup
                    # so we can still run normal KB/web fallback logic.
                    if (
                        round_idx == 0
                        and _looks_like_neet_factual_query(request.question)
                        and not kb_attempted
                    ):
                        forced_args: Dict[str, str] = {"query": request.question}
                        if user_state:
                            forced_args["state"] = user_state
                        log(f"[V2] 🛟 Forced KB lookup (tool skipped by LLM): {forced_args}")
                        t_tool = time.perf_counter()
                        tool_result, success = execute_tool_call("search_knowledge_base", forced_args)
                        tool_exec_ms = _elapsed_ms(t_tool)
                        suff_ms = 0.0
                        kb_attempted = True
                        if success:
                            t_suff = time.perf_counter()
                            is_sufficient, reason = assess_kb_sufficiency_with_llm(
                                client=client,
                                user_question=request.question,
                                kb_tool_result=tool_result,
                            )
                            suff_ms = _elapsed_ms(t_suff)
                            log(f"[V2] 🧪 KB sufficiency (forced): {is_sufficient} | reason: {reason}")
                            if not is_sufficient and web_fallback_enabled:
                                force_web_search_next_round = True
                                kb_marked_insufficient = True
                            elif not is_sufficient and not web_fallback_enabled:
                                kb_insufficient_and_web_disabled = True
                                kb_marked_insufficient = True
                        else:
                            log(f"[V2] ⚠️ Forced KB lookup error: {tool_result[:120]}")

                        round_stats.append(
                            {
                                "i": round_idx + 1,
                                "llm": llm_round_ms,
                                "tool": "forced_search_knowledge_base",
                                "tool_exec": tool_exec_ms,
                                "suff": suff_ms,
                            }
                        )

                        messages.append({
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [{
                                "id": "forced_kb_lookup",
                                "type": "function",
                                "function": {
                                    "name": "search_knowledge_base",
                                    "arguments": json.dumps(forced_args),
                                }
                            }]
                        })
                        messages.append({
                            "role": "tool",
                            "tool_call_id": "forced_kb_lookup",
                            "content": tool_result,
                        })
                        # Continue loop so final response generation can use tool context.
                        continue

                    round_stats.append(
                        {
                            "i": round_idx + 1,
                            "llm": llm_round_ms,
                            "tool": None,
                            "tool_exec": 0.0,
                            "suff": 0.0,
                        }
                    )
                    break

                tool_call = assistant_message.tool_calls[0]
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments or "{}")
                log(f"[V2] 🔧 TOOL CALL: {tool_name}")
                log(f"[V2]    Args: {tool_args}")

                t_tool = time.perf_counter()
                tool_result, success = execute_tool_call(tool_name, tool_args)
                tool_exec_ms = _elapsed_ms(t_tool)
                if success:
                    log("[V2] ✅ Tool returned results")
                else:
                    log(f"[V2] ⚠️ Tool error: {tool_result[:120]}")

                suff_ms = 0.0
                if tool_name == "search_knowledge_base":
                    kb_attempted = True
                    t_suff = time.perf_counter()
                    is_sufficient, reason = assess_kb_sufficiency_with_llm(
                        client=client,
                        user_question=request.question,
                        kb_tool_result=tool_result
                    )
                    # Generic guardrails for sufficiency decisions (not fee-only):
                    # 1) If query is entity-specific and entity is missing in KB text, force insufficient.
                    # 2) If LLM marked insufficient but KB has strong structured signals for non-entity asks,
                    #    allow a cautious sufficient override to reduce unnecessary web fallback.
                    qn = _normalize_text(request.question)
                    kb_txt = _normalize_text(tool_result)
                    entity_hint = _extract_entity_hint(request.question)
                    is_entity_query = _is_entity_specific_query(request.question)
                    entity_present_in_kb = True if not entity_hint else (entity_hint in kb_txt)
                    kb_has_multiple_hits = ("[1]" in tool_result or "[2]" in tool_result)
                    kb_has_structured_signals = _has_structured_data_signals(tool_result)

                    if is_entity_query and not entity_present_in_kb:
                        is_sufficient = False
                        reason = "Entity-specific query but exact entity not found in KB evidence"
                    elif (not is_entity_query) and kb_has_multiple_hits and kb_has_structured_signals and not is_sufficient:
                        is_sufficient = True
                        reason = "Heuristic override: non-entity query with strong structured KB signals"
                    suff_ms = _elapsed_ms(t_suff)
                    log(f"[V2] 🧪 KB sufficiency: {is_sufficient} | reason: {reason}")
                    if not is_sufficient:
                        kb_marked_insufficient = True
                        if web_fallback_enabled:
                            force_web_search_next_round = True
                        else:
                            kb_insufficient_and_web_disabled = True
                round_stats.append(
                    {
                        "i": round_idx + 1,
                        "llm": llm_round_ms,
                        "tool": tool_name,
                        "tool_exec": tool_exec_ms,
                        "suff": suff_ms,
                    }
                )
                if tool_name == "search_web":
                    used_web_fallback = True
                    force_web_search_next_round = False

                # Emit sources for frontend when possible
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

                if sources:
                    yield f"data: {json.dumps({'type': 'sources', 'sources': sources[:5]})}\n\n"
                    log(f"[V2] 📚 Parsed {len(sources)} source references from {tool_name}")
                    for idx, src in enumerate(sources[:5], 1):
                        log(
                            f"[V2]    Source {idx}: "
                            f"file={src.get('file_name', 'Unknown')} | "
                            f"page={src.get('page', 'N/A')} | "
                            f"state={src.get('state', 'N/A')} | "
                            f"type={src.get('document_type', 'N/A')}"
                        )

                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": tool_call.function.arguments
                        }
                    }]
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result
                })

                if kb_insufficient_and_web_disabled:
                    forced_fallback_response = (
                        "I want to make sure you get accurate details.\n\n"
                        "Right now, I do not have this exact information in the current knowledge base. "
                        "Please verify from official MCC/state counselling websites for the latest confirmed values."
                    )
                    messages.append({
                        "role": "system",
                        "content": (
                            "KB sufficiency flag is FALSE and web search is disabled. "
                            "Do not provide assumptions or transferred values. "
                            "Respond politely that this specific information is not available in the knowledge base "
                            "and suggest checking official NTA/state counselling websites."
                        )
                    })
                    break

                # When KB is insufficient and web is used, explicitly prevent leakage of KB values.
                if tool_name == "search_web" and kb_marked_insufficient:
                    # Build a web-only context for final answer generation (no KB tool outputs).
                    web_only_messages = [
                        {"role": "system", "content": get_system_prompt()},
                        {
                            "role": "system",
                            "content": (
                                "Final answer mode: WEB-ONLY.\n"
                                "Knowledge-base retrieval was marked insufficient.\n"
                                "Use ONLY the latest web_search tool output as evidence.\n"
                                "Do NOT use any numbers/details from knowledge-base chunks.\n"
                                "If web snippets do not confirm exact values, clearly say not confirmed."
                            )
                        },
                        {"role": "user", "content": request.question},
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [{
                                "id": tool_call.id,
                                "type": "function",
                                "function": {
                                    "name": tool_name,
                                    "arguments": tool_call.function.arguments
                                }
                            }]
                        },
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": tool_result
                        },
                    ]
                    messages.append({
                        "role": "system",
                        "content": (
                            "CRITICAL: Earlier knowledge-base chunks were marked INSUFFICIENT for this question. "
                            "Do NOT use any numeric values, fees, or entity details from KB chunks in final answer. "
                            "Use only web_search evidence from this round. "
                            "If web snippets still lack exact numeric details, clearly say the exact fee is not confirmed."
                        )
                    })

            # ========== FINAL RESPONSE ==========
            if forced_fallback_response:
                full_response = _apply_response_policy(forced_fallback_response, request.question)
                t_out = time.perf_counter()
                _first_out = True
                yield f"data: {json.dumps({'type': 'suggested_replies', 'replies': ['Show more options', 'Compare colleges', 'Try another state']})}\n\n"
                for token in sse_tokens_preserving_formatting(full_response):
                    if _first_out:
                        final_ttft_ms = _elapsed_ms(t_out)
                        _first_out = False
                    yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"
                    await asyncio.sleep(0.01)
                final_stream_ms = _elapsed_ms(t_out)
            elif assistant_message and not assistant_message.tool_calls and (assistant_message.content or "").strip():
                full_response = _apply_response_policy(assistant_message.content or "", request.question)
                t_out = time.perf_counter()
                _first_out = True
                yield f"data: {json.dumps({'type': 'suggested_replies', 'replies': MEDBUDDY_DEFAULT_REPLIES})}\n\n"
                for token in sse_tokens_preserving_formatting(full_response):
                    if _first_out:
                        final_ttft_ms = _elapsed_ms(t_out)
                        _first_out = False
                    yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"
                    await asyncio.sleep(0.01)
                final_stream_ms = _elapsed_ms(t_out)
            else:
                log("[V2] 🤖 Generating final response with context...")
                final_messages = web_only_messages if (used_web_fallback and web_only_messages) else messages
                t_out = time.perf_counter()
                final_response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=final_messages,
                    temperature=0.3,
                    max_tokens=1500
                )
                full_response = final_response.choices[0].message.content or ""
                full_response = _apply_response_policy(full_response, request.question)
                _first_out = True
                yield f"data: {json.dumps({'type': 'suggested_replies', 'replies': MEDBUDDY_DEFAULT_REPLIES})}\n\n"
                for token in sse_tokens_preserving_formatting(full_response):
                    if _first_out:
                        final_ttft_ms = _elapsed_ms(t_out)
                        _first_out = False
                    yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"
                final_stream_ms = _elapsed_ms(t_out)

            if used_web_fallback:
                yield f"data: {json.dumps({'type': 'meta', 'web_fallback_used': True})}\n\n"
                log("[V2] 🌐 Final response generated with web fallback context")
            else:
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
                parts.append(f"answer_stream_total={final_stream_ms:.0f}ms")
                parts.append("save_db=deferred(background)")
                log(
                    "[V2] ⏱ TIMING SUMMARY (V2_TIMING_LOG=false to hide) → "
                    + " ".join(parts)
                )
            log(f"{'='*60}")
            
            # Generate title AFTER done event (non-blocking for user).
            # Also handle conversations that started with greeting-only turns.
            should_generate_title = (
                request.user_id
                and conversation_id
                and not _is_greeting_only(request.question)
                and await _v2_conversation_needs_title(conversation_id)
            )
            if should_generate_title:
                from services.conversation_memory import generate_conversation_title, update_conversation_title
                try:
                    generated_title = await generate_conversation_title(request.question)
                    async with async_session_maker() as title_db:
                        await update_conversation_title(title_db, conversation_id, generated_title)
                    log(f"[V2] 🏷️ Generated title: {generated_title}")
                    # Send title as separate event
                    yield f"data: {json.dumps({'type': 'title', 'title': generated_title, 'conversation_id': conversation_id})}\n\n"
                except Exception as title_err:
                    log(f"[V2] ⚠️ Title generation error: {title_err}")
            
        except Exception as e:
            import traceback
            log(f"[V2] ❌ ERROR: {e}")
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
