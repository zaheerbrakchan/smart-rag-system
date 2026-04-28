"""
Query Router Service
Classifies user query intent before applying vector-store metadata filters
"""

import os
import sys
import re
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from dataclasses import dataclass
from openai import OpenAI

# Use uvicorn's logger
uvicorn_logger = logging.getLogger("uvicorn.error")

def log(msg):
    uvicorn_logger.info(msg)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ============== INTENT DEFINITIONS ==============

class QueryIntent(Enum):
    """Four types of query intents"""
    EXAM_INFO = "exam_info"              # Syllabus, pattern, dates - NTA bulletin only
    CENTRAL_COUNSELLING = "central_counselling"  # MCC, AIQ, AIIMS - NTA bulletin only
    STATE_COUNSELLING = "state_counselling"      # State specific - State brochure only
    MIXED = "mixed"                      # Could be both central and state level
    NEEDS_CLARIFICATION = "needs_clarification"  # Ambiguous - ask user to clarify


@dataclass
class QueryRouting:
    """Result of query classification"""
    intent: QueryIntent
    detected_state: Optional[str]       # State explicitly mentioned in query
    use_user_preference: bool           # Whether to use user's registered state
    categories_to_search: List[str]     # Specific categories to filter by
    confidence: float                   # How confident we are (0-1)
    needs_clarification: bool = False   # Whether we need user to clarify
    clarification_options: List[str] = None  # Options to show user


# Keywords that CLEARLY indicate central-only queries (no ambiguity)
CLEARLY_CENTRAL_KEYWORDS = [
    "nta", "neet exam", "syllabus", "pattern", "admit card", "hall ticket",
    "application form", "registration", "exam date", "exam center",
    "mcc", "all india quota", "aiq", "15% quota", "central counselling",
    "aiims", "jipmer", "deemed university", "central university"
]

# Keywords that suggest STATE-SPECIFIC info (ambiguous without state mention)
AMBIGUOUS_STATE_KEYWORDS = [
    "reservation", "seat matrix", "counselling", "counseling", "fee structure",
    "cutoff", "cut-off", "rank required", "closing rank", "seat", "quota",
    "eligibility", "document", "process", "round", "mop up", "college"
]


# ============== KEYWORD-BASED ROUTING ==============

# Keywords that indicate EXAM_INFO intent (NTA bulletin, no state needed)
EXAM_INFO_KEYWORDS = [
    "syllabus", "pattern", "exam pattern", "marking scheme", "negative marking",
    "how many questions", "total marks", "duration", "admit card", "hall ticket",
    "exam date", "neet date", "exam time", "exam center", "test center",
    "nta", "application form", "apply for neet", "registration",
    "eligibility for neet", "who can give neet", "age limit for neet",
    "attempt limit", "how to fill", "application fee", "correction window",
    "what is neet", "about neet", "neet exam"
]

# Keywords that indicate CENTRAL_COUNSELLING intent
CENTRAL_COUNSELLING_KEYWORDS = [
    "mcc", "medical counselling committee", "all india quota", "aiq", "15% quota",
    "aiims", "jipmer", "bhu", "amu", "du medical", "central university",
    "deemed university", "esic", "afmc", "central counselling",
    "central pool", "national level counselling"
]

# Keywords that indicate STATE_COUNSELLING intent
STATE_COUNSELLING_KEYWORDS = [
    "state quota", "85%", "state counselling", "state counseling",
    "state seat", "college in", "medical college", "government college",
    "private college", "management quota", "nri quota", "fee in",
    "seat matrix", "how many seats", "counselling round", "round 1", "round 2",
    "mop up", "document verification", "reporting", "domicile",
    "state reservation", "local quota"
]

# Keywords that suggest MIXED intent (both central and state info needed)
MIXED_KEYWORDS = [
    "reservation", "obc reservation", "sc reservation", "st reservation",
    "ews reservation", "category wise", "quota", "reserved seats"
]

# State name mappings
STATES = {
    "karnataka": "Karnataka",
    "kerala": "Kerala",
    "tamil nadu": "Tamil Nadu",
    "tamilnadu": "Tamil Nadu",
    "andhra pradesh": "Andhra Pradesh",
    "andhra": "Andhra Pradesh",
    "ap": "Andhra Pradesh",
    "telangana": "Telangana",
    "maharashtra": "Maharashtra",
    "gujarat": "Gujarat",
    "rajasthan": "Rajasthan",
    "madhya pradesh": "Madhya Pradesh",
    "mp": "Madhya Pradesh",
    "uttar pradesh": "Uttar Pradesh",
    "up": "Uttar Pradesh",
    "bihar": "Bihar",
    "west bengal": "West Bengal",
    "bengal": "West Bengal",
    "wb": "West Bengal",
    "odisha": "Odisha",
    "assam": "Assam",
    "punjab": "Punjab",
    "haryana": "Haryana",
    "delhi": "Delhi",
    "jharkhand": "Jharkhand",
    "chhattisgarh": "Chhattisgarh",
    "uttarakhand": "Uttarakhand",
    "himachal": "Himachal Pradesh",
    "hp": "Himachal Pradesh",
    "goa": "Goa",
    "jammu": "Jammu & Kashmir",
    "kashmir": "Jammu & Kashmir",
    "j&k": "Jammu & Kashmir",
    "jk": "Jammu & Kashmir",
    "j & k": "Jammu & Kashmir",
    "puducherry": "Puducherry",
    "pondicherry": "Puducherry",
    "all-india": "All-India",
    "all india": "All-India",
    # Union Territories (names + common short forms)
    "chandigarh": "Chandigarh",
    "ladakh": "Ladakh",
    "lakshadweep": "Lakshadweep",
    "andaman and nicobar islands": "Andaman and Nicobar Islands",
    "andaman": "Andaman and Nicobar Islands",
    "dadra and nagar haveli and daman and diu": "Dadra and Nagar Haveli and Daman and Diu",
    "dadra and nagar haveli": "Dadra and Nagar Haveli and Daman and Diu",
    "daman and diu": "Dadra and Nagar Haveli and Daman and Diu",
    "daman": "Dadra and Nagar Haveli and Daman and Diu",
    "dnhdd": "Dadra and Nagar Haveli and Daman and Diu",
    # Northeast + other states (full names + common short forms)
    "arunachal pradesh": "Arunachal Pradesh",
    "arunachal": "Arunachal Pradesh",
    "manipur": "Manipur",
    "meghalaya": "Meghalaya",
    "mizoram": "Mizoram",
    "nagaland": "Nagaland",
    "sikkim": "Sikkim",
    "tripura": "Tripura",
}

# Query expansion - abbreviations to expand for better semantic search
# Split into SAFE (always expand) and AMBIGUOUS (only expand with context)
SAFE_EXPANSIONS = {
    # These are unambiguous - always safe to expand
    r"\bj&k\b": "Jammu and Kashmir",
    r"\bjk\b": "Jammu and Kashmir", 
    r"\bj & k\b": "Jammu and Kashmir",
    r"\bj and k\b": "Jammu and Kashmir",
}

AMBIGUOUS_EXPANSIONS = {
    # These could mean other things - only expand with state context
    r"\bmp\b": "Madhya Pradesh",      # could be "MP3", "member of parliament"
    r"\bup\b": "Uttar Pradesh",       # could be "up" (direction)
    r"\bap\b": "Andhra Pradesh",      # could be "AP exam", "application"
    r"\bwb\b": "West Bengal",         # could be "web"
    r"\bhp\b": "Himachal Pradesh",    # could be "HP laptop", "horsepower"
    r"\buk\b": "Uttarakhand",         # could be "UK" (United Kingdom)
    r"\bhr\b": "Haryana",             # could be "HR" (human resources)
}

# Context keywords that suggest we're talking about Indian states
STATE_CONTEXT_KEYWORDS = [
    "eligibility", "counselling", "counseling", "quota", "state", "neet",
    "admission", "reservation", "domicile", "medical", "college", "seat",
    "cutoff", "cut-off", "rank", "allotment", "mbbs", "bds", "aiims",
    "government", "private", "deemed", "category", "obc", "sc", "st",
    "ews", "general", "pwd", "ug", "pg", "mop-up", "stray"
]

def expand_query(query: str) -> str:
    """
    Expand abbreviations in query for better semantic search.
    - Safe abbreviations (j&k) are always expanded
    - Ambiguous ones (ap, up, mp) only expand if state context keywords present
    """
    expanded = query
    
    # Always expand safe abbreviations
    for pattern, replacement in SAFE_EXPANSIONS.items():
        expanded = re.sub(pattern, replacement, expanded, flags=re.IGNORECASE)
    
    # Check if query has state-related context
    query_lower = query.lower()
    has_state_context = any(kw in query_lower for kw in STATE_CONTEXT_KEYWORDS)
    
    if has_state_context:
        # Only expand ambiguous abbreviations if we have state context
        for pattern, replacement in AMBIGUOUS_EXPANSIONS.items():
            expanded = re.sub(pattern, replacement, expanded, flags=re.IGNORECASE)
    
    if expanded != query:
        log(f"   🔄 Query expanded: '{query}' -> '{expanded}'")
    
    return expanded


def detect_state_in_query(query: str) -> Optional[str]:
    """Detect if a specific state is mentioned in the query using word boundaries"""
    query_lower = query.lower()
    for key, value in STATES.items():
        # Use word boundary regex to avoid false matches like 'mp' in 'important'
        pattern = r'\b' + re.escape(key) + r'\b'
        if re.search(pattern, query_lower):
            return value
    return None


def _canonicalize_state_string(raw: str) -> Optional[str]:
    """Map LLM / fuzzy text to a STATES canonical value."""
    if not raw or raw.strip().upper() == "NONE":
        return None
    cleaned = raw.strip()
    cl = cleaned.lower()
    if cl in STATES:
        return STATES[cl]
    for key, val in STATES.items():
        if val.lower() == cl:
            return val
    # Substring on full state names
    for canonical in sorted(set(STATES.values()), key=len, reverse=True):
        if canonical.lower() in cl or cl in canonical.lower():
            return canonical
    return None


def _parse_json_from_llm_response(text: str) -> Optional[dict]:
    """Extract JSON object from model output (plain or fenced)."""
    raw = text.strip()
    if "```" in raw:
        m = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
        if m:
            raw = m.group(1).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return None


def resolve_routing_filters_llm(query: str, user_state: Optional[str] = None) -> Optional[dict]:
    """
    Single LLM call: infer retrieval scope and state. The application does NOT map cities or
    colleges to states in code—you must infer geography (city → state, college → state, etc.)
    yourself and output the closest matching `inferred_state` from the allowed list.
    """
    allowed_states = sorted(set(STATES.values()) | {"All-India"})
    states_csv = ", ".join(allowed_states)

    prompt = f"""You route questions for a NEET UG counselling RAG assistant.

HOW OUR INDEX IS STORED (each chunk has metadata):
- state: which state/UT the PDF belongs to, or "All-India" for all-India NTA material
- document_type: nta_bulletin | state_counseling | college_info | cutoffs | faq | other
- doc_topic: file-level tag (e.g. general, fees, eligibility, cutoff, process, documents, colleges, dates)
- chunk-level labels describe paragraph topics

YOUR JOB — infer routing from the user question alone:
- Users may name **cities**, **colleges**, **AIIMS/GMC/institutes**, or **local areas** without ever saying the state. You must use **your own knowledge** of Indian geography and where institutions are located to choose the correct **inferred_state** from the allowed list (e.g. a city in Bihar → Bihar; a college campus → that state/UT).
- If the question is clearly about **All India / MCC / AIQ / national NTA** material only, use scope central_level or exam_nta and set inferred_state to null or "All-India" as appropriate.
- Do not assume the app will guess state from city names in code—**you** provide inferred_state.

User question: "{query}"
User profile state (weak hint only; use when it aligns with the question): {user_state or "not provided"}

Allowed values for inferred_state (exactly one of these strings, or null):
{states_csv}

Respond with ONLY valid JSON (no markdown):
{{
  "inferred_state": "<exact string from allowed list or null>",
  "scope": "state_level" | "central_level" | "exam_nta" | "mixed",
  "needs_user_clarification": false,
  "brief_reason": "one short sentence"
}}

scope:
- state_level: need state counselling / college / cutoff PDFs — set inferred_state whenever the question implies a place you can map to a state/UT on the list.
- central_level: MCC, AIQ, central counselling, national pools (primary retrieval is not one state’s brochure).
- exam_nta: NEET exam (syllabus, dates, application, admit card) from national bulletin.
- mixed: need both national and state-level sources.

needs_user_clarification: true only if the question is so vague that **you** cannot pick scope or state (not merely because the user omitted the word "state"). Prefer false and infer."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=220,
        )
        text = response.choices[0].message.content or ""
        data = _parse_json_from_llm_response(text)
        if not isinstance(data, dict):
            log(f"   ⚠️ LLM routing JSON parse failed: {text[:200]}")
            return None
        log(f"   🤖 LLM routing JSON: {data}")
        return data
    except Exception as e:
        log(f"   ⚠️ resolve_routing_filters_llm error: {e}")
        return None


def routing_from_llm_resolution(
    data: Optional[dict], user_state: Optional[str]
) -> Optional[QueryRouting]:
    """Turn LLM JSON into QueryRouting, or None to fall back to clarification."""
    if not data:
        return None
    if data.get("needs_user_clarification"):
        log(f"   LLM requested user clarification: {data.get('brief_reason', '')}")
        return None

    scope = (data.get("scope") or "state_level").lower()
    raw_state = data.get("inferred_state")
    st: Optional[str] = None
    if raw_state is not None and str(raw_state).strip().lower() not in ("", "null", "none"):
        st = _canonicalize_state_string(str(raw_state).strip())

    if scope == "exam_nta":
        return QueryRouting(
            intent=QueryIntent.EXAM_INFO,
            detected_state=None,
            use_user_preference=False,
            categories_to_search=[],
            confidence=0.88,
            needs_clarification=False,
        )

    if scope == "central_level":
        return QueryRouting(
            intent=QueryIntent.CENTRAL_COUNSELLING,
            detected_state=None,
            use_user_preference=False,
            categories_to_search=[],
            confidence=0.88,
            needs_clarification=False,
        )

    if scope == "mixed":
        ds = st or user_state
        if ds:
            use_pref = st is None and user_state is not None
            return QueryRouting(
                intent=QueryIntent.MIXED,
                detected_state=ds,
                use_user_preference=use_pref,
                categories_to_search=[],
                confidence=0.85,
                needs_clarification=False,
            )
        return None

    # state_level
    if st:
        return QueryRouting(
            intent=QueryIntent.STATE_COUNSELLING,
            detected_state=st,
            use_user_preference=False,
            categories_to_search=[],
            confidence=0.88,
            needs_clarification=False,
        )
    if user_state:
        log(f"   LLM did not return inferred_state; using profile state: {user_state}")
        return QueryRouting(
            intent=QueryIntent.STATE_COUNSELLING,
            detected_state=user_state,
            use_user_preference=True,
            categories_to_search=[],
            confidence=0.72,
            needs_clarification=False,
        )
    return None


def normalize_clarified_scope(raw: str, user_state: Optional[str]) -> str:
    """
    Map a free-text reply (after we asked central vs state) to
    'central', 'preference', or a canonical state name from STATES.
    """
    if not raw or not str(raw).strip():
        return "central"
    s = str(raw).strip()
    sl = s.lower()

    if sl in ("central", "preference"):
        return sl

    if re.search(r"\baiq\b", sl) or re.search(r"\bmcc\b", sl):
        return "central"

    central_signals = [
        "all india",
        "all-india",
        "medical counselling committee",
        "central counselling",
        "central counseling",
        "national level",
        "15%",
        "15 percent",
        "deemed",
        "aiims",
        "jipmer",
        "central quota",
        "all india quota",
    ]
    if any(sig in sl for sig in central_signals):
        return "central"

    preference_signals = [
        "my state",
        "my profile",
        "profile state",
        "saved state",
        "preference",
        "home state",
        "domicile",
        "85%",
        "85 percent",
        "state quota",
        "my domicile",
        "registered state",
    ]
    if user_state and any(sig in sl for sig in preference_signals):
        return "preference"

    st = detect_state_in_query(s)
    if st:
        return st

    # Substring match on canonical names (e.g. "for Karnataka" or "Tamil Nadu please")
    for canonical in sorted(set(STATES.values()), key=len, reverse=True):
        if canonical.lower() in sl:
            return canonical

    if user_state and sl in ("state", "state counselling", "state counseling", "state counseling."):
        return "preference"

    log(f"   ⚠️ Clarification not clearly matched, defaulting to central: {raw!r}")
    return "central"


def classify_by_keywords(query: str) -> Tuple[Optional[QueryIntent], float]:
    """
    Try to classify query intent using keyword matching (no LLM cost)
    Returns (intent, confidence) or (None, 0) if not confident
    """
    query_lower = query.lower()
    
    # Count matches for each intent
    exam_score = sum(1 for kw in EXAM_INFO_KEYWORDS if kw in query_lower)
    central_score = sum(1 for kw in CENTRAL_COUNSELLING_KEYWORDS if kw in query_lower)
    state_score = sum(1 for kw in STATE_COUNSELLING_KEYWORDS if kw in query_lower)
    mixed_score = sum(1 for kw in MIXED_KEYWORDS if kw in query_lower)
    
    log(f"   📊 Keyword scores: exam={exam_score}, central={central_score}, state={state_score}, mixed={mixed_score}")
    
    # Check for state mention
    has_state_mention = detect_state_in_query(query) is not None
    
    # Decision logic
    total = exam_score + central_score + state_score + mixed_score
    
    if total == 0:
        log("   ⚠️ No keyword matches - will use LLM")
        return None, 0  # Need LLM
    
    # If state is mentioned, boost state_score
    if has_state_mention:
        state_score += 2
        log(f"   State mentioned - boosted state_score to {state_score}")
    
    # Determine intent
    if exam_score >= 2 and exam_score > max(central_score, state_score):
        log("   ✅ Keyword match: EXAM_INFO")
        return QueryIntent.EXAM_INFO, min(0.9, exam_score / 5)
    
    if central_score >= 2 and central_score > max(exam_score, state_score):
        log("   ✅ Keyword match: CENTRAL_COUNSELLING")
        return QueryIntent.CENTRAL_COUNSELLING, min(0.9, central_score / 5)
    
    if state_score >= 2 and state_score > max(exam_score, central_score):
        log("   ✅ Keyword match: STATE_COUNSELLING")
        return QueryIntent.STATE_COUNSELLING, min(0.9, state_score / 5)
    
    if mixed_score >= 1:
        log("   ✅ Keyword match: MIXED")
        return QueryIntent.MIXED, 0.7
    
    log("   ⚠️ Low confidence - will use LLM")
    return None, 0  # Not confident enough


# ============== LLM-BASED ROUTING ==============

def classify_by_llm(query: str) -> Tuple[QueryIntent, float]:
    """
    Classify query intent using GPT-4o-mini when keywords are not sufficient
    """
    log("   🤖 Using LLM for intent classification...")
    
    prompt = f"""Classify this NEET exam query into exactly ONE of these 4 intents:

QUERY: "{query}"

INTENTS:
1. EXAM_INFO - Questions about NEET exam itself: syllabus, pattern, dates, admit card, eligibility, application, how to apply, marking scheme, exam centers. These are answered from NTA bulletin which applies to ALL students.

2. CENTRAL_COUNSELLING - Questions about central level counselling: MCC, All India Quota (15%), AIIMS admission, JIPMER admission, central universities, deemed universities, ESIC, AFMC. These are also from NTA bulletin.

3. STATE_COUNSELLING - Questions about state level counselling: state quota (85%), state specific seats, specific state colleges, state domicile rules, state counselling rounds, fee in specific state colleges, seat matrix of a state.

4. MIXED - Questions that need BOTH central and state information: reservation percentages (could be central or state level), general quota questions without specifying central or state.

IMPORTANT RULES:
- If asking about "NEET 2026 dates" or "important dates" → EXAM_INFO (this is from NTA bulletin, applies to all)
- If asking about "syllabus" or "pattern" → EXAM_INFO
- If asking about a SPECIFIC STATE → STATE_COUNSELLING
- If asking "OBC reservation" without specifying state → MIXED
- If asking about AIIMS, JIPMER, MCC → CENTRAL_COUNSELLING

Respond with ONLY the intent name (EXAM_INFO, CENTRAL_COUNSELLING, STATE_COUNSELLING, or MIXED):"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=20
        )
        
        result = response.choices[0].message.content.strip().upper()
        log(f"   ✅ LLM classified as: {result}")
        
        intent_map = {
            "EXAM_INFO": QueryIntent.EXAM_INFO,
            "CENTRAL_COUNSELLING": QueryIntent.CENTRAL_COUNSELLING,
            "STATE_COUNSELLING": QueryIntent.STATE_COUNSELLING,
            "MIXED": QueryIntent.MIXED
        }
        
        return intent_map.get(result, QueryIntent.MIXED), 0.85
        
    except Exception as e:
        log(f"   ❌ LLM routing error: {e}")
        # Default to MIXED to be safe
        return QueryIntent.MIXED, 0.5


# ============== MAIN ROUTING FUNCTION ==============

def route_query(
    query: str, 
    user_state: Optional[str] = None, 
    clarified_scope: Optional[str] = None,
    conversation_context: Optional[Dict[str, Any]] = None
) -> QueryRouting:
    """
    Main function to classify a query and determine routing.
    
    Args:
        query: The user's question
        user_state: User's registered state preference (from their profile)
        clarified_scope: User's clarification response - "central", "preference", or state name
        conversation_context: Context extracted from conversation memory:
            - detected_state: State mentioned in previous messages
            - detected_topic: Topic being discussed
            - last_user_question: Previous question for context
            - is_followup: Whether this appears to be a follow-up
    
    Returns:
        QueryRouting object with all routing information
    """
    log("🔀 QUERY ROUTING:")
    
    # Extract conversation context
    conv_state = None
    conv_topic = None
    is_followup = False
    if conversation_context:
        conv_state = conversation_context.get("detected_state")
        conv_topic = conversation_context.get("detected_topic")
        is_followup = conversation_context.get("is_followup", False)
        if conv_state:
            log(f"   📝 Conversation context state: {conv_state}")
        if conv_topic:
            log(f"   📝 Conversation context topic: {conv_topic}")
        if is_followup:
            log(f"   📝 Detected as follow-up question")
    
    # Step 1: Detect any state explicitly mentioned in the query
    detected_state = detect_state_in_query(query)
    if detected_state:
        log(f"   📍 Detected state in query: {detected_state}")
    
    # Step 1.1: For follow-up questions without explicit state, inherit from conversation
    elif is_followup and conv_state and not detected_state:
        detected_state = conv_state
        log(f"   📍 Using state from conversation context: {detected_state}")
    
    # Step 1.5: If user already provided clarification, use it directly
    if clarified_scope:
        clarified_scope = normalize_clarified_scope(clarified_scope, user_state)
        log(f"   ✅ User clarified scope (normalized): {clarified_scope}")
        if clarified_scope.lower() == "central":
            return QueryRouting(
                intent=QueryIntent.CENTRAL_COUNSELLING,
                detected_state=None,
                use_user_preference=False,
                categories_to_search=[],
                confidence=1.0,
                needs_clarification=False
            )
        elif clarified_scope.lower() == "preference":
            return QueryRouting(
                intent=QueryIntent.STATE_COUNSELLING,
                detected_state=user_state,  # Use their preference state
                use_user_preference=True,
                categories_to_search=[],
                confidence=1.0,
                needs_clarification=False
            )
        else:
            # User specified a different state name
            return QueryRouting(
                intent=QueryIntent.STATE_COUNSELLING,
                detected_state=clarified_scope,
                use_user_preference=False,
                categories_to_search=[],
                confidence=1.0,
                needs_clarification=False
            )
    
    # Step 2: Check if query is CLEARLY about central/NTA info (no ambiguity)
    query_lower = query.lower()
    is_clearly_central = any(kw in query_lower for kw in CLEARLY_CENTRAL_KEYWORDS)
    has_ambiguous_keywords = any(kw in query_lower for kw in AMBIGUOUS_STATE_KEYWORDS)
    
    # Step 3: If state is explicitly mentioned -> STATE_COUNSELLING, no ambiguity
    if detected_state:
        log(f"   🔄 State explicitly mentioned - forcing STATE_COUNSELLING intent")
        return QueryRouting(
            intent=QueryIntent.STATE_COUNSELLING,
            detected_state=detected_state,
            use_user_preference=False,
            categories_to_search=[],
            confidence=0.95,
            needs_clarification=False
        )
    
    # Step 4: If clearly central, no clarification needed
    if is_clearly_central and not has_ambiguous_keywords:
        log(f"   ✅ Clearly central/NTA query - no clarification needed")
        return QueryRouting(
            intent=QueryIntent.EXAM_INFO,
            detected_state=None,
            use_user_preference=False,
            categories_to_search=[],
            confidence=0.9,
            needs_clarification=False
        )
    
    # Step 5: Ambiguous keywords but no explicit state — LLM infers scope/state from metadata model (no regex tables)
    if has_ambiguous_keywords and not detected_state:
        log(f"   🤖 Ambiguous query without explicit state — resolving via LLM (indexed metadata context)")
        llm_data = resolve_routing_filters_llm(query, user_state)
        qr = routing_from_llm_resolution(llm_data, user_state)
        if qr:
            return qr
        options = ["Central/AIQ (All India)"]
        if user_state:
            options.append(f"My State ({user_state})")
        options.append("Other State")
        return QueryRouting(
            intent=QueryIntent.NEEDS_CLARIFICATION,
            detected_state=None,
            use_user_preference=False,
            categories_to_search=[],
            confidence=0.5,
            needs_clarification=True,
            clarification_options=options,
        )
    
    # Step 6: Default - try keyword then LLM classification
    log("   🔑 Trying keyword classification (FREE)...")
    intent, confidence = classify_by_keywords(query)
    
    if intent is None or confidence < 0.6:
        log(f"   ⚠️ Keyword confidence too low ({confidence:.2f}), using LLM...")
        intent, confidence = classify_by_llm(query)
    else:
        log(f"   💰 Saved LLM call! Keyword confidence: {confidence:.2f}")
    
    # Step 7: SIMPLIFIED APPROACH - If state-specific query but no state mentioned, ASK!
    # Don't assume user's profile state - that causes confusion
    
    if intent == QueryIntent.STATE_COUNSELLING and not detected_state:
        log(f"   🤖 STATE intent but no state — resolving via LLM")
        llm_data = resolve_routing_filters_llm(query, user_state)
        qr = routing_from_llm_resolution(llm_data, user_state)
        if qr:
            return qr
        options = ["All India (AIQ/NTA)"]
        if user_state:
            options.append(f"My State ({user_state})")
        options.append("Other State")
        return QueryRouting(
            intent=QueryIntent.NEEDS_CLARIFICATION,
            detected_state=None,
            use_user_preference=False,
            categories_to_search=[],
            confidence=0.5,
            needs_clarification=True,
            clarification_options=options,
        )
    
    # For EXAM_INFO and CENTRAL_COUNSELLING - no state needed
    use_user_preference = False
    
    if intent == QueryIntent.MIXED:
        if not detected_state:
            log(f"   🤖 MIXED intent but no state — resolving via LLM")
            llm_data = resolve_routing_filters_llm(query, user_state)
            qr = routing_from_llm_resolution(llm_data, user_state)
            if qr:
                return qr
            options = ["All India (AIQ/NTA)"]
            if user_state:
                options.append(f"My State ({user_state})")
            options.append("Other State")
            return QueryRouting(
                intent=QueryIntent.NEEDS_CLARIFICATION,
                detected_state=None,
                use_user_preference=False,
                categories_to_search=[],
                confidence=0.5,
                needs_clarification=True,
                clarification_options=options,
            )
    
    return QueryRouting(
        intent=intent,
        detected_state=detected_state,
        use_user_preference=use_user_preference,
        categories_to_search=[],
        confidence=confidence,
        needs_clarification=False
    )


# ============== VECTOR FILTER BUILDERS ==============

# State-scoped uploads besides the main counselling brochure (admin "Document type")
STATE_LEVEL_DOCUMENT_TYPES = (
    "state_counseling",
    "college_info",  # college / fee-only PDFs per state
    "cutoffs",
    "other",
)


def _filters_for_state_and_doc_types(state: str) -> List[Dict]:
    """One metadata filter per document type — PGVector query runs separately and merges."""
    out: List[Dict] = []
    for dt in STATE_LEVEL_DOCUMENT_TYPES:
        out.append(
            {
                "$and": [
                    {"document_type": {"$eq": dt}},
                    {"state": {"$eq": state}},
                ]
            }
        )
    return out


def build_vector_filters(routing: QueryRouting, user_state: Optional[str] = None) -> List[Dict]:
    """
    Build vector metadata filter(s) based on routing decision.
    
    IMPORTANT: We only filter by document_type and state - NOT by category.
    Category metadata is useful for display but not for retrieval filtering.
    The semantic search (embedding similarity) will naturally find the most
    relevant chunks regardless of what category they were classified as.
    
    State-level questions must search ALL state-scoped document types uploaded in admin
    (counselling brochure, college/fee PDFs, cutoffs, etc.) — not only ``state_counseling``.
    
    Args:
        routing: QueryRouting result from route_query()
        user_state: User's registered state from profile
    
    Returns:
        List of vector filter dictionaries
    """
    
    filters = []
    
    # Determine which state to use
    state_to_use = routing.detected_state
    if not state_to_use and routing.use_user_preference and user_state:
        state_to_use = user_state
    
    if routing.intent == QueryIntent.EXAM_INFO:
        # Search NTA bulletin only - no category filter, let semantic search work
        filters.append({
            "document_type": {"$eq": "nta_bulletin"}
        })
        
    elif routing.intent == QueryIntent.CENTRAL_COUNSELLING:
        # Search NTA bulletin - no category filter, semantic search will find counselling content
        filters.append({
            "document_type": {"$eq": "nta_bulletin"}
        })
        
    elif routing.intent == QueryIntent.STATE_COUNSELLING:
        # Search every state-scoped document type for this state (brochure + college fee PDFs + …)
        if state_to_use:
            filters.extend(_filters_for_state_and_doc_types(state_to_use))
            log(
                f"   STATE_COUNSELLING: {len(STATE_LEVEL_DOCUMENT_TYPES)} parallel searches "
                f"for state={state_to_use} (state_counseling, college_info, cutoffs, other)"
            )
        else:
            # No state resolved — keep retrieval narrow (main brochure only)
            filters.append({
                "document_type": {"$eq": "state_counseling"}
            })
            
    elif routing.intent == QueryIntent.MIXED:
        # NTA bulletin, then each state-level document type for the state
        filters.append({
            "document_type": {"$eq": "nta_bulletin"}
        })
        
        if state_to_use:
            filters.extend(_filters_for_state_and_doc_types(state_to_use))
    
    return filters


def format_mixed_response_prompt(central_context: str, state_context: str, state_name: str, question: str, conversation_history: str = "") -> str:
    """
    Format prompt for MIXED intent that shows both central and state info clearly
    """
    history_section = ""
    if conversation_history.strip():
        history_section = f"""
CONVERSATION HISTORY (use this to understand follow-up questions):
{conversation_history}
---
"""
    
    return f"""You are an expert NEET UG 2026 counselling assistant. Answer this question using BOTH central level and state level information.

{history_section}CURRENT QUESTION: {question}

CENTRAL LEVEL INFORMATION (from NTA Bulletin - applies to all students):
{central_context}

STATE LEVEL INFORMATION (from {state_name} Counselling Brochure):
{state_context}

CRITICAL RULES:
1. First explain the CENTRAL level policy/information
2. Then explain the STATE level ({state_name}) policy/information  
3. Clearly distinguish between central and state level in your answer
4. If context partially covers the question, give what is present and state what is missing
5. NEVER make up numbers, percentages, fees, or cutoffs not in the context
6. If this is a follow-up question, refer to conversation history to understand the context

Answer:"""
