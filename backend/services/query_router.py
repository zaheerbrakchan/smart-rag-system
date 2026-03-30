"""
Query Router Service
Classifies user query intent before hitting Pinecone to apply correct filters
"""

import os
import sys
import re
import logging
from typing import Dict, List, Optional, Tuple
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
    "j & k": "Jammu & Kashmir"
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

def route_query(query: str, user_state: Optional[str] = None, clarified_scope: Optional[str] = None) -> QueryRouting:
    """
    Main function to classify a query and determine routing.
    
    Args:
        query: The user's question
        user_state: User's registered state preference (from their profile)
        clarified_scope: User's clarification response - "central", "preference", or state name
    
    Returns:
        QueryRouting object with all routing information
    """
    log("🔀 QUERY ROUTING:")
    
    # Step 1: Detect any state explicitly mentioned in the query
    detected_state = detect_state_in_query(query)
    if detected_state:
        log(f"   📍 Detected state in query: {detected_state}")
    
    # Step 1.5: If user already provided clarification, use it directly
    if clarified_scope:
        log(f"   ✅ User clarified scope: {clarified_scope}")
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
    
    # Step 5: If has ambiguous keywords but no state -> NEEDS CLARIFICATION
    if has_ambiguous_keywords and not detected_state:
        log(f"   ❓ Ambiguous query - needs clarification (central vs state)")
        
        # Build clarification options
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
            clarification_options=options
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
        log(f"   ❓ STATE intent but no state mentioned - asking user to clarify")
        # Build clarification options
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
            clarification_options=options
        )
    
    # For EXAM_INFO and CENTRAL_COUNSELLING - no state needed
    use_user_preference = False
    
    if intent == QueryIntent.MIXED:
        # MIXED also needs clarification if no state detected
        if not detected_state:
            log(f"   ❓ MIXED intent but no state mentioned - asking user to clarify")
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
                clarification_options=options
            )
    
    return QueryRouting(
        intent=intent,
        detected_state=detected_state,
        use_user_preference=use_user_preference,
        categories_to_search=[],
        confidence=confidence,
        needs_clarification=False
    )


# ============== PINECONE FILTER BUILDERS ==============

def build_pinecone_filters(routing: QueryRouting, user_state: Optional[str] = None) -> List[Dict]:
    """
    Build Pinecone filter(s) based on routing decision.
    
    IMPORTANT: We only filter by document_type and state - NOT by category.
    Category metadata is useful for display but not for retrieval filtering.
    The semantic search (embedding similarity) will naturally find the most
    relevant chunks regardless of what category they were classified as.
    
    Args:
        routing: QueryRouting result from route_query()
        user_state: User's registered state from profile
    
    Returns:
        List of Pinecone filter dictionaries
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
        # Search state brochure only
        if state_to_use:
            filters.append({
                "$and": [
                    {"document_type": {"$eq": "state_counseling"}},
                    {"state": {"$eq": state_to_use}}
                ]
            })
        else:
            # No state available, search all state brochures
            filters.append({
                "document_type": {"$eq": "state_counseling"}
            })
            
    elif routing.intent == QueryIntent.MIXED:
        # Two separate queries: NTA bulletin first, then state brochure
        # Filter 1: NTA bulletin
        filters.append({
            "document_type": {"$eq": "nta_bulletin"}
        })
        
        # Filter 2: State brochure (if state available)
        if state_to_use:
            filters.append({
                "$and": [
                    {"document_type": {"$eq": "state_counseling"}},
                    {"state": {"$eq": state_to_use}}
                ]
            })
    
    return filters


def format_mixed_response_prompt(central_context: str, state_context: str, state_name: str, question: str) -> str:
    """
    Format prompt for MIXED intent that shows both central and state info clearly
    """
    return f"""You are a helpful NEET UG 2026 counseling assistant. Answer this question using BOTH central level and state level information provided.

QUESTION: {question}

CENTRAL LEVEL INFORMATION (from NTA Bulletin - applies to all students):
{central_context}

STATE LEVEL INFORMATION (from {state_name} Counselling Brochure):
{state_context}

RULES:
1. First explain the CENTRAL level policy/information
2. Then explain the STATE level ({state_name}) policy/information
3. Clearly distinguish between central and state level in your answer
4. If any information is missing, say so professionally
5. NEVER make up numbers or percentages

Answer:"""
