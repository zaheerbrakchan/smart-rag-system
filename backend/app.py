"""
RAG Chatbot - FastAPI Backend (OpenAI + Neon pgvector)
Production RAG with Admin Document Management & Smart Query Routing
"""

import os
import re
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

OUT_OF_DOMAIN_RESPONSE = """I'm sorry, I can only help with questions related to:

🎓 **NEET UG / JEE** - Entrance exams for medical and engineering
📋 **Admissions & Counseling** - Process, eligibility, documents
🏫 **Colleges** - Medical, engineering, deemed universities
📅 **Important Dates** - Application deadlines, exam schedules
💰 **Fees & Payments** - Application fees, college fees

Please ask something related to these topics, and I'll be happy to assist!"""


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
                        "You are a strict retrieval sufficiency evaluator.\n"
                        "Given a user question and retrieved knowledge-base content, decide if the KB content "
                        "is enough to answer accurately without assumptions.\n"
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
            
            # ========== FAQ CHECK FIRST (BEFORE EVERYTHING) ==========
            # If FAQ matches with high confidence, skip domain check, routing, RAG.
            # Default 0.85: 0.95 was too strict—near-duplicate wording often scores ~0.78–0.92.
            FAQ_SCORE_THRESHOLD = float(os.getenv("FAQ_SCORE_THRESHOLD", "0.85"))
            try:
                log("[INFO] 🔍 Checking FAQs FIRST...")
                faq_matches = await search_faq(request.question, state_filter=user_state, top_k=1)
                
                if faq_matches and faq_matches[0]["score"] >= FAQ_SCORE_THRESHOLD:
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
        save_message_to_db
    )
    from database.connection import async_session_maker
    
    log(f"\n{'='*60}")
    log(f"[V2] 📥 NEW QUERY: {request.question[:100]}...")
    if request.conversation_id:
        log(f"[V2] 💬 CONVERSATION ID: {request.conversation_id}")
    
    async def generate_stream() -> AsyncGenerator[str, None]:
        conversation_memory = None
        conversation_id = request.conversation_id
        start_time = datetime.now()
        
        try:
            if not request.question.strip():
                yield f"data: {json.dumps({'error': 'Question cannot be empty'})}\n\n"
                return
            
            # ========== LOAD/CREATE CONVERSATION ==========
            if request.user_id:
                try:
                    async with async_session_maker() as conv_db:
                        if request.conversation_id:
                            conversation_memory = ConversationMemory(
                                conversation_id=request.conversation_id,
                                user_id=request.user_id
                            )
                            await conversation_memory.load_from_db(conv_db)
                            msg_count = len(conversation_memory.get_chat_history())
                            if msg_count > 0:
                                log(f"[V2] 🧠 MEMORY LOADED: {msg_count} messages")
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
                            log(f"[V2] 📝 NEW CONVERSATION: id={conversation_id}")
                except Exception as mem_err:
                    log(f"[V2] ⚠️ Conversation error: {mem_err}")
                    conversation_memory = None
            
            # ========== FAQ CHECK (FAST PATH) ==========
            user_state = None
            if request.user_preferences and request.user_preferences.preferred_state:
                user_state = request.user_preferences.preferred_state
            
            FAQ_SCORE_THRESHOLD = float(os.getenv("FAQ_SCORE_THRESHOLD", "0.85"))
            try:
                log("[V2] 🔍 Checking FAQs...")
                faq_matches = await search_faq(request.question, state_filter=user_state, top_k=1)
                
                if faq_matches and faq_matches[0]["score"] >= FAQ_SCORE_THRESHOLD:
                    faq_match = faq_matches[0]
                    log(f"[V2] ✅ FAQ MATCH! Score: {faq_match['score']:.3f}")
                    
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
                    
                    for token in sse_tokens_preserving_formatting(faq_match["answer"]):
                        yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"
                        await asyncio.sleep(0.01)
                    
                    # Save FAQ response to conversation
                    is_new_conversation = not request.conversation_id and conversation_id
                    if request.user_id and conversation_id:
                        async with async_session_maker() as conv_db:
                            await save_message_to_db(conv_db, conversation_id, "user", request.question)
                            await save_message_to_db(
                                conv_db, conversation_id, "assistant", faq_match["answer"],
                                sources=[faq_source], was_faq_match=True, faq_confidence=faq_match["score"]
                            )
                    
                    # Send done event IMMEDIATELY - don't wait for title
                    yield f"data: {json.dumps({'type': 'done', 'from_faq': True, 'conversation_id': conversation_id})}\n\n"
                    log("[V2] ✅ Response from FAQ")
                    
                    # Generate title AFTER done event (non-blocking for user)
                    if is_new_conversation and request.user_id:
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
            web_fallback_enabled = await is_web_search_fallback_enabled()
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
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    tools=round_tools,
                    tool_choice=tool_choice_mode,
                    temperature=0.3,
                    max_tokens=1500
                )
                assistant_message = response.choices[0].message

                if not assistant_message.tool_calls:
                    break

                tool_call = assistant_message.tool_calls[0]
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments or "{}")
                log(f"[V2] 🔧 TOOL CALL: {tool_name}")
                log(f"[V2]    Args: {tool_args}")

                tool_result, success = execute_tool_call(tool_name, tool_args)
                if success:
                    log("[V2] ✅ Tool returned results")
                else:
                    log(f"[V2] ⚠️ Tool error: {tool_result[:120]}")

                if tool_name == "search_knowledge_base":
                    kb_attempted = True
                    is_sufficient, reason = assess_kb_sufficiency_with_llm(
                        client=client,
                        user_question=request.question,
                        kb_tool_result=tool_result
                    )
                    log(f"[V2] 🧪 KB sufficiency: {is_sufficient} | reason: {reason}")
                    if not is_sufficient:
                        kb_marked_insufficient = True
                        if web_fallback_enabled:
                            force_web_search_next_round = True
                        else:
                            kb_insufficient_and_web_disabled = True
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
                        "Hi! Sorry, we don't have these specific details in our knowledge base as of now. "
                        "Please visit official NTA/state counselling websites for the latest confirmed information."
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
                full_response = forced_fallback_response
                for token in sse_tokens_preserving_formatting(full_response):
                    yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"
                    await asyncio.sleep(0.01)
            elif assistant_message and not assistant_message.tool_calls and (assistant_message.content or "").strip():
                full_response = assistant_message.content or ""
                for token in sse_tokens_preserving_formatting(full_response):
                    yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"
                    await asyncio.sleep(0.01)
            else:
                log("[V2] 🤖 Generating final response with context...")
                final_messages = web_only_messages if (used_web_fallback and web_only_messages) else messages
                final_response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=final_messages,
                    temperature=0.3,
                    max_tokens=1500,
                    stream=True
                )
                full_response = ""
                for chunk in final_response:
                    if chunk.choices[0].delta.content:
                        text = chunk.choices[0].delta.content
                        full_response += text
                        yield f"data: {json.dumps({'type': 'token', 'token': text})}\n\n"

            if used_web_fallback:
                yield f"data: {json.dumps({'type': 'meta', 'web_fallback_used': True})}\n\n"
                log("[V2] 🌐 Final response generated with web fallback context")
            else:
                log("[V2] 🧠 Final response generated with RAG tool context")
                
            log(f"[V2] ✅ RESPONSE COMPLETE: {len(full_response)} chars")
            
            # ========== SAVE TO CONVERSATION ==========
            is_new_conversation = not request.conversation_id and conversation_id
            if request.user_id and conversation_id:
                try:
                    response_time = int((datetime.now() - start_time).total_seconds() * 1000)
                    async with async_session_maker() as conv_db:
                        await save_message_to_db(conv_db, conversation_id, "user", request.question)
                        await save_message_to_db(
                            conv_db, conversation_id, "assistant", full_response,
                            response_time_ms=response_time
                        )
                        log(f"[V2] 💾 Saved to conversation {conversation_id}")
                except Exception as save_err:
                    log(f"[V2] ⚠️ Save error: {save_err}")
            
            # Send done event IMMEDIATELY - don't wait for title
            yield f"data: {json.dumps({'type': 'done', 'conversation_id': conversation_id})}\n\n"
            log(f"{'='*60}")
            
            # Generate title AFTER done event (non-blocking for user)
            if is_new_conversation and request.user_id:
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
