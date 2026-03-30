"""
RAG Chatbot - FastAPI Backend (OpenAI + Pinecone)
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
from typing import Optional, List, Dict, AsyncGenerator
from dotenv import load_dotenv

# Use uvicorn's logger (same one that shows "INFO: Uvicorn running on...")
uvicorn_logger = logging.getLogger("uvicorn.error")

def log(msg):
    """Log using uvicorn's logger"""
    uvicorn_logger.info(msg)

from llama_index.core import (
    VectorStoreIndex,
    Settings,
    StorageContext,
    Document
)
from llama_index.core.prompts import PromptTemplate
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter, FilterOperator
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone
from pypdf import PdfReader
from openai import OpenAI  # For domain classification

# Database imports
from database.connection import engine, Base, init_db, close_db
from models import User, Conversation, Message, PendingQA, ActivityLog

# Smart routing services
from services.query_router import route_query, build_pinecone_filters, QueryIntent, format_mixed_response_prompt, expand_query
from services.chunk_classifier import classify_chunk

# Supabase storage import (lazy - imported when needed to avoid blocking startup)
# from services.supabase_storage import upload_pdf_to_supabase, delete_pdf_from_supabase

# Load environment variables from script directory
ENV_PATH = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

# Configuration
DATA_DIR = Path(__file__).parent / "data"


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
    
    # Pre-warm Pinecone cache in background (non-blocking)
    async def warm_pinecone_cache():
        try:
            from routes.admin import get_cached_pinecone_vector_count
            get_cached_pinecone_vector_count()
            print("✅ Pinecone cache pre-warmed")
        except Exception as e:
            print(f"⚠️ Pinecone cache warm failed (will retry on first request): {e}")
    
    asyncio.create_task(warm_pinecone_cache())
    print("📡 Pinecone cache warming in background...")
    
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
    target_exams: Optional[List[str]] = None

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
pinecone_index = None
vector_store = None

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
    
    return LlamaOpenAI(
        model="gpt-4o-mini",
        api_key=api_key,
        temperature=0.1
    )


def get_pinecone_index():
    """Get or create Pinecone index connection"""
    global pinecone_index, vector_store
    
    if pinecone_index is not None:
        return pinecone_index, vector_store
    
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_api_key:
        raise ValueError("PINECONE_API_KEY not set in environment")
    
    pc = Pinecone(api_key=pinecone_api_key)
    pinecone_index = pc.Index("neet-assistant")
    # Explicitly set text_key to ensure text content is stored/retrieved from metadata
    vector_store = PineconeVectorStore(
        pinecone_index=pinecone_index,
        text_key="text"  # Ensure text is stored in metadata for retrieval
    )
    
    return pinecone_index, vector_store


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
    """Load existing index from Pinecone"""
    global index
    
    if index is not None:
        return index
    
    # Configure LLM
    llm = get_llm()
    Settings.llm = llm
    
    # Get Pinecone connection
    _, vs = get_pinecone_index()
    
    # Load index from vector store
    index = VectorStoreIndex.from_vector_store(vs)
    print("Index loaded from Pinecone!")
    
    return index


def extract_text_from_pdf(file_path: Path) -> List[Dict]:
    """
    Extract text from PDF with page numbers.
    Uses pdfplumber for better table extraction, falls back to pypdf.
    """
    pages = []
    
    try:
        # Try pdfplumber first - better at tables
        import pdfplumber
        
        with pdfplumber.open(str(file_path)) as pdf:
            for page_num, page in enumerate(pdf.pages):
                # Extract text including tables
                text = page.extract_text()
                
                # Also try to extract tables specifically
                tables = page.extract_tables()
                if tables:
                    table_text = ""
                    for table in tables:
                        for row in table:
                            # Filter out None values and join
                            row_text = " | ".join([str(cell) if cell else "" for cell in row])
                            table_text += row_text + "\n"
                    
                    # Append table text if we got meaningful content
                    if table_text.strip():
                        text = (text or "") + "\n\n[TABLE DATA]\n" + table_text
                
                if text and text.strip():
                    pages.append({
                        "text": text,
                        "page_num": page_num + 1
                    })
        
        if pages:
            print(f"✅ Extracted {len(pages)} pages using pdfplumber")
            return pages
            
    except ImportError:
        print("⚠️ pdfplumber not installed, falling back to pypdf")
    except Exception as e:
        print(f"⚠️ pdfplumber failed ({e}), falling back to pypdf")
    
    # Fallback to pypdf
    reader = PdfReader(str(file_path))
    
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            pages.append({
                "text": text,
                "page_num": page_num + 1
            })
    
    print(f"✅ Extracted {len(pages)} pages using pypdf")
    return pages


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
        pc_index, _ = get_pinecone_index()
        stats = pc_index.describe_index_stats()
        return {
            "status": "healthy",
            "index_loaded": index is not None,
            "pinecone_connected": True,
            "total_vectors": stats.get('total_vector_count', 0),
            "vector_store": "Pinecone"
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
                "target_exams": request.user_preferences.target_exams,
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
1. Answer ONLY using information from the context below
2. If the specific information is NOT in the context, say: "I'm sorry, this information is not available at the moment. Please check the official {state_context} website for accurate details."
3. NEVER make up numbers, fees, dates, or percentages
4. Be concise, accurate, and professional
5. When mentioning data, mention which state/document it's from

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
                    document_type=metadata.get("document_type")
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
    """Streaming chat endpoint with smart query routing"""
    
    # Log immediately when endpoint is hit (before generator starts)
    log(f"\n{'='*60}")
    log(f"[INFO] 📥 NEW QUERY: {request.question[:100]}...")
    
    async def generate_stream() -> AsyncGenerator[str, None]:
        try:
            if not request.question.strip():
                yield f"data: {json.dumps({'error': 'Question cannot be empty'})}\n\n"
                return
            
            # Extract user's registered state from preferences (needed for FAQ search)
            user_state = None
            if request.user_preferences and request.user_preferences.preferred_state:
                user_state = request.user_preferences.preferred_state
            
            # ========== FAQ CHECK FIRST (BEFORE EVERYTHING) ==========
            # If FAQ matches with high confidence, skip domain check, routing, RAG
            FAQ_SCORE_THRESHOLD = float(os.getenv("FAQ_SCORE_THRESHOLD", "0.95"))
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
                    for word in faq_answer.split():
                        yield f"data: {json.dumps({'type': 'token', 'token': word + ' '})}\n\n"
                        await asyncio.sleep(0.01)
                    
                    yield f"data: {json.dumps({'type': 'done', 'from_faq': True})}\n\n"
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
                for word in OUT_OF_DOMAIN_RESPONSE.split():
                    yield f"data: {json.dumps({'type': 'token', 'token': word + ' '})}\n\n"
                    await asyncio.sleep(0.02)
                yield f"data: {json.dumps({'type': 'done', 'out_of_domain': True})}\n\n"
                return
            
            log("[INFO] ✅ DOMAIN CHECK: Query is in domain")
            log(f"[INFO] 👤 USER STATE: {user_state or 'Not set'}")
            
            # ========== SMART QUERY ROUTING ==========
            # FAQ already checked above - if we're here, no FAQ match or score too low
            routing = route_query(request.question, user_state, request.clarified_scope)
            log(f"[INFO] 🎯 ROUTING:")
            log(f"[INFO]    Intent: {routing.intent.value}")
            log(f"[INFO]    Detected State: {routing.detected_state or 'None'}")
            log(f"[INFO]    Use User Preference: {routing.use_user_preference}")
            log(f"[INFO]    Confidence: {routing.confidence:.2f}")
            
            # ========== HANDLE CLARIFICATION NEEDED ==========
            if routing.needs_clarification:
                log("[INFO] ❓ CLARIFICATION NEEDED - asking user to specify scope")
                yield f"data: {json.dumps({'type': 'clarification_needed', 'options': routing.clarification_options, 'message': 'Please clarify: Are you asking about Central/All India level or a specific state?'})}\n\n"
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                return
            
            # Build Pinecone filters based on routing
            pinecone_filters = build_pinecone_filters(routing, user_state)
            log(f"[INFO] 🔍 PINECONE FILTERS: {pinecone_filters}")
            
            # ========== PINECONE RAG RETRIEVAL ==========
            try:
                pc_index, _ = get_pinecone_index()
                
                # Expand abbreviations in query for better semantic search
                expanded_query = expand_query(request.question)
                
                # Generate query embedding
                log("[INFO] 🧮 Generating query embedding...")
                from llama_index.embeddings.openai import OpenAIEmbedding
                embed_model = OpenAIEmbedding(model="text-embedding-3-small")
                query_embedding = embed_model.get_text_embedding(expanded_query)
                log(f"[INFO] ✅ Embedding generated (dim={len(query_embedding)})")
                
                all_sources = []
                all_context_texts = []
                central_context = ""
                state_context = ""
                state_name = routing.detected_state or user_state or "your state"
                
                # Execute queries based on filters (may be 1 or 2 for MIXED intent)
                for filter_idx, pc_filter in enumerate(pinecone_filters):
                    log(f"[INFO] 🔎 PINECONE QUERY {filter_idx + 1}: filter={pc_filter}")
                    
                    results = pc_index.query(
                        vector=query_embedding,
                        top_k=6,
                        include_metadata=True,
                        filter=pc_filter
                    )
                    
                    result_matches = results.get("matches", []) if isinstance(results, dict) else getattr(results, 'matches', [])
                    log(f"[INFO]    ↳ Found {len(result_matches)} matches")
                    
                    query_texts = []
                    for match_idx, match in enumerate(result_matches):
                        if isinstance(match, dict):
                            metadata = match.get("metadata", {})
                            score = match.get("score", 0)
                        else:
                            metadata = match.metadata if hasattr(match, 'metadata') else {}
                            score = match.score if hasattr(match, 'score') else 0
                        
                        # ALWAYS try _node_content first - it has the actual chunk content
                        # The 'text' metadata field is unreliable (often just headers/abbreviations)
                        text = ""
                        node_content = metadata.get("_node_content", "")
                        if node_content:
                            try:
                                import json as json_parser
                                parsed = json_parser.loads(node_content)
                                text = parsed.get("text", "")
                            except:
                                pass
                        
                        # Fallback to text metadata only if _node_content failed
                        if not text:
                            text = metadata.get("text", "")
                        
                        # Debug: log first match's text extraction
                        if match_idx == 0:
                            log(f"[DEBUG]    Text length: {len(text)}, Preview: {text[:100]}...")
                        
                        if text and len(text) > 50:
                            ascii_ratio = sum(1 for c in text if ord(c) < 128) / max(len(text), 1)
                            if ascii_ratio >= 0.3:
                                query_texts.append(text)
                                all_context_texts.append(text)
                                all_sources.append({
                                    "file_name": metadata.get("file_name", "Unknown"),
                                    "page": metadata.get("page_label"),
                                    "text_snippet": text[:250] + "..." if len(text) > 250 else text,
                                    "state": metadata.get("state"),
                                    "document_type": metadata.get("document_type"),
                                    "category": metadata.get("category"),
                                    "score": round(score, 3)
                                })
                                # Log top 3 matches
                                if match_idx < 3:
                                    log(f"[INFO]    📄 Match {match_idx + 1}: score={score:.3f} | {metadata.get('file_name', 'Unknown')} | page {metadata.get('page_label')} | cat={metadata.get('category')}")
                    
                    # For MIXED intent, separate central and state context
                    if routing.intent == QueryIntent.MIXED:
                        if filter_idx == 0:
                            central_context = "\n\n".join(query_texts[:3])
                        else:
                            state_context = "\n\n".join(query_texts[:3])
                
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
                    
                    for word in no_result_msg.split():
                        yield f"data: {json.dumps({'type': 'token', 'token': word + ' '})}\n\n"
                        await asyncio.sleep(0.02)
                    yield f"data: {json.dumps({'type': 'done', 'intent': routing.intent.value, 'no_data': True})}\n\n"
                    return
                
                # Build prompt based on intent
                log("[INFO] 🤖 LLM GENERATION:")
                if routing.intent == QueryIntent.MIXED and central_context and state_context:
                    # Special prompt for MIXED intent
                    log("[INFO]    Using MIXED prompt (central + state)")
                    prompt = format_mixed_response_prompt(central_context, state_context, state_name, request.question)
                else:
                    # Standard prompt
                    context_str = "\n\n---\n\n".join(all_context_texts[:5])
                    
                    if routing.intent == QueryIntent.EXAM_INFO:
                        source_label = "NTA NEET UG Bulletin"
                    elif routing.intent == QueryIntent.CENTRAL_COUNSELLING:
                        source_label = "NTA NEET UG Bulletin (Central Counselling)"
                    elif routing.intent == QueryIntent.STATE_COUNSELLING:
                        source_label = f"{state_name} State Counselling Brochure"
                    else:
                        source_label = "official NEET documents"
                    
                    log(f"[INFO]    Source: {source_label}")
                    log(f"[INFO]    Context length: {len(context_str)} chars")
                    
                    prompt = f"""You are a helpful NEET UG 2026 counseling assistant. Answer the question based ONLY on the provided context from {source_label}.

RULES:
1. Answer ONLY using information from the context below
2. If the information is NOT in the context, say: "I'm sorry, this information is not available at the moment."
3. NEVER make up numbers, fees, dates, or percentages
4. Be concise, accurate, and professional
5. Mention the source (NTA bulletin / state brochure) when relevant

Context:
{context_str}

Question: {request.question}

Answer:"""
                
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
                        from sqlalchemy import select, func
                        
                        # Clean the question and answer (remove null bytes)
                        clean_question = request.question.replace('\x00', '').strip()
                        clean_answer = full_response.replace('\x00', '').strip()
                        
                        # IMPORTANT: Append state to question for FAQ (makes it specific)
                        # e.g., "what is ST fee in NEET 2026?" -> "what is ST fee in NEET 2026 for Kerala?"
                        faq_state = routing.detected_state or user_state
                        if faq_state and faq_state.lower() not in clean_question.lower():
                            # Append state if not already in question
                            clean_question = f"{clean_question} for {faq_state}"
                        elif not faq_state:
                            # No state - mark as All India
                            clean_question = f"{clean_question} (All India)"
                        
                        # Truncate if too long
                        if len(clean_answer) > 5000:
                            clean_answer = clean_answer[:5000] + "..."
                        
                        async with async_session_maker() as db:
                            # Check if similar question already exists
                            # Use first 100 chars for matching to catch variations
                            question_pattern = clean_question[:100].lower()
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
                                    question=clean_question,
                                    original_answer=clean_answer,
                                    detected_state=routing.detected_state or user_state,
                                    detected_exam="NEET",
                                    detected_category=all_sources[0].get("category") if all_sources else None,
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
                
                log("=" * 60)
                yield f"data: {json.dumps({'type': 'done', 'intent': routing.intent.value, 'source': 'rag'})}\n\n"
                
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
    """Get Pinecone index statistics"""
    try:
        pc_index, _ = get_pinecone_index()
        stats = pc_index.describe_index_stats()
        
        # Convert namespaces to simple dict
        namespaces = {}
        if stats.get('namespaces'):
            for ns_name, ns_data in stats['namespaces'].items():
                namespaces[ns_name] = {
                    "vector_count": ns_data.vector_count if hasattr(ns_data, 'vector_count') else ns_data.get('vector_count', 0)
                }
        
        return {
            "total_vectors": stats.get('total_vector_count', 0),
            "namespaces": namespaces,
            "index_name": "neet-assistant"
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
    """Upload and index a new document with metadata (stored in Supabase)"""
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
        
        # Extract text from PDF
        pages = extract_text_from_pdf(temp_file_path)
        
        if not pages:
            raise HTTPException(status_code=400, detail="Could not extract text from PDF")
        
        print(f"Extracted {len(pages)} pages from PDF")
        
        # Create documents with metadata (text added separately after chunking)
        documents = []
        for page_data in pages:
            page_text = page_data["text"]
            doc = Document(
                text=page_text,
                metadata={
                    "file_name": file.filename,
                    "file_id": file_id,
                    "page_label": str(page_data["page_num"]),
                    "state": state,
                    "document_type": document_type,
                    "category": category,
                    "year": year,
                    "description": description or "",
                    "uploaded_at": datetime.now().isoformat(),
                }
            )
            documents.append(doc)
        
        print(f"Created {len(documents)} document chunks")
        
        # Get Pinecone connection
        _, vs = get_pinecone_index()
        
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
        Settings.chunk_size = 1024  # Text chunk size
        Settings.chunk_overlap = 100
        
        # Create node parser that adds text to metadata for Pinecone retrieval
        node_parser = SentenceSplitter(
            chunk_size=1024,
            chunk_overlap=100,
            include_metadata=True,
            include_prev_next_rel=False,
        )
        
        # Parse documents into nodes and add text to metadata
        all_nodes = node_parser.get_nodes_from_documents(documents, show_progress=True)
        
        # Classify each node and add enhanced metadata
        print(f"Classifying {len(all_nodes)} chunks...")
        for i, node in enumerate(all_nodes):
            chunk_text = node.get_content()
            node.metadata["text"] = chunk_text[:2000]  # Store first 2000 chars
            
            # Use chunk classifier to get proper category
            classification = classify_chunk(
                text=chunk_text,
                document_type=document_type,
                state=state
            )
            
            # Update metadata with classification
            node.metadata["category"] = classification["category"]
            node.metadata["section"] = classification["section"]
            node.metadata["importance"] = classification["importance"]
            
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
        
        # Get updated stats
        pc_index, _ = get_pinecone_index()
        stats = pc_index.describe_index_stats()
        total_vectors = stats.get('total_vector_count', 0)
        
        # Upload PDF to Supabase Storage (lazy import to avoid blocking startup)
        try:
            from services.supabase_storage import upload_pdf_to_supabase
            storage_path, storage_url = await upload_pdf_to_supabase(
                file_content=file_content,
                file_id=file_id,
                original_filename=file.filename,
                state=state,
                document_type=document_type
            )
            print(f"✅ Uploaded to Supabase: {storage_path}")
        except Exception as storage_err:
            print(f"⚠️ Supabase upload failed (continuing without storage): {storage_err}")
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
                    
                    # Deactivate old versions and delete their vectors from Pinecone
                    for old_doc in existing_docs:
                        old_doc.is_active = False
                        old_doc.index_status = "superseded"
                        deactivated_docs.append(old_doc.file_id)
                        
                        # Delete old vectors from Pinecone
                        try:
                            # Pinecone delete by metadata filter
                            pc_index.delete(filter={"file_id": {"$eq": old_doc.file_id}})
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
                    total_pages=len(pages),
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
            "message": f"Successfully indexed {total_indexed} chunks from {file.filename} ({len(pages)} pages)",
            "file_id": file_id,
            "version": new_version,
            "pages_indexed": len(pages),
            "chunks_indexed": total_indexed,
            "metadata": {
                "state": state,
                "document_type": document_type,
                "category": category,
                "year": year
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
            "Karnataka", "Tamil Nadu", "Maharashtra", "Andhra Pradesh",
            "Telangana", "Kerala", "Gujarat", "Rajasthan", "Uttar Pradesh",
            "Madhya Pradesh", "West Bengal", "Bihar", "Odisha", "Punjab",
            "Haryana", "Delhi", "Assam", "Jharkhand", "Chhattisgarh",
            "Uttarakhand", "Himachal Pradesh", "Goa", "Jammu & Kashmir"
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
    """Clear ALL vectors from Pinecone index (use with caution!)"""
    global index
    try:
        pc_index, _ = get_pinecone_index()
        
        # Get current stats
        stats_before = pc_index.describe_index_stats()
        total_before = stats_before.get('total_vector_count', 0)
        
        # Delete all vectors
        pc_index.delete(delete_all=True)
        
        # Reset cached index
        index = None
        
        return {
            "success": True,
            "message": f"Deleted {total_before} vectors from Pinecone",
            "action": "Please re-upload your documents to rebuild the knowledge base"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear vectors: {str(e)}")


@app.get("/admin/vectors/sample")
async def get_sample_vectors():
    """Get sample vectors to check metadata structure"""
    try:
        pc_index, _ = get_pinecone_index()
        
        # Get stats
        stats = pc_index.describe_index_stats()
        
        # Query with a random vector to see sample metadata
        from llama_index.embeddings.openai import OpenAIEmbedding
        embed_model = OpenAIEmbedding(model="text-embedding-3-small")
        sample_query = embed_model.get_text_embedding("NEET eligibility")
        
        results = pc_index.query(
            vector=sample_query,
            top_k=3,
            include_metadata=True
        )
        
        samples = []
        result_matches = results.get("matches", []) if isinstance(results, dict) else getattr(results, 'matches', [])
        
        for match in result_matches:
            if isinstance(match, dict):
                metadata = match.get("metadata", {})
                score = match.get("score", 0)
                vec_id = match.get("id", "unknown")
            else:
                metadata = match.metadata if hasattr(match, 'metadata') else {}
                score = match.score if hasattr(match, 'score') else 0
                vec_id = match.id if hasattr(match, 'id') else "unknown"
            
            samples.append({
                "id": vec_id,
                "score": score,
                "metadata_keys": list(metadata.keys()),
                "has_text": "text" in metadata,
                "text_preview": metadata.get("text", "")[:100] + "..." if metadata.get("text") else None,
                "file_name": metadata.get("file_name"),
                "state": metadata.get("state"),
                "is_faq": metadata.get("is_faq", False)
            })
        
        return {
            "total_vectors": stats.get('total_vector_count', 0),
            "samples": samples
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
