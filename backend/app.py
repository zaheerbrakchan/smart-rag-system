"""
RAG Chatbot - FastAPI Backend (OpenAI Only)
Simple RAG implementation using LlamaIndex + OpenAI
"""

import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from dotenv import load_dotenv

from llama_index.core import (
    VectorStoreIndex,
    Settings,
    StorageContext,
    load_index_from_storage,
    Document
)
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.openai import OpenAI
from pypdf import PdfReader

# Load environment variables from script directory
ENV_PATH = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

# Configuration
DATA_DIR = Path(__file__).parent / "data"
STORAGE_DIR = Path(__file__).parent / "storage"

# Initialize FastAPI app
app = FastAPI(
    title="RAG Chatbot API",
    description="A RAG-based chatbot using LlamaIndex + OpenAI",
    version="1.0.0"
)

# CORS: localhost + optional ALLOWED_ORIGINS (comma-separated, e.g. your Vercel URL)
# Vercel previews/production *.vercel.app allowed via regex unless CORS_ALLOW_VERCEL_PREVIEWS=false
_default_origins = ["http://localhost:3000", "http://127.0.0.1:3000"]
_extra_origins = os.getenv("ALLOWED_ORIGINS", "").strip()
_cors_origins = list(_default_origins)
if _extra_origins:
    _cors_origins.extend([o.strip() for o in _extra_origins.split(",") if o.strip()])

_allow_vercel_regex = os.getenv("CORS_ALLOW_VERCEL_PREVIEWS", "true").lower() in ("1", "true", "yes")
_cors_regex = r"https://.*\.vercel\.app" if _allow_vercel_regex else None

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_origin_regex=_cors_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class ChatRequest(BaseModel):
    question: str
    model: str = "openai"

class Source(BaseModel):
    file_name: str
    page: Optional[int] = None
    text_snippet: str

class ChatResponse(BaseModel):
    answer: str
    sources: Optional[List[Source]] = None
    model_used: str

# Global index
index: Optional[VectorStoreIndex] = None


def get_llm():
    """Get OpenAI LLM instance"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set in environment")
    
    return OpenAI(
        model="gpt-4o-mini",
        api_key=api_key,
        temperature=0.1
    )


def load_or_create_index() -> VectorStoreIndex:
    """Load existing index or create new one from documents"""
    global index
    
    if index is not None:
        return index
    
    # Configure LLM
    llm = get_llm()
    Settings.llm = llm
    
    # Try to load existing index
    if STORAGE_DIR.exists():
        try:
            print("Loading existing index...")
            storage_context = StorageContext.from_defaults(persist_dir=str(STORAGE_DIR))
            index = load_index_from_storage(storage_context)
            print("Index loaded successfully!")
            return index
        except Exception as e:
            print(f"Could not load index: {e}")
    
    # Create new index from documents
    print("Creating new index from documents...")
    
    if not DATA_DIR.exists():
        DATA_DIR.mkdir(parents=True)
        raise ValueError(f"No documents found. Add PDFs to {DATA_DIR}")
    
    # Load documents using pypdf for proper text extraction
    documents = []
    pdf_files = list(DATA_DIR.glob("*.pdf"))
    
    if not pdf_files:
        raise ValueError(f"No PDF documents found in {DATA_DIR}")
    
    for pdf_path in pdf_files:
        print(f"Loading {pdf_path.name}...")
        reader = PdfReader(str(pdf_path))
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                doc = Document(
                    text=text,
                    metadata={
                        "file_name": pdf_path.name,
                        "page_label": str(page_num + 1),
                        "file_path": str(pdf_path)
                    }
                )
                documents.append(doc)
    
    if not documents:
        raise ValueError(f"No text content found in PDF documents")
    
    print(f"Loaded {len(documents)} pages from PDFs")
    
    # Create index
    index = VectorStoreIndex.from_documents(documents, show_progress=True)
    
    # Persist index
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    index.storage_context.persist(persist_dir=str(STORAGE_DIR))
    print(f"Index persisted to {STORAGE_DIR}")
    
    return index


@app.on_event("startup")
async def startup_event():
    """Initialize index on startup"""
    try:
        load_or_create_index()
        print("RAG Chatbot ready!")
    except Exception as e:
        print(f"Warning: Could not initialize index: {e}")
        print("Index will be created on first query")


@app.get("/")
async def root():
    """Health check"""
    return {"status": "healthy", "message": "RAG Chatbot API is running"}


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "index_loaded": index is not None,
        "available_models": ["openai"]
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat endpoint - answers questions using RAG"""
    try:
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        # Get or create index
        idx = load_or_create_index()
        
        # Create query engine
        query_engine = idx.as_query_engine(
            similarity_top_k=3,
            response_mode="compact"
        )
        
        # Custom RAG prompt
        qa_template = PromptTemplate(
            """You are a helpful AI assistant that answers questions based ONLY on the provided context.

RULES:
1. Answer ONLY using information from the context below
2. If the answer is not in the context, say: "This information is not available in the document."
3. Do NOT make up information
4. Be concise and accurate

Context:
{context_str}

Question: {query_str}

Answer:"""
        )
        
        query_engine.update_prompts({"response_synthesizer:text_qa_template": qa_template})
        
        # Execute query
        response = query_engine.query(request.question)
        
        # Extract sources
        sources = []
        if hasattr(response, 'source_nodes'):
            for node in response.source_nodes[:3]:
                text = node.node.text if hasattr(node.node, 'text') else str(node.node)
                source = Source(
                    file_name=node.node.metadata.get("file_name", "Unknown") if hasattr(node.node, 'metadata') else "Unknown",
                    page=node.node.metadata.get("page_label") if hasattr(node.node, 'metadata') else None,
                    text_snippet=text[:200] + "..." if len(text) > 200 else text
                )
                sources.append(source)
        
        return ChatResponse(
            answer=str(response),
            sources=sources if sources else None,
            model_used="openai"
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


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


@app.post("/ingest")
async def trigger_ingestion():
    """Re-index documents"""
    try:
        global index
        import shutil
        
        # Clear existing index
        if STORAGE_DIR.exists():
            shutil.rmtree(STORAGE_DIR)
        index = None
        
        # Recreate
        load_or_create_index()
        return {"status": "success", "message": "Documents re-indexed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
