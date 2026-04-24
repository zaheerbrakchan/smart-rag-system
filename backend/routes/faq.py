"""
FAQ Routes
Manages FAQ knowledge base - bulk upload, review, approve/reject
Supports hybrid retrieval (FAQ first, then RAG fallback)
"""

import os
import uuid
from datetime import datetime
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, status, Query, UploadFile, File
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_
from pydantic import BaseModel, Field
import json

from database.connection import get_db
from models.user import User
from models.pending_qa import PendingQA, QAStatus
from dependencies.auth import get_current_admin
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters, FilterOperator
from services.vector_store_factory import get_vector_store

router = APIRouter(prefix="/faq", tags=["FAQ"])

# Lazy-loaded embedding model (initialized on first use to avoid blocking startup)
_embed_model = None

def get_embed_model():
    """Get or initialize embedding model (lazy loading)"""
    global _embed_model
    if _embed_model is None:
        _embed_model = OpenAIEmbedding(model="text-embedding-3-small")  # Same as RAG docs
    return _embed_model


# ============== SCHEMAS ==============

class FAQCreateRequest(BaseModel):
    question: str = Field(..., min_length=5)
    answer: str = Field(..., min_length=10)
    state: Optional[str] = None
    exam: Optional[str] = "NEET"
    category: Optional[str] = None


class FAQBulkUploadItem(BaseModel):
    question: str
    answer: str
    state: Optional[str] = None
    exam: Optional[str] = "NEET"
    category: Optional[str] = None


class FAQUpdateRequest(BaseModel):
    modified_answer: Optional[str] = None
    detected_state: Optional[str] = None
    detected_exam: Optional[str] = None
    detected_category: Optional[str] = None


class FAQReviewRequest(BaseModel):
    action: str = Field(..., pattern="^(approve|reject|modify)$")
    modified_answer: Optional[str] = None
    review_notes: Optional[str] = None


class FAQResponse(BaseModel):
    id: int
    question: str
    original_answer: str
    modified_answer: Optional[str]
    detected_state: Optional[str]
    detected_exam: Optional[str]
    detected_category: Optional[str]
    status: str
    occurrence_count: int
    faq_vector_id: Optional[str]
    created_at: datetime
    reviewed_at: Optional[datetime]
    
    class Config:
        from_attributes = True


class FAQListResponse(BaseModel):
    faqs: List[FAQResponse]
    total: int
    page: int
    page_size: int


class FAQSearchResult(BaseModel):
    id: int
    question: str
    answer: str
    similarity_score: float
    state: Optional[str]
    category: Optional[str]


# ============== HELPER FUNCTIONS ==============

async def vectorize_and_store_faq(faq: PendingQA) -> str:
    """Vectorize FAQ question and store in pgvector with special metadata."""
    embedding = get_embed_model().get_text_embedding(faq.question)
    vector_id = f"faq_{faq.id}_{uuid.uuid4().hex[:8]}"

    metadata = {
        "is_faq": True,
        "faq_id": faq.id,
        "question": faq.question,
        "answer": faq.modified_answer or faq.original_answer,
        "state": faq.detected_state or "All-India",
        "exam": faq.detected_exam or "NEET",
        "category": faq.detected_category or "general",
        "document_type": "faq",
    }

    node = TextNode(
        id_=vector_id,
        text=faq.question,
        metadata=metadata,
        embedding=embedding,
    )
    vs = get_vector_store()
    await vs.async_add([node])
    return vector_id


async def search_faq(query: str, state_filter: Optional[str] = None, top_k: int = 3) -> List[dict]:
    """Search FAQ vectors in pgvector.

    IMPORTANT: Do not filter on ``is_faq`` (boolean). LlamaIndex's Postgres SQL builder
    turns ``True`` into ``float(1.0)`` and compares ``(metadata_->>'is_faq')::float``,
    but JSONB booleans extract as the text ``'true'``, which breaks the filter—so
    FAQ search returned no rows. Filter on string ``document_type == 'faq'`` instead
    (same metadata we set in ``vectorize_and_store_faq``). ``state_filter`` is unused;
    state lives in metadata for display only.
    """
    try:
        query_embedding = get_embed_model().get_text_embedding(query)

        mf = MetadataFilters(
            filters=[
                MetadataFilter(
                    key="document_type",
                    value="faq",
                    operator=FilterOperator.EQ,
                )
            ]
        )
        vs = get_vector_store()
        vq = VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=top_k,
            filters=mf,
            mode=VectorStoreQueryMode.DEFAULT,
        )
        result = await vs.aquery(vq)

        matches = []
        for i, node in enumerate(result.nodes):
            score = result.similarities[i] if i < len(result.similarities) else 0
            metadata = dict(node.metadata) if node.metadata else {}
            matches.append({
                "faq_id": metadata.get("faq_id"),
                "question": metadata.get("question", ""),
                "answer": metadata.get("answer", ""),
                "state": metadata.get("state"),
                "category": metadata.get("category"),
                "score": float(score),
            })

        return matches
    except Exception as e:
        print(f"FAQ search error: {e}")
        import traceback
        traceback.print_exc()
        return []


# ============== ROUTES ==============

@router.post("/create", response_model=FAQResponse)
async def create_faq(
    request: FAQCreateRequest,
    db: AsyncSession = Depends(get_db),
    current_admin: User = Depends(get_current_admin)
):
    """Create a single FAQ entry (auto-approved by admin)"""
    faq = PendingQA(
        question=request.question,
        original_answer=request.answer,
        detected_state=request.state,
        detected_exam=request.exam,
        detected_category=request.category,
        status=QAStatus.APPROVED,
        reviewed_by=current_admin.id,
        reviewed_at=datetime.utcnow()
    )
    db.add(faq)
    await db.commit()
    await db.refresh(faq)
    
    # Vectorize and store in Pinecone
    try:
        vector_id = await vectorize_and_store_faq(faq)
        faq.faq_vector_id = vector_id
        await db.commit()
    except Exception as e:
        print(f"Warning: Could not vectorize FAQ: {e}")
    
    return faq


@router.post("/bulk-upload")
async def bulk_upload_faqs(
    faqs: List[FAQBulkUploadItem],
    auto_approve: bool = Query(default=False, description="Auto-approve all uploaded FAQs"),
    db: AsyncSession = Depends(get_db),
    current_admin: User = Depends(get_current_admin)
):
    """Bulk upload FAQs from JSON array"""
    created = 0
    failed = 0
    
    for item in faqs:
        try:
            faq = PendingQA(
                question=item.question,
                original_answer=item.answer,
                detected_state=item.state,
                detected_exam=item.exam,
                detected_category=item.category,
                status=QAStatus.APPROVED if auto_approve else QAStatus.PENDING,
                reviewed_by=current_admin.id if auto_approve else None,
                reviewed_at=datetime.utcnow() if auto_approve else None
            )
            db.add(faq)
            await db.flush()
            
            # Vectorize if auto-approved
            if auto_approve:
                try:
                    vector_id = await vectorize_and_store_faq(faq)
                    faq.faq_vector_id = vector_id
                except Exception as e:
                    print(f"Warning: Could not vectorize FAQ {faq.id}: {e}")
            
            created += 1
        except Exception as e:
            print(f"Failed to create FAQ: {e}")
            failed += 1
    
    await db.commit()
    
    return {
        "success": True,
        "created": created,
        "failed": failed,
        "auto_approved": auto_approve
    }


@router.post("/upload-json")
async def upload_faq_json(
    file: UploadFile = File(...),
    auto_approve: bool = Query(default=False),
    db: AsyncSession = Depends(get_db),
    current_admin: User = Depends(get_current_admin)
):
    """Upload FAQs from JSON file"""
    if not file.filename.endswith('.json'):
        raise HTTPException(status_code=400, detail="Only JSON files are supported")
    
    try:
        content = await file.read()
        faqs_data = json.loads(content)
        
        if not isinstance(faqs_data, list):
            raise HTTPException(status_code=400, detail="JSON must be an array of FAQ objects")
        
        # Convert to FAQBulkUploadItem
        items = [FAQBulkUploadItem(**item) for item in faqs_data]
        
        # Use bulk upload
        return await bulk_upload_faqs(items, auto_approve, db, current_admin)
    
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format")


@router.get("/list", response_model=FAQListResponse)
async def list_faqs(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status_filter: Optional[str] = Query(None, description="pending|approved|rejected"),
    state_filter: Optional[str] = None,
    search: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
    current_admin: User = Depends(get_current_admin)
):
    """List FAQs with pagination and filtering"""
    query = select(PendingQA)
    count_query = select(func.count(PendingQA.id))
    
    # Apply filters
    if status_filter:
        try:
            status_enum = QAStatus(status_filter)
            query = query.where(PendingQA.status == status_enum)
            count_query = count_query.where(PendingQA.status == status_enum)
        except ValueError:
            pass
    
    if state_filter:
        query = query.where(PendingQA.detected_state == state_filter)
        count_query = count_query.where(PendingQA.detected_state == state_filter)
    
    if search:
        search_filter = or_(
            PendingQA.question.ilike(f"%{search}%"),
            PendingQA.original_answer.ilike(f"%{search}%")
        )
        query = query.where(search_filter)
        count_query = count_query.where(search_filter)
    
    # Get total count
    total = await db.scalar(count_query) or 0
    
    # Apply pagination
    offset = (page - 1) * page_size
    query = query.offset(offset).limit(page_size).order_by(PendingQA.id.desc())
    
    result = await db.execute(query)
    faqs = result.scalars().all()
    
    return FAQListResponse(
        faqs=[FAQResponse(
            id=f.id,
            question=f.question,
            original_answer=f.original_answer,
            modified_answer=f.modified_answer,
            detected_state=f.detected_state,
            detected_exam=f.detected_exam,
            detected_category=f.detected_category,
            status=f.status.value,
            occurrence_count=f.occurrence_count,
            faq_vector_id=f.faq_vector_id,
            created_at=f.created_at,
            reviewed_at=f.reviewed_at
        ) for f in faqs],
        total=total,
        page=page,
        page_size=page_size
    )


@router.get("/{faq_id}", response_model=FAQResponse)
async def get_faq(
    faq_id: int,
    db: AsyncSession = Depends(get_db),
    current_admin: User = Depends(get_current_admin)
):
    """Get single FAQ by ID"""
    faq = await db.get(PendingQA, faq_id)
    if not faq:
        raise HTTPException(status_code=404, detail="FAQ not found")
    
    return FAQResponse(
        id=faq.id,
        question=faq.question,
        original_answer=faq.original_answer,
        modified_answer=faq.modified_answer,
        detected_state=faq.detected_state,
        detected_exam=faq.detected_exam,
        detected_category=faq.detected_category,
        status=faq.status.value,
        occurrence_count=faq.occurrence_count,
        faq_vector_id=faq.faq_vector_id,
        created_at=faq.created_at,
        reviewed_at=faq.reviewed_at
    )


@router.post("/{faq_id}/review")
async def review_faq(
    faq_id: int,
    request: FAQReviewRequest,
    db: AsyncSession = Depends(get_db),
    current_admin: User = Depends(get_current_admin)
):
    """Review (approve/reject/modify) a pending FAQ"""
    faq = await db.get(PendingQA, faq_id)
    if not faq:
        raise HTTPException(status_code=404, detail="FAQ not found")
    
    if request.action == "approve":
        faq.status = QAStatus.APPROVED

        try:
            if faq.faq_vector_id:
                try:
                    get_vector_store().delete_nodes(node_ids=[faq.faq_vector_id])
                except Exception:
                    pass
            vector_id = await vectorize_and_store_faq(faq)
            faq.faq_vector_id = vector_id
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Could not vectorize FAQ: {e}")

    elif request.action == "reject":
        faq.status = QAStatus.REJECTED

        if faq.faq_vector_id:
            try:
                get_vector_store().delete_nodes(node_ids=[faq.faq_vector_id])
            except Exception:
                pass
            faq.faq_vector_id = None

    elif request.action == "modify":
        if not request.modified_answer:
            raise HTTPException(status_code=400, detail="modified_answer required for modify action")
        faq.status = QAStatus.MODIFIED
        faq.modified_answer = request.modified_answer

        try:
            if faq.faq_vector_id:
                try:
                    get_vector_store().delete_nodes(node_ids=[faq.faq_vector_id])
                except Exception:
                    pass
            vector_id = await vectorize_and_store_faq(faq)
            faq.faq_vector_id = vector_id
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Could not vectorize FAQ: {e}")
    
    faq.reviewed_by = current_admin.id
    faq.reviewed_at = datetime.utcnow()
    faq.review_notes = request.review_notes
    
    await db.commit()
    
    return {
        "success": True,
        "action": request.action,
        "faq_id": faq_id,
        "status": faq.status.value,
        "vector_id": faq.faq_vector_id
    }


@router.delete("/{faq_id}")
async def delete_faq(
    faq_id: int,
    db: AsyncSession = Depends(get_db),
    current_admin: User = Depends(get_current_admin)
):
    """Delete an FAQ"""
    faq = await db.get(PendingQA, faq_id)
    if not faq:
        raise HTTPException(status_code=404, detail="FAQ not found")
    
    if faq.faq_vector_id:
        try:
            get_vector_store().delete_nodes(node_ids=[faq.faq_vector_id])
        except Exception as e:
            print(f"Warning: Could not delete FAQ vector: {e}")
    
    await db.delete(faq)
    await db.commit()
    
    return {"success": True, "deleted_id": faq_id}


@router.get("/search/query")
async def search_faq_endpoint(
    q: str = Query(..., min_length=3),
    state: Optional[str] = None,
    threshold: float = Query(0.75, ge=0, le=1, description="Minimum similarity score"),
    db: AsyncSession = Depends(get_db)
):
    """
    Search FAQs by semantic similarity.
    This is the endpoint used by hybrid retrieval.
    Returns matches above the threshold.
    """
    try:
        matches = await search_faq(q, state_filter=state, top_k=5)
        
        # Filter by threshold
        filtered = [m for m in matches if m["score"] >= threshold]
        
        return {
            "query": q,
            "matches": filtered,
            "has_good_match": len(filtered) > 0 and filtered[0]["score"] >= 0.85
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")


# ============== STATS ==============

@router.get("/stats/overview")
async def get_faq_stats(
    db: AsyncSession = Depends(get_db),
    current_admin: User = Depends(get_current_admin)
):
    """Get FAQ statistics"""
    total = await db.scalar(select(func.count(PendingQA.id)))
    pending = await db.scalar(
        select(func.count(PendingQA.id)).where(PendingQA.status == QAStatus.PENDING)
    )
    approved = await db.scalar(
        select(func.count(PendingQA.id)).where(
            or_(PendingQA.status == QAStatus.APPROVED, PendingQA.status == QAStatus.MODIFIED)
        )
    )
    rejected = await db.scalar(
        select(func.count(PendingQA.id)).where(PendingQA.status == QAStatus.REJECTED)
    )
    
    return {
        "total": total or 0,
        "pending_review": pending or 0,
        "approved": approved or 0,
        "rejected": rejected or 0
    }


@router.post("/revectorize-all")
async def revectorize_all_faqs(
    db: AsyncSession = Depends(get_db),
    current_admin: User = Depends(get_current_admin)
):
    """Re-vectorize all approved FAQs with the current embedding model"""
    # Get all approved/modified FAQs
    result = await db.execute(
        select(PendingQA).where(
            or_(PendingQA.status == QAStatus.APPROVED, PendingQA.status == QAStatus.MODIFIED)
        )
    )
    faqs = result.scalars().all()
    
    if not faqs:
        return {"success": True, "message": "No approved FAQs to revectorize", "count": 0}
    
    success_count = 0
    errors = []
    
    for faq in faqs:
        try:
            # Delete old vector if exists
            if faq.faq_vector_id:
                try:
                    get_vector_store().delete_nodes(node_ids=[faq.faq_vector_id])
                except Exception:
                    pass
            
            # Re-vectorize
            vector_id = await vectorize_and_store_faq(faq)
            faq.faq_vector_id = vector_id
            success_count += 1
        except Exception as e:
            errors.append({"faq_id": faq.id, "error": str(e)})
    
    await db.commit()
    
    return {
        "success": True,
        "message": f"Re-vectorized {success_count}/{len(faqs)} FAQs",
        "count": success_count,
        "errors": errors if errors else None
    }


# ============== AUTO-LEARNING SETTINGS ==============

from models.system_settings import SystemSettings, SettingsKeys


async def get_auto_learning_status(db: AsyncSession) -> bool:
    """Get current auto-learning enabled status from database"""
    setting = await db.get(SystemSettings, SettingsKeys.AUTO_LEARNING_ENABLED)
    if setting is None:
        return True  # Default: enabled
    return setting.value.lower() == "true"


async def get_web_search_fallback_status(db: AsyncSession) -> bool:
    """Get current web search fallback status from database."""
    setting = await db.get(SystemSettings, SettingsKeys.WEB_SEARCH_FALLBACK_ENABLED)
    if setting is None:
        return False  # Default: disabled
    return setting.value.lower() == "true"


async def get_chat_references_enabled_status(db: AsyncSession) -> bool:
    """Get current chat reference visibility setting."""
    setting = await db.get(SystemSettings, SettingsKeys.CHAT_REFERENCES_ENABLED)
    if setting is None:
        return True  # Default: enabled for transparency
    return setting.value.lower() == "true"


async def get_cutoff_college_result_limit(db: AsyncSession) -> int:
    """Get current cutoff result row limit."""
    setting = await db.get(SystemSettings, SettingsKeys.CUTOFF_COLLEGE_RESULT_LIMIT)
    if setting is None:
        return 10
    try:
        value = int(setting.value)
    except (TypeError, ValueError):
        return 10
    return max(1, min(200, value))


@router.get("/settings/auto-learning")
async def get_auto_learning_setting(
    db: AsyncSession = Depends(get_db),
    current_admin: User = Depends(get_current_admin)
):
    """Get auto-learning configuration"""
    is_enabled = await get_auto_learning_status(db)
    
    # Get last update info
    setting = await db.get(SystemSettings, SettingsKeys.AUTO_LEARNING_ENABLED)
    
    return {
        "enabled": is_enabled,
        "updated_at": setting.updated_at.isoformat() if setting else None,
        "updated_by": setting.updated_by if setting else None,
        "description": "When enabled, the system automatically captures Q&A pairs from RAG responses for admin review."
    }


@router.post("/settings/auto-learning")
async def toggle_auto_learning(
    enable: bool = Query(..., description="Enable or disable auto-learning"),
    db: AsyncSession = Depends(get_db),
    current_admin: User = Depends(get_current_admin)
):
    """Toggle auto-learning on/off"""
    from datetime import datetime
    
    setting = await db.get(SystemSettings, SettingsKeys.AUTO_LEARNING_ENABLED)
    
    if setting is None:
        # Create new setting
        setting = SystemSettings(
            key=SettingsKeys.AUTO_LEARNING_ENABLED,
            value=str(enable).lower(),
            description="Enable/disable automatic FAQ learning from RAG responses",
            updated_by=current_admin.id
        )
        db.add(setting)
    else:
        # Update existing
        setting.value = str(enable).lower()
        setting.updated_by = current_admin.id
    
    await db.commit()
    await db.refresh(setting)
    
    status = "enabled" if enable else "paused"
    return {
        "success": True,
        "enabled": enable,
        "message": f"Auto-learning has been {status}",
        "updated_at": setting.updated_at.isoformat(),
        "updated_by": current_admin.id
    }


@router.get("/settings/web-search-fallback")
async def get_web_search_fallback_setting(
    db: AsyncSession = Depends(get_db),
    current_admin: User = Depends(get_current_admin)
):
    """Get web search fallback configuration."""
    is_enabled = await get_web_search_fallback_status(db)
    setting = await db.get(SystemSettings, SettingsKeys.WEB_SEARCH_FALLBACK_ENABLED)

    return {
        "enabled": is_enabled,
        "updated_at": setting.updated_at.isoformat() if setting else None,
        "updated_by": setting.updated_by if setting else None,
        "description": (
            "When enabled, the system can perform web search ONLY when a NEET in-domain query "
            "has no relevant information in RAG."
        )
    }


@router.post("/settings/web-search-fallback")
async def toggle_web_search_fallback(
    enable: bool = Query(..., description="Enable or disable web-search fallback"),
    db: AsyncSession = Depends(get_db),
    current_admin: User = Depends(get_current_admin)
):
    """Toggle web-search fallback on/off."""
    setting = await db.get(SystemSettings, SettingsKeys.WEB_SEARCH_FALLBACK_ENABLED)

    if setting is None:
        setting = SystemSettings(
            key=SettingsKeys.WEB_SEARCH_FALLBACK_ENABLED,
            value=str(enable).lower(),
            description="Enable/disable web search fallback when RAG has no relevant context",
            updated_by=current_admin.id
        )
        db.add(setting)
    else:
        setting.value = str(enable).lower()
        setting.updated_by = current_admin.id

    await db.commit()
    await db.refresh(setting)

    status = "enabled" if enable else "disabled"
    return {
        "success": True,
        "enabled": enable,
        "message": f"Web-search fallback has been {status}",
        "updated_at": setting.updated_at.isoformat(),
        "updated_by": current_admin.id
    }


@router.get("/settings/cutoff-result-limit")
async def get_cutoff_result_limit_setting(
    db: AsyncSession = Depends(get_db),
    current_admin: User = Depends(get_current_admin)
):
    """Get cutoff SQL result limit configuration."""
    value = await get_cutoff_college_result_limit(db)
    setting = await db.get(SystemSettings, SettingsKeys.CUTOFF_COLLEGE_RESULT_LIMIT)
    return {
        "limit": value,
        "updated_at": setting.updated_at.isoformat() if setting else None,
        "updated_by": setting.updated_by if setting else None,
        "description": "Maximum number of cutoff rows shown to user from SQL output."
    }


@router.get("/settings/chat-references")
async def get_chat_references_setting(
    db: AsyncSession = Depends(get_db),
    current_admin: User = Depends(get_current_admin)
):
    """Get chat references visibility configuration."""
    is_enabled = await get_chat_references_enabled_status(db)
    setting = await db.get(SystemSettings, SettingsKeys.CHAT_REFERENCES_ENABLED)
    return {
        "enabled": is_enabled,
        "updated_at": setting.updated_at.isoformat() if setting else None,
        "updated_by": setting.updated_by if setting else None,
        "description": "Show/hide reference badges and reference list in student chat responses."
    }


@router.get("/settings/chat-references/public")
async def get_chat_references_setting_public(
    db: AsyncSession = Depends(get_db),
):
    """Public read-only endpoint used by chat UI to decide reference visibility."""
    is_enabled = await get_chat_references_enabled_status(db)
    return {"enabled": is_enabled}


@router.post("/settings/chat-references")
async def toggle_chat_references(
    enable: bool = Query(..., description="Enable or disable chat references in UI"),
    db: AsyncSession = Depends(get_db),
    current_admin: User = Depends(get_current_admin)
):
    """Toggle chat reference visibility."""
    setting = await db.get(SystemSettings, SettingsKeys.CHAT_REFERENCES_ENABLED)
    if setting is None:
        setting = SystemSettings(
            key=SettingsKeys.CHAT_REFERENCES_ENABLED,
            value=str(enable).lower(),
            description="Enable/disable chat reference badges and source list visibility",
            updated_by=current_admin.id
        )
        db.add(setting)
    else:
        setting.value = str(enable).lower()
        setting.updated_by = current_admin.id

    await db.commit()
    await db.refresh(setting)
    status = "enabled" if enable else "disabled"
    return {
        "success": True,
        "enabled": enable,
        "message": f"Chat references have been {status}",
        "updated_at": setting.updated_at.isoformat(),
        "updated_by": current_admin.id
    }


@router.post("/settings/cutoff-result-limit")
async def update_cutoff_result_limit(
    limit: int = Query(..., ge=1, le=200, description="Cutoff result limit (1-200)"),
    db: AsyncSession = Depends(get_db),
    current_admin: User = Depends(get_current_admin)
):
    """Update cutoff SQL result limit configuration."""
    setting = await db.get(SystemSettings, SettingsKeys.CUTOFF_COLLEGE_RESULT_LIMIT)
    if setting is None:
        setting = SystemSettings(
            key=SettingsKeys.CUTOFF_COLLEGE_RESULT_LIMIT,
            value=str(limit),
            description="Maximum number of cutoff rows shown to user from SQL output",
            updated_by=current_admin.id
        )
        db.add(setting)
    else:
        setting.value = str(limit)
        setting.updated_by = current_admin.id

    await db.commit()
    await db.refresh(setting)
    return {
        "success": True,
        "limit": limit,
        "message": f"Cutoff result limit updated to {limit}",
        "updated_at": setting.updated_at.isoformat(),
        "updated_by": current_admin.id
    }
