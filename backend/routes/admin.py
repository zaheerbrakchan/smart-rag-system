"""
Admin Routes
Handles user management and document administration
Protected by admin authentication
"""

import os
import time
from datetime import datetime
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, status, Query
from fastapi.responses import Response
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, or_
from pydantic import BaseModel, EmailStr, Field
from database.connection import get_db
from models.user import User, UserRole
from models.indexed_document import IndexedDocument
from models.activity_log import ActionType
from repositories.user_repository import UserRepository
from repositories.activity_log_repository import ActivityLogRepository
from services.auth_service import get_password_hash
from dependencies.auth import get_current_admin
from services.vector_store_factory import count_vectors_sync
from services.pdf_extraction import extract_text_from_pdf
from services.document_chunking import (
    prepare_pages_for_indexing,
    format_page_label,
    get_chunk_settings_for_document,
)

router = APIRouter(prefix="/admin", tags=["Admin"])

# Cache for expensive stats
_stats_cache = {
    "pg_vectors": {"value": 0, "expires": 0},
    "dashboard_stats": {"value": None, "expires": 0}
}
PINECONE_CACHE_TTL = 300  # Cache vector table stats for 5 minutes
DASHBOARD_CACHE_TTL = 30  # Cache dashboard stats for 30 seconds


def get_cached_pinecone_vector_count(use_background_refresh: bool = True):
    """Cached row count in pgvector table (name kept for minimal dashboard churn)."""
    global _stats_cache
    now = time.time()

    if _stats_cache["pg_vectors"]["expires"] > now:
        return _stats_cache["pg_vectors"]["value"]

    if use_background_refresh and _stats_cache["pg_vectors"]["value"] > 0:
        return _stats_cache["pg_vectors"]["value"]

    try:
        count = count_vectors_sync()
        _stats_cache["pg_vectors"] = {
            "value": count,
            "expires": now + PINECONE_CACHE_TTL,
        }
        return count
    except Exception as e:
        print(f"Warning: Could not fetch pgvector stats: {e}")

    return _stats_cache["pg_vectors"]["value"]

# ============== REQUEST/RESPONSE SCHEMAS ==============

class UserListResponse(BaseModel):
    id: int
    username: str
    email: str
    full_name: str
    phone: Optional[str]
    age: Optional[int]
    role: str
    is_active: bool
    is_verified: bool
    target_exams: List[str]
    created_at: datetime
    last_login_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class UserUpdateRequest(BaseModel):
    full_name: Optional[str] = None
    email: Optional[EmailStr] = None
    phone: Optional[str] = None
    age: Optional[int] = Field(None, ge=10, le=100)
    role: Optional[str] = None
    is_active: Optional[bool] = None
    is_verified: Optional[bool] = None
    target_exams: Optional[List[str]] = None


class PaginatedUsersResponse(BaseModel):
    users: List[UserListResponse]
    total: int
    page: int
    page_size: int
    total_pages: int


def _to_user_list_response(u: User) -> UserListResponse:
    """Map User ORM object to response, tolerating removed legacy columns."""
    return UserListResponse(
        id=u.id,
        username=u.username,
        email=u.email,
        full_name=u.full_name,
        phone=getattr(u, "phone", None),
        age=getattr(u, "age", None),
        role=u.role.value if hasattr(u.role, "value") else str(u.role),
        is_active=u.is_active,
        is_verified=u.is_verified,
        target_exams=getattr(u, "target_exams", []) or [],
        created_at=u.created_at,
        last_login_at=getattr(u, "last_login_at", None),
    )


class DashboardStatsResponse(BaseModel):
    total_users: int
    active_users: int
    verified_users: int
    admin_users: int
    total_documents: int
    total_vectors: int
    users_by_role: dict
    recent_signups: int  # Last 7 days


class DocumentListResponse(BaseModel):
    id: int
    file_id: str
    filename: str
    original_filename: str
    state: str
    document_type: str
    category: str
    year: str
    description: Optional[str]
    total_pages: int
    total_vectors: int
    file_size_kb: float
    is_active: bool
    index_status: str
    indexed_at: datetime
    storage_path: Optional[str] = None
    storage_url: Optional[str] = None
    uploaded_by: Optional[int]
    
    class Config:
        from_attributes = True


class PaginatedDocumentsResponse(BaseModel):
    documents: List[DocumentListResponse]
    total: int
    page: int
    page_size: int
    total_pages: int


class CombinedOverviewResponse(BaseModel):
    """Combined response for dashboard overview - reduces API round trips"""
    stats: DashboardStatsResponse
    recent_users: List[UserListResponse]
    recent_documents: List[DocumentListResponse]


# ============== DASHBOARD ROUTES ==============

@router.get("/dashboard/overview")
async def get_dashboard_overview(
    db: AsyncSession = Depends(get_db),
    current_admin: User = Depends(get_current_admin)
):
    """
    Combined endpoint for dashboard overview - fetches stats, recent users, and documents
    in a single API call to reduce latency for remote database connections.
    """
    from datetime import timedelta
    import asyncio
    
    global _stats_cache
    now = time.time()
    
    week_ago = datetime.utcnow() - timedelta(days=7)
    
    # Run ALL queries in parallel - stats + recent users + recent documents
    results = await asyncio.gather(
        # Stats queries
        db.scalar(select(func.count(User.id))),  # 0: total_users
        db.scalar(select(func.count(User.id)).where(User.is_active == True)),  # 1: active_users
        db.scalar(select(func.count(User.id)).where(User.is_verified == True)),  # 2: verified_users
        db.scalar(select(func.count(User.id)).where(
            or_(User.role == UserRole.ADMIN, User.role == UserRole.SUPER_ADMIN)
        )),  # 3: admin_users
        db.scalar(select(func.count(User.id)).where(User.created_at >= week_ago)),  # 4: recent_signups
        db.scalar(select(func.count(IndexedDocument.id))),  # 5: total_docs
        db.scalar(select(func.sum(IndexedDocument.total_vectors))),  # 6: total_vectors from DB
        db.execute(select(User.role, func.count(User.id)).group_by(User.role)),  # 7: role counts
        # Recent users (5 most recent)
        db.execute(select(User).order_by(User.created_at.desc()).limit(5)),  # 8: recent users
        # Recent documents (5 most recent)
        db.execute(select(IndexedDocument).order_by(IndexedDocument.indexed_at.desc()).limit(5)),  # 9: recent docs
        return_exceptions=True
    )
    
    # Process stats
    total_users = results[0] or 0
    active_users = results[1] or 0
    verified_users = results[2] or 0
    admin_users = results[3] or 0
    recent_signups = results[4] or 0
    total_docs = results[5] or 0
    db_vectors = results[6] or 0
    
    role_counts = {role.value: 0 for role in UserRole}
    if not isinstance(results[7], Exception):
        for row in results[7]:
            role_counts[row[0].value] = row[1]
    
    # Use DB vector count for fast response, Pinecone count only if already cached
    # This ensures first load is fast (no waiting for Pinecone API)
    total_vectors = int(db_vectors) if db_vectors else 0
    cached_pinecone_count = get_cached_pinecone_vector_count(use_background_refresh=True)
    if cached_pinecone_count > 0:
        total_vectors = cached_pinecone_count
    
    stats = DashboardStatsResponse(
        total_users=total_users,
        active_users=active_users,
        verified_users=verified_users,
        admin_users=admin_users,
        total_documents=total_docs,
        total_vectors=int(total_vectors),
        users_by_role=role_counts,
        recent_signups=recent_signups
    )
    
    # Process recent users
    recent_users = []
    if not isinstance(results[8], Exception):
        for u in results[8].scalars().all():
            recent_users.append(_to_user_list_response(u))
    
    # Process recent documents
    recent_documents = []
    if not isinstance(results[9], Exception):
        for d in results[9].scalars().all():
            recent_documents.append(DocumentListResponse(
                id=d.id,
                file_id=d.file_id,
                filename=d.filename,
                original_filename=d.original_filename,
                state=d.state,
                document_type=d.document_type,
                category=d.category,
                year=d.year,
                description=d.description,
                total_pages=d.total_pages,
                total_vectors=d.total_vectors,
                file_size_kb=d.file_size_kb,
                is_active=d.is_active,
                index_status=d.index_status.value if hasattr(d.index_status, 'value') else str(d.index_status),
                indexed_at=d.indexed_at,
                storage_path=d.storage_path,
                storage_url=d.storage_url,
                uploaded_by=d.uploaded_by
            ))
    
    return {
        "stats": stats,
        "recent_users": recent_users,
        "recent_documents": recent_documents
    }


@router.get("/dashboard/stats", response_model=DashboardStatsResponse)
async def get_dashboard_stats(
    db: AsyncSession = Depends(get_db),
    current_admin: User = Depends(get_current_admin)
):
    """Get comprehensive dashboard statistics - OPTIMIZED with caching and parallel queries"""
    from datetime import timedelta
    import asyncio
    
    global _stats_cache
    now = time.time()
    
    # Return cached dashboard stats if not expired (30 second cache)
    if _stats_cache["dashboard_stats"]["expires"] > now and _stats_cache["dashboard_stats"]["value"]:
        return _stats_cache["dashboard_stats"]["value"]
    
    week_ago = datetime.utcnow() - timedelta(days=7)
    
    # Run ALL database queries in parallel using asyncio.gather
    results = await asyncio.gather(
        db.scalar(select(func.count(User.id))),  # total_users
        db.scalar(select(func.count(User.id)).where(User.is_active == True)),  # active_users
        db.scalar(select(func.count(User.id)).where(User.is_verified == True)),  # verified_users
        db.scalar(select(func.count(User.id)).where(
            or_(User.role == UserRole.ADMIN, User.role == UserRole.SUPER_ADMIN)
        )),  # admin_users
        db.scalar(select(func.count(User.id)).where(User.created_at >= week_ago)),  # recent_signups
        db.scalar(select(func.count(IndexedDocument.id))),  # total_docs
        db.scalar(select(func.sum(IndexedDocument.total_vectors))),  # total_vectors from DB (fallback)
        # Role counts in single query with GROUP BY
        db.execute(
            select(User.role, func.count(User.id)).group_by(User.role)
        ),
        return_exceptions=True
    )
    
    total_users = results[0] or 0
    active_users = results[1] or 0
    verified_users = results[2] or 0
    admin_users = results[3] or 0
    recent_signups = results[4] or 0
    total_docs = results[5] or 0
    db_vectors = results[6] or 0
    
    # Process role counts from GROUP BY result
    role_counts = {role.value: 0 for role in UserRole}  # Initialize all roles to 0
    if not isinstance(results[7], Exception):
        for row in results[7]:
            role_counts[row[0].value] = row[1]
    
    # Use DB vector count for fast response, Pinecone count only if already cached
    total_vectors = int(db_vectors) if db_vectors else 0
    cached_pinecone_count = get_cached_pinecone_vector_count(use_background_refresh=True)
    if cached_pinecone_count > 0:
        total_vectors = cached_pinecone_count
    
    response = DashboardStatsResponse(
        total_users=total_users,
        active_users=active_users,
        verified_users=verified_users,
        admin_users=admin_users,
        total_documents=total_docs,
        total_vectors=int(total_vectors),
        users_by_role=role_counts,
        recent_signups=recent_signups
    )
    
    # Cache the response
    _stats_cache["dashboard_stats"] = {
        "value": response,
        "expires": now + DASHBOARD_CACHE_TTL
    }
    
    return response


# ============== USER MANAGEMENT ROUTES ==============

@router.get("/users", response_model=PaginatedUsersResponse)
async def list_users(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    search: Optional[str] = None,
    role: Optional[str] = None,
    is_active: Optional[bool] = None,
    db: AsyncSession = Depends(get_db),
    current_admin: User = Depends(get_current_admin)
):
    """List all users with pagination and filtering"""
    
    # Build query
    query = select(User)
    count_query = select(func.count(User.id))
    
    # Apply filters
    if search:
        search_filter = or_(
            User.username.ilike(f"%{search}%"),
            User.email.ilike(f"%{search}%"),
            User.full_name.ilike(f"%{search}%"),
            User.phone.ilike(f"%{search}%")
        )
        query = query.where(search_filter)
        count_query = count_query.where(search_filter)
    
    if role:
        try:
            role_enum = UserRole(role)
            query = query.where(User.role == role_enum)
            count_query = count_query.where(User.role == role_enum)
        except ValueError:
            pass
    
    if is_active is not None:
        query = query.where(User.is_active == is_active)
        count_query = count_query.where(User.is_active == is_active)
    
    # Get total count
    total = await db.scalar(count_query) or 0
    
    # Apply pagination
    offset = (page - 1) * page_size
    query = query.order_by(User.created_at.desc()).offset(offset).limit(page_size)
    
    result = await db.execute(query)
    users = result.scalars().all()
    
    return PaginatedUsersResponse(
        users=[_to_user_list_response(u) for u in users],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=(total + page_size - 1) // page_size
    )


@router.get("/users/{user_id}", response_model=UserListResponse)
async def get_user(
    user_id: int,
    db: AsyncSession = Depends(get_db),
    current_admin: User = Depends(get_current_admin)
):
    """Get a specific user by ID"""
    user = await db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return _to_user_list_response(user)


@router.patch("/users/{user_id}", response_model=UserListResponse)
async def update_user(
    user_id: int,
    request: UserUpdateRequest,
    db: AsyncSession = Depends(get_db),
    current_admin: User = Depends(get_current_admin)
):
    """Update a user's information"""
    user = await db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Prevent modifying super_admin unless you are super_admin
    if user.role == UserRole.SUPER_ADMIN and current_admin.role != UserRole.SUPER_ADMIN:
        raise HTTPException(
            status_code=403, 
            detail="Only super admins can modify other super admins"
        )
    
    # Update fields if provided
    if request.full_name is not None:
        user.full_name = request.full_name
    if request.email is not None:
        # Check if email is already taken
        existing = await db.scalar(
            select(User).where(User.email == request.email.lower(), User.id != user_id)
        )
        if existing:
            raise HTTPException(status_code=409, detail="Email already in use")
        user.email = request.email.lower()
    if request.phone is not None:
        user.phone = request.phone
    if request.age is not None and hasattr(user, "age"):
        user.age = request.age
    if request.role is not None:
        # Only super_admin can promote to admin/super_admin
        try:
            new_role = UserRole(request.role)
            if new_role in [UserRole.ADMIN, UserRole.SUPER_ADMIN]:
                if current_admin.role != UserRole.SUPER_ADMIN:
                    raise HTTPException(
                        status_code=403,
                        detail="Only super admins can assign admin roles"
                    )
            user.role = new_role
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid role")
    if request.is_active is not None:
        user.is_active = request.is_active
    if request.is_verified is not None:
        user.is_verified = request.is_verified
    if request.target_exams is not None and hasattr(user, "target_exams"):
        user.target_exams = request.target_exams
    
    await db.commit()
    await db.refresh(user)
    
    return _to_user_list_response(user)


@router.delete("/users/{user_id}")
async def delete_user(
    user_id: int,
    db: AsyncSession = Depends(get_db),
    current_admin: User = Depends(get_current_admin)
):
    """Delete a user (soft delete - sets is_active to False)"""
    user = await db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Prevent deleting super_admin
    if user.role == UserRole.SUPER_ADMIN:
        raise HTTPException(
            status_code=403, 
            detail="Cannot delete super admin accounts"
        )
    
    # Prevent self-deletion
    if user.id == current_admin.id:
        raise HTTPException(
            status_code=400,
            detail="Cannot delete your own account"
        )
    
    # Soft delete
    user.is_active = False
    await db.commit()
    
    return {"success": True, "message": f"User {user.username} has been deactivated"}


@router.post("/users/{user_id}/reset-password")
async def reset_user_password(
    user_id: int,
    db: AsyncSession = Depends(get_db),
    current_admin: User = Depends(get_current_admin)
):
    """Reset a user's password to a temporary one"""
    import secrets
    
    user = await db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Generate temporary password
    temp_password = secrets.token_urlsafe(12)
    user.password_hash = get_password_hash(temp_password)
    await db.commit()
    
    return {
        "success": True,
        "message": f"Password reset for {user.username}",
        "temporary_password": temp_password
    }


# ============== DOCUMENT MANAGEMENT ROUTES ==============

@router.get("/documents", response_model=PaginatedDocumentsResponse)
async def list_documents(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    state: Optional[str] = None,
    document_type: Optional[str] = None,
    search: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
    current_admin: User = Depends(get_current_admin)
):
    """List all indexed documents with pagination and filtering"""
    
    query = select(IndexedDocument)
    count_query = select(func.count(IndexedDocument.id))
    
    # Apply filters
    if state:
        query = query.where(IndexedDocument.state == state)
        count_query = count_query.where(IndexedDocument.state == state)
    
    if document_type:
        query = query.where(IndexedDocument.document_type == document_type)
        count_query = count_query.where(IndexedDocument.document_type == document_type)
    
    if search:
        search_filter = or_(
            IndexedDocument.filename.ilike(f"%{search}%"),
            IndexedDocument.original_filename.ilike(f"%{search}%"),
            IndexedDocument.description.ilike(f"%{search}%")
        )
        query = query.where(search_filter)
        count_query = count_query.where(search_filter)
    
    # Get total
    total = await db.scalar(count_query) or 0
    
    # Apply pagination
    offset = (page - 1) * page_size
    query = query.order_by(IndexedDocument.indexed_at.desc()).offset(offset).limit(page_size)
    
    result = await db.execute(query)
    documents = result.scalars().all()
    
    return PaginatedDocumentsResponse(
        documents=[DocumentListResponse.model_validate(doc) for doc in documents],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=(total + page_size - 1) // page_size
    )


@router.get("/documents/{doc_id}", response_model=DocumentListResponse)
async def get_document(
    doc_id: int,
    db: AsyncSession = Depends(get_db),
    current_admin: User = Depends(get_current_admin)
):
    """Get a specific document by ID"""
    doc = await db.get(IndexedDocument, doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return DocumentListResponse.model_validate(doc)


@router.patch("/documents/{doc_id}")
async def update_document(
    doc_id: int,
    is_active: Optional[bool] = None,
    description: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
    current_admin: User = Depends(get_current_admin)
):
    """Update document metadata"""
    doc = await db.get(IndexedDocument, doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    if is_active is not None:
        doc.is_active = is_active
    if description is not None:
        doc.description = description
    
    await db.commit()
    await db.refresh(doc)
    
    return {"success": True, "document": doc.to_dict()}


@router.post("/documents/{doc_id}/reindex")
async def reindex_document(
    doc_id: int,
    db: AsyncSession = Depends(get_db),
    current_admin: User = Depends(get_current_admin)
):
    """
    Reindex a document - downloads from object storage, reclassifies chunks,
    deletes old vectors from pgvector, and re-uploads with updated metadata.
    """
    import tempfile
    import asyncio
    from pathlib import Path
    from pypdf import PdfReader
    from llama_index.core import Document, VectorStoreIndex, Settings, StorageContext
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.core.vector_stores import MetadataFilter, MetadataFilters, FilterOperator
    from services.chunk_classifier import classify_chunk
    from services.vector_store_factory import get_vector_store
    from services.r2_storage import get_pdf_from_r2, build_storage_path_from_metadata
    
    doc = await db.get(IndexedDocument, doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    can_fetch = (
        doc.storage_path
        or doc.storage_url
        or (doc.file_id and doc.original_filename)
    )
    if not can_fetch:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Cannot reindex: Document '{doc.original_filename}' has no storage reference. "
                "Delete and re-upload if it predates cloud storage."
            ),
        )
    
    temp_file_path = None
    
    try:
        # Update status to processing
        doc.index_status = "processing"
        await db.commit()
        
        # 1. Download PDF: prefer R2 S3 API (works with private buckets); public URL often returns 400/403
        pdf_bytes = None
        r2_paths_tried: list[str] = []
        if doc.storage_path:
            r2_paths_tried.append(doc.storage_path)
            pdf_bytes = await get_pdf_from_r2(doc.storage_path)
        if not pdf_bytes and doc.file_id and doc.original_filename:
            inferred = build_storage_path_from_metadata(
                doc.state, doc.document_type, doc.file_id, doc.original_filename
            )
            if inferred not in r2_paths_tried:
                r2_paths_tried.append(inferred)
                pdf_bytes = await get_pdf_from_r2(inferred)
        if pdf_bytes:
            print(f"Downloaded PDF from R2 ({r2_paths_tried[-1] if r2_paths_tried else 'key'})")
        elif doc.storage_url:
            import httpx
            async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
                response = await client.get(doc.storage_url)
                if response.status_code == 200:
                    pdf_bytes = response.content
                    print("Downloaded PDF via public storage_url (fallback)")
                else:
                    raise HTTPException(
                        status_code=500,
                        detail=(
                            f"Could not fetch PDF from R2 or public URL. "
                            f"Public URL returned HTTP {response.status_code}. "
                            "Check R2_PUBLIC_BASE_URL, bucket access, and that R2 credentials match this bucket."
                        ),
                    )
        if not pdf_bytes:
            raise HTTPException(
                status_code=500,
                detail=(
                    "Could not download PDF from R2 (check R2_BUCKET_NAME, keys, and credentials) "
                    "and no usable public URL fallback."
                ),
            )
        
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(pdf_bytes)
            temp_file_path = Path(tmp.name)
        
        print(f"Saved document to {temp_file_path}")
        
        # 2. Extract text (pdfplumber when available), drop blanks, merge fee pages for college_info+fees
        pages_raw = extract_text_from_pdf(temp_file_path)
        try:
            total_pdf_pages = len(PdfReader(str(temp_file_path)).pages)
        except Exception:
            total_pdf_pages = max((p["page_num"] for p in pages_raw), default=0) if pages_raw else 0
        pages = prepare_pages_for_indexing(pages_raw, doc.document_type, doc.category)

        if not pages:
            raise HTTPException(
                status_code=400,
                detail="Could not extract text from PDF (all pages empty or too short)",
            )

        print(f"Prepared {len(pages)} indexing unit(s) from {total_pdf_pages} PDF page(s)")
        
        # 3. Delete old vectors from pgvector
        vs = get_vector_store()
        if doc.file_id:
            mf = MetadataFilters(
                filters=[
                    MetadataFilter(
                        key="file_id",
                        value=doc.file_id,
                        operator=FilterOperator.EQ,
                    )
                ]
            )
            vs.delete_nodes(filters=mf)
            print(f"Deleted old vectors for file_id={doc.file_id}")
        
        # 4. Create new documents with metadata
        documents = []
        for page_data in pages:
            llama_doc = Document(
                text=page_data["text"],
                metadata={
                    "file_name": doc.original_filename,
                    "file_id": doc.file_id,
                    "page_label": format_page_label(page_data),
                    "state": doc.state,
                    "document_type": doc.document_type,
                    "doc_topic": doc.category,
                    "year": doc.year,
                    "description": doc.description or "",
                    "reindexed_at": datetime.now().isoformat(),
                },
            )
            documents.append(llama_doc)
        
        # 5. Configure embedding and chunking (larger windows for merged college fee text)
        Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
        chunk_size, chunk_overlap = get_chunk_settings_for_document(doc.document_type, doc.category)
        Settings.chunk_size = chunk_size
        Settings.chunk_overlap = chunk_overlap

        node_parser = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            include_metadata=True,
            include_prev_next_rel=False,
        )
        
        # 6. Parse into nodes and classify each chunk
        all_nodes = node_parser.get_nodes_from_documents(documents, show_progress=True)
        print(f"Created {len(all_nodes)} nodes, classifying...")
        
        for i, node in enumerate(all_nodes):
            chunk_text = node.get_content()
            # Chunk body lives on the node; avoid duplicating in metadata JSONB.

            # Use chunk classifier to get proper category
            classification = classify_chunk(
                text=chunk_text,
                document_type=doc.document_type,
                state=doc.state
            )
            
            node.metadata["chunk_category"] = classification["category"]
            node.metadata["chunk_section"] = classification["section"]
            node.metadata["chunk_importance"] = classification["importance"]
            
            if (i + 1) % 20 == 0:
                print(f"Classified {i + 1}/{len(all_nodes)} chunks...")
        
        print(f"Classified all {len(all_nodes)} chunks")
        
        # 7. Index to pgvector
        vector_store = get_vector_store()
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        BATCH_SIZE = 10
        total_indexed = 0
        
        for i in range(0, len(all_nodes), BATCH_SIZE):
            batch = all_nodes[i:i + BATCH_SIZE]
            VectorStoreIndex(
                nodes=batch,
                storage_context=storage_context,
                show_progress=False
            )
            total_indexed += len(batch)
            
            if i + BATCH_SIZE < len(all_nodes):
                import time
                time.sleep(0.3)
        
        print(f"Indexed {total_indexed} vectors to pgvector")
        
        # 8. Update document record
        doc.total_pages = total_pdf_pages
        doc.total_vectors = total_indexed
        doc.index_status = "indexed"
        doc.updated_at = datetime.now()
        
        # Store classification stats in extra_metadata
        category_counts = {}
        for node in all_nodes:
            cat = node.metadata.get("chunk_category") or node.metadata.get("category", "general")
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        doc.extra_metadata = {
            **(doc.extra_metadata or {}),
            "reindexed_at": datetime.now().isoformat(),
            "category_distribution": category_counts
        }
        
        await db.commit()
        await db.refresh(doc)
        
        return {
            "success": True,
            "message": f"Document reindexed successfully with {total_indexed} vectors",
            "document": doc.to_dict(),
            "category_distribution": category_counts
        }
        
    except HTTPException:
        doc.index_status = "failed"
        await db.commit()
        raise
    except Exception as e:
        # Revert status on failure
        doc.index_status = "failed"
        await db.commit()
        
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Reindex failed: {str(e)}")
    
    finally:
        # Cleanup temp file
        if temp_file_path and temp_file_path.exists():
            temp_file_path.unlink()


@router.delete("/documents/{doc_id}")
async def delete_document(
    doc_id: int,
    db: AsyncSession = Depends(get_db),
    current_admin: User = Depends(get_current_admin)
):
    """Delete a document - removes from R2 storage, pgvector rows, and database"""
    doc = await db.get(IndexedDocument, doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    errors = []
    
    # 1. Delete from R2 object storage
    from services.r2_storage import (
        delete_pdf_from_r2,
        build_storage_path_from_metadata,
    )
    storage_path_to_delete = doc.storage_path
    if not storage_path_to_delete and doc.file_id and doc.original_filename:
        storage_path_to_delete = build_storage_path_from_metadata(
            doc.state, doc.document_type, doc.file_id, doc.original_filename
        )
        print(
            f"ℹ️ storage_path was empty; trying inferred path: {storage_path_to_delete}"
        )
    if storage_path_to_delete:
        try:
            await delete_pdf_from_r2(storage_path_to_delete)
            print(f"✅ Deleted from R2: {storage_path_to_delete}")
        except Exception as e:
            errors.append(f"R2 storage: {str(e)}")
            print(f"⚠️ R2 delete failed: {e}")

    # 2. Delete vectors from pgvector
    try:
        from llama_index.core.vector_stores import MetadataFilter, MetadataFilters, FilterOperator
        from services.vector_store_factory import get_vector_store

        if doc.file_id:
            vs = get_vector_store()
            mf = MetadataFilters(
                filters=[
                    MetadataFilter(
                        key="file_id",
                        value=doc.file_id,
                        operator=FilterOperator.EQ,
                    )
                ]
            )
            vs.delete_nodes(filters=mf)
            print(f"✅ Deleted vectors from pgvector: file_id={doc.file_id}")
    except Exception as e:
        errors.append(f"pgvector: {str(e)}")
        print(f"⚠️ pgvector delete failed: {e}")
    
    # 3. Hard delete from database (not soft delete)
    await db.delete(doc)
    await db.commit()
    
    return {
        "success": True, 
        "message": f"Document {doc.filename} has been permanently deleted",
        "warnings": errors if errors else None
    }


@router.get("/documents/{doc_id}/file")
async def get_document_file(
    doc_id: int,
    download: bool = Query(False, description="Set true to force file download"),
    db: AsyncSession = Depends(get_db),
    current_admin: User = Depends(get_current_admin)
):
    """Serve document PDF from R2 via authenticated admin endpoint."""
    from services.r2_storage import get_pdf_from_r2, build_storage_path_from_metadata

    doc = await db.get(IndexedDocument, doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    storage_path = doc.storage_path
    if not storage_path and doc.file_id and doc.original_filename:
        storage_path = build_storage_path_from_metadata(
            doc.state, doc.document_type, doc.file_id, doc.original_filename
        )

    if not storage_path:
        raise HTTPException(
            status_code=400,
            detail="This document has no storage path and cannot be fetched."
        )

    pdf_bytes = await get_pdf_from_r2(storage_path)
    if not pdf_bytes:
        raise HTTPException(
            status_code=404,
            detail="Document file not found in storage."
        )

    disposition = "attachment" if download else "inline"
    safe_name = (doc.original_filename or "document.pdf").replace('"', "")

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'{disposition}; filename="{safe_name}"'
        },
    )


# ============== ACTIVITY LOG ROUTES ==============

@router.get("/activity-logs")
async def get_activity_logs(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    user_id: Optional[int] = None,
    action_type: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
    current_admin: User = Depends(get_current_admin)
):
    """Get activity logs with filtering"""
    from models.activity_log import ActivityLog
    
    query = select(ActivityLog)
    count_query = select(func.count(ActivityLog.id))
    
    if user_id:
        query = query.where(ActivityLog.user_id == user_id)
        count_query = count_query.where(ActivityLog.user_id == user_id)
    
    if action_type:
        try:
            action_enum = ActionType(action_type)
            query = query.where(ActivityLog.action_type == action_enum)
            count_query = count_query.where(ActivityLog.action_type == action_enum)
        except ValueError:
            pass
    
    total = await db.scalar(count_query) or 0
    
    offset = (page - 1) * page_size
    query = query.order_by(ActivityLog.created_at.desc()).offset(offset).limit(page_size)
    
    result = await db.execute(query)
    logs = result.scalars().all()
    
    return {
        "logs": [
            {
                "id": log.id,
                "action_type": log.action_type.value,
                "description": log.description,
                "user_id": log.user_id,
                "target_type": log.target_type,
                "target_id": log.target_id,
                "ip_address": log.ip_address,
                "created_at": log.created_at.isoformat()
            }
            for log in logs
        ],
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": (total + page_size - 1) // page_size
    }
