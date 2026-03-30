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
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, or_
from pydantic import BaseModel, EmailStr, Field
from pinecone import Pinecone

from database.connection import get_db
from models.user import User, UserRole
from models.indexed_document import IndexedDocument
from models.activity_log import ActionType
from repositories.user_repository import UserRepository
from repositories.activity_log_repository import ActivityLogRepository
from services.auth_service import get_password_hash
from dependencies.auth import get_current_admin

router = APIRouter(prefix="/admin", tags=["Admin"])

# Cached Pinecone connection
_pinecone_index = None

# Cache for expensive stats
_stats_cache = {
    "pinecone_vectors": {"value": 0, "expires": 0},
    "dashboard_stats": {"value": None, "expires": 0}
}
PINECONE_CACHE_TTL = 300  # Cache Pinecone stats for 5 minutes
DASHBOARD_CACHE_TTL = 30  # Cache dashboard stats for 30 seconds

def get_cached_pinecone_index():
    """Get or create cached Pinecone index connection"""
    global _pinecone_index
    if _pinecone_index is None:
        api_key = os.getenv("PINECONE_API_KEY")
        if api_key:
            pc = Pinecone(api_key=api_key)
            _pinecone_index = pc.Index("neet-assistant")
    return _pinecone_index

def get_cached_pinecone_vector_count(use_background_refresh: bool = True):
    """Get Pinecone vector count with caching to avoid slow API calls"""
    global _stats_cache
    now = time.time()
    
    # Return cached value if not expired
    if _stats_cache["pinecone_vectors"]["expires"] > now:
        return _stats_cache["pinecone_vectors"]["value"]
    
    # If we need fast response and have a stale value, return it and refresh in background
    if use_background_refresh and _stats_cache["pinecone_vectors"]["value"] > 0:
        # Return stale value immediately, refresh in background later
        return _stats_cache["pinecone_vectors"]["value"]
    
    # Fetch fresh value (blocking - only on first call or explicit refresh)
    try:
        pc_index = get_cached_pinecone_index()
        if pc_index:
            stats = pc_index.describe_index_stats()
            count = stats.get('total_vector_count', 0)
            _stats_cache["pinecone_vectors"] = {
                "value": count,
                "expires": now + PINECONE_CACHE_TTL
            }
            return count
    except Exception as e:
        print(f"Warning: Could not fetch Pinecone stats: {e}")
    
    return _stats_cache["pinecone_vectors"]["value"]  # Return last known value

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
            recent_users.append(UserListResponse(
                id=u.id,
                username=u.username,
                email=u.email,
                full_name=u.full_name,
                phone=u.phone,
                age=u.age,
                role=u.role.value,
                is_active=u.is_active,
                is_verified=u.is_verified,
                target_exams=u.target_exams or [],
                created_at=u.created_at,
                last_login_at=getattr(u, 'last_login_at', None)
            ))
    
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
        users=[UserListResponse(
            id=u.id,
            username=u.username,
            email=u.email,
            full_name=u.full_name,
            phone=u.phone,
            age=u.age,
            role=u.role.value,
            is_active=u.is_active,
            is_verified=u.is_verified,
            target_exams=u.target_exams or [],
            created_at=u.created_at,
            last_login_at=getattr(u, 'last_login_at', None)
        ) for u in users],
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
    
    return UserListResponse(
        id=user.id,
        username=user.username,
        email=user.email,
        full_name=user.full_name,
        phone=user.phone,
        age=user.age,
        role=user.role.value,
        is_active=user.is_active,
        is_verified=user.is_verified,
        target_exams=user.target_exams or [],
        created_at=user.created_at,
        last_login_at=getattr(user, 'last_login_at', None)
    )


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
    if request.age is not None:
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
    if request.target_exams is not None:
        user.target_exams = request.target_exams
    
    await db.commit()
    await db.refresh(user)
    
    return UserListResponse(
        id=user.id,
        username=user.username,
        email=user.email,
        full_name=user.full_name,
        phone=user.phone,
        age=user.age,
        role=user.role.value,
        is_active=user.is_active,
        is_verified=user.is_verified,
        target_exams=user.target_exams or [],
        created_at=user.created_at,
        last_login_at=getattr(user, 'last_login_at', None)
    )


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
    Reindex a document - downloads from Supabase, reclassifies chunks with new categories,
    deletes old vectors from Pinecone, and re-uploads with updated metadata.
    """
    import tempfile
    import asyncio
    from pathlib import Path
    from pypdf import PdfReader
    from llama_index.core import Document, VectorStoreIndex, Settings, StorageContext
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.vector_stores.pinecone import PineconeVectorStore
    from services.chunk_classifier import classify_chunk
    
    doc = await db.get(IndexedDocument, doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Check for storage URL - required for reindexing
    if not doc.storage_url:
        raise HTTPException(
            status_code=400, 
            detail=f"Cannot reindex: Document '{doc.original_filename}' was uploaded before cloud storage was enabled. Please delete and re-upload this document to enable reindexing."
        )
    
    temp_file_path = None
    
    try:
        # Update status to processing
        doc.index_status = "processing"
        await db.commit()
        
        # 1. Download PDF from Supabase
        import httpx
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.get(doc.storage_url)
            if response.status_code != 200:
                raise HTTPException(status_code=500, detail=f"Failed to download document from storage (HTTP {response.status_code})")
            
            # Save to temp file
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
                tmp.write(response.content)
                temp_file_path = Path(tmp.name)
        
        print(f"Downloaded document to {temp_file_path}")
        
        # 2. Extract text from PDF
        pages = []
        reader = PdfReader(str(temp_file_path))
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                pages.append({"text": text, "page_num": page_num + 1})
        
        if not pages:
            raise HTTPException(status_code=400, detail="Could not extract text from PDF")
        
        print(f"Extracted {len(pages)} pages")
        
        # 3. Delete old vectors from Pinecone
        pc_index = get_cached_pinecone_index()
        if pc_index and doc.file_id:
            pc_index.delete(filter={"file_id": {"$eq": doc.file_id}})
            print(f"Deleted old vectors for file_id={doc.file_id}")
        
        # 4. Create new documents with metadata
        documents = []
        for page_data in pages:
            llama_doc = Document(
                text=page_data["text"],
                metadata={
                    "file_name": doc.original_filename,
                    "file_id": doc.file_id,
                    "page_label": str(page_data["page_num"]),
                    "state": doc.state,
                    "document_type": doc.document_type,
                    "year": doc.year,
                    "description": doc.description or "",
                    "reindexed_at": datetime.now().isoformat(),
                }
            )
            documents.append(llama_doc)
        
        # 5. Configure embedding and chunking
        Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
        Settings.chunk_size = 1024
        Settings.chunk_overlap = 100
        
        node_parser = SentenceSplitter(
            chunk_size=1024,
            chunk_overlap=100,
            include_metadata=True,
            include_prev_next_rel=False,
        )
        
        # 6. Parse into nodes and classify each chunk
        all_nodes = node_parser.get_nodes_from_documents(documents, show_progress=True)
        print(f"Created {len(all_nodes)} nodes, classifying...")
        
        for i, node in enumerate(all_nodes):
            chunk_text = node.get_content()
            node.metadata["text"] = chunk_text[:2000]
            
            # Use chunk classifier to get proper category
            classification = classify_chunk(
                text=chunk_text,
                document_type=doc.document_type,
                state=doc.state
            )
            
            # Update metadata with classification
            node.metadata["category"] = classification["category"]
            node.metadata["section"] = classification["section"]
            node.metadata["importance"] = classification["importance"]
            
            if (i + 1) % 20 == 0:
                print(f"Classified {i + 1}/{len(all_nodes)} chunks...")
        
        print(f"Classified all {len(all_nodes)} chunks")
        
        # 7. Index to Pinecone
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        pinecone_index = pc.Index("neet-assistant")
        vector_store = PineconeVectorStore(pinecone_index=pinecone_index, text_key="text")
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
        
        print(f"Indexed {total_indexed} vectors to Pinecone")
        
        # 8. Update document record
        doc.total_pages = len(pages)
        doc.total_vectors = total_indexed
        doc.index_status = "indexed"
        doc.updated_at = datetime.now()
        
        # Store classification stats in extra_metadata
        category_counts = {}
        for node in all_nodes:
            cat = node.metadata.get("category", "general")
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
    """Delete a document - removes from Supabase Storage, Pinecone vectors, and database"""
    doc = await db.get(IndexedDocument, doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    errors = []
    
    # 1. Delete from Supabase Storage
    if doc.storage_path:
        try:
            from services.supabase_storage import delete_pdf_from_supabase
            await delete_pdf_from_supabase(doc.storage_path)
            print(f"✅ Deleted from Supabase: {doc.storage_path}")
        except Exception as e:
            errors.append(f"Supabase storage: {str(e)}")
            print(f"⚠️ Supabase delete failed: {e}")
    
    # 2. Delete vectors from Pinecone
    try:
        pc_index = get_cached_pinecone_index()
        if pc_index and doc.file_id:
            pc_index.delete(filter={"file_id": {"$eq": doc.file_id}})
            print(f"✅ Deleted vectors from Pinecone: file_id={doc.file_id}")
    except Exception as e:
        errors.append(f"Pinecone: {str(e)}")
        print(f"⚠️ Pinecone delete failed: {e}")
    
    # 3. Hard delete from database (not soft delete)
    await db.delete(doc)
    await db.commit()
    
    return {
        "success": True, 
        "message": f"Document {doc.filename} has been permanently deleted",
        "warnings": errors if errors else None
    }


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
