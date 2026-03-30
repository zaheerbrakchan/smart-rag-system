"""
Supabase Storage Service
Handles PDF file uploads to Supabase Storage bucket
"""

import os
from pathlib import Path
from typing import Optional, Tuple
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "documents")

_supabase_client: Optional[Client] = None


def get_supabase_client() -> Client:
    """Get or create Supabase client"""
    global _supabase_client
    
    if _supabase_client is None:
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in environment")
        _supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    return _supabase_client


async def upload_pdf_to_supabase(
    file_content: bytes,
    file_id: str,
    original_filename: str,
    state: str,
    document_type: str
) -> Tuple[str, str]:
    """
    Upload PDF to Supabase Storage
    
    Args:
        file_content: Raw file bytes
        file_id: Unique file identifier
        original_filename: Original filename
        state: State metadata for folder organization
        document_type: Document type for folder organization
    
    Returns:
        Tuple of (storage_path, public_url)
    """
    client = get_supabase_client()
    
    # Create folder structure: state/document_type/file_id_filename.pdf
    safe_filename = original_filename.replace(" ", "_").replace("/", "_")
    storage_path = f"{state}/{document_type}/{file_id}_{safe_filename}"
    
    try:
        # Upload to Supabase Storage
        response = client.storage.from_(SUPABASE_BUCKET).upload(
            path=storage_path,
            file=file_content,
            file_options={"content-type": "application/pdf"}
        )
        
        # Get public URL
        public_url = client.storage.from_(SUPABASE_BUCKET).get_public_url(storage_path)
        
        print(f"✅ Uploaded to Supabase: {storage_path}")
        return storage_path, public_url
        
    except Exception as e:
        print(f"❌ Supabase upload failed: {e}")
        raise


async def delete_pdf_from_supabase(storage_path: str) -> bool:
    """
    Delete PDF from Supabase Storage
    
    Args:
        storage_path: Path in Supabase storage
    
    Returns:
        True if deleted successfully
    """
    client = get_supabase_client()
    
    try:
        client.storage.from_(SUPABASE_BUCKET).remove([storage_path])
        print(f"✅ Deleted from Supabase: {storage_path}")
        return True
    except Exception as e:
        print(f"❌ Supabase delete failed: {e}")
        return False


async def get_pdf_from_supabase(storage_path: str) -> Optional[bytes]:
    """
    Download PDF from Supabase Storage
    
    Args:
        storage_path: Path in Supabase storage
    
    Returns:
        File content as bytes, or None if not found
    """
    client = get_supabase_client()
    
    try:
        response = client.storage.from_(SUPABASE_BUCKET).download(storage_path)
        return response
    except Exception as e:
        print(f"❌ Supabase download failed: {e}")
        return None


def list_files_in_bucket(prefix: str = "") -> list:
    """
    List files in Supabase Storage bucket
    
    Args:
        prefix: Folder prefix to filter by
    
    Returns:
        List of file objects
    """
    client = get_supabase_client()
    
    try:
        response = client.storage.from_(SUPABASE_BUCKET).list(prefix)
        return response
    except Exception as e:
        print(f"❌ Supabase list failed: {e}")
        return []
