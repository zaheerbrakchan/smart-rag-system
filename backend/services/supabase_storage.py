"""
Deprecated: use services.r2_storage (Cloudflare R2).
Re-exports compatible names for any legacy imports.
"""

from services.r2_storage import (
    build_storage_path_from_metadata,
    upload_pdf_to_r2 as upload_pdf_to_supabase,
    delete_pdf_from_r2 as delete_pdf_from_supabase,
    get_pdf_from_r2 as get_pdf_from_supabase,
)
