"""
Cloudflare R2 (S3-compatible) object storage — replaces Supabase Storage.
Same key layout: {state}/{document_type}/{file_id}_{safe_filename}
"""

import os
from typing import Any, Optional, Tuple

import boto3
from botocore.config import Config
from dotenv import load_dotenv

load_dotenv()

R2_BUCKET = os.getenv("R2_BUCKET_NAME", "neet-documents")
R2_ENDPOINT = os.getenv("R2_ENDPOINT_URL", "").rstrip("/")
R2_ACCESS_KEY = os.getenv("R2_ACCESS_KEY_ID", "")
R2_SECRET_KEY = os.getenv("R2_SECRET_ACCESS_KEY", "")
# Public base for browser downloads (bucket public URL or custom domain), no trailing slash
R2_PUBLIC_BASE = os.getenv("R2_PUBLIC_BASE_URL", "").rstrip("/")

_s3: Any = None


def _client():
    global _s3
    if _s3 is None:
        if not R2_ENDPOINT or not R2_ACCESS_KEY or not R2_SECRET_KEY:
            raise ValueError(
                "R2 storage requires R2_ENDPOINT_URL, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY"
            )
        _s3 = boto3.client(
            "s3",
            endpoint_url=R2_ENDPOINT,
            aws_access_key_id=R2_ACCESS_KEY,
            aws_secret_access_key=R2_SECRET_KEY,
            config=Config(signature_version="s3v4"),
            region_name="auto",
        )
    return _s3


def build_storage_path_from_metadata(
    state: str,
    document_type: str,
    file_id: str,
    original_filename: str,
) -> str:
    safe_filename = original_filename.replace(" ", "_").replace("/", "_")
    return f"{state}/{document_type}/{file_id}_{safe_filename}"


async def upload_pdf_to_r2(
    file_content: bytes,
    file_id: str,
    original_filename: str,
    state: str,
    document_type: str,
) -> Tuple[str, str]:
    """
    Upload PDF bytes to R2. Returns (storage_path, public_url).
    public_url uses R2_PUBLIC_BASE_URL if set, else virtual-host style URL.
    """
    path = build_storage_path_from_metadata(
        state, document_type, file_id, original_filename
    )
    client = _client()
    client.put_object(
        Bucket=R2_BUCKET,
        Key=path,
        Body=file_content,
        ContentType="application/pdf",
    )
    if R2_PUBLIC_BASE:
        public_url = f"{R2_PUBLIC_BASE}/{path}"
    else:
        public_url = f"{R2_ENDPOINT}/{R2_BUCKET}/{path}"
    print(f"✅ Uploaded to R2: s3://{R2_BUCKET}/{path}")
    return path, public_url


async def delete_pdf_from_r2(storage_path: str) -> bool:
    client = _client()
    try:
        client.delete_object(Bucket=R2_BUCKET, Key=storage_path)
        print(f"✅ Deleted from R2: {storage_path}")
        return True
    except Exception as e:
        print(f"❌ R2 delete failed: {e}")
        raise RuntimeError(f"R2 delete failed for '{storage_path}': {e}") from e


async def get_pdf_from_r2(storage_path: str) -> Optional[bytes]:
    client = _client()
    try:
        resp = client.get_object(Bucket=R2_BUCKET, Key=storage_path)
        return resp["Body"].read()
    except Exception as e:
        print(f"❌ R2 download failed: {e}")
        return None
