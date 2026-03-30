"""Add storage columns to indexed_documents

Revision ID: 003_add_storage_columns
Revises: 002_indexed_documents
Create Date: 2026-03-27

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '003_add_storage_columns'
down_revision: Union[str, None] = '002_indexed_documents'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add version column if not exists
    op.add_column(
        'indexed_documents',
        sa.Column('version', sa.Integer(), nullable=True, default=1)
    )
    
    # Add storage_path column for Supabase Storage path
    op.add_column(
        'indexed_documents',
        sa.Column('storage_path', sa.String(length=500), nullable=True,
                  comment='Path in Supabase Storage bucket')
    )
    
    # Add storage_url column for public URL
    op.add_column(
        'indexed_documents',
        sa.Column('storage_url', sa.String(length=1000), nullable=True,
                  comment='Public URL for the file in Supabase Storage')
    )


def downgrade() -> None:
    op.drop_column('indexed_documents', 'storage_url')
    op.drop_column('indexed_documents', 'storage_path')
    op.drop_column('indexed_documents', 'version')
