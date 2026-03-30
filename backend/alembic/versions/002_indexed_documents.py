"""Add indexed_documents table

Revision ID: 002_indexed_documents
Revises: 001_initial
Create Date: 2026-03-26

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '002_indexed_documents'
down_revision: Union[str, None] = '001_initial'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create indexed_documents table
    op.create_table(
        'indexed_documents',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('file_id', sa.String(length=100), nullable=False),
        sa.Column('filename', sa.String(length=255), nullable=False),
        sa.Column('original_filename', sa.String(length=255), nullable=False),
        sa.Column('state', sa.String(length=100), nullable=False),
        sa.Column('document_type', sa.String(length=100), nullable=False),
        sa.Column('category', sa.String(length=100), nullable=False),
        sa.Column('year', sa.String(length=10), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('total_pages', sa.Integer(), nullable=True, default=0),
        sa.Column('total_vectors', sa.Integer(), nullable=True, default=0),
        sa.Column('file_size_kb', sa.Float(), nullable=True, default=0.0),
        sa.Column('is_active', sa.Boolean(), nullable=False, default=True),
        sa.Column('index_status', sa.String(length=50), nullable=True, default='indexed'),
        sa.Column('extra_metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('indexed_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('uploaded_by', sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes
    op.create_index('ix_indexed_documents_file_id', 'indexed_documents', ['file_id'], unique=True)
    op.create_index('ix_indexed_documents_state', 'indexed_documents', ['state'], unique=False)
    op.create_index('ix_indexed_documents_document_type', 'indexed_documents', ['document_type'], unique=False)


def downgrade() -> None:
    op.drop_index('ix_indexed_documents_document_type', table_name='indexed_documents')
    op.drop_index('ix_indexed_documents_state', table_name='indexed_documents')
    op.drop_index('ix_indexed_documents_file_id', table_name='indexed_documents')
    op.drop_table('indexed_documents')
