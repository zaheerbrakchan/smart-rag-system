"""Add system_settings table

Revision ID: 004_system_settings
Revises: 003_add_storage_columns
Create Date: 2026-04-13

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '004_system_settings'
down_revision: Union[str, None] = '003_add_storage_columns'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create system_settings table for app configuration
    op.create_table(
        'system_settings',
        sa.Column('key', sa.String(length=100), nullable=False),
        sa.Column('value', sa.Text(), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_by', sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint('key')
    )
    
    # Insert default settings
    op.execute("""
        INSERT INTO system_settings (key, value, description) VALUES 
        ('auto_learning_enabled', 'true', 'Enable/disable automatic FAQ learning from RAG responses')
        ON CONFLICT (key) DO NOTHING
    """)


def downgrade() -> None:
    op.drop_table('system_settings')
