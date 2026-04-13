"""Remove age and target_exams from users table

Revision ID: 005_remove_user_fields
Revises: 004_system_settings
Create Date: 2026-04-13

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '005_remove_user_fields'
down_revision: Union[str, None] = '004_system_settings'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Remove age and target_exams columns from users table."""
    # Drop age column
    op.drop_column('users', 'age')
    
    # Drop target_exams column
    op.drop_column('users', 'target_exams')


def downgrade() -> None:
    """Re-add age and target_exams columns to users table."""
    # Add target_exams column back
    op.add_column('users', sa.Column(
        'target_exams',
        postgresql.JSONB(astext_type=sa.Text()),
        nullable=True,
        comment='List of target exams: NEET, JEE, etc.'
    ))
    
    # Add age column back
    op.add_column('users', sa.Column(
        'age',
        sa.Integer(),
        nullable=True
    ))
