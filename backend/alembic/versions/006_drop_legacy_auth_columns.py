"""Drop legacy username/email/password columns from users

Revision ID: 006_drop_legacy_auth_columns
Revises: 005_remove_user_fields
Create Date: 2026-04-14
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "006_drop_legacy_auth_columns"
down_revision: Union[str, None] = "005_remove_user_fields"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Drop indexes first (if present), then columns.
    op.drop_index("ix_users_username", table_name="users")
    op.drop_index("ix_users_email", table_name="users")
    op.drop_column("users", "username")
    op.drop_column("users", "email")
    op.drop_column("users", "password_hash")


def downgrade() -> None:
    op.add_column("users", sa.Column("password_hash", sa.String(length=255), nullable=False))
    op.add_column("users", sa.Column("email", sa.String(length=255), nullable=False))
    op.add_column("users", sa.Column("username", sa.String(length=50), nullable=False))
    op.create_index("ix_users_email", "users", ["email"], unique=True)
    op.create_index("ix_users_username", "users", ["username"], unique=True)
