"""Initial migration - Create all tables

Revision ID: 001_initial
Revises: 
Create Date: 2026-03-24

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '001_initial'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create users table
    op.create_table(
        'users',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('username', sa.String(length=50), nullable=False),
        sa.Column('email', sa.String(length=255), nullable=False),
        sa.Column('password_hash', sa.String(length=255), nullable=False),
        sa.Column('full_name', sa.String(length=100), nullable=False),
        sa.Column('phone', sa.String(length=20), nullable=True),
        sa.Column('age', sa.Integer(), nullable=True),
        sa.Column('role', sa.Enum('student', 'admin', 'super_admin', name='userrole'), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False, default=True),
        sa.Column('is_verified', sa.Boolean(), nullable=False, default=False),
        sa.Column('target_exams', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('preferences', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('profile_data', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('last_login_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_users_username', 'users', ['username'], unique=True)
    op.create_index('ix_users_email', 'users', ['email'], unique=True)
    op.create_index('ix_users_role', 'users', ['role'], unique=False)

    # Create conversations table
    op.create_table(
        'conversations',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('title', sa.String(length=255), nullable=True),
        sa.Column('summary', sa.Text(), nullable=True),
        sa.Column('context_data', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False, default=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_conversations_user_id', 'conversations', ['user_id'], unique=False)

    # Create messages table
    op.create_table(
        'messages',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('conversation_id', sa.Integer(), nullable=False),
        sa.Column('role', sa.Enum('user', 'assistant', 'system', name='messagerole'), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('sources', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('model_used', sa.String(length=50), nullable=True),
        sa.Column('filters_applied', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('response_time_ms', sa.Integer(), nullable=True),
        sa.Column('tokens_used', sa.Integer(), nullable=True),
        sa.Column('was_faq_match', sa.Boolean(), nullable=False, default=False),
        sa.Column('faq_confidence', sa.Float(), nullable=True),
        sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['conversation_id'], ['conversations.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_messages_conversation_id', 'messages', ['conversation_id'], unique=False)
    op.create_index('ix_messages_created_at', 'messages', ['created_at'], unique=False)

    # Create pending_qa table
    op.create_table(
        'pending_qa',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('question', sa.Text(), nullable=False),
        sa.Column('original_answer', sa.Text(), nullable=False),
        sa.Column('modified_answer', sa.Text(), nullable=True),
        sa.Column('source_conversation_id', sa.Integer(), nullable=True),
        sa.Column('source_documents', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('detected_state', sa.String(length=50), nullable=True),
        sa.Column('detected_exam', sa.String(length=50), nullable=True),
        sa.Column('detected_category', sa.String(length=100), nullable=True),
        sa.Column('original_confidence', sa.Float(), nullable=True),
        sa.Column('occurrence_count', sa.Integer(), nullable=False, default=1),
        sa.Column('status', sa.Enum('pending', 'approved', 'rejected', 'modified', name='qastatus'), nullable=False),
        sa.Column('reviewed_by', sa.Integer(), nullable=True),
        sa.Column('review_notes', sa.Text(), nullable=True),
        sa.Column('faq_vector_id', sa.String(length=100), nullable=True),
        sa.Column('extra_data', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('reviewed_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['source_conversation_id'], ['conversations.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['reviewed_by'], ['users.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_pending_qa_status', 'pending_qa', ['status'], unique=False)
    op.create_index('ix_pending_qa_detected_state', 'pending_qa', ['detected_state'], unique=False)
    op.create_index('ix_pending_qa_detected_exam', 'pending_qa', ['detected_exam'], unique=False)

    # Create activity_logs table
    op.create_table(
        'activity_logs',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=True),
        sa.Column('action_type', sa.Enum(
            'login', 'logout', 'register', 'password_change',
            'chat_start', 'chat_message',
            'document_upload', 'document_delete', 'document_deactivate',
            'faq_approve', 'faq_reject', 'faq_modify', 'faq_bulk_upload',
            'user_create', 'user_update', 'user_deactivate', 'user_role_change',
            'system_error', 'api_rate_limit',
            name='actiontype'
        ), nullable=False),
        sa.Column('description', sa.Text(), nullable=False),
        sa.Column('target_type', sa.String(length=50), nullable=True),
        sa.Column('target_id', sa.String(length=100), nullable=True),
        sa.Column('ip_address', sa.String(length=45), nullable=True),
        sa.Column('user_agent', sa.String(length=500), nullable=True),
        sa.Column('request_data', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('response_data', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('error_details', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('success', sa.Boolean(), nullable=False, default=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_activity_logs_user_id', 'activity_logs', ['user_id'], unique=False)
    op.create_index('ix_activity_logs_action_type', 'activity_logs', ['action_type'], unique=False)
    op.create_index('ix_activity_logs_created_at', 'activity_logs', ['created_at'], unique=False)


def downgrade() -> None:
    op.drop_table('activity_logs')
    op.drop_table('pending_qa')
    op.drop_table('messages')
    op.drop_table('conversations')
    op.drop_table('users')
    
    # Drop enums
    op.execute('DROP TYPE IF EXISTS actiontype')
    op.execute('DROP TYPE IF EXISTS qastatus')
    op.execute('DROP TYPE IF EXISTS messagerole')
    op.execute('DROP TYPE IF EXISTS userrole')
