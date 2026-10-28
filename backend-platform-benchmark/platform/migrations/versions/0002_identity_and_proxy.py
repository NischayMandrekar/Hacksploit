"""Add user identity, refresh-token sessions and user-owned jobs."""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0002_identity"
down_revision = "0001_init"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("email", sa.String(320), nullable=False),
        sa.Column("password_hash", sa.String(512), nullable=False),
        sa.Column("role", sa.String(20), nullable=False, server_default="user"),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("role IN ('user','admin')", name="ck_users_role"),
    )
    op.create_index("uq_users_email_normalized", "users", [sa.text("lower(email)")], unique=True)

    op.create_table(
        "auth_sessions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("family_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("token_hash", sa.String(64), nullable=False, unique=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("replaced_by_hash", sa.String(64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_auth_sessions_user_expiry", "auth_sessions", ["user_id", "expires_at"])
    op.create_index("ix_auth_sessions_family", "auth_sessions", ["family_id"])

    op.add_column("api_keys", sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=True))
    op.create_foreign_key("fk_api_keys_user_id_users", "api_keys", "users", ["user_id"], ["id"], ondelete="CASCADE")
    op.create_index("ix_api_keys_user_id", "api_keys", ["user_id"])

    op.add_column("jobs", sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=True))
    op.alter_column("jobs", "api_key_id", existing_type=postgresql.UUID(as_uuid=True), nullable=True)
    op.create_foreign_key("fk_jobs_user_id_users", "jobs", "users", ["user_id"], ["id"], ondelete="RESTRICT")
    op.create_check_constraint("ck_jobs_has_owner", "jobs", "api_key_id IS NOT NULL OR user_id IS NOT NULL")
    op.create_index("ix_jobs_user_created", "jobs", ["user_id", "created_at"])
    op.create_index(
        "uq_jobs_user_idempotency", "jobs", ["user_id", "idempotency_key"], unique=True,
        postgresql_where=sa.text("idempotency_key IS NOT NULL AND user_id IS NOT NULL AND api_key_id IS NULL"),
    )


def downgrade() -> None:
    op.drop_index("uq_jobs_user_idempotency", table_name="jobs")
    op.drop_index("ix_jobs_user_created", table_name="jobs")
    op.drop_constraint("ck_jobs_has_owner", "jobs", type_="check")
    op.drop_constraint("fk_jobs_user_id_users", "jobs", type_="foreignkey")
    op.drop_column("jobs", "user_id")
    op.alter_column("jobs", "api_key_id", existing_type=postgresql.UUID(as_uuid=True), nullable=False)
    op.drop_index("ix_api_keys_user_id", table_name="api_keys")
    op.drop_constraint("fk_api_keys_user_id_users", "api_keys", type_="foreignkey")
    op.drop_column("api_keys", "user_id")
    op.drop_index("ix_auth_sessions_family", table_name="auth_sessions")
    op.drop_index("ix_auth_sessions_user_expiry", table_name="auth_sessions")
    op.drop_table("auth_sessions")
    op.drop_index("uq_users_email_normalized", table_name="users")
    op.drop_table("users")
