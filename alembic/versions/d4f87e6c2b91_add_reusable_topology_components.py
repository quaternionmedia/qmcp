"""add reusable topology components

Revision ID: d4f87e6c2b91
Revises: 626146afaf3f
Create Date: 2026-10-05
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "d4f87e6c2b91"
down_revision: str | None = "626146afaf3f"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "topology_components",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=64), nullable=False),
        sa.Column("description", sa.String(length=512), nullable=False),
        sa.Column("instruction", sa.String(length=2048), nullable=False),
        sa.Column("version", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    with op.batch_alter_table("topology_components", schema=None) as batch_op:
        batch_op.create_index(
            "ix_topology_components_name", ["name"], unique=True
        )


def downgrade() -> None:
    with op.batch_alter_table("topology_components", schema=None) as batch_op:
        batch_op.drop_index("ix_topology_components_name")
    op.drop_table("topology_components")
