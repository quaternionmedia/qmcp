"""scope topology and component names to a project

Revision ID: e5a1c9d37f02
Revises: d4f87e6c2b91
Create Date: 2026-10-05
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "e5a1c9d37f02"
down_revision: str | None = "d4f87e6c2b91"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLES = {
    "topologies": ("ix_topologies_name", "uq_topologies_project_name"),
    "topology_components": (
        "ix_topology_components_name", "uq_topology_components_project_name"
    ),
}


def upgrade() -> None:
    for table, (name_index, unique_name) in _TABLES.items():
        with op.batch_alter_table(table, schema=None) as batch_op:
            batch_op.add_column(
                sa.Column(
                    "project", sa.String(length=64), nullable=False,
                    server_default="default",
                )
            )
            batch_op.drop_index(name_index)
            batch_op.create_index(name_index, ["name"], unique=False)
            batch_op.create_index(f"ix_{table}_project", ["project"], unique=False)
            batch_op.create_unique_constraint(unique_name, ["project", "name"])


def downgrade() -> None:
    for table, (name_index, unique_name) in _TABLES.items():
        with op.batch_alter_table(table, schema=None) as batch_op:
            batch_op.drop_constraint(unique_name, type_="unique")
            batch_op.drop_index(f"ix_{table}_project")
            batch_op.drop_index(name_index)
            batch_op.create_index(name_index, ["name"], unique=True)
            batch_op.drop_column("project")
