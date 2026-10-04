"""add the instructions table

Revision ID: b7ecf8eefe0a
Revises: adf10cb9cbff
Create Date: 2026-10-03 15:30:53.090728
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
import sqlmodel
from alembic import op
from sqlalchemy.dialects import sqlite  # noqa: F401 -- reflected types use it

revision: str = 'b7ecf8eefe0a'
down_revision: str | None = 'adf10cb9cbff'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # As autogenerate emitted it. Neither adjustment the previous revision
    # records applies to a new table: no column is added to a table that has
    # rows, so no server default is owed, and there is no foreign key to name.
    op.create_table('instructions',
    sa.Column('id', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
    sa.Column('text', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
    sa.Column('project', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
    sa.Column('source', sa.Enum('VOICE', 'TYPED', 'PAGE', name='instructionsource'), nullable=False),
    sa.Column('status', sa.Enum('RECORDED', 'UNRESOLVED', name='instructionstatus'), nullable=False),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.Column('updated_at', sa.DateTime(), nullable=False),
    sa.Column('detail', sa.JSON(), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('instructions', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_instructions_created_at'), ['created_at'], unique=False)
        batch_op.create_index(batch_op.f('ix_instructions_project'), ['project'], unique=False)
        batch_op.create_index(batch_op.f('ix_instructions_source'), ['source'], unique=False)
        batch_op.create_index(batch_op.f('ix_instructions_status'), ['status'], unique=False)

    # ### end Alembic commands ###


def downgrade() -> None:
    # As autogenerate emitted it; `tests/test_db_migrations.py` runs it.
    with op.batch_alter_table('instructions', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_instructions_status'))
        batch_op.drop_index(batch_op.f('ix_instructions_source'))
        batch_op.drop_index(batch_op.f('ix_instructions_project'))
        batch_op.drop_index(batch_op.f('ix_instructions_created_at'))

    op.drop_table('instructions')
    # ### end Alembic commands ###
