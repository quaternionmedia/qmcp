"""add the columns an acted-on instruction carries

Revision ID: 626146afaf3f
Revises: b7ecf8eefe0a
Create Date: 2026-10-03 20:49:48.757021
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
import sqlmodel
from alembic import op
from sqlalchemy.dialects import sqlite  # noqa: F401 -- reflected types use it

revision: str = '626146afaf3f'
down_revision: str | None = 'b7ecf8eefe0a'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


INBOX_STATUSES = ('RECORDED', 'UNRESOLVED')
ACT_STATUSES = ('ASKING', 'CONSENTED', 'REFUSED', 'UNANSWERED', 'ACTING', 'DONE', 'FAILED')


def upgrade() -> None:
    # Adjusted from autogenerate, which emitted the columns and the index and
    # not the enum: it does not compare enum members, and SQLite stores the
    # column as a VARCHAR whose length the new members happen to fit. The type
    # is altered anyway, so the schema a reader reflects states the vocabulary
    # the model has rather than the two words it started with. Every column is
    # nullable, so no server default is owed to a table that has rows.
    with op.batch_alter_table('instructions', schema=None) as batch_op:
        batch_op.alter_column(
            'status',
            existing_type=sa.Enum(*INBOX_STATUSES, name='instructionstatus'),
            type_=sa.Enum(*INBOX_STATUSES, *ACT_STATUSES, name='instructionstatus'),
            existing_nullable=False)
        batch_op.add_column(sa.Column('consent_request_id', sqlmodel.sql.sqltypes.AutoString(), nullable=True))
        batch_op.add_column(sa.Column('runtime', sqlmodel.sql.sqltypes.AutoString(), nullable=True))
        batch_op.add_column(sa.Column('cwd', sqlmodel.sql.sqltypes.AutoString(), nullable=True))
        batch_op.add_column(sa.Column('outcome_text', sqlmodel.sql.sqltypes.AutoString(), nullable=True))
        batch_op.add_column(sa.Column('exit_code', sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column('acted_at', sa.DateTime(), nullable=True))
        batch_op.add_column(sa.Column('declared', sa.JSON(), nullable=True))
        batch_op.create_index(batch_op.f('ix_instructions_consent_request_id'), ['consent_request_id'], unique=False)

    # ### end Alembic commands ###


def downgrade() -> None:
    # As autogenerate emitted it, plus the enum going back to the inbox's two
    # words. A row that was acted on keeps a status the narrower type does not
    # name; SQLite does not check it, and `tests/test_db_migrations.py` runs
    # this against an empty table.
    with op.batch_alter_table('instructions', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_instructions_consent_request_id'))
        batch_op.drop_column('declared')
        batch_op.drop_column('acted_at')
        batch_op.drop_column('exit_code')
        batch_op.drop_column('outcome_text')
        batch_op.drop_column('cwd')
        batch_op.drop_column('runtime')
        batch_op.drop_column('consent_request_id')
        batch_op.alter_column(
            'status',
            existing_type=sa.Enum(*INBOX_STATUSES, *ACT_STATUSES, name='instructionstatus'),
            type_=sa.Enum(*INBOX_STATUSES, name='instructionstatus'),
            existing_nullable=False)

    # ### end Alembic commands ###
