"""add the columns the models gained

Revision ID: adf10cb9cbff
Revises: 3dda531e16d2
Create Date: 2026-08-17 22:43:40.490092
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
import sqlmodel
from alembic import op

revision: str = 'adf10cb9cbff'
down_revision: str | None = '3dda531e16d2'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Adjusted from autogenerate, which is what its own comment asks for.
    #
    # Server defaults are added to the NOT NULL enum columns, so the
    # migration succeeds against tables that have rows ("Cannot add a NOT NULL
    # column with default value NULL" otherwise). The values match the models'
    # defaults: HealthStatus.UNKNOWN, Priority.NORMAL.
    #
    # Foreign keys are named, because batch mode -- which SQLite needs, as it
    # cannot ALTER a constraint -- rebuilds the table and must name what it
    # rebuilds. An unnamed one raises `ValueError: Constraint must have a name`
    # after the preceding add_column has committed. `alembic/env.py` sets a
    # naming convention so autogenerate output carries names of its own.
    with op.batch_alter_table('agent_instances', schema=None) as batch_op:
        batch_op.add_column(sa.Column('health', sa.Enum('HEALTHY', 'DEGRADED', 'UNHEALTHY', 'UNKNOWN', name='healthstatus'), nullable=False, server_default='UNKNOWN'))

    with op.batch_alter_table('executions', schema=None) as batch_op:
        batch_op.add_column(sa.Column('priority', sa.Enum('CRITICAL', 'HIGH', 'NORMAL', 'LOW', 'BACKGROUND', name='priority'), nullable=False, server_default='NORMAL'))
        batch_op.add_column(sa.Column('parent_execution_id', sa.Uuid(), nullable=True))
        batch_op.create_foreign_key(
            'fk_executions_parent_execution_id', 'executions',
            ['parent_execution_id'], ['id'])

    with op.batch_alter_table('messages', schema=None) as batch_op:
        batch_op.add_column(sa.Column('priority', sa.Enum('CRITICAL', 'HIGH', 'NORMAL', 'LOW', 'BACKGROUND', name='priority'), nullable=False, server_default='NORMAL'))
        batch_op.add_column(sa.Column('parent_message_id', sa.Uuid(), nullable=True))
        batch_op.create_foreign_key(
            'fk_messages_parent_message_id', 'messages',
            ['parent_message_id'], ['id'])

    # ### end Alembic commands ###


def downgrade() -> None:
    # Adjusted from autogenerate, the same way `upgrade` was.
    #
    # The constraints are named here too. Autogenerate emitted
    # `drop_constraint(None, ...)`, which the naming convention renders as
    # `fk_messages_` -- the column half of the pattern is empty because there
    # is no name to take it from -- and the downgrade then fails with
    # `No such constraint`. A downgrade nobody runs is a downgrade that does
    # not work, which is why there is a test that runs it.
    with op.batch_alter_table('messages', schema=None) as batch_op:
        batch_op.drop_constraint('fk_messages_parent_message_id', type_='foreignkey')
        batch_op.drop_column('parent_message_id')
        batch_op.drop_column('priority')

    with op.batch_alter_table('executions', schema=None) as batch_op:
        batch_op.drop_constraint('fk_executions_parent_execution_id', type_='foreignkey')
        batch_op.drop_column('parent_execution_id')
        batch_op.drop_column('priority')

    with op.batch_alter_table('agent_instances', schema=None) as batch_op:
        batch_op.drop_column('health')

    # ### end Alembic commands ###
