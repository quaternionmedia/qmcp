"""Base utilities and common imports for the models module."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import field_validator
from sqlalchemy import JSON, Column
from sqlmodel import Field, SQLModel


def utc_now() -> datetime:
    """Return the current UTC datetime."""
    return datetime.now(UTC)


def generate_uuid() -> UUID:
    """Generate a new UUID4."""
    return uuid4()


def validate_identifier(value: str) -> str:
    """Validate and normalize an identifier string."""
    if not value.replace("_", "").replace("-", "").isalnum():
        raise ValueError("Identifier must be alphanumeric with underscores/hyphens")
    return value.lower()


__all__ = [
    "UTC",
    "datetime",
    "Any",
    "UUID",
    "uuid4",
    "field_validator",
    "JSON",
    "Column",
    "Field",
    "SQLModel",
    "utc_now",
    "generate_uuid",
    "validate_identifier",
]
