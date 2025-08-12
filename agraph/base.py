"""
Base classes for the agraph package.

This module provides abstract base classes and shared functionality
for all data models in the agraph package.
"""

import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

from .mixins import SerializableMixin


class GraphNodeBase(BaseModel, SerializableMixin, ABC):
    """Abstract base class for all graph nodes (entities and relations).

    Provides common functionality for identification, confidence scoring,
    and basic validation.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0, description="Confidence score")
    source: str = Field(default="", description="Source of the information")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")

    class Config:
        """Pydantic configuration."""

        use_enum_values = True
        extra = "allow"
        validate_assignment = True

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        """Validate confidence score is between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        return v

    def touch(self) -> None:
        """Update the timestamp to current time."""
        self.updated_at = datetime.now()

    @abstractmethod
    def is_valid(self) -> bool:
        """Check if the node is valid."""

    def __hash__(self) -> int:
        """Return hash value based on ID."""
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        """Check equality based on ID."""
        if not isinstance(other, GraphNodeBase):
            return NotImplemented
        return self.id == other.id


class TextChunkMixin:
    """Mixin for objects that can be connected to text chunks."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if not hasattr(self, "text_chunks"):
            self.text_chunks: set[str] = set()

    def add_text_chunk(self, chunk_id: str) -> None:
        """Add a text chunk connection."""
        if not chunk_id:
            raise ValueError("Chunk ID cannot be empty")
        self.text_chunks.add(chunk_id)
        if hasattr(self, "touch"):
            self.touch()

    def remove_text_chunk(self, chunk_id: str) -> None:
        """Remove a text chunk connection."""
        self.text_chunks.discard(chunk_id)
        if hasattr(self, "touch"):
            self.touch()

    def has_text_chunk(self, chunk_id: str) -> bool:
        """Check if connected to a specific text chunk."""
        return chunk_id in self.text_chunks

    def get_text_chunk_count(self) -> int:
        """Get the number of connected text chunks."""
        return len(self.text_chunks)
