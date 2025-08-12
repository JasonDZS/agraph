"""
Text chunk related data structures and operations.

This module defines the TextChunk class and related operations for managing
text segments and their relationships with entities and relations in a knowledge graph.
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field

from agraph.base.entities import Entity

from .mixins import SerializableMixin
from .relations import Relation


class TextChunk(BaseModel, SerializableMixin):
    """Represents a text chunk with connections to entities and relations.

    A TextChunk represents a segment of text (document, paragraph, sentence, etc.)
    that can be connected to entities and relations in a knowledge graph for
    contextual understanding and retrieval.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str = Field(default="")
    title: str = Field(default="")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    source: str = Field(default="")
    start_index: Optional[int] = Field(default=None)
    end_index: Optional[int] = Field(default=None)
    chunk_type: str = Field(default="paragraph")
    language: str = Field(default="zh")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    embedding: Optional[List[float]] = Field(default=None)
    entities: Set[str] = Field(default_factory=set)
    relations: Set[str] = Field(default_factory=set)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    class Config:
        """Pydantic configuration."""

        extra = "allow"

    def __hash__(self) -> int:
        """Return hash value of the text chunk based on its ID."""
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        """Check equality between text chunks based on their IDs."""
        if isinstance(other, TextChunk):
            return self.id == other.id
        return False

    def touch(self) -> None:
        """Update the timestamp to current time."""
        self.updated_at = datetime.now()

    def add_entity(self, entity_id: str) -> None:
        """Add an entity connection to the text chunk."""
        self.entities.add(entity_id)
        self.touch()

    def remove_entity(self, entity_id: str) -> None:
        """Remove an entity connection from the text chunk."""
        self.entities.discard(entity_id)
        self.touch()

    def add_relation(self, relation_id: str) -> None:
        """Add a relation connection to the text chunk."""
        self.relations.add(relation_id)
        self.touch()

    def remove_relation(self, relation_id: str) -> None:
        """Remove a relation connection from the text chunk."""
        self.relations.discard(relation_id)
        self.touch()

    def has_entity(self, entity_id: str) -> bool:
        """Check if the text chunk is connected to a specific entity.

        Args:
            entity_id (str): The ID of the entity to check.

        Returns:
            bool: True if the entity is connected, False otherwise.
        """
        return entity_id in self.entities

    def has_relation(self, relation_id: str) -> bool:
        """Check if the text chunk is connected to a specific relation.

        Args:
            relation_id (str): The ID of the relation to check.

        Returns:
            bool: True if the relation is connected, False otherwise.
        """
        return relation_id in self.relations

    def get_connected_entities(self, entities_map: Dict[str, Entity]) -> List[Entity]:
        """Get all entities connected to this text chunk.

        Args:
            entities_map (Dict[str, Entity]): Map of entity IDs to Entity objects.

        Returns:
            List[Entity]: List of connected Entity objects.
        """
        return [entities_map[entity_id] for entity_id in self.entities if entity_id in entities_map]

    def get_connected_relations(self, relations_map: Dict[str, Relation]) -> List[Relation]:
        """Get all relations connected to this text chunk.

        Args:
            relations_map (Dict[str, Relation]): Map of relation IDs to Relation objects.

        Returns:
            List[Relation]: List of connected Relation objects.
        """
        return [
            relations_map[relation_id]
            for relation_id in self.relations
            if relation_id in relations_map
        ]

    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to the text chunk."""
        self.metadata[key] = value
        self.touch()

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata value from the text chunk."""
        return self.metadata.get(key, default)

    def get_text_length(self) -> int:
        """Get the length of the text content.

        Returns:
            int: Number of characters in the content.
        """
        return len(self.content)

    def get_position_info(self) -> Dict[str, Optional[int]]:
        """Get position information in the original document.

        Returns:
            Dict[str, Optional[int]]: Dictionary with start_index, end_index, and length.
        """
        length = None
        if self.start_index is not None and self.end_index is not None:
            length = self.end_index - self.start_index
        return {
            "start_index": self.start_index,
            "end_index": self.end_index,
            "length": length,
        }

    def is_valid(self) -> bool:
        """Check if the text chunk is valid.

        A text chunk is considered valid if it has non-empty content.

        Returns:
            bool: True if the text chunk is valid, False otherwise.
        """
        return bool(self.content.strip())

    def get_summary(self) -> str:
        """Get a summary of the text chunk.

        Returns:
            str: A truncated version of the content for display purposes.
        """
        max_length = 100
        if len(self.content) <= max_length:
            return self.content
        return self.content[:max_length] + "..."

    def to_dict(self) -> Dict[str, Any]:
        """Convert the text chunk to a dictionary representation.

        Serializes the text chunk object to a dictionary format suitable
        for storage or transmission.

        Returns:
            Dict[str, Any]: Dictionary representation of the text chunk.
        """
        return {
            "id": self.id,
            "content": self.content,
            "title": self.title,
            "metadata": self.metadata,
            "source": self.source,
            "start_index": self.start_index,
            "end_index": self.end_index,
            "chunk_type": self.chunk_type,
            "language": self.language,
            "confidence": self.confidence,
            "embedding": self.embedding,
            "entities": list(self.entities),
            "relations": list(self.relations),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], **kwargs: Any) -> "TextChunk":
        """Create a text chunk from a dictionary representation.

        Deserializes a text chunk object from a dictionary format.

        Args:
            data (Dict[str, Any]): Dictionary containing text chunk data.

        Returns:
            TextChunk: The reconstructed text chunk object.
        """
        chunk = cls(
            id=data.get("id", str(uuid.uuid4())),
            content=data.get("content", ""),
            title=data.get("title", ""),
            metadata=data.get("metadata", {}),
            source=data.get("source", ""),
            start_index=data.get("start_index"),
            end_index=data.get("end_index"),
            chunk_type=data.get("chunk_type", "paragraph"),
            language=data.get("language", "zh"),
            confidence=data.get("confidence", 1.0),
            embedding=data.get("embedding"),
            entities=set(data.get("entities", [])),
            relations=set(data.get("relations", [])),
        )

        if "created_at" in data:
            chunk.created_at = datetime.fromisoformat(data["created_at"])
        if "updated_at" in data:
            chunk.updated_at = datetime.fromisoformat(data["updated_at"])

        return chunk

    def calculate_similarity(self, other: "TextChunk") -> float:
        """Calculate similarity with another text chunk based on shared entities and relations.

        Args:
            other (TextChunk): Another text chunk to compare with.

        Returns:
            float: Similarity score between 0.0 and 1.0.
        """
        # Calculate entity overlap
        entity_intersection = len(self.entities.intersection(other.entities))
        entity_union = len(self.entities.union(other.entities))
        entity_similarity = entity_intersection / entity_union if entity_union > 0 else 0.0

        # Calculate relation overlap
        relation_intersection = len(self.relations.intersection(other.relations))
        relation_union = len(self.relations.union(other.relations))
        relation_similarity = relation_intersection / relation_union if relation_union > 0 else 0.0

        # Weighted average (can be adjusted based on use case)
        return entity_similarity * 0.6 + relation_similarity * 0.4
