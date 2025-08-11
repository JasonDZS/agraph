"""
Text chunk related data structures and operations.

This module defines the TextChunk class and related operations for managing
text segments and their relationships with entities and relations in a knowledge graph.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from .entities import Entity
from .relations import Relation


@dataclass
class TextChunk:
    """Represents a text chunk with connections to entities and relations.

    A TextChunk represents a segment of text (document, paragraph, sentence, etc.)
    that can be connected to entities and relations in a knowledge graph for
    contextual understanding and retrieval.

    Attributes:
        id (str): Unique identifier for the text chunk.
        content (str): The actual text content of the chunk.
        title (str): Optional title or header for the text chunk.
        metadata (Dict[str, Any]): Additional metadata about the text chunk.
        source (str): Source document or origin of the text chunk.
        start_index (Optional[int]): Start position in the original document.
        end_index (Optional[int]): End position in the original document.
        chunk_type (str): Type of text chunk (paragraph, sentence, document, etc.).
        language (str): Language of the text content.
        confidence (float): Confidence score for the text chunk extraction (0.0 to 1.0).
        embedding (Optional[List[float]]): Vector embedding of the text chunk.
        entities (Set[str]): Set of entity IDs mentioned/extracted from this chunk.
        relations (Set[str]): Set of relation IDs extracted from this chunk.
        created_at (datetime): Timestamp when the chunk was created.
        updated_at (datetime): Timestamp when the chunk was last updated.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    title: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = ""
    start_index: Optional[int] = None
    end_index: Optional[int] = None
    chunk_type: str = "paragraph"
    language: str = "zh"
    confidence: float = 1.0
    embedding: Optional[List[float]] = None
    entities: Set[str] = field(default_factory=set)
    relations: Set[str] = field(default_factory=set)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def __hash__(self) -> int:
        """Return hash value of the text chunk based on its ID."""
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        """Check equality between text chunks based on their IDs."""
        if isinstance(other, TextChunk):
            return self.id == other.id
        return False

    def add_entity(self, entity_id: str) -> None:
        """Add an entity connection to the text chunk.

        Args:
            entity_id (str): The ID of the entity to connect.
        """
        self.entities.add(entity_id)
        self.updated_at = datetime.now()

    def remove_entity(self, entity_id: str) -> None:
        """Remove an entity connection from the text chunk.

        Args:
            entity_id (str): The ID of the entity to disconnect.
        """
        self.entities.discard(entity_id)
        self.updated_at = datetime.now()

    def add_relation(self, relation_id: str) -> None:
        """Add a relation connection to the text chunk.

        Args:
            relation_id (str): The ID of the relation to connect.
        """
        self.relations.add(relation_id)
        self.updated_at = datetime.now()

    def remove_relation(self, relation_id: str) -> None:
        """Remove a relation connection from the text chunk.

        Args:
            relation_id (str): The ID of the relation to disconnect.
        """
        self.relations.discard(relation_id)
        self.updated_at = datetime.now()

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
        return [relations_map[relation_id] for relation_id in self.relations if relation_id in relations_map]

    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to the text chunk.

        Args:
            key (str): The metadata key.
            value (Any): The metadata value.
        """
        self.metadata[key] = value
        self.updated_at = datetime.now()

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata value from the text chunk.

        Args:
            key (str): The metadata key to retrieve.
            default (Any, optional): Default value if key not found. Defaults to None.

        Returns:
            Any: The metadata value or default if not found.
        """
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
    def from_dict(cls, data: Dict[str, Any]) -> "TextChunk":
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


@dataclass
class TextChunkConnection:
    """Represents a connection between a text chunk and an entity or relation.

    This class provides additional metadata about the connection, such as
    the type of mention, confidence, and position within the text.
    """

    chunk_id: str
    target_id: str  # Entity or Relation ID
    target_type: str  # "entity" or "relation"
    connection_type: str = "mention"  # mention, extraction, reference, etc.
    confidence: float = 1.0
    start_position: Optional[int] = None
    end_position: Optional[int] = None
    context: str = ""  # Surrounding context
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert connection to dictionary representation."""
        return {
            "chunk_id": self.chunk_id,
            "target_id": self.target_id,
            "target_type": self.target_type,
            "connection_type": self.connection_type,
            "confidence": self.confidence,
            "start_position": self.start_position,
            "end_position": self.end_position,
            "context": self.context,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TextChunkConnection":
        """Create connection from dictionary representation."""
        connection = cls(
            chunk_id=data["chunk_id"],
            target_id=data["target_id"],
            target_type=data["target_type"],
            connection_type=data.get("connection_type", "mention"),
            confidence=data.get("confidence", 1.0),
            start_position=data.get("start_position"),
            end_position=data.get("end_position"),
            context=data.get("context", ""),
            metadata=data.get("metadata", {}),
        )

        if "created_at" in data:
            connection.created_at = datetime.fromisoformat(data["created_at"])

        return connection
