"""
Entity-related data structures and operations.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Set

from .types import EntityType, EntityTypeType


@dataclass
class Entity:
    """
    Entity class representing a knowledge graph entity.

    Attributes:
        id: Unique identifier for the entity
        name: Name of the entity
        entity_type: Type classification of the entity
        description: Detailed description of the entity
        properties: Additional properties as key-value pairs
        aliases: Alternative names for the entity
        confidence: Confidence score of entity extraction (0.0 to 1.0)
        source: Source document or origin of the entity
        text_chunks: Set of text chunk IDs that mention or contain this entity
        created_at: Timestamp when the entity was created
        updated_at: Timestamp when the entity was last updated
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    entity_type: EntityTypeType = EntityType.UNKNOWN
    description: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)
    aliases: List[str] = field(default_factory=list)
    confidence: float = 1.0
    source: str = ""
    text_chunks: Set[str] = field(default_factory=set)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def __hash__(self) -> int:
        """Return hash value of the entity based on its ID."""
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        """Check equality between entities based on their IDs."""
        if not isinstance(other, Entity):
            return NotImplemented
        return self.id == other.id

    def add_alias(self, alias: str) -> None:
        """
        Add an alias to the entity.

        Args:
            alias: The alias name to add
        """
        if alias and alias not in self.aliases:
            self.aliases.append(alias)

    def add_property(self, key: str, value: Any) -> None:
        """
        Add a property to the entity.

        Args:
            key: Property key
            value: Property value
        """
        self.properties[key] = value

    def get_property(self, key: str, default: Any = None) -> Any:
        """
        Get a property value from the entity.

        Args:
            key: Property key to retrieve
            default: Default value if key is not found

        Returns:
            Property value or default if not found
        """
        return self.properties.get(key, default)

    def add_text_chunk(self, chunk_id: str) -> None:
        """
        Add a text chunk connection to the entity.

        Args:
            chunk_id: The ID of the text chunk to connect
        """
        self.text_chunks.add(chunk_id)
        self.updated_at = datetime.now()

    def remove_text_chunk(self, chunk_id: str) -> None:
        """
        Remove a text chunk connection from the entity.

        Args:
            chunk_id: The ID of the text chunk to disconnect
        """
        self.text_chunks.discard(chunk_id)
        self.updated_at = datetime.now()

    def has_text_chunk(self, chunk_id: str) -> bool:
        """
        Check if the entity is connected to a specific text chunk.

        Args:
            chunk_id: The ID of the text chunk to check

        Returns:
            True if the text chunk is connected, False otherwise
        """
        return chunk_id in self.text_chunks

    def get_text_chunk_count(self) -> int:
        """
        Get the number of text chunks connected to this entity.

        Returns:
            Number of connected text chunks
        """
        return len(self.text_chunks)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert entity to dictionary representation.

        Returns:
            Dictionary containing all entity data with serializable values
        """
        from .utils import get_type_value

        return {
            "id": self.id,
            "name": self.name,
            "entity_type": get_type_value(self.entity_type),
            "description": self.description,
            "properties": self.properties,
            "aliases": self.aliases,
            "confidence": self.confidence,
            "source": self.source,
            "text_chunks": list(self.text_chunks),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Entity":
        """
        Create entity from dictionary data.

        Args:
            data: Dictionary containing entity data

        Returns:
            Entity instance created from the dictionary data
        """
        entity_type_value = data.get("entity_type", "UNKNOWN")
        try:
            entity_type = EntityType(entity_type_value)
        except (ValueError, AttributeError):
            entity_type = EntityType.UNKNOWN

        entity = cls(
            id=data.get("id", str(uuid.uuid4())),
            name=data.get("name", ""),
            entity_type=entity_type,
            description=data.get("description", ""),
            properties=data.get("properties", {}),
            aliases=data.get("aliases", []),
            confidence=data.get("confidence", 1.0),
            source=data.get("source", ""),
            text_chunks=set(data.get("text_chunks", [])),
        )
        if "created_at" in data:
            entity.created_at = datetime.fromisoformat(data["created_at"])
        if "updated_at" in data:
            entity.updated_at = datetime.fromisoformat(data["updated_at"])
        return entity
