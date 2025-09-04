"""
Entity-related data structures and operations.
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List

from pydantic import Field, field_validator

from ...utils import get_type_value
from ..core.base import GraphNodeBase, TextChunkMixin
from ..core.mixins import PropertyMixin
from ..core.types import EntityType, EntityTypeType
from .positioning import PositionMixin


class Entity(GraphNodeBase, TextChunkMixin, PropertyMixin, PositionMixin):
    """Entity class representing a knowledge graph entity.

    Attributes:
        name: Name of the entity
        entity_type: Type classification of the entity
        description: Detailed description of the entity
        aliases: Alternative names for the entity
    """

    name: str = Field(default="", description="Name of the entity")
    entity_type: EntityTypeType = Field(default=EntityType.UNKNOWN, description="Entity type")
    description: str = Field(default="", description="Detailed description")
    aliases: List[str] = Field(default_factory=list, description="Alternative names")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Additional properties")
    text_chunks: set = Field(default_factory=set, description="Connected text chunk IDs")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate entity name is not empty when provided."""
        if v is not None and len(v.strip()) == 0:
            raise ValueError("Entity name cannot be empty string")
        return v.strip() if v else ""

    @field_validator("aliases")
    @classmethod
    def validate_aliases(cls, v: List[str]) -> List[str]:
        """Validate and clean aliases list."""
        return [alias.strip() for alias in v if alias and alias.strip()]

    def add_alias(self, alias: str) -> None:
        """Add an alias to the entity.

        Args:
            alias: The alias name to add
        """
        if alias and alias.strip() and alias.strip() not in self.aliases:
            self.aliases.append(alias.strip())
            self.touch()

    def is_valid(self) -> bool:
        """Check if the entity is valid.

        Returns:
            True if entity has a non-empty name, False otherwise
        """
        return bool(self.name and self.name.strip())

    def to_dict(self) -> Dict[str, Any]:
        """Convert entity to dictionary representation.

        Returns:
            Dictionary containing all entity data with serializable values
        """
        result = {
            "id": self.id,
            "name": self.name,
            "entity_type": get_type_value(self.entity_type),
            "description": self.description,
            "properties": self.properties,
            "aliases": self.aliases,
            # pylint: disable-next=duplicate-code
            "confidence": self.confidence,
            "source": self.source,
            "text_chunks": list(self.text_chunks),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

        # Add position information if available
        if self.position is not None:
            result["position"] = self.position.to_dict()

        return result

    @classmethod
    # pylint: disable=duplicate-code
    def from_dict(cls, data: Dict[str, Any], **kwargs: Any) -> "Entity":
        """
        Create entity from dictionary data.

        Args:
            data: Dictionary containing entity data

        Returns:
            Entity instance created from the dictionary data
        """
        from .positioning import Position  # Import here to avoid circular imports

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

        # Set position information if available
        if "position" in data and data["position"] is not None:
            entity.position = Position.from_dict(data["position"])

        if "created_at" in data:
            entity.created_at = datetime.fromisoformat(data["created_at"])
        if "updated_at" in data:
            entity.updated_at = datetime.fromisoformat(data["updated_at"])
        return entity
