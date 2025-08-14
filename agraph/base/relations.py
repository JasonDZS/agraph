"""
Relation-related data structures and operations.

This module defines the Relation class and related operations for managing
relationships between entities in a knowledge graph.
"""

from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, Optional

from pydantic import Field, model_validator

from ..utils import get_type_value
from .base import GraphNodeBase, TextChunkMixin
from .mixins import PropertyMixin
from .types import RelationType, RelationTypeType

if TYPE_CHECKING:
    from .entities import Entity


class Relation(GraphNodeBase, TextChunkMixin, PropertyMixin):
    """Represents a relationship between two entities in a knowledge graph.

    A Relation connects two entities (head and tail) with a specific relationship
    type and additional metadata.

    Attributes:
        head_entity: The source entity of the relationship
        tail_entity: The target entity of the relationship
        relation_type: The type of relationship
        description: Human-readable description of the relation
    """

    head_entity: Optional["Entity"] = Field(default=None, description="Source entity")
    tail_entity: Optional["Entity"] = Field(default=None, description="Target entity")
    relation_type: RelationTypeType = Field(
        default=RelationType.RELATED_TO, description="Relation type"
    )
    description: str = Field(default="", description="Human-readable description")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Additional properties")
    text_chunks: set = Field(default_factory=set, description="Connected text chunk IDs")

    @model_validator(mode="after")
    def validate_entities_different(self) -> "Relation":
        """Ensure head and tail entities are different."""
        if self.head_entity and self.tail_entity and self.head_entity.id == self.tail_entity.id:
            raise ValueError("Head and tail entities must be different")
        return self

    def is_valid(self) -> bool:
        """Check if the relation is valid.

        A relation is considered valid if both head and tail entities exist
        and are different from each other.

        Returns:
            bool: True if the relation is valid, False otherwise.
        """
        return (
            self.head_entity is not None
            and self.tail_entity is not None
            and self.head_entity != self.tail_entity
        )

    def reverse(self) -> "Relation":
        """Create a reverse relation.

        Creates a new relation with head and tail entities swapped
        and the appropriate reverse relation type.

        Returns:
            Relation: A new relation with reversed direction.
        """
        return Relation(
            head_entity=self.tail_entity,
            tail_entity=self.head_entity,
            relation_type=self._get_reverse_relation_type(),
            properties=dict(self.properties),
            confidence=self.confidence,
            source=self.source,
            description=self.description,
            text_chunks=set(self.text_chunks),
        )

    def _get_reverse_relation_type(self) -> RelationTypeType:
        """Get the reverse relation type for the current relation.

        Maps certain relation types to their logical reverse. For symmetric
        relations, returns the same type.

        Returns:
            The reverse relation type.
        """
        # Handle enum or string types
        if isinstance(self.relation_type, str):
            try:
                rel_type = RelationType(self.relation_type)
            except ValueError:
                return self.relation_type
        else:
            rel_type = self.relation_type

        reverse_map = {
            RelationType.CONTAINS: RelationType.BELONGS_TO,
            RelationType.BELONGS_TO: RelationType.CONTAINS,
            RelationType.LOCATED_IN: RelationType.CONTAINS,
            RelationType.WORKS_FOR: RelationType.CONTAINS,
            RelationType.PART_OF: RelationType.CONTAINS,
            RelationType.FOUNDED_BY: RelationType.CREATES,
            RelationType.CREATES: RelationType.FOUNDED_BY,
            RelationType.DEVELOPS: RelationType.FOUNDED_BY,
            # Symmetric relations
            RelationType.REFERENCES: RelationType.REFERENCES,
            RelationType.SIMILAR_TO: RelationType.SIMILAR_TO,
            RelationType.SYNONYMS: RelationType.SYNONYMS,
            RelationType.RELATED_TO: RelationType.RELATED_TO,
        }
        return reverse_map.get(rel_type, rel_type)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the relation to a dictionary representation.

        Serializes the relation object to a dictionary format suitable
        for storage or transmission.

        Returns:
            Dict[str, Any]: Dictionary representation of the relation.
        """
        return {
            "id": self.id,
            "head_entity_id": self.head_entity.id if self.head_entity else None,
            "tail_entity_id": self.tail_entity.id if self.tail_entity else None,
            "relation_type": get_type_value(self.relation_type),
            "properties": self.properties,
            "confidence": self.confidence,
            "source": self.source,
            "description": self.description,
            "text_chunks": list(self.text_chunks),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(
        cls, data: Dict[str, Any], entities_map: Optional[Dict[str, "Entity"]] = None, **kwargs: Any
    ) -> "Relation":
        """Create a relation from a dictionary representation.

        Deserializes a relation object from a dictionary format, resolving
        entity references using the provided entities map.

        Args:
            data (Dict[str, Any]): Dictionary containing relation data.
            entities_map (Dict[str, Entity]): Map of entity IDs to Entity objects.

        Returns:
            Relation: The reconstructed relation object.
        """
        head_entity_id = data.get("head_entity_id")
        tail_entity_id = data.get("tail_entity_id")
        head_entity = entities_map.get(head_entity_id) if entities_map and head_entity_id else None
        tail_entity = entities_map.get(tail_entity_id) if entities_map and tail_entity_id else None

        relation_type_value = data.get("relation_type", "RELATED_TO")
        try:
            relation_type = RelationType(relation_type_value)
        except (ValueError, AttributeError):
            relation_type = RelationType.RELATED_TO

        relation_data = {
            "head_entity": head_entity,
            "tail_entity": tail_entity,
            "relation_type": relation_type,
            "properties": data.get("properties", {}),
            "confidence": data.get("confidence", 0.8),
            "source": data.get("source", ""),
            "description": data.get("description", ""),
            "text_chunks": set(data.get("text_chunks", [])),
        }

        if "id" in data:
            relation_data["id"] = data["id"]

        relation = cls(**relation_data)
        if "created_at" in data:
            relation.created_at = datetime.fromisoformat(data["created_at"])
        if "updated_at" in data:
            relation.updated_at = datetime.fromisoformat(data["updated_at"])
        return relation
