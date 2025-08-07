"""
Relation-related data structures and operations.

This module defines the Relation class and related operations for managing
relationships between entities in a knowledge graph.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

from .entities import Entity
from .types import RelationType, RelationTypeType


@dataclass
class Relation:
    """Represents a relationship between two entities in a knowledge graph.

    A Relation connects two entities (head and tail) with a specific relationship
    type and additional metadata such as confidence score, properties, and source.

    Attributes:
        id (str): Unique identifier for the relation.
        head_entity (Optional[Entity]): The source entity of the relationship.
        tail_entity (Optional[Entity]): The target entity of the relationship.
        relation_type (Any): The type of relationship (from RelationType enum).
        properties (Dict[str, Any]): Additional properties for the relation.
        confidence (float): Confidence score for the relation (0.0 to 1.0).
        source (str): Source of the relation information.
        description (str): Human-readable description of the relation.
        created_at (datetime): Timestamp when the relation was created.
        updated_at (datetime): Timestamp when the relation was last updated.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    head_entity: Optional[Entity] = None
    tail_entity: Optional[Entity] = None
    relation_type: RelationTypeType = RelationType.RELATED_TO
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    source: str = ""
    description: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Relation):
            return self.id == other.id
        return False

    def add_property(self, key: str, value: Any) -> None:
        """Add a property to the relation.

        Args:
            key (str): The property key.
            value (Any): The property value.
        """
        self.properties[key] = value

    def get_property(self, key: str, default: Any = None) -> Any:
        """Get a property value from the relation.

        Args:
            key (str): The property key to retrieve.
            default (Any, optional): Default value if key not found. Defaults to None.

        Returns:
            Any: The property value or default if not found.
        """
        return self.properties.get(key, default)

    def is_valid(self) -> bool:
        """Check if the relation is valid.

        A relation is considered valid if both head and tail entities exist
        and are different from each other.

        Returns:
            bool: True if the relation is valid, False otherwise.
        """
        return self.head_entity is not None and self.tail_entity is not None and self.head_entity != self.tail_entity

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
            properties=self.properties.copy(),
            confidence=self.confidence,
            source=self.source,
        )

    def _get_reverse_relation_type(self) -> Any:
        """Get the reverse relation type for the current relation.

        Maps certain relation types to their logical reverse. For symmetric
        relations, returns the same type.

        Returns:
            Any: The reverse relation type.
        """
        reverse_map = {
            getattr(RelationType, "CONTAINS", None): getattr(RelationType, "BELONGS_TO", None),
            getattr(RelationType, "BELONGS_TO", None): getattr(RelationType, "CONTAINS", None),
            getattr(RelationType, "REFERENCES", None): getattr(RelationType, "REFERENCES", None),
            getattr(RelationType, "SIMILAR_TO", None): getattr(RelationType, "SIMILAR_TO", None),
            getattr(RelationType, "SYNONYMS", None): getattr(RelationType, "SYNONYMS", None),
        }
        return reverse_map.get(self.relation_type, self.relation_type)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the relation to a dictionary representation.

        Serializes the relation object to a dictionary format suitable
        for storage or transmission.

        Returns:
            Dict[str, Any]: Dictionary representation of the relation.
        """
        from .utils import get_type_value

        return {
            "id": self.id,
            "head_entity_id": self.head_entity.id if self.head_entity else None,
            "tail_entity_id": self.tail_entity.id if self.tail_entity else None,
            "relation_type": get_type_value(self.relation_type),
            "properties": self.properties,
            "confidence": self.confidence,
            "source": self.source,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], entities_map: Dict[str, Entity]) -> "Relation":
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
        head_entity = entities_map.get(head_entity_id) if head_entity_id else None
        tail_entity = entities_map.get(tail_entity_id) if tail_entity_id else None

        relation_type_value = data.get("relation_type", "RELATED_TO")
        try:
            relation_type = RelationType(relation_type_value)
        except (ValueError, AttributeError):
            relation_type = RelationType.RELATED_TO

        relation = cls(
            id=data.get("id", str(uuid.uuid4())),
            head_entity=head_entity,
            tail_entity=tail_entity,
            relation_type=relation_type,
            properties=data.get("properties", {}),
            confidence=data.get("confidence", 1.0),
            source=data.get("source", ""),
        )
        if "created_at" in data:
            relation.created_at = datetime.fromisoformat(data["created_at"])
        return relation
