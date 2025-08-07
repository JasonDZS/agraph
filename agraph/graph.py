"""
Knowledge graph core data structures.

This module provides the main KnowledgeGraph class for storing and managing
entities and relations in a graph structure with efficient indexing.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from .entities import Entity
from .relations import Relation


@dataclass
class KnowledgeGraph:
    """Knowledge graph class for managing entities and relations.

    This class provides a complete knowledge graph implementation with
    efficient indexing, CRUD operations, and graph traversal capabilities.

    Attributes:
        id: Unique identifier for the knowledge graph.
        name: Human-readable name for the knowledge graph.
        entities: Dictionary mapping entity IDs to Entity objects.
        relations: Dictionary mapping relation IDs to Relation objects.
        entity_index: Index mapping entity types to sets of entity IDs.
        relation_index: Index mapping relation types to sets of relation IDs.
        created_at: Timestamp when the graph was created.
        updated_at: Timestamp when the graph was last updated.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    entities: Dict[str, Entity] = field(default_factory=dict)
    relations: Dict[str, Relation] = field(default_factory=dict)
    entity_index: Dict[str, Set[str]] = field(default_factory=dict)  # Index entities by type
    relation_index: Dict[str, Set[str]] = field(default_factory=dict)  # Index relations by type
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def add_entity(self, entity: Entity) -> bool:
        """Add an entity to the knowledge graph.

        Args:
            entity: The Entity object to add.

        Returns:
            bool: True if entity was added successfully, False if it already exists.
        """
        if entity.id in self.entities:
            return False

        self.entities[entity.id] = entity
        self._index_entity(entity)
        self.updated_at = datetime.now()
        return True

    def add_relation(self, relation: Relation) -> bool:
        """Add a relation to the knowledge graph.

        Args:
            relation: The Relation object to add.

        Returns:
            bool: True if relation was added successfully, False if invalid or already exists.
        """
        if not relation.is_valid() or relation.id in self.relations:
            return False

        # Ensure related entities exist
        if (
            relation.head_entity is None
            or relation.tail_entity is None
            or relation.head_entity.id not in self.entities
            or relation.tail_entity.id not in self.entities
        ):
            return False

        self.relations[relation.id] = relation
        self._index_relation(relation)
        self.updated_at = datetime.now()
        return True

    def remove_entity(self, entity_id: str) -> bool:
        """Remove an entity and all its associated relations.

        Args:
            entity_id: The ID of the entity to remove.

        Returns:
            bool: True if entity was removed successfully, False if not found.
        """
        if entity_id not in self.entities:
            return False

        entity = self.entities[entity_id]

        # Remove associated relations
        relations_to_remove = []
        for relation in self.relations.values():
            if (relation.head_entity is not None and relation.head_entity.id == entity_id) or (
                relation.tail_entity is not None and relation.tail_entity.id == entity_id
            ):
                relations_to_remove.append(relation.id)

        for relation_id in relations_to_remove:
            self.remove_relation(relation_id)

        # Remove entity
        del self.entities[entity_id]
        self._unindex_entity(entity)
        self.updated_at = datetime.now()
        return True

    def remove_relation(self, relation_id: str) -> bool:
        """Remove a relation from the knowledge graph.

        Args:
            relation_id: The ID of the relation to remove.

        Returns:
            bool: True if relation was removed successfully, False if not found.
        """
        if relation_id not in self.relations:
            return False

        relation = self.relations[relation_id]
        del self.relations[relation_id]
        self._unindex_relation(relation)
        self.updated_at = datetime.now()
        return True

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get an entity by its ID.

        Args:
            entity_id: The ID of the entity to retrieve.

        Returns:
            Optional[Entity]: The entity if found, None otherwise.
        """
        return self.entities.get(entity_id)

    def get_relation(self, relation_id: str) -> Optional[Relation]:
        """Get a relation by its ID.

        Args:
            relation_id: The ID of the relation to retrieve.

        Returns:
            Optional[Relation]: The relation if found, None otherwise.
        """
        return self.relations.get(relation_id)

    def get_entities_by_type(self, entity_type: Any) -> List[Entity]:
        """Get all entities of a specific type.

        Args:
            entity_type: The type of entities to retrieve.

        Returns:
            List[Entity]: List of entities matching the specified type.
        """
        entity_type_value = getattr(entity_type, "value", str(entity_type))
        entity_ids = self.entity_index.get(entity_type_value, set())
        return [self.entities[entity_id] for entity_id in entity_ids if entity_id in self.entities]

    def get_relations_by_type(self, relation_type: Any) -> List[Relation]:
        """Get all relations of a specific type.

        Args:
            relation_type: The type of relations to retrieve.

        Returns:
            List[Relation]: List of relations matching the specified type.
        """
        relation_type_value = getattr(relation_type, "value", str(relation_type))
        relation_ids = self.relation_index.get(relation_type_value, set())
        return [self.relations[relation_id] for relation_id in relation_ids if relation_id in self.relations]

    def get_entity_relations(
        self, entity_id: str, relation_type: Optional[Any] = None, direction: str = "both"
    ) -> List[Relation]:
        """Get all relations involving a specific entity.

        Args:
            entity_id: The ID of the entity.
            relation_type: Optional filter by relation type.
            direction: Direction filter - "in", "out", or "both".

        Returns:
            List[Relation]: List of relations involving the entity.
        """
        if entity_id not in self.entities:
            return []

        entity_relations = []
        for relation in self.relations.values():
            if relation_type and relation.relation_type != relation_type:
                continue

            if (
                direction in ["out", "both"]
                and relation.head_entity is not None
                and relation.head_entity.id == entity_id
            ):
                entity_relations.append(relation)
            elif (
                direction in ["in", "both"]
                and relation.tail_entity is not None
                and relation.tail_entity.id == entity_id
            ):
                entity_relations.append(relation)

        return entity_relations

    def get_neighbors(
        self, entity_id: str, relation_type: Optional[Any] = None, direction: str = "both"
    ) -> List[Entity]:
        """Get neighboring entities connected to a specific entity.

        Args:
            entity_id: The ID of the entity.
            relation_type: Optional filter by relation type.
            direction: Direction filter - "in", "out", or "both".

        Returns:
            List[Entity]: List of neighboring entities.
        """
        relations = self.get_entity_relations(entity_id, relation_type, direction)
        neighbors: List[Entity] = []

        for relation in relations:
            if (
                relation.head_entity is not None
                and relation.head_entity.id == entity_id
                and relation.tail_entity is not None
            ):
                neighbors.append(relation.tail_entity)
            elif (
                relation.tail_entity is not None
                and relation.tail_entity.id == entity_id
                and relation.head_entity is not None
            ):
                neighbors.append(relation.head_entity)

        return neighbors

    def get_basic_statistics(self) -> Dict[str, Any]:
        """Get basic statistics about the knowledge graph.

        Returns:
            Dict[str, Any]: Dictionary containing basic graph statistics.
        """
        return {
            "total_entities": len(self.entities),
            "total_relations": len(self.relations),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    def _index_entity(self, entity: Entity) -> None:
        """Add entity to type index for efficient lookup.

        Args:
            entity: The entity to index.
        """
        entity_type = getattr(entity.entity_type, "value", str(entity.entity_type))
        if entity_type not in self.entity_index:
            self.entity_index[entity_type] = set()
        self.entity_index[entity_type].add(entity.id)

    def _unindex_entity(self, entity: Entity) -> None:
        """Remove entity from type index.

        Args:
            entity: The entity to remove from index.
        """
        entity_type = getattr(entity.entity_type, "value", str(entity.entity_type))
        if entity_type in self.entity_index:
            self.entity_index[entity_type].discard(entity.id)

    def _index_relation(self, relation: Relation) -> None:
        """Add relation to type index for efficient lookup.

        Args:
            relation: The relation to index.
        """
        relation_type = getattr(relation.relation_type, "value", str(relation.relation_type))
        if relation_type not in self.relation_index:
            self.relation_index[relation_type] = set()
        self.relation_index[relation_type].add(relation.id)

    def _unindex_relation(self, relation: Relation) -> None:
        """Remove relation from type index.

        Args:
            relation: The relation to remove from index.
        """
        relation_type = getattr(relation.relation_type, "value", str(relation.relation_type))
        if relation_type in self.relation_index:
            self.relation_index[relation_type].discard(relation.id)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the knowledge graph to a dictionary.

        Returns:
            Dict[str, Any]: Dictionary representation of the knowledge graph.
        """
        return {
            "id": self.id,
            "name": self.name,
            "entities": {eid: entity.to_dict() for eid, entity in self.entities.items()},
            "relations": {rid: relation.to_dict() for rid, relation in self.relations.items()},
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeGraph":
        """Create a knowledge graph from a dictionary.

        Args:
            data: Dictionary containing knowledge graph data.

        Returns:
            KnowledgeGraph: A new KnowledgeGraph instance.
        """
        graph = cls(id=data.get("id", str(uuid.uuid4())), name=data.get("name", ""))

        # Restore entities
        entities_data = data.get("entities", {})
        for entity_data in entities_data.values():
            entity = Entity.from_dict(entity_data)
            graph.add_entity(entity)

        # Restore relations
        relations_data = data.get("relations", {})
        for relation_data in relations_data.values():
            relation = Relation.from_dict(relation_data, graph.entities)
            if relation.head_entity and relation.tail_entity:
                graph.add_relation(relation)

        if "created_at" in data:
            graph.created_at = datetime.fromisoformat(data["created_at"])
        if "updated_at" in data:
            graph.updated_at = datetime.fromisoformat(data["updated_at"])

        return graph
