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
from .text import TextChunk


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
        text_chunks: Dictionary mapping text chunk IDs to TextChunk objects.
        entity_index: Index mapping entity types to sets of entity IDs.
        relation_index: Index mapping relation types to sets of relation IDs.
        text_chunk_index: Index mapping chunk types to sets of text chunk IDs.
        created_at: Timestamp when the graph was created.
        updated_at: Timestamp when the graph was last updated.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    entities: Dict[str, Entity] = field(default_factory=dict)
    relations: Dict[str, Relation] = field(default_factory=dict)
    text_chunks: Dict[str, TextChunk] = field(default_factory=dict)
    entity_index: Dict[str, Set[str]] = field(default_factory=dict)  # Index entities by type
    relation_index: Dict[str, Set[str]] = field(default_factory=dict)  # Index relations by type
    text_chunk_index: Dict[str, Set[str]] = field(default_factory=dict)  # Index text chunks by type
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

    def add_text_chunk(self, text_chunk: TextChunk) -> bool:
        """Add a text chunk to the knowledge graph.

        Args:
            text_chunk: The TextChunk object to add.

        Returns:
            bool: True if text chunk was added successfully, False if it already exists.
        """
        if text_chunk.id in self.text_chunks:
            return False

        self.text_chunks[text_chunk.id] = text_chunk
        self._index_text_chunk(text_chunk)
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

    def remove_text_chunk(self, chunk_id: str) -> bool:
        """Remove a text chunk from the knowledge graph.

        Args:
            chunk_id: The ID of the text chunk to remove.

        Returns:
            bool: True if text chunk was removed successfully, False if not found.
        """
        if chunk_id not in self.text_chunks:
            return False

        text_chunk = self.text_chunks[chunk_id]

        # Remove connections from entities and relations
        for entity in self.entities.values():
            entity.remove_text_chunk(chunk_id)

        for relation in self.relations.values():
            relation.remove_text_chunk(chunk_id)

        del self.text_chunks[chunk_id]
        self._unindex_text_chunk(text_chunk)
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

    def get_text_chunk(self, chunk_id: str) -> Optional[TextChunk]:
        """Get a text chunk by its ID.

        Args:
            chunk_id: The ID of the text chunk to retrieve.

        Returns:
            Optional[TextChunk]: The text chunk if found, None otherwise.
        """
        return self.text_chunks.get(chunk_id)

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

    def get_text_chunks_by_type(self, chunk_type: str) -> List[TextChunk]:
        """Get all text chunks of a specific type.

        Args:
            chunk_type: The type of text chunks to retrieve.

        Returns:
            List[TextChunk]: List of text chunks matching the specified type.
        """
        chunk_ids = self.text_chunk_index.get(chunk_type, set())
        return [self.text_chunks[chunk_id] for chunk_id in chunk_ids if chunk_id in self.text_chunks]

    def get_text_chunks_by_source(self, source: str) -> List[TextChunk]:
        """Get all text chunks from a specific source.

        Args:
            source: The source to filter by.

        Returns:
            List[TextChunk]: List of text chunks from the specified source.
        """
        return [chunk for chunk in self.text_chunks.values() if chunk.source == source]

    def get_entity_text_chunks(self, entity_id: str) -> List[TextChunk]:
        """Get all text chunks connected to a specific entity.

        Args:
            entity_id: The ID of the entity.

        Returns:
            List[TextChunk]: List of text chunks connected to the entity.
        """
        if entity_id not in self.entities:
            return []

        return [chunk for chunk in self.text_chunks.values() if chunk.has_entity(entity_id)]

    def get_relation_text_chunks(self, relation_id: str) -> List[TextChunk]:
        """Get all text chunks connected to a specific relation.

        Args:
            relation_id: The ID of the relation.

        Returns:
            List[TextChunk]: List of text chunks connected to the relation.
        """
        if relation_id not in self.relations:
            return []

        return [chunk for chunk in self.text_chunks.values() if chunk.has_relation(relation_id)]

    def connect_text_chunk_to_entity(self, chunk_id: str, entity_id: str) -> bool:
        """Connect a text chunk to an entity.

        Args:
            chunk_id: The ID of the text chunk.
            entity_id: The ID of the entity.

        Returns:
            bool: True if connection was successful, False otherwise.
        """
        if chunk_id not in self.text_chunks or entity_id not in self.entities:
            return False

        text_chunk = self.text_chunks[chunk_id]
        entity = self.entities[entity_id]

        text_chunk.add_entity(entity_id)
        entity.add_text_chunk(chunk_id)

        self.updated_at = datetime.now()
        return True

    def connect_text_chunk_to_relation(self, chunk_id: str, relation_id: str) -> bool:
        """Connect a text chunk to a relation.

        Args:
            chunk_id: The ID of the text chunk.
            relation_id: The ID of the relation.

        Returns:
            bool: True if connection was successful, False otherwise.
        """
        if chunk_id not in self.text_chunks or relation_id not in self.relations:
            return False

        text_chunk = self.text_chunks[chunk_id]
        relation = self.relations[relation_id]

        text_chunk.add_relation(relation_id)
        relation.add_text_chunk(chunk_id)

        self.updated_at = datetime.now()
        return True

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
            "total_text_chunks": len(self.text_chunks),
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

    def _index_text_chunk(self, text_chunk: TextChunk) -> None:
        """Add text chunk to type index for efficient lookup.

        Args:
            text_chunk: The text chunk to index.
        """
        chunk_type = text_chunk.chunk_type
        if chunk_type not in self.text_chunk_index:
            self.text_chunk_index[chunk_type] = set()
        self.text_chunk_index[chunk_type].add(text_chunk.id)

    def _unindex_text_chunk(self, text_chunk: TextChunk) -> None:
        """Remove text chunk from type index.

        Args:
            text_chunk: The text chunk to remove from index.
        """
        chunk_type = text_chunk.chunk_type
        if chunk_type in self.text_chunk_index:
            self.text_chunk_index[chunk_type].discard(text_chunk.id)

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
            "text_chunks": {cid: chunk.to_dict() for cid, chunk in self.text_chunks.items()},
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

        # Restore text chunks
        text_chunks_data = data.get("text_chunks", {})
        for chunk_data in text_chunks_data.values():
            text_chunk = TextChunk.from_dict(chunk_data)
            graph.add_text_chunk(text_chunk)

        if "created_at" in data:
            graph.created_at = datetime.fromisoformat(data["created_at"])
        if "updated_at" in data:
            graph.updated_at = datetime.fromisoformat(data["updated_at"])

        return graph
