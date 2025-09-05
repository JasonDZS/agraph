"""
Manager classes for knowledge graph operations.

This module provides manager classes to handle different aspects of knowledge graph
operations, reducing the complexity of the main KnowledgeGraph class.
"""

from typing import Callable, Dict, List, Optional, Union

from ..core.types import ClusterType, EntityType, RelationType
from ..models.clusters import Cluster
from ..models.entities import Entity
from ..models.relations import Relation
from ..models.text import TextChunk


class EntityManager:
    """Manager for entity operations in the knowledge graph."""

    def __init__(self, entities: Dict[str, Entity], touch_callback: Callable[[], None]) -> None:
        """Initialize the EntityManager.

        Args:
            entities: Dictionary of entities
            touch_callback: Callback to update graph timestamp
        """
        self.entities = entities
        self._touch = touch_callback

    def add_entity(self, entity: Entity) -> None:
        """Add an entity to the knowledge graph."""
        self.entities[entity.id] = entity
        self._touch()

    def remove_entity(
        self,
        entity_id: str,
        relations: Dict[str, Relation],
        clusters: Dict[str, Cluster],
        text_chunks: Dict[str, TextChunk],
    ) -> bool:
        """Remove an entity and update references."""
        if entity_id not in self.entities:
            return False

        # Remove entity from relations
        relations_to_remove = []
        for relation in relations.values():
            if (relation.head_entity and relation.head_entity.id == entity_id) or (
                relation.tail_entity and relation.tail_entity.id == entity_id
            ):
                relations_to_remove.append(relation.id)

        for rel_id in relations_to_remove:
            relations.pop(rel_id, None)

        # Remove entity from clusters
        for cluster in clusters.values():
            cluster.remove_entity(entity_id)

        # Remove entity from text chunks
        for text_chunk in text_chunks.values():
            text_chunk.remove_entity(entity_id)

        del self.entities[entity_id]
        self._touch()
        return True

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get an entity by ID."""
        return self.entities.get(entity_id)

    def get_entities_by_type(self, entity_type: Union[EntityType, str]) -> List[Entity]:
        """Get all entities of a specific type."""
        return [entity for entity in self.entities.values() if entity.entity_type == entity_type]

    def search_entities(self, query: str, limit: int = 10) -> List[Entity]:
        """Search entities by name or description."""
        query_lower = query.lower()
        matches = []

        for entity in self.entities.values():
            if (
                query_lower in entity.name.lower()
                or query_lower in entity.description.lower()
                or any(query_lower in alias.lower() for alias in entity.aliases)
            ):
                matches.append(entity)

        return matches[:limit]


class RelationManager:
    """Manager for relation operations in the knowledge graph."""

    def __init__(self, relations: Dict[str, Relation], touch_callback: Callable[[], None]) -> None:
        """Initialize the RelationManager.

        Args:
            relations: Dictionary of relations
            touch_callback: Callback to update graph timestamp
        """
        self.relations = relations
        self._touch = touch_callback

    def add_relation(self, relation: Relation) -> None:
        """Add a relation to the knowledge graph."""
        self.relations[relation.id] = relation
        self._touch()

    def remove_relation(
        self, relation_id: str, clusters: Dict[str, Cluster], text_chunks: Dict[str, TextChunk]
    ) -> bool:
        """Remove a relation and update references."""
        if relation_id not in self.relations:
            return False

        # Remove relation from clusters
        for cluster in clusters.values():
            cluster.remove_relation(relation_id)

        # Remove relation from text chunks
        for text_chunk in text_chunks.values():
            text_chunk.remove_relation(relation_id)

        del self.relations[relation_id]
        self._touch()
        return True

    def get_relation(self, relation_id: str) -> Optional[Relation]:
        """Get a relation by ID."""
        return self.relations.get(relation_id)

    def get_relations_by_type(self, relation_type: Union[RelationType, str]) -> List[Relation]:
        """Get all relations of a specific type."""
        return [relation for relation in self.relations.values() if relation.relation_type == relation_type]

    def get_entity_relations(self, entity_id: str, direction: str = "both") -> List[Relation]:
        """Get all relations connected to an entity."""
        relations = []

        for relation in self.relations.values():
            if direction in ("both", "outgoing"):
                if relation.head_entity and relation.head_entity.id == entity_id:
                    relations.append(relation)

            if direction in ("both", "incoming"):
                if relation.tail_entity and relation.tail_entity.id == entity_id:
                    relations.append(relation)

        return relations


class ClusterManager:
    """Manager for cluster operations in the knowledge graph."""

    def __init__(self, clusters: Dict[str, Cluster], touch_callback: Callable[[], None]) -> None:
        """Initialize the ClusterManager.

        Args:
            clusters: Dictionary of clusters
            touch_callback: Callback to update graph timestamp
        """
        self.clusters = clusters
        self._touch = touch_callback

    def add_cluster(self, cluster: Cluster) -> None:
        """Add a cluster to the knowledge graph."""
        self.clusters[cluster.id] = cluster
        self._touch()

    def remove_cluster(self, cluster_id: str) -> bool:
        """Remove a cluster and update parent/child relationships."""
        if cluster_id not in self.clusters:
            return False

        cluster = self.clusters[cluster_id]
        if cluster.parent_cluster_id:
            parent = self.get_cluster(cluster.parent_cluster_id)
            if parent:
                parent.remove_child_cluster(cluster_id)

        for child_id in cluster.child_clusters.copy():
            child = self.get_cluster(child_id)
            if child:
                child.parent_cluster_id = ""

        del self.clusters[cluster_id]
        self._touch()
        return True

    def get_cluster(self, cluster_id: str) -> Optional[Cluster]:
        """Get a cluster by ID."""
        return self.clusters.get(cluster_id)

    def get_clusters_by_type(self, cluster_type: Union[ClusterType, str]) -> List[Cluster]:
        """Get all clusters of a specific type."""
        return [cluster for cluster in self.clusters.values() if cluster.cluster_type == cluster_type]


class TextChunkManager:
    """Manager for text chunk operations in the knowledge graph."""

    def __init__(self, text_chunks: Dict[str, TextChunk], touch_callback: Callable[[], None]) -> None:
        """Initialize the TextChunkManager.

        Args:
            text_chunks: Dictionary of text chunks
            touch_callback: Callback to update graph timestamp
        """
        self.text_chunks = text_chunks
        self._touch = touch_callback

    def add_text_chunk(self, text_chunk: TextChunk) -> None:
        """Add a text chunk to the knowledge graph."""
        self.text_chunks[text_chunk.id] = text_chunk
        self._touch()

    def remove_text_chunk(
        self,
        chunk_id: str,
        entities: Dict[str, Entity],
        relations: Dict[str, Relation],
        clusters: Dict[str, Cluster],
    ) -> bool:
        """Remove a text chunk and update references."""
        if chunk_id not in self.text_chunks:
            return False

        # Remove text chunk references from entities
        for entity in entities.values():
            if hasattr(entity, "text_chunks"):
                entity.text_chunks.discard(chunk_id)

        # Remove text chunk references from relations
        for relation in relations.values():
            if hasattr(relation, "text_chunks"):
                relation.text_chunks.discard(chunk_id)

        # Remove text chunk references from clusters
        for cluster in clusters.values():
            if hasattr(cluster, "text_chunks"):
                cluster.text_chunks.discard(chunk_id)

        del self.text_chunks[chunk_id]
        self._touch()
        return True

    def get_text_chunk(self, chunk_id: str) -> Optional[TextChunk]:
        """Get a text chunk by ID."""
        return self.text_chunks.get(chunk_id)

    def search_text_chunks(self, query: str, limit: int = 10) -> List[TextChunk]:
        """Search text chunks by content or title."""
        query_lower = query.lower()
        matches = []

        for chunk in self.text_chunks.values():
            if query_lower in chunk.content.lower() or query_lower in chunk.title.lower():
                matches.append(chunk)

        return matches[:limit]
