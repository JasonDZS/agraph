"""
Knowledge graph data structure and operations.

This module defines the KnowledgeGraph class for managing entities, relations,
clusters, and text chunks in a comprehensive knowledge graph system.
"""

import uuid
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Union

from pydantic import BaseModel, Field

from agraph.base.clusters import Cluster
from agraph.base.entities import Entity
from agraph.base.types import ClusterType, EntityType, RelationType

from .managers import ClusterManager, EntityManager, RelationManager, TextChunkManager
from .mixins import SerializableMixin
from .relations import Relation
from .text import TextChunk


# pylint: disable=too-many-public-methods
class KnowledgeGraph(BaseModel, SerializableMixin):
    """Comprehensive knowledge graph containing entities, relations, clusters, and text chunks.

    The KnowledgeGraph serves as the main container for all knowledge graph components,
    providing methods for adding, removing, querying, and analyzing the graph structure.

    Attributes:
        name: Name of the knowledge graph
        description: Description of the knowledge graph
        entities: Dictionary mapping entity IDs to Entity objects
        relations: Dictionary mapping relation IDs to Relation objects
        clusters: Dictionary mapping cluster IDs to Cluster objects
        text_chunks: Dictionary mapping text chunk IDs to TextChunk objects
        metadata: Additional metadata for the graph
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Graph ID")
    name: str = Field(default="", description="Name of the knowledge graph")
    description: str = Field(default="", description="Description of the graph")
    entities: Dict[str, Entity] = Field(default_factory=dict, description="Entity storage")
    relations: Dict[str, Relation] = Field(default_factory=dict, description="Relation storage")
    clusters: Dict[str, Cluster] = Field(default_factory=dict, description="Cluster storage")
    text_chunks: Dict[str, TextChunk] = Field(
        default_factory=dict, description="Text chunk storage"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Graph metadata")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Update timestamp")

    class Config:
        """Pydantic configuration."""

        extra = "allow"
        validate_assignment = True

    def __init__(self, **data: Any) -> None:
        """Initialize the KnowledgeGraph with manager instances."""
        super().__init__(**data)
        self._entity_manager = EntityManager(self.entities, self.touch)
        self._relation_manager = RelationManager(self.relations, self.touch)
        self._cluster_manager = ClusterManager(self.clusters, self.touch)
        self._text_chunk_manager = TextChunkManager(self.text_chunks, self.touch)

    def touch(self) -> None:
        """Update the timestamp to current time."""
        self.updated_at = datetime.now()

    # Entity Management
    def add_entity(self, entity: Entity) -> None:
        """Add an entity to the knowledge graph."""
        self._entity_manager.add_entity(entity)

    def remove_entity(self, entity_id: str) -> bool:
        """Remove an entity from the knowledge graph."""
        return self._entity_manager.remove_entity(
            entity_id, self.relations, self.clusters, self.text_chunks
        )

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get an entity by ID."""
        return self._entity_manager.get_entity(entity_id)

    def get_entities_by_type(self, entity_type: Union[EntityType, str]) -> List[Entity]:
        """Get all entities of a specific type."""
        return self._entity_manager.get_entities_by_type(entity_type)

    def search_entities(self, query: str, limit: int = 10) -> List[Entity]:
        """Search entities by name or description."""
        return self._entity_manager.search_entities(query, limit)

    # Relation Management
    def add_relation(self, relation: Relation) -> None:
        """Add a relation to the knowledge graph."""
        self._relation_manager.add_relation(relation)

    def remove_relation(self, relation_id: str) -> bool:
        """Remove a relation from the knowledge graph."""
        return self._relation_manager.remove_relation(relation_id, self.clusters, self.text_chunks)

    def get_relation(self, relation_id: str) -> Optional[Relation]:
        """Get a relation by ID."""
        return self._relation_manager.get_relation(relation_id)

    def get_relations_by_type(self, relation_type: Union[RelationType, str]) -> List[Relation]:
        """Get all relations of a specific type."""
        return self._relation_manager.get_relations_by_type(relation_type)

    def get_entity_relations(self, entity_id: str, direction: str = "both") -> List[Relation]:
        """Get all relations connected to an entity."""
        return self._relation_manager.get_entity_relations(entity_id, direction)

    # Cluster Management
    def add_cluster(self, cluster: Cluster) -> None:
        """Add a cluster to the knowledge graph."""
        self._cluster_manager.add_cluster(cluster)

    def remove_cluster(self, cluster_id: str) -> bool:
        """Remove a cluster from the knowledge graph."""
        return self._cluster_manager.remove_cluster(cluster_id)

    def get_cluster(self, cluster_id: str) -> Optional[Cluster]:
        """Get a cluster by ID."""
        return self._cluster_manager.get_cluster(cluster_id)

    def get_clusters_by_type(self, cluster_type: Union[ClusterType, str]) -> List[Cluster]:
        """Get all clusters of a specific type."""
        return self._cluster_manager.get_clusters_by_type(cluster_type)

    # Text Chunk Management
    def add_text_chunk(self, text_chunk: TextChunk) -> None:
        """Add a text chunk to the knowledge graph."""
        self._text_chunk_manager.add_text_chunk(text_chunk)

    def remove_text_chunk(self, chunk_id: str) -> bool:
        """Remove a text chunk from the knowledge graph."""
        return self._text_chunk_manager.remove_text_chunk(
            chunk_id, self.entities, self.relations, self.clusters
        )

    def get_text_chunk(self, chunk_id: str) -> Optional[TextChunk]:
        """Get a text chunk by ID."""
        return self._text_chunk_manager.get_text_chunk(chunk_id)

    def search_text_chunks(self, query: str, limit: int = 10) -> List[TextChunk]:
        """Search text chunks by content or title."""
        return self._text_chunk_manager.search_text_chunks(query, limit)

    # Graph Analysis
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the knowledge graph.

        Returns:
            Dictionary containing various graph statistics
        """
        entity_types: Dict[str, int] = defaultdict(int)
        relation_types: Dict[str, int] = defaultdict(int)
        cluster_types: Dict[str, int] = defaultdict(int)

        for entity in self.entities.values():
            entity_types[str(entity.entity_type)] += 1

        for relation in self.relations.values():
            relation_types[str(relation.relation_type)] += 1

        for cluster in self.clusters.values():
            cluster_types[str(cluster.cluster_type)] += 1

        return {
            "total_entities": len(self.entities),
            "total_relations": len(self.relations),
            "total_clusters": len(self.clusters),
            "total_text_chunks": len(self.text_chunks),
            "entity_types": dict(entity_types),
            "relation_types": dict(relation_types),
            "cluster_types": dict(cluster_types),
            "average_entity_degree": self._calculate_average_degree(),
        }

    def _calculate_average_degree(self) -> float:
        """Calculate the average degree (number of connections) per entity."""
        if not self.entities:
            return 0.0

        total_degree = 0
        for entity_id in self.entities:
            degree = len(self._relation_manager.get_entity_relations(entity_id))
            total_degree += degree

        return total_degree / len(self.entities)

    def get_connected_components(self) -> List[Set[str]]:
        """Get all connected components in the graph.

        Returns:
            List of sets, each containing entity IDs in a connected component
        """
        visited = set()
        components = []

        def dfs(entity_id: str, component: Set[str]) -> None:
            if entity_id in visited:
                return
            visited.add(entity_id)
            component.add(entity_id)

            for relation in self.get_entity_relations(entity_id):
                if relation.head_entity and relation.head_entity.id != entity_id:
                    dfs(relation.head_entity.id, component)
                if relation.tail_entity and relation.tail_entity.id != entity_id:
                    dfs(relation.tail_entity.id, component)

        for entity_id in self.entities:
            if entity_id not in visited:
                component: Set[str] = set()
                dfs(entity_id, component)
                components.append(component)

        return components

    # Validation
    def is_valid(self) -> bool:
        """Check if the knowledge graph is in a valid state.

        Returns:
            True if the graph is valid, False otherwise
        """
        # Check all entities are valid
        for entity in self.entities.values():
            if not entity.is_valid():
                return False

        # Check all relations are valid
        for relation in self.relations.values():
            if not relation.is_valid():
                return False

        # Check all clusters are valid
        for cluster in self.clusters.values():
            if not cluster.is_valid():
                return False

        # Check all text chunks are valid
        for text_chunk in self.text_chunks.values():
            if not text_chunk.is_valid():
                return False

        return True

    def validate_integrity(self) -> List[str]:
        """Validate the integrity of the knowledge graph.

        Returns:
            List of error messages, empty if no issues found
        """
        errors = []
        errors.extend(self._validate_relation_references())
        errors.extend(self._validate_cluster_references())
        errors.extend(self._validate_text_chunk_references())
        return errors

    def _validate_relation_references(self) -> List[str]:
        """Validate that relations reference existing entities."""
        errors = []
        for relation in self.relations.values():
            if relation.head_entity and relation.head_entity.id not in self.entities:
                errors.append(f"Relation {relation.id} references non-existent head entity")
            if relation.tail_entity and relation.tail_entity.id not in self.entities:
                errors.append(f"Relation {relation.id} references non-existent tail entity")
        return errors

    def _validate_cluster_references(self) -> List[str]:
        """Validate that clusters reference existing entities."""
        errors = []
        for cluster in self.clusters.values():
            for entity_id in cluster.entities:
                if entity_id not in self.entities:
                    errors.append(
                        f"Cluster {cluster.id} references non-existent entity {entity_id}"
                    )
        return errors

    def _validate_text_chunk_references(self) -> List[str]:
        """Validate that text chunks reference existing entities and relations."""
        errors = []
        for chunk in self.text_chunks.values():
            for entity_id in chunk.entities:
                if entity_id not in self.entities:
                    errors.append(
                        f"TextChunk {chunk.id} references non-existent entity {entity_id}"
                    )
            for relation_id in chunk.relations:
                if relation_id not in self.relations:
                    errors.append(
                        f"TextChunk {chunk.id} references non-existent relation {relation_id}"
                    )
        return errors

    # Serialization
    def to_dict(self) -> Dict[str, Any]:
        """Convert the knowledge graph to dictionary representation.

        Returns:
            Dictionary containing all graph data
        """
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "entities": {eid: entity.to_dict() for eid, entity in self.entities.items()},
            "relations": {rid: relation.to_dict() for rid, relation in self.relations.items()},
            "clusters": {cid: cluster.to_dict() for cid, cluster in self.clusters.items()},
            "text_chunks": {tid: chunk.to_dict() for tid, chunk in self.text_chunks.items()},
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], **kwargs: Any) -> "KnowledgeGraph":
        """Create a knowledge graph from dictionary data.

        Args:
            data: Dictionary containing knowledge graph data

        Returns:
            KnowledgeGraph instance created from the dictionary data
        """
        # Create entities first
        # pylint: disable=too-many-locals
        entities = {}
        for eid, entity_data in data.get("entities", {}).items():
            entities[eid] = Entity.from_dict(entity_data)

        # Create relations with entity references
        relations = {}
        for rid, relation_data in data.get("relations", {}).items():
            relations[rid] = Relation.from_dict(relation_data, entities_map=entities)

        # Create clusters
        clusters = {}
        for cid, cluster_data in data.get("clusters", {}).items():
            clusters[cid] = Cluster.from_dict(cluster_data)

        # Create text chunks
        text_chunks = {}
        for tid, chunk_data in data.get("text_chunks", {}).items():
            text_chunks[tid] = TextChunk.from_dict(chunk_data)

        # Create the knowledge graph
        kg = cls(
            id=data.get("id", str(uuid.uuid4())),
            name=data.get("name", ""),
            description=data.get("description", ""),
            entities=entities,
            relations=relations,
            clusters=clusters,
            text_chunks=text_chunks,
            metadata=data.get("metadata", {}),
        )

        if "created_at" in data:
            kg.created_at = datetime.fromisoformat(data["created_at"])
        if "updated_at" in data:
            kg.updated_at = datetime.fromisoformat(data["updated_at"])

        return kg

    def clear(self) -> None:
        """Clear all data from the knowledge graph."""
        self.entities.clear()
        self.relations.clear()
        self.clusters.clear()
        self.text_chunks.clear()
        self.metadata.clear()
        self.touch()

    def merge(self, other: "KnowledgeGraph") -> None:
        """Merge another knowledge graph into this one.

        Args:
            other: Another KnowledgeGraph to merge
        """
        # Merge entities
        self.entities.update(other.entities)

        # Merge relations
        self.relations.update(other.relations)

        # Merge clusters
        self.clusters.update(other.clusters)

        # Merge text chunks
        self.text_chunks.update(other.text_chunks)

        # Merge metadata
        self.metadata.update(other.metadata)

        self.touch()
