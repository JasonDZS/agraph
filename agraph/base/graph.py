"""
Knowledge graph data structure and operations.

This module defines the KnowledgeGraph class for managing entities, relations,
clusters, and text chunks in a comprehensive knowledge graph system.
"""

import ast
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import networkx as nx
from pydantic import BaseModel, Field

from .clusters import Cluster
from .entities import Entity
from .managers import ClusterManager, EntityManager, RelationManager, TextChunkManager
from .mixins import ImportExportMixin, SerializableMixin
from .relations import Relation
from .text import TextChunk
from .types import ClusterType, EntityType, RelationType


# pylint: disable=too-many-public-methods
class KnowledgeGraph(BaseModel, SerializableMixin, ImportExportMixin):
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

    # ImportExportMixin implementation
    def _export_data(self) -> Dict[str, Any]:
        """Export data to dictionary format (backup method for ImportExportMixin)."""
        return self.to_dict()

    @classmethod
    def _import_data(cls, data: Dict[str, Any], **kwargs: Any) -> "KnowledgeGraph":
        """Import data from dictionary format (backup method for ImportExportMixin)."""
        return cls.from_dict(data, **kwargs)

    # GraphML Format Support
    def export_to_graphml(self, file_path: Union[str, Path], **kwargs: Any) -> None:
        """Export the knowledge graph to GraphML format.

        Args:
            file_path: Path where the GraphML file will be saved
            **kwargs: Additional arguments (currently unused)
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Create NetworkX graph
        G = nx.MultiDiGraph()

        # Add nodes (entities)
        for entity_id, entity in self.entities.items():
            G.add_node(
                entity_id,
                # Basic attributes
                name=entity.name,
                type=str(entity.entity_type),
                description=entity.description or "",
                confidence=entity.confidence,
                # Metadata
                created_at=entity.created_at.isoformat(),
                updated_at=entity.updated_at.isoformat(),
                # Additional properties as string representation
                properties=str(entity.properties) if entity.properties else "",
                # Text chunks as comma-separated string
                text_chunks=",".join(entity.text_chunks) if entity.text_chunks else "",
            )

        # Add edges (relations)
        for relation_id, relation in self.relations.items():
            if relation.head_entity and relation.tail_entity:
                G.add_edge(
                    relation.head_entity.id,
                    relation.tail_entity.id,
                    key=relation_id,  # Use relation ID as edge key for MultiDiGraph
                    # Basic attributes
                    relation_id=relation_id,
                    type=str(relation.relation_type),
                    description=relation.description or "",
                    confidence=relation.confidence,
                    weight=relation.properties.get("weight", 1.0),  # Get weight from properties
                    # Metadata
                    created_at=relation.created_at.isoformat(),
                    updated_at=relation.updated_at.isoformat(),
                    # Additional properties as string representation
                    properties=str(relation.properties) if relation.properties else "",
                    # Text chunks as comma-separated string
                    text_chunks=",".join(relation.text_chunks) if relation.text_chunks else "",
                )

        # Add graph-level metadata as graph attributes
        G.graph.update(
            {
                "id": self.id,
                "name": self.name,
                "description": self.description,
                "created_at": self.created_at.isoformat(),
                "updated_at": self.updated_at.isoformat(),
                "metadata": str(self.metadata) if self.metadata else "",
                # Statistics
                "total_entities": len(self.entities),
                "total_relations": len(self.relations),
                "total_clusters": len(self.clusters),
                "total_text_chunks": len(self.text_chunks),
            }
        )

        # Write to GraphML file
        nx.write_graphml(G, file_path, encoding="utf-8", prettyprint=True)

    @classmethod
    def import_from_graphml(cls, file_path: Union[str, Path], **kwargs: Any) -> "KnowledgeGraph":
        """Import a knowledge graph from GraphML format.

        Args:
            file_path: Path to the GraphML file to import
            **kwargs: Additional arguments for graph creation

        Returns:
            KnowledgeGraph instance created from the GraphML data
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"GraphML file not found: {file_path}")

        # Read GraphML file
        G = nx.read_graphml(file_path)

        # Create entities from nodes
        entities = {}
        for node_id, node_data in G.nodes(data=True):
            # Parse entity type
            entity_type_str = node_data.get("type", "concept")
            try:
                entity_type = EntityType(entity_type_str)
            except ValueError:
                entity_type = EntityType.CONCEPT  # Default fallback

            # Parse properties from string representation
            properties = {}
            if node_data.get("properties"):
                try:
                    properties = ast.literal_eval(node_data["properties"])
                except (ValueError, SyntaxError):
                    properties = {}

            # Parse text chunks from comma-separated string
            text_chunks = set()
            if node_data.get("text_chunks"):
                text_chunks = set(
                    chunk.strip() for chunk in node_data["text_chunks"].split(",") if chunk.strip()
                )

            entity = Entity(
                id=node_id,
                name=node_data.get("name", ""),
                entity_type=entity_type,
                description=node_data.get("description", ""),
                confidence=float(node_data.get("confidence", 1.0)),
                properties=properties,
                text_chunks=text_chunks,
            )

            # Set timestamps if available
            if node_data.get("created_at"):
                try:
                    entity.created_at = datetime.fromisoformat(node_data["created_at"])
                except ValueError:
                    pass
            if node_data.get("updated_at"):
                try:
                    entity.updated_at = datetime.fromisoformat(node_data["updated_at"])
                except ValueError:
                    pass

            entities[node_id] = entity

        # Create relations from edges
        relations = {}
        for head, tail, edge_data in G.edges(data=True):
            relation_id = edge_data.get("relation_id", str(uuid.uuid4()))

            # Parse relation type
            relation_type_str = edge_data.get("type", "references")
            try:
                relation_type = RelationType(relation_type_str)
            except ValueError:
                relation_type = RelationType.REFERENCES  # Default fallback

            # Parse properties from string representation
            properties = {}
            if edge_data.get("properties"):
                try:
                    properties = ast.literal_eval(edge_data["properties"])
                except (ValueError, SyntaxError):
                    properties = {}

            # Store weight in properties if present
            if edge_data.get("weight") and float(edge_data["weight"]) != 1.0:
                properties["weight"] = float(edge_data["weight"])

            # Parse text chunks from comma-separated string
            text_chunks = set()
            if edge_data.get("text_chunks"):
                text_chunks = set(
                    chunk.strip() for chunk in edge_data["text_chunks"].split(",") if chunk.strip()
                )

            relation = Relation(
                id=relation_id,
                head_entity=entities.get(head),
                tail_entity=entities.get(tail),
                relation_type=relation_type,
                description=edge_data.get("description", ""),
                confidence=float(edge_data.get("confidence", 1.0)),
                # Note: weight is stored in properties since Relation doesn't have weight attribute
                properties=properties,
                text_chunks=text_chunks,
            )

            # Set timestamps if available
            if edge_data.get("created_at"):
                try:
                    relation.created_at = datetime.fromisoformat(edge_data["created_at"])
                except ValueError:
                    pass
            if edge_data.get("updated_at"):
                try:
                    relation.updated_at = datetime.fromisoformat(edge_data["updated_at"])
                except ValueError:
                    pass

            relations[relation_id] = relation

        # Extract graph metadata
        graph_data = G.graph
        kg = cls(
            id=graph_data.get("id", str(uuid.uuid4())),
            name=graph_data.get("name", ""),
            description=graph_data.get("description", ""),
            entities=entities,
            relations=relations,
            # Note: Clusters and text_chunks are not preserved in GraphML format
            # They could be added as separate nodes with special types if needed
            clusters={},
            text_chunks={},
            metadata=(
                ast.literal_eval(graph_data.get("metadata", "{}"))
                if graph_data.get("metadata")
                else {}
            ),
        )

        # Set timestamps if available
        if graph_data.get("created_at"):
            try:
                kg.created_at = datetime.fromisoformat(graph_data["created_at"])
            except ValueError:
                pass
        if graph_data.get("updated_at"):
            try:
                kg.updated_at = datetime.fromisoformat(graph_data["updated_at"])
            except ValueError:
                pass

        return kg
