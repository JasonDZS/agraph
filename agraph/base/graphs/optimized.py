"""
Optimized KnowledgeGraph implementation with indexing and caching.

This module provides an enhanced KnowledgeGraph class that uses indexing
and caching to significantly improve performance for common operations.
"""

import time
import uuid
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Union

from pydantic import BaseModel, Field

from ..core.mixins import ImportExportMixin, SerializableMixin
from ..core.types import ClusterType, EntityType, RelationType
from ..infrastructure.cache import CacheManager, cached
from ..infrastructure.indexes import IndexManager
from ..managers.optimized import OptimizedEntityManager, OptimizedRelationManager
from ..models.clusters import Cluster
from ..models.entities import Entity
from ..models.relations import Relation
from ..models.text import TextChunk


class KnowledgeGraph(BaseModel, SerializableMixin, ImportExportMixin):
    """
    Optimized knowledge graph with indexing and caching support.

    This enhanced version provides significant performance improvements over
    the original KnowledgeGraph through:
    - Index-based fast lookups (O(1) instead of O(n))
    - Intelligent caching of expensive operations
    - Optimized cascade operations
    - Performance monitoring and metrics
    """

    # pylint: disable=too-many-public-methods

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Graph ID")
    name: str = Field(default="", description="Name of the knowledge graph")
    description: str = Field(default="", description="Description of the graph")
    entities: Dict[str, Entity] = Field(default_factory=dict, description="Entity storage")
    relations: Dict[str, Relation] = Field(default_factory=dict, description="Relation storage")
    clusters: Dict[str, Cluster] = Field(default_factory=dict, description="Cluster storage")
    text_chunks: Dict[str, TextChunk] = Field(default_factory=dict, description="Text chunk storage")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Graph metadata")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Update timestamp")

    class Config:
        """Pydantic configuration."""

        extra = "allow"
        validate_assignment = True

    def __init__(self, **data: Any) -> None:
        """Initialize the KnowledgeGraph with managers and optimization components."""
        super().__init__(**data)

        # Initialize optimization components
        self.index_manager = IndexManager()
        self.cache_manager = CacheManager(max_size=2000, default_ttl=300)  # 5 min default TTL

        # Initialize optimized managers
        self._entity_manager = OptimizedEntityManager(self.entities, self.touch, self.index_manager, self.cache_manager)
        self._relation_manager = OptimizedRelationManager(
            self.relations, self.touch, self.index_manager, self.cache_manager
        )

        # Performance metrics
        self._performance_metrics = {
            "total_operations": 0,
            "cache_hits": 0,
            "index_hits": 0,
            "expensive_operations_time": 0.0,
        }

        # Wrap methods with cache decorators now that cache_manager is available
        self.get_graph_statistics = cached(
            cache_manager=self.cache_manager, ttl=300, tags={"statistics", "analysis"}  # 5 minutes
        )(self._get_graph_statistics_impl)

        self.get_connected_components = cached(
            cache_manager=self.cache_manager, ttl=600, tags={"topology", "analysis"}  # 10 minutes
        )(self._get_connected_components_impl)

        # Build indexes if data already exists
        if self.entities or self.relations or self.clusters or self.text_chunks:
            self._rebuild_all_indexes()

    def touch(self) -> None:
        """Update the timestamp to current time."""
        self.updated_at = datetime.now()

    def _rebuild_all_indexes(self) -> None:
        """Rebuild all indexes from current data."""
        self.index_manager.rebuild_indexes(self)

    # Entity Management with Optimization
    def add_entity(self, entity: Entity) -> None:
        """Add an entity to the knowledge graph (optimized)."""
        self._entity_manager.add_entity(entity)
        self._performance_metrics["total_operations"] += 1

    def remove_entity(self, entity_id: str) -> bool:
        """Remove an entity from the knowledge graph (optimized)."""
        result = self._entity_manager.remove_entity(entity_id, self.relations, self.clusters, self.text_chunks)
        self._performance_metrics["total_operations"] += 1
        return result

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get an entity by ID (cached)."""
        return self._entity_manager.get_entity(entity_id)

    def get_entities_by_type(self, entity_type: Union[EntityType, str]) -> List[Entity]:
        """Get all entities of a specific type (indexed - O(1) lookup)."""
        return self._entity_manager.get_entities_by_type(entity_type)

    def search_entities(self, query: str, limit: int = 10) -> List[Entity]:
        """Search entities by name or description (cached)."""
        return self._entity_manager.search_entities(query, limit)

    # Relation Management with Optimization
    def add_relation(self, relation: Relation) -> None:
        """Add a relation to the knowledge graph (optimized)."""
        self._relation_manager.add_relation(relation)
        self._performance_metrics["total_operations"] += 1

    def remove_relation(self, relation_id: str) -> bool:
        """Remove a relation from the knowledge graph (optimized)."""
        result = self._relation_manager.remove_relation(relation_id, self.clusters, self.text_chunks)
        self._performance_metrics["total_operations"] += 1
        return result

    def get_relation(self, relation_id: str) -> Optional[Relation]:
        """Get a relation by ID (cached)."""
        return self._relation_manager.get_relation(relation_id)

    def get_relations_by_type(self, relation_type: Union[RelationType, str]) -> List[Relation]:
        """Get all relations of a specific type (cached)."""
        return self._relation_manager.get_relations_by_type(relation_type)

    def get_entity_relations(self, entity_id: str, direction: str = "both") -> List[Relation]:
        """Get all relations connected to an entity (indexed - O(1) lookup)."""
        return self._relation_manager.get_entity_relations(entity_id, direction)

    # Cached Graph Analysis Methods - will be wrapped with cache decorator in __init__
    def _get_graph_statistics_impl(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the knowledge graph (cached)."""
        start_time = time.time()

        entity_types: Dict[str, int] = defaultdict(int)
        relation_types: Dict[str, int] = defaultdict(int)
        cluster_types: Dict[str, int] = defaultdict(int)

        for entity in self.entities.values():
            entity_types[str(entity.entity_type)] += 1

        for relation in self.relations.values():
            relation_types[str(relation.relation_type)] += 1

        for cluster in self.clusters.values():
            cluster_types[str(cluster.cluster_type)] += 1

        statistics = {
            "total_entities": len(self.entities),
            "total_relations": len(self.relations),
            "total_clusters": len(self.clusters),
            "total_text_chunks": len(self.text_chunks),
            "entity_types": dict(entity_types),
            "relation_types": dict(relation_types),
            "cluster_types": dict(cluster_types),
            "average_entity_degree": self._calculate_average_degree(),
        }

        # Record expensive operation time
        operation_time = time.time() - start_time
        self._performance_metrics["expensive_operations_time"] += operation_time

        return statistics

    def _calculate_average_degree(self) -> float:
        """Calculate the average degree (cached internally)."""
        if not self.entities:
            return 0.0

        total_degree = 0
        for entity_id in self.entities:
            # Use optimized index-based lookup
            relations = self._relation_manager.get_entity_relations(entity_id)
            total_degree += len(relations)

        return total_degree / len(self.entities)

    def _get_connected_components_impl(self) -> List[Set[str]]:
        """Get all connected components in the graph (cached)."""
        start_time = time.time()

        visited = set()
        components = []

        def dfs(entity_id: str, component: Set[str]) -> None:
            if entity_id in visited:
                return
            visited.add(entity_id)
            component.add(entity_id)

            # Use optimized relation lookup
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

        # Record expensive operation time
        operation_time = time.time() - start_time
        self._performance_metrics["expensive_operations_time"] += operation_time

        return components

    # Performance and Optimization Methods
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        entity_metrics = self._entity_manager.get_performance_metrics()
        relation_metrics = self._relation_manager.get_performance_metrics()

        return {
            "graph_metrics": self._performance_metrics,
            "entity_manager": entity_metrics,
            "relation_manager": relation_metrics,
            "index_statistics": self.index_manager.get_statistics(),
            "cache_statistics": self.cache_manager.get_statistics(),
            "optimization_summary": {
                "total_index_hits": (entity_metrics.get("index_hits", 0) + relation_metrics.get("index_hits", 0)),
                "total_cache_hits": (entity_metrics.get("cache_hits", 0) + relation_metrics.get("cache_hits", 0)),
                "average_operation_time": (
                    (entity_metrics.get("total_time", 0) + relation_metrics.get("total_time", 0))
                    / max(
                        1,
                        entity_metrics.get("operations_count", 0) + relation_metrics.get("operations_count", 0),
                    )
                ),
            },
        }

    def optimize_performance(self) -> Dict[str, Any]:
        """Perform optimization operations and return summary."""
        optimization_summary = {"cache_cleanup": 0, "index_rebuild": False, "memory_freed": 0}

        # Clean up expired cache entries
        expired_count = self.cache_manager.cleanup_expired()
        optimization_summary["cache_cleanup"] = expired_count

        # Rebuild indexes if needed (based on cache hit ratio)
        cache_stats = self.cache_manager.get_statistics()
        if cache_stats.get("hit_ratio", 0) < 0.5:  # Low hit ratio indicates stale cache
            self._rebuild_all_indexes()
            optimization_summary["index_rebuild"] = True

        return optimization_summary

    def clear_caches(self) -> None:
        """Clear all caches to free memory."""
        self.cache_manager.clear()

    def rebuild_indexes(self) -> None:
        """Rebuild all indexes for optimal performance."""
        self._rebuild_all_indexes()

    # Cluster Management (unchanged but using optimized base)
    def add_cluster(self, cluster: Cluster) -> None:
        """Add a cluster to the knowledge graph."""
        self.clusters[cluster.id] = cluster

        # Update indexes for cluster-entity relationships
        for entity_id in cluster.entities:
            self.index_manager.add_entity_to_cluster_index(entity_id, cluster.id)

        # Invalidate relevant caches
        self.cache_manager.invalidate_by_tags({"clusters", "statistics"})
        self.touch()

    def remove_cluster(self, cluster_id: str) -> bool:
        """Remove a cluster from the knowledge graph."""
        if cluster_id not in self.clusters:
            return False

        cluster = self.clusters[cluster_id]

        # Remove from indexes
        for entity_id in cluster.entities:
            self.index_manager.remove_entity_from_cluster_index(entity_id, cluster_id)

        # Handle parent/child relationships
        if cluster.parent_cluster_id:
            parent = self.clusters.get(cluster.parent_cluster_id)
            if parent:
                parent.remove_child_cluster(cluster_id)

        for child_id in cluster.child_clusters.copy():
            child = self.clusters.get(child_id)
            if child:
                child.parent_cluster_id = ""

        del self.clusters[cluster_id]

        # Invalidate relevant caches
        self.cache_manager.invalidate_by_tags({"clusters", "statistics"})
        self.touch()
        return True

    def get_cluster(self, cluster_id: str) -> Optional[Cluster]:
        """Get a cluster by ID."""
        return self.clusters.get(cluster_id)

    def get_clusters_by_type(self, cluster_type: Union[ClusterType, str]) -> List[Cluster]:
        """Get all clusters of a specific type."""
        return [cluster for cluster in self.clusters.values() if cluster.cluster_type == cluster_type]

    # Text Chunk Management (unchanged but using optimized base)
    def add_text_chunk(self, text_chunk: TextChunk) -> None:
        """Add a text chunk to the knowledge graph."""
        self.text_chunks[text_chunk.id] = text_chunk

        # Update indexes for text chunk relationships
        for entity_id in text_chunk.entities:
            self.index_manager.add_entity_to_text_chunk_index(entity_id, text_chunk.id)

        # Invalidate relevant caches
        self.cache_manager.invalidate_by_tags({"text_chunks", "statistics"})
        self.touch()

    def remove_text_chunk(self, chunk_id: str) -> bool:
        """Remove a text chunk from the knowledge graph."""
        if chunk_id not in self.text_chunks:
            return False

        text_chunk = self.text_chunks[chunk_id]

        # Remove from indexes
        for entity_id in text_chunk.entities:
            self.index_manager.remove_entity_from_text_chunk_index(entity_id, chunk_id)

        # Remove text chunk references from entities
        for entity in self.entities.values():
            if hasattr(entity, "text_chunks"):
                entity.text_chunks.discard(chunk_id)

        # Remove text chunk references from relations
        for relation in self.relations.values():
            if hasattr(relation, "text_chunks"):
                relation.text_chunks.discard(chunk_id)

        # Remove text chunk references from clusters
        for cluster in self.clusters.values():
            if hasattr(cluster, "text_chunks"):
                cluster.text_chunks.discard(chunk_id)

        del self.text_chunks[chunk_id]

        # Invalidate relevant caches
        self.cache_manager.invalidate_by_tags({"text_chunks", "statistics"})
        self.touch()
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
                if len(matches) >= limit:
                    break

        return matches

    # Enhanced Validation with Performance Monitoring
    def is_valid(self) -> bool:
        """Check if the knowledge graph is in a valid state (cached)."""
        cache_key = "graph_validation"
        cached_result = self.cache_manager.get(cache_key)
        if cached_result is not None:
            return bool(cached_result)

        # Check all entities are valid
        for entity in self.entities.values():
            if not entity.is_valid():
                self.cache_manager.put(cache_key, False, ttl=60)
                return False

        # Check all relations are valid
        for relation in self.relations.values():
            if not relation.is_valid():
                self.cache_manager.put(cache_key, False, ttl=60)
                return False

        # Check all clusters are valid
        for cluster in self.clusters.values():
            if not cluster.is_valid():
                self.cache_manager.put(cache_key, False, ttl=60)
                return False

        # Check all text chunks are valid
        for text_chunk in self.text_chunks.values():
            if not text_chunk.is_valid():
                self.cache_manager.put(cache_key, False, ttl=60)
                return False

        # Cache positive result for shorter time
        self.cache_manager.put(cache_key, True, ttl=30)
        return True

    # Inherit other methods from the original KnowledgeGraph
    # (validate_integrity, to_dict, from_dict, etc. remain the same)

    def validate_integrity(self) -> List[str]:
        """Validate the integrity of the knowledge graph."""
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
                    errors.append(f"Cluster {cluster.id} references non-existent entity {entity_id}")
        return errors

    def _validate_text_chunk_references(self) -> List[str]:
        """Validate that text chunks reference existing entities and relations."""
        errors = []
        for chunk in self.text_chunks.values():
            for entity_id in chunk.entities:
                if entity_id not in self.entities:
                    errors.append(f"TextChunk {chunk.id} references non-existent entity {entity_id}")
            for relation_id in chunk.relations:
                if relation_id not in self.relations:
                    errors.append(f"TextChunk {chunk.id} references non-existent relation {relation_id}")
        return errors

    # SerializableMixin abstract methods implementation
    def to_dict(self) -> Dict[str, Any]:
        """Convert the optimized knowledge graph to dictionary representation.

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
        """Create an optimized knowledge graph from dictionary data.

        Args:
            data: Dictionary containing knowledge graph data
            **kwargs: Additional arguments

        Returns:
            KnowledgeGraph instance created from the dictionary data
        """
        # Create entities first
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

        # Parse timestamps
        created_at = datetime.fromisoformat(data.get("created_at", datetime.now().isoformat()))
        updated_at = datetime.fromisoformat(data.get("updated_at", datetime.now().isoformat()))

        # Create optimized knowledge graph instance
        kg = cls(
            id=data.get("id", str(uuid.uuid4())),
            name=data.get("name", ""),
            description=data.get("description", ""),
            entities=entities,
            relations=relations,
            clusters=clusters,
            text_chunks=text_chunks,
            metadata=data.get("metadata", {}),
            created_at=created_at,
            updated_at=updated_at,
            **kwargs,
        )

        # Rebuild indexes after loading data
        kg._rebuild_indexes()

        return kg

    def _rebuild_indexes(self) -> None:
        """Rebuild all indexes after loading data."""
        # Clear existing indexes
        self.index_manager = IndexManager()

        # Rebuild entity type indexes
        for entity_id, entity in self.entities.items():
            self.index_manager.add_entity_to_type_index(entity_id, entity.entity_type)

        # Rebuild relation indexes
        for relation_id, relation in self.relations.items():
            if relation.head_entity and relation.tail_entity:
                self.index_manager.add_relation_to_index(relation_id, relation.head_entity.id, relation.tail_entity.id)

    # ImportExportMixin abstract methods implementation
    def _export_data(self) -> Dict[str, Any]:
        """Export data to dictionary format (backup method for ImportExportMixin)."""
        return self.to_dict()

    @classmethod
    def _import_data(cls, data: Dict[str, Any], **kwargs: Any) -> "KnowledgeGraph":
        """Import data from dictionary format (backup method for ImportExportMixin)."""
        return cls.from_dict(data, **kwargs)

    # Compatibility methods to match KnowledgeGraph API
    def clear(self) -> None:
        """Clear all data from the knowledge graph (with index cleanup)."""
        # Clear main data structures
        self.entities.clear()
        self.relations.clear()
        self.clusters.clear()
        self.text_chunks.clear()
        self.metadata.clear()

        # Clear optimization structures
        self.index_manager = IndexManager()
        self.cache_manager.clear()

        # Reset performance metrics
        self._performance_metrics = {
            "total_operations": 0,
            "cache_hits": 0,
            "index_hits": 0,
            "expensive_operations_time": 0.0,
        }

        self.touch()

    def merge(self, other: "KnowledgeGraph") -> None:
        """Merge another knowledge graph into this one (with index updates).

        Args:
            other: Another KnowledgeGraph to merge
        """
        # Merge entities and update indexes
        for entity in other.entities.values():
            self.add_entity(entity)

        # Merge relations and update indexes
        for relation in other.relations.values():
            self.add_relation(relation)

        # Merge clusters
        for cluster in other.clusters.values():
            self.add_cluster(cluster)

        # Merge text chunks
        self.text_chunks.update(other.text_chunks)

        # Merge metadata
        self.metadata.update(other.metadata)

        # Clear caches since data structure changed significantly
        self.cache_manager.clear()

        self.touch()
