"""
Cluster-related data structures and operations.

This module defines the Cluster class for grouping and organizing
entities and relations in a knowledge graph.
"""

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, Set

from pydantic import Field, field_validator

from ..utils import get_type_value
from .base import GraphNodeBase, TextChunkMixin
from .mixins import PropertyMixin
from .types import ClusterType, ClusterTypeType

if TYPE_CHECKING:
    pass  # No actual imports needed at runtime


class Cluster(GraphNodeBase, TextChunkMixin, PropertyMixin):
    """Represents a cluster of entities and relations in a knowledge graph.

    A Cluster groups together related entities and relations based on various
    clustering algorithms or domain-specific criteria.

    Attributes:
        name: Name of the cluster
        cluster_type: The type/algorithm used for clustering
        description: Detailed description of the cluster
        entities: Set of entity IDs belonging to this cluster
        relations: Set of relation IDs belonging to this cluster
        centroid: Optional centroid entity representing the cluster
        parent_cluster: Optional parent cluster for hierarchical clustering
        child_clusters: Set of child cluster IDs for hierarchical clustering
        size: Number of entities in the cluster
        cohesion_score: Measure of cluster cohesion (0.0 to 1.0)
    """

    name: str = Field(default="", description="Name of the cluster")
    cluster_type: ClusterTypeType = Field(default=ClusterType.OTHER, description="Cluster type")
    description: str = Field(default="", description="Detailed description")
    entities: Set[str] = Field(default_factory=set, description="Entity IDs in cluster")
    relations: Set[str] = Field(default_factory=set, description="Relation IDs in cluster")
    centroid_entity_id: str = Field(default="", description="ID of centroid entity")
    parent_cluster_id: str = Field(default="", description="Parent cluster ID")
    child_clusters: Set[str] = Field(default_factory=set, description="Child cluster IDs")
    cohesion_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Cluster cohesion score")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Additional properties")
    text_chunks: set = Field(default_factory=set, description="Connected text chunk IDs")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate cluster name is not empty when provided."""
        if v is not None and len(v.strip()) == 0:
            raise ValueError("Cluster name cannot be empty string")
        return v.strip() if v else ""

    @field_validator("cohesion_score")
    @classmethod
    def validate_cohesion_score(cls, v: float) -> float:
        """Validate cohesion score is between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Cohesion score must be between 0.0 and 1.0")
        return v

    @property
    def size(self) -> int:
        """Get the size of the cluster (number of entities)."""
        return len(self.entities)

    def add_entity(self, entity_id: str) -> None:
        """Add an entity to the cluster.

        Args:
            entity_id: The ID of the entity to add
        """
        if entity_id and entity_id.strip():
            self.entities.add(entity_id.strip())
            self.touch()

    def remove_entity(self, entity_id: str) -> None:
        """Remove an entity from the cluster.

        Args:
            entity_id: The ID of the entity to remove
        """
        self.entities.discard(entity_id)
        self.touch()

    def add_relation(self, relation_id: str) -> None:
        """Add a relation to the cluster.

        Args:
            relation_id: The ID of the relation to add
        """
        if relation_id and relation_id.strip():
            self.relations.add(relation_id.strip())
            self.touch()

    def remove_relation(self, relation_id: str) -> None:
        """Remove a relation from the cluster.

        Args:
            relation_id: The ID of the relation to remove
        """
        self.relations.discard(relation_id)
        self.touch()

    def add_child_cluster(self, cluster_id: str) -> None:
        """Add a child cluster for hierarchical clustering.

        Args:
            cluster_id: The ID of the child cluster to add
        """
        if cluster_id and cluster_id.strip():
            self.child_clusters.add(cluster_id.strip())
            self.touch()

    def remove_child_cluster(self, cluster_id: str) -> None:
        """Remove a child cluster.

        Args:
            cluster_id: The ID of the child cluster to remove
        """
        self.child_clusters.discard(cluster_id)
        self.touch()

    def has_entity(self, entity_id: str) -> bool:
        """Check if an entity belongs to this cluster.

        Args:
            entity_id: The ID of the entity to check

        Returns:
            True if the entity belongs to this cluster, False otherwise
        """
        return entity_id in self.entities

    def has_relation(self, relation_id: str) -> bool:
        """Check if a relation belongs to this cluster.

        Args:
            relation_id: The ID of the relation to check

        Returns:
            True if the relation belongs to this cluster, False otherwise
        """
        return relation_id in self.relations

    def is_empty(self) -> bool:
        """Check if the cluster is empty.

        Returns:
            True if the cluster has no entities, False otherwise
        """
        return len(self.entities) == 0

    def is_hierarchical(self) -> bool:
        """Check if this cluster is part of a hierarchy.

        Returns:
            True if cluster has parent or children, False otherwise
        """
        return bool(self.parent_cluster_id or self.child_clusters)

    def merge_with(self, other: "Cluster") -> None:
        """Merge another cluster into this one.

        Args:
            other: The cluster to merge into this one
        """
        self.entities.update(other.entities)
        self.relations.update(other.relations)
        self.child_clusters.update(other.child_clusters)

        # Update cohesion score as weighted average
        total_size = self.size + other.size
        if total_size > 0:
            self.cohesion_score = (
                self.cohesion_score * self.size + other.cohesion_score * other.size
            ) / total_size

        self.touch()

    def is_valid(self) -> bool:
        """Check if the cluster is valid.

        Returns:
            True if cluster has at least one entity, False otherwise
        """
        return len(self.entities) > 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert cluster to dictionary representation.

        Returns:
            Dictionary containing all cluster data with serializable values
        """
        return {
            "id": self.id,
            "name": self.name,
            "cluster_type": get_type_value(self.cluster_type),
            "description": self.description,
            "entities": list(self.entities),
            "relations": list(self.relations),
            "centroid_entity_id": self.centroid_entity_id,
            "parent_cluster_id": self.parent_cluster_id,
            "child_clusters": list(self.child_clusters),
            "cohesion_score": self.cohesion_score,
            "properties": self.properties,
            # pylint: disable-next=duplicate-code
            "confidence": self.confidence,
            "source": self.source,
            "text_chunks": list(self.text_chunks),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    # pylint: disable=duplicate-code
    def from_dict(cls, data: Dict[str, Any], **kwargs: Any) -> "Cluster":
        """Create cluster from dictionary data.

        Args:
            data: Dictionary containing cluster data

        Returns:
            Cluster instance created from the dictionary data
        """
        cluster_type_value = data.get("cluster_type", "OTHER")
        try:
            cluster_type = ClusterType(cluster_type_value)
        except (ValueError, AttributeError):
            cluster_type = ClusterType.OTHER

        cluster = cls(
            id=data.get("id", str(uuid.uuid4())),
            name=data.get("name", ""),
            cluster_type=cluster_type,
            description=data.get("description", ""),
            entities=set(data.get("entities", [])),
            relations=set(data.get("relations", [])),
            centroid_entity_id=data.get("centroid_entity_id", ""),
            parent_cluster_id=data.get("parent_cluster_id", ""),
            child_clusters=set(data.get("child_clusters", [])),
            cohesion_score=data.get("cohesion_score", 0.0),
            properties=data.get("properties", {}),
            confidence=data.get("confidence", 0.8),
            source=data.get("source", ""),
            text_chunks=set(data.get("text_chunks", [])),
        )

        if "created_at" in data:
            cluster.created_at = datetime.fromisoformat(data["created_at"])
        if "updated_at" in data:
            cluster.updated_at = datetime.fromisoformat(data["updated_at"])

        return cluster
