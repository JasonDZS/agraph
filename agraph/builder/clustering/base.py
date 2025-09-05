"""
Base class for clustering algorithms.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set

from ...base.core.types import ClusterType
from ...base.models.clusters import Cluster
from ...base.models.entities import Entity
from ...base.models.relations import Relation


class ClusterAlgorithm(ABC):
    """Abstract base class for clustering algorithms."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize clustering algorithm.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.min_cluster_size = self.config.get("min_cluster_size", 2)
        self.max_cluster_size = self.config.get("max_cluster_size", 50)

    @abstractmethod
    def cluster(self, entities: List[Entity], relations: List[Relation]) -> List[Cluster]:
        """Perform clustering on entities based on relations.

        Args:
            entities: List of entities to cluster
            relations: List of relations between entities

        Returns:
            List of generated clusters
        """

    def build_adjacency_matrix(self, entities: List[Entity], relations: List[Relation]) -> Dict[str, Dict[str, float]]:
        """Build adjacency matrix from entities and relations.

        Args:
            entities: List of entities
            relations: List of relations

        Returns:
            Adjacency matrix as dict of dicts
        """
        # Initialize adjacency matrix
        adjacency: Dict[str, Dict[str, float]] = {e.id: {} for e in entities}

        # Add relations to adjacency matrix
        for relation in relations:
            if (
                relation.head_entity
                and relation.tail_entity
                and relation.head_entity.id in adjacency
                and relation.tail_entity.id in adjacency
            ):

                head_id = relation.head_entity.id
                tail_id = relation.tail_entity.id

                # Use confidence as edge weight
                weight = relation.confidence

                # Add bidirectional edges (treat as undirected graph for clustering)
                adjacency[head_id][tail_id] = max(adjacency[head_id].get(tail_id, 0), weight)
                adjacency[tail_id][head_id] = max(adjacency[tail_id].get(head_id, 0), weight)

        return adjacency

    def get_entity_neighbors(self, entity_id: str, adjacency: Dict[str, Dict[str, float]]) -> Set[str]:
        """Get neighbors of an entity.

        Args:
            entity_id: ID of the entity
            adjacency: Adjacency matrix

        Returns:
            Set of neighbor entity IDs
        """
        return set(adjacency.get(entity_id, {}).keys())

    def calculate_cluster_cohesion(self, entity_ids: Set[str], adjacency: Dict[str, Dict[str, float]]) -> float:
        """Calculate cohesion score for a cluster.

        Args:
            entity_ids: Set of entity IDs in the cluster
            adjacency: Adjacency matrix

        Returns:
            Cohesion score (0.0 to 1.0)
        """
        if len(entity_ids) < 2:
            return 0.0

        total_possible_edges = len(entity_ids) * (len(entity_ids) - 1) // 2
        if total_possible_edges == 0:
            return 0.0

        total_weight = 0.0
        edge_count = 0

        for entity1 in entity_ids:
            for entity2 in entity_ids:
                if entity1 < entity2:  # Avoid double counting
                    weight = adjacency.get(entity1, {}).get(entity2, 0.0)
                    if weight > 0:
                        total_weight += weight
                        edge_count += 1

        if edge_count == 0:
            return 0.0

        # Normalize by number of possible edges and average weight
        avg_weight = total_weight / edge_count
        edge_density = edge_count / total_possible_edges

        return avg_weight * edge_density

    def create_cluster(
        self,
        entity_ids: Set[str],
        entities_map: Dict[str, Entity],
        cluster_type: ClusterType = ClusterType.TOPIC,
    ) -> Optional[Cluster]:
        """Create a cluster from entity IDs.

        Args:
            entity_ids: Set of entity IDs
            entities_map: Mapping from entity ID to Entity object
            cluster_type: Type of cluster

        Returns:
            Created cluster or None if invalid
        """
        if len(entity_ids) < self.min_cluster_size:
            return None

        if len(entity_ids) > self.max_cluster_size:
            # If cluster is too large, it might not be meaningful
            # Could implement splitting logic here
            return None

        # Get entities in cluster
        cluster_entities = [entities_map[eid] for eid in entity_ids if eid in entities_map]

        if len(cluster_entities) < self.min_cluster_size:
            return None

        # Generate cluster name based on entity types and names
        cluster_name = self._generate_cluster_name(cluster_entities)

        # Generate cluster description
        cluster_description = self._generate_cluster_description(cluster_entities)

        cluster = Cluster(
            name=cluster_name,
            cluster_type=cluster_type,
            description=cluster_description,
            entities=entity_ids,
        )

        return cluster

    def _generate_cluster_name(self, entities: List[Entity]) -> str:
        """Generate name for a cluster based on its entities.

        Args:
            entities: List of entities in the cluster

        Returns:
            Generated cluster name
        """
        if not entities:
            return "Empty Cluster"

        # Count entity types
        type_counts: Dict[str, int] = {}
        for entity in entities:
            entity_type = str(entity.entity_type)
            type_counts[entity_type] = type_counts.get(entity_type, 0) + 1

        # Get most common type
        most_common_type = max(type_counts.keys(), key=lambda x: type_counts[x])

        # Generate name based on type and first few entities
        entity_names = [e.name for e in entities[:3]]

        if len(entities) <= 3:
            return f"{most_common_type.title()} Cluster: {', '.join(entity_names)}"

        return f"{most_common_type.title()} Cluster: {', '.join(entity_names)} and {len(entities) - 3} others"

    def _generate_cluster_description(self, entities: List[Entity]) -> str:
        """Generate description for a cluster.

        Args:
            entities: List of entities in the cluster

        Returns:
            Generated cluster description
        """
        if not entities:
            return "Empty cluster"

        # Count entity types
        type_counts: Dict[str, int] = {}
        for entity in entities:
            entity_type = str(entity.entity_type)
            type_counts[entity_type] = type_counts.get(entity_type, 0) + 1

        type_summary = []
        for entity_type, count in sorted(type_counts.items()):
            type_summary.append(f"{count} {entity_type}{'s' if count > 1 else ''}")

        return f"Cluster containing {', '.join(type_summary)} ({len(entities)} entities total)"

    def validate_cluster(self, cluster: Cluster) -> bool:
        """Validate a generated cluster.

        Args:
            cluster: Cluster to validate

        Returns:
            True if cluster is valid
        """
        if not cluster.entities:
            return False

        if len(cluster.entities) < self.min_cluster_size:
            return False

        if len(cluster.entities) > self.max_cluster_size:
            return False

        return True

    def post_process_clusters(self, clusters: List[Cluster]) -> List[Cluster]:
        """Post-process generated clusters.

        Args:
            clusters: List of clusters to process

        Returns:
            Processed list of clusters
        """
        # Filter out invalid clusters
        valid_clusters = [c for c in clusters if self.validate_cluster(c)]

        # Remove overlapping clusters (keep the one with higher cohesion)
        return self._remove_overlapping_clusters(valid_clusters)

    def _remove_overlapping_clusters(self, clusters: List[Cluster]) -> List[Cluster]:
        """Remove overlapping clusters, keeping the better one.

        Args:
            clusters: List of clusters

        Returns:
            List of non-overlapping clusters
        """
        if len(clusters) <= 1:
            return clusters

        # Sort clusters by size (larger first)
        clusters.sort(key=lambda c: len(c.entities), reverse=True)

        final_clusters = []
        used_entities: Set[str] = set()

        for cluster in clusters:
            # Check overlap with already selected clusters
            overlap = len(cluster.entities.intersection(used_entities))
            overlap_ratio = overlap / len(cluster.entities) if cluster.entities else 0

            # If overlap is small, keep the cluster
            if overlap_ratio < 0.5:  # Less than 50% overlap
                final_clusters.append(cluster)
                used_entities.update(cluster.entities)

        return final_clusters
