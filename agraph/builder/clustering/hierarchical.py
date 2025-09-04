"""
Hierarchical clustering algorithm.
"""

from typing import Any, Dict, List, Optional, Set, Tuple

from ...base.core.types import ClusterType
from ...base.models.clusters import Cluster
from ...base.models.entities import Entity
from ...base.models.relations import Relation
from .base import ClusterAlgorithm


class HierarchicalClusteringAlgorithm(ClusterAlgorithm):
    """Hierarchical clustering algorithm using agglomerative clustering."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize hierarchical clustering algorithm.

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.linkage = self.config.get("linkage", "average")  # 'single', 'complete', 'average'
        self.distance_threshold = self.config.get("distance_threshold", 0.5)
        self.max_clusters = self.config.get("max_clusters", 20)

    def cluster(self, entities: List[Entity], relations: List[Relation]) -> List[Cluster]:
        """Perform hierarchical clustering.

        Args:
            entities: List of entities to cluster
            relations: List of relations between entities

        Returns:
            List of hierarchical clusters
        """
        if len(entities) < self.min_cluster_size:
            return []

        # Build adjacency matrix
        adjacency = self.build_adjacency_matrix(entities, relations)
        entities_map = {e.id: e for e in entities}

        # Calculate distance matrix
        distance_matrix = self._calculate_distance_matrix(adjacency)

        # Perform agglomerative clustering
        clusters_hierarchy = self._agglomerative_clustering(list(adjacency.keys()), distance_matrix)

        # Convert to final clusters
        final_clusters = []
        for cluster_entities in clusters_hierarchy:
            cluster = self.create_cluster(cluster_entities, entities_map, ClusterType.TOPIC)
            if cluster:
                final_clusters.append(cluster)

        return self.post_process_clusters(final_clusters)

    def _calculate_distance_matrix(
        self, adjacency: Dict[str, Dict[str, float]]
    ) -> Dict[Tuple[str, str], float]:
        """Calculate distance matrix from adjacency matrix.

        Args:
            adjacency: Adjacency matrix (similarity)

        Returns:
            Distance matrix
        """
        nodes = list(adjacency.keys())
        distance_matrix = {}

        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes):
                if i >= j:
                    continue

                # Convert similarity to distance
                similarity = adjacency.get(node1, {}).get(node2, 0.0)
                distance = 1.0 - similarity  # Simple conversion

                distance_matrix[(node1, node2)] = distance
                distance_matrix[(node2, node1)] = distance

        return distance_matrix

    def _agglomerative_clustering(
        self, nodes: List[str], distance_matrix: Dict[Tuple[str, str], float]
    ) -> List[Set[str]]:
        """Perform agglomerative clustering.

        Args:
            nodes: List of node IDs
            distance_matrix: Distance matrix between nodes

        Returns:
            List of clusters (sets of node IDs)
        """
        # Initialize each node as its own cluster
        clusters = [{node} for node in nodes]

        # Keep merging until stopping criteria
        while len(clusters) > 1:
            if len(clusters) <= self.max_clusters:
                # Check if all remaining clusters are above distance threshold
                min_distance = float("inf")
                for i, cluster1 in enumerate(clusters):
                    for j, cluster2 in enumerate(clusters):
                        if i >= j:
                            continue

                        distance = self._calculate_cluster_distance(
                            cluster1, cluster2, distance_matrix
                        )

                        min_distance = min(min_distance, distance)

                if min_distance > self.distance_threshold:
                    break

            # Find closest pair of clusters
            min_distance = float("inf")
            merge_i, merge_j = -1, -1

            for i, cluster1 in enumerate(clusters):
                for j, cluster2 in enumerate(clusters):
                    if i >= j:
                        continue

                    distance = self._calculate_cluster_distance(cluster1, cluster2, distance_matrix)

                    if distance < min_distance:
                        min_distance = distance
                        merge_i, merge_j = i, j

            if merge_i == -1 or merge_j == -1:
                break

            # Merge the closest clusters
            merged_cluster = clusters[merge_i].union(clusters[merge_j])

            # Remove the original clusters (remove higher index first)
            if merge_i < merge_j:
                clusters.pop(merge_j)
                clusters.pop(merge_i)
            else:
                clusters.pop(merge_i)
                clusters.pop(merge_j)

            # Add merged cluster
            clusters.append(merged_cluster)

        # Filter clusters by minimum size
        return [cluster for cluster in clusters if len(cluster) >= self.min_cluster_size]

    def _calculate_cluster_distance(
        self, cluster1: Set[str], cluster2: Set[str], distance_matrix: Dict[Tuple[str, str], float]
    ) -> float:
        """Calculate distance between two clusters.

        Args:
            cluster1: First cluster
            cluster2: Second cluster
            distance_matrix: Distance matrix between individual nodes

        Returns:
            Distance between clusters
        """
        if self.linkage == "single":
            # Single linkage: minimum distance
            min_distance = float("inf")
            for node1 in cluster1:
                for node2 in cluster2:
                    distance = distance_matrix.get((node1, node2), 1.0)
                    min_distance = min(min_distance, distance)
            return min_distance

        if self.linkage == "complete":
            # Complete linkage: maximum distance
            max_distance = 0.0
            for node1 in cluster1:
                for node2 in cluster2:
                    distance = distance_matrix.get((node1, node2), 1.0)
                    max_distance = max(max_distance, distance)
            return max_distance

        # Average linkage (default)
        # Average linkage: average distance
        total_distance = 0.0
        pair_count = 0

        for node1 in cluster1:
            for node2 in cluster2:
                distance = distance_matrix.get((node1, node2), 1.0)
                total_distance += distance
                pair_count += 1

        if pair_count == 0:
            return 1.0

        return total_distance / pair_count
