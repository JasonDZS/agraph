"""
Graph Path Finding Service

This service provides comprehensive path finding and graph traversal algorithms
for knowledge graphs. It implements various search strategies to discover
relationships and connections between entities.

Key Features:
- Shortest path finding using BFS algorithm
- Comprehensive path enumeration with DFS
- Filtered path search by relation types
- Cycle detection in directed graphs
- Configurable search depth limits
"""

import logging
from collections import deque
from typing import List, Optional, Set, Union

from ..graph import KnowledgeGraph
from ..relations import Relation
from ..types import RelationType

logger = logging.getLogger(__name__)


class GraphPathFinder:
    """
    Service for graph path finding and traversal algorithms.

    This service implements various path finding algorithms to discover connections
    between entities in knowledge graphs. It supports both breadth-first and
    depth-first search strategies with configurable constraints.

    The path finder can:
    - Find shortest paths between entities
    - Enumerate all possible paths with depth limits
    - Filter paths by relation types
    - Detect cycles in the graph structure
    """

    def __init__(self) -> None:
        """Initialize path finder"""
        pass

    def find_shortest_path(
        self, graph: KnowledgeGraph, start_entity_id: str, end_entity_id: str, max_depth: int = 5
    ) -> Optional[List[str]]:
        """
        Find the shortest path between two entities using breadth-first search.

        This method uses BFS to guarantee finding the shortest path (minimum number
        of hops) between the start and end entities. The search respects the
        maximum depth limit to prevent excessive computation.

        Args:
            graph: Knowledge graph to search within
            start_entity_id: ID of the starting entity for path search
            end_entity_id: ID of the target entity to reach
            max_depth: Maximum number of hops to search (default: 5)

        Returns:
            Optional[List[str]]: List of entity IDs representing the shortest path
                from start to end, including both endpoints. Returns None if no
                path exists within the depth limit or if entities don't exist.

        Note:
            - Path includes both start and end entity IDs
            - Empty list is never returned; None indicates no path found
            - Search considers undirected connections between entities
        """
        if start_entity_id not in graph.entities or end_entity_id not in graph.entities:
            return None

        if start_entity_id == end_entity_id:
            return [start_entity_id]

        try:
            queue = deque([(start_entity_id, [start_entity_id])])
            visited: Set[str] = {start_entity_id}

            while queue:
                current_id, path = queue.popleft()

                if len(path) > max_depth:
                    continue

                # Get neighbors
                neighbors = graph.get_neighbors(current_id)

                for neighbor in neighbors:
                    neighbor_id = neighbor.id

                    if neighbor_id == end_entity_id:
                        return path + [neighbor_id]

                    if neighbor_id not in visited:
                        visited.add(neighbor_id)
                        queue.append((neighbor_id, path + [neighbor_id]))

            return None

        except Exception as e:
            logger.error(f"Error finding shortest path: {e}")
            return None

    def find_all_paths(
        self, graph: KnowledgeGraph, start_entity_id: str, end_entity_id: str, max_depth: int = 3
    ) -> List[List[Relation]]:
        """
        Find all paths between two entities using DFS

        Args:
            graph: Knowledge graph
            start_entity_id: Start entity ID
            end_entity_id: End entity ID
            max_depth: Maximum search depth

        Returns:
            List[List[Relation]]: List of paths, each path is list of relations
        """
        if start_entity_id not in graph.entities or end_entity_id not in graph.entities:
            return []

        try:
            paths = []
            visited: Set[str] = set()

            def dfs(current_id: str, target_id: str, path: List[Relation], depth: int) -> None:
                if depth > max_depth:
                    return

                if current_id == target_id and path:
                    paths.append(path.copy())
                    return

                if current_id in visited:
                    return

                visited.add(current_id)

                # Explore outgoing relations
                relations = graph.get_entity_relations(current_id, direction="out")
                for relation in relations:
                    if relation.tail_entity and relation.tail_entity.id not in visited:
                        path.append(relation)
                        dfs(relation.tail_entity.id, target_id, path, depth + 1)
                        path.pop()

                visited.remove(current_id)

            dfs(start_entity_id, end_entity_id, [], 0)
            return paths

        except Exception as e:
            logger.error(f"Error finding all paths: {e}")
            return []

    def find_paths_with_relation_type(
        self,
        graph: KnowledgeGraph,
        start_entity_id: str,
        end_entity_id: str,
        relation_types: List[Union[RelationType, str]],
        max_depth: int = 3,
        max_paths: int = 10,
    ) -> List[List[Relation]]:
        """Find paths between entities with specific relation types."""
        from ..utils import get_type_value

        """
        Find paths between entities using only specified relation types

        Args:
            graph: Knowledge graph
            start_entity_id: Start entity ID
            end_entity_id: End entity ID
            relation_types: Allowed relation types
            max_depth: Maximum search depth

        Returns:
            List[List[Relation]]: Filtered paths
        """
        if start_entity_id not in graph.entities or end_entity_id not in graph.entities:
            return []

        try:
            paths = []
            visited: Set[str] = set()

            def dfs(current_id: str, target_id: str, path: List[Relation], depth: int) -> None:
                if depth > max_depth:
                    return

                if current_id == target_id and path:
                    paths.append(path.copy())
                    return

                if current_id in visited:
                    return

                visited.add(current_id)

                # Explore relations with allowed types only
                relations = graph.get_entity_relations(current_id, direction="out")
                for relation in relations:
                    if (
                        get_type_value(relation.relation_type) in [get_type_value(rt) for rt in relation_types]
                        and relation.tail_entity
                        and relation.tail_entity.id not in visited
                    ):

                        path.append(relation)
                        dfs(relation.tail_entity.id, target_id, path, depth + 1)
                        path.pop()

                visited.remove(current_id)

            dfs(start_entity_id, end_entity_id, [], 0)
            return paths

        except Exception as e:
            logger.error(f"Error finding paths with relation types: {e}")
            return []

    def find_cycles(self, graph: KnowledgeGraph, max_cycle_length: int = 5) -> List[List[str]]:
        """
        Detect cycles in the graph

        Args:
            graph: Knowledge graph
            max_cycle_length: Maximum cycle length to detect

        Returns:
            List[List[str]]: List of cycles, each cycle is list of entity IDs
        """
        cycles = []
        visited: Set[str] = set()
        rec_stack: Set[str] = set()

        def dfs_cycle(entity_id: str, path: List[str]) -> bool:
            if len(path) > max_cycle_length:
                return False

            if entity_id in rec_stack:
                # Found cycle
                try:
                    cycle_start = path.index(entity_id)
                    cycle = path[cycle_start:] + [entity_id]
                    cycles.append(cycle)
                    return True
                except ValueError:
                    pass

            if entity_id in visited:
                return False

            visited.add(entity_id)
            rec_stack.add(entity_id)
            path.append(entity_id)

            # Check outgoing relations
            relations = graph.get_entity_relations(entity_id, direction="out")
            for relation in relations:
                if relation.tail_entity:
                    if dfs_cycle(relation.tail_entity.id, path):
                        break  # Found a cycle, can stop exploring this branch

            path.pop()
            rec_stack.remove(entity_id)
            return False

        try:
            for entity_id in graph.entities:
                if entity_id not in visited:
                    dfs_cycle(entity_id, [])

            return cycles

        except Exception as e:
            logger.error(f"Error detecting cycles: {e}")
            return []
