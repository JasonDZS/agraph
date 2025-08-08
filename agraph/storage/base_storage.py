"""
Graph storage base class.

Provides abstract base class for graph storage implementations.
"""

import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ..entities import Entity
from ..graph import KnowledgeGraph
from ..logger import logger
from ..relations import Relation


class GraphStorage(ABC):
    """Abstract base class for graph storage implementations."""

    def __init__(self) -> None:
        """Initialize the graph storage."""
        self.connection = None
        self._is_connected = False

    def is_connected(self) -> bool:
        """
        Check if connected to storage backend.

        Returns:
            bool: True if connected
        """
        return self._is_connected

    def set_connected(self, value: bool) -> None:
        """
        Set connection status.

        Args:
            value: Connection status to set
        """
        self._is_connected = value

    @abstractmethod
    def connect(self) -> bool:
        """
        Connect to storage backend.

        Returns:
            bool: True if connection successful
        """

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from storage backend."""

    @abstractmethod
    def save_graph(self, graph: KnowledgeGraph) -> bool:
        """
        Save knowledge graph to storage.

        Args:
            graph: Knowledge graph to save

        Returns:
            bool: True if save successful
        """

    @abstractmethod
    def load_graph(self, graph_id: str) -> Optional[KnowledgeGraph]:
        """
        Load knowledge graph from storage.

        Args:
            graph_id: Graph identifier

        Returns:
            KnowledgeGraph: Loaded knowledge graph, None if not found
        """

    @abstractmethod
    def delete_graph(self, graph_id: str) -> bool:
        """
        Delete knowledge graph from storage.

        Args:
            graph_id: Graph identifier

        Returns:
            bool: True if delete successful
        """

    @abstractmethod
    def list_graphs(self) -> List[Dict[str, Any]]:
        """
        List all available graphs.

        Returns:
            List[Dict[str, Any]]: Graph metadata list
        """

    @abstractmethod
    def query_entities(self, conditions: Dict[str, Any]) -> List[Entity]:
        """
        Query entities based on conditions.

        Args:
            conditions: Query conditions

        Returns:
            List[Entity]: Matching entities
        """

    @abstractmethod
    def query_relations(
        self,
        head_entity: Optional[str] = None,
        tail_entity: Optional[str] = None,
        relation_type: Optional[Any] = None,
    ) -> List[Relation]:
        """
        Query relations based on conditions.

        Args:
            head_entity: Head entity ID
            tail_entity: Tail entity ID
            relation_type: Relation type

        Returns:
            List[Relation]: Matching relations
        """

    @abstractmethod
    def add_entity(self, graph_id: str, entity: Entity) -> bool:
        """
        Add entity to graph.

        Args:
            graph_id: Graph identifier
            entity: Entity object to add

        Returns:
            bool: True if add successful
        """

    @abstractmethod
    def add_relation(self, graph_id: str, relation: Relation) -> bool:
        """
        Add relation to graph.

        Args:
            graph_id: Graph identifier
            relation: Relation object to add

        Returns:
            bool: True if add successful
        """

    @abstractmethod
    def update_entity(self, graph_id: str, entity: Entity) -> bool:
        """
        Update entity in graph.

        Args:
            graph_id: Graph identifier
            entity: Entity object to update

        Returns:
            bool: True if update successful
        """

    @abstractmethod
    def update_relation(self, graph_id: str, relation: Relation) -> bool:
        """
        Update relation in graph.

        Args:
            graph_id: Graph identifier
            relation: Relation object to update

        Returns:
            bool: True if update successful
        """

    @abstractmethod
    def remove_entity(self, graph_id: str, entity_id: str) -> bool:
        """
        Remove entity from graph.

        Args:
            graph_id: Graph identifier
            entity_id: Entity ID to remove

        Returns:
            bool: True if remove successful
        """

    @abstractmethod
    def remove_relation(self, graph_id: str, relation_id: str) -> bool:
        """
        Remove relation from graph.

        Args:
            graph_id: Graph identifier
            relation_id: Relation ID to remove

        Returns:
            bool: True if remove successful
        """

    def get_graph_statistics(self, graph_id: str) -> Dict[str, Any]:
        """
        Get graph statistics.

        Args:
            graph_id: Graph identifier

        Returns:
            Dict[str, Any]: Statistics information
        """
        try:
            graph = self.load_graph(graph_id)
            if graph:
                return graph.get_basic_statistics()
            return {}
        except Exception as e:
            logger.error("Error getting graph statistics: %s", e)
            return {}

    def backup_graph(self, graph_id: str, backup_path: str) -> bool:
        """
        Backup graph to file.

        Args:
            graph_id: Graph identifier
            backup_path: Backup file path

        Returns:
            bool: True if backup successful
        """
        try:
            graph = self.load_graph(graph_id)
            if not graph:
                return False

            # Subclasses can override this method for specific backup logic
            graph_data = graph.to_dict()

            with open(backup_path, "w", encoding="utf-8") as f:
                json.dump(graph_data, f, ensure_ascii=False, indent=2)

            logger.info("Graph %s backed up to %s", graph_id, backup_path)
            return True

        except Exception as e:
            logger.error("Error backing up graph %s: %s", graph_id, e)
            return False

    def restore_graph(self, backup_path: str) -> Optional[str]:
        """
        Restore graph from backup file.

        Args:
            backup_path: Backup file path

        Returns:
            str: Restored graph ID, None if failed
        """
        try:

            with open(backup_path, "r", encoding="utf-8") as f:
                graph_data = json.load(f)

            graph = KnowledgeGraph.from_dict(graph_data)

            if self.save_graph(graph):
                logger.info("Graph restored from %s with ID %s", backup_path, graph.id)
                return graph.id

            return None

        except Exception as e:
            logger.error("Error restoring graph from %s: %s", backup_path, e)
            return None

    def export_graph(self, graph_id: str, outformat: str = "json") -> Optional[Dict[str, Any]]:
        """
        Export graph data in specified format.

        Args:
            graph_id: Graph identifier
            outformat: Export format ('json', 'csv', 'graphml', etc.)

        Returns:
            Dict[str, Any]: Exported data, None if failed
        """
        try:
            graph = self.load_graph(graph_id)
            if not graph:
                return None

            if outformat.lower() == "json":
                return graph.to_dict()
            if outformat.lower() == "csv":
                return self._export_to_csv_format(graph)
            if outformat.lower() == "graphml":
                return self._export_to_graphml_format(graph)
            logger.warning("Unsupported export outformat: %s", outformat)
            return None

        except Exception as e:
            logger.error("Error exporting graph %s: %s", graph_id, e)
            return None

    def _export_to_csv_format(self, graph: KnowledgeGraph) -> Dict[str, Any]:
        """Export graph data to CSV format."""
        from ..utils import get_type_value

        entities_data = []
        for entity in graph.entities.values():
            entities_data.append(
                {
                    "id": entity.id,
                    "name": entity.name,
                    "type": get_type_value(entity.entity_type),
                    "description": entity.description,
                    "confidence": entity.confidence,
                    "source": entity.source,
                }
            )

        relations_data = []
        for relation in graph.relations.values():
            if relation.head_entity is None or relation.tail_entity is None:
                continue
            relations_data.append(
                {
                    "id": relation.id,
                    "head_entity": relation.head_entity.name,
                    "tail_entity": relation.tail_entity.name,
                    "relation_type": get_type_value(relation.relation_type),
                    "confidence": relation.confidence,
                    "source": relation.source,
                }
            )

        return {"entities": entities_data, "relations": relations_data}

    def _export_to_graphml_format(self, graph: KnowledgeGraph) -> Dict[str, Any]:
        """Export graph data to GraphML format."""
        from ..utils import get_type_value

        # Simplified GraphML format
        nodes = []
        edges = []

        for entity in graph.entities.values():
            nodes.append(
                {
                    "id": entity.id,
                    "label": entity.name,
                    "type": get_type_value(entity.entity_type),
                    "description": entity.description,
                }
            )

        for relation in graph.relations.values():
            if relation.head_entity is None or relation.tail_entity is None:
                continue
            edges.append(
                {
                    "id": relation.id,
                    "source": relation.head_entity.id,
                    "target": relation.tail_entity.id,
                    "label": get_type_value(relation.relation_type),
                    "confidence": relation.confidence,
                }
            )

        return {"graph": {"nodes": nodes, "edges": edges}}
