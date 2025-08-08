"""
JSON file storage implementation.

Provides JSON file-based storage for knowledge graphs.
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..config import settings
from ..entities import Entity
from ..graph import KnowledgeGraph
from ..logger import logger
from ..relations import Relation
from .base_storage import GraphStorage


class JsonStorage(GraphStorage):
    """JSON file-based graph storage implementation."""

    def __init__(self, storage_dir: Optional[str] = None):
        """
        Initialize JSON storage.

        Args:
            storage_dir: Directory to store JSON files. If None, uses workdir from settings.
        """
        super().__init__()
        if storage_dir is not None:
            self.storage_dir = storage_dir
        else:
            self.storage_dir = settings.workdir
        self.graphs_file = os.path.join(self.storage_dir, "graphs.json")
        self.ensure_storage_dir()

    def ensure_storage_dir(self) -> None:
        """Ensure storage directory exists."""
        if not os.path.exists(self.storage_dir):
            os.makedirs(self.storage_dir)
            logger.info("Created storage directory: %s", self.storage_dir)

    def connect(self) -> bool:
        """Connect to file system storage."""
        try:
            self.ensure_storage_dir()
            self._is_connected = True
            logger.info("Connected to JSON storage at %s", self.storage_dir)
            return True
        except Exception as e:
            logger.error("Failed to connect to JSON storage: %s", e)
            return False

    def disconnect(self) -> None:
        """Disconnect from storage."""
        self._is_connected = False
        logger.info("Disconnected from JSON storage")

    def save_graph(self, graph: KnowledgeGraph) -> bool:
        """Save knowledge graph to JSON file."""
        if not self.is_connected():
            if not self.connect():
                return False

        try:
            # 保存图谱数据到单独文件
            graph_file = os.path.join(self.storage_dir, f"{graph.id}.json")
            graph_data = graph.to_dict()

            with open(graph_file, "w", encoding="utf-8") as f:
                json.dump(graph_data, f, ensure_ascii=False, indent=2)

            # 更新图谱索引
            self._update_graph_index(graph)

            logger.info("Graph %s saved to %s", graph.id, graph_file)
            return True

        except Exception as e:
            logger.error("Error saving graph to JSON: %s", e)
            return False

    def load_graph(self, graph_id: str) -> Optional[KnowledgeGraph]:
        """Load knowledge graph from JSON file."""
        if not self.is_connected():
            logger.error("Not connected to storage")
            return None

        try:
            graph_file = os.path.join(self.storage_dir, f"{graph_id}.json")

            if not os.path.exists(graph_file):
                logger.warning("Graph file not found: %s", graph_file)
                return None

            with open(graph_file, "r", encoding="utf-8") as f:
                graph_data = json.load(f)

            graph = KnowledgeGraph.from_dict(graph_data)
            logger.info("Graph %s loaded from %s", graph_id, graph_file)
            return graph

        except Exception as e:
            logger.error("Error loading graph from JSON: %s", e)
            return None

    def delete_graph(self, graph_id: str) -> bool:
        """Delete knowledge graph from storage."""
        if not self.is_connected():
            logger.error("Not connected to storage")
            return False

        try:
            graph_file = os.path.join(self.storage_dir, f"{graph_id}.json")

            if os.path.exists(graph_file):
                os.remove(graph_file)
                logger.info("Graph file deleted: %s", graph_file)

            # Remove from index
            self._remove_from_graph_index(graph_id)

            return True

        except Exception as e:
            logger.error("Error deleting graph: %s", e)
            return False

    def list_graphs(self) -> List[Dict[str, Any]]:
        """List all available graphs."""
        if not self.is_connected():
            logger.error("Not connected to storage")
            return []

        try:
            if not os.path.exists(self.graphs_file):
                return []

            with open(self.graphs_file, "r", encoding="utf-8") as f:
                graphs_index = json.load(f)

            result: list[dict[str, Any]] = graphs_index.get("graphs", [])
            return result

        except Exception as e:
            logger.error("Error listing graphs: %s", e)
            return []

    def query_entities(self, conditions: Dict[str, Any]) -> List[Entity]:
        """Query entities based on specified conditions."""
        if not self.is_connected():
            logger.error("Not connected to storage")
            return []

        try:
            graph_id = conditions.get("graph_id")
            if not graph_id:
                logger.error("graph_id is required for entity query")
                return []

            graph = self.load_graph(graph_id)
            if not graph:
                return []

            entities = list(graph.entities.values())

            # Apply filter conditions
            if "entity_type" in conditions:
                entity_type = conditions["entity_type"]
                entities = [e for e in entities if getattr(e.entity_type, "value", str(e.entity_type)) == entity_type]

            if "name" in conditions:
                name_filter = conditions["name"].lower()
                entities = [e for e in entities if name_filter in e.name.lower()]

            if "min_confidence" in conditions:
                min_confidence = conditions["min_confidence"]
                entities = [e for e in entities if e.confidence >= min_confidence]

            # Limit results
            limit = conditions.get("limit", 100)
            return entities[:limit]

        except Exception as e:
            logger.error("Error querying entities: %s", e)
            return []

    def query_relations(
        self,
        head_entity: Optional[str] = None,
        tail_entity: Optional[str] = None,
        relation_type: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[Relation]:
        """Query relations based on specified conditions."""
        if not self.is_connected():
            logger.error("Not connected to storage")
            return []

        try:
            graph_id = kwargs.get("graph_id")
            if not graph_id:
                logger.error("graph_id is required for relation query")
                return []

            graph = self.load_graph(graph_id)
            if not graph:
                return []

            relations = list(graph.relations.values())

            # Apply filter conditions
            if head_entity:
                relations = [r for r in relations if r.head_entity and r.head_entity.id == head_entity]

            if tail_entity:
                relations = [r for r in relations if r.tail_entity and r.tail_entity.id == tail_entity]

            if relation_type:
                relations = [r for r in relations if r.relation_type == relation_type]

            return relations

        except Exception as e:
            logger.error("Error querying relations: %s", e)
            return []

    def add_entity(self, graph_id: str, entity: Entity) -> bool:
        """Add entity to specified graph."""
        try:
            graph = self.load_graph(graph_id)
            if not graph:
                logger.error("Graph %s not found", graph_id)
                return False

            graph.add_entity(entity)
            return self.save_graph(graph)

        except Exception as e:
            logger.error("Error adding entity: %s", e)
            return False

    def add_relation(self, graph_id: str, relation: Relation) -> bool:
        """Add relation to specified graph."""
        try:
            graph = self.load_graph(graph_id)
            if not graph:
                logger.error("Graph %s not found", graph_id)
                return False

            graph.add_relation(relation)
            return self.save_graph(graph)

        except Exception as e:
            logger.error("Error adding relation: %s", e)
            return False

    def update_entity(self, graph_id: str, entity: Entity) -> bool:
        """Update entity in specified graph."""
        try:
            graph = self.load_graph(graph_id)
            if not graph:
                logger.error("Graph %s not found", graph_id)
                return False

            if entity.id in graph.entities:
                graph.entities[entity.id] = entity
                graph.updated_at = datetime.now()
                return self.save_graph(graph)
            return self.add_entity(graph_id, entity)

        except Exception as e:
            logger.error("Error updating entity: %s", e)
            return False

    def update_relation(self, graph_id: str, relation: Relation) -> bool:
        """Update relation in specified graph."""
        try:
            graph = self.load_graph(graph_id)
            if not graph:
                logger.error("Graph %s not found", graph_id)
                return False

            if relation.id in graph.relations:
                graph.relations[relation.id] = relation
                graph.updated_at = datetime.now()
                return self.save_graph(graph)
            return self.add_relation(graph_id, relation)

        except Exception as e:
            logger.error("Error updating relation: %s", e)
            return False

    def remove_entity(self, graph_id: str, entity_id: str) -> bool:
        """Remove entity from specified graph."""
        try:
            graph = self.load_graph(graph_id)
            if not graph:
                logger.error("Graph %s not found", graph_id)
                return False

            success = graph.remove_entity(entity_id)
            if success:
                return self.save_graph(graph)
            return False

        except Exception as e:
            logger.error("Error removing entity: %s", e)
            return False

    def remove_relation(self, graph_id: str, relation_id: str) -> bool:
        """Remove relation from specified graph."""
        try:
            graph = self.load_graph(graph_id)
            if not graph:
                logger.error("Graph %s not found", graph_id)
                return False

            success = graph.remove_relation(relation_id)
            if success:
                return self.save_graph(graph)
            return False

        except Exception as e:
            logger.error("Error removing relation: %s", e)
            return False

    def _update_graph_index(self, graph: KnowledgeGraph) -> None:
        """Update graph index with new graph information."""
        try:
            graphs_index: Dict[str, Any] = {"graphs": []}

            if os.path.exists(self.graphs_file):
                with open(self.graphs_file, "r", encoding="utf-8") as f:
                    graphs_index = json.load(f)

            # Remove existing graph record
            graphs_list = graphs_index.get("graphs", [])
            graphs_list = [g for g in graphs_list if g.get("id") != graph.id]

            # Add new record
            graph_info = {
                "id": graph.id,
                "name": graph.name,
                "created_at": graph.created_at.isoformat(),
                "updated_at": graph.updated_at.isoformat(),
                "entity_count": len(graph.entities),
                "relation_count": len(graph.relations),
            }
            graphs_list.append(graph_info)

            # Sort by update time
            graphs_list.sort(key=lambda x: x["updated_at"], reverse=True)

            graphs_index["graphs"] = graphs_list

            with open(self.graphs_file, "w", encoding="utf-8") as f:
                json.dump(graphs_index, f, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.error("Error updating graph index: %s", e)

    def _remove_from_graph_index(self, graph_id: str) -> None:
        """Remove graph from index."""
        try:
            if not os.path.exists(self.graphs_file):
                return

            with open(self.graphs_file, "r", encoding="utf-8") as f:
                graphs_index = json.load(f)

            graphs_list = graphs_index.get("graphs", [])
            graphs_list = [g for g in graphs_list if g.get("id") != graph_id]

            graphs_index["graphs"] = graphs_list

            with open(self.graphs_file, "w", encoding="utf-8") as f:
                json.dump(graphs_index, f, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.error("Error removing from graph index: %s", e)

    def compact_storage(self) -> None:
        """Compact storage by removing invalid index entries."""
        try:
            # Clean up non-existent graph indexes
            graphs_list = self.list_graphs()
            valid_graphs = []

            for graph_info in graphs_list:
                graph_id = graph_info.get("id")
                graph_file = os.path.join(self.storage_dir, f"{graph_id}.json")

                if os.path.exists(graph_file):
                    valid_graphs.append(graph_info)
                else:
                    logger.info("Removing invalid graph index entry: %s", graph_id)

            # Update index
            graphs_index = {"graphs": valid_graphs}
            with open(self.graphs_file, "w", encoding="utf-8") as f:
                json.dump(graphs_index, f, ensure_ascii=False, indent=2)

            logger.info("Storage compaction completed")

        except Exception as e:
            logger.error("Error compacting storage: %s", e)

    def get_storage_info(self) -> Dict[str, Any]:
        """Get storage information and statistics."""
        try:
            total_size = 0
            file_count = 0

            for filename in os.listdir(self.storage_dir):
                if filename.endswith(".json"):
                    file_path = os.path.join(self.storage_dir, filename)
                    total_size += os.path.getsize(file_path)
                    file_count += 1

            graphs_count = len(self.list_graphs())

            return {
                "storage_dir": self.storage_dir,
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "file_count": file_count,
                "graphs_count": graphs_count,
                "is_connected": self.is_connected(),
            }

        except Exception as e:
            logger.error("Error getting storage info: %s", e)
            return {}
