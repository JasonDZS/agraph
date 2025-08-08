"""
Improved LightRAG builders following Interface Segregation Principle

These builders implement only the interfaces they need, demonstrating
proper ISP compliance through composition and focused responsibilities.

Based on LightRAG framework for knowledge graph construction with automatic
entity and relation extraction from documents, generating GraphML format files.
"""

import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, TypeVar

from ..config import settings
from ..entities import Entity
from ..graph import KnowledgeGraph
from ..logger import logger
from ..relations import Relation
from ..types import EntityType, RelationType
from ..utils import get_type_value
from .interfaces import (
    BasicGraphBuilder,
    BatchGraphBuilder,
    FullFeaturedGraphBuilder,
    GraphExporter,
    StreamingGraphBuilder,
    UpdatableGraphBuilder,
)
from .mixins import (
    GraphExporterMixin,
    GraphMergerMixin,
    GraphStatisticsMixin,
    GraphValidatorMixin,
    IncrementalBuilderMixin,
)

# 定义类型变量
T = TypeVar("T")


class LightRAGCore:
    """Core LightRAG functionality class with single responsibility for managing LightRAG instances and GraphML parsing.

    This class encapsulates all LightRAG-specific operations including initialization,
    document processing, graph building, and GraphML file handling. It follows the
    single responsibility principle by focusing solely on LightRAG operations.

    Attributes:
        working_dir: Directory path for LightRAG storage and generated files.
        rag_instance: The initialized LightRAG instance.
        _initialized: Flag indicating whether the LightRAG instance is properly initialized.
    """

    def __init__(self, working_dir: str = "lightrag_storage"):
        """Initialize the LightRAG core functionality.

        Args:
            working_dir: Directory path for storing LightRAG data and generated files.
                Defaults to "lightrag_storage".
        """
        self.working_dir = Path(working_dir)
        self.rag_instance: Optional[Any] = None
        self._initialized: bool = False

    def __del__(self) -> None:
        """Ensure resources are properly cleaned up during garbage collection."""
        if self._initialized and self.rag_instance:
            try:
                self.cleanup()
            except Exception:
                pass

    async def initialize_lightrag(self) -> Any:
        """Initialize the LightRAG instance with custom configuration.

        This method sets up the LightRAG instance with custom LLM and embedding functions
        based on the application settings. It ensures that the instance is only initialized
        once and handles all necessary storage and pipeline initialization.

        Returns:
            Any: The initialized LightRAG instance.

        Raises:
            ImportError: If LightRAG is not installed.
            RuntimeError: If initialization fails for any reason.
        """
        if self._initialized and self.rag_instance is not None:
            return self.rag_instance

        try:
            # pylint: disable=import-outside-toplevel
            # Lazy import of LightRAG to avoid dependency issues
            from lightrag import LightRAG
            from lightrag.kg.shared_storage import initialize_pipeline_status
            from lightrag.llm.openai import openai_complete_if_cache, openai_embed
            from lightrag.utils import EmbeddingFunc

            # Ensure working directory exists
            self.working_dir.mkdir(parents=True, exist_ok=True)

            # Create custom LLM function
            async def custom_llm_complete(
                prompt: str,
                system_prompt: Optional[str] = None,
                history_messages: Optional[List[Any]] = None,
                **kwargs: Any,
            ) -> str:
                result = await openai_complete_if_cache(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    history_messages=history_messages or [],
                    model=settings.LLM_MODEL,
                    base_url=settings.OPENAI_API_BASE,
                    api_key=settings.OPENAI_API_KEY,
                    **kwargs,
                )
                return str(result)

            # Create custom embedding function
            async def custom_embed(texts: List[str]) -> List[List[float]]:
                result = await openai_embed(
                    texts,
                    model=settings.EMBEDDING_MODEL,
                    base_url=settings.OPENAI_API_BASE,
                    api_key=settings.OPENAI_API_KEY,
                )
                return list(result)

            # Initialize LightRAG instance
            self.rag_instance = LightRAG(
                working_dir=str(self.working_dir),
                llm_model_func=custom_llm_complete,
                embedding_func=EmbeddingFunc(
                    embedding_dim=settings.EMBEDDING_DIM,
                    max_token_size=settings.EMBEDDING_MAX_TOKEN_SIZE,
                    func=custom_embed,
                ),
            )

            # Critical: Both initialization calls are required!
            if self.rag_instance:
                await self.rag_instance.initialize_storages()  # Initialize storage backends
            await initialize_pipeline_status()  # Initialize processing pipeline

            self._initialized = True
            logger.info("LightRAG initialized with working directory: %s", self.working_dir)
            return self.rag_instance

        except ImportError:
            logger.error("LightRAG not installed. Please install with: pip install lightrag")
            self.rag_instance = None
            raise
        except Exception as e:
            logger.error("Failed to initialize LightRAG: %s", e)
            self.rag_instance = None
            self._initialized = False
            raise

    async def build_graph_async(
        self,
        texts: Optional[List[str]] = None,
        database_schema: Optional[Dict[str, Any]] = None,
        graph_name: str = "lightrag_graph",
    ) -> KnowledgeGraph:
        """Asynchronously build a knowledge graph from input data.

        This is the main entry point for graph building operations. It delegates
        to the internal implementation while providing a clean public interface.

        Args:
            texts: List of text documents to process for entity and relation extraction.
            database_schema: Database schema information (currently not used but kept
                for interface compatibility).
            graph_name: Name identifier for the generated knowledge graph.

        Returns:
            KnowledgeGraph: Constructed knowledge graph containing extracted entities
                and relations from the input texts.
        """
        return await self.abuild_graph(texts, database_schema, graph_name)

    async def abuild_graph(
        self,
        texts: Optional[List[str]] = None,
        database_schema: Optional[Dict[str, Any]] = None,
        graph_name: str = "lightrag_graph",
    ) -> KnowledgeGraph:
        """Internal implementation for asynchronously building knowledge graphs.

        This method handles the core logic of graph construction using LightRAG,
        including text processing, GraphML file generation, and graph loading.

        Args:
            texts: List of text documents to process.
            database_schema: Database schema information (currently unused).
            graph_name: Name for the generated graph.

        Returns:
            KnowledgeGraph: The constructed knowledge graph.

        Raises:
            RuntimeError: If LightRAG instance initialization fails.
            Exception: For any other errors during graph building.
        """
        # Initialize RAG instance
        await self.initialize_lightrag()

        # Ensure instance is initialized
        if self.rag_instance is None:
            raise RuntimeError("Failed to initialize LightRAG instance")

        try:
            logger.info("Building knowledge graph with LightRAG: %s", graph_name)

            # Process text inputs
            if texts:
                for i, text in enumerate(texts):
                    logger.info("Inserting text document %d/%d", i + 1, len(texts))
                    # Use asynchronous version to insert documents
                    await self.rag_instance.ainsert(text)

            # Wait for LightRAG to complete processing and generate GraphML file
            graphml_file = self.working_dir / "graph_chunk_entity_relation.graphml"

            if graphml_file.exists():
                # Load knowledge graph from GraphML file
                graph = self._load_graph_from_graphml(str(graphml_file), graph_name)
                logger.info(
                    "Successfully built graph: %d entities, %d relations",
                    len(graph.entities),
                    len(graph.relations),
                )
                return graph

            logger.warning("GraphML file not generated by LightRAG, creating empty graph")
            return KnowledgeGraph(name=graph_name)

        except Exception as e:
            logger.error("Error building graph with LightRAG: %s", e)
            raise

    async def update_graph_with_texts_async(
        self,
        texts: List[str],
        graph_name: Optional[str] = None,
    ) -> KnowledgeGraph:
        """Update the knowledge graph with new texts asynchronously.

        This method provides the main entry point for updating an existing
        knowledge graph with additional text documents.

        Args:
            texts: List of new text documents to add to the graph.
            graph_name: Optional name for the updated graph.

        Returns:
            KnowledgeGraph: Updated knowledge graph containing both existing
                and new entities and relations.
        """
        return await self.aadd_documents(texts, graph_name)

    async def aadd_documents(self, documents: List[str], graph_name: Optional[str] = None) -> KnowledgeGraph:
        """Internal implementation for asynchronously adding documents to the graph.

        This method handles the core logic of document addition, including
        processing new documents through LightRAG and reloading the updated graph.

        Args:
            documents: List of documents to add to the knowledge graph.
            graph_name: Optional name for the updated graph. If None,
                defaults to "lightrag_graph".

        Returns:
            KnowledgeGraph: Updated knowledge graph with new documents processed.

        Raises:
            RuntimeError: If LightRAG instance initialization fails.
            Exception: For any other errors during document addition.
        """
        # Initialize RAG instance
        await self.initialize_lightrag()

        # Ensure instance is initialized
        if self.rag_instance is None:
            raise RuntimeError("Failed to initialize LightRAG instance")

        try:
            logger.info("Adding %d documents to knowledge graph", len(documents))

            for i, doc in enumerate(documents):
                logger.info("Adding document %d/%d", i + 1, len(documents))
                await self.rag_instance.ainsert(doc)

            # Reload the graph
            graphml_file = self.working_dir / "graph_chunk_entity_relation.graphml"
            if graphml_file.exists():
                graph = self._load_graph_from_graphml(str(graphml_file), graph_name or "lightrag_graph")
                logger.info(
                    "Updated graph: %d entities, %d relations",
                    len(graph.entities),
                    len(graph.relations),
                )
                return graph
            logger.warning("GraphML file not found after adding documents")
            return KnowledgeGraph(name=graph_name or "lightrag_graph")

        except Exception as e:
            logger.error("Error adding documents: %s", e)
            raise

    async def asearch_graph(
        self, query: str, search_type: Literal["naive", "local", "global", "hybrid"] = "hybrid"
    ) -> Dict[str, Any]:
        """Internal implementation for asynchronously searching the knowledge graph.

        This method provides semantic search capabilities over the knowledge graph
        using different search strategies supported by LightRAG.

        Args:
            query: Search query string to find relevant information.
            search_type: Type of search strategy to use. Options are:
                - "naive": Simple keyword-based search
                - "local": Local neighborhood search
                - "global": Global graph search
                - "hybrid": Combination of local and global search (default)

        Returns:
            Dict[str, Any]: Search results containing query, search type,
                result data, and timestamp.

        Raises:
            RuntimeError: If LightRAG instance initialization fails.
            Exception: For any other errors during search operation.
        """
        # Initialize RAG instance
        await self.initialize_lightrag()

        # Ensure instance is initialized
        if self.rag_instance is None:
            raise RuntimeError("Failed to initialize LightRAG instance")

        try:
            logger.info("Searching graph with query: %s, type: %s", query, search_type)

            # Call appropriate query method based on search type
            from lightrag import QueryParam  # pylint: disable=import-outside-toplevel

            param = QueryParam(mode=search_type)
            result = await self.rag_instance.aquery(query, param=param)

            return {
                "query": query,
                "search_type": search_type,
                "result": result,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error("Error searching graph: %s", e)
            raise

    def cleanup(self) -> None:
        """Clean up LightRAG resources using synchronous approach to avoid event loop issues.

        This method performs resource cleanup in a synchronous manner to prevent
        issues with event loop creation during cleanup operations. It safely
        disposes of the LightRAG instance and resets the initialization state.
        """
        if self.rag_instance:
            try:
                # Attempt synchronous cleanup to avoid creating new event loops
                # Set to None directly and let Python garbage collection handle cleanup
                logger.info("LightRAG resources cleaned up (sync mode)")
                self.rag_instance = None
                self._initialized = False
            except Exception as e:
                logger.error("Error during cleanup: %s", e)
                self.rag_instance = None
                self._initialized = False

    def _load_graph_from_graphml(self, graphml_file: str, graph_name: str) -> KnowledgeGraph:
        """Load a knowledge graph from a GraphML file.

        This method parses a GraphML file generated by LightRAG and converts
        it into a KnowledgeGraph object with proper entity and relation mappings.

        Args:
            graphml_file: Path to the GraphML file to load.
            graph_name: Name to assign to the loaded knowledge graph.

        Returns:
            KnowledgeGraph: The loaded knowledge graph with entities and relations.

        Raises:
            Exception: If there are errors parsing the GraphML file or creating entities/relations.
        """
        try:
            # Parse GraphML file
            tree = ET.parse(graphml_file)
            root = tree.getroot()

            # GraphML namespace
            ns = {"graphml": "http://graphml.graphdrawing.org/xmlns"}

            # Create knowledge graph
            graph = KnowledgeGraph(name=graph_name)

            # Parse nodes (entities)
            entities_map: Dict[str, Entity] = {}
            for node in root.findall(".//graphml:node", ns):
                node_id = node.get("id")
                if node_id is None:
                    continue
                entity = self._parse_graphml_node(node, ns)
                if entity:
                    entities_map[node_id] = entity
                    graph.add_entity(entity)

            # Parse edges (relations)
            for edge in root.findall(".//graphml:edge", ns):
                relation = self._parse_graphml_edge(edge, ns, entities_map)
                if relation:
                    graph.add_relation(relation)

            logger.info(
                "Loaded graph from GraphML: %d entities, %d relations",
                len(graph.entities),
                len(graph.relations),
            )
            return graph

        except Exception as e:
            logger.error("Error loading graph from GraphML file %s: %s", graphml_file, e)
            raise

    def _parse_graphml_node(self, node: Any, ns: Dict[str, str]) -> Optional[Entity]:
        """Parse a GraphML node into an Entity object.

        This method extracts entity information from GraphML node elements
        according to LightRAG's GraphML format specifications.

        Args:
            node: XML node element to parse.
            ns: XML namespace dictionary for GraphML parsing.

        Returns:
            Entity: Parsed entity object, or None if parsing fails.
        """
        try:
            node_id = node.get("id")
            entity_data = {}

            # Extract node attributes
            for data in node.findall("graphml:data", ns):
                key = data.get("key")
                value = data.text
                if key and value:
                    # Map attributes according to LightRAG's GraphML format
                    if key == "d0":  # entity_id
                        entity_data["entity_id"] = value
                    elif key == "d1":  # entity_type
                        entity_data["entity_type"] = value
                    elif key == "d2":  # description
                        entity_data["description"] = value
                    elif key == "d3":  # source_id
                        entity_data["source_id"] = value
                    elif key == "d4":  # file_path
                        entity_data["file_path"] = value
                    elif key == "d5":  # created_at
                        entity_data["created_at"] = value

            # Map entity type
            entity_type_str = entity_data.get("entity_type", "unknown").lower()
            entity_type = self._map_entity_type(entity_type_str)

            # Create entity
            entity = Entity(
                id=node_id,
                name=entity_data.get("entity_id", node_id),
                entity_type=entity_type,
                description=entity_data.get("description", ""),
                source="lightrag",
                properties={
                    "source_id": entity_data.get("source_id", ""),
                    "file_path": entity_data.get("file_path", ""),
                    "created_at": entity_data.get("created_at", ""),
                },
            )

            return entity

        except Exception as e:
            logger.error("Error parsing GraphML node: %s", e)
            return None

    def _parse_graphml_edge(self, edge: Any, ns: Dict[str, str], entities_map: Dict[str, Entity]) -> Optional[Relation]:
        """Parse a GraphML edge into a Relation object.

        This method extracts relationship information from GraphML edge elements
        and creates appropriate relation objects linking entities.

        Args:
            edge: XML edge element to parse.
            ns: XML namespace dictionary for GraphML parsing.
            entities_map: Dictionary mapping node IDs to Entity objects.

        Returns:
            Relation: Parsed relation object, or None if parsing fails
                or referenced entities don't exist.
        """
        try:
            source_id = edge.get("source")
            target_id = edge.get("target")

            if source_id not in entities_map or target_id not in entities_map:
                return None

            edge_data = {}
            # Extract edge attributes
            for data in edge.findall("graphml:data", ns):
                key = data.get("key")
                value = data.text
                if key and value:
                    if key == "d6":  # weight
                        edge_data["weight"] = float(value)
                    elif key == "d7":  # description
                        edge_data["description"] = value
                    elif key == "d8":  # keywords
                        edge_data["keywords"] = value
                    elif key == "d9":  # source_id
                        edge_data["source_id"] = value
                    elif key == "d10":  # file_path
                        edge_data["file_path"] = value
                    elif key == "d11":  # created_at
                        edge_data["created_at"] = value

            # Create relation
            relation = Relation(
                head_entity=entities_map[source_id],
                tail_entity=entities_map[target_id],
                relation_type=RelationType.RELATED_TO,  # LightRAG typically doesn't distinguish relation types
                confidence=edge_data.get("weight", 1.0),
                source="lightrag",
                properties={
                    "description": edge_data.get("description", ""),
                    "keywords": edge_data.get("keywords", ""),
                    "source_id": edge_data.get("source_id", ""),
                    "file_path": edge_data.get("file_path", ""),
                    "created_at": edge_data.get("created_at", ""),
                },
            )

            return relation

        except Exception as e:
            logger.error("Error parsing GraphML edge: %s", e)
            return None

    def _map_entity_type(self, entity_type_str: str) -> "EntityType":
        """Map entity type string to EntityType enumeration.

        This method provides a mapping from string representations of entity
        types (as found in GraphML files) to the corresponding EntityType
        enumeration values used by the system.

        Args:
            entity_type_str: String representation of the entity type.

        Returns:
            EntityType: Corresponding EntityType enumeration value.
                Returns EntityType.UNKNOWN for unrecognized types.
        """
        type_mapping = {
            "person": EntityType.PERSON,
            "organization": EntityType.ORGANIZATION,
            "location": EntityType.LOCATION,
            "concept": EntityType.CONCEPT,
            "document": EntityType.DOCUMENT,
            "keyword": EntityType.KEYWORD,
            "table": EntityType.TABLE,
            "column": EntityType.COLUMN,
            "database": EntityType.DATABASE,
        }

        return type_mapping.get(entity_type_str.lower(), EntityType.UNKNOWN)

    def export_to_graphml(self, graph: KnowledgeGraph, output_path: str) -> bool:
        """Export a knowledge graph to GraphML format.

        This method converts a KnowledgeGraph object into a GraphML file
        that is compatible with LightRAG's format specifications.

        Args:
            graph: Knowledge graph to export.
            output_path: File path where the GraphML file should be written.

        Returns:
            bool: True if export was successful, False otherwise.
        """
        try:
            from xml.dom import minidom  # pylint: disable=import-outside-toplevel

            # Create GraphML root element
            root = ET.Element("graphml")
            root.set("xmlns", "http://graphml.graphdrawing.org/xmlns")
            root.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
            root.set(
                "xsi:schemaLocation",
                "http://graphml.graphdrawing.org/xmlns http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd",
            )

            # Define attribute keys
            keys = [
                ("d0", "node", "entity_id", "string"),
                ("d1", "node", "entity_type", "string"),
                ("d2", "node", "description", "string"),
                ("d3", "node", "source_id", "string"),
                ("d4", "node", "file_path", "string"),
                ("d5", "node", "created_at", "long"),
                ("d6", "edge", "weight", "double"),
                ("d7", "edge", "description", "string"),
                ("d8", "edge", "keywords", "string"),
                ("d9", "edge", "source_id", "string"),
                ("d10", "edge", "file_path", "string"),
                ("d11", "edge", "created_at", "long"),
            ]

            for key_id, for_type, attr_name, attr_type in keys:
                key_elem = ET.SubElement(root, "key")
                key_elem.set("id", key_id)
                key_elem.set("for", for_type)
                key_elem.set("attr.name", attr_name)
                key_elem.set("attr.type", attr_type)

            # Create graph element
            graph_elem = ET.SubElement(root, "graph")
            graph_elem.set("edgedefault", "undirected")

            # Add nodes
            for entity in graph.entities.values():
                node_elem = ET.SubElement(graph_elem, "node")
                node_elem.set("id", entity.id)

                # Add node attributes
                data_attrs = [
                    ("d0", entity.name),
                    ("d1", get_type_value(entity.entity_type)),
                    ("d2", entity.description),
                    ("d3", entity.properties.get("source_id", "")),
                    ("d4", entity.properties.get("file_path", "")),
                    ("d5", str(int(entity.created_at.timestamp()))),
                ]

                for key, value in data_attrs:
                    if value:
                        data_elem = ET.SubElement(node_elem, "data")
                        data_elem.set("key", key)
                        data_elem.text = str(value)

            # Add edges
            for relation in graph.relations.values():
                if relation.head_entity and relation.tail_entity:
                    edge_elem = ET.SubElement(graph_elem, "edge")
                    edge_elem.set("source", relation.head_entity.id)
                    edge_elem.set("target", relation.tail_entity.id)

                    # Add edge attributes
                    data_attrs = [
                        ("d6", str(relation.confidence)),
                        ("d7", relation.properties.get("description", "")),
                        ("d8", relation.properties.get("keywords", "")),
                        ("d9", relation.properties.get("source_id", "")),
                        ("d10", relation.properties.get("file_path", "")),
                        ("d11", str(int(relation.created_at.timestamp()))),
                    ]

                    for key, value in data_attrs:
                        if value:
                            data_elem = ET.SubElement(edge_elem, "data")
                            data_elem.set("key", key)
                            data_elem.text = str(value)

            # Write to file
            rough_string = ET.tostring(root, "unicode")
            reparsed = minidom.parseString(rough_string)

            with open(output_path, "w", encoding="utf-8") as f:
                reparsed.writexml(f, indent="  ", addindent="  ", newl="\n", encoding="utf-8")

            logger.info("Graph exported to GraphML: %s", output_path)
            return True

        except Exception as e:
            logger.error("Error exporting graph to GraphML: %s", e)
            return False

    def get_basic_statistics(self) -> Dict[str, Any]:
        """Get basic statistics about the knowledge graph.

        This method provides information about the current state of the
        knowledge graph including entity and relation counts.

        Returns:
            Dict[str, Any]: Dictionary containing graph statistics including:
                - entities_count: Number of entities in the graph
                - relations_count: Number of relations in the graph
                - graphml_file: Path to the GraphML file (if exists)
                - last_modified: Timestamp of last modification
                - status: Current status of the graph
        """
        try:
            graphml_file = self.working_dir / "graph_chunk_entity_relation.graphml"
            if not graphml_file.exists():
                return {"entities_count": 0, "relations_count": 0, "status": "no_graph"}

            tree = ET.parse(str(graphml_file))
            root = tree.getroot()

            ns = {"graphml": "http://graphml.graphdrawing.org/xmlns"}

            entities_count = len(root.findall(".//graphml:node", ns))
            relations_count = len(root.findall(".//graphml:edge", ns))

            return {
                "entities_count": entities_count,
                "relations_count": relations_count,
                "graphml_file": str(graphml_file),
                "last_modified": datetime.fromtimestamp(graphml_file.stat().st_mtime).isoformat(),
                "status": "ready",
            }

        except Exception as e:
            logger.error("Error getting graph statistics: %s", e)
            return {"entities_count": 0, "relations_count": 0, "status": "error", "error": str(e)}


# ============================================================================
# ISP-compliant LightRAG Builder Implementations
# ============================================================================


class MinimalLightRAGBuilder(BasicGraphBuilder):
    """Minimal LightRAG graph builder implementing only core building functionality.

    This class provides a lightweight implementation for basic knowledge graph
    construction using LightRAG. It's designed for clients that only need
    essential graph building capabilities without additional features like
    updates, merging, or other advanced functionality.

    Attributes:
        lightrag_core: Core LightRAG functionality handler.
    """

    def __init__(self, working_dir: str = "minimal_lightrag_storage"):
        """Initialize the minimal LightRAG graph builder.

        Args:
            working_dir: Directory path for LightRAG storage. Defaults to
                "minimal_lightrag_storage".
        """
        self.lightrag_core = LightRAGCore(working_dir)

    async def build_graph(
        self,
        texts: Optional[List[str]] = None,
        database_schema: Optional[Dict[str, Any]] = None,
        graph_name: str = "minimal_lightrag_graph",
    ) -> KnowledgeGraph:
        """Build a knowledge graph asynchronously using LightRAG.

        Args:
            texts: List of text documents to process for entity and relation extraction.
            database_schema: Database schema information (currently not used).
            graph_name: Name identifier for the generated graph.

        Returns:
            KnowledgeGraph: Constructed knowledge graph from input texts.
        """
        return await self.lightrag_core.build_graph_async(texts, database_schema, graph_name)

    def cleanup(self) -> None:
        """Clean up LightRAG resources."""
        self.lightrag_core.cleanup()


class FlexibleLightRAGBuilder(UpdatableGraphBuilder):
    """Flexible LightRAG graph builder supporting building and updating operations.

    This builder extends the basic functionality with update capabilities
    using LightRAG's document insertion mechanism. It's suitable for clients
    that need graph updates but don't require advanced features like
    merging or validation.

    Attributes:
        lightrag_core: Core LightRAG functionality handler.
    """

    def __init__(self, working_dir: str = "flexible_lightrag_storage"):
        """Initialize the flexible LightRAG graph builder.

        Args:
            working_dir: Directory path for LightRAG storage. Defaults to
                "flexible_lightrag_storage".
        """
        self.lightrag_core = LightRAGCore(working_dir)

    async def build_graph(
        self,
        texts: Optional[List[str]] = None,
        database_schema: Optional[Dict[str, Any]] = None,
        graph_name: str = "flexible_lightrag_graph",
    ) -> KnowledgeGraph:
        """Build a knowledge graph asynchronously using LightRAG.

        Args:
            texts: List of text documents to process for entity and relation extraction.
            database_schema: Database schema information (currently not used).
            graph_name: Name identifier for the generated graph.

        Returns:
            KnowledgeGraph: Constructed knowledge graph from input texts.
        """
        return await self.lightrag_core.build_graph_async(texts, database_schema, graph_name)

    async def update_graph(
        self,
        graph: KnowledgeGraph,
        new_entities: Optional[List[Entity]] = None,
        new_relations: Optional[List[Relation]] = None,
    ) -> KnowledgeGraph:
        """Update the existing graph (LightRAG limitation notice).

        Note: LightRAG does not support direct entity/relation updates.
        This method returns the original graph unchanged and logs a warning.

        Args:
            graph: Existing knowledge graph to update.
            new_entities: New entities to add (ignored due to LightRAG limitations).
            new_relations: New relations to add (ignored due to LightRAG limitations).

        Returns:
            KnowledgeGraph: The original graph unchanged.
        """
        logger.warning("LightRAG does not support direct entity/relation updates")
        logger.info("To update the graph, please add new documents using update_graph_with_texts")
        return graph

    async def update_graph_with_texts(
        self,
        texts: List[str],
        graph_name: Optional[str] = None,
    ) -> KnowledgeGraph:
        """Update the graph by adding new text documents.

        This method provides the proper way to update a LightRAG-based
        knowledge graph by processing additional text documents.

        Args:
            texts: List of new text documents to add to the graph.
            graph_name: Optional name for the updated graph.

        Returns:
            KnowledgeGraph: Updated knowledge graph with new documents processed.
        """
        return await self.lightrag_core.update_graph_with_texts_async(texts, graph_name)

    def cleanup(self) -> None:
        """Clean up LightRAG resources."""
        self.lightrag_core.cleanup()


class LightRAGBuilder(
    GraphMergerMixin,
    GraphValidatorMixin,
    GraphExporterMixin,
    GraphStatisticsMixin,
    FullFeaturedGraphBuilder,
):
    """Comprehensive LightRAG graph builder with all available features.

    This is the most feature-complete builder that combines all available
    functionality through multiple mixins. It provides building, updating,
    merging, validation, exporting, and statistics capabilities using LightRAG.

    Only use this builder when you actually need ALL the functionality.
    For most use cases, consider using more focused builders like
    MinimalLightRAGBuilder or FlexibleLightRAGBuilder for better performance.

    Attributes:
        lightrag_core: Core LightRAG functionality handler.
    """

    def __init__(self, working_dir: str = "comprehensive_lightrag_storage"):
        """Initialize the comprehensive LightRAG graph builder.

        Args:
            working_dir: Directory path for LightRAG storage. Defaults to
                "comprehensive_lightrag_storage".
        """
        super().__init__()
        self.lightrag_core = LightRAGCore(working_dir)

    async def build_graph(
        self,
        texts: Optional[List[str]] = None,
        database_schema: Optional[Dict[str, Any]] = None,
        graph_name: str = "comprehensive_lightrag_graph",
    ) -> KnowledgeGraph:
        """Build a comprehensive knowledge graph with full processing capabilities.

        This method leverages LightRAG's functionality to construct a knowledge
        graph and additionally provides validation through the mixin capabilities.

        Args:
            texts: List of text documents to process for entity and relation extraction.
            database_schema: Database schema information (currently not used).
            graph_name: Name identifier for the generated graph.

        Returns:
            KnowledgeGraph: Comprehensive graph with validation performed.
        """
        # Delegate to core builder
        graph = await self.lightrag_core.build_graph_async(texts, database_schema, graph_name)

        # Perform validation
        validation_result = await self.validate_graph(graph)
        if not validation_result.get("valid", True):
            logger.warning(f"Graph validation issues: {validation_result.get('issues', [])}")

        return graph

    async def update_graph(
        self,
        graph: KnowledgeGraph,
        new_entities: Optional[List[Entity]] = None,
        new_relations: Optional[List[Relation]] = None,
    ) -> KnowledgeGraph:
        """Update the graph with validation (LightRAG limitation notice).

        Note: LightRAG does not support direct entity/relation updates.
        This method performs validation on the existing graph.

        Args:
            graph: Existing knowledge graph to update.
            new_entities: New entities to add (ignored due to LightRAG limitations).
            new_relations: New relations to add (ignored due to LightRAG limitations).

        Returns:
            KnowledgeGraph: The original graph with validation performed.
        """
        logger.warning("LightRAG does not support direct entity/relation updates")
        logger.info("To update the graph, please add new documents using update_graph_with_texts")

        # Perform validation
        validation_result = await self.validate_graph(graph)
        if not validation_result.get("valid", True):
            logger.warning(f"Graph validation issues: {validation_result.get('issues', [])}")

        return graph

    async def update_graph_with_texts(
        self,
        texts: List[str],
        graph_name: Optional[str] = None,
    ) -> KnowledgeGraph:
        """Update the graph with new texts and perform validation.

        This method provides the proper way to update a LightRAG-based
        knowledge graph by processing additional text documents, followed
        by validation.

        Args:
            texts: List of new text documents to add to the graph.
            graph_name: Optional name for the updated graph.

        Returns:
            KnowledgeGraph: Updated and validated knowledge graph.
        """
        graph = await self.lightrag_core.update_graph_with_texts_async(texts, graph_name)

        # 执行验证
        validation_result = await self.validate_graph(graph)
        if not validation_result.get("valid", True):
            logger.warning(f"Graph validation issues: {validation_result.get('issues', [])}")

        return graph

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.lightrag_core.get_basic_statistics()

    def cleanup(self) -> None:
        """清理资源"""
        self.lightrag_core.cleanup()


class StreamingLightRAGBuilder(StreamingGraphBuilder, IncrementalBuilderMixin):
    """Streaming LightRAG graph builder for real-time incremental updates.

    This builder is optimized for real-time applications that need to process
    documents as they arrive in a stream. It provides incremental update
    capabilities without requiring merging or validation features that
    might slow down streaming operations.

    Attributes:
        lightrag_core: Core LightRAG functionality handler.
    """

    def __init__(self, working_dir: str = "streaming_lightrag_storage"):
        """Initialize the streaming LightRAG graph builder.

        Args:
            working_dir: Directory path for LightRAG storage. Defaults to
                "streaming_lightrag_storage".
        """
        super().__init__()
        self.lightrag_core = LightRAGCore(working_dir)

    async def build_graph(
        self,
        texts: Optional[List[str]] = None,
        database_schema: Optional[Dict[str, Any]] = None,
        graph_name: str = "streaming_lightrag_graph",
    ) -> KnowledgeGraph:
        """Build initial graph for streaming operations."""
        graph = await self.lightrag_core.build_graph_async(texts, database_schema, graph_name)
        self._current_graph = graph
        return graph

    async def add_documents(self, documents: List[str], document_ids: Optional[List[str]] = None) -> KnowledgeGraph:
        """Add new documents to streaming graph."""
        if not documents:
            return self._current_graph or KnowledgeGraph()

        logger.info(f"Adding {len(documents)} documents to streaming LightRAG graph")

        # 直接使用LightRAG的增量更新功能
        graph = await self.lightrag_core.update_graph_with_texts_async(documents)
        self._current_graph = graph

        # 记录文档（简化版，基于时间戳）
        doc_timestamp = datetime.now().timestamp()
        for i, _ in enumerate(documents):
            doc_id = f"lightrag_doc_{doc_timestamp}_{i}"
            self._document_registry[doc_id] = [f"lightrag_entities_{doc_timestamp}_{i}"]

        return graph

    async def remove_documents(self, document_ids: List[str]) -> KnowledgeGraph:
        """Remove documents (LightRAG does not directly support this, returns current graph)."""
        logger.warning("LightRAG does not support document removal")
        logger.info("Consider rebuilding the graph from scratch if document removal is required")

        # 清理注册表
        for doc_id in document_ids:
            if doc_id in self._document_registry:
                del self._document_registry[doc_id]

        return self._current_graph or KnowledgeGraph()

    def cleanup(self) -> None:
        """清理资源"""
        self.lightrag_core.cleanup()


class BatchLightRAGBuilder(GraphMergerMixin, BatchGraphBuilder):
    """
    批量LightRAG图构建器 - 优化批量处理多个数据源

    适用于需要处理多个数据源并合并它们，但不需要增量更新或验证的场景。
    """

    def __init__(self, working_dir: str = "batch_lightrag_storage"):
        """初始化批量LightRAG图构建器"""
        super().__init__()
        self.lightrag_core = LightRAGCore(working_dir)

    async def build_graph(
        self,
        texts: Optional[List[str]] = None,
        database_schema: Optional[Dict[str, Any]] = None,
        graph_name: str = "batch_lightrag_graph",
    ) -> KnowledgeGraph:
        """构建批量图谱"""
        return await self.lightrag_core.build_graph_async(texts, database_schema, graph_name)

    async def build_from_multiple_sources(
        self, sources: List[Dict[str, Any]], graph_name: str = "multi_source_lightrag_graph"
    ) -> KnowledgeGraph:
        """从多个异构数据源构建图谱"""
        # LightRAG处理多个数据源的策略：顺序处理并合并到同一个工作目录
        all_texts = []

        for source in sources:
            source_type = source.get("type")
            source_data = source.get("data")

            if source_type == "text":
                texts = source_data if isinstance(source_data, list) else [source_data]
                all_texts.extend(texts)
            elif source_type == "mixed":
                if source_data is not None:
                    texts = source_data.get("texts", [])
                    all_texts.extend(texts)
            else:
                logger.warning(f"Unknown source type: {source_type}")

        if not all_texts:
            return KnowledgeGraph(name=graph_name)

        # 使用单个LightRAG实例处理所有文本
        logger.info(f"Building LightRAG graph from {len(all_texts)} texts across {len(sources)} sources")
        graph = await self.lightrag_core.build_graph_async(all_texts, None, graph_name)

        return graph

    def cleanup(self) -> None:
        """清理资源"""
        self.lightrag_core.cleanup()


class LightRAGSearchBuilder(GraphExporter):
    """
    LightRAG搜索构建器 - 专门用于搜索和导出功能

    遵循ISP：只实现搜索和导出相关的接口，不包含构建功能。
    """

    def __init__(self, working_dir: str = "search_lightrag_storage"):
        """初始化搜索构建器"""
        self.lightrag_core = LightRAGCore(working_dir)

    async def search_graph(
        self, query: str, search_type: Literal["naive", "local", "global", "hybrid"] = "hybrid"
    ) -> Dict[str, Any]:
        """搜索图谱"""
        return await self.lightrag_core.asearch_graph(query, search_type)

    async def export_to_format(self, graph: KnowledgeGraph, format: str) -> Dict[str, Any]:
        """导出图谱到指定格式"""
        format_lower = format.lower()

        if format_lower == "graphml":
            # LightRAG原生支持GraphML
            success = self.lightrag_core.export_to_graphml(graph, f"{graph.name}.graphml")
            return {"success": success, "format": "graphml", "message": "Exported to GraphML"}
        elif format_lower == "json":
            return graph.to_dict()
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.lightrag_core.get_basic_statistics()

    def cleanup(self) -> None:
        """清理资源"""
        self.lightrag_core.cleanup()


# 导出所有公共类
__all__ = [
    "LightRAGCore",
    "MinimalLightRAGBuilder",
    "FlexibleLightRAGBuilder",
    "LightRAGBuilder",
    "StreamingLightRAGBuilder",
    "BatchLightRAGBuilder",
    "LightRAGSearchBuilder",
]
