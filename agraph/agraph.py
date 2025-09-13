"""AGraph: Unified knowledge graph construction, vector storage and conversation system.

This module provides a unified AGraph class that integrates:
1. Knowledge graph construction functionality (based on KnowledgeGraphBuilder)
2. Vector storage functionality (based on VectorStore interface)
3. Knowledge base conversation functionality (RAG system)
"""

# pylint: disable=too-many-lines
import asyncio
import json
import os
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union

from .base.core.result import Result
from .base.graphs.optimized import KnowledgeGraph
from .base.infrastructure.dao import MemoryDataAccessLayer

# Import unified architecture components
from .base.managers.factory import create_managers
from .base.managers.interfaces import ClusterManager, EntityManager, RelationManager, TextChunkManager
from .base.models.entities import Entity
from .base.models.relations import Relation
from .base.models.text import TextChunk
from .builder.builder import KnowledgeGraphBuilder
from .chunker import TokenChunker
from .config import BuildSteps, Settings, get_settings
from .logger import logger
from .processor import DocumentProcessorFactory
from .vectordb.factory import VectorStoreFactory, create_chroma_store
from .vectordb.interfaces import VectorStore


class AGraph:
    """Unified knowledge graph system supporting construction, storage and conversation functions.

    Simplified initialization: only requires a Settings instance for all configuration.
    All other parameters are optional overrides for specific use cases.

    Example:
        # Simple initialization with just settings
        settings = get_settings()  # or custom Settings instance
        agraph = AGraph(settings=settings)

        # Or with optional overrides
        agraph = AGraph(settings=settings, collection_name="custom_collection")
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        collection_name: Optional[str] = None,
        persist_directory: Optional[str] = None,
        vector_store_type: Optional[str] = None,
        enable_knowledge_graph: Optional[bool] = None,
        **_kwargs: Any,
    ) -> None:
        """Initialize AGraph system with unified settings configuration.

        Args:
            settings: Settings instance (primary configuration source).
            collection_name: Override collection name (optional).
            persist_directory: Override storage persistence directory (optional).
            vector_store_type: Override vector store type (optional).
            enable_knowledge_graph: Override KG enablement (optional).
            **kwargs: Other parameters (for backward compatibility).
        """
        # Initialize settings with automatic config file handling
        self.settings = self._initialize_settings(settings)

        # Derive configurations from settings with optional overrides
        self.collection_name = collection_name or "agraph_knowledge"
        self.persist_directory = persist_directory or self.settings.workdir
        self.vector_store_type = vector_store_type or "chroma"
        self.enable_knowledge_graph = enable_knowledge_graph if enable_knowledge_graph is not None else True

        # Derive other configurations from settings
        self.use_openai_embeddings = self.settings.embedding.provider == "openai"

        # Initialize configuration from settings
        self.config = self.settings.builder

        # Update builder configuration with AGraph-specific settings
        self.config.cache_dir = os.path.join(self.persist_directory, "cache")

        # Initialize components
        self.vector_store: Optional[VectorStore] = None
        self.builder: Optional[KnowledgeGraphBuilder] = None
        self.knowledge_graph: Optional[KnowledgeGraph] = None
        self._is_initialized = False
        self._background_tasks: List[asyncio.Task] = []

        # Initialize unified architecture components
        self._dao: Optional[MemoryDataAccessLayer] = None
        self._managers: Optional[Dict[str, Any]] = None
        self._use_unified_architecture = True  # Flag to control architecture usage

        logger.info(
            f"AGraph initialization completed, collection: {self.collection_name}, persist_dir: {self.persist_directory}, enable_kg: {self.enable_knowledge_graph}"
        )

    def _initialize_settings(self, settings: Optional[Settings] = None) -> Settings:
        """Initialize settings with automatic config file handling.

        Priority:
        1. If settings parameter is provided, use it and save to its workdir
        2. If config.json exists in workdir, load from it
        3. Use default settings and save to workdir/config.json

        Args:
            settings: Optional settings instance

        Returns:
            Settings instance
        """
        final_settings = None

        if settings is not None:
            logger.info("Using provided settings instance")
            final_settings = settings
        else:
            # Try to load from workdir/config.json
            default_settings = get_settings()
            config_paths_to_try = [
                # 1. Current directory (highest priority when no settings provided)
                os.path.join(os.getcwd(), "config.json"),
                # 2. Default settings workdir
                os.path.join(default_settings.workdir, "config.json"),
            ]

            for config_file_path in config_paths_to_try:
                if os.path.exists(config_file_path):
                    try:
                        logger.info(f"Loading settings from {config_file_path}")
                        with open(config_file_path, "r", encoding="utf-8") as f:
                            config_data = json.load(f)

                        # Create Settings instance from config data
                        final_settings = Settings.from_dict(config_data)
                        logger.info(f"Successfully loaded settings from {config_file_path}")
                        break

                    except Exception as e:
                        logger.warning(f"Failed to load settings from {config_file_path}: {e}")
                        continue

            if final_settings is None:
                logger.info(f"No config file found at: {', '.join(config_paths_to_try)}")
                final_settings = default_settings

        # Always ensure config.json exists in the final settings' workdir
        if final_settings:
            config_file_path = os.path.join(final_settings.workdir, "config.json")
            try:
                # Ensure workdir exists
                os.makedirs(final_settings.workdir, exist_ok=True)

                # Save final settings to config.json (create or update)
                with open(config_file_path, "w", encoding="utf-8") as f:
                    json.dump(final_settings.to_dict(), f, indent=2, ensure_ascii=False)

                logger.info(f"Saved settings to {config_file_path}")

            except Exception as e:
                logger.warning(f"Failed to save settings to {config_file_path}: {e}")

        return final_settings

    async def initialize(self) -> None:
        """Asynchronously initialize all components."""
        if self._is_initialized:
            logger.warning("AGraph is already initialized")
            return

        logger.info("Starting AGraph component initialization...")

        try:
            # 1. Initialize vector store
            await self._initialize_vector_store()

            # 2. Initialize unified architecture (if enabled)
            if self._use_unified_architecture:
                self._initialize_unified_architecture()

            # 3. Initialize knowledge graph builder (if enabled)
            if self.enable_knowledge_graph:
                self._initialize_builder()

            # 4. Try to load existing knowledge graph from disk
            if self.enable_knowledge_graph:
                loaded = self.load_knowledge_graph_from_disk()
                if loaded:
                    logger.info("Existing knowledge graph loaded successfully")
                else:
                    logger.info("No existing knowledge graph found, will create new one when needed")

            self._is_initialized = True
            logger.info("AGraph initialization successful")

        except Exception as e:
            logger.error(f"AGraph initialization failed: {e}")
            raise

    async def _initialize_vector_store(self) -> None:
        """Initialize vector store."""
        try:
            if self.vector_store_type.lower() == "chroma":
                self.vector_store = create_chroma_store(
                    collection_name=self.collection_name,
                    persist_directory=self.persist_directory + "/chroma",
                    use_openai_embeddings=self.use_openai_embeddings,
                )
            else:
                self.vector_store = VectorStoreFactory.create_store(
                    store_type=self.vector_store_type,
                    collection_name=self.collection_name,
                    use_openai_embeddings=self.use_openai_embeddings,
                )

            await self.vector_store.initialize()
            logger.info(f"Vector store ({self.vector_store_type}) initialization successful")

        except Exception as e:
            logger.error(f"Vector store initialization failed: {e}")
            raise

    def _initialize_unified_architecture(self) -> None:
        """Initialize unified architecture components."""
        try:
            # Initialize DAO
            self._dao = MemoryDataAccessLayer()

            # Create managers using optimized factory for better performance
            manager_type = "optimized" if self.enable_knowledge_graph else "default"
            self._managers = create_managers(manager_type, dao=self._dao)

            logger.info(f"Unified architecture initialization successful (type: {manager_type})")
            logger.info(f"Available managers: {list(self._managers.keys())}")

        except Exception as e:
            logger.error(f"Unified architecture initialization failed: {e}")
            raise

    def _initialize_builder(self) -> None:
        """Initialize knowledge graph builder."""
        try:
            self.builder = KnowledgeGraphBuilder(config=self.config, enable_knowledge_graph=self.enable_knowledge_graph)
            logger.info("Knowledge graph builder initialization successful")
        except Exception as e:
            logger.error(f"Knowledge graph builder initialization failed: {e}")
            raise

    # =============== Unified Architecture Access Methods ===============

    @property
    def entity_manager(self) -> Optional[EntityManager]:
        """Get the entity manager from unified architecture."""
        if self._managers:
            return self._managers.get("entity_manager")
        return None

    @property
    def relation_manager(self) -> Optional[RelationManager]:
        """Get the relation manager from unified architecture."""
        if self._managers:
            return self._managers.get("relation_manager")
        return None

    @property
    def cluster_manager(self) -> Optional[ClusterManager]:
        """Get the cluster manager from unified architecture."""
        if self._managers:
            return self._managers.get("cluster_manager")
        return None

    @property
    def text_chunk_manager(self) -> Optional[TextChunkManager]:
        """Get the text chunk manager from unified architecture."""
        if self._managers:
            return self._managers.get("text_chunk_manager")
        return None

    def get_unified_stats(self) -> Optional[Result[Dict[str, Any]]]:
        """Get comprehensive statistics from all unified managers."""
        if not self._managers:
            return None

        try:
            stats = {}

            # Get entity statistics
            if self.entity_manager:
                entity_stats = self.entity_manager.get_statistics()
                if entity_stats.is_ok():
                    stats["entities"] = entity_stats.data

            # Get relation statistics
            if self.relation_manager:
                relation_stats = self.relation_manager.get_statistics()
                if relation_stats.is_ok():
                    stats["relations"] = relation_stats.data

            # Get cluster statistics
            if self.cluster_manager:
                cluster_stats = self.cluster_manager.get_statistics()
                if cluster_stats.is_ok():
                    stats["clusters"] = cluster_stats.data

            # Get text chunk statistics
            if self.text_chunk_manager:
                chunk_stats = self.text_chunk_manager.get_statistics()
                if chunk_stats.is_ok():
                    stats["text_chunks"] = chunk_stats.data

            return Result.ok(stats)

        except Exception as e:
            return Result.internal_error(e)

    # =============== Knowledge Graph Construction Functions ===============

    async def build_from_documents(
        self,
        documents: Union[List[Union[str, Path]], str, Path],
        graph_name: str = "Knowledge Graph",
        graph_description: str = "Built by AGraph",
        use_cache: bool = True,
        save_to_vector_store: bool = True,
    ) -> KnowledgeGraph:
        """Build knowledge graph from documents.

        Args:
            documents: Document path list or single document path.
            graph_name: Graph name.
            graph_description: Graph description.
            use_cache: Whether to use cache.
            save_to_vector_store: Whether to save to vector store.

        Returns:
            Built knowledge graph.
        """
        if not self._is_initialized:
            raise RuntimeError("AGraph not initialized, please call initialize() first")

        if not self.enable_knowledge_graph:
            # When KG is disabled, process documents for text chunks only
            return await self._build_text_chunks_from_documents(
                documents, graph_name, graph_description, use_cache, save_to_vector_store
            )

        if not self.builder:
            raise RuntimeError("Knowledge graph builder not initialized")

        # Process document path parameters
        if isinstance(documents, (str, Path)):
            documents = [Path(documents)]
        elif isinstance(documents, list):
            documents = [Path(doc) if isinstance(doc, str) else doc for doc in documents]

        # Convert to Union[str, Path] list
        documents_list: List[Union[str, Path]] = [str(doc) for doc in documents]

        logger.info(f"Starting to build knowledge graph from {len(documents_list)} documents: {graph_name}")

        try:
            # Use builder to construct knowledge graph
            self.knowledge_graph = await self.builder.build_from_documents(
                documents=documents_list,
                graph_name=graph_name,
                graph_description=graph_description,
                use_cache=use_cache,
            )

            # Asynchronously save to vector store
            # Save to disk for persistence
            if self.knowledge_graph:
                self.save_knowledge_graph_to_disk()

            if save_to_vector_store and self.knowledge_graph:
                asyncio.create_task(self._save_to_vector_store())

            if self.knowledge_graph:
                logger.info(
                    f"Knowledge graph construction completed: {len(self.knowledge_graph.entities)} entities, "
                    f"{len(self.knowledge_graph.relations)} relations, "
                    f"{len(self.knowledge_graph.text_chunks)} text chunks"
                )

                return self.knowledge_graph
            raise RuntimeError("Knowledge graph construction failed, returned None")

        except Exception as e:
            logger.error(f"Knowledge graph construction failed: {e}")
            raise

    async def build_from_texts(
        self,
        texts: List[str],
        graph_name: str = "Knowledge Graph",
        graph_description: str = "Built by AGraph from texts",
        use_cache: bool = True,
        save_to_vector_store: bool = True,
    ) -> KnowledgeGraph:
        """Build knowledge graph from text list.

        Args:
            texts: Text list.
            graph_name: Graph name.
            graph_description: Graph description.
            use_cache: Whether to use cache.
            save_to_vector_store: Whether to save to vector store.

        Returns:
            Built knowledge graph.
        """
        if not self._is_initialized:
            raise RuntimeError("AGraph not initialized, please call initialize() first")

        if not self.enable_knowledge_graph:
            # When KG is disabled, process texts for text chunks only
            return await self._build_text_chunks_from_texts(
                texts, graph_name, graph_description, use_cache, save_to_vector_store
            )

        if not self.builder:
            raise RuntimeError("Knowledge graph builder not initialized")

        logger.info(f"Starting to build knowledge graph from {len(texts)} texts: {graph_name}")

        try:
            # Use builder's build_from_text method (accepts text list)
            self.knowledge_graph = await self.builder.build_from_text(
                texts=texts,
                graph_name=graph_name,
                graph_description=graph_description,
                use_cache=use_cache,
            )

            # Save to disk for persistence
            if self.knowledge_graph:
                self.save_knowledge_graph_to_disk()

            # Check if new content was built (not all from cache) before saving to vector store
            should_save_to_vector_store = self._should_save_to_vector_store(save_to_vector_store, use_cache)

            # Asynchronously save to vector store only if needed
            if should_save_to_vector_store:
                task = asyncio.create_task(self._save_to_vector_store())
                self._background_tasks.append(task)

            if self.knowledge_graph:
                logger.info(
                    f"Knowledge graph construction completed: {len(self.knowledge_graph.entities)} entities, "
                    f"{len(self.knowledge_graph.relations)} relations, "
                    f"{len(self.knowledge_graph.text_chunks)} text chunks"
                )

                return self.knowledge_graph
            raise RuntimeError("Knowledge graph construction failed")

        except Exception as e:
            logger.error(f"Building knowledge graph from texts failed: {e}")
            raise

    # =============== Text-Only Processing Functions (KG Disabled) ===============

    async def _build_text_chunks_from_documents(
        self,
        documents: Union[List[Union[str, Path]], str, Path],
        graph_name: str,
        graph_description: str,
        use_cache: bool,
        save_to_vector_store: bool,
    ) -> KnowledgeGraph:
        """Build knowledge graph with text chunks only from documents (no entities/relations/clusters)."""
        # Process document path parameters
        if isinstance(documents, (str, Path)):
            documents = [Path(documents)]
        elif isinstance(documents, list):
            documents = [Path(doc) if isinstance(doc, str) else doc for doc in documents]

        # Convert to Union[str, Path] list
        documents_list: List[Union[str, Path]] = [str(doc) for doc in documents]

        logger.info(f"Starting text-only processing from {len(documents_list)} documents: {graph_name}")

        try:
            # Initialize components we need
            processor_factory = DocumentProcessorFactory()
            chunker = TokenChunker(chunk_size=self.config.chunk_size, chunk_overlap=self.config.chunk_overlap)

            # Step 1: Process documents
            logger.info(f"Step 1: Processing {len(documents_list)} documents")
            texts = []
            for doc_path in documents_list:
                try:
                    processor = processor_factory.get_processor(doc_path)
                    text = processor.process(doc_path)
                    texts.append(text)
                except Exception as e:
                    logger.warning(f"Failed to process document {doc_path}: {e}")

            logger.info(f"Document processing completed - extracted {len(texts)} texts")

            # Step 2: Chunk texts
            logger.info(f"Step 2: Chunking {len(texts)} texts")
            chunks = []
            for i, text in enumerate(texts):
                chunks_with_positions = chunker.split_text_with_positions(text)
                for j, (chunk_text, start_idx, end_idx) in enumerate(chunks_with_positions):

                    chunk = TextChunk(
                        id=f"chunk_{i}_{j}",
                        content=chunk_text,
                        title=f"Document {i} Chunk {j}",
                        start_index=start_idx,
                        end_index=end_idx,
                        source=Path(documents_list[i]).name,
                    )
                    chunks.append(chunk)

            logger.info(f"Text chunking completed - created {len(chunks)} chunks")

            # Step 3: Create knowledge graph with only text chunks
            kg = KnowledgeGraph(
                name=graph_name,
                description=graph_description,
                entities={},  # Empty
                relations={},  # Empty
                clusters={},  # Empty
                text_chunks={chunk.id: chunk for chunk in chunks},
            )

            # Step 4: Save to vector store if requested
            if save_to_vector_store and chunks:
                self.knowledge_graph = kg  # Set temporarily for saving
                asyncio.create_task(self._save_to_vector_store())

            logger.info(f"Text-only processing completed: {len(chunks)} text chunks")
            return kg

        except Exception as e:
            logger.error(f"Text-only processing failed: {e}")
            raise

    async def _build_text_chunks_from_texts(
        self,
        texts: List[str],
        graph_name: str,
        graph_description: str,
        use_cache: bool,
        save_to_vector_store: bool,
    ) -> KnowledgeGraph:
        """Build knowledge graph with text chunks only from texts (no entities/relations/clusters)."""
        logger.info(f"Starting text-only processing from {len(texts)} texts: {graph_name}")

        try:
            # Initialize chunker
            chunker = TokenChunker(chunk_size=self.config.chunk_size, chunk_overlap=self.config.chunk_overlap)

            # Step 1: Chunk texts
            logger.info(f"Step 1: Chunking {len(texts)} texts")
            chunks = []
            for i, text in enumerate(texts):
                chunks_with_positions = chunker.split_text_with_positions(text)
                for j, (chunk_text, start_idx, end_idx) in enumerate(chunks_with_positions):
                    chunk = TextChunk(
                        id=f"chunk_{i}_{j}",
                        content=chunk_text,
                        title=f"Text {i} Chunk {j}",
                        start_index=start_idx,
                        end_index=end_idx,
                        source=f"text_{i}",
                    )
                    chunks.append(chunk)

            logger.info(f"Text chunking completed - created {len(chunks)} chunks")

            # Step 2: Create knowledge graph with only text chunks
            kg = KnowledgeGraph(
                name=graph_name,
                description=graph_description,
                entities={},  # Empty
                relations={},  # Empty
                clusters={},  # Empty
                text_chunks={chunk.id: chunk for chunk in chunks},
            )

            # Step 3: Save to vector store if requested
            if save_to_vector_store and chunks:
                self.knowledge_graph = kg  # Set temporarily for saving
                asyncio.create_task(self._save_to_vector_store())

            logger.info(f"Text-only processing completed: {len(chunks)} text chunks")
            return kg

        except Exception as e:
            logger.error(f"Text-only processing failed: {e}")
            raise

    def _should_save_to_vector_store(self, save_to_vector_store: bool, use_cache: bool) -> bool:
        """Determine if we should save to vector store based on cache usage analysis.

        Args:
            save_to_vector_store: Initial intention to save to vector store
            use_cache: Whether cache was used during building

        Returns:
            True if should save to vector store, False otherwise
        """
        if not save_to_vector_store:
            return False

        if not use_cache:
            return True

        # Check if builder has cache usage information
        if (
            self.builder is None
            or not hasattr(self.builder, "last_build_context")
            or not self.builder.last_build_context
        ):
            logger.debug("No build context available for cache analysis")
            return True

        context = self.builder.last_build_context

        # Check actual cache usage based on execution times
        # If major steps completed very quickly (< 1s), likely used cache
        new_content_built = False
        major_steps = [
            BuildSteps.ENTITY_EXTRACTION,
            BuildSteps.RELATION_EXTRACTION,
            BuildSteps.CLUSTER_FORMATION,
        ]

        for step in major_steps:
            if context.is_step_completed(step):
                execution_time = context.step_execution_times.get(step, float("inf"))
                # If execution time > 1 second, likely built new content
                # If execution time < 1 second, likely used cache
                if execution_time > 1.0:
                    new_content_built = True
                    logger.debug(f"Step {step} took {execution_time:.2f}s, indicating new content processing")
                else:
                    logger.debug(f"Step {step} took {execution_time:.2f}s, indicating cache usage")

        if not new_content_built:
            logger.info("All major components loaded from cache, skipping vector store save")
            return False
        logger.info("New content detected, will save to vector store")
        return True

    # =============== Vector Storage Functions ===============

    async def _save_to_vector_store(self) -> None:
        """Save knowledge graph to vector store."""
        if not self.vector_store or not self.knowledge_graph:
            logger.warning("Vector store or knowledge graph not initialized, skipping save")
            return

        try:
            logger.info("Starting to save knowledge graph to vector store...")

            # Batch save entities (if knowledge graph is enabled)
            if self.enable_knowledge_graph and self.knowledge_graph.entities:
                await self.vector_store.batch_add_entities(list(self.knowledge_graph.entities.values()))
                logger.info(f"Saved {len(self.knowledge_graph.entities)} entities")

            # Batch save relations (if knowledge graph is enabled)
            if self.enable_knowledge_graph and self.knowledge_graph.relations:
                await self.vector_store.batch_add_relations(list(self.knowledge_graph.relations.values()))
                logger.info(f"Saved {len(self.knowledge_graph.relations)} relations")

            # Batch save clusters (if knowledge graph is enabled)
            if self.enable_knowledge_graph and self.knowledge_graph.clusters:
                await self.vector_store.batch_add_clusters(list(self.knowledge_graph.clusters.values()))
                logger.info(f"Saved {len(self.knowledge_graph.clusters)} clusters")

            # Batch save text chunks
            if self.knowledge_graph.text_chunks:
                await self.vector_store.batch_add_text_chunks(list(self.knowledge_graph.text_chunks.values()))
                logger.info(f"Saved {len(self.knowledge_graph.text_chunks)} text chunks")

            logger.info("Knowledge graph saved to vector store completed")

        except Exception as e:
            logger.error(f"Saving to vector store failed: {e}")
            raise
        finally:
            # Clean up completed tasks
            self._cleanup_completed_tasks()

    def _cleanup_completed_tasks(self) -> None:
        """Clean up completed background tasks."""
        self._background_tasks = [task for task in self._background_tasks if not task.done()]

    async def save_knowledge_graph(self) -> None:
        """Explicitly save knowledge graph to vector store."""
        await self._save_to_vector_store()

    def save_knowledge_graph_to_disk(self) -> None:
        """Save knowledge graph to disk for persistence across restarts."""
        if not self.knowledge_graph:
            logger.warning("No knowledge graph to save")
            return

        try:
            # Create knowledge graph storage directory
            kg_storage_dir = os.path.join(self.persist_directory, "knowledge_graphs")
            os.makedirs(kg_storage_dir, exist_ok=True)

            # Save knowledge graph as JSON
            kg_file_path = os.path.join(kg_storage_dir, f"{self.collection_name}_kg.json")
            kg_data = self.knowledge_graph.to_dict()

            with open(kg_file_path, "w", encoding="utf-8") as f:
                json.dump(kg_data, f, indent=2, ensure_ascii=False, default=str)

            logger.info(f"Knowledge graph saved to {kg_file_path}")

        except Exception as e:
            logger.error(f"Failed to save knowledge graph to disk: {e}")

    def delete_chroma_files(self) -> bool:
        """Delete ChromaDB persistent files from disk.

        Returns:
            True if files were deleted or didn't exist, False if deletion failed.
        """
        try:
            import os
            import shutil

            # ChromaDB persistent directory path
            chroma_dir = os.path.join(self.persist_directory, "chroma")

            if os.path.exists(chroma_dir):
                # Delete all contents inside the chroma directory, but keep the directory itself
                for filename in os.listdir(chroma_dir):
                    file_path = os.path.join(chroma_dir, filename)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    except Exception as e:
                        logger.warning(f"Failed to delete {file_path}: {e}")

                logger.info(f"ChromaDB persistent files content deleted from: {chroma_dir}")
            else:
                logger.info(f"No ChromaDB persistent directory found: {chroma_dir}")

            return True
        except Exception as e:
            logger.error(f"Failed to delete ChromaDB files: {e}")
            return False

    def delete_knowledge_graph_from_disk(self) -> bool:
        """Delete knowledge graph file from disk.

        Returns:
            True if file was deleted or didn't exist, False if deletion failed.
        """
        try:
            # Get knowledge graph file path
            kg_storage_dir = os.path.join(self.persist_directory, "knowledge_graphs")
            kg_file_path = os.path.join(kg_storage_dir, f"{self.collection_name}_kg.json")

            if os.path.exists(kg_file_path):
                os.remove(kg_file_path)
                logger.info(f"Knowledge graph file deleted: {kg_file_path}")
            else:
                logger.info(f"No knowledge graph file found to delete: {kg_file_path}")

            # Also try to remove the storage directory if it's empty
            try:
                if os.path.exists(kg_storage_dir) and not os.listdir(kg_storage_dir):
                    os.rmdir(kg_storage_dir)
                    logger.info(f"Empty knowledge graph storage directory removed: {kg_storage_dir}")
            except OSError:
                # Directory not empty or other issues, ignore
                pass

            return True
        except Exception as e:
            logger.error(f"Failed to delete knowledge graph from disk: {e}")
            return False

    def load_knowledge_graph_from_disk(self) -> bool:
        """Load knowledge graph from disk if it exists.

        Returns:
            True if knowledge graph was loaded successfully, False otherwise.
        """
        try:
            # Check if knowledge graph file exists
            kg_storage_dir = os.path.join(self.persist_directory, "knowledge_graphs")
            kg_file_path = os.path.join(kg_storage_dir, f"{self.collection_name}_kg.json")

            if not os.path.exists(kg_file_path):
                logger.info(f"No saved knowledge graph found at {kg_file_path}")
                return False

            # Load knowledge graph from JSON
            with open(kg_file_path, "r", encoding="utf-8") as f:
                kg_data = json.load(f)

            self.knowledge_graph = KnowledgeGraph.from_dict(kg_data)
            logger.info(f"Knowledge graph loaded from {kg_file_path}")
            logger.info(
                f"Loaded: {len(self.knowledge_graph.entities)} entities, "
                f"{len(self.knowledge_graph.relations)} relations, "
                f"{len(self.knowledge_graph.clusters)} clusters, "
                f"{len(self.knowledge_graph.text_chunks)} text chunks"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to load knowledge graph from disk: {e}")
            return False

    # =============== Search and Retrieval Functions ===============

    async def search_entities(
        self, query: str, top_k: int = 10, filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Entity, float]]:
        """Search entities.

        Args:
            query: Query text.
            top_k: Number of results to return.
            filter_dict: Filter conditions.

        Returns:
            Entity list and similarity scores.
        """
        if not self.vector_store:
            raise RuntimeError("Vector store not initialized")

        return await self.vector_store.search_entities(query, top_k, filter_dict)

    async def search_relations(
        self, query: str, top_k: int = 10, filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Relation, float]]:
        """Search relations.

        Args:
            query: Query text.
            top_k: Number of results to return.
            filter_dict: Filter conditions.

        Returns:
            Relation list and similarity scores.
        """
        if not self.vector_store:
            raise RuntimeError("Vector store not initialized")

        return await self.vector_store.search_relations(query, top_k, filter_dict)

    async def search_text_chunks(
        self, query: str, top_k: int = 10, filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[TextChunk, float]]:
        """Search text chunks.

        Args:
            query: Query text.
            top_k: Number of results to return.
            filter_dict: Filter conditions.

        Returns:
            Text chunk list and similarity scores.
        """
        if not self.vector_store:
            raise RuntimeError("Vector store not initialized")

        return await self.vector_store.search_text_chunks(query, top_k, filter_dict)

    # =============== Knowledge Base Conversation Functions ===============

    async def chat(
        self,
        question: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        entity_top_k: int = 5,
        relation_top_k: int = 5,
        text_chunk_top_k: int = 5,
        response_type: str = "详细回答",
        stream: bool = False,
    ) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        """Knowledge base conversation functionality.

        Args:
            question: User question.
            conversation_history: Conversation history.
            entity_top_k: Number of entities to retrieve.
            relation_top_k: Number of relations to retrieve.
            text_chunk_top_k: Number of text chunks to retrieve.
            response_type: Response type.
            stream: Whether to return stream.

        Returns:
            If stream=False, returns a dict containing answer and context info.
            If stream=True, returns async generator, each yield contains chunk and partial_answer dict.
        """
        if not self.vector_store:
            raise RuntimeError("Vector store not initialized")

        logger.info(f"Starting to process user question: {question}")

        try:
            # 1. Retrieve relevant information
            context_info = await self._retrieve_context(question, entity_top_k, relation_top_k, text_chunk_top_k)
            logger.info(f"Retrieved context information: {context_info}")

            # 2. Build prompt
            prompt = self._build_chat_prompt(question, context_info, conversation_history, response_type)
            logger.info(f"Model: {self.settings.llm}, Built prompt: {prompt[:100]}...")  # Log first 100 chars of prompt

            # 3. Call LLM to generate answer
            if stream:
                # Stream answer - directly return async generator
                return self._generate_stream_response(prompt, question, context_info)

            # Non-stream answer
            response = await self._generate_response(prompt)
            return {
                "question": question,
                "answer": response,
                "context": context_info,
                "prompt": prompt,
            }

        except Exception as e:
            logger.error(f"Conversation processing failed: {e}")
            raise

    async def _retrieve_context(
        self, query: str, entity_top_k: int, relation_top_k: int, text_chunk_top_k: int
    ) -> Dict[str, Any]:
        """Retrieve relevant context information."""
        context: Dict[str, Any] = {"entities": [], "relations": [], "text_chunks": []}

        try:
            # Concurrent retrieval of different types of information
            tasks = []

            # Only add entity and relation search tasks if knowledge graph is enabled
            if self.enable_knowledge_graph:
                tasks.extend(
                    [
                        self.search_entities(query, entity_top_k),
                        self.search_relations(query, relation_top_k),
                    ]
                )

            # Always add text chunks search
            tasks.append(self.search_text_chunks(query, text_chunk_top_k))

            results = await asyncio.gather(*tasks, return_exceptions=True)

            result_index = 0

            # Process entity results (if knowledge graph is enabled)
            if self.enable_knowledge_graph:
                if result_index < len(results) and isinstance(results[result_index], list):
                    entity_results = results[result_index]
                    if isinstance(entity_results, list):  # Type guard for mypy
                        context["entities"] = [{"entity": entity, "score": score} for entity, score in entity_results]
                result_index += 1

                # Process relation results (if knowledge graph is enabled)
                if result_index < len(results) and isinstance(results[result_index], list):
                    relation_results = results[result_index]
                    if isinstance(relation_results, list):  # Type guard for mypy
                        context["relations"] = [
                            {"relation": relation, "score": score} for relation, score in relation_results
                        ]
                result_index += 1

            # Process text chunk results (always present)
            if result_index < len(results) and isinstance(results[result_index], list):
                chunk_results = results[result_index]
                if isinstance(chunk_results, list):  # Type guard for mypy
                    context["text_chunks"] = [{"text_chunk": chunk, "score": score} for chunk, score in chunk_results]

        except Exception as e:
            logger.error(f"Context retrieval failed: {e}")

        return context

    def _build_chat_prompt(
        self,
        question: str,
        context_info: Dict[str, Any],
        conversation_history: Optional[List[Dict[str, str]]],
        response_type: str,
    ) -> str:
        """Build conversation prompt."""
        # Build conversation history
        history_text = ""
        if conversation_history:
            history_parts = []
            for turn in conversation_history[-5:]:  # Only take the last 5 conversations
                if turn.get("user"):
                    history_parts.append(f"用户: {turn['user']}")
                if turn.get("assistant"):
                    history_parts.append(f"助手: {turn['assistant']}")
            history_text = "\n".join(history_parts)

        # Build knowledge graph context
        kg_context_parts = []

        # Add entity information (if knowledge graph is enabled)
        if self.enable_knowledge_graph and context_info.get("entities"):
            kg_context_parts.append("相关实体:")
            for item in context_info["entities"][:3]:
                entity = item["entity"]
                kg_context_parts.append(f"- {entity.name} ({entity.entity_type}): {entity.description or 'N/A'}")

        # Add relation information (if knowledge graph is enabled)
        if self.enable_knowledge_graph and context_info.get("relations"):
            kg_context_parts.append("\n相关关系:")
            for item in context_info["relations"][:3]:
                relation = item["relation"]
                head_name = relation.head_entity.name if relation.head_entity else "未知"
                tail_name = relation.tail_entity.name if relation.tail_entity else "未知"
                kg_context_parts.append(f"- {head_name} --[{relation.relation_type}]--> {tail_name}")

        # Add text chunk information
        if context_info.get("text_chunks"):
            kg_context_parts.append("\n相关文档内容:")
            for item in context_info["text_chunks"][:3]:
                chunk = item["text_chunk"]
                content_preview = chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content
                kg_context_parts.append(f"- {content_preview}")

        kg_context = "\n".join(kg_context_parts)

        # Use system prompt template from configuration
        prompt = self.settings.rag.system_prompt.format(
            history=history_text, kg_context=kg_context, response_type=response_type
        )

        # Add user question
        prompt += f"\n\n---用户问题---\n{question}\n\n请根据上述数据源回答问题："

        return prompt

    async def _generate_stream_response(
        self, prompt: str, question: str, context_info: Dict[str, Any]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Generate streaming response."""
        try:
            # Use OpenAI compatible API call
            import openai  # pylint: disable=import-outside-toplevel

            client = openai.AsyncOpenAI(api_key=self.settings.openai.api_key, base_url=self.settings.openai.api_base)

            stream = await client.chat.completions.create(
                model=self.settings.llm.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.settings.llm.temperature,
                max_tokens=4096,
                stream=True,
            )

            answer_chunks = []
            async for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    answer_chunks.append(content)
                    yield {
                        "question": question,
                        "chunk": content,
                        "partial_answer": "".join(answer_chunks),
                        "context": context_info,
                        "finished": False,
                    }

            # Send completion signal
            yield {
                "question": question,
                "chunk": "",
                "partial_answer": "".join(answer_chunks),
                "answer": "".join(answer_chunks),
                "context": context_info,
                "finished": True,
            }

        except ImportError:
            logger.warning("OpenAI package not installed, using mock streaming response")
            async for item in self._mock_stream_response(question, context_info):
                yield item
        except Exception as e:
            logger.error(f"Streaming LLM call failed: {e}")
            async for item in self._mock_stream_response(question, context_info):
                yield item

    async def _generate_response(self, prompt: str) -> str:
        """Generate response using LLM."""
        try:
            # Use OpenAI compatible API call
            import openai  # pylint: disable=import-outside-toplevel

            client = openai.AsyncOpenAI(api_key=self.settings.openai.api_key, base_url=self.settings.openai.api_base)

            response = await client.chat.completions.create(
                model=self.settings.llm.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.settings.llm.temperature,
                max_tokens=4096,
            )

            content = response.choices[0].message.content
            return content.strip() if content else ""

        except ImportError:
            logger.warning("OpenAI package not installed, using mock response")
            return self._mock_response(prompt)
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return self._mock_response(prompt)

    def _mock_response(self, _prompt: str) -> str:
        """Mock LLM response for testing."""
        return (
            "Based on the provided knowledge graph information, I understand your question. "
            "Due to the current use of mock mode, I cannot provide a specific answer. "
            "Please configure the correct LLM API for full functionality."
        )

    async def _mock_stream_response(
        self, question: str, context_info: Dict[str, Any]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Mock streaming LLM response for testing."""
        mock_response = (
            "Based on the provided knowledge graph information, I understand your question. "
            "Due to the current use of mock mode, I cannot provide a specific answer. "
            "Please configure the correct LLM API for full functionality."
        )

        # Simulate character-by-character output
        answer_chunks = []
        for char in mock_response:
            answer_chunks.append(char)
            await asyncio.sleep(0.01)  # Simulate network latency
            yield {
                "question": question,
                "chunk": char,
                "partial_answer": "".join(answer_chunks),
                "context": context_info,
                "finished": False,
            }

        # Send completion signal
        yield {
            "question": question,
            "chunk": "",
            "partial_answer": mock_response,
            "answer": mock_response,
            "context": context_info,
            "finished": True,
        }

    # =============== Management Functions ===============

    async def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        stats: Dict[str, Any] = {}

        # Vector store statistics
        if self.vector_store:
            try:
                vector_stats = await self.vector_store.get_stats()
                stats["vector_store"] = vector_stats
            except Exception as e:
                logger.warning(f"Getting vector store statistics failed: {e}")
                stats["vector_store"] = {"error": str(e)}

        # Knowledge graph statistics (if enabled)
        if self.enable_knowledge_graph and self.knowledge_graph:
            stats["knowledge_graph"] = {
                "entities": len(self.knowledge_graph.entities),
                "relations": len(self.knowledge_graph.relations),
                "clusters": len(self.knowledge_graph.clusters),
                "text_chunks": len(self.knowledge_graph.text_chunks),
            }

        # Builder statistics
        if self.builder:
            try:
                build_status = self.builder.get_build_status()
                cache_info = self.builder.get_cache_info()
                stats["builder"] = {"build_status": build_status, "cache_info": cache_info}
            except Exception as e:
                logger.warning(f"Getting builder statistics failed: {e}")
                stats["builder"] = {"error": str(e)}

        return stats

    async def clear_all(self) -> bool:
        """Clear all data including vector store, cache, and disk files."""
        try:
            # Clear vector store
            if self.vector_store:
                await self.vector_store.clear_all()

            # Clear builder cache
            if self.builder and hasattr(self.builder, "clear_cache"):
                self.builder.clear_cache()

            # Delete knowledge graph file from disk
            self.delete_knowledge_graph_from_disk()

            # Delete ChromaDB persistent files from disk
            self.delete_chroma_files()

            # Close current vector store to release file handles
            if self.vector_store:
                await self.vector_store.close()
                self.vector_store = None

            # Re-initialize vector store to ensure clean state
            await self._initialize_vector_store()

            # Clear in-memory knowledge graph
            self.knowledge_graph = None

            logger.info(
                "All data cleared including disk files and ChromaDB persistent files, vector store reinitialized"
            )
            return True

        except Exception as e:
            logger.error(f"Clearing data failed: {e}")
            return False

    async def close(self) -> None:
        """Close AGraph system."""
        try:
            # Wait for all background tasks to complete
            if self._background_tasks:
                logger.info(f"Waiting for {len(self._background_tasks)} background tasks to complete...")
                await asyncio.gather(*self._background_tasks, return_exceptions=True)
                self._background_tasks.clear()
                logger.info("All background tasks completed")

            # Close builder resources
            if self.builder and hasattr(self.builder, "aclose"):
                await self.builder.aclose()

            if self.vector_store:
                await self.vector_store.close()

            logger.info("AGraph system closed")

        except Exception as e:
            logger.error(f"Closing system failed: {e}")

    # =============== Context Manager Support ===============

    async def __aenter__(self) -> "AGraph":
        """Async context manager entry."""
        if not self._is_initialized:
            await self.initialize()
        return self

    async def __aexit__(self, _exc_type: Any, _exc_val: Any, _exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()

    # =============== Properties and State Checking ===============

    @property
    def is_initialized(self) -> bool:
        """Check if initialized."""
        return self._is_initialized

    @property
    def has_knowledge_graph(self) -> bool:
        """Check if has knowledge graph."""
        return self.knowledge_graph is not None

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"AGraph(collection='{self.collection_name}', "
            f"store_type='{self.vector_store_type}', "
            f"initialized={self.is_initialized}, "
            f"enable_kg={self.enable_knowledge_graph}, "
            f"has_kg={self.has_knowledge_graph})"
        )
