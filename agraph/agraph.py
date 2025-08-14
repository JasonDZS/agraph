"""AGraph: Unified knowledge graph construction, vector storage and conversation system.

This module provides a unified AGraph class that integrates:
1. Knowledge graph construction functionality (based on KnowledgeGraphBuilder)
2. Vector storage functionality (based on VectorStore interface)
3. Knowledge base conversation functionality (RAG system)
"""

import asyncio
import os
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union

from .base.entities import Entity
from .base.graph import KnowledgeGraph
from .base.relations import Relation
from .base.text import TextChunk
from .builder.builder import KnowledgeGraphBuilder
from .config import BuilderConfig, get_settings
from .logger import logger
from .vectordb.factory import VectorStoreFactory, create_chroma_store
from .vectordb.interfaces import VectorStore


class AGraph:
    """Unified knowledge graph system supporting construction, storage and conversation functions."""

    def __init__(
        self,
        collection_name: str = "agraph_knowledge",
        persist_directory: Optional[str] = None,
        vector_store_type: str = "chroma",
        config: Optional[BuilderConfig] = None,
        use_openai_embeddings: bool = True,
        **_kwargs: Any,
    ) -> None:
        """Initialize AGraph system.

        Args:
            collection_name: Collection name.
            persist_directory: Storage persistence directory.
            vector_store_type: Vector store type ('chroma', 'memory').
            config: Builder configuration.
            use_openai_embeddings: Whether to use OpenAI embeddings.
            **kwargs: Other parameters.
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory or "./agraph_vectordb"
        self.vector_store_type = vector_store_type
        self.use_openai_embeddings = use_openai_embeddings
        self.settings = get_settings()

        # Initialize configuration
        if config is None:
            config = BuilderConfig(
                chunk_size=self.settings.text.max_chunk_size,
                chunk_overlap=self.settings.text.chunk_overlap,
                llm_provider=self.settings.llm.provider,
                llm_model=self.settings.llm.model,
                entity_confidence_threshold=0.7,
                relation_confidence_threshold=0.6,
                cache_dir=os.path.join(self.persist_directory, "cache"),
            )
        self.config = config

        # Initialize components
        self.vector_store: Optional[VectorStore] = None
        self.builder: Optional[KnowledgeGraphBuilder] = None
        self.knowledge_graph: Optional[KnowledgeGraph] = None
        self._is_initialized = False
        self._background_tasks: List[asyncio.Task] = []

        logger.info(
            f"AGraph initialization completed, collection: {collection_name}, persist_dir: {self.persist_directory}"
        )

    async def initialize(self) -> None:
        """Asynchronously initialize all components."""
        if self._is_initialized:
            logger.warning("AGraph is already initialized")
            return

        logger.info("Starting AGraph component initialization...")

        try:
            # 1. Initialize vector store
            await self._initialize_vector_store()

            # 2. Initialize knowledge graph builder
            self._initialize_builder()

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

    def _initialize_builder(self) -> None:
        """Initialize knowledge graph builder."""
        try:
            self.builder = KnowledgeGraphBuilder(config=self.config)
            logger.info("Knowledge graph builder initialization successful")
        except Exception as e:
            logger.error(f"Knowledge graph builder initialization failed: {e}")
            raise

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

        if not self.builder:
            raise RuntimeError("Knowledge graph builder not initialized")

        # Process document path parameters
        if isinstance(documents, (str, Path)):
            documents = [Path(documents)]
        elif isinstance(documents, list):
            documents = [Path(doc) if isinstance(doc, str) else doc for doc in documents]

        # Convert to Union[str, Path] list
        documents_list: List[Union[str, Path]] = [str(doc) for doc in documents]

        logger.info(
            f"Starting to build knowledge graph from {len(documents_list)} documents: {graph_name}"
        )

        try:
            # Use builder to construct knowledge graph
            self.knowledge_graph = await self.builder.build_from_documents(
                documents=documents_list,
                graph_name=graph_name,
                graph_description=graph_description,
                use_cache=use_cache,
            )

            # Asynchronously save to vector store
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

            # Asynchronously save to vector store
            if save_to_vector_store:
                task = asyncio.create_task(self._save_to_vector_store())
                self._background_tasks.append(task)

            logger.info(
                f"Knowledge graph construction completed: {len(self.knowledge_graph.entities)} entities, "
                f"{len(self.knowledge_graph.relations)} relations, "
                f"{len(self.knowledge_graph.text_chunks)} text chunks"
            )

            return self.knowledge_graph

        except Exception as e:
            logger.error(f"Building knowledge graph from texts failed: {e}")
            raise

    # =============== Vector Storage Functions ===============

    async def _save_to_vector_store(self) -> None:
        """Save knowledge graph to vector store."""
        if not self.vector_store or not self.knowledge_graph:
            logger.warning("Vector store or knowledge graph not initialized, skipping save")
            return

        try:
            logger.info("Starting to save knowledge graph to vector store...")

            # Batch save entities
            if self.knowledge_graph.entities:
                await self.vector_store.batch_add_entities(
                    list(self.knowledge_graph.entities.values())
                )
                logger.info(f"Saved {len(self.knowledge_graph.entities)} entities")

            # Batch save relations
            if self.knowledge_graph.relations:
                await self.vector_store.batch_add_relations(
                    list(self.knowledge_graph.relations.values())
                )
                logger.info(f"Saved {len(self.knowledge_graph.relations)} relations")

            # Batch save clusters
            if self.knowledge_graph.clusters:
                await self.vector_store.batch_add_clusters(
                    list(self.knowledge_graph.clusters.values())
                )
                logger.info(f"Saved {len(self.knowledge_graph.clusters)} clusters")

            # Batch save text chunks
            if self.knowledge_graph.text_chunks:
                await self.vector_store.batch_add_text_chunks(
                    list(self.knowledge_graph.text_chunks.values())
                )
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
            context_info = await self._retrieve_context(
                question, entity_top_k, relation_top_k, text_chunk_top_k
            )
            logger.info(f"Retrieved context information: {context_info}")

            # 2. Build prompt
            prompt = self._build_chat_prompt(
                question, context_info, conversation_history, response_type
            )

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
            tasks = [
                self.search_entities(query, entity_top_k),
                self.search_relations(query, relation_top_k),
                self.search_text_chunks(query, text_chunk_top_k),
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process entity results
            if isinstance(results[0], list):
                context["entities"] = [
                    {"entity": entity, "score": score} for entity, score in results[0]
                ]

            # Process relation results
            if isinstance(results[1], list):
                context["relations"] = [
                    {"relation": relation, "score": score} for relation, score in results[1]
                ]

            # Process text chunk results
            if isinstance(results[2], list):
                context["text_chunks"] = [
                    {"text_chunk": chunk, "score": score} for chunk, score in results[2]
                ]

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

        # Add entity information
        if context_info.get("entities"):
            kg_context_parts.append("相关实体:")
            for item in context_info["entities"][:3]:
                entity = item["entity"]
                kg_context_parts.append(
                    f"- {entity.name} ({entity.entity_type}): {entity.description or 'N/A'}"
                )

        # Add relation information
        if context_info.get("relations"):
            kg_context_parts.append("\n相关关系:")
            for item in context_info["relations"][:3]:
                relation = item["relation"]
                head_name = relation.head_entity.name if relation.head_entity else "未知"
                tail_name = relation.tail_entity.name if relation.tail_entity else "未知"
                kg_context_parts.append(
                    f"- {head_name} --[{relation.relation_type}]--> {tail_name}"
                )

        # Add text chunk information
        if context_info.get("text_chunks"):
            kg_context_parts.append("\n相关文档内容:")
            for item in context_info["text_chunks"][:3]:
                chunk = item["text_chunk"]
                content_preview = (
                    chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content
                )
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

            client = openai.AsyncOpenAI(
                api_key=self.settings.openai.api_key, base_url=self.settings.openai.api_base
            )

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

            client = openai.AsyncOpenAI(
                api_key=self.settings.openai.api_key, base_url=self.settings.openai.api_base
            )

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

        # Knowledge graph statistics
        if self.knowledge_graph:
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
        """Clear all data."""
        try:
            if self.vector_store:
                await self.vector_store.clear_all()

            if self.builder and hasattr(self.builder, "clear_cache"):
                self.builder.clear_cache()

            self.knowledge_graph = None
            logger.info("All data cleared")
            return True

        except Exception as e:
            logger.error(f"Clearing data failed: {e}")
            return False

    async def close(self) -> None:
        """Close AGraph system."""
        try:
            # Wait for all background tasks to complete
            if self._background_tasks:
                logger.info(
                    f"Waiting for {len(self._background_tasks)} background tasks to complete..."
                )
                await asyncio.gather(*self._background_tasks, return_exceptions=True)
                self._background_tasks.clear()
                logger.info("All background tasks completed")

            if self.vector_store:
                await self.vector_store.close()

            logger.info("AGraph system closed")

        except Exception as e:
            logger.error(f"Closing system failed: {e}")

    # =============== Context Manager Support ===============

    async def __aenter__(self) -> "AGraph":
        """Async context manager entry."""
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
            f"has_kg={self.has_knowledge_graph})"
        )
