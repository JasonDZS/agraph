"""
图嵌入处理器模块

专注于知识图谱的嵌入向量生成和管理，不包含检索功能
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np
from openai import AsyncOpenAI

from ..config import settings
from ..entities import Entity
from ..graph import KnowledgeGraph
from ..logger import logger
from ..relations import Relation
from ..storage import JsonVectorStorage, VectorStorage
from ..text import TextChunk
from ..utils import get_type_value


class GraphEmbedding(ABC):
    """图嵌入基类 - 专注于图相关的嵌入向量生成和管理"""

    def __init__(
        self,
        embedding_dim: int = 128,
        vector_storage: Optional[VectorStorage] = None,
        openai_api_key: Optional[str] = None,
        embedding_model: str = "text-embedding-3-small",
    ):
        self.embedding_dim = embedding_dim
        self.vector_storage = vector_storage or JsonVectorStorage("embeddings.json")
        self.entity_to_id: Dict[str, int] = {}
        self.id_to_entity: Dict[int, str] = {}
        self.relation_to_id: Dict[str, int] = {}
        self.id_to_relation: Dict[int, str] = {}
        self.text_chunk_to_id: Dict[str, int] = {}
        self.id_to_text_chunk: Dict[int, str] = {}
        self.openai_client = AsyncOpenAI(api_key=openai_api_key) if openai_api_key else None
        self.embedding_model = embedding_model

    @abstractmethod
    def fit(self, graph: KnowledgeGraph) -> bool:
        """
        训练图嵌入模型

        Args:
            graph: 知识图谱

        Returns:
            bool: 是否成功
        """
        pass

    @abstractmethod
    def get_entity_embedding(self, entity_id: str) -> Optional[np.ndarray]:
        """
        获取实体嵌入向量

        Args:
            entity_id: 实体ID

        Returns:
            np.ndarray: 嵌入向量，如果不存在则返回None
        """
        pass

    def get_relation_embedding(self, relation_id: str) -> Optional[np.ndarray]:
        """
        获取关系嵌入向量

        Args:
            relation_id: 关系ID

        Returns:
            np.ndarray: 嵌入向量，如果不存在则返回None
        """
        return self.vector_storage.get_vector(f"relation_{relation_id}")

    def get_text_chunk_embedding(self, text_chunk_id: str) -> Optional[np.ndarray]:
        """
        获取文本块嵌入向量

        Args:
            text_chunk_id: 文本块ID

        Returns:
            np.ndarray: 嵌入向量，如果不存在则返回None
        """
        return self.vector_storage.get_vector(f"text_chunk_{text_chunk_id}")

    def compute_entity_similarity(self, entity1_id: str, entity2_id: str) -> float:
        """
        计算实体相似度

        Args:
            entity1_id: 实体1 ID
            entity2_id: 实体2 ID

        Returns:
            float: 相似度分数 (0-1)
        """
        try:
            emb1 = self.get_entity_embedding(entity1_id)
            emb2 = self.get_entity_embedding(entity2_id)

            if emb1 is None or emb2 is None:
                return 0.0

            return self.vector_storage.compute_cosine_similarity(emb1, emb2)

        except Exception as e:
            logger.error(f"Error computing entity similarity: {e}")
            return 0.0

    def recommend_entities(
        self, entity_id: str, top_k: int = 10, similarity_threshold: float = 0.5
    ) -> List[Tuple[str, float]]:
        """
        基于嵌入的实体推荐

        Args:
            entity_id: 输入实体ID
            top_k: 推荐数量
            similarity_threshold: 相似度阈值

        Returns:
            List[Tuple[str, float]]: 推荐实体列表及相似度
        """
        try:
            entity_embedding = self.get_entity_embedding(entity_id)
            if entity_embedding is None:
                return []

            # 使用向量存储搜索相似向量
            similar_vectors = self.vector_storage.search_similar_vectors(
                entity_embedding, top_k + 1, similarity_threshold
            )

            # 过滤掉自己，并移除entity_前缀
            results = []
            for vector_id, similarity in similar_vectors:
                if vector_id.startswith("entity_"):
                    actual_entity_id = vector_id[7:]  # 移除"entity_"前缀
                    if actual_entity_id != entity_id:
                        results.append((actual_entity_id, similarity))

            return results[:top_k]

        except Exception as e:
            logger.error(f"Error recommending entities: {e}")
            return []

    async def embed_text(self, text: str) -> Optional[np.ndarray]:
        """
        将文本转换为嵌入向量

        Args:
            text: 输入文本

        Returns:
            np.ndarray: 嵌入向量，如果失败返回None
        """
        if not self.openai_client:
            logger.warning("No OpenAI client configured for text embedding")
            return None

        try:
            response = await self.openai_client.embeddings.create(input=text, model=self.embedding_model)
            embedding = np.array(response.data[0].embedding, dtype=np.float32)
            return embedding
        except Exception as e:
            logger.error("Error embedding text: %s", e)
            return None

    async def build_text_embeddings(self, graph: KnowledgeGraph) -> bool:
        """
        为图中的实体、关系和文本块构建文本嵌入

        Args:
            graph: 知识图谱

        Returns:
            bool: 是否成功
        """
        if not self.openai_client:
            logger.warning("No OpenAI client configured for text embedding")
            return False

        try:
            entity_vectors = {}
            relation_vectors = {}
            text_chunk_vectors = {}

            # 为实体构建文本嵌入
            for entity_id, entity in graph.entities.items():
                entity_text = self._entity_to_text(entity)
                embedding = await self.embed_text(entity_text)
                if embedding is not None:
                    entity_vectors[f"entity_text_{entity_id}"] = embedding

            # 为关系构建文本嵌入
            for relation_id, relation in graph.relations.items():
                relation_text = self._relation_to_text(relation)
                embedding = await self.embed_text(relation_text)
                if embedding is not None:
                    relation_vectors[f"relation_text_{relation_id}"] = embedding

            # 为文本块构建文本嵌入
            for text_chunk_id, text_chunk in graph.text_chunks.items():
                text_chunk_text = self._text_chunk_to_text(text_chunk)
                embedding = await self.embed_text(text_chunk_text)
                if embedding is not None:
                    text_chunk_vectors[f"text_chunk_text_{text_chunk_id}"] = embedding

            # 保存到向量存储
            all_vectors = {**entity_vectors, **relation_vectors, **text_chunk_vectors}
            success = self.vector_storage.save_vectors(
                all_vectors, {"embedding_model": self.embedding_model, "embedding_type": "text"}
            )

            if success:
                logger.info(
                    "Built text embeddings for %d entities, %d relations and %d text chunks",
                    len(entity_vectors),
                    len(relation_vectors),
                    len(text_chunk_vectors),
                )
            return success

        except Exception as e:
            logger.error("Error building text embeddings: %s", e)
            return False

    def save_embeddings(self, storage_path: Optional[str] = None) -> bool:
        """
        保存嵌入向量和映射信息

        Args:
            storage_path: 可选的存储路径，仅对JsonVectorStorage有效

        Returns:
            bool: 保存是否成功
        """
        try:
            # 保存映射信息到向量存储的元数据
            mapping_metadata = {
                "entity_to_id": self.entity_to_id,
                "id_to_entity": self.id_to_entity,
                "relation_to_id": self.relation_to_id,
                "id_to_relation": self.id_to_relation,
                "text_chunk_to_id": self.text_chunk_to_id,
                "id_to_text_chunk": self.id_to_text_chunk,
                "embedding_dim": self.embedding_dim,
                "embedding_model": self.embedding_model,
            }

            # 如果是JsonVectorStorage且提供了路径，更新路径
            if storage_path and isinstance(self.vector_storage, JsonVectorStorage):
                self.vector_storage.file_path = storage_path

            success = self.vector_storage.save_vectors({}, mapping_metadata)

            if success:
                logger.info("Graph embeddings saved successfully")
            return success

        except Exception as e:
            logger.error("Error saving graph embeddings: %s", e)
            return False

    def load_embeddings(self, storage_path: Optional[str] = None) -> bool:
        """
        加载嵌入向量和映射信息

        Args:
            storage_path: 可选的存储路径，仅对JsonVectorStorage有效

        Returns:
            bool: 加载是否成功
        """
        try:
            # 如果是JsonVectorStorage且提供了路径，更新路径
            if storage_path and isinstance(self.vector_storage, JsonVectorStorage):
                self.vector_storage.file_path = storage_path

            vectors, metadata = self.vector_storage.load_vectors()

            # 恢复映射信息
            self.entity_to_id = metadata.get("entity_to_id", {})
            self.id_to_entity = metadata.get("id_to_entity", {})
            self.relation_to_id = metadata.get("relation_to_id", {})
            self.id_to_relation = metadata.get("id_to_relation", {})
            self.text_chunk_to_id = metadata.get("text_chunk_to_id", {})
            self.id_to_text_chunk = metadata.get("id_to_text_chunk", {})
            self.embedding_dim = metadata.get("embedding_dim", self.embedding_dim)
            self.embedding_model = metadata.get("embedding_model", self.embedding_model)

            logger.info("Graph embeddings loaded successfully")
            return True

        except Exception as e:
            logger.error("Error loading graph embeddings: %s", e)
            return False

    def _entity_to_text(self, entity: Entity) -> str:
        """将实体转换为文本描述"""
        text_parts = [f"Entity: {entity.name}"]

        if entity.description:
            text_parts.append(f"Description: {entity.description}")

        if entity.aliases:
            text_parts.append(f"Aliases: {', '.join(entity.aliases)}")

        text_parts.append(f"Type: {get_type_value(entity.entity_type)}")

        if entity.properties:
            properties_text = ", ".join([f"{k}: {v}" for k, v in entity.properties.items()])
            text_parts.append(f"Properties: {properties_text}")

        return " | ".join(text_parts)

    def _relation_to_text(self, relation: Relation) -> str:
        """将关系转换为文本描述"""
        text_parts = []

        if relation.head_entity and relation.tail_entity:
            text_parts.append(
                f"Relation: {relation.head_entity.name} {get_type_value(relation.relation_type)} {relation.tail_entity.name}"
            )

        if relation.description:
            text_parts.append(f"Description: {relation.description}")

        if relation.properties:
            properties_text = ", ".join([f"{k}: {v}" for k, v in relation.properties.items()])
            text_parts.append(f"Properties: {properties_text}")

        return " | ".join(text_parts)

    def _text_chunk_to_text(self, text_chunk: TextChunk) -> str:
        """将文本块转换为文本描述"""
        text_parts = []

        # 添加标题
        if text_chunk.title:
            text_parts.append(f"Title: {text_chunk.title}")

        # 添加内容
        if text_chunk.content:
            text_parts.append(f"Content: {text_chunk.content}")

        # 添加来源
        if text_chunk.source:
            text_parts.append(f"Source: {text_chunk.source}")

        # 添加类型
        if text_chunk.chunk_type:
            text_parts.append(f"Type: {text_chunk.chunk_type}")

        # 添加元数据
        if text_chunk.metadata:
            metadata_text = ", ".join([f"{k}: {v}" for k, v in text_chunk.metadata.items()])
            text_parts.append(f"Metadata: {metadata_text}")

        return " | ".join(text_parts)

    def _build_entity_mapping(self, graph: KnowledgeGraph) -> None:
        """构建实体映射"""
        entity_list = list(graph.entities.keys())
        self.entity_to_id = {entity: i for i, entity in enumerate(entity_list)}
        self.id_to_entity = {i: entity for entity, i in self.entity_to_id.items()}

    def _build_relation_mapping(self, graph: KnowledgeGraph) -> None:
        """构建关系映射"""
        relation_types = set()
        for relation in graph.relations.values():
            relation_types.add(get_type_value(relation.relation_type))

        relation_list = list(relation_types)
        self.relation_to_id = {relation: i for i, relation in enumerate(relation_list)}
        self.id_to_relation = {i: relation for relation, i in self.relation_to_id.items()}

    def _build_text_chunk_mapping(self, graph: KnowledgeGraph) -> None:
        """构建文本块映射"""
        text_chunk_list = list(graph.text_chunks.keys())
        self.text_chunk_to_id = {text_chunk: i for i, text_chunk in enumerate(text_chunk_list)}
        self.id_to_text_chunk = {i: text_chunk for text_chunk, i in self.text_chunk_to_id.items()}


class OpenAIEmbedding(GraphEmbedding):
    """基于OpenAI API的图嵌入实现 - 专注于嵌入向量生成"""

    def __init__(
        self,
        vector_storage: Optional[VectorStorage] = None,
        openai_api_key: Optional[str] = None,
        openai_api_base: Optional[str] = None,
        embedding_model: str = "text-embedding-3-small",
        batch_size: int = 64,
        max_concurrent: int = 5,
    ):
        """
        初始化OpenAI嵌入处理器

        Args:
            vector_storage: 向量存储后端
            openai_api_key: OpenAI API密钥
            openai_api_base: OpenAI API基础URL
            embedding_model: 嵌入模型名称
            batch_size: 批处理大小
            max_concurrent: 最大并发数
        """
        # 从OpenAI获取实际的embedding维度
        embedding_dim = self._get_embedding_dim_for_model(embedding_model)

        super().__init__(
            embedding_dim=embedding_dim,
            vector_storage=vector_storage,
            openai_api_key=openai_api_key,
            embedding_model=embedding_model,
        )

        self.openai_client = AsyncOpenAI(api_key=openai_api_key, base_url=openai_api_base)
        self.batch_size = batch_size
        self.max_concurrent = max_concurrent
        self._entity_embeddings_cache: Dict[str, np.ndarray] = {}
        self._relation_embeddings_cache: Dict[str, np.ndarray] = {}
        self._text_chunk_embeddings_cache: Dict[str, np.ndarray] = {}

    def _get_embedding_dim_for_model(self, model: str) -> int:
        """根据模型名称获取embedding维度"""
        if settings.EMBEDDING_DIM:
            return settings.EMBEDDING_DIM
        model_dims = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        return model_dims.get(model, 1536)  # 默认1536

    def fit(self, graph: KnowledgeGraph) -> bool:
        """
        使用OpenAI API为图中的实体和关系构建嵌入

        Args:
            graph: 知识图谱

        Returns:
            bool: 是否成功
        """
        try:
            # 使用asyncio运行异步方法
            return asyncio.run(self._fit_async(graph))
        except Exception as e:
            logger.error(f"Error in OpenAI embedding fit: {e}")
            return False

    async def _fit_async(self, graph: KnowledgeGraph) -> bool:
        """异步版本的fit方法"""
        try:
            # 构建文本嵌入
            await self.build_text_embeddings(graph)
            return True
        except Exception as e:
            logger.error(f"Error in async fit: {e}")
            return False

    async def build_text_embeddings(self, graph: KnowledgeGraph) -> bool:
        """为图中的实体、关系和文本块构建文本嵌入"""
        try:
            logger.info("Building text embeddings using OpenAI API...")

            # 收集所有需要嵌入的文本
            texts_to_embed = []
            text_to_id = {}

            # 实体文本
            for entity_id, entity in graph.entities.items():
                text = self._entity_to_text(entity)
                texts_to_embed.append(text)
                text_to_id[text] = f"entity_{entity_id}"

            # 关系文本
            for relation_id, relation in graph.relations.items():
                text = self._relation_to_text(relation)
                texts_to_embed.append(text)
                text_to_id[text] = f"relation_{relation_id}"

            # 文本块文本
            for text_chunk_id, text_chunk in graph.text_chunks.items():
                text = self._text_chunk_to_text(text_chunk)
                texts_to_embed.append(text)
                text_to_id[text] = f"text_chunk_{text_chunk_id}"

            if not texts_to_embed:
                logger.warning("No texts to embed")
                return True

            # 批量获取嵌入
            embeddings = await self._get_embeddings_batch(texts_to_embed)

            # 存储嵌入
            for text, embedding in zip(texts_to_embed, embeddings):
                item_id = text_to_id[text]
                self.vector_storage.add_vector(item_id, embedding, metadata={"text": text})

                # 更新缓存
                if item_id.startswith("entity_"):
                    entity_id = item_id[7:]  # 去掉"entity_"前缀
                    self._entity_embeddings_cache[entity_id] = embedding
                elif item_id.startswith("relation_"):
                    relation_id = item_id[9:]  # 去掉"relation_"前缀
                    self._relation_embeddings_cache[relation_id] = embedding
                elif item_id.startswith("text_chunk_"):
                    text_chunk_id = item_id[11:]  # 去掉"text_chunk_"前缀
                    self._text_chunk_embeddings_cache[text_chunk_id] = embedding

            logger.info(f"Successfully built embeddings for {len(texts_to_embed)} items")
            return True
        except Exception as e:
            logger.error(f"Error building text embeddings: {e}")
            return False

    def _entity_to_text(self, entity: Entity) -> str:
        """将实体转换为文本"""
        parts = [entity.name]
        if entity.description:
            parts.append(entity.description)
        if entity.aliases:
            parts.append(f"别名: {', '.join(entity.aliases)}")
        return " ".join(parts)

    def _relation_to_text(self, relation: Relation) -> str:
        """将关系转换为文本"""
        parts = []
        if relation.head_entity and relation.tail_entity:
            parts.append(
                f"{relation.head_entity.name} {get_type_value(relation.relation_type)} {relation.tail_entity.name}"
            )
        if relation.description:
            parts.append(relation.description)
        return " ".join(parts) if parts else get_type_value(relation.relation_type)

    async def _get_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """批量获取文本嵌入"""
        all_embeddings = []

        # 使用信号量控制并发
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def get_batch_embedding(batch_texts: List[str]) -> List[np.ndarray]:
            async with semaphore:
                try:
                    if self.openai_client is None:
                        raise ValueError("OpenAI client is not initialized")
                    response = await self.openai_client.embeddings.create(model=self.embedding_model, input=batch_texts)
                    return [np.array(item.embedding) for item in response.data]
                except Exception as e:
                    logger.error(f"Error getting batch embedding: {e}")
                    # 返回零向量作为备用
                    return [np.zeros(self.embedding_dim) for _ in batch_texts]

        # 分批处理
        tasks = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            tasks.append(get_batch_embedding(batch))

        # 并发执行
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        # 合并结果
        for i, result in enumerate(batch_results):
            if isinstance(result, Exception):
                logger.error(f"Batch embedding failed: {result}")
                # 为失败的批次添加零向量
                batch_start = i * self.batch_size
                batch_end = min(batch_start + self.batch_size, len(texts))
                batch_size = batch_end - batch_start
                all_embeddings.extend([np.zeros(self.embedding_dim)] * batch_size)
            else:
                # 确保 result 是正确的类型
                if isinstance(result, list):
                    all_embeddings.extend(result)
                else:
                    logger.error(f"Unexpected result type: {type(result)}")
                    # 添加零向量作为替代
                    batch_start = i * self.batch_size
                    batch_end = min(batch_start + self.batch_size, len(texts))
                    batch_size = batch_end - batch_start
                    all_embeddings.extend([np.zeros(self.embedding_dim)] * batch_size)

        return all_embeddings[: len(texts)]  # 确保长度匹配

    def get_entity_embedding(self, entity_id: str) -> Optional[np.ndarray]:
        """获取实体嵌入向量"""
        # 先检查缓存
        if entity_id in self._entity_embeddings_cache:
            return self._entity_embeddings_cache[entity_id]

        # 从存储中获取
        embedding = self.vector_storage.get_vector(f"entity_{entity_id}")
        if embedding is not None:
            self._entity_embeddings_cache[entity_id] = embedding

        return embedding

    def get_relation_embedding(self, relation_id: str) -> Optional[np.ndarray]:
        """获取关系嵌入向量"""
        # 先检查缓存
        if relation_id in self._relation_embeddings_cache:
            return self._relation_embeddings_cache[relation_id]

        # 从存储中获取
        embedding = self.vector_storage.get_vector(f"relation_{relation_id}")
        if embedding is not None:
            self._relation_embeddings_cache[relation_id] = embedding

        return embedding

    def get_text_chunk_embedding(self, text_chunk_id: str) -> Optional[np.ndarray]:
        """获取文本块嵌入向量"""
        # 先检查缓存
        if text_chunk_id in self._text_chunk_embeddings_cache:
            return self._text_chunk_embeddings_cache[text_chunk_id]

        # 从存储中获取
        embedding = self.vector_storage.get_vector(f"text_chunk_{text_chunk_id}")
        if embedding is not None:
            self._text_chunk_embeddings_cache[text_chunk_id] = embedding

        return embedding

    async def _get_single_embedding(self, text: str) -> Optional[np.ndarray]:
        """获取单个文本的嵌入"""
        try:
            if self.openai_client is None:
                raise ValueError("OpenAI client is not initialized")
            response = await self.openai_client.embeddings.create(model=self.embedding_model, input=[text])
            return np.array(response.data[0].embedding)
        except Exception as e:
            logger.error(f"Error getting single embedding: {e}")
            return None

    def save_embeddings(self, storage_path: Optional[str] = None) -> bool:
        """保存嵌入到存储"""
        try:
            # 调用 vector_storage 的保存方法
            if hasattr(self.vector_storage, "save"):
                self.vector_storage.save()
            logger.info("OpenAI embeddings saved to storage")
            return True
        except Exception as e:
            logger.error(f"Error saving embeddings: {e}")
            return False
