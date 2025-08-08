"""
基于Interface Segregation Principle的LLM知识图谱构建器

遵循ISP原则，将LLM构建器分解为多个专门的接口和实现，
客户端只需要依赖他们实际使用的功能。
"""

import asyncio
import json
import os.path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from ..config import settings
from ..embeddings import OpenAIEmbedding
from ..entities import Entity
from ..extractors.llm_entity_extractor import LLMEntityExtractor
from ..extractors.llm_relation_extractor import LLMRelationExtractor
from ..graph import KnowledgeGraph
from ..logger import logger
from ..relations import Relation
from ..storage import JsonVectorStorage, VectorStorage
from ..types import EntityType, RelationType
from ..utils import get_type_value
from .interfaces import (
    BasicGraphBuilder,
    BatchGraphBuilder,
    FullFeaturedGraphBuilder,
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


class LLMUsageTracker:
    """LLM使用统计跟踪器 - 独立的关注点"""

    def __init__(self, llm_model: str, embedding_model: str):
        self.usage_stats: Dict[str, Any] = {
            "llm_model": {
                "model_name": llm_model,
                "total_calls": 0,
                "entity_extraction_calls": 0,
                "relation_extraction_calls": 0,
                "deduplication_calls": 0,
                "errors": 0,
                "call_history": [],
            },
            "embedding_model": {
                "model_name": embedding_model,
                "total_calls": 0,
                "entity_embedding_calls": 0,
                "relation_embedding_calls": 0,
                "errors": 0,
                "call_history": [],
            },
            "session_info": {
                "start_time": datetime.now(),
                "end_time": None,
                "total_texts_processed": 0,
                "total_entities_extracted": 0,
                "total_relations_extracted": 0,
            },
        }

    def track_llm_call(
        self,
        call_type: str,
        input_msg: str = "",
        output_msg: str = "",
        success: bool = True,
        error_msg: Optional[str] = None,
    ) -> None:
        """跟踪LLM API调用统计信息"""
        llm_stats = self.usage_stats["llm_model"]

        if success:
            llm_stats["total_calls"] += 1
            if call_type == "entity_extraction":
                llm_stats["entity_extraction_calls"] += 1
            elif call_type == "relation_extraction":
                llm_stats["relation_extraction_calls"] += 1
            elif call_type == "deduplication":
                llm_stats["deduplication_calls"] += 1
        else:
            llm_stats["errors"] += 1

        # 记录调用历史
        call_record = {
            "timestamp": datetime.now(),
            "call_type": call_type,
            "input_msg": input_msg,
            "output_msg": output_msg,
            "success": success,
            "error_msg": error_msg,
        }
        llm_stats["call_history"].append(call_record)

    def track_embedding_call(self, call_type: str, success: bool = True, error_msg: Optional[str] = None) -> None:
        """跟踪embedding API调用统计信息"""
        embedding_stats = self.usage_stats["embedding_model"]

        if success:
            embedding_stats["total_calls"] += 1
            if call_type == "entity_embedding":
                embedding_stats["entity_embedding_calls"] += 1
            elif call_type == "relation_embedding":
                embedding_stats["relation_embedding_calls"] += 1
        else:
            embedding_stats["errors"] += 1

        # 记录调用历史
        call_record = {
            "timestamp": datetime.now(),
            "call_type": call_type,
            "success": success,
            "error_msg": error_msg,
        }
        embedding_stats["call_history"].append(call_record)

    def update_session_stats(self, texts_processed: int, entities_extracted: int, relations_extracted: int) -> None:
        """更新会话统计"""
        session = self.usage_stats["session_info"]
        session["total_texts_processed"] += texts_processed
        session["total_entities_extracted"] += entities_extracted
        session["total_relations_extracted"] += relations_extracted

    def get_usage_statistics(self) -> Dict[str, Any]:
        """获取使用统计信息"""
        return self.usage_stats.copy()

    def export_usage_stats(self, file_path: str) -> None:
        """导出使用统计到JSON文件"""
        stats_copy = json.loads(json.dumps(self.usage_stats, default=str))
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(stats_copy, f, ensure_ascii=False, indent=2)
        logger.info(f"Usage statistics exported to {file_path}")

    def print_usage_summary(self) -> None:
        """打印使用统计摘要"""
        stats = self.usage_stats

        print("\n" + "=" * 60)
        print("📊 LLM Graph Builder 使用统计")
        print("=" * 60)

        # 会话信息
        session = stats["session_info"]
        start_time = session["start_time"]
        end_time = session.get("end_time") or datetime.now()
        duration = end_time - start_time

        print(f"⏱️  会话时长: {duration}")
        print(f"📄 处理文本数: {session['total_texts_processed']}")
        print(f"🏷️  提取实体数: {session['total_entities_extracted']}")
        print(f"🔗 提取关系数: {session['total_relations_extracted']}")
        print()

        # LLM统计
        llm = stats["llm_model"]
        print(f"🤖 LLM模型: {llm['model_name']}")
        print(f"   📞 总调用次数: {llm['total_calls']}")
        print(f"   ❌ 错误次数: {llm['errors']}")
        print(f"   ├── 实体提取: {llm['entity_extraction_calls']} 次")
        print(f"   ├── 关系提取: {llm['relation_extraction_calls']} 次")
        print(f"   └── 实体去重: {llm['deduplication_calls']} 次")
        print()

        # Embedding统计
        embedding = stats["embedding_model"]
        print(f"🔤 Embedding模型: {embedding['model_name']}")
        print(f"   📞 总调用次数: {embedding['total_calls']}")
        print(f"   ❌ 错误次数: {embedding['errors']}")
        print(f"   ├── 实体嵌入: {embedding['entity_embedding_calls']} 次")
        print(f"   └── 关系嵌入: {embedding['relation_embedding_calls']} 次")
        print()

    def cleanup(self) -> None:
        """清理资源"""
        self.usage_stats["session_info"]["end_time"] = str(datetime.now())


class LLMAsyncProcessor:
    """LLM异步处理器 - 独立的处理逻辑"""

    def __init__(
        self,
        entity_extractor: LLMEntityExtractor,
        relation_extractor: LLMRelationExtractor,
        usage_tracker: LLMUsageTracker,
        max_concurrent: int = 10,
    ):
        self.entity_extractor = entity_extractor
        self.relation_extractor = relation_extractor
        self.usage_tracker = usage_tracker
        self.max_concurrent = max_concurrent
        self._deduplication_calls_count = 0

    async def process_texts_async(self, texts: List[str]) -> Tuple[List[Dict], List[Dict]]:
        """异步处理文本列表"""
        if len(texts) <= 1:
            return await self._process_texts_sequential(texts)

        all_entities: List[Dict[str, Any]] = []
        all_relations: List[Dict[str, Any]] = []

        # 使用信号量控制并发数
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def process_with_semaphore(
            text: str, index: int
        ) -> Tuple[int, List[Dict[str, Any]], List[Dict[str, Any]]]:
            async with semaphore:
                return await self._process_single_text_async(text, index)

        # 创建异步任务
        tasks = [process_with_semaphore(text, i) for i, text in enumerate(texts)]

        # 批量执行
        logger.info(f"Processing {len(texts)} texts with max_concurrent={self.max_concurrent}")
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 处理结果
        valid_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Error in async execution: {result}")
            else:
                # 确保 result 是正确的类型
                if isinstance(result, tuple) and len(result) == 3:
                    valid_results.append(result)
                else:
                    logger.warning(f"Unexpected result format: {result}")

        # 按索引排序并合并结果
        if valid_results:
            valid_results.sort(key=lambda x: x[0])
            for _, entities_data, relations_data in valid_results:
                all_entities.extend(entities_data)
                all_relations.extend(relations_data)

        return all_entities, all_relations

    async def _process_single_text_async(self, text: str, index: int) -> Tuple[int, List[Dict], List[Dict]]:
        """异步处理单个文本"""
        try:
            # 异步提取实体
            entities = await self.entity_extractor._extract_entities_async(text)
            self.usage_tracker.track_llm_call(
                call_type="entity_extraction",
                input_msg=text,
                output_msg=str(entities),
                success=True,
            )

            entities_data = [
                {
                    "name": entity.name,
                    "type": get_type_value(entity.entity_type),
                    "description": entity.description,
                    "aliases": entity.aliases,
                    "properties": entity.properties,
                }
                for entity in entities
            ]

            # 异步提取关系
            relations = await self.relation_extractor._extract_relations_async(text, entities)
            self.usage_tracker.track_llm_call(call_type="relation_extraction", success=True)

            relations_data = [
                {
                    "head_entity": relation.head_entity.name if relation.head_entity else "",
                    "tail_entity": relation.tail_entity.name if relation.tail_entity else "",
                    "relation_type": get_type_value(relation.relation_type),
                    "description": relation.description,
                    "properties": relation.properties,
                    "confidence": relation.confidence,
                }
                for relation in relations
            ]

            return index, entities_data, relations_data

        except Exception as e:
            logger.error(f"Error processing text {index}: {e}")
            self.usage_tracker.track_llm_call(call_type="entity_extraction", success=False, error_msg=str(e))
            return index, [], []

    async def _process_texts_sequential(self, texts: List[str]) -> Tuple[List[Dict], List[Dict]]:
        """串行处理文本列表"""
        all_entities = []
        all_relations = []

        for i, text in enumerate(texts):
            logger.info(f"Processing text {i+1}/{len(texts)}")
            try:
                entities = await self.entity_extractor._extract_entities_async(text)
                self.usage_tracker.track_llm_call(call_type="entity_extraction", success=True)

                entities_data = [
                    {
                        "name": entity.name,
                        "type": get_type_value(entity.entity_type),
                        "description": entity.description,
                        "aliases": entity.aliases,
                        "properties": entity.properties,
                    }
                    for entity in entities
                ]
                all_entities.extend(entities_data)

                relations = await self.relation_extractor._extract_relations_async(text, entities)
                self.usage_tracker.track_llm_call(call_type="relation_extraction", success=True)

                relations_data = [
                    {
                        "head_entity": relation.head_entity.name if relation.head_entity else "",
                        "tail_entity": relation.tail_entity.name if relation.tail_entity else "",
                        "relation_type": get_type_value(relation.relation_type),
                        "description": relation.description,
                        "properties": relation.properties,
                        "confidence": relation.confidence,
                    }
                    for relation in relations
                ]
                all_relations.extend(relations_data)

            except Exception as e:
                logger.error(f"Error processing text {i+1}: {e}")
                self.usage_tracker.track_llm_call(call_type="entity_extraction", success=False, error_msg=str(e))

        return all_entities, all_relations

    async def deduplicate_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """优化的实体去重"""
        if len(entities) <= 1:
            return entities

        logger.info(f"Starting optimized deduplication for {len(entities)} entities")

        # Convert dict entities to Entity objects
        entity_objects = []
        for entity_data in entities:
            entity = Entity(
                id=self._generate_entity_id(entity_data["name"]),
                name=entity_data["name"],
                entity_type=EntityType(entity_data["type"]),
                description=entity_data.get("description", ""),
                aliases=entity_data.get("aliases", []),
                properties=entity_data.get("properties", {}),
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )
            entity_objects.append(entity)

        # 使用优化的并发去重方法
        deduplicated_entities = await self._deduplicate_entities_concurrent(entity_objects)

        # 跟踪去重调用统计
        if len(entity_objects) > 1:
            actual_calls = self._deduplication_calls_count
            for _ in range(actual_calls):
                self.usage_tracker.track_llm_call(call_type="deduplication", success=True)

        logger.info(f"Deduplication completed: {len(entity_objects)} -> {len(deduplicated_entities)} entities")

        # Convert back to dict format
        return [
            {
                "name": entity.name,
                "type": get_type_value(entity.entity_type),
                "description": entity.description,
                "aliases": entity.aliases,
                "properties": entity.properties,
            }
            for entity in deduplicated_entities
        ]

    async def _deduplicate_entities_concurrent(self, entities: List[Entity]) -> List[Entity]:
        """并发优化的实体去重方法"""
        if len(entities) <= 1:
            return entities

        self._deduplication_calls_count = 0

        # 第一步：快速预筛选
        logger.info("Step 1: Fast pre-filtering based on name and type similarity")
        candidate_pairs = self._prefilter_duplicate_candidates(entities)

        if not candidate_pairs:
            logger.info("No potential duplicates found in pre-filtering")
            return entities

        logger.info(f"Found {len(candidate_pairs)} potential duplicate pairs for LLM verification")

        # 第二步：并发LLM验证
        logger.info("Step 2: Concurrent LLM verification of potential duplicates")
        duplicate_pairs = await self._verify_duplicates_concurrent(entities, candidate_pairs)

        # 第三步：合并重复实体
        logger.info("Step 3: Merging duplicate entities")
        return self._merge_duplicate_entities(entities, duplicate_pairs)

    def _prefilter_duplicate_candidates(self, entities: List[Entity]) -> List[Tuple[int, int]]:
        """预筛选潜在重复实体对"""
        candidate_pairs = []

        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                entity1, entity2 = entities[i], entities[j]

                # 只对相同类型的实体进行比较
                if entity1.entity_type != entity2.entity_type:
                    continue

                # 计算名称相似度
                name_similarity = self._calculate_name_similarity(entity1.name, entity2.name)

                # 检查别名匹配
                alias_match = self._check_alias_match(entity1, entity2)

                # 如果名称相似度高或有别名匹配，则认为是候选重复对
                if name_similarity > 0.7 or alias_match:
                    candidate_pairs.append((i, j))

        return candidate_pairs

    def _calculate_name_similarity(self, name1: str, name2: str) -> float:
        """计算两个名称的相似度"""
        name1_lower = name1.lower().strip()
        name2_lower = name2.lower().strip()

        if name1_lower == name2_lower:
            return 1.0

        if name1_lower in name2_lower or name2_lower in name1_lower:
            return 0.8

        return self._levenshtein_similarity(name1_lower, name2_lower)

    def _levenshtein_similarity(self, s1: str, s2: str) -> float:
        """计算编辑距离相似度"""
        if len(s1) == 0 or len(s2) == 0:
            return 0.0

        if len(s1) > len(s2):
            s1, s2 = s2, s1

        distances = list(range(len(s1) + 1))
        for i2, c2 in enumerate(s2):
            new_distances = [i2 + 1]
            for i1, c1 in enumerate(s1):
                if c1 == c2:
                    new_distances.append(distances[i1])
                else:
                    new_distances.append(1 + min(distances[i1], distances[i1 + 1], new_distances[-1]))
            distances = new_distances

        max_len = max(len(s1), len(s2))
        return 1.0 - (distances[-1] / max_len)

    def _check_alias_match(self, entity1: Entity, entity2: Entity) -> bool:
        """检查两个实体是否有别名匹配"""
        all_names1 = {entity1.name.lower()} | {alias.lower() for alias in entity1.aliases}
        all_names2 = {entity2.name.lower()} | {alias.lower() for alias in entity2.aliases}
        return bool(all_names1 & all_names2)

    async def _verify_duplicates_concurrent(
        self, entities: List[Entity], candidate_pairs: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        """并发验证潜在重复实体对"""
        if not candidate_pairs:
            return []

        semaphore = asyncio.Semaphore(min(self.max_concurrent, 5))

        async def verify_single_pair(pair_idx: int, entity_pair: Tuple[int, int]) -> Tuple[int, bool]:
            async with semaphore:
                entity1_idx, entity2_idx = entity_pair
                entity1, entity2 = entities[entity1_idx], entities[entity2_idx]
                is_duplicate = await self.entity_extractor._check_entity_duplicate_llm(entity1, entity2)
                self._deduplication_calls_count += 1
                return pair_idx, is_duplicate

        tasks = [verify_single_pair(i, pair) for i, pair in enumerate(candidate_pairs)]

        logger.info(f"Verifying {len(candidate_pairs)} potential duplicate pairs concurrently")
        results = await asyncio.gather(*tasks, return_exceptions=True)

        duplicate_pairs = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Error in duplicate verification: {result}")
            elif isinstance(result, tuple) and len(result) == 2:
                pair_idx, is_duplicate = result
                if is_duplicate:
                    duplicate_pairs.append(candidate_pairs[pair_idx])
            else:
                logger.error(f"Unexpected result format: {result}")

        logger.info(f"Found {len(duplicate_pairs)} confirmed duplicate pairs")
        return duplicate_pairs

    def _merge_duplicate_entities(self, entities: List[Entity], duplicate_pairs: List[Tuple[int, int]]) -> List[Entity]:
        """合并重复实体"""
        if not duplicate_pairs:
            return entities

        # 使用并查集处理传递性重复关系
        parent = list(range(len(entities)))

        def find(x: int) -> int:
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x: int, y: int) -> None:
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        for i, j in duplicate_pairs:
            union(i, j)

        # 分组实体
        groups: Dict[int, List[int]] = {}
        for i in range(len(entities)):
            root = find(i)
            if root not in groups:
                groups[root] = []
            groups[root].append(i)

        # 合并每个组的实体
        merged_entities = []
        for group_indices in groups.values():
            if len(group_indices) == 1:
                merged_entities.append(entities[group_indices[0]])
            else:
                merged_entity = self._merge_multiple_entities([entities[i] for i in group_indices])
                merged_entities.append(merged_entity)

        return merged_entities

    def _merge_multiple_entities(self, entities_to_merge: List[Entity]) -> Entity:
        """合并多个实体为一个"""
        if not entities_to_merge:
            raise ValueError("No entities to merge")

        if len(entities_to_merge) == 1:
            return entities_to_merge[0]

        base_entity = entities_to_merge[0]
        all_aliases = set(base_entity.aliases)
        all_descriptions = [base_entity.description] if base_entity.description else []
        merged_properties = base_entity.properties.copy()

        for entity in entities_to_merge[1:]:
            all_aliases.update(entity.aliases)
            if entity.name != base_entity.name:
                all_aliases.add(entity.name)

            if entity.description and entity.description not in all_descriptions:
                all_descriptions.append(entity.description)

            merged_properties.update(entity.properties)

        merged_entity = Entity(
            id=base_entity.id,
            name=base_entity.name,
            entity_type=base_entity.entity_type,
            description="; ".join(all_descriptions),
            aliases=list(all_aliases),
            properties=merged_properties,
            created_at=base_entity.created_at,
            updated_at=datetime.now(),
        )

        return merged_entity

    @staticmethod
    def _generate_entity_id(name: str) -> str:
        """生成实体ID"""
        import hashlib

        return f"entity_{hashlib.md5(name.encode()).hexdigest()[:8]}"


class LLMGraphUtils:
    """LLM图构建工具类 - 公共工具方法"""

    @staticmethod
    def generate_entity_id(name: str) -> str:
        """生成实体ID"""
        import hashlib

        return f"entity_{hashlib.md5(name.encode()).hexdigest()[:8]}"

    @staticmethod
    def generate_relation_id(head: str, tail: str, relation_type: str) -> str:
        """生成关系ID"""
        import hashlib

        relation_str = f"{head}_{relation_type}_{tail}"
        return f"relation_{hashlib.md5(relation_str.encode()).hexdigest()[:8]}"

    @staticmethod
    def find_similar_entity(graph: KnowledgeGraph, entity_data: Dict[str, Any]) -> Optional[Entity]:
        """在现有图谱中查找相似实体"""
        entity_name = entity_data.get("name", "").lower()

        for entity in graph.entities.values():
            if entity.name.lower() == entity_name:
                return entity

            if entity_name in [alias.lower() for alias in entity.aliases]:
                return entity

            for alias in entity_data.get("aliases", []):
                if alias.lower() == entity.name.lower():
                    return entity
                if alias.lower() in [a.lower() for a in entity.aliases]:
                    return entity

        return None

    @staticmethod
    def merge_entity_data(entity: Entity, entity_data: Dict[str, Any]) -> None:
        """将新的实体数据合并到现有实体"""
        new_desc = entity_data.get("description", "")
        if new_desc and new_desc not in entity.description:
            entity.description = f"{entity.description}; {new_desc}" if entity.description else new_desc

        new_aliases = entity_data.get("aliases", [])
        for alias in new_aliases:
            if alias not in entity.aliases:
                entity.aliases.append(alias)

        new_props = entity_data.get("properties", {})
        entity.properties.update(new_props)
        entity.updated_at = datetime.now()

    @staticmethod
    def find_existing_relation(
        graph: KnowledgeGraph, head_name: str, tail_name: str, relation_type: str
    ) -> Optional[Relation]:
        """查找现有关系"""
        for relation in graph.relations.values():
            if (
                relation.head_entity
                and relation.tail_entity
                and relation.head_entity.name == head_name
                and relation.tail_entity.name == tail_name
                and get_type_value(relation.relation_type) == relation_type
            ):
                return relation
        return None

    @staticmethod
    def build_entities_from_data(graph: KnowledgeGraph, entity_data_list: List[Dict[str, Any]]) -> Dict[str, str]:
        """从数据构建实体并返回名称到ID的映射"""
        entity_mapping = {}

        for entity_data in entity_data_list:
            entity = Entity(
                id=LLMGraphUtils.generate_entity_id(entity_data["name"]),
                name=entity_data["name"],
                entity_type=EntityType(entity_data["type"]),
                description=entity_data.get("description", ""),
                aliases=entity_data.get("aliases", []),
                properties=entity_data.get("properties", {}),
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )
            graph.add_entity(entity)
            entity_mapping[entity_data["name"]] = entity.id

        return entity_mapping

    @staticmethod
    def build_relations_from_data(
        graph: KnowledgeGraph, relations_data: List[Dict[str, Any]], entity_mapping: Dict[str, str]
    ) -> int:
        """从数据构建关系并返回添加的关系数量"""
        relations_added = 0

        for relation_data in relations_data:
            head_name = relation_data["head_entity"]
            tail_name = relation_data["tail_entity"]

            if head_name in entity_mapping and tail_name in entity_mapping:
                head_entity = graph.get_entity(entity_mapping[head_name])
                tail_entity = graph.get_entity(entity_mapping[tail_name])

                if head_entity and tail_entity:
                    relation = Relation(
                        id=LLMGraphUtils.generate_relation_id(head_name, tail_name, relation_data["relation_type"]),
                        head_entity=head_entity,
                        tail_entity=tail_entity,
                        relation_type=RelationType(relation_data["relation_type"]),
                        description=relation_data.get("description", ""),
                        properties=relation_data.get("properties", {}),
                        confidence=relation_data.get("confidence", 1.0),
                        created_at=datetime.now(),
                        updated_at=datetime.now(),
                    )
                    graph.add_relation(relation)
                    relations_added += 1

        return relations_added


# ============================================================================
# ISP-compliant LLM Builder Implementations
# ============================================================================


class MinimalLLMGraphBuilder(BasicGraphBuilder):
    """
    最小化的LLM图构建器 - 只实现核心构建功能

    适用于只需要基本图构建而不需要更新、合并或其他高级功能的客户端。
    """

    def __init__(
        self,
        openai_api_key: str,
        openai_api_base: Optional[str] = None,
        llm_model: str = "gpt-4o-mini",
        max_tokens: int = 4000,
        temperature: float = 0.1,
        entity_extractor: Optional[LLMEntityExtractor] = None,
        relation_extractor: Optional[LLMRelationExtractor] = None,
    ):
        """
        初始化最小化LLM图构建器

        Args:
            openai_api_key: OpenAI API密钥
            openai_api_base: OpenAI API基础URL
            llm_model: 使用的LLM模型
            max_tokens: 最大token数
            temperature: 温度参数
            entity_extractor: 实体提取器
            relation_extractor: 关系提取器
        """
        self.entity_extractor = entity_extractor or LLMEntityExtractor(
            openai_api_key=openai_api_key,
            openai_api_base=openai_api_base,
            llm_model=llm_model,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        self.relation_extractor = relation_extractor or LLMRelationExtractor(
            openai_api_key=openai_api_key,
            openai_api_base=openai_api_base,
            llm_model=llm_model,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    async def build_graph(
        self,
        texts: Optional[List[str]] = None,
        database_schema: Optional[Dict[str, Any]] = None,
        graph_name: str = "minimal_llm_graph",
    ) -> KnowledgeGraph:
        """构建知识图谱 - 异步版本"""
        if not texts:
            return KnowledgeGraph(name=graph_name)

        try:
            graph = KnowledgeGraph(name=graph_name)
            all_entities = []
            all_relations = []

            # 串行处理每个文本
            for text in texts:
                # 异步提取实体
                entities = await self.entity_extractor._extract_entities_async(text)
                entity_data = [
                    {
                        "name": entity.name,
                        "type": get_type_value(entity.entity_type),
                        "description": entity.description,
                        "aliases": entity.aliases,
                        "properties": entity.properties,
                    }
                    for entity in entities
                ]
                all_entities.extend(entity_data)

                # 异步提取关系
                relations = await self.relation_extractor._extract_relations_async(text, entities)
                relation_data = [
                    {
                        "head_entity": relation.head_entity.name if relation.head_entity else "",
                        "tail_entity": relation.tail_entity.name if relation.tail_entity else "",
                        "relation_type": get_type_value(relation.relation_type),
                        "description": relation.description,
                        "properties": relation.properties,
                        "confidence": relation.confidence,
                    }
                    for relation in relations
                ]
                all_relations.extend(relation_data)

            # 构建图谱
            entity_mapping = LLMGraphUtils.build_entities_from_data(graph, all_entities)
            LLMGraphUtils.build_relations_from_data(graph, all_relations, entity_mapping)

            logger.info(f"Minimal LLM graph built: {len(graph.entities)} entities, {len(graph.relations)} relations")
            return graph

        except Exception as e:
            logger.error(f"Error building minimal LLM graph: {e}")
            raise


class FlexibleLLMGraphBuilder(UpdatableGraphBuilder):
    """
    灵活的LLM图构建器 - 支持构建和更新

    适用于需要更新图谱但不需要合并、验证等高级功能的客户端。
    """

    def __init__(
        self,
        openai_api_key: str,
        openai_api_base: Optional[str] = None,
        llm_model: str = "gpt-4o-mini",
        embedding_model: str = "text-embedding-3-small",
        max_tokens: int = 4000,
        temperature: float = 0.1,
        vector_storage: Optional[VectorStorage] = None,
        entity_extractor: Optional[LLMEntityExtractor] = None,
        relation_extractor: Optional[LLMRelationExtractor] = None,
        max_concurrent: int = 10,
    ):
        """
        初始化灵活LLM图构建器

        Args:
            openai_api_key: OpenAI API密钥
            openai_api_base: OpenAI API基础URL
            llm_model: 使用的LLM模型
            embedding_model: 使用的嵌入模型
            max_tokens: 最大token数
            temperature: 温度参数
            vector_storage: 向量存储后端
            entity_extractor: 实体提取器
            relation_extractor: 关系提取器
            max_concurrent: 最大并发数
        """
        self.entity_extractor = entity_extractor or LLMEntityExtractor(
            openai_api_key=openai_api_key,
            openai_api_base=openai_api_base,
            llm_model=llm_model,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        self.relation_extractor = relation_extractor or LLMRelationExtractor(
            openai_api_key=openai_api_key,
            openai_api_base=openai_api_base,
            llm_model=llm_model,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        # 初始化使用统计跟踪器
        self.usage_tracker = LLMUsageTracker(llm_model, embedding_model)

        # 初始化异步处理器
        self.async_processor = LLMAsyncProcessor(
            self.entity_extractor,
            self.relation_extractor,
            self.usage_tracker,
            max_concurrent,
        )

        # 初始化向量嵌入（可选）
        self.vector_storage = vector_storage or JsonVectorStorage("flexible_llm_graph_vectors.json")
        self.graph_embedding = OpenAIEmbedding(
            vector_storage=self.vector_storage,
            openai_api_key=openai_api_key,
            openai_api_base=openai_api_base,
            embedding_model=embedding_model,
        )

    async def build_graph(
        self,
        texts: Optional[List[str]] = None,
        database_schema: Optional[Dict[str, Any]] = None,
        graph_name: str = "flexible_llm_graph",
    ) -> KnowledgeGraph:
        """异步构建知识图谱"""
        if not texts:
            return KnowledgeGraph(name=graph_name)

        logger.info("Starting flexible LLM-based knowledge graph construction")

        try:
            graph = KnowledgeGraph(name=graph_name)

            # 更新会话统计
            self.usage_tracker.update_session_stats(len(texts), 0, 0)

            # 异步处理文本
            all_entities, all_relations = await self.async_processor.process_texts_async(texts)

            # 去重实体
            deduplicated_entities = await self.async_processor.deduplicate_entities(all_entities)

            # 构建图谱
            entity_mapping = LLMGraphUtils.build_entities_from_data(graph, deduplicated_entities)
            relations_added = LLMGraphUtils.build_relations_from_data(graph, all_relations, entity_mapping)

            # 更新统计
            self.usage_tracker.update_session_stats(0, len(deduplicated_entities), relations_added)

            # 构建嵌入（可选）
            if self.graph_embedding:
                await self._build_embeddings(graph)

            logger.info(f"Flexible LLM graph built: {len(graph.entities)} entities, {len(graph.relations)} relations")
            return graph

        except Exception as e:
            logger.error(f"Error building flexible LLM graph: {e}")
            raise

    async def update_graph(
        self,
        graph: KnowledgeGraph,
        new_entities: Optional[List[Entity]] = None,
        new_relations: Optional[List[Relation]] = None,
    ) -> KnowledgeGraph:
        """更新现有图谱"""
        logger.info("Updating flexible LLM graph")

        try:
            if new_entities:
                for entity in new_entities:
                    if entity.id not in graph.entities:
                        graph.add_entity(entity)

            if new_relations:
                for relation in new_relations:
                    if (
                        relation.head_entity
                        and relation.head_entity.id in graph.entities
                        and relation.tail_entity
                        and relation.tail_entity.id in graph.entities
                        and relation.id not in graph.relations
                    ):
                        graph.add_relation(relation)

            # 更新嵌入
            if self.graph_embedding:
                await self._build_embeddings(graph)

            logger.info(f"Flexible LLM graph updated: {len(graph.entities)} entities, {len(graph.relations)} relations")
            return graph

        except Exception as e:
            logger.error(f"Error updating flexible LLM graph: {e}")
            raise

    async def update_graph_with_texts(
        self,
        graph: KnowledgeGraph,
        texts: List[str],
    ) -> KnowledgeGraph:
        """使用新文本更新图谱"""
        if not texts:
            return graph

        logger.info(f"Updating graph with {len(texts)} new texts")

        # 处理新文本
        all_entities, all_relations = await self.async_processor.process_texts_async(texts)

        # 与现有实体进行匹配和去重
        new_entities = []
        for entity_data in all_entities:
            existing_entity = LLMGraphUtils.find_similar_entity(graph, entity_data)
            if existing_entity:
                LLMGraphUtils.merge_entity_data(existing_entity, entity_data)
            else:
                new_entities.append(entity_data)

        # 添加新实体
        entity_mapping = {entity.name: entity.id for entity in graph.entities.values()}
        new_entity_mapping = LLMGraphUtils.build_entities_from_data(graph, new_entities)
        entity_mapping.update(new_entity_mapping)

        # 添加新关系
        relations_added = LLMGraphUtils.build_relations_from_data(graph, all_relations, entity_mapping)

        # 更新嵌入
        if self.graph_embedding:
            await self._build_embeddings(graph)

        logger.info(f"Graph updated with texts: {len(new_entities)} new entities, {relations_added} new relations")
        return graph

    async def _build_embeddings(self, graph: KnowledgeGraph) -> None:
        """构建图谱嵌入"""
        try:
            await self.graph_embedding.build_text_embeddings(graph)
            self.usage_tracker.track_embedding_call("entity_embedding", success=True)
            self.usage_tracker.track_embedding_call("relation_embedding", success=True)
            self.graph_embedding.save_embeddings()
        except Exception as e:
            logger.error(f"Error building embeddings: {e}")
            self.usage_tracker.track_embedding_call("entity_embedding", success=False, error_msg=str(e))

    def get_usage_statistics(self) -> Dict[str, Any]:
        """获取使用统计信息"""
        return self.usage_tracker.get_usage_statistics()

    def print_usage_summary(self) -> None:
        """打印使用统计摘要"""
        self.usage_tracker.print_usage_summary()

    def cleanup(self) -> None:
        """清理资源"""
        self.usage_tracker.cleanup()
        if self.graph_embedding:
            self.graph_embedding.save_embeddings()


class LLMGraphBuilder(
    GraphMergerMixin,
    GraphValidatorMixin,
    GraphExporterMixin,
    GraphStatisticsMixin,
    FullFeaturedGraphBuilder,
):
    """
    全功能LLM图构建器 - 包含所有功能

    只有需要所有功能的客户端才应该使用这个类。
    大多数客户端应该使用更专注的接口。
    """

    def __init__(
        self,
        openai_api_key: str,
        openai_api_base: Optional[str] = None,
        llm_model: str = "gpt-4o-mini",
        embedding_model: str = "text-embedding-3-small",
        max_tokens: int = 4000,
        temperature: float = 0.1,
        vector_storage: Optional[VectorStorage] = None,
        entity_extractor: Optional[LLMEntityExtractor] = None,
        relation_extractor: Optional[LLMRelationExtractor] = None,
        max_concurrent: int = 10,
    ):
        """初始化全功能LLM图构建器"""
        super().__init__()

        # 使用组合而不是继承来复用FlexibleLLMGraphBuilder的功能
        self.flexible_builder = FlexibleLLMGraphBuilder(
            openai_api_key=openai_api_key,
            openai_api_base=openai_api_base,
            llm_model=llm_model,
            embedding_model=embedding_model,
            max_tokens=max_tokens,
            temperature=temperature,
            vector_storage=vector_storage,
            entity_extractor=entity_extractor,
            relation_extractor=relation_extractor,
            max_concurrent=max_concurrent,
        )

    async def build_graph(
        self,
        texts: Optional[List[str]] = None,
        database_schema: Optional[Dict[str, Any]] = None,
        graph_name: str = "comprehensive_llm_graph",
    ) -> KnowledgeGraph:
        """构建全功能图谱"""
        # 委托给灵活构建器
        graph = await self.flexible_builder.build_graph(texts, database_schema, graph_name)

        # 执行验证
        validation_result = await self.validate_graph(graph)
        if not validation_result.get("valid", True):
            logger.warning(f"Graph validation issues: {validation_result.get('issues', [])}")
        with open(os.path.join(settings.workdir, "graph.json"), "w") as f:
            json.dump(graph.to_dict(), f, ensure_ascii=False, indent=2)
        self.flexible_builder.vector_storage.save()
        return graph

    async def update_graph(
        self,
        graph: KnowledgeGraph,
        new_entities: Optional[List[Entity]] = None,
        new_relations: Optional[List[Relation]] = None,
    ) -> KnowledgeGraph:
        """更新图谱并验证"""
        # 委托给灵活构建器
        updated_graph = await self.flexible_builder.update_graph(graph, new_entities, new_relations)

        # 执行验证
        validation_result = await self.validate_graph(updated_graph)
        if not validation_result.get("valid", True):
            logger.warning(f"Graph validation issues: {validation_result.get('issues', [])}")

        return updated_graph

    def get_usage_statistics(self) -> Dict[str, Any]:
        """获取使用统计信息"""
        return self.flexible_builder.get_usage_statistics()

    def print_usage_summary(self) -> None:
        """打印使用统计摘要"""
        self.flexible_builder.print_usage_summary()

    def cleanup(self) -> None:
        """清理资源"""
        self.flexible_builder.cleanup()


class StreamingLLMGraphBuilder(StreamingGraphBuilder, IncrementalBuilderMixin, GraphStatisticsMixin):
    """
    流式LLM图构建器 - 适用于实时增量更新

    专为需要实时处理文档流而不需要合并或验证功能的应用设计。
    """

    def __init__(
        self,
        openai_api_key: str,
        openai_api_base: Optional[str] = None,
        llm_model: str = "gpt-4o-mini",
        max_tokens: int = 4000,
        temperature: float = 0.1,
        entity_extractor: Optional[LLMEntityExtractor] = None,
        relation_extractor: Optional[LLMRelationExtractor] = None,
        max_concurrent: int = 5,  # 流式处理使用较小的并发数
    ):
        """初始化流式LLM图构建器"""
        super().__init__()

        self.entity_extractor = entity_extractor or LLMEntityExtractor(
            openai_api_key=openai_api_key,
            openai_api_base=openai_api_base,
            llm_model=llm_model,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        self.relation_extractor = relation_extractor or LLMRelationExtractor(
            openai_api_key=openai_api_key,
            openai_api_base=openai_api_base,
            llm_model=llm_model,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        self.usage_tracker = LLMUsageTracker(llm_model, "")
        self.async_processor = LLMAsyncProcessor(
            self.entity_extractor,
            self.relation_extractor,
            self.usage_tracker,
            max_concurrent,
        )

    async def build_graph(
        self,
        texts: Optional[List[str]] = None,
        database_schema: Optional[Dict[str, Any]] = None,
        graph_name: str = "streaming_llm_graph",
    ) -> KnowledgeGraph:
        """构建初始图谱用于流式处理"""
        if not texts:
            graph = KnowledgeGraph(name=graph_name)
        else:
            # 使用最小构建器创建初始图谱
            minimal_builder = MinimalLLMGraphBuilder(
                openai_api_key=self.entity_extractor.openai_client.api_key,
                openai_api_base=(
                    str(self.entity_extractor.openai_client.base_url)
                    if self.entity_extractor.openai_client.base_url
                    else None
                ),
                llm_model=self.entity_extractor.llm_model,
                entity_extractor=self.entity_extractor,
                relation_extractor=self.relation_extractor,
            )
            graph = await minimal_builder.build_graph(texts, database_schema, graph_name)

        self._current_graph = graph
        return graph

    async def add_documents_async(
        self, documents: List[str], document_ids: Optional[List[str]] = None
    ) -> KnowledgeGraph:
        """异步添加文档到流式图谱"""
        if document_ids is None:
            document_ids = [f"doc_{i}_{datetime.now().timestamp()}" for i in range(len(documents))]

        if len(document_ids) != len(documents):
            raise ValueError("Number of document IDs must match number of documents")

        logger.info(f"Adding {len(documents)} documents to streaming graph")

        try:
            # 处理新文档
            all_entities, all_relations = await self.async_processor.process_texts_async(documents)

            # 简单去重（流式处理避免复杂去重）
            simple_dedup_entities = self._simple_entity_dedup(all_entities)

            # 确保当前图谱存在
            if self._current_graph is None:
                raise ValueError("Current graph is not initialized. Call build_initial_graph first.")

            # 添加到当前图谱
            entity_mapping = {entity.name: entity.id for entity in self._current_graph.entities.values()}

            # 构建新实体
            new_entities = []
            for entity_data in simple_dedup_entities:
                if entity_data["name"] not in entity_mapping:
                    new_entities.append(entity_data)

            new_entity_mapping = LLMGraphUtils.build_entities_from_data(self._current_graph, new_entities)
            entity_mapping.update(new_entity_mapping)

            # 构建新关系
            relations_added = LLMGraphUtils.build_relations_from_data(
                self._current_graph, all_relations, entity_mapping
            )

            # 记录文档映射
            for doc_id in document_ids:
                self._document_registry[doc_id] = list(new_entity_mapping.values())

            logger.info(f"Added {len(new_entities)} entities and {relations_added} relations from streaming documents")
            return self._current_graph

        except Exception as e:
            logger.error(f"Error adding documents to streaming graph: {e}")
            raise

    def _simple_entity_dedup(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """简单的实体去重 - 基于名称和类型"""
        seen = set()
        deduplicated = []

        for entity_data in entities:
            key = (entity_data["name"].lower(), entity_data["type"])
            if key not in seen:
                seen.add(key)
                deduplicated.append(entity_data)

        return deduplicated

    async def add_documents(self, documents: List[str], document_ids: Optional[List[str]] = None) -> KnowledgeGraph:
        """异步版本的添加文档方法 - 实现IncrementalBuilder接口"""
        return await self.add_documents_async(documents, document_ids)

    async def remove_documents(self, document_ids: List[str]) -> KnowledgeGraph:
        """移除文档方法 - 实现IncrementalBuilder接口"""
        if self._current_graph is None:
            raise ValueError("No graph exists to remove documents from")

        try:
            entities_to_remove = set()

            # 收集要移除的文档的实体
            for doc_id in document_ids:
                if doc_id in self._document_registry:
                    entities_to_remove.update(self._document_registry[doc_id])
                    del self._document_registry[doc_id]

            # 移除实体（这也会移除相关关系）
            for entity_id in entities_to_remove:
                if entity_id in self._current_graph.entities:
                    self._current_graph.remove_entity(entity_id)

            return self._current_graph

        except Exception as e:
            logger.error(f"Error removing documents: {e}")
            raise

    def get_usage_statistics(self) -> Dict[str, Any]:
        """获取使用统计信息"""
        return self.usage_tracker.get_usage_statistics()


class BatchLLMGraphBuilder(GraphMergerMixin, BatchGraphBuilder):
    """
    批量LLM图构建器 - 优化批量处理多个数据源

    适用于需要处理多个数据源并合并它们，但不需要增量更新或验证的场景。
    """

    def __init__(
        self,
        openai_api_key: str,
        openai_api_base: Optional[str] = None,
        llm_model: str = "gpt-4o-mini",
        embedding_model: str = "text-embedding-3-small",
        max_tokens: int = 4000,
        temperature: float = 0.1,
        vector_storage: Optional[VectorStorage] = None,
        entity_extractor: Optional[LLMEntityExtractor] = None,
        relation_extractor: Optional[LLMRelationExtractor] = None,
        max_concurrent: int = 15,  # 批量处理使用更高的并发数
    ):
        """初始化批量LLM图构建器"""
        super().__init__()

        # 使用组合复用FlexibleLLMGraphBuilder
        self.flexible_builder = FlexibleLLMGraphBuilder(
            openai_api_key=openai_api_key,
            openai_api_base=openai_api_base,
            llm_model=llm_model,
            embedding_model=embedding_model,
            max_tokens=max_tokens,
            temperature=temperature,
            vector_storage=vector_storage,
            entity_extractor=entity_extractor,
            relation_extractor=relation_extractor,
            max_concurrent=max_concurrent,
        )

    async def build_graph(
        self,
        texts: Optional[List[str]] = None,
        database_schema: Optional[Dict[str, Any]] = None,
        graph_name: str = "batch_llm_graph",
    ) -> KnowledgeGraph:
        """构建批量图谱"""
        return await self.flexible_builder.build_graph(texts, database_schema, graph_name)

    async def build_from_multiple_sources(
        self, sources: List[Dict[str, Any]], graph_name: str = "multi_source_batch_llm_graph"
    ) -> KnowledgeGraph:
        """从多个异构数据源构建图谱"""
        sub_graphs = []

        # 并行处理多个数据源
        tasks = []
        for i, source in enumerate(sources):
            source_type = source.get("type")
            source_data = source.get("data")
            source_name = f"{graph_name}_source_{i}"

            if source_type == "text":
                texts = source_data if isinstance(source_data, list) else [source_data]
                task = self.flexible_builder.build_graph(texts=texts, graph_name=source_name)
                tasks.append(task)
            elif source_type == "mixed":
                if source_data is not None:
                    texts = source_data.get("texts", [])
                else:
                    texts = []
                task = self.flexible_builder.build_graph(texts=texts, graph_name=source_name)
                tasks.append(task)
            else:
                logger.warning(f"Unknown source type: {source_type}")

        if not tasks:
            return KnowledgeGraph(name=graph_name)

        # 并行执行所有构建任务
        logger.info(f"Building {len(tasks)} sub-graphs in parallel")
        sub_graphs = await asyncio.gather(*tasks, return_exceptions=True)

        # 过滤异常结果
        valid_graphs: List[KnowledgeGraph] = []
        for result in sub_graphs:
            if isinstance(result, Exception):
                logger.error(f"Error building sub-graph: {result}")
            elif isinstance(result, KnowledgeGraph):
                valid_graphs.append(result)

        if not valid_graphs:
            return KnowledgeGraph(name=graph_name)

        # 合并所有子图
        logger.info(f"Merging {len(valid_graphs)} sub-graphs")
        merged_graph = await self.merge_graphs(valid_graphs)
        merged_graph.name = graph_name
        return merged_graph

    def get_usage_statistics(self) -> Dict[str, Any]:
        """获取使用统计信息"""
        return self.flexible_builder.get_usage_statistics()

    def cleanup(self) -> None:
        """清理资源"""
        self.flexible_builder.cleanup()


# 导出所有公共类和函数
__all__ = [
    "LLMUsageTracker",
    "LLMAsyncProcessor",
    "LLMGraphUtils",
    "MinimalLLMGraphBuilder",
    "FlexibleLLMGraphBuilder",
    "LLMGraphBuilder",
    "StreamingLLMGraphBuilder",
    "BatchLLMGraphBuilder",
]
