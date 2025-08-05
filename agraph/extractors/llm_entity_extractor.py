"""
LLM-based entity extractor module
"""

import asyncio
import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI

from ..entities import Entity
from ..types import EntityType
from .entity_extractor import BaseEntityExtractor

logger = logging.getLogger(__name__)


class LLMEntityExtractor(BaseEntityExtractor):
    """LLM-based entity extractor"""

    def __init__(
        self,
        openai_api_key: str,
        openai_api_base: Optional[str] = None,
        llm_model: str = "gpt-4o-mini",
        max_tokens: int = 4000,
        temperature: float = 0.1,
    ):
        """
        Initialize LLM entity extractor

        Args:
            openai_api_key: OpenAI API key
            openai_api_base: OpenAI API base URL
            llm_model: LLM model to use
            max_tokens: Maximum tokens
            temperature: Temperature parameter
        """
        super().__init__()
        self.openai_client = AsyncOpenAI(api_key=openai_api_key, base_url=openai_api_base)
        self.llm_model = llm_model
        self.max_tokens = max_tokens
        self.temperature = temperature

        # Entity extraction prompt
        self.entity_extraction_prompt = """
你是一个专业的知识图谱构建专家。请从以下文本中提取实体信息。

文本内容：
{text}

请按照以下JSON格式返回提取的实体：
{{
    "entities": [
        {{
            "name": "实体名称",
            "type": "实体类型(PERSON/ORGANIZATION/LOCATION/CONCEPT/EVENT/OTHER)",
            "description": "实体描述",
            "aliases": ["别名1", "别名2"],
            "properties": {{"属性名": "属性值"}}
        }}
    ]
}}

要求：
1. 提取所有重要的实体，包括人名、地名、机构名、概念等
2. 为每个实体分配合适的类型
3. 提供简洁准确的描述
4. 如果有别名或其他称呼，请包含在aliases中
5. 重要属性可以放在properties中
6. 只返回JSON格式，不要包含其他文本
"""

        # Entity deduplication prompt
        self.entity_deduplication_prompt = """
你是一个专业的实体去重专家。请判断以下两个实体是否指向同一个事物。

实体1：
名称：{entity1_name}
类型：{entity1_type}
描述：{entity1_description}
别名：{entity1_aliases}

实体2：
名称：{entity2_name}
类型：{entity2_type}
描述：{entity2_description}
别名：{entity2_aliases}

请按照以下JSON格式返回判断结果：
{{
    "is_duplicate": true/false,
    "confidence": 0.95,
    "reason": "判断理由",
    "merged_entity": {{
        "name": "合并后的实体名称",
        "type": "实体类型",
        "description": "合并后的描述",
        "aliases": ["所有别名"],
        "properties": {{"合并后的属性"}}
    }}
}}

要求：
1. 如果是同一实体，is_duplicate为true，并提供合并后的实体信息
2. 如果不是同一实体，is_duplicate为false
3. 提供判断的置信度和理由
4. 只返回JSON格式，不要包含其他文本
"""

    def extract_from_text(self, text: str, context: Optional[Dict[str, Any]] = None) -> List[Entity]:
        """
        从文本中提取实体

        Args:
            text: 输入文本
            context: 上下文信息

        Returns:
            List[Entity]: 提取的实体列表
        """
        # Use asyncio to run the async method
        return asyncio.run(self._extract_entities_async(text, context))

    async def _extract_entities_async(self, text: str, context: Optional[Dict[str, Any]] = None) -> List[Entity]:
        """Async version of entity extraction"""
        try:
            entities_data = await self._extract_entities_llm(text)
            entities = []

            for entity_data in entities_data:
                entity = Entity(
                    id=self._generate_entity_id(entity_data["name"]),
                    name=entity_data["name"],
                    entity_type=self._normalize_entity_type(entity_data["type"]),
                    description=entity_data.get("description", ""),
                    aliases=entity_data.get("aliases", []),
                    properties=entity_data.get("properties", {}),
                    confidence=self._calculate_entity_confidence(entity_data["name"], context),
                    source="llm_extraction",
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                )
                entities.append(entity)

            return self.deduplicate_entities(entities)

        except Exception as e:
            logger.error(f"Error extracting entities from text: {e}")
            return []

    async def _extract_entities_llm(self, text: str) -> List[Dict[str, Any]]:
        """Use LLM to extract entities from text"""
        try:
            prompt = self.entity_extraction_prompt.format(text=text)

            response = await self.openai_client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )

            content = response.choices[0].message.content
            if not content:
                return []

            # Parse JSON response
            try:
                result = json.loads(content)
                entities = result.get("entities", [])
                return entities if isinstance(entities, list) else []
            except json.JSONDecodeError:
                # Try to extract JSON part
                json_match = re.search(r"\{.*\}", content, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                    entities = result.get("entities", [])
                    return entities if isinstance(entities, list) else []
                return []

        except Exception as e:
            logger.error(f"Error extracting entities with LLM: {e}")
            return []

    async def deduplicate_entities_llm(self, entities: List[Entity]) -> List[Entity]:
        """Use LLM to deduplicate entities"""
        if len(entities) <= 1:
            return entities

        deduplicated = []
        processed_indices = set()

        for i, entity1 in enumerate(entities):
            if i in processed_indices:
                continue

            merged_entity = entity1

            # Compare with subsequent entities
            for j, entity2 in enumerate(entities[i + 1 :], i + 1):
                if j in processed_indices:
                    continue

                is_duplicate = await self._check_entity_duplicate_llm(entity1, entity2)
                if is_duplicate:
                    # Merge entity information
                    merged_entity = self._merge_entities_data(merged_entity, entity2)
                    processed_indices.add(j)

            deduplicated.append(merged_entity)
            processed_indices.add(i)

        return deduplicated

    async def _check_entity_duplicate_llm(self, entity1: Entity, entity2: Entity) -> bool:
        """Use LLM to check if two entities are duplicates"""
        try:
            prompt = self.entity_deduplication_prompt.format(
                entity1_name=entity1.name,
                entity1_type=entity1.entity_type.value,
                entity1_description=entity1.description,
                entity1_aliases=", ".join(entity1.aliases),
                entity2_name=entity2.name,
                entity2_type=entity2.entity_type.value,
                entity2_description=entity2.description,
                entity2_aliases=", ".join(entity2.aliases),
            )

            response = await self.openai_client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )

            content = response.choices[0].message.content
            if not content:
                return False

            # Parse JSON response
            try:
                result = json.loads(content)
                is_duplicate = result.get("is_duplicate", False)
                return bool(is_duplicate) if isinstance(is_duplicate, (bool, int, str)) else False
            except json.JSONDecodeError:
                # Simple name matching as fallback
                return entity1.name.lower() == entity2.name.lower()

        except Exception as e:
            logger.error(f"Error checking entity duplicate: {e}")
            # Simple name matching as fallback
            return entity1.name.lower() == entity2.name.lower()

    def _merge_entities_data(self, entity1: Entity, entity2: Entity) -> Entity:
        """Merge two entities"""
        # Create a new merged entity
        merged_entity = Entity(
            id=entity1.id,
            name=entity1.name,
            entity_type=entity1.entity_type,
            description=entity1.description,
            aliases=entity1.aliases.copy(),
            properties=entity1.properties.copy(),
            confidence=max(entity1.confidence, entity2.confidence),
            source=entity1.source,
            created_at=entity1.created_at,
            updated_at=datetime.now(),
        )

        # Merge descriptions
        if entity2.description and entity2.description not in merged_entity.description:
            if merged_entity.description:
                merged_entity.description = f"{merged_entity.description}; {entity2.description}"
            else:
                merged_entity.description = entity2.description

        # Merge aliases
        for alias in entity2.aliases:
            if alias not in merged_entity.aliases:
                merged_entity.aliases.append(alias)

        # Merge properties
        merged_entity.properties.update(entity2.properties)

        return merged_entity

    def extract_from_database(self, schema: Dict[str, Any]) -> List[Entity]:
        """
        LLM entity extractor does not handle database schemas
        Use DatabaseEntityExtractor for database schema extraction
        """
        return []

    def _normalize_entity_type(self, entity_type: str) -> Any:
        """
        Normalize entity type string to EntityType enum value

        Args:
            entity_type: Entity type string from LLM (case-insensitive)

        Returns:
            EntityType: Normalized entity type
        """
        # Mapping from common LLM output variations to EntityType values
        type_mapping = {
            "PERSON": EntityType.PERSON,
            "PEOPLE": EntityType.PERSON,
            "ORGANIZATION": EntityType.ORGANIZATION,
            "ORG": EntityType.ORGANIZATION,
            "COMPANY": EntityType.ORGANIZATION,
            "LOCATION": EntityType.LOCATION,
            "PLACE": EntityType.LOCATION,
            "CONCEPT": EntityType.CONCEPT,
            "EVENT": EntityType.EVENT,
            "OTHER": EntityType.OTHER,
            "MISC": EntityType.OTHER,
            "TABLE": EntityType.TABLE,
            "COLUMN": EntityType.COLUMN,
            "DATABASE": EntityType.DATABASE,
            "DOCUMENT": EntityType.DOCUMENT,
            "KEYWORD": EntityType.KEYWORD,
            "PRODUCT": EntityType.PRODUCT,
            "SOFTWARE": EntityType.SOFTWARE,
            "UNKNOWN": EntityType.UNKNOWN,
        }

        # Normalize to uppercase for lookup
        normalized_type = entity_type.strip().upper()

        # Try direct mapping first
        if normalized_type in type_mapping:
            return type_mapping[normalized_type]

        # Try to match against enum values (case-insensitive)
        for enum_value in EntityType:
            if enum_value.value.upper() == normalized_type:
                return enum_value

        # Fallback to UNKNOWN for unrecognized types
        logger.warning(f"Unknown entity type '{entity_type}', falling back to UNKNOWN")
        return EntityType.UNKNOWN

    def _generate_entity_id(self, name: str) -> str:
        """Generate entity ID"""
        import hashlib

        return f"entity_{hashlib.md5(name.encode()).hexdigest()[:8]}"
