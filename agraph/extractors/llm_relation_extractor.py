"""
LLM-based relation extractor module
"""

import asyncio
import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI

from ..entities import Entity
from ..relations import Relation
from ..types import RelationType
from .relation_extractor import BaseRelationExtractor

logger = logging.getLogger(__name__)


class LLMRelationExtractor(BaseRelationExtractor):
    """LLM-based relation extractor"""

    def __init__(
        self,
        openai_api_key: str,
        openai_api_base: Optional[str] = None,
        llm_model: str = "gpt-4o-mini",
        max_tokens: int = 4000,
        temperature: float = 0.1,
    ):
        """
        Initialize LLM relation extractor

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

        # Relation extraction prompt
        self.relation_extraction_prompt = """
你是一个专业的知识图谱构建专家。请从以下文本中提取实体间的关系。

文本内容：
{text}

已识别的实体：
{entities}

请按照以下JSON格式返回提取的关系：
{{
    "relations": [
        {{
            "head_entity": "头实体名称",
            "tail_entity": "尾实体名称",
            "relation_type": "关系类型(BELONGS_TO/LOCATED_IN/WORKS_FOR/RELATED_TO/CAUSES/PART_OF/IS_A/OTHER)",
            "description": "关系描述",
            "properties": {{"属性名": "属性值"}},
            "confidence": 0.9
        }}
    ]
}}

要求：
1. 只提取文本中明确表述的关系
2. 头实体和尾实体必须在已识别实体列表中
3. 选择最适合的关系类型
4. 提供关系的简洁描述
5. 置信度范围0-1，表示关系的确定程度
6. 只返回JSON格式，不要包含其他文本
"""

    def extract_from_text(self, text: str, entities: List[Entity]) -> List[Relation]:
        """
        从文本中提取关系

        Args:
            text: 输入文本
            entities: 已识别的实体列表

        Returns:
            List[Relation]: 提取的关系列表
        """
        # Use asyncio to run the async method
        return asyncio.run(self._extract_relations_async(text, entities))

    async def _extract_relations_async(self, text: str, entities: List[Entity]) -> List[Relation]:
        """Async version of relation extraction"""
        try:
            relations_data = await self._extract_relations_llm(text, entities)
            relations = []

            # Create entity mapping for quick lookup
            entity_mapping = {entity.name: entity for entity in entities}

            # Also create mapping with lowercase names for fuzzy matching
            entity_mapping_lower = {entity.name.lower(): entity for entity in entities}

            for relation_data in relations_data:
                head_name = relation_data["head_entity"]
                tail_name = relation_data["tail_entity"]

                # Find entities by name (exact match first, then fuzzy)
                head_entity = self._find_entity_by_name(head_name, entity_mapping, entity_mapping_lower)
                tail_entity = self._find_entity_by_name(tail_name, entity_mapping, entity_mapping_lower)

                if head_entity and tail_entity:
                    relation = Relation(
                        id=self._generate_relation_id(head_name, tail_name, relation_data["relation_type"]),
                        head_entity=head_entity,
                        tail_entity=tail_entity,
                        relation_type=self._normalize_relation_type(relation_data["relation_type"]),
                        description=relation_data.get("description", ""),
                        properties=relation_data.get("properties", {}),
                        confidence=relation_data.get("confidence", 1.0),
                        source="llm_extraction",
                        created_at=datetime.now(),
                        updated_at=datetime.now(),
                    )

                    if self.validate_relation(relation):
                        relations.append(relation)

            return relations

        except Exception as e:
            logger.error(f"Error extracting relations from text: {e}")
            return []

    async def _extract_relations_llm(self, text: str, entities: List[Entity]) -> List[Dict[str, Any]]:
        """Use LLM to extract relations from text"""
        try:
            entities_str = "\n".join([f"- {entity.name} ({entity.entity_type.value})" for entity in entities])
            prompt = self.relation_extraction_prompt.format(text=text, entities=entities_str)

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
                relations = result.get("relations", [])
                return relations if isinstance(relations, list) else []
            except json.JSONDecodeError:
                # Try to extract JSON part
                json_match = re.search(r"\{.*\}", content, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                    relations = result.get("relations", [])
                    return relations if isinstance(relations, list) else []
                return []

        except Exception as e:
            logger.error(f"Error extracting relations with LLM: {e}")
            return []

    def _find_entity_by_name(
        self, name: str, entity_mapping: Dict[str, Entity], entity_mapping_lower: Dict[str, Entity]
    ) -> Optional[Entity]:
        """Find entity by name with exact and fuzzy matching"""
        # Exact match
        if name in entity_mapping:
            return entity_mapping[name]

        # Lowercase match
        name_lower = name.lower()
        if name_lower in entity_mapping_lower:
            return entity_mapping_lower[name_lower]

        # Fuzzy match - check if name is contained in entity names or aliases
        for entity in entity_mapping.values():
            if name_lower in entity.name.lower() or entity.name.lower() in name_lower:
                return entity

            # Check aliases
            for alias in entity.aliases:
                if name_lower in alias.lower() or alias.lower() in name_lower:
                    return entity

        return None

    def extract_from_database(self, schema: Dict[str, Any], entities: List[Entity]) -> List[Relation]:
        """
        LLM relation extractor does not handle database schemas
        Use DatabaseRelationExtractor for database schema extraction
        """
        return []

    def _normalize_relation_type(self, relation_type: str) -> Any:
        """
        Normalize relation type string to RelationType enum value

        Args:
            relation_type: Relation type string from LLM (case-insensitive)

        Returns:
            RelationType: Normalized relation type
        """
        # Mapping from common LLM output variations to RelationType values
        type_mapping = {
            "BELONGS_TO": RelationType.BELONGS_TO,
            "LOCATED_IN": RelationType.LOCATED_IN,
            "WORKS_FOR": RelationType.WORKS_FOR,
            "CAUSES": RelationType.CAUSES,
            "PART_OF": RelationType.PART_OF,
            "IS_A": RelationType.IS_A,
            "CONTAINS": RelationType.CONTAINS,
            "REFERENCES": RelationType.REFERENCES,
            "SIMILAR_TO": RelationType.SIMILAR_TO,
            "RELATED_TO": RelationType.RELATED_TO,
            "DEPENDS_ON": RelationType.DEPENDS_ON,
            "FOREIGN_KEY": RelationType.FOREIGN_KEY,
            "MENTIONS": RelationType.MENTIONS,
            "DESCRIBES": RelationType.DESCRIBES,
            "SYNONYMS": RelationType.SYNONYMS,
            "DEVELOPS": RelationType.DEVELOPS,
            "CREATES": RelationType.CREATES,
            "FOUNDED_BY": RelationType.FOUNDED_BY,
            "OTHER": RelationType.OTHER,
            # Common variations
            "HAS": RelationType.CONTAINS,
            "INCLUDES": RelationType.CONTAINS,
            "MEMBER_OF": RelationType.BELONGS_TO,
            "EMPLOYED_BY": RelationType.WORKS_FOR,
            "BASED_IN": RelationType.LOCATED_IN,
            "SAME_AS": RelationType.SIMILAR_TO,
            "ASSOCIATED_WITH": RelationType.RELATED_TO,
        }

        # Normalize to uppercase for lookup
        normalized_type = relation_type.strip().upper()

        # Try direct mapping first
        if normalized_type in type_mapping:
            return type_mapping[normalized_type]

        # Try to match against enum values (case-insensitive)
        for enum_value in RelationType:
            if enum_value.value.upper() == normalized_type:
                return enum_value

        # Fallback to OTHER for unrecognized types
        logger.warning(f"Unknown relation type '{relation_type}', falling back to OTHER")
        return RelationType.OTHER

    def _generate_relation_id(self, head: str, tail: str, relation_type: str) -> str:
        """Generate relation ID"""
        import hashlib

        relation_str = f"{head}_{relation_type}_{tail}"
        return f"relation_{hashlib.md5(relation_str.encode()).hexdigest()[:8]}"
