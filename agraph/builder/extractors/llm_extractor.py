"""
LLM-based entity and relation extractors.
"""

import asyncio
import json
import re
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI

from ...base.entities import Entity
from ...base.relations import Relation
from ...base.text import TextChunk
from ...base.types import EntityType, RelationType
from ...config import get_settings
from ...logger import logger
from .base import EntityExtractor, RelationExtractor


class LLMEntityExtractor(EntityExtractor):
    """LLM-based entity extractor."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize LLM entity extractor.

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.settings = get_settings()
        self.llm_provider = (
            config.get("llm_provider", self.settings.llm.provider)
            if config
            else self.settings.llm.provider
        )
        self.llm_model = (
            config.get("llm_model", self.settings.llm.model) if config else self.settings.llm.model
        )
        self.batch_size = config.get("batch_size", 5) if config else 5

        # Initialize AsyncOpenAI client
        self.client = AsyncOpenAI(
            api_key=self.settings.openai.api_key, base_url=self.settings.openai.api_base
        )

    async def extract(self, chunks: List[TextChunk]) -> List[Entity]:
        """Extract entities from text chunks."""
        all_entities = []

        # Process chunks in batches
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i : i + self.batch_size]
            batch_entities = await self._extract_batch(batch)
            all_entities.extend(batch_entities)

        return self.post_process_entities(all_entities)

    async def extract_from_text(self, text: str, chunk_id: Optional[str] = None) -> List[Entity]:
        """Extract entities from raw text."""
        # Create temporary text chunk
        temp_chunk = TextChunk(
            id=chunk_id or "temp_chunk",
            content=text,
            title="Temporary Chunk",
            start_index=0,
            end_index=len(text),
        )

        return await self._extract_from_chunk(temp_chunk)

    async def _extract_batch(self, chunks: List[TextChunk]) -> List[Entity]:
        """Extract entities from a batch of chunks."""
        tasks = []
        for chunk in chunks:
            tasks.append(self._extract_from_chunk(chunk))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_entities: List[Entity] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error extracting entities from chunk {chunks[i].id}: {result}")
                continue
            if isinstance(result, list):
                all_entities.extend(result)

        return all_entities

    async def _extract_from_chunk(self, chunk: TextChunk) -> List[Entity]:
        """Extract entities from a single chunk."""
        prompt = self._build_entity_prompt(chunk.content)

        try:
            response = await self._call_llm(prompt)
            entities = self._parse_entity_response(response, chunk.id)
            return entities
        except Exception as e:
            logger.error(f"Error calling LLM for entity extraction: {e}")
            return []

    def _build_entity_prompt(self, text: str) -> str:
        """Build prompt for entity extraction."""
        entity_types = [et.value for et in EntityType]

        prompt = f"""
Extract named entities from the following text. Focus on identifying significant entities that would be useful in a knowledge graph.

Entity Types (choose the most appropriate):
{', '.join(entity_types)}

Text to analyze:
{text}

For each entity, provide:
- name: The exact entity name as it appears in the text
- type: One of the entity types listed above
- description: A concise description explaining what this entity represents
- confidence: Confidence score between 0.0 and 1.0 (only include entities with confidence >= 0.7)

Return ONLY a valid JSON array with no additional text:
[
  {{
    "name": "entity name",
    "type": "entity_type",
    "description": "brief description",
    "confidence": 0.9
  }}
]

Requirements:
- Only extract clearly identifiable entities
- Avoid generic terms or common words
- Ensure entity names are properly capitalized
- Return empty array [] if no valid entities found
"""
        return prompt.strip()

    def _parse_entity_response(self, response: str, chunk_id: str) -> List[Entity]:
        """Parse LLM response into Entity objects."""
        entities: List[Entity] = []

        try:
            # Try to parse as direct JSON first
            try:
                entity_data = json.loads(response.strip())
            except json.JSONDecodeError:
                # Fallback: Extract JSON from response using regex
                json_match = re.search(r"\[.*\]", response, re.DOTALL)
                if not json_match:
                    logger.warning(f"No JSON array found in entity response: {response[:100]}...")
                    return entities
                json_str = json_match.group(0)
                entity_data = json.loads(json_str)

            if not isinstance(entity_data, list):
                logger.warning(f"Entity response is not a list: {type(entity_data)}")
                return entities

            for item in entity_data:
                if not isinstance(item, dict):
                    continue

                # Validate required fields
                name = item.get("name", "").strip()
                if not name:
                    continue

                # Parse entity type
                entity_type_str = item.get("type", "concept")
                try:
                    entity_type = EntityType(entity_type_str.lower())
                except ValueError:
                    logger.warning(f"Unknown entity type '{entity_type_str}', using 'concept'")
                    entity_type = EntityType.CONCEPT

                # Validate confidence
                confidence = float(item.get("confidence", 0.7))
                if confidence < 0.7:  # Skip low confidence entities
                    continue

                entity = Entity(
                    name=name,
                    entity_type=entity_type,
                    description=item.get("description", ""),
                    confidence=confidence,
                    text_chunks={chunk_id},
                )

                entities.append(entity)

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Error parsing entity response: {e}")

        return entities

    async def _call_llm(self, prompt: str) -> str:
        """Call LLM with the given prompt."""
        try:
            response = await self.client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert entity extraction assistant. Return only valid JSON format as requested.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self.settings.llm.temperature,
                max_tokens=4096,
            )

            return response.choices[0].message.content or ""

        except Exception as e:
            logger.error(f"Error calling LLM API: {e}")
            raise


class LLMRelationExtractor(RelationExtractor):
    """LLM-based relation extractor."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize LLM relation extractor.

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.settings = get_settings()
        self.llm_provider = (
            config.get("llm_provider", self.settings.llm.provider)
            if config
            else self.settings.llm.provider
        )
        self.llm_model = (
            config.get("llm_model", self.settings.llm.model) if config else self.settings.llm.model
        )
        self.batch_size = config.get("batch_size", 5) if config else 5

        # Initialize AsyncOpenAI client
        self.client = AsyncOpenAI(
            api_key=self.settings.openai.api_key, base_url=self.settings.openai.api_base
        )

    async def extract(self, chunks: List[TextChunk], entities: List[Entity]) -> List[Relation]:
        """Extract relations from text chunks and entities."""
        all_relations: List[Relation] = []

        # Process chunks in batches
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i : i + self.batch_size]
            batch_relations = await self._extract_batch(batch, entities)
            all_relations.extend(batch_relations)

        return self.post_process_relations(all_relations)

    async def extract_from_text(
        self, text: str, entities: List[Entity], chunk_id: Optional[str] = None
    ) -> List[Relation]:
        """Extract relations from raw text and entities."""
        # Create temporary text chunk
        temp_chunk = TextChunk(
            id=chunk_id or "temp_chunk",
            content=text,
            title="Temporary Chunk",
            start_index=0,
            end_index=len(text),
        )

        return await self._extract_from_chunk(temp_chunk, entities)

    async def _extract_batch(
        self, chunks: List[TextChunk], entities: List[Entity]
    ) -> List[Relation]:
        """Extract relations from a batch of chunks."""
        tasks = []
        for chunk in chunks:
            # Find entities mentioned in this chunk
            chunk_entities = self.find_entities_in_text(chunk.content, entities)
            if len(chunk_entities) >= 2:  # Need at least 2 entities to form relations
                tasks.append(self._extract_from_chunk(chunk, chunk_entities))

        if not tasks:
            return []

        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_relations: List[Relation] = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Error extracting relations from chunk: {result}")
                continue
            if isinstance(result, list):
                all_relations.extend(result)

        return all_relations

    async def _extract_from_chunk(self, chunk: TextChunk, entities: List[Entity]) -> List[Relation]:
        """Extract relations from a single chunk."""
        if len(entities) < 2:
            return []

        prompt = self._build_relation_prompt(chunk.content, entities)

        try:
            response = await self._call_llm(prompt)
            relations = self._parse_relation_response(response, entities, chunk.id)
            return relations
        except Exception as e:
            logger.error(f"Error calling LLM for relation extraction: {e}")
            return []

    def _build_relation_prompt(self, text: str, entities: List[Entity]) -> str:
        """Build prompt for relation extraction."""
        relation_types = [rt.value for rt in RelationType]
        entity_details = [
            f"- {e.name} ({e.entity_type.value if hasattr(e.entity_type, 'value') else e.entity_type})"
            for e in entities
        ]

        prompt = f"""
Analyze the text below and identify relationships between the provided entities. Focus on relationships that are explicitly mentioned or clearly implied.

Available Entities:
{chr(10).join(entity_details)}

Relation Types (choose the most appropriate):
{', '.join(relation_types)}

Text to analyze:
{text}

For each relationship found, provide:
- head_entity: Name of the source/subject entity (must match exactly from entity list)
- tail_entity: Name of the target/object entity (must match exactly from entity list)
- type: Most appropriate relation type from the list above
- description: Brief explanation of how the entities are related in the text
- confidence: Confidence score between 0.0 and 1.0 (only include relations with confidence >= 0.6)

Return ONLY a valid JSON array with no additional text:
[
  {{
    "head_entity": "entity1 name",
    "tail_entity": "entity2 name",
    "type": "relation_type",
    "description": "brief description",
    "confidence": 0.8
  }}
]

Requirements:
- Only extract relationships explicitly mentioned in the text
- Entity names must match exactly from the provided list
- Avoid inferring relationships not supported by the text
- Return empty array [] if no valid relationships found
"""
        return prompt.strip()

    def _parse_relation_response(
        self, response: str, entities: List[Entity], chunk_id: str
    ) -> List[Relation]:
        """Parse LLM response into Relation objects."""
        relations: List[Relation] = []

        # Create entity lookup (both exact match and lowercase)
        entity_lookup = {e.name: e for e in entities}
        entity_lookup_lower = {e.name.lower(): e for e in entities}

        try:
            # Try to parse as direct JSON first
            try:
                relation_data = json.loads(response.strip())
            except json.JSONDecodeError:
                # Fallback: Extract JSON from response using regex
                json_match = re.search(r"\[.*\]", response, re.DOTALL)
                if not json_match:
                    logger.warning(f"No JSON array found in relation response: {response[:100]}...")
                    return relations
                json_str = json_match.group(0)
                relation_data = json.loads(json_str)

            if not isinstance(relation_data, list):
                logger.warning(f"Relation response is not a list: {type(relation_data)}")
                return relations

            for item in relation_data:
                if not isinstance(item, dict):
                    continue

                # Find head and tail entities with exact match first, then lowercase
                head_name = item.get("head_entity", "").strip()
                tail_name = item.get("tail_entity", "").strip()

                if not head_name or not tail_name:
                    continue

                head_entity = entity_lookup.get(head_name) or entity_lookup_lower.get(
                    head_name.lower()
                )
                tail_entity = entity_lookup.get(tail_name) or entity_lookup_lower.get(
                    tail_name.lower()
                )

                if not head_entity or not tail_entity:
                    logger.debug(
                        f"Could not find entities: '{head_name}' or '{tail_name}' in entity list"
                    )
                    continue

                # Skip self-relations
                if head_entity.name == tail_entity.name:
                    continue

                # Validate confidence
                confidence = float(item.get("confidence", 0.7))
                if confidence < 0.6:  # Skip low confidence relations
                    continue

                # Parse relation type
                relation_type_str = item.get("type", "references")
                try:
                    relation_type = RelationType(relation_type_str.lower())
                except ValueError:
                    logger.warning(
                        f"Unknown relation type '{relation_type_str}', using 'references'"
                    )
                    relation_type = RelationType.REFERENCES

                relation = Relation(
                    head_entity=head_entity,
                    tail_entity=tail_entity,
                    relation_type=relation_type,
                    description=item.get("description", ""),
                    confidence=confidence,
                    text_chunks={chunk_id},
                )

                relations.append(relation)

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Error parsing relation response: {e}")

        return relations

    async def _call_llm(self, prompt: str) -> str:
        """Call LLM with the given prompt."""
        try:
            response = await self.client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert relation extraction assistant. Return only valid JSON format as requested.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self.settings.llm.temperature,
                max_tokens=4096,
            )

            return response.choices[0].message.content or ""

        except Exception as e:
            logger.error(f"Error calling LLM API: {e}")
            raise
