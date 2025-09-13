"""
Basic configuration classes for AGraph.

This module contains the fundamental configuration classes that define
the core settings for different components of the system.
"""

import os
from typing import List

from pydantic import BaseModel, Field


class OpenAIConfig(BaseModel):
    """OpenAI API configuration."""

    api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    api_base: str = Field(default_factory=lambda: os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"))

    def mask_sensitive_data(self) -> dict:
        """Return configuration with masked sensitive data for safe serialization."""
        data = self.model_dump()
        if data["api_key"]:
            # Show first 3 and last 4 characters, mask the rest
            key = data["api_key"]
            if len(key) > 7:
                data["api_key"] = f"{key[:3]}{'*' * (len(key) - 7)}{key[-4:]}"
            else:
                data["api_key"] = "*" * len(key)
        return data


class LLMConfig(BaseModel):
    """Language model configuration."""

    model: str = Field(default_factory=lambda: os.getenv("LLM_MODEL", "gpt-3.5-turbo"))
    temperature: float = Field(default=0.0)
    max_tokens: int = Field(default_factory=lambda: int(os.getenv("LLM_MAX_TOKENS", "4096")))
    provider: str = Field(default_factory=lambda: os.getenv("LLM_PROVIDER", "openai"))


class EmbeddingConfig(BaseModel):
    """Embedding model configuration."""

    model: str = Field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"))
    provider: str = Field(default_factory=lambda: os.getenv("EMBEDDING_PROVIDER", "openai"))
    dimension: int = Field(default_factory=lambda: int(os.getenv("EMBEDDING_DIM", "1024")))
    max_token_size: int = Field(default_factory=lambda: int(os.getenv("EMBEDDING_MAX_TOKENS", "8192")))
    batch_size: int = Field(default_factory=lambda: int(os.getenv("EMBEDDING_BATCH_SIZE", "32")))


class GraphConfig(BaseModel):
    """Knowledge graph configuration."""

    entity_types: List[str] = Field(
        default_factory=lambda: [
            "person",
            "organization",
            "location",
            "concept",
            "event",
            "other",
            "table",
            "column",
            "database",
            "document",
            "keyword",
            "product",
            "software",
            "unknown",
        ]
    )

    relation_types: List[str] = Field(
        default_factory=lambda: [
            "contains",
            "belongs_to",
            "located_in",
            "works_for",
            "causes",
            "part_of",
            "is_a",
            "references",
            "similar_to",
            "related_to",
            "depends_on",
            "foreign_key",
            "mentions",
            "describes",
            "synonyms",
            "develops",
            "creates",
            "founded_by",
            "other",
        ]
    )


class TextConfig(BaseModel):
    """Text processing configuration."""

    max_chunk_size: int = Field(default_factory=lambda: int(os.getenv("MAX_CHUNK_SIZE", "512")))
    chunk_overlap: int = Field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "100")))


class RAGConfig(BaseModel):
    """RAG system configuration."""

    system_prompt: str = Field(
        default="""# System Role

You are an expert knowledge assistant specializing in information retrieval and synthesis from structured knowledge graphs and document collections.

## Objective

Provide comprehensive, well-structured responses to user queries by synthesizing information from the provided data sources. Your responses must be grounded exclusively in the given data sources while maintaining accuracy, clarity, and proper attribution.

## Data Sources Available

**Knowledge Graph (KG)**: Structured entities, relationships, and semantic connections
**Document Chunks (DC)**: Relevant text segments from documents with contextual information

### Temporal Information Handling
- Each data point includes a `created_at` timestamp indicating knowledge acquisition time
- For conflicting information, evaluate both content relevance and temporal context
- Prioritize content-based temporal information over creation timestamps
- Apply contextual judgment rather than defaulting to most recent information

---

## Conversation Context
{history}

## Available Knowledge Sources
{kg_context}

---

## Response Requirements

### Format and Structure
- **Response Type**: {response_type}
- **Language**: Respond in the same language as the user's question
- **Formatting**: Use markdown with clear section headers and proper structure
- **Continuity**: Maintain coherence with conversation history

### Content Organization
- Structure responses with focused sections addressing distinct aspects
- Use descriptive section headers that clearly indicate content focus
- Present information in logical, easily digestible segments

### Citation System
- **Inline Citations**: Use the format `[ID:reference_number]` immediately after each statement or claim that references data sources
  - Example: `The system processes over 10,000 queries daily [ID:1].`
  - Example: `According to the latest research findings [ID:2], performance improved significantly.`
  - Place citations at the end of sentences or clauses, before punctuation

- **References Section**: Always conclude with a "# References" section containing:
  - Format: `ID:number - [Source_Type] Brief description of the source content`
  - Source type indicators: `[KG]` for Knowledge Graph, `[DC]` for Document Chunks
  - Maximum 5 most relevant references

### Reference Format Template
```
### References
- ID:1 [KG] Entity relationship describing system performance metrics
- ID:2 [DC] Research document excerpt about performance improvements
- ID:3 [KG] Semantic connection between entities showing growth trends
```

### Quality Standards
- **Accuracy**: Base all claims exclusively on provided data sources
- **Transparency**: Clearly distinguish between different source types
- **Completeness**: Address all relevant aspects found in the data sources
- **Honesty**: State limitations clearly when information is insufficient
- **No Fabrication**: Never generate information not present in the provided sources

If the available data sources are insufficient to answer the query, explicitly state this limitation and describe what additional information would be needed."""
    )
