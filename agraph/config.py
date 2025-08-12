import os
from typing import List

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv(".env")


class OpenAIConfig(BaseModel):
    """OpenAI API configuration."""

    api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    api_base: str = Field(
        default_factory=lambda: os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
    )


class LLMConfig(BaseModel):
    """Language model configuration."""

    model: str = Field(default_factory=lambda: os.getenv("LLM_MODEL", "gpt-3.5-turbo"))
    temperature: float = Field(default=0.0)
    max_tokens: int = Field(default_factory=lambda: int(os.getenv("LLM_MAX_TOKENS", "4096")))
    provider: str = Field(default_factory=lambda: os.getenv("LLM_PROVIDER", "openai"))


class EmbeddingConfig(BaseModel):
    """Embedding model configuration."""

    model: str = Field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "Pro/BAAI/bge-m3"))
    provider: str = Field(default_factory=lambda: os.getenv("EMBEDDING_PROVIDER", "openai"))
    dimension: int = Field(default_factory=lambda: int(os.getenv("EMBEDDING_DIM", "1024")))
    max_token_size: int = Field(
        default_factory=lambda: int(os.getenv("EMBEDDING_MAX_TOKENS", "8192"))
    )


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
        default="""---Role---

You are a helpful assistant responding to user query about Data Sources provided below.


---Goal---

Generate a concise response based on Data Sources and follow Response Rules, considering both
the conversation history and the current query. Data sources contain two parts: Knowledge
Graph(KG) and Document Chunks(DC). Summarize all information in the provided Data Sources,
and incorporating general knowledge relevant to the Data Sources. Do not include information
not provided by Data Sources.

When handling information with timestamps:
1. Each piece of information (both relationships and content) has a "created_at" timestamp
   indicating when we acquired this knowledge
2. When encountering conflicting information, consider both the content/relationship
   and the timestamp
3. Don't automatically prefer the most recent information - use judgment based on
   the context
4. For time-specific queries, prioritize temporal information in the content before
   considering creation timestamps

---Conversation History---
{history}

---Data Sources---

1. From Knowledge Graph(KG):
{kg_context}

---Response Rules---

- Target format and length: {response_type}
- Use markdown formatting with appropriate section headings
- Please respond in the same language as the user's question.
- Ensure the response maintains continuity with the conversation history.
- Organize answer in sesctions focusing on one main point or aspect of the answer
- Use clear and descriptive section titles that reflect the content
- List up to 5 most important reference sources at the end under "References" sesction.
  Clearly indicating whether each source is from Knowledge Graph (KG) or Vector Data (DC),
  in the following format: [KG/DC] Source content
- If you don't know the answer, just say so. Do not make anything up.
- Do not include information not provided by the Data Sources."""
    )


class Settings(BaseModel):
    """Main application settings."""

    workdir: str = Field(default="workdir")
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    graph: GraphConfig = Field(default_factory=GraphConfig)
    text: TextConfig = Field(default_factory=TextConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)


def get_settings() -> Settings:
    """Get application settings instance."""
    return Settings()
