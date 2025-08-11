import os
from typing import Any, ClassVar, List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv(".env")


class Settings(BaseModel):
    _instance: ClassVar[Optional["Settings"]] = None

    workdir: str = Field(default="workdir")

    # OpenAI settings
    OPENAI_API_KEY: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    OPENAI_API_BASE: str = Field(default_factory=lambda: os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"))

    # Model settings
    LLM_MODEL: str = Field(default_factory=lambda: os.getenv("LLM_MODEL", "gpt-3.5-turbo"))
    LLM_TEMPERATURE: float = Field(default=0.0)
    LLM_MAX_TOKENS: int = Field(
        default_factory=lambda: int(os.getenv("LLM_MAX_TOKENS", "4096"))
    )  # Default max tokens for GPT-3.5 Turbo
    LLM_PROVIDER: str = Field(
        default_factory=lambda: os.getenv("LLM_PROVIDER", "openai")
    )  # Options: "openai", "hf", "ollama", "custom"

    # Embedding settings
    EMBEDDING_MODEL: str = Field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "Pro/BAAI/bge-m3"))
    EMBEDDING_PROVIDER: str = Field(
        default_factory=lambda: os.getenv("EMBEDDING_PROVIDER", "openai")
    )  # Options: "openai", "hf", "ollama"
    EMBEDDING_DIM: int = Field(
        default_factory=lambda: int(os.getenv("EMBEDDING_DIM", "1024"))
    )  # Default dimension for BGE models
    EMBEDDING_MAX_TOKEN_SIZE: int = Field(
        default_factory=lambda: int(os.getenv("EMBEDDING_MAX_TOKENS", "8192"))
    )  # Default max token size for BGE models

    # Entity and Relation Type settings
    ENTITY_TYPES: List[str] = Field(
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

    RELATION_TYPES: List[str] = Field(
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
    # pylint: disable=line-too-long
    RAG_SYS_PROMPT: str = Field(
        default="""---Role---

You are a helpful assistant responding to user query about Data Sources provided below.


---Goal---

Generate a concise response based on Data Sources and follow Response Rules, considering both the conversation history and the current query. Data sources contain two parts: Knowledge Graph(KG) and Document Chunks(DC). Summarize all information in the provided Data Sources, and incorporating general knowledge relevant to the Data Sources. Do not include information not provided by Data Sources.

When handling information with timestamps:
1. Each piece of information (both relationships and content) has a "created_at" timestamp indicating when we acquired this knowledge
2. When encountering conflicting information, consider both the content/relationship and the timestamp
3. Don't automatically prefer the most recent information - use judgment based on the context
4. For time-specific queries, prioritize temporal information in the content before considering creation timestamps

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
- List up to 5 most important reference sources at the end under "References" sesction. Clearly indicating whether each source is from Knowledge Graph (KG) or Vector Data (DC), in the following format: [KG/DC] Source content
- If you don't know the answer, just say so. Do not make anything up.
- Do not include information not provided by the Data Sources."""
    )

    MAX_CHUNK_SIZE: int = Field(
        default_factory=lambda: int(os.getenv("MAX_CHUNK_SIZE", "512"))
    )  # Maximum size of each text chunk in tokens
    CHUNK_OVERLAP: int = Field(
        default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "100"))
    )  # Overlap size between text chunks in tokens

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        if not hasattr(cls, "_instance") or cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if hasattr(self, "_initialized"):
            return
        super().__init__(*args, **kwargs)
        self._initialized = True

    class Config:
        arbitrary_types_allowed = True


# Create singleton instance
settings = Settings()
