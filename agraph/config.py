import os
from typing import Any, ClassVar, List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv(".env")


class Settings(BaseModel):
    _instance: ClassVar[Optional["Settings"]] = None

    # API settings
    APP_TITLE: str = Field(default="Data Visualization Agent")
    APP_VERSION: str = Field(default="0.1.0")
    HOST: str = Field(default="0.0.0.0")
    PORT: int = Field(default=8000)
    LOG_LEVEL: str = Field(default="debug")  # Options: "debug", "info", "warning", "error", "critical"

    # CORS settings
    CORS_ORIGINS: List[str] = Field(default_factory=lambda: ["*"])
    CORS_METHODS: List[str] = Field(default_factory=lambda: ["GET", "POST", "PUT", "DELETE", "OPTIONS"])
    CORS_HEADERS: List[str] = Field(default_factory=lambda: ["*"])
    CORS_CREDENTIALS: bool = Field(default=False)

    # OpenAI settings
    OPENAI_API_KEY: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    OPENAI_API_BASE: str = Field(default_factory=lambda: os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"))

    # Database settings
    DATABASES_DIR: str = Field(default="databases")
    LOGS_DIR: str = Field(default="logs")
    TEMPLATES_DIR: str = Field(default="templates")

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
