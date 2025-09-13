"""API models and schemas for AGraph."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class ResponseStatus(str, Enum):
    """Response status enumeration."""

    SUCCESS = "success"
    ERROR = "error"
    PROCESSING = "processing"


class BaseResponse(BaseModel):
    """Base response model."""

    model_config = ConfigDict(json_encoders={datetime: lambda v: v.isoformat()})

    status: ResponseStatus
    message: str
    timestamp: datetime = Field(default_factory=datetime.now)


class ConfigUpdateRequest(BaseModel):
    """Configuration update request for unified Settings."""

    # Global settings
    workdir: Optional[str] = None
    current_project: Optional[str] = None
    max_current: Optional[int] = None

    # OpenAI settings
    openai_api_key: Optional[str] = None
    openai_api_base: Optional[str] = None

    # LLM settings
    llm_model: Optional[str] = None
    llm_temperature: Optional[float] = None
    llm_max_tokens: Optional[int] = None
    llm_provider: Optional[str] = None

    # Embedding settings
    embedding_model: Optional[str] = None
    embedding_provider: Optional[str] = None
    embedding_dimension: Optional[int] = None
    embedding_max_token_size: Optional[int] = None
    embedding_batch_size: Optional[int] = None

    # Graph settings
    entity_types: Optional[List[str]] = None
    relation_types: Optional[List[str]] = None

    # Text processing settings
    max_chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None

    # RAG system prompt
    system_prompt: Optional[str] = None

    # Builder settings (unified via Settings.builder)
    builder_enable_cache: Optional[bool] = None
    builder_cache_dir: Optional[str] = None
    builder_cache_ttl: Optional[int] = None
    builder_auto_cleanup: Optional[bool] = None
    builder_chunk_size: Optional[int] = None
    builder_chunk_overlap: Optional[int] = None
    builder_entity_confidence_threshold: Optional[float] = None
    builder_relation_confidence_threshold: Optional[float] = None
    builder_cluster_algorithm: Optional[str] = None
    builder_min_cluster_size: Optional[int] = None
    builder_enable_user_interaction: Optional[bool] = None
    builder_auto_save_edits: Optional[bool] = None

    # Legacy support for backward compatibility
    collection_name: Optional[str] = None
    persist_directory: Optional[str] = None
    vector_store_type: Optional[str] = None
    use_openai_embeddings: Optional[bool] = None
    enable_knowledge_graph: Optional[bool] = None
    chunk_size: Optional[int] = None  # Alias for max_chunk_size
    entity_confidence_threshold: Optional[float] = None  # Alias for builder_entity_confidence_threshold
    relation_confidence_threshold: Optional[float] = None  # Alias for builder_relation_confidence_threshold


class ConfigResponse(BaseResponse):
    """Configuration response."""

    data: Optional[Dict[str, Any]] = None


class ConfigFileRequest(BaseModel):
    """Configuration file operation request."""

    file_path: Optional[str] = None


class ConfigFileResponse(BaseResponse):
    """Configuration file operation response."""

    data: Optional[Dict[str, Any]] = None
    file_path: Optional[str] = None


# Project Management Models
class ProjectCreateRequest(BaseModel):
    """Project creation request."""

    name: str = Field(..., min_length=1, max_length=50)
    description: Optional[str] = None


class ProjectResponse(BaseResponse):
    """Project operation response."""

    data: Optional[Dict[str, Any]] = None
    project_name: Optional[str] = None


class ProjectListResponse(BaseResponse):
    """Project list response."""

    projects: List[str] = Field(default_factory=list)


class ProjectSwitchRequest(BaseModel):
    """Project switch request."""

    project_name: Optional[str] = None  # None to switch to no project


class ProjectDeleteRequest(BaseModel):
    """Project deletion request."""

    project_name: str = Field(..., min_length=1)
    confirm: bool = Field(default=False)  # Safety confirmation


class DocumentUploadRequest(BaseModel):
    """Document upload request."""

    texts: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None


class DocumentUploadResponse(BaseResponse):
    """Document upload response."""

    data: Optional[Dict[str, Any]] = None


class DocumentListRequest(BaseModel):
    """Document list request."""

    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=10, ge=1, le=100)
    tag_filter: Optional[List[str]] = None
    search_query: Optional[str] = None


class DocumentListResponse(BaseResponse):
    """Document list response."""

    data: Optional[Dict[str, Any]] = None


class DocumentDeleteRequest(BaseModel):
    """Document delete request."""

    document_ids: List[str]


class DocumentDeleteResponse(BaseResponse):
    """Document delete response."""

    data: Optional[Dict[str, Any]] = None


class KnowledgeGraphBuildRequest(BaseModel):
    """Knowledge graph build request."""

    document_ids: Optional[List[str]] = None
    texts: Optional[List[str]] = None
    graph_name: str = "Knowledge Graph"
    graph_description: str = "Built by AGraph"
    use_cache: bool = True
    save_to_vector_store: bool = True
    from_step: Optional[str] = None
    enable_graph: bool = True


class KnowledgeGraphBuildResponse(BaseResponse):
    """Knowledge graph build response."""

    data: Optional[Dict[str, Any]] = None


class KnowledgeGraphUpdateRequest(BaseModel):
    """Knowledge graph update request."""

    additional_document_ids: Optional[List[str]] = None
    additional_texts: Optional[List[str]] = None
    use_cache: bool = True
    save_to_vector_store: bool = True


class KnowledgeGraphUpdateResponse(BaseResponse):
    """Knowledge graph update response."""

    data: Optional[Dict[str, Any]] = None


class KnowledgeGraphStatusResponse(BaseResponse):
    """Knowledge graph status response."""

    data: Optional[Dict[str, Any]] = None


class KnowledgeGraphGetRequest(BaseModel):
    """Knowledge graph get request."""

    include_text_chunks: bool = Field(default=False, description="Include text chunks in response")
    include_clusters: bool = Field(default=False, description="Include clusters in response")
    entity_limit: Optional[int] = Field(default=None, description="Limit number of entities")
    relation_limit: Optional[int] = Field(default=None, description="Limit number of relations")


class KnowledgeGraphGetResponse(BaseResponse):
    """Knowledge graph get response."""

    data: Optional[Dict[str, Any]] = None


class KnowledgeGraphVisualizationRequest(BaseModel):
    """Knowledge graph visualization data request."""

    include_clusters: bool = Field(default=True, description="Include cluster information")
    include_text_chunks: bool = Field(default=False, description="Include text chunk information")
    max_entities: int = Field(default=500, description="Maximum number of entities to return")
    max_relations: int = Field(default=1000, description="Maximum number of relations to return")
    min_confidence: float = Field(default=0.0, description="Minimum confidence threshold")
    entity_types: Optional[List[str]] = Field(default=None, description="Filter by entity types")
    relation_types: Optional[List[str]] = Field(default=None, description="Filter by relation types")
    cluster_layout: bool = Field(default=False, description="Apply cluster-based layout")


class KnowledgeGraphVisualizationResponse(BaseResponse):
    """Knowledge graph visualization data response."""

    data: Optional[Dict[str, Any]] = None


class TextChunkSearchRequest(BaseModel):
    """Text chunk search request."""

    search: Optional[str] = Field(default=None, description="Search query for text chunks")
    entity_id: Optional[str] = Field(default=None, description="Filter by entity ID")
    limit: int = Field(default=20, description="Maximum number of text chunks to return")
    offset: int = Field(default=0, description="Offset for pagination")


class TextChunkSearchResponse(BaseResponse):
    """Text chunk search response."""

    data: Optional[Dict[str, Any]] = None


class SearchRequest(BaseModel):
    """Search request."""

    query: str
    top_k: int = 10
    filter_dict: Optional[Dict[str, Any]] = None
    search_type: str = Field(default="text_chunks", description="entities, relations, text_chunks")


class SearchResponse(BaseResponse):
    """Search response."""

    data: Optional[Dict[str, Any]] = None


class ChatRequest(BaseModel):
    """Chat request."""

    question: str
    conversation_history: Optional[List[Dict[str, str]]] = None
    entity_top_k: int = 5
    relation_top_k: int = 5
    text_chunk_top_k: int = 5
    response_type: str = "详细回答"
    stream: bool = False


class ChatResponse(BaseResponse):
    """Chat response."""

    data: Optional[Dict[str, Any]] = None


class StreamChatResponse(BaseModel):
    """Stream chat response."""

    question: str
    chunk: str = ""
    partial_answer: str = ""
    answer: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    finished: bool = False


class StatsResponse(BaseResponse):
    """Stats response."""

    data: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    """Health check response."""

    model_config = ConfigDict(json_encoders={datetime: lambda v: v.isoformat()})

    status: str = "healthy"
    timestamp: datetime = Field(default_factory=datetime.now)
    version: str = "1.0.0"


class BuildStatusResponse(BaseResponse):
    """Build status response."""

    data: Optional[Dict[str, Any]] = None


class CacheInfoResponse(BaseResponse):
    """Cache info response."""

    data: Optional[Dict[str, Any]] = None


class ClearCacheRequest(BaseModel):
    """Clear cache request."""

    from_step: Optional[str] = None


class CacheDataRequest(BaseModel):
    """Cache data view request."""

    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=10, ge=1, le=100)
    filter_by: Optional[Dict[str, Any]] = None


class CacheDataResponse(BaseResponse):
    """Cache data response."""

    data: Optional[Dict[str, Any]] = None


class EntityModel(BaseModel):
    """Entity model for API responses."""

    id: str
    name: str
    entity_type: str
    description: Optional[str] = None
    properties: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = 0.0


class RelationModel(BaseModel):
    """Relation model for API responses."""

    id: str
    head_entity_id: str
    tail_entity_id: str
    relation_type: str
    description: Optional[str] = None
    properties: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = 0.0


class TextChunkModel(BaseModel):
    """Text chunk model for API responses."""

    id: str
    content: str
    title: Optional[str] = None
    source: Optional[str] = None
    start_index: Optional[int] = None
    end_index: Optional[int] = None


class ClusterModel(BaseModel):
    """Cluster model for API responses."""

    id: str
    name: str
    description: Optional[str] = None
    entities: List[str] = Field(default_factory=list)
    relations: List[str] = Field(default_factory=list)


class KnowledgeGraphModel(BaseModel):
    """Knowledge graph model for API responses."""

    name: str
    description: str
    entities: List[EntityModel] = Field(default_factory=list)
    relations: List[RelationModel] = Field(default_factory=list)
    clusters: List[ClusterModel] = Field(default_factory=list)
    text_chunks: List[TextChunkModel] = Field(default_factory=list)


class ErrorResponse(BaseResponse):
    """Error response model."""

    error_code: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None
