export interface OpenAIConfig {
  api_key: string;
  api_base: string;
}

export interface LLMConfig {
  model: string;
  temperature: number;
  max_tokens: number;
  provider: string;
}

export interface EmbeddingConfig {
  model: string;
  provider: string;
  dimension: number;
  max_token_size: number;
  batch_size: number;
}

export interface GraphConfig {
  entity_types: string[];
  relation_types: string[];
}

export interface TextConfig {
  max_chunk_size: number;
  chunk_overlap: number;
}

export interface RAGConfig {
  system_prompt: string;
}

// New Builder configuration interface
export interface BuilderConfig {
  enable_cache: boolean;
  cache_dir: string;
  cache_ttl: number;
  auto_cleanup: boolean;
  chunk_size: number;
  chunk_overlap: number;
  entity_confidence_threshold: number;
  entity_types: string[];
  relation_confidence_threshold: number;
  relation_types: string[];
  cluster_algorithm: string;
  min_cluster_size: number;
  enable_user_interaction: boolean;
  auto_save_edits: boolean;
}

export interface ProjectSettings {
  workdir: string;
  current_project?: string;
  max_current: number;
  openai: OpenAIConfig;
  llm: LLMConfig;
  embedding: EmbeddingConfig;
  graph: GraphConfig;
  text: TextConfig;
  rag: RAGConfig;
  builder: BuilderConfig;  // Added builder configuration
}

export interface ProjectInfo {
  project_name: string;
  description: string;
  created_at: string;
  version: string;
  paths?: {
    project_dir: string;
    config_file: string;
    document_storage: string;
    vector_db: string;
    cache: string;
    logs: string;
  };
}

export interface RuntimeInfo {
  collection_name: string;
  persist_directory: string;
  vector_store_type: string;
  use_openai_embeddings: boolean;
  enable_knowledge_graph: boolean;
  is_initialized: boolean;
  has_knowledge_graph: boolean;
  builder_config?: any;
}

export interface ProjectConfig {
  project_info: ProjectInfo;
  settings: ProjectSettings;
  config_source?: string;
  config_file_path?: string;
  is_project_specific?: boolean;
  runtime_info?: RuntimeInfo;
}

// Updated ConfigUpdateRequest to include all builder settings
export interface ConfigUpdateRequest {
  // Global settings
  workdir?: string;
  current_project?: string;
  max_current?: number;

  // OpenAI settings
  openai_api_key?: string;
  openai_api_base?: string;

  // LLM settings
  llm_model?: string;
  llm_temperature?: number;
  llm_max_tokens?: number;
  llm_provider?: string;

  // Embedding settings
  embedding_model?: string;
  embedding_provider?: string;
  embedding_dimension?: number;
  embedding_max_token_size?: number;
  embedding_batch_size?: number;

  // Graph settings
  entity_types?: string[];
  relation_types?: string[];

  // Text processing settings
  max_chunk_size?: number;
  chunk_overlap?: number;

  // RAG system prompt
  system_prompt?: string;

  // Builder settings (unified via Settings.builder)
  builder_enable_cache?: boolean;
  builder_cache_dir?: string;
  builder_cache_ttl?: number;
  builder_auto_cleanup?: boolean;
  builder_chunk_size?: number;
  builder_chunk_overlap?: number;
  builder_entity_confidence_threshold?: number;
  builder_relation_confidence_threshold?: number;
  builder_cluster_algorithm?: string;
  builder_min_cluster_size?: number;
  builder_enable_user_interaction?: boolean;
  builder_auto_save_edits?: boolean;

  // Legacy support for backward compatibility
  collection_name?: string;
  persist_directory?: string;
  vector_store_type?: string;
  use_openai_embeddings?: boolean;
  enable_knowledge_graph?: boolean;
  chunk_size?: number;  // Alias for max_chunk_size
  entity_confidence_threshold?: number;  // Alias for builder_entity_confidence_threshold
  relation_confidence_threshold?: number;  // Alias for builder_relation_confidence_threshold
}

export interface ConfigResponse {
  status: string;
  message: string;
  data: ProjectConfig | ProjectSettings;
  timestamp?: string;
}

// New interfaces for backup and recovery operations
export interface BackupStatus {
  project_name: string;
  backup_file_path: string;
  backup_exists: boolean;
  has_complete_settings?: boolean;
  settings_version?: string;
  created_at?: string;
  updated_at?: string;
  file_size_kb?: number;
  settings_sections?: string[];
  settings_count?: number;
  config_summary?: {
    has_openai_config: boolean;
    has_llm_config: boolean;
    has_builder_config: boolean;
    current_project?: string;
    workdir?: string;
  };
  parse_error?: string;
  is_valid?: boolean;
  reason?: string;
}

export interface ConfigFileRequest {
  file_path?: string;
}

export interface ConfigFileResponse {
  status: string;
  message: string;
  data?: any;
  file_path?: string;
}
