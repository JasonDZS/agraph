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

export interface ProjectSettings {
  workdir: string;
  current_project?: string;
  openai: OpenAIConfig;
  llm: LLMConfig;
  embedding: EmbeddingConfig;
  graph: GraphConfig;
  text: TextConfig;
  rag: RAGConfig;
}

export interface ProjectInfo {
  project_name: string;
  description: string;
  created_at: string;
  version: string;
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
  runtime_info?: RuntimeInfo;
}

export interface ConfigUpdateRequest {
  workdir?: string;
  openai_api_key?: string;
  openai_api_base?: string;
  llm_model?: string;
  llm_temperature?: number;
  llm_max_tokens?: number;
  llm_provider?: string;
  embedding_model?: string;
  embedding_provider?: string;
  embedding_dimension?: number;
  embedding_max_token_size?: number;
  embedding_batch_size?: number;
  entity_types?: string[];
  relation_types?: string[];
  max_chunk_size?: number;
  chunk_overlap?: number;
  system_prompt?: string;
}

export interface ConfigResponse {
  status: string;
  message: string;
  data: ProjectConfig | ProjectSettings;
  timestamp?: string;
}
