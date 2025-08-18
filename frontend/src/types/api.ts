export interface BaseApiResponse {
  status: 'success' | 'error' | 'processing';
  message: string;
  timestamp: string;
}

export interface Project {
  name: string;
  description?: string;
  created_at?: string;
  updated_at?: string;
  document_count?: number;
  entity_count?: number;
  relation_count?: number;
}

export interface ProjectCreateRequest {
  name: string;
  description?: string;
}

export interface ProjectSwitchRequest {
  project_name?: string;
}

export interface ProjectDeleteRequest {
  project_name: string;
  confirm: boolean;
}

export interface Document {
  id: string;
  title?: string;
  content: string;
  source?: string;
  metadata?: Record<string, any>;
  tags?: string[];
  created_at?: string;
  updated_at?: string;
  file_type?: string;
  file_size?: number;
  // Backend-specific fields
  filename?: string;
  content_length?: number;
  content_type?: string;
  stored_at?: string;
  project_name?: string;
  content_hash?: string;
}

export interface DocumentUploadRequest {
  texts?: string[];
  metadata?: Record<string, any>;
  tags?: string[];
}

export interface DocumentListRequest {
  page?: number;
  page_size?: number;
  tag_filter?: string[];
  search_query?: string;
}

export interface DocumentDeleteRequest {
  document_ids: string[];
}

export interface Entity {
  id: string;
  name: string;
  entity_type: string;
  description?: string;
  properties: Record<string, any>;
  confidence: number;
}

export interface Relation {
  id: string;
  head_entity_id: string;
  tail_entity_id: string;
  relation_type: string;
  description?: string;
  properties: Record<string, any>;
  confidence: number;
}

export interface TextChunk {
  id: string;
  content: string;
  title?: string;
  source?: string;
  start_index?: number;
  end_index?: number;
  entities: string[];
  relations: string[];
}

export interface Cluster {
  id: string;
  name: string;
  description?: string;
  entities: string[];
  relations: string[];
}

export interface KnowledgeGraph {
  name: string;
  description: string;
  entities: Entity[];
  relations: Relation[];
  clusters: Cluster[];
  text_chunks: TextChunk[];
}

export interface KnowledgeGraphBuildRequest {
  document_ids?: string[];
  texts?: string[];
  graph_name?: string;
  graph_description?: string;
  use_cache?: boolean;
  save_to_vector_store?: boolean;
  from_step?: string;
}

export interface KnowledgeGraphUpdateRequest {
  additional_document_ids?: string[];
  additional_texts?: string[];
  use_cache?: boolean;
  save_to_vector_store?: boolean;
}

export interface SearchRequest {
  query: string;
  top_k?: number;
  filter_dict?: Record<string, any>;
  search_type?: 'entities' | 'relations' | 'text_chunks';
}

export interface SearchResult {
  id: string;
  content: string;
  score: number;
  type: 'entity' | 'relation' | 'text_chunk';
  metadata?: Record<string, any>;
}

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
}

export interface ChatRequest {
  question: string;
  conversation_history?: ChatMessage[];
  entity_top_k?: number;
  relation_top_k?: number;
  text_chunk_top_k?: number;
  response_type?: string;
  stream?: boolean;
}

export interface ChatResponse {
  answer: string;
  context?: {
    entities: Entity[];
    relations: Relation[];
    text_chunks: TextChunk[];
  };
  reasoning?: string;
}

export interface StreamChatChunk {
  question: string;
  chunk: string;
  partial_answer: string;
  answer?: string;
  context?: {
    entities: Entity[];
    relations: Relation[];
    text_chunks: TextChunk[];
  };
  finished: boolean;
}

export interface AppConfig {
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

export interface ConfigUpdateRequest extends Partial<AppConfig> {}

export interface SystemStats {
  total_projects: number;
  total_documents: number;
  total_entities: number;
  total_relations: number;
  memory_usage: number;
  cache_size: number;
  uptime: string;
}

export interface CacheInfo {
  cache_keys: string[];
  cache_size_mb: number;
  total_items: number;
}

export interface ClearCacheRequest {
  from_step?: string;
}

export interface CacheDataRequest {
  page?: number;
  page_size?: number;
  filter_by?: Record<string, any>;
}

export interface BuildStatus {
  is_building: boolean;
  current_step?: string;
  progress_percentage?: number;
  estimated_time_remaining?: number;
  error_message?: string;
}
