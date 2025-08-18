/**
 * Document module types
 */
import type { Document as ApiDocument } from '@/types/api';

export interface Document extends ApiDocument {
  filename?: string;
  content_length?: number;
  content_type?: string;
  stored_at?: string;
  project_name?: string;
  content_hash?: string;
  extracted_metadata?: Record<string, any>;
}

export interface DocumentUploadItem {
  id: string;
  filename: string;
  content_type?: string;
  size: number;
  content_length: number;
  extracted_metadata?: Record<string, any>;
}

export interface DocumentUploadProgress {
  file: File;
  progress: number;
  status: 'pending' | 'uploading' | 'success' | 'error';
  error?: string;
  result?: DocumentUploadItem;
}

export interface DocumentListParams {
  page?: number;
  page_size?: number;
  tag_filter?: string[];
  search_query?: string;
  project_name?: string;
}

export interface DocumentListResponse {
  documents: Document[];
  pagination: {
    page: number;
    page_size: number;
    total_count: number;
    total_pages: number;
  };
  filters: {
    tag_filter: string[];
    search_query: string;
  };
}

export interface DocumentStats {
  total_documents: number;
  total_content_size: number;
  storage_path: string;
  project_name?: string;
  tags: Record<string, number>;
}

export interface DocumentFilterState {
  searchQuery: string;
  selectedTags: string[];
  selectedDocuments: string[];
  sortBy: 'stored_at' | 'filename' | 'content_length';
  sortOrder: 'asc' | 'desc';
}

export interface UploadOptions {
  metadata?: Record<string, any>;
  tags?: string[];
  project_name?: string;
}
