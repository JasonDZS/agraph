import { apiClient, ApiResponse, RequestConfig } from './api';
import type { Document, BaseApiResponse } from '@/types/api';

export interface DocumentUploadResponse extends BaseApiResponse {
  data?: {
    document_ids: string[];
    uploaded_count: number;
    failed_count: number;
    failed_documents?: Array<{
      index: number;
      error: string;
    }>;
  };
}

export interface DocumentListResponse extends BaseApiResponse {
  data?: {
    documents: Document[];
    total: number;
    page: number;
    page_size: number;
    total_pages: number;
  };
}

export interface DocumentDeleteResponse extends BaseApiResponse {
  data?: {
    deleted_count: number;
    failed_count: number;
    failed_documents?: Array<{
      document_id: string;
      error: string;
    }>;
  };
}

export interface UploadProgress {
  loaded: number;
  total: number;
  percentage: number;
  speed?: number;
  estimatedTime?: number;
}

export interface FileUploadOptions {
  onProgress?: (progress: UploadProgress) => void;
  chunkSize?: number;
  concurrent?: boolean;
  timeout?: number;
}

class DocumentService {
  private readonly baseEndpoint = '/documents';
  private readonly SUPPORTED_FILE_TYPES = [
    'text/plain',
    'text/markdown',
    'text/csv',
    'application/json',
    'application/pdf',
    'application/msword',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    'application/vnd.ms-excel',
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
  ];

  async uploadDocuments(
    request: DocumentUploadRequest,
    options?: FileUploadOptions
  ): Promise<ApiResponse<DocumentUploadResponse>> {
    const config: RequestConfig = {
      timeout: options?.timeout || 300000, // 5 minutes for file uploads
    };

    return apiClient.post<DocumentUploadResponse>(
      `${this.baseEndpoint}/upload`,
      request,
      config
    );
  }

  async uploadFiles(
    files: File[],
    options?: { metadata?: any; tags?: string[]; project_name?: string }
  ): Promise<ApiResponse<any>> {
    const formData = new FormData();

    files.forEach(file => {
      formData.append('files', file);
    });

    if (options?.metadata) {
      formData.append('metadata', JSON.stringify(options.metadata));
    }

    if (options?.tags) {
      formData.append('tags', JSON.stringify(options.tags));
    }

    // Build URL with query parameters
    const params = new URLSearchParams();
    if (options?.project_name) {
      params.append('project_name', options.project_name);
    }

    const endpoint = params.toString()
      ? `${this.baseEndpoint}/upload?${params.toString()}`
      : `${this.baseEndpoint}/upload`;

    const config: RequestConfig = {
      timeout: 300000,
      method: 'POST',
      body: formData,
    };

    return apiClient.request<any>(endpoint, config);
  }

  async uploadFilesWithProgress(
    files: File[],
    options?: FileUploadOptions
  ): Promise<ApiResponse<DocumentUploadResponse>> {
    return new Promise((resolve, reject) => {
      const formData = new FormData();

      files.forEach(file => {
        formData.append('files', file);
      });

      const xhr = new XMLHttpRequest();

      if (options?.onProgress) {
        xhr.upload.addEventListener('progress', event => {
          if (event.lengthComputable) {
            const progress: UploadProgress = {
              loaded: event.loaded,
              total: event.total,
              percentage: Math.round((event.loaded / event.total) * 100),
            };
            options.onProgress!(progress);
          }
        });
      }

      xhr.addEventListener('load', () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          try {
            const response = JSON.parse(xhr.responseText);
            resolve({
              data: response,
              success: true,
              status: 'success',
            });
          } catch (error) {
            reject(new Error('Invalid JSON response'));
          }
        } else {
          reject(new Error(`HTTP ${xhr.status}: ${xhr.statusText}`));
        }
      });

      xhr.addEventListener('error', () => {
        reject(new Error('Network error'));
      });

      xhr.addEventListener('timeout', () => {
        reject(new Error('Upload timeout'));
      });

      xhr.timeout = options?.timeout || 300000;
      xhr.open(
        'POST',
        `${apiClient['baseURL']}${this.baseEndpoint}/upload-files`
      );

      // Add authorization header if available
      const authHeader = (apiClient as any).defaultHeaders['Authorization'];
      if (authHeader) {
        xhr.setRequestHeader('Authorization', authHeader);
      }

      xhr.send(formData);
    });
  }

  async listDocuments(request?: {
    page?: number;
    page_size?: number;
    search_query?: string;
    tag_filter?: string[];
    project_name?: string;
  }): Promise<ApiResponse<any>> {
    const params = new URLSearchParams();

    if (request) {
      if (request.page) params.append('page', request.page.toString());
      if (request.page_size)
        params.append('page_size', request.page_size.toString());
      if (request.search_query)
        params.append('search_query', request.search_query);
      if (request.tag_filter?.length) {
        params.append('tag_filter', JSON.stringify(request.tag_filter));
      }
      if (request.project_name) {
        params.append('project_name', request.project_name);
      }
    }

    const endpoint = params.toString()
      ? `${this.baseEndpoint}/list?${params.toString()}`
      : `${this.baseEndpoint}/list`;

    return apiClient.get<any>(endpoint, { cache: true });
  }

  async getDocument(
    documentId: string,
    projectName?: string
  ): Promise<ApiResponse<any>> {
    const params = new URLSearchParams();
    if (projectName) {
      params.append('project_name', projectName);
    }

    const url = params.toString()
      ? `${this.baseEndpoint}/${encodeURIComponent(documentId)}?${params.toString()}`
      : `${this.baseEndpoint}/${encodeURIComponent(documentId)}`;

    return apiClient.get<any>(url, {
      cache: true,
    });
  }

  async deleteDocuments(
    documentIds: string[],
    projectName?: string
  ): Promise<ApiResponse<any>> {
    const params = new URLSearchParams();
    if (projectName) {
      params.append('project_name', projectName);
    }

    const url = params.toString()
      ? `${this.baseEndpoint}/delete?${params.toString()}`
      : `${this.baseEndpoint}/delete`;

    return apiClient.post<any>(url, {
      document_ids: documentIds,
    });
  }

  async getDocumentContent(
    documentId: string
  ): Promise<ApiResponse<{ content: string }>> {
    return apiClient.get<{ content: string }>(
      `${this.baseEndpoint}/${encodeURIComponent(documentId)}/content`
    );
  }

  async updateDocumentMetadata(
    documentId: string,
    metadata: Record<string, any>
  ): Promise<ApiResponse<Document>> {
    return apiClient.put<Document>(
      `${this.baseEndpoint}/${encodeURIComponent(documentId)}/metadata`,
      {
        metadata,
      }
    );
  }

  async updateDocumentTags(
    documentId: string,
    tags: string[]
  ): Promise<ApiResponse<Document>> {
    return apiClient.put<Document>(
      `${this.baseEndpoint}/${encodeURIComponent(documentId)}/tags`,
      {
        tags,
      }
    );
  }

  async getDocumentsByTag(
    tag: string
  ): Promise<ApiResponse<DocumentListResponse>> {
    return apiClient.get<DocumentListResponse>(
      `${this.baseEndpoint}/by-tag/${encodeURIComponent(tag)}`,
      {
        cache: true,
      }
    );
  }

  async searchDocuments(
    query: string,
    options?: {
      page?: number;
      page_size?: number;
      filters?: Record<string, any>;
    }
  ): Promise<ApiResponse<DocumentListResponse>> {
    const params = new URLSearchParams({
      q: query,
    });

    if (options?.page) params.append('page', options.page.toString());
    if (options?.page_size)
      params.append('page_size', options.page_size.toString());
    if (options?.filters) {
      Object.entries(options.filters).forEach(([key, value]) => {
        params.append(`filter_${key}`, value.toString());
      });
    }

    return apiClient.get<DocumentListResponse>(
      `${this.baseEndpoint}/search?${params.toString()}`
    );
  }

  async exportDocuments(
    documentIds: string[],
    format: 'json' | 'csv' | 'txt' = 'json'
  ): Promise<ApiResponse<Blob>> {
    return apiClient.post(
      `${this.baseEndpoint}/export`,
      {
        document_ids: documentIds,
        format,
      },
      {
        headers: {
          Accept:
            format === 'json'
              ? 'application/json'
              : format === 'csv'
                ? 'text/csv'
                : 'text/plain',
        },
      }
    );
  }

  validateFile(file: File): { valid: boolean; error?: string } {
    // Check file size (max 50MB)
    const maxSize = 50 * 1024 * 1024;
    if (file.size > maxSize) {
      return { valid: false, error: '文件大小不能超过 50MB' };
    }

    // Check file type
    if (!this.SUPPORTED_FILE_TYPES.includes(file.type)) {
      return { valid: false, error: '不支持的文件格式' };
    }

    return { valid: true };
  }

  validateFiles(files: File[]): { valid: boolean; errors: string[] } {
    const errors: string[] = [];

    for (const file of files) {
      const validation = this.validateFile(file);
      if (!validation.valid) {
        errors.push(`${file.name}: ${validation.error}`);
      }
    }

    return { valid: errors.length === 0, errors };
  }

  getSupportedFileTypes(): string[] {
    return [...this.SUPPORTED_FILE_TYPES];
  }

  formatFileSize(bytes: number): string {
    const units = ['B', 'KB', 'MB', 'GB'];
    let size = bytes;
    let unitIndex = 0;

    while (size >= 1024 && unitIndex < units.length - 1) {
      size /= 1024;
      unitIndex++;
    }

    return `${size.toFixed(unitIndex > 0 ? 1 : 0)} ${units[unitIndex]}`;
  }

  async uploadTexts(
    texts: string[],
    options?: { metadata?: any; tags?: string[]; project_name?: string }
  ): Promise<ApiResponse<any>> {
    const params = new URLSearchParams();
    if (options?.project_name) {
      params.append('project_name', options.project_name);
    }

    const url = params.toString()
      ? `${this.baseEndpoint}/from-text?${params.toString()}`
      : `${this.baseEndpoint}/from-text`;

    return apiClient.post<any>(url, {
      texts: texts,
      metadata: options?.metadata || {},
      tags: options?.tags || [],
    });
  }

  async getStats(projectName?: string): Promise<ApiResponse<any>> {
    const params = new URLSearchParams();
    if (projectName) {
      params.append('project_name', projectName);
    }

    const url = params.toString()
      ? `${this.baseEndpoint}/stats/summary?${params.toString()}`
      : `${this.baseEndpoint}/stats/summary`;

    return apiClient.get<any>(url);
  }

  clearCache(): void {
    apiClient.clearCache();
  }
}

export const documentService = new DocumentService();
