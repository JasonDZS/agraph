# AGraph API 集成指南

## API 集成架构

### 服务层架构设计

```typescript
// services/base/ApiClient.ts
import axios, { AxiosInstance, AxiosRequestConfig, AxiosResponse } from 'axios';

interface ApiClientConfig {
  baseURL: string;
  timeout: number;
  retryAttempts: number;
  retryDelay: number;
}

class ApiClient {
  private instance: AxiosInstance;
  private config: ApiClientConfig;

  constructor(config: ApiClientConfig) {
    this.config = config;
    this.instance = axios.create({
      baseURL: config.baseURL,
      timeout: config.timeout,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    this.setupInterceptors();
  }

  private setupInterceptors() {
    // 请求拦截器
    this.instance.interceptors.request.use(
      (config) => this.handleRequest(config),
      (error) => this.handleRequestError(error)
    );

    // 响应拦截器
    this.instance.interceptors.response.use(
      (response) => this.handleResponse(response),
      (error) => this.handleResponseError(error)
    );
  }

  private handleRequest(config: AxiosRequestConfig): AxiosRequestConfig {
    // 添加项目上下文
    const currentProject = this.getCurrentProject();
    if (currentProject) {
      config.params = {
        ...config.params,
        project_name: currentProject,
      };
    }

    // 添加请求标识
    config.metadata = {
      ...config.metadata,
      requestId: this.generateRequestId(),
      timestamp: Date.now(),
    };

    // 记录请求开始
    this.logRequest(config);

    return config;
  }

  private handleResponse(response: AxiosResponse): AxiosResponse {
    this.logResponse(response);
    return response;
  }

  private async handleResponseError(error: any): Promise<never> {
    const { config, response } = error;

    // 记录错误
    this.logError(error);

    // 重试逻辑
    if (this.shouldRetry(error) && config.retryCount < this.config.retryAttempts) {
      config.retryCount = (config.retryCount || 0) + 1;

      await this.delay(this.config.retryDelay * config.retryCount);
      return this.instance(config);
    }

    // 统一错误处理
    throw this.normalizeError(error);
  }

  private shouldRetry(error: any): boolean {
    if (!error.response) return true; // 网络错误

    const status = error.response.status;
    return status >= 500 || status === 408 || status === 429;
  }

  private normalizeError(error: any): ApiError {
    const { response, request, message } = error;

    if (response) {
      // 服务器响应错误
      return new ApiError({
        type: 'SERVER_ERROR',
        status: response.status,
        message: response.data?.message || message,
        details: response.data,
        originalError: error,
      });
    } else if (request) {
      // 网络错误
      return new ApiError({
        type: 'NETWORK_ERROR',
        message: '网络连接失败',
        originalError: error,
      });
    } else {
      // 其他错误
      return new ApiError({
        type: 'UNKNOWN_ERROR',
        message: message || '未知错误',
        originalError: error,
      });
    }
  }

  // 公共API方法
  async get<T>(url: string, config?: AxiosRequestConfig): Promise<ApiResponse<T>> {
    const response = await this.instance.get(url, config);
    return this.transformResponse(response);
  }

  async post<T>(url: string, data?: any, config?: AxiosRequestConfig): Promise<ApiResponse<T>> {
    const response = await this.instance.post(url, data, config);
    return this.transformResponse(response);
  }

  async put<T>(url: string, data?: any, config?: AxiosRequestConfig): Promise<ApiResponse<T>> {
    const response = await this.instance.put(url, data, config);
    return this.transformResponse(response);
  }

  async delete<T>(url: string, config?: AxiosRequestConfig): Promise<ApiResponse<T>> {
    const response = await this.instance.delete(url, config);
    return this.transformResponse(response);
  }

  // 流式请求支持
  async streamPost<T>(
    url: string,
    data?: any,
    config?: AxiosRequestConfig
  ): Promise<AsyncIterable<T>> {
    const response = await this.instance.post(url, data, {
      ...config,
      responseType: 'stream',
    });

    return this.parseStreamResponse<T>(response.data);
  }

  private async* parseStreamResponse<T>(stream: ReadableStream): AsyncIterable<T> {
    const reader = stream.getReader();
    let buffer = '';

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += new TextDecoder().decode(value);
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.trim() && line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              yield data as T;
            } catch (error) {
              console.warn('Failed to parse stream data:', line);
            }
          }
        }
      }
    } finally {
      reader.releaseLock();
    }
  }
}
```

### API 错误处理

```typescript
// services/base/ApiError.ts

interface ApiErrorConfig {
  type: 'SERVER_ERROR' | 'NETWORK_ERROR' | 'VALIDATION_ERROR' | 'UNKNOWN_ERROR';
  status?: number;
  message: string;
  details?: any;
  originalError?: Error;
}

class ApiError extends Error {
  public readonly type: string;
  public readonly status?: number;
  public readonly details?: any;
  public readonly originalError?: Error;

  constructor(config: ApiErrorConfig) {
    super(config.message);
    this.name = 'ApiError';
    this.type = config.type;
    this.status = config.status;
    this.details = config.details;
    this.originalError = config.originalError;
  }

  // 错误分类方法
  isNetworkError(): boolean {
    return this.type === 'NETWORK_ERROR';
  }

  isServerError(): boolean {
    return this.type === 'SERVER_ERROR';
  }

  isValidationError(): boolean {
    return this.type === 'VALIDATION_ERROR';
  }

  isRetryable(): boolean {
    return this.isNetworkError() || (this.isServerError() && this.status >= 500);
  }

  // 获取用户友好的错误消息
  getUserMessage(): string {
    switch (this.type) {
      case 'NETWORK_ERROR':
        return '网络连接失败，请检查网络状态';
      case 'SERVER_ERROR':
        if (this.status === 401) return '认证失败，请重新登录';
        if (this.status === 403) return '没有权限执行此操作';
        if (this.status === 404) return '请求的资源不存在';
        if (this.status >= 500) return '服务器内部错误，请稍后重试';
        return this.message;
      case 'VALIDATION_ERROR':
        return `数据验证失败: ${this.message}`;
      default:
        return this.message || '发生未知错误';
    }
  }
}
```

## 项目管理 API 集成

### 项目服务实现

```typescript
// services/ProjectService.ts

interface ProjectService {
  list(includeStats?: boolean): Promise<ProjectListResponse>;
  create(data: CreateProjectData): Promise<ProjectResponse>;
  get(projectName: string): Promise<ProjectResponse>;
  getCurrent(): Promise<ProjectResponse>;
  switch(projectName: string | null): Promise<ProjectResponse>;
  delete(projectName: string, confirm: boolean): Promise<ProjectResponse>;
  getOverview(includeStats?: boolean): Promise<ProjectListResponse>;
}

class ProjectServiceImpl implements ProjectService {
  constructor(private apiClient: ApiClient) {}

  async list(includeStats = false): Promise<ProjectListResponse> {
    try {
      const response = await this.apiClient.get<ProjectListResponse>('/projects/list', {
        params: { include_stats: includeStats },
      });

      return response.data;
    } catch (error) {
      throw this.handleProjectError(error, 'list');
    }
  }

  async create(data: CreateProjectData): Promise<ProjectResponse> {
    try {
      // 数据验证
      this.validateCreateProjectData(data);

      const response = await this.apiClient.post<ProjectResponse>('/projects/create', data);

      // 成功后更新本地状态
      this.notifyProjectCreated(response.data);

      return response.data;
    } catch (error) {
      throw this.handleProjectError(error, 'create');
    }
  }

  async get(projectName: string): Promise<ProjectResponse> {
    try {
      if (!projectName) {
        throw new ApiError({
          type: 'VALIDATION_ERROR',
          message: '项目名称不能为空',
        });
      }

      const response = await this.apiClient.get<ProjectResponse>(`/projects/${projectName}`);
      return response.data;
    } catch (error) {
      throw this.handleProjectError(error, 'get');
    }
  }

  async getCurrent(): Promise<ProjectResponse> {
    try {
      const response = await this.apiClient.get<ProjectResponse>('/projects/current');
      return response.data;
    } catch (error) {
      throw this.handleProjectError(error, 'getCurrent');
    }
  }

  async switch(projectName: string | null): Promise<ProjectResponse> {
    try {
      const response = await this.apiClient.post<ProjectResponse>('/projects/switch', {
        project_name: projectName,
      });

      // 切换成功后清理相关缓存
      this.clearProjectRelatedCache();
      this.notifyProjectSwitched(projectName);

      return response.data;
    } catch (error) {
      throw this.handleProjectError(error, 'switch');
    }
  }

  async delete(projectName: string, confirm = false): Promise<ProjectResponse> {
    try {
      if (!confirm) {
        throw new ApiError({
          type: 'VALIDATION_ERROR',
          message: '删除项目需要确认',
        });
      }

      const response = await this.apiClient.post<ProjectResponse>('/projects/delete', {
        project_name: projectName,
        confirm: true,
      });

      // 删除成功后清理本地数据
      this.notifyProjectDeleted(projectName);

      return response.data;
    } catch (error) {
      throw this.handleProjectError(error, 'delete');
    }
  }

  async getOverview(includeStats = false): Promise<ProjectListResponse> {
    try {
      const response = await this.apiClient.get<ProjectListResponse>('/projects/', {
        params: { include_stats: includeStats },
      });

      return response.data;
    } catch (error) {
      throw this.handleProjectError(error, 'overview');
    }
  }

  // 私有方法
  private validateCreateProjectData(data: CreateProjectData): void {
    if (!data.name) {
      throw new ApiError({
        type: 'VALIDATION_ERROR',
        message: '项目名称不能为空',
      });
    }

    if (data.name.length > 50) {
      throw new ApiError({
        type: 'VALIDATION_ERROR',
        message: '项目名称不能超过50个字符',
      });
    }

    if (!/^[a-zA-Z0-9_-]+$/.test(data.name)) {
      throw new ApiError({
        type: 'VALIDATION_ERROR',
        message: '项目名称只能包含字母、数字、下划线和连字符',
      });
    }
  }

  private handleProjectError(error: any, operation: string): ApiError {
    if (error instanceof ApiError) {
      return error;
    }

    return new ApiError({
      type: 'SERVER_ERROR',
      message: `项目${operation}操作失败: ${error.message}`,
      originalError: error,
    });
  }

  private notifyProjectCreated(project: ProjectResponse): void {
    // 通知全局状态管理器
    eventBus.emit('project:created', project);
  }

  private notifyProjectSwitched(projectName: string | null): void {
    eventBus.emit('project:switched', projectName);
  }

  private notifyProjectDeleted(projectName: string): void {
    eventBus.emit('project:deleted', projectName);
  }

  private clearProjectRelatedCache(): void {
    // 清理知识图谱缓存
    // 清理文档缓存
    // 清理对话缓存
    eventBus.emit('cache:clear', ['knowledge-graph', 'documents', 'chat']);
  }
}
```

## 文档管理 API 集成

### 文档服务实现

```typescript
// services/DocumentService.ts

interface UploadProgressCallback {
  (progressEvent: { loaded: number; total: number }): void;
}

interface DocumentService {
  upload(formData: FormData, options?: { onUploadProgress?: UploadProgressCallback }): Promise<DocumentUploadResponse>;
  uploadTexts(texts: string[], metadata?: any, tags?: string[]): Promise<DocumentUploadResponse>;
  list(params: DocumentListParams): Promise<DocumentListResponse>;
  get(documentId: string): Promise<Document>;
  delete(documentIds: string[]): Promise<DocumentDeleteResponse>;
  download(documentId: string): Promise<Blob>;
  getStats(): Promise<DocumentStatsResponse>;
}

interface DocumentListParams {
  page?: number;
  pageSize?: number;
  search?: string;
  tags?: string[];
  dateRange?: [string, string];
  fileTypes?: string[];
}

class DocumentServiceImpl implements DocumentService {
  constructor(private apiClient: ApiClient) {}

  async upload(
    formData: FormData,
    options: { onUploadProgress?: UploadProgressCallback } = {}
  ): Promise<DocumentUploadResponse> {
    try {
      // 验证文件数据
      this.validateUploadData(formData);

      const response = await this.apiClient.post<DocumentUploadResponse>(
        '/documents/upload',
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
          onUploadProgress: options.onUploadProgress ? (progressEvent) => {
            options.onUploadProgress!({
              loaded: progressEvent.loaded || 0,
              total: progressEvent.total || 0,
            });
          } : undefined,
          timeout: 300000, // 5分钟超时
        }
      );

      // 上传成功后更新文档缓存
      this.invalidateDocumentCache();

      return response.data;
    } catch (error) {
      throw this.handleDocumentError(error, 'upload');
    }
  }

  async uploadTexts(
    texts: string[],
    metadata?: any,
    tags?: string[]
  ): Promise<DocumentUploadResponse> {
    try {
      if (!texts || texts.length === 0) {
        throw new ApiError({
          type: 'VALIDATION_ERROR',
          message: '文本内容不能为空',
        });
      }

      const response = await this.apiClient.post<DocumentUploadResponse>(
        '/documents/from-text',
        {
          texts,
          metadata: metadata || {},
          tags: tags || [],
        }
      );

      this.invalidateDocumentCache();

      return response.data;
    } catch (error) {
      throw this.handleDocumentError(error, 'uploadTexts');
    }
  }

  async list(params: DocumentListParams): Promise<DocumentListResponse> {
    try {
      const queryParams = this.buildDocumentListParams(params);

      const response = await this.apiClient.get<DocumentListResponse>(
        '/documents/list',
        { params: queryParams }
      );

      return response.data;
    } catch (error) {
      throw this.handleDocumentError(error, 'list');
    }
  }

  async get(documentId: string): Promise<Document> {
    try {
      if (!documentId) {
        throw new ApiError({
          type: 'VALIDATION_ERROR',
          message: '文档ID不能为空',
        });
      }

      const response = await this.apiClient.get<{ data: Document }>(`/documents/${documentId}`);
      return response.data.data;
    } catch (error) {
      throw this.handleDocumentError(error, 'get');
    }
  }

  async delete(documentIds: string[]): Promise<DocumentDeleteResponse> {
    try {
      if (!documentIds || documentIds.length === 0) {
        throw new ApiError({
          type: 'VALIDATION_ERROR',
          message: '请选择要删除的文档',
        });
      }

      const response = await this.apiClient.delete<DocumentDeleteResponse>(
        '/documents/delete',
        {
          data: { document_ids: documentIds }
        }
      );

      this.invalidateDocumentCache();

      return response.data;
    } catch (error) {
      throw this.handleDocumentError(error, 'delete');
    }
  }

  async download(documentId: string): Promise<Blob> {
    try {
      const response = await this.apiClient.get(`/documents/${documentId}/download`, {
        responseType: 'blob',
      });

      return response.data;
    } catch (error) {
      throw this.handleDocumentError(error, 'download');
    }
  }

  async getStats(): Promise<DocumentStatsResponse> {
    try {
      const response = await this.apiClient.get<DocumentStatsResponse>('/documents/stats/summary');
      return response.data;
    } catch (error) {
      throw this.handleDocumentError(error, 'getStats');
    }
  }

  // 私有方法
  private validateUploadData(formData: FormData): void {
    const files = formData.getAll('files') as File[];

    if (!files || files.length === 0) {
      throw new ApiError({
        type: 'VALIDATION_ERROR',
        message: '请选择要上传的文件',
      });
    }

    const maxFileSize = 50 * 1024 * 1024; // 50MB
    const maxFiles = 10;

    if (files.length > maxFiles) {
      throw new ApiError({
        type: 'VALIDATION_ERROR',
        message: `最多只能上传 ${maxFiles} 个文件`,
      });
    }

    for (const file of files) {
      if (file.size > maxFileSize) {
        throw new ApiError({
          type: 'VALIDATION_ERROR',
          message: `文件 ${file.name} 超过大小限制 (50MB)`,
        });
      }
    }
  }

  private buildDocumentListParams(params: DocumentListParams): any {
    const queryParams: any = {
      page: params.page || 1,
      page_size: params.pageSize || 20,
    };

    if (params.search) {
      queryParams.search_query = params.search;
    }

    if (params.tags && params.tags.length > 0) {
      queryParams.tag_filter = JSON.stringify(params.tags);
    }

    return queryParams;
  }

  private handleDocumentError(error: any, operation: string): ApiError {
    if (error instanceof ApiError) {
      return error;
    }

    return new ApiError({
      type: 'SERVER_ERROR',
      message: `文档${operation}操作失败: ${error.message}`,
      originalError: error,
    });
  }

  private invalidateDocumentCache(): void {
    // 通知缓存失效
    eventBus.emit('cache:invalidate', 'documents');
  }
}
```

## 知识图谱 API 集成

### 知识图谱服务实现

```typescript
// services/KnowledgeGraphService.ts

interface KnowledgeGraphService {
  build(config: BuildGraphConfig): Promise<KnowledgeGraphBuildResponse>;
  update(config: UpdateGraphConfig): Promise<KnowledgeGraphUpdateResponse>;
  getStatus(): Promise<KnowledgeGraphStatusResponse>;
  buildWithProgress(config: BuildGraphConfig): AsyncIterable<BuildProgressEvent>;
}

interface BuildProgressEvent {
  step: string;
  progress: number;
  message: string;
  data?: any;
}

class KnowledgeGraphServiceImpl implements KnowledgeGraphService {
  constructor(private apiClient: ApiClient) {}

  async build(config: BuildGraphConfig): Promise<KnowledgeGraphBuildResponse> {
    try {
      this.validateBuildConfig(config);

      const response = await this.apiClient.post<KnowledgeGraphBuildResponse>(
        '/knowledge-graph/build',
        config,
        {
          timeout: 600000, // 10分钟超时
        }
      );

      // 构建成功后通知状态更新
      this.notifyGraphBuilt(response.data);

      return response.data;
    } catch (error) {
      throw this.handleKnowledgeGraphError(error, 'build');
    }
  }

  async update(config: UpdateGraphConfig): Promise<KnowledgeGraphUpdateResponse> {
    try {
      this.validateUpdateConfig(config);

      const response = await this.apiClient.post<KnowledgeGraphUpdateResponse>(
        '/knowledge-graph/update',
        config,
        {
          timeout: 600000, // 10分钟超时
        }
      );

      this.notifyGraphUpdated(response.data);

      return response.data;
    } catch (error) {
      throw this.handleKnowledgeGraphError(error, 'update');
    }
  }

  async getStatus(): Promise<KnowledgeGraphStatusResponse> {
    try {
      const response = await this.apiClient.get<KnowledgeGraphStatusResponse>(
        '/knowledge-graph/status'
      );

      return response.data;
    } catch (error) {
      throw this.handleKnowledgeGraphError(error, 'getStatus');
    }
  }

  async *buildWithProgress(config: BuildGraphConfig): AsyncIterable<BuildProgressEvent> {
    try {
      this.validateBuildConfig(config);

      // 使用Server-Sent Events或WebSocket实现进度推送
      const eventSource = new EventSource(`/knowledge-graph/build-stream?${new URLSearchParams(config as any)}`);

      let resolve: (value: IteratorResult<BuildProgressEvent>) => void;
      let reject: (reason?: any) => void;
      const queue: BuildProgressEvent[] = [];
      let finished = false;

      eventSource.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data) as BuildProgressEvent;

          if (data.step === 'completed') {
            finished = true;
            eventSource.close();
          }

          queue.push(data);

          if (resolve) {
            const value = queue.shift()!;
            resolve({ value, done: false });
            resolve = null!;
          }
        } catch (error) {
          reject?.(error);
        }
      };

      eventSource.onerror = (error) => {
        eventSource.close();
        reject?.(new ApiError({
          type: 'SERVER_ERROR',
          message: '知识图谱构建过程中断',
          originalError: error as any,
        }));
      };

      while (!finished || queue.length > 0) {
        if (queue.length > 0) {
          yield queue.shift()!;
        } else {
          await new Promise<void>((res, rej) => {
            resolve = (result) => {
              if (!result.done) {
                res();
              }
            };
            reject = rej;
          });
        }
      }
    } catch (error) {
      throw this.handleKnowledgeGraphError(error, 'buildWithProgress');
    }
  }

  // 私有方法
  private validateBuildConfig(config: BuildGraphConfig): void {
    if (!config.document_ids && !config.texts) {
      throw new ApiError({
        type: 'VALIDATION_ERROR',
        message: '请提供文档ID或文本内容',
      });
    }

    if (config.document_ids && config.document_ids.length === 0) {
      throw new ApiError({
        type: 'VALIDATION_ERROR',
        message: '文档ID列表不能为空',
      });
    }

    if (config.texts && config.texts.length === 0) {
      throw new ApiError({
        type: 'VALIDATION_ERROR',
        message: '文本列表不能为空',
      });
    }
  }

  private validateUpdateConfig(config: UpdateGraphConfig): void {
    if (!config.additional_document_ids && !config.additional_texts) {
      throw new ApiError({
        type: 'VALIDATION_ERROR',
        message: '请提供要添加的文档ID或文本内容',
      });
    }
  }

  private handleKnowledgeGraphError(error: any, operation: string): ApiError {
    if (error instanceof ApiError) {
      return error;
    }

    return new ApiError({
      type: 'SERVER_ERROR',
      message: `知识图谱${operation}操作失败: ${error.message}`,
      originalError: error,
    });
  }

  private notifyGraphBuilt(response: KnowledgeGraphBuildResponse): void {
    eventBus.emit('knowledge-graph:built', response);
  }

  private notifyGraphUpdated(response: KnowledgeGraphUpdateResponse): void {
    eventBus.emit('knowledge-graph:updated', response);
  }
}
```

## 搜索服务 API 集成

### 搜索服务实现

```typescript
// services/SearchService.ts

interface SearchService {
  searchEntities(query: string, options?: SearchOptions): Promise<EntitySearchResult[]>;
  searchRelations(query: string, options?: SearchOptions): Promise<RelationSearchResult[]>;
  searchTextChunks(query: string, options?: SearchOptions): Promise<TextChunkSearchResult[]>;
  search(query: string, type: SearchType, options?: SearchOptions): Promise<SearchResponse>;
  advancedSearch(criteria: AdvancedSearchCriteria): Promise<SearchResponse>;
}

interface SearchOptions {
  topK?: number;
  filters?: Record<string, any>;
  includeContext?: boolean;
}

interface AdvancedSearchCriteria {
  query: string;
  entityTypes?: string[];
  relationTypes?: string[];
  confidenceThreshold?: number;
  dateRange?: [string, string];
  sources?: string[];
}

class SearchServiceImpl implements SearchService {
  constructor(private apiClient: ApiClient) {}

  async searchEntities(query: string, options: SearchOptions = {}): Promise<EntitySearchResult[]> {
    try {
      this.validateSearchQuery(query);

      const response = await this.apiClient.post<SearchResponse>('/search', {
        query,
        search_type: 'entities',
        top_k: options.topK || 10,
        filter_dict: options.filters,
      });

      return response.data.data.results.map(this.transformEntityResult);
    } catch (error) {
      throw this.handleSearchError(error, 'searchEntities');
    }
  }

  async searchRelations(query: string, options: SearchOptions = {}): Promise<RelationSearchResult[]> {
    try {
      this.validateSearchQuery(query);

      const response = await this.apiClient.post<SearchResponse>('/search', {
        query,
        search_type: 'relations',
        top_k: options.topK || 10,
        filter_dict: options.filters,
      });

      return response.data.data.results.map(this.transformRelationResult);
    } catch (error) {
      throw this.handleSearchError(error, 'searchRelations');
    }
  }

  async searchTextChunks(query: string, options: SearchOptions = {}): Promise<TextChunkSearchResult[]> {
    try {
      this.validateSearchQuery(query);

      const response = await this.apiClient.post<SearchResponse>('/search', {
        query,
        search_type: 'text_chunks',
        top_k: options.topK || 10,
        filter_dict: options.filters,
      });

      return response.data.data.results.map(this.transformTextChunkResult);
    } catch (error) {
      throw this.handleSearchError(error, 'searchTextChunks');
    }
  }

  async search(query: string, type: SearchType, options: SearchOptions = {}): Promise<SearchResponse> {
    try {
      this.validateSearchQuery(query);

      const response = await this.apiClient.post<SearchResponse>('/search', {
        query,
        search_type: type,
        top_k: options.topK || 10,
        filter_dict: options.filters,
      });

      // 记录搜索历史
      this.recordSearchHistory(query, type, response.data.data.total_count);

      return response.data;
    } catch (error) {
      throw this.handleSearchError(error, 'search');
    }
  }

  async advancedSearch(criteria: AdvancedSearchCriteria): Promise<SearchResponse> {
    try {
      this.validateAdvancedSearchCriteria(criteria);

      // 构建高级搜索过滤器
      const filters = this.buildAdvancedSearchFilters(criteria);

      const response = await this.apiClient.post<SearchResponse>('/search', {
        query: criteria.query,
        search_type: 'text_chunks', // 默认搜索文本块
        top_k: 20,
        filter_dict: filters,
      });

      return response.data;
    } catch (error) {
      throw this.handleSearchError(error, 'advancedSearch');
    }
  }

  // 私有方法
  private validateSearchQuery(query: string): void {
    if (!query || query.trim().length === 0) {
      throw new ApiError({
        type: 'VALIDATION_ERROR',
        message: '搜索查询不能为空',
      });
    }

    if (query.length > 500) {
      throw new ApiError({
        type: 'VALIDATION_ERROR',
        message: '搜索查询不能超过500个字符',
      });
    }
  }

  private validateAdvancedSearchCriteria(criteria: AdvancedSearchCriteria): void {
    this.validateSearchQuery(criteria.query);

    if (criteria.confidenceThreshold && (criteria.confidenceThreshold < 0 || criteria.confidenceThreshold > 1)) {
      throw new ApiError({
        type: 'VALIDATION_ERROR',
        message: '置信度阈值必须在0-1之间',
      });
    }
  }

  private buildAdvancedSearchFilters(criteria: AdvancedSearchCriteria): Record<string, any> {
    const filters: Record<string, any> = {};

    if (criteria.entityTypes && criteria.entityTypes.length > 0) {
      filters.entity_types = criteria.entityTypes;
    }

    if (criteria.relationTypes && criteria.relationTypes.length > 0) {
      filters.relation_types = criteria.relationTypes;
    }

    if (criteria.confidenceThreshold !== undefined) {
      filters.min_confidence = criteria.confidenceThreshold;
    }

    if (criteria.dateRange) {
      filters.date_range = criteria.dateRange;
    }

    if (criteria.sources && criteria.sources.length > 0) {
      filters.sources = criteria.sources;
    }

    return filters;
  }

  private transformEntityResult(result: any): EntitySearchResult {
    return {
      entity: result.entity,
      score: result.score,
      highlights: this.extractHighlights(result),
    };
  }

  private transformRelationResult(result: any): RelationSearchResult {
    return {
      relation: result.relation,
      score: result.score,
      highlights: this.extractHighlights(result),
    };
  }

  private transformTextChunkResult(result: any): TextChunkSearchResult {
    return {
      textChunk: result.text_chunk,
      score: result.score,
      highlights: this.extractHighlights(result),
    };
  }

  private extractHighlights(result: any): string[] {
    // 提取搜索结果中的高亮片段
    return result.highlights || [];
  }

  private recordSearchHistory(query: string, type: SearchType, resultCount: number): void {
    const searchRecord = {
      query,
      type,
      resultCount,
      timestamp: new Date().toISOString(),
    };

    // 保存到本地存储或发送到分析服务
    this.saveSearchHistory(searchRecord);
  }

  private saveSearchHistory(record: any): void {
    try {
      const history = JSON.parse(localStorage.getItem('searchHistory') || '[]');
      history.unshift(record);

      // 只保留最近100条记录
      if (history.length > 100) {
        history.splice(100);
      }

      localStorage.setItem('searchHistory', JSON.stringify(history));
    } catch (error) {
      console.warn('Failed to save search history:', error);
    }
  }

  private handleSearchError(error: any, operation: string): ApiError {
    if (error instanceof ApiError) {
      return error;
    }

    return new ApiError({
      type: 'SERVER_ERROR',
      message: `搜索${operation}操作失败: ${error.message}`,
      originalError: error,
    });
  }
}
```

## 对话系统 API 集成

### 对话服务实现

```typescript
// services/ChatService.ts

interface ChatService {
  chat(request: ChatRequest): Promise<ChatResponse>;
  streamChat(request: ChatRequest): AsyncIterable<StreamChatResponse>;
  getConversationHistory(conversationId: string): Promise<ChatMessage[]>;
  clearConversation(conversationId: string): Promise<void>;
}

class ChatServiceImpl implements ChatService {
  constructor(private apiClient: ApiClient) {}

  async chat(request: ChatRequest): Promise<ChatResponse> {
    try {
      this.validateChatRequest(request);

      const response = await this.apiClient.post<ChatResponse>('/chat', {
        ...request,
        stream: false,
      }, {
        timeout: 120000, // 2分钟超时
      });

      // 保存对话历史
      this.saveConversationMessage(request, response.data);

      return response.data;
    } catch (error) {
      throw this.handleChatError(error, 'chat');
    }
  }

  async *streamChat(request: ChatRequest): AsyncIterable<StreamChatResponse> {
    try {
      this.validateChatRequest(request);

      const streamResponse = await this.apiClient.streamPost<StreamChatResponse>(
        '/chat/stream',
        { ...request, stream: true }
      );

      let fullResponse = '';
      let finalContext: any = null;

      for await (const chunk of streamResponse) {
        if (chunk.chunk) {
          fullResponse += chunk.chunk;
        }

        if (chunk.context) {
          finalContext = chunk.context;
        }

        yield chunk;

        if (chunk.finished) {
          // 保存完整的对话
          this.saveStreamConversation(request, fullResponse, finalContext);
          break;
        }
      }
    } catch (error) {
      throw this.handleChatError(error, 'streamChat');
    }
  }

  async getConversationHistory(conversationId: string): Promise<ChatMessage[]> {
    try {
      // 从本地存储获取对话历史
      const history = this.loadConversationHistory(conversationId);
      return history;
    } catch (error) {
      throw this.handleChatError(error, 'getConversationHistory');
    }
  }

  async clearConversation(conversationId: string): Promise<void> {
    try {
      this.clearConversationHistory(conversationId);
    } catch (error) {
      throw this.handleChatError(error, 'clearConversation');
    }
  }

  // 私有方法
  private validateChatRequest(request: ChatRequest): void {
    if (!request.question || request.question.trim().length === 0) {
      throw new ApiError({
        type: 'VALIDATION_ERROR',
        message: '问题不能为空',
      });
    }

    if (request.question.length > 2000) {
      throw new ApiError({
        type: 'VALIDATION_ERROR',
        message: '问题不能超过2000个字符',
      });
    }

    if (request.conversation_history && request.conversation_history.length > 20) {
      throw new ApiError({
        type: 'VALIDATION_ERROR',
        message: '对话历史不能超过20轮',
      });
    }
  }

  private saveConversationMessage(request: ChatRequest, response: ChatResponse): void {
    try {
      const conversationId = this.generateConversationId();
      const history = this.loadConversationHistory(conversationId);

      // 添加用户消息
      history.push({
        id: generateId(),
        role: 'user',
        content: request.question,
        timestamp: new Date(),
      });

      // 添加助手回复
      if (response.data?.answer) {
        history.push({
          id: generateId(),
          role: 'assistant',
          content: response.data.answer,
          timestamp: new Date(),
          context: response.data.context,
        });
      }

      this.saveConversationHistory(conversationId, history);
    } catch (error) {
      console.warn('Failed to save conversation:', error);
    }
  }

  private saveStreamConversation(request: ChatRequest, fullResponse: string, context: any): void {
    try {
      const conversationId = this.generateConversationId();
      const history = this.loadConversationHistory(conversationId);

      // 添加用户消息
      history.push({
        id: generateId(),
        role: 'user',
        content: request.question,
        timestamp: new Date(),
      });

      // 添加完整的助手回复
      history.push({
        id: generateId(),
        role: 'assistant',
        content: fullResponse,
        timestamp: new Date(),
        context,
      });

      this.saveConversationHistory(conversationId, history);
    } catch (error) {
      console.warn('Failed to save stream conversation:', error);
    }
  }

  private loadConversationHistory(conversationId: string): ChatMessage[] {
    try {
      const stored = localStorage.getItem(`conversation_${conversationId}`);
      return stored ? JSON.parse(stored) : [];
    } catch (error) {
      console.warn('Failed to load conversation history:', error);
      return [];
    }
  }

  private saveConversationHistory(conversationId: string, history: ChatMessage[]): void {
    try {
      localStorage.setItem(`conversation_${conversationId}`, JSON.stringify(history));
    } catch (error) {
      console.warn('Failed to save conversation history:', error);
    }
  }

  private clearConversationHistory(conversationId: string): void {
    try {
      localStorage.removeItem(`conversation_${conversationId}`);
    } catch (error) {
      console.warn('Failed to clear conversation history:', error);
    }
  }

  private generateConversationId(): string {
    // 使用当前项目和时间戳生成对话ID
    const currentProject = this.apiClient.getCurrentProject();
    const timestamp = Date.now();
    return `${currentProject || 'default'}_${timestamp}`;
  }

  private handleChatError(error: any, operation: string): ApiError {
    if (error instanceof ApiError) {
      return error;
    }

    return new ApiError({
      type: 'SERVER_ERROR',
      message: `对话${operation}操作失败: ${error.message}`,
      originalError: error,
    });
  }
}
```

## API 缓存策略

### 响应缓存实现

```typescript
// services/base/CacheManager.ts

interface CacheOptions {
  ttl?: number; // 缓存存活时间(毫秒)
  maxSize?: number; // 最大缓存条目数
  serialize?: (data: any) => string;
  deserialize?: (data: string) => any;
}

class CacheManager {
  private cache = new Map<string, CacheEntry>();
  private options: Required<CacheOptions>;

  constructor(options: CacheOptions = {}) {
    this.options = {
      ttl: options.ttl || 5 * 60 * 1000, // 默认5分钟
      maxSize: options.maxSize || 100,
      serialize: options.serialize || JSON.stringify,
      deserialize: options.deserialize || JSON.parse,
    };
  }

  get<T>(key: string): T | null {
    const entry = this.cache.get(key);

    if (!entry) {
      return null;
    }

    if (Date.now() > entry.expireAt) {
      this.cache.delete(key);
      return null;
    }

    // 更新访问时间
    entry.lastAccess = Date.now();

    try {
      return this.options.deserialize(entry.data);
    } catch (error) {
      console.warn('Cache deserialize error:', error);
      this.cache.delete(key);
      return null;
    }
  }

  set<T>(key: string, data: T, ttl?: number): void {
    try {
      const serializedData = this.options.serialize(data);
      const expireAt = Date.now() + (ttl || this.options.ttl);

      // 检查缓存大小限制
      if (this.cache.size >= this.options.maxSize) {
        this.evictLeastRecentlyUsed();
      }

      this.cache.set(key, {
        data: serializedData,
        expireAt,
        lastAccess: Date.now(),
      });
    } catch (error) {
      console.warn('Cache serialize error:', error);
    }
  }

  delete(key: string): boolean {
    return this.cache.delete(key);
  }

  clear(): void {
    this.cache.clear();
  }

  has(key: string): boolean {
    const entry = this.cache.get(key);
    if (!entry) return false;

    if (Date.now() > entry.expireAt) {
      this.cache.delete(key);
      return false;
    }

    return true;
  }

  size(): number {
    return this.cache.size;
  }

  // 清理过期缓存
  cleanup(): void {
    const now = Date.now();
    for (const [key, entry] of this.cache.entries()) {
      if (now > entry.expireAt) {
        this.cache.delete(key);
      }
    }
  }

  // LRU淘汰策略
  private evictLeastRecentlyUsed(): void {
    let lruKey = '';
    let lruTime = Date.now();

    for (const [key, entry] of this.cache.entries()) {
      if (entry.lastAccess < lruTime) {
        lruTime = entry.lastAccess;
        lruKey = key;
      }
    }

    if (lruKey) {
      this.cache.delete(lruKey);
    }
  }
}

interface CacheEntry {
  data: string;
  expireAt: number;
  lastAccess: number;
}

// 创建全局缓存实例
export const apiCache = new CacheManager({
  ttl: 5 * 60 * 1000, // 5分钟
  maxSize: 200,
});

export const documentCache = new CacheManager({
  ttl: 10 * 60 * 1000, // 10分钟
  maxSize: 50,
});

export const projectCache = new CacheManager({
  ttl: 30 * 60 * 1000, // 30分钟
  maxSize: 20,
});
```

### 带缓存的API客户端

```typescript
// services/base/CachedApiClient.ts

class CachedApiClient extends ApiClient {
  constructor(
    config: ApiClientConfig,
    private cacheManager: CacheManager
  ) {
    super(config);
  }

  async get<T>(url: string, config?: AxiosRequestConfig): Promise<ApiResponse<T>> {
    const cacheKey = this.buildCacheKey('GET', url, config?.params);

    // 尝试从缓存获取
    const cached = this.cacheManager.get<ApiResponse<T>>(cacheKey);
    if (cached) {
      return cached;
    }

    // 从API获取数据
    const response = await super.get<T>(url, config);

    // 缓存响应(只缓存成功的响应)
    if (response.status === 'success') {
      this.cacheManager.set(cacheKey, response);
    }

    return response;
  }

  async post<T>(url: string, data?: any, config?: AxiosRequestConfig): Promise<ApiResponse<T>> {
    // POST请求通常不缓存，但可以清理相关缓存
    const response = await super.post<T>(url, data, config);

    // 清理相关缓存
    this.invalidateRelatedCache(url);

    return response;
  }

  async put<T>(url: string, data?: any, config?: AxiosRequestConfig): Promise<ApiResponse<T>> {
    const response = await super.put<T>(url, data, config);
    this.invalidateRelatedCache(url);
    return response;
  }

  async delete<T>(url: string, config?: AxiosRequestConfig): Promise<ApiResponse<T>> {
    const response = await super.delete<T>(url, config);
    this.invalidateRelatedCache(url);
    return response;
  }

  private buildCacheKey(method: string, url: string, params?: any): string {
    const paramString = params ? JSON.stringify(params) : '';
    return `${method}:${url}:${paramString}`;
  }

  private invalidateRelatedCache(url: string): void {
    // 根据URL模式清理相关缓存
    if (url.includes('/projects')) {
      this.cacheManager.clear(); // 项目相关操作清理所有缓存
    } else if (url.includes('/documents')) {
      // 清理文档相关缓存
      // 实现具体的缓存清理逻辑
    } else if (url.includes('/knowledge-graph')) {
      // 清理知识图谱相关缓存
    }
  }
}
```

## 事件总线集成

### 事件系统实现

```typescript
// utils/EventBus.ts

type EventCallback<T = any> = (data: T) => void;

class EventBus {
  private events = new Map<string, Set<EventCallback>>();

  on<T = any>(event: string, callback: EventCallback<T>): () => void {
    if (!this.events.has(event)) {
      this.events.set(event, new Set());
    }

    this.events.get(event)!.add(callback);

    // 返回取消订阅函数
    return () => {
      this.off(event, callback);
    };
  }

  off<T = any>(event: string, callback: EventCallback<T>): void {
    const callbacks = this.events.get(event);
    if (callbacks) {
      callbacks.delete(callback);
      if (callbacks.size === 0) {
        this.events.delete(event);
      }
    }
  }

  emit<T = any>(event: string, data?: T): void {
    const callbacks = this.events.get(event);
    if (callbacks) {
      callbacks.forEach(callback => {
        try {
          callback(data);
        } catch (error) {
          console.error(`Event callback error for ${event}:`, error);
        }
      });
    }
  }

  once<T = any>(event: string, callback: EventCallback<T>): void {
    const onceCallback = (data: T) => {
      callback(data);
      this.off(event, onceCallback);
    };

    this.on(event, onceCallback);
  }

  clear(): void {
    this.events.clear();
  }
}

export const eventBus = new EventBus();

// 预定义事件类型
export const Events = {
  PROJECT_CREATED: 'project:created',
  PROJECT_SWITCHED: 'project:switched',
  PROJECT_DELETED: 'project:deleted',

  DOCUMENT_UPLOADED: 'document:uploaded',
  DOCUMENT_DELETED: 'document:deleted',

  KNOWLEDGE_GRAPH_BUILT: 'knowledge-graph:built',
  KNOWLEDGE_GRAPH_UPDATED: 'knowledge-graph:updated',

  CACHE_CLEAR: 'cache:clear',
  CACHE_INVALIDATE: 'cache:invalidate',
} as const;
```

这个API集成指南提供了完整的前端与agraph API交互的架构设计，包括错误处理、缓存策略、事件系统等核心功能，确保前端应用能够稳定、高效地与后端API进行通信。
