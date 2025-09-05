// Export all services
export { apiClient } from './api';
import { apiClient } from './api';
export { projectService } from './projectService';
export { documentService } from './documentService';
export { knowledgeGraphService } from './knowledgeGraphService';
export { chatService } from './chatService';
export { searchService } from './searchService';
export { configService } from './configService';
export { configManager } from './configManager';

// Export state management utilities
export {
  requestStateManager,
  createRequestHandler,
  requestHandlers,
} from './requestStateManager';
export type { RequestState, RequestOptions } from './requestStateManager';

// Export types
export type { ApiResponse, ApiError, RequestConfig } from './api';
export type { ProjectResponse, ProjectListResponse } from './projectService';
export type {
  DocumentUploadResponse,
  DocumentListResponse,
  UploadProgress,
} from './documentService';
export type {
  KnowledgeGraphResponse,
  BuildResponse,
  GraphStatsResponse,
  GraphVisualizationData,
} from './knowledgeGraphService';
export type {
  ChatHistoryResponse,
  ConversationResponse,
  StreamEventData,
} from './chatService';
export type {
  SearchResponse,
  AdvancedSearchRequest,
  AdvancedSearchResponse,
  SearchSuggestion,
} from './searchService';
// Config service types
export type {
  ConfigResponse,
  ConfigUpdateRequest,
  BackupStatus,
  ConfigFileRequest,
  ConfigFileResponse,
  BuilderConfig,
  ProjectSettings
} from '@/types/config';

// Service manager for centralized control
class ServiceManager {
  clearAllCaches(): void {
    apiClient.clearCache();
  }

  setGlobalTimeout(timeout: number): void {
    // This would require modifying the ApiClient to support global timeout settings
    console.log(`Setting global timeout to ${timeout}ms`);
  }

  setupGlobalInterceptors(): void {
    // Add common request interceptor for authentication
    apiClient.addRequestInterceptor((config: any) => {
      // Add timestamp to prevent caching issues
      if (config.url) {
        const url = new URL(config.url, window.location.origin);
        url.searchParams.set('_t', Date.now().toString());
        config.url = url.toString();
      }

      return config;
    });

    // Add common response interceptor for error handling
    apiClient.addResponseInterceptor({
      onFulfilled: (response: any) => {
        // Log successful responses in development
        if (process.env.NODE_ENV === 'development') {
          console.log(`✅ ${response.url}:`, response.status);
        }
        return response;
      },
      onRejected: (error: any) => {
        // Log errors in development
        if (process.env.NODE_ENV === 'development') {
          console.error(`❌ API Error:`, error);
        }
        throw error;
      },
    });
  }

  async checkConnectivity(): Promise<boolean> {
    try {
      const response = await apiClient.get('/health', { timeout: 5000 });
      return response.success;
    } catch {
      return false;
    }
  }

  getServiceStatus(): Record<string, boolean> {
    return {
      projects: true,
      documents: true,
      knowledgeGraph: true,
      chat: true,
      search: true,
      config: true,
    };
  }
}

export const serviceManager = new ServiceManager();

// Initialize global interceptors
serviceManager.setupGlobalInterceptors();
