import { apiClient, ApiResponse } from './api';
import type {
  SearchRequest,
  SearchResult,
  Entity,
  Relation,
  TextChunk,
  BaseApiResponse,
} from '@/types/api';

export interface SearchResponse extends BaseApiResponse {
  data?: {
    results: SearchResult[];
    total: number;
    query: string;
    search_type: string;
    execution_time: number;
  };
}

export interface AdvancedSearchRequest {
  query: string;
  search_types: Array<'entities' | 'relations' | 'text_chunks'>;
  filters: {
    entity_types?: string[];
    relation_types?: string[];
    min_confidence?: number;
    date_range?: {
      start: string;
      end: string;
    };
    sources?: string[];
    tags?: string[];
  };
  sort_by?: 'relevance' | 'confidence' | 'date';
  sort_order?: 'asc' | 'desc';
  top_k?: number;
}

export interface AdvancedSearchResponse extends BaseApiResponse {
  data?: {
    entities: Entity[];
    relations: Relation[];
    text_chunks: TextChunk[];
    total_results: number;
    query: string;
    execution_time: number;
  };
}

export interface SearchSuggestion {
  text: string;
  type: 'entity' | 'relation' | 'concept';
  confidence: number;
}

class SearchService {
  private readonly baseEndpoint = '/search';

  async search(request: SearchRequest): Promise<ApiResponse<SearchResponse>> {
    return apiClient.post<SearchResponse>(`${this.baseEndpoint}`, request);
  }

  async advancedSearch(
    request: AdvancedSearchRequest
  ): Promise<ApiResponse<AdvancedSearchResponse>> {
    return apiClient.post<AdvancedSearchResponse>(
      `${this.baseEndpoint}/advanced`,
      request
    );
  }

  async searchEntities(
    query: string,
    options?: {
      entity_types?: string[];
      min_confidence?: number;
      top_k?: number;
    }
  ): Promise<ApiResponse<{ entities: Entity[]; total: number }>> {
    return apiClient.post(`${this.baseEndpoint}/entities`, {
      query,
      ...options,
    });
  }

  async searchRelations(
    query: string,
    options?: {
      relation_types?: string[];
      min_confidence?: number;
      top_k?: number;
    }
  ): Promise<ApiResponse<{ relations: Relation[]; total: number }>> {
    return apiClient.post(`${this.baseEndpoint}/relations`, {
      query,
      ...options,
    });
  }

  async searchTextChunks(
    query: string,
    options?: {
      sources?: string[];
      top_k?: number;
    }
  ): Promise<ApiResponse<{ text_chunks: TextChunk[]; total: number }>> {
    return apiClient.post(`${this.baseEndpoint}/text-chunks`, {
      query,
      ...options,
    });
  }

  async getSuggestions(
    query: string
  ): Promise<ApiResponse<{ suggestions: SearchSuggestion[] }>> {
    return apiClient.post(`${this.baseEndpoint}/suggestions`, { query });
  }

  async getPopularSearches(): Promise<
    ApiResponse<{ searches: Array<{ query: string; count: number }> }>
  > {
    return apiClient.get(`${this.baseEndpoint}/popular`, { cache: true });
  }

  async getSearchHistory(): Promise<
    ApiResponse<{ history: Array<{ query: string; timestamp: string }> }>
  > {
    return apiClient.get(`${this.baseEndpoint}/history`, { cache: true });
  }

  async saveSearch(
    query: string,
    results: SearchResult[]
  ): Promise<ApiResponse<BaseApiResponse>> {
    return apiClient.post(`${this.baseEndpoint}/save`, {
      query,
      results,
      timestamp: new Date().toISOString(),
    });
  }

  async deleteSearchHistory(): Promise<ApiResponse<BaseApiResponse>> {
    return apiClient.delete(`${this.baseEndpoint}/history`);
  }

  async similaritySearch(
    entityId: string,
    options?: {
      top_k?: number;
      include_relations?: boolean;
    }
  ): Promise<
    ApiResponse<{
      similar_entities: Entity[];
      related_entities?: Entity[];
      relations?: Relation[];
    }>
  > {
    return apiClient.post(`${this.baseEndpoint}/similarity`, {
      entity_id: entityId,
      ...options,
    });
  }

  async semanticSearch(
    query: string,
    options?: {
      search_types?: Array<'entities' | 'relations' | 'text_chunks'>;
      top_k?: number;
      threshold?: number;
    }
  ): Promise<ApiResponse<AdvancedSearchResponse>> {
    return apiClient.post(`${this.baseEndpoint}/semantic`, {
      query,
      ...options,
    });
  }

  buildAdvancedSearchRequest(params: {
    query: string;
    entityTypes?: string[];
    relationTypes?: string[];
    minConfidence?: number;
    sources?: string[];
    tags?: string[];
    dateRange?: { start: string; end: string };
    sortBy?: 'relevance' | 'confidence' | 'date';
    sortOrder?: 'asc' | 'desc';
    topK?: number;
  }): AdvancedSearchRequest {
    return {
      query: params.query,
      search_types: ['entities', 'relations', 'text_chunks'],
      filters: {
        entity_types: params.entityTypes,
        relation_types: params.relationTypes,
        min_confidence: params.minConfidence,
        sources: params.sources,
        tags: params.tags,
        date_range: params.dateRange,
      },
      sort_by: params.sortBy || 'relevance',
      sort_order: params.sortOrder || 'desc',
      top_k: params.topK || 10,
    };
  }

  highlightSearchTerms(text: string, query: string): string {
    if (!query.trim()) return text;

    const terms = query.toLowerCase().split(/\s+/);
    let highlighted = text;

    terms.forEach(term => {
      if (term.length > 2) {
        const regex = new RegExp(`(${term})`, 'gi');
        highlighted = highlighted.replace(regex, '<mark>$1</mark>');
      }
    });

    return highlighted;
  }

  extractSearchKeywords(query: string): string[] {
    return query
      .toLowerCase()
      .replace(/[^\w\s]/g, ' ')
      .split(/\s+/)
      .filter(word => word.length > 2)
      .slice(0, 10); // Limit to 10 keywords
  }

  formatSearchResult(result: SearchResult): string {
    switch (result.type) {
      case 'entity':
        return `${result.content} (${result.metadata?.entity_type || 'Entity'})`;
      case 'relation':
        return `${result.content} (${result.metadata?.relation_type || 'Relation'})`;
      case 'text_chunk':
        return (
          result.content.slice(0, 200) +
          (result.content.length > 200 ? '...' : '')
        );
      default:
        return result.content;
    }
  }

  validateSearchQuery(query: string): { valid: boolean; error?: string } {
    if (!query || query.trim().length === 0) {
      return { valid: false, error: '搜索查询不能为空' };
    }

    if (query.length > 500) {
      return { valid: false, error: '搜索查询不能超过 500 个字符' };
    }

    // Check for potentially dangerous patterns
    const dangerousPatterns = /<script|javascript:|data:/i;
    if (dangerousPatterns.test(query)) {
      return { valid: false, error: '搜索查询包含不安全的内容' };
    }

    return { valid: true };
  }

  clearCache(): void {
    apiClient.clearCache();
  }
}

export const searchService = new SearchService();
