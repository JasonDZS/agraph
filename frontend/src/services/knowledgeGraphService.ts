import { apiClient, ApiResponse } from './api';
import { useProjectStore } from '@/store/projectStore';
import { projectSync } from '@/utils/projectSync';
import type {
  KnowledgeGraph,
  KnowledgeGraphBuildRequest,
  KnowledgeGraphUpdateRequest,
  Entity,
  Relation,
  TextChunk,
  Cluster,
  BuildStatus,
  BaseApiResponse,
} from '@/types/api';

export interface KnowledgeGraphResponse extends BaseApiResponse {
  data?: {
    graph_name: string;
    graph_description: string;
    entities: Entity[];
    relations: Relation[];
    clusters?: Cluster[];
    text_chunks?: TextChunk[];
  };
}

export interface BuildResponse extends BaseApiResponse {
  data?: {
    graph_name: string;
    graph_description: string;
    entities_count: number;
    relations_count: number;
    clusters_count: number;
    text_chunks_count: number;
    source_documents: Array<{
      id: string;
      filename: string;
      content_length: number;
    }>;
    total_texts_processed: number;
    from_stored_documents: number;
    from_direct_texts: number;
  };
}

export interface GraphStatsResponse extends BaseApiResponse {
  data?: {
    exists: boolean;
    graph_name: string | null;
    graph_description: string | null;
    statistics: {
      entities: number;
      relations: number;
      clusters: number;
      text_chunks: number;
    };
    entity_types?: Record<string, number>;
    relation_types?: Record<string, number>;
    system_info?: {
      agraph_initialized: boolean;
      vector_store_type: string;
      enable_knowledge_graph: boolean;
    };
  };
}

export interface GraphVisualizationData {
  nodes: Array<{
    id: string;
    label: string;
    type: 'entity' | 'cluster';
    entityType?: string;
    confidence?: number;
    properties?: Record<string, any>;
    x?: number;
    y?: number;
  }>;
  edges: Array<{
    id: string;
    source: string;
    target: string;
    label: string;
    relationType: string;
    confidence: number;
    properties?: Record<string, any>;
  }>;
  text_chunks?: TextChunk[];
}

class KnowledgeGraphService {
  private readonly baseEndpoint = '/knowledge-graph';

  private async getCurrentProjectName(): Promise<string | null> {
    // First try to get from store
    let projectName = useProjectStore.getState().currentProject?.name;

    // If not found, try to sync with backend
    if (!projectName) {
      projectName = await projectSync.ensureProjectSync();
    }

    return projectName;
  }

  private async addProjectParam(url: string): Promise<string> {
    const projectName = await this.getCurrentProjectName();
    if (!projectName) return url;

    const separator = url.includes('?') ? '&' : '?';
    return `${url}${separator}project_name=${encodeURIComponent(projectName)}`;
  }

  async buildGraph(
    request: KnowledgeGraphBuildRequest
  ): Promise<ApiResponse<BuildResponse>> {
    return apiClient.post<BuildResponse>(
      await this.addProjectParam(`${this.baseEndpoint}/build`),
      request,
      {
        timeout: 600000, // 10 minutes for graph building
      }
    );
  }

  async updateGraph(
    request: KnowledgeGraphUpdateRequest
  ): Promise<ApiResponse<BuildResponse>> {
    return apiClient.post<BuildResponse>(
      await this.addProjectParam(`${this.baseEndpoint}/update`),
      request,
      {
        timeout: 600000,
      }
    );
  }

  async getGraph(options?: {
    include_text_chunks?: boolean;
    include_clusters?: boolean;
    entity_limit?: number;
    relation_limit?: number;
  }): Promise<ApiResponse<KnowledgeGraphResponse>> {
    const params = new URLSearchParams();

    if (options) {
      if (options.include_text_chunks !== undefined)
        params.append(
          'include_text_chunks',
          options.include_text_chunks.toString()
        );
      if (options.include_clusters !== undefined)
        params.append('include_clusters', options.include_clusters.toString());
      if (options.entity_limit)
        params.append('entity_limit', options.entity_limit.toString());
      if (options.relation_limit)
        params.append('relation_limit', options.relation_limit.toString());
    }

    const endpoint = params.toString()
      ? `${this.baseEndpoint}/get?${params.toString()}`
      : `${this.baseEndpoint}/get`;

    return apiClient.get<KnowledgeGraphResponse>(
      await this.addProjectParam(endpoint),
      {
        cache: true,
      }
    );
  }

  async getGraphStatus(): Promise<ApiResponse<GraphStatsResponse>> {
    return apiClient.get<GraphStatsResponse>(
      await this.addProjectParam(`${this.baseEndpoint}/status`),
      {
        cache: false,
      }
    );
  }

  async getGraphStats(): Promise<ApiResponse<GraphStatsResponse>> {
    // Use the status endpoint since it now provides comprehensive stats
    return this.getGraphStatus();
  }

  async getEntities(filters?: {
    entity_type?: string;
    min_confidence?: number;
    search?: string;
    limit?: number;
    offset?: number;
  }): Promise<ApiResponse<{ entities: Entity[]; total: number }>> {
    // Use the main getGraph method and filter on the client side for now
    const graphResponse = await this.getGraph({
      entity_limit: filters?.limit || 1000,
    });

    if (graphResponse.success && graphResponse.data?.data?.entities) {
      let entities = graphResponse.data.data.entities;

      // Apply client-side filtering if needed
      if (filters) {
        if (filters.entity_type) {
          entities = entities.filter(
            e => e.entity_type === filters.entity_type
          );
        }
        if (filters.min_confidence !== undefined) {
          entities = entities.filter(
            e => e.confidence >= filters.min_confidence!
          );
        }
        if (filters.search) {
          const searchLower = filters.search.toLowerCase();
          entities = entities.filter(
            e =>
              e.name.toLowerCase().includes(searchLower) ||
              e.description?.toLowerCase().includes(searchLower)
          );
        }
        if (filters.offset) {
          entities = entities.slice(filters.offset);
        }
        if (filters.limit) {
          entities = entities.slice(0, filters.limit);
        }
      }

      return {
        success: true,
        data: { entities, total: entities.length },
      };
    }

    return {
      success: false,
      error: graphResponse.error || 'Failed to get entities',
      data: { entities: [], total: 0 },
    };
  }

  async getEntity(entityId: string): Promise<ApiResponse<Entity>> {
    // Get entity from the full graph data
    const graphResponse = await this.getGraph();

    if (graphResponse.success && graphResponse.data?.data?.entities) {
      const entity = graphResponse.data.data.entities.find(
        e => e.id === entityId
      );
      if (entity) {
        return {
          success: true,
          data: entity,
        };
      }
    }

    return {
      success: false,
      error: 'Entity not found',
      data: undefined,
    };
  }

  async getEntityRelations(
    entityId: string
  ): Promise<ApiResponse<{ relations: Relation[] }>> {
    // Get relations from the full graph data
    const graphResponse = await this.getGraph();

    if (graphResponse.success && graphResponse.data?.data?.relations) {
      const relations = graphResponse.data.data.relations.filter(
        r => r.head_entity_id === entityId || r.tail_entity_id === entityId
      );

      return {
        success: true,
        data: { relations },
      };
    }

    return {
      success: false,
      error: 'Failed to get entity relations',
      data: { relations: [] },
    };
  }

  async getRelations(filters?: {
    relation_type?: string;
    head_entity_id?: string;
    tail_entity_id?: string;
    min_confidence?: number;
    limit?: number;
    offset?: number;
  }): Promise<ApiResponse<{ relations: Relation[]; total: number }>> {
    // Use the main getGraph method and filter on the client side
    const graphResponse = await this.getGraph({
      relation_limit: filters?.limit || 1000,
    });

    if (graphResponse.success && graphResponse.data?.data?.relations) {
      let relations = graphResponse.data.data.relations;

      // Apply client-side filtering if needed
      if (filters) {
        if (filters.relation_type) {
          relations = relations.filter(
            r => r.relation_type === filters.relation_type
          );
        }
        if (filters.head_entity_id) {
          relations = relations.filter(
            r => r.head_entity_id === filters.head_entity_id
          );
        }
        if (filters.tail_entity_id) {
          relations = relations.filter(
            r => r.tail_entity_id === filters.tail_entity_id
          );
        }
        if (filters.min_confidence !== undefined) {
          relations = relations.filter(
            r => r.confidence >= filters.min_confidence!
          );
        }
        if (filters.offset) {
          relations = relations.slice(filters.offset);
        }
        if (filters.limit) {
          relations = relations.slice(0, filters.limit);
        }
      }

      return {
        success: true,
        data: { relations, total: relations.length },
      };
    }

    return {
      success: false,
      error: graphResponse.error || 'Failed to get relations',
      data: { relations: [], total: 0 },
    };
  }

  async getRelation(relationId: string): Promise<ApiResponse<Relation>> {
    // Get relation from the full graph data
    const graphResponse = await this.getGraph();

    if (graphResponse.success && graphResponse.data?.data?.relations) {
      const relation = graphResponse.data.data.relations.find(
        r => r.id === relationId
      );
      if (relation) {
        return {
          success: true,
          data: relation,
        };
      }
    }

    return {
      success: false,
      error: 'Relation not found',
      data: undefined,
    };
  }

  async getClusters(): Promise<ApiResponse<{ clusters: Cluster[] }>> {
    // Use the main getGraph method with clusters included
    const graphResponse = await this.getGraph({ include_clusters: true });

    if (graphResponse.success && graphResponse.data?.data?.clusters) {
      return {
        success: true,
        data: { clusters: graphResponse.data.data.clusters },
      };
    }

    return {
      success: false,
      error: graphResponse.error || 'Failed to get clusters',
      data: { clusters: [] },
    };
  }

  async getCluster(clusterId: string): Promise<ApiResponse<Cluster>> {
    // Get cluster from the full graph data
    const graphResponse = await this.getGraph({ include_clusters: true });

    if (graphResponse.success && graphResponse.data?.data?.clusters) {
      const cluster = graphResponse.data.data.clusters.find(
        c => c.id === clusterId
      );
      if (cluster) {
        return {
          success: true,
          data: cluster,
        };
      }
    }

    return {
      success: false,
      error: 'Cluster not found',
      data: undefined,
    };
  }

  async getTextChunks(filters?: {
    search?: string;
    entity_id?: string;
    limit?: number;
    offset?: number;
  }): Promise<
    ApiResponse<{
      data: { text_chunks: TextChunk[]; pagination: any; filters: any };
    }>
  > {
    const request = {
      search: filters?.search,
      entity_id: filters?.entity_id,
      limit: filters?.limit ?? 20,
      offset: filters?.offset ?? 0,
    };

    const response = await apiClient.post<{
      data: { text_chunks: TextChunk[]; pagination: any; filters: any };
    }>(
      await this.addProjectParam(`${this.baseEndpoint}/text-chunks`),
      request,
      { cache: true }
    );

    // The backend response already contains the correct structure
    return response;
  }

  async getVisualizationData(options?: {
    include_clusters?: boolean;
    include_text_chunks?: boolean;
    max_entities?: number;
    max_relations?: number;
    min_confidence?: number;
    entity_types?: string[];
    relation_types?: string[];
    cluster_layout?: boolean;
    use_cache?: boolean;
  }): Promise<ApiResponse<{ data: GraphVisualizationData }>> {
    const request = {
      include_clusters: options?.include_clusters ?? true,
      include_text_chunks: options?.include_text_chunks ?? false,
      max_entities: options?.max_entities ?? 500,
      max_relations: options?.max_relations ?? 1000,
      min_confidence: options?.min_confidence ?? 0.0,
      entity_types: options?.entity_types,
      relation_types: options?.relation_types,
      cluster_layout: options?.cluster_layout ?? false,
    };

    const response = await apiClient.post<{ data: GraphVisualizationData }>(
      await this.addProjectParam(`${this.baseEndpoint}/visualization-data`),
      request,
      { cache: options?.use_cache !== false }
    );

    // The backend response structure contains nodes and edges directly in data
    return response;
  }

  async exportGraph(
    format: 'json' | 'graphml' | 'gexf' = 'json'
  ): Promise<ApiResponse<Blob>> {
    return apiClient.get(`${this.baseEndpoint}/export`, {
      headers: {
        Accept:
          format === 'json'
            ? 'application/json'
            : format === 'graphml'
              ? 'application/xml'
              : 'application/gexf+xml',
      },
    });
  }

  async importGraph(
    graphData: any,
    merge = false
  ): Promise<ApiResponse<BuildResponse>> {
    return apiClient.post<BuildResponse>(`${this.baseEndpoint}/import`, {
      graph_data: graphData,
      merge,
    });
  }

  async clearGraph(): Promise<ApiResponse<BaseApiResponse>> {
    return apiClient.post<BaseApiResponse>(`${this.baseEndpoint}/clear`);
  }

  async updateEntity(
    entityId: string,
    updates: Partial<Entity>
  ): Promise<ApiResponse<Entity>> {
    return apiClient.put<Entity>(
      `${this.baseEndpoint}/entities/${encodeURIComponent(entityId)}`,
      updates
    );
  }

  async deleteEntity(entityId: string): Promise<ApiResponse<BaseApiResponse>> {
    return apiClient.delete(
      `${this.baseEndpoint}/entities/${encodeURIComponent(entityId)}`
    );
  }

  async updateRelation(
    relationId: string,
    updates: Partial<Relation>
  ): Promise<ApiResponse<Relation>> {
    return apiClient.put<Relation>(
      `${this.baseEndpoint}/relations/${encodeURIComponent(relationId)}`,
      updates
    );
  }

  async deleteRelation(
    relationId: string
  ): Promise<ApiResponse<BaseApiResponse>> {
    return apiClient.delete(
      `${this.baseEndpoint}/relations/${encodeURIComponent(relationId)}`
    );
  }

  async createEntity(entity: Omit<Entity, 'id'>): Promise<ApiResponse<Entity>> {
    return apiClient.post<Entity>(`${this.baseEndpoint}/entities`, entity);
  }

  async createRelation(
    relation: Omit<Relation, 'id'>
  ): Promise<ApiResponse<Relation>> {
    return apiClient.post<Relation>(`${this.baseEndpoint}/relations`, relation);
  }

  async getEntityTypes(): Promise<string[]> {
    try {
      const response = await apiClient.get(`${this.baseEndpoint}/entity-types`);
      return response.data?.entity_types || [];
    } catch {
      return [
        'person',
        'organization',
        'location',
        'concept',
        'event',
        'object',
      ];
    }
  }

  async getRelationTypes(): Promise<string[]> {
    try {
      const response = await apiClient.get(
        `${this.baseEndpoint}/relation-types`
      );
      return response.data?.relation_types || [];
    } catch {
      return [
        'contains',
        'belongs_to',
        'references',
        'related_to',
        'located_in',
        'works_for',
      ];
    }
  }

  convertToVisualizationData(graph: KnowledgeGraph): GraphVisualizationData {
    const nodes = [
      ...graph.entities.map(entity => ({
        id: entity.id,
        label: entity.name,
        type: 'entity' as const,
        entityType: entity.entity_type,
        confidence: entity.confidence,
        properties: entity.properties,
      })),
      ...graph.clusters.map(cluster => ({
        id: cluster.id,
        label: cluster.name,
        type: 'cluster' as const,
        properties: { description: cluster.description },
      })),
    ];

    const edges = graph.relations.map(relation => ({
      id: relation.id,
      source: relation.head_entity_id,
      target: relation.tail_entity_id,
      label: relation.relation_type,
      relationType: relation.relation_type,
      confidence: relation.confidence,
      properties: relation.properties,
    }));

    return { nodes, edges };
  }

  clearCache(): void {
    apiClient.clearCache();
  }
}

export const knowledgeGraphService = new KnowledgeGraphService();
