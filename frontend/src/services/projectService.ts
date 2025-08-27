import { apiClient, ApiResponse } from './api';
import type {
  Project,
  ProjectCreateRequest,
  ProjectSwitchRequest,
  ProjectDeleteRequest,
  BaseApiResponse,
} from '@/types/api';
import type { KnowledgeGraphBuildRequest } from '@/modules/Projects/types/project';

export interface ProjectStatistics {
  document_count: number;
  has_vector_db: boolean;
  size_mb: number;
  entity_count?: number;
  relation_count?: number;
  error?: string;
}

export interface ProjectListResponse extends BaseApiResponse {
  projects: string[];
  data?: {
    current_project?: string;
    project_details?: Record<string, any>;
    project_statistics?: Record<string, ProjectStatistics>;
    total_count?: number;
  };
}

export interface ProjectResponse extends BaseApiResponse {
  data?: {
    project_name?: string;
    project_details?: Project;
    current_project?: string;
  };
}

export interface KnowledgeGraphBuildResponse extends BaseApiResponse {
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

class ProjectService {
  private readonly baseEndpoint = '/projects';

  async createProject(
    request: ProjectCreateRequest
  ): Promise<ApiResponse<ProjectResponse>> {
    return apiClient.post<ProjectResponse>(
      `${this.baseEndpoint}/create`,
      request
    );
  }

  async listProjects(
    includeStats: boolean = false,
    useCache: boolean = true
  ): Promise<ApiResponse<ProjectListResponse>> {
    const endpoint = includeStats
      ? this.baseEndpoint
      : `${this.baseEndpoint}/list`;

    const params = includeStats ? '?include_stats=true' : '';

    return apiClient.get<ProjectListResponse>(`${endpoint}${params}`, {
      cache: useCache,
    });
  }

  async getCurrentProject(
    useCache: boolean = true
  ): Promise<ApiResponse<ProjectResponse>> {
    return apiClient.get<ProjectResponse>(`${this.baseEndpoint}/current`, {
      cache: useCache,
    });
  }

  async switchProject(
    request: ProjectSwitchRequest
  ): Promise<ApiResponse<ProjectResponse>> {
    return apiClient.post<ProjectResponse>(
      `${this.baseEndpoint}/switch`,
      request
    );
  }

  async deleteProject(
    request: ProjectDeleteRequest
  ): Promise<ApiResponse<ProjectResponse>> {
    return apiClient.post<ProjectResponse>(
      `${this.baseEndpoint}/delete`,
      request
    );
  }

  async getProjectDetails(
    projectName: string
  ): Promise<ApiResponse<ProjectResponse>> {
    return apiClient.get<ProjectResponse>(
      `${this.baseEndpoint}/${encodeURIComponent(projectName)}`,
      {
        cache: true,
      }
    );
  }

  async getProjectStats(projectName?: string): Promise<ApiResponse<any>> {
    const endpoint = projectName
      ? `${this.baseEndpoint}/${encodeURIComponent(projectName)}/stats`
      : `${this.baseEndpoint}/stats`;

    return apiClient.get(endpoint, { cache: true });
  }

  validateProjectName(name: string): boolean {
    if (!name || name.length === 0 || name.length > 50) {
      return false;
    }

    // Check for invalid characters
    const invalidChars = /[<>:"/\\|?*]/;
    if (invalidChars.test(name)) {
      return false;
    }

    // Check for reserved names
    const reservedNames = ['con', 'prn', 'aux', 'nul'];
    if (reservedNames.includes(name.toLowerCase())) {
      return false;
    }

    return true;
  }

  async checkProjectExists(name: string): Promise<boolean> {
    try {
      const response = await this.listProjects();
      return response.data?.projects?.includes(name) || false;
    } catch {
      return false;
    }
  }

  async exportProject(
    projectName: string,
    format: 'json' | 'yaml' = 'json'
  ): Promise<ApiResponse<any>> {
    return apiClient.get(
      `${this.baseEndpoint}/${encodeURIComponent(projectName)}/export`,
      {
        headers: {
          Accept: format === 'json' ? 'application/json' : 'application/x-yaml',
        },
      }
    );
  }

  async importProject(
    projectData: any,
    overwrite = false
  ): Promise<ApiResponse<ProjectResponse>> {
    return apiClient.post<ProjectResponse>(`${this.baseEndpoint}/import`, {
      project_data: projectData,
      overwrite,
    });
  }

  async buildKnowledgeGraph(
    projectName: string,
    request: KnowledgeGraphBuildRequest
  ): Promise<ApiResponse<KnowledgeGraphBuildResponse>> {
    return apiClient.post<KnowledgeGraphBuildResponse>(
      '/knowledge-graph/build',
      request,
      {
        params: {
          project_name: projectName,
        },
      }
    );
  }

  clearCache(): void {
    apiClient.clearCache();
  }
}

export const projectService = new ProjectService();
