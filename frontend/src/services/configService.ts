import { apiClient, ApiResponse } from './api';
import type {
  ConfigUpdateRequest,
  ConfigResponse,
  BackupStatus,
  ConfigFileRequest,
  ConfigFileResponse
} from '@/types/config';

class ConfigService {
  private readonly baseEndpoint = '/config';

  /**
   * Get project configuration with automatic recovery support
   */
  async getProjectConfig(
    projectName: string
  ): Promise<ApiResponse<ConfigResponse>> {
    return apiClient.get<ConfigResponse>(
      `${this.baseEndpoint}?project_name=${encodeURIComponent(projectName)}`,
      { cache: false }
    );
  }

  /**
   * Get global configuration (no project context)
   */
  async getGlobalConfig(): Promise<ApiResponse<ConfigResponse>> {
    return apiClient.get<ConfigResponse>(
      this.baseEndpoint,
      { cache: false }
    );
  }

  /**
   * Update project configuration with automatic backup sync
   */
  async updateProjectConfig(
    projectName: string,
    updates: ConfigUpdateRequest
  ): Promise<ApiResponse<ConfigResponse>> {
    return apiClient.post<ConfigResponse>(
      `${this.baseEndpoint}?project_name=${encodeURIComponent(projectName)}`,
      updates
    );
  }

  /**
   * Update global configuration
   */
  async updateGlobalConfig(
    updates: ConfigUpdateRequest
  ): Promise<ApiResponse<ConfigResponse>> {
    return apiClient.post<ConfigResponse>(
      this.baseEndpoint,
      updates
    );
  }

  /**
   * Reset project configuration to default values
   */
  async resetProjectConfig(
    projectName: string
  ): Promise<ApiResponse<ConfigResponse>> {
    return apiClient.post<ConfigResponse>(
      `${this.baseEndpoint}/reset?project_name=${encodeURIComponent(projectName)}`,
      {}
    );
  }

  /**
   * Reset global configuration to default values
   */
  async resetGlobalConfig(): Promise<ApiResponse<ConfigResponse>> {
    return apiClient.post<ConfigResponse>(
      `${this.baseEndpoint}/reset`,
      {}
    );
  }

  /**
   * Get project configuration information (legacy endpoint)
   */
  async getProjectInfo(projectName: string): Promise<ApiResponse<any>> {
    return apiClient.get<any>(
      `${this.baseEndpoint}/projects/${encodeURIComponent(projectName)}/info`,
      { cache: false }
    );
  }

  /**
   * Get project backup status and details
   */
  async getProjectBackupStatus(
    projectName: string
  ): Promise<ApiResponse<{ data: BackupStatus }>> {
    return apiClient.get<{ data: BackupStatus }>(
      `${this.baseEndpoint}/projects/${encodeURIComponent(projectName)}/backup-status`,
      { cache: false }
    );
  }

  /**
   * Load project configuration from backup file
   */
  async loadProjectConfigFromBackup(
    projectName: string
  ): Promise<ApiResponse<ConfigResponse>> {
    return apiClient.post<ConfigResponse>(
      `${this.baseEndpoint}/projects/${encodeURIComponent(projectName)}/load-from-backup`,
      {}
    );
  }

  /**
   * Copy configuration from one project to another
   */
  async copyProjectConfig(
    projectName: string,
    sourceProject: string
  ): Promise<ApiResponse<ConfigResponse>> {
    return apiClient.post<ConfigResponse>(
      `${this.baseEndpoint}/projects/${encodeURIComponent(projectName)}/copy-from/${encodeURIComponent(sourceProject)}`,
      {}
    );
  }

  /**
   * Save current project configuration to file
   */
  async saveProjectConfigToFile(
    projectName: string,
    filePath?: string
  ): Promise<ApiResponse<ConfigFileResponse>> {
    return apiClient.post<ConfigFileResponse>(
      `${this.baseEndpoint}/save?project_name=${encodeURIComponent(projectName)}`,
      { file_path: filePath }
    );
  }

  /**
   * Save current global configuration to file
   */
  async saveGlobalConfigToFile(
    filePath?: string
  ): Promise<ApiResponse<ConfigFileResponse>> {
    return apiClient.post<ConfigFileResponse>(
      `${this.baseEndpoint}/save`,
      { file_path: filePath }
    );
  }

  /**
   * Load project configuration from file
   */
  async loadProjectConfigFromFile(
    projectName: string,
    filePath?: string
  ): Promise<ApiResponse<ConfigFileResponse>> {
    return apiClient.post<ConfigFileResponse>(
      `${this.baseEndpoint}/load?project_name=${encodeURIComponent(projectName)}`,
      { file_path: filePath }
    );
  }

  /**
   * Load global configuration from file
   */
  async loadGlobalConfigFromFile(
    filePath?: string
  ): Promise<ApiResponse<ConfigFileResponse>> {
    return apiClient.post<ConfigFileResponse>(
      `${this.baseEndpoint}/load`,
      { file_path: filePath }
    );
  }

  /**
   * Get configuration file path for project or global
   */
  async getConfigFilePath(
    projectName?: string
  ): Promise<ApiResponse<ConfigFileResponse>> {
    const params = projectName
      ? `?project_name=${encodeURIComponent(projectName)}`
      : '';

    return apiClient.get<ConfigFileResponse>(
      `${this.baseEndpoint}/file-path${params}`,
      { cache: false }
    );
  }

  // Utility methods for builder configuration

  /**
   * Update only builder configuration for a project
   */
  async updateBuilderConfig(
    projectName: string,
    builderConfig: Partial<{
      enable_cache: boolean;
      cache_dir: string;
      cache_ttl: number;
      auto_cleanup: boolean;
      chunk_size: number;
      chunk_overlap: number;
      entity_confidence_threshold: number;
      relation_confidence_threshold: number;
      cluster_algorithm: string;
      min_cluster_size: number;
      enable_user_interaction: boolean;
      auto_save_edits: boolean;
    }>
  ): Promise<ApiResponse<ConfigResponse>> {
    // Convert builder config to request format
    const updates: ConfigUpdateRequest = {};

    if (builderConfig.enable_cache !== undefined) {
      updates.builder_enable_cache = builderConfig.enable_cache;
    }
    if (builderConfig.cache_dir !== undefined) {
      updates.builder_cache_dir = builderConfig.cache_dir;
    }
    if (builderConfig.cache_ttl !== undefined) {
      updates.builder_cache_ttl = builderConfig.cache_ttl;
    }
    if (builderConfig.auto_cleanup !== undefined) {
      updates.builder_auto_cleanup = builderConfig.auto_cleanup;
    }
    if (builderConfig.chunk_size !== undefined) {
      updates.builder_chunk_size = builderConfig.chunk_size;
    }
    if (builderConfig.chunk_overlap !== undefined) {
      updates.builder_chunk_overlap = builderConfig.chunk_overlap;
    }
    if (builderConfig.entity_confidence_threshold !== undefined) {
      updates.builder_entity_confidence_threshold = builderConfig.entity_confidence_threshold;
    }
    if (builderConfig.relation_confidence_threshold !== undefined) {
      updates.builder_relation_confidence_threshold = builderConfig.relation_confidence_threshold;
    }
    if (builderConfig.cluster_algorithm !== undefined) {
      updates.builder_cluster_algorithm = builderConfig.cluster_algorithm;
    }
    if (builderConfig.min_cluster_size !== undefined) {
      updates.builder_min_cluster_size = builderConfig.min_cluster_size;
    }
    if (builderConfig.enable_user_interaction !== undefined) {
      updates.builder_enable_user_interaction = builderConfig.enable_user_interaction;
    }
    if (builderConfig.auto_save_edits !== undefined) {
      updates.builder_auto_save_edits = builderConfig.auto_save_edits;
    }

    return this.updateProjectConfig(projectName, updates);
  }

  /**
   * Validate configuration before updating
   */
  validateConfigUpdate(updates: ConfigUpdateRequest): { valid: boolean; errors: string[] } {
    const errors: string[] = [];

    // Validate temperature range
    if (updates.llm_temperature !== undefined &&
        (updates.llm_temperature < 0 || updates.llm_temperature > 2)) {
      errors.push('LLM temperature must be between 0 and 2');
    }

    // Validate token limits
    if (updates.llm_max_tokens !== undefined && updates.llm_max_tokens <= 0) {
      errors.push('LLM max tokens must be positive');
    }

    // Validate chunk sizes
    if (updates.max_chunk_size !== undefined && updates.max_chunk_size <= 0) {
      errors.push('Chunk size must be positive');
    }

    if (updates.chunk_overlap !== undefined && updates.chunk_overlap < 0) {
      errors.push('Chunk overlap cannot be negative');
    }

    // Validate builder confidence thresholds
    if (updates.builder_entity_confidence_threshold !== undefined &&
        (updates.builder_entity_confidence_threshold < 0 || updates.builder_entity_confidence_threshold > 1)) {
      errors.push('Entity confidence threshold must be between 0 and 1');
    }

    if (updates.builder_relation_confidence_threshold !== undefined &&
        (updates.builder_relation_confidence_threshold < 0 || updates.builder_relation_confidence_threshold > 1)) {
      errors.push('Relation confidence threshold must be between 0 and 1');
    }

    // Validate cache TTL
    if (updates.builder_cache_ttl !== undefined && updates.builder_cache_ttl <= 0) {
      errors.push('Cache TTL must be positive');
    }

    return {
      valid: errors.length === 0,
      errors
    };
  }
}

export const configService = new ConfigService();
export default configService;
