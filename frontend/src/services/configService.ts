import { apiClient, ApiResponse } from './api';
import type { ConfigUpdateRequest, ConfigResponse } from '@/types/config';

class ConfigService {
  private readonly baseEndpoint = '/config';

  async getProjectConfig(
    projectName: string
  ): Promise<ApiResponse<ConfigResponse>> {
    return apiClient.get<ConfigResponse>(
      `${this.baseEndpoint}?project_name=${encodeURIComponent(projectName)}`,
      { cache: false }
    );
  }

  async updateProjectConfig(
    projectName: string,
    updates: ConfigUpdateRequest
  ): Promise<ApiResponse<ConfigResponse>> {
    return apiClient.post<ConfigResponse>(
      `${this.baseEndpoint}?project_name=${encodeURIComponent(projectName)}`,
      updates
    );
  }

  async resetProjectConfig(
    projectName: string
  ): Promise<ApiResponse<ConfigResponse>> {
    return apiClient.post<ConfigResponse>(
      `${this.baseEndpoint}/reset?project_name=${encodeURIComponent(projectName)}`,
      {}
    );
  }

  async getProjectInfo(projectName: string): Promise<ApiResponse<any>> {
    return apiClient.get<any>(
      `${this.baseEndpoint}/projects/${encodeURIComponent(projectName)}/info`,
      { cache: false }
    );
  }

  async copyProjectConfig(
    projectName: string,
    sourceProject: string
  ): Promise<ApiResponse<ConfigResponse>> {
    return apiClient.post<ConfigResponse>(
      `${this.baseEndpoint}/projects/${encodeURIComponent(projectName)}/copy-from/${encodeURIComponent(sourceProject)}`,
      {}
    );
  }

  async saveConfigToFile(
    projectName: string,
    filePath?: string
  ): Promise<ApiResponse<any>> {
    return apiClient.post<any>(
      `${this.baseEndpoint}/save?project_name=${encodeURIComponent(projectName)}`,
      { file_path: filePath }
    );
  }

  async loadConfigFromFile(
    projectName: string,
    filePath?: string
  ): Promise<ApiResponse<any>> {
    return apiClient.post<any>(
      `${this.baseEndpoint}/load?project_name=${encodeURIComponent(projectName)}`,
      { file_path: filePath }
    );
  }
}

export const configService = new ConfigService();
export default configService;
