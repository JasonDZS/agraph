import { configService } from './configService';
import { projectService } from './projectService';
import type {
  ConfigUpdateRequest,
  ProjectSettings,
  BuilderConfig,
  BackupStatus
} from '@/types/config';
import type { ApiResponse } from './api';

/**
 * Advanced configuration manager with unified Settings support
 * Provides high-level operations for managing project configurations
 */
class ConfigManager {

  /**
   * Get complete project configuration with backup status
   */
  async getProjectConfigWithStatus(projectName: string) {
    try {
      const [configResponse, backupStatusResponse] = await Promise.all([
        configService.getProjectConfig(projectName),
        configService.getProjectBackupStatus(projectName)
      ]);

      return {
        config: configResponse.data,
        backupStatus: backupStatusResponse.data.data,
        hasBackup: backupStatusResponse.data.data.backup_exists,
        isHealthy: configResponse.success && backupStatusResponse.success
      };
    } catch (error) {
      console.error('Failed to get project config with status:', error);
      throw error;
    }
  }

  /**
   * Update project configuration with validation and backup
   */
  async updateProjectConfigSafe(
    projectName: string,
    updates: ConfigUpdateRequest,
    options: { validateFirst?: boolean; createBackup?: boolean } = {}
  ) {
    const { validateFirst = true, createBackup = false } = options;

    try {
      // Validate configuration if requested
      if (validateFirst) {
        const validation = configService.validateConfigUpdate(updates);
        if (!validation.valid) {
          throw new Error(`Configuration validation failed: ${validation.errors.join(', ')}`);
        }
      }

      // Create backup if requested
      if (createBackup) {
        await configService.saveProjectConfigToFile(projectName);
      }

      // Update configuration
      const response = await configService.updateProjectConfig(projectName, updates);

      return {
        success: response.success,
        config: response.data,
        backupCreated: createBackup
      };

    } catch (error) {
      console.error('Failed to update project config safely:', error);
      throw error;
    }
  }

  /**
   * Recover project configuration from backup with verification
   */
  async recoverProjectConfigSafe(projectName: string) {
    try {
      // First check if backup exists and is valid
      const backupStatus = await configService.getProjectBackupStatus(projectName);

      if (!backupStatus.data.data.backup_exists) {
        throw new Error('No backup file found for project');
      }

      if (!backupStatus.data.data.has_complete_settings) {
        throw new Error('Backup file does not contain complete settings');
      }

      // Attempt recovery
      const recoveryResponse = await configService.loadProjectConfigFromBackup(projectName);

      if (recoveryResponse.success) {
        // Verify recovery by getting current config
        const verificationResponse = await configService.getProjectConfig(projectName);

        return {
          success: true,
          recoveredConfig: recoveryResponse.data,
          verifiedConfig: verificationResponse.data,
          backupInfo: backupStatus.data.data
        };
      }

      throw new Error('Recovery operation failed');

    } catch (error) {
      console.error('Failed to recover project config safely:', error);
      throw error;
    }
  }

  /**
   * Clone configuration from one project to another
   */
  async cloneProjectConfig(
    sourceProject: string,
    targetProject: string,
    options: {
      overrideProjectName?: boolean;
      excludeSections?: string[];
      createBackup?: boolean;
    } = {}
  ) {
    const {
      overrideProjectName = true,
      excludeSections = [],
      createBackup = false
    } = options;

    try {
      // Create backup of target project if requested
      if (createBackup) {
        await configService.saveProjectConfigToFile(targetProject);
      }

      // Get source configuration
      const sourceConfigResponse = await configService.getProjectConfig(sourceProject);
      if (!sourceConfigResponse.success) {
        throw new Error(`Failed to get source project config: ${sourceProject}`);
      }

      const sourceSettings = sourceConfigResponse.data.data as ProjectSettings;

      // Prepare update request, excluding specified sections
      const updates: ConfigUpdateRequest = {};

      if (!excludeSections.includes('workdir')) {
        updates.workdir = sourceSettings.workdir;
      }

      if (!excludeSections.includes('openai')) {
        updates.openai_api_key = sourceSettings.openai.api_key;
        updates.openai_api_base = sourceSettings.openai.api_base;
      }

      if (!excludeSections.includes('llm')) {
        updates.llm_model = sourceSettings.llm.model;
        updates.llm_temperature = sourceSettings.llm.temperature;
        updates.llm_max_tokens = sourceSettings.llm.max_tokens;
        updates.llm_provider = sourceSettings.llm.provider;
      }

      if (!excludeSections.includes('embedding')) {
        updates.embedding_model = sourceSettings.embedding.model;
        updates.embedding_provider = sourceSettings.embedding.provider;
        updates.embedding_dimension = sourceSettings.embedding.dimension;
        updates.embedding_max_token_size = sourceSettings.embedding.max_token_size;
        updates.embedding_batch_size = sourceSettings.embedding.batch_size;
      }

      if (!excludeSections.includes('graph')) {
        updates.entity_types = sourceSettings.graph.entity_types;
        updates.relation_types = sourceSettings.graph.relation_types;
      }

      if (!excludeSections.includes('text')) {
        updates.max_chunk_size = sourceSettings.text.max_chunk_size;
        updates.chunk_overlap = sourceSettings.text.chunk_overlap;
      }

      if (!excludeSections.includes('rag')) {
        updates.system_prompt = sourceSettings.rag.system_prompt;
      }

      // Clone builder configuration
      if (!excludeSections.includes('builder')) {
        const builder = sourceSettings.builder;
        updates.builder_enable_cache = builder.enable_cache;
        updates.builder_cache_dir = builder.cache_dir;
        updates.builder_cache_ttl = builder.cache_ttl;
        updates.builder_auto_cleanup = builder.auto_cleanup;
        updates.builder_chunk_size = builder.chunk_size;
        updates.builder_chunk_overlap = builder.chunk_overlap;
        updates.builder_entity_confidence_threshold = builder.entity_confidence_threshold;
        updates.builder_relation_confidence_threshold = builder.relation_confidence_threshold;
        updates.builder_cluster_algorithm = builder.cluster_algorithm;
        updates.builder_min_cluster_size = builder.min_cluster_size;
        updates.builder_enable_user_interaction = builder.enable_user_interaction;
        updates.builder_auto_save_edits = builder.auto_save_edits;
      }

      // Override project name if requested
      if (overrideProjectName) {
        updates.current_project = targetProject;
      }

      // Apply configuration to target project
      const updateResponse = await configService.updateProjectConfig(targetProject, updates);

      return {
        success: updateResponse.success,
        clonedSections: Object.keys(updates),
        targetConfig: updateResponse.data,
        backupCreated: createBackup
      };

    } catch (error) {
      console.error('Failed to clone project config:', error);
      throw error;
    }
  }

  /**
   * Compare configurations between two projects
   */
  async compareProjectConfigs(projectA: string, projectB: string) {
    try {
      const [configA, configB] = await Promise.all([
        configService.getProjectConfig(projectA),
        configService.getProjectConfig(projectB)
      ]);

      if (!configA.success || !configB.success) {
        throw new Error('Failed to retrieve project configurations');
      }

      const settingsA = configA.data.data as ProjectSettings;
      const settingsB = configB.data.data as ProjectSettings;

      const differences: Record<string, { projectA: any; projectB: any }> = {};

      // Compare each configuration section
      const sections = ['workdir', 'current_project', 'max_current', 'openai', 'llm', 'embedding', 'graph', 'text', 'rag', 'builder'];

      sections.forEach(section => {
        const valueA = (settingsA as any)[section];
        const valueB = (settingsB as any)[section];

        if (JSON.stringify(valueA) !== JSON.stringify(valueB)) {
          differences[section] = {
            projectA: valueA,
            projectB: valueB
          };
        }
      });

      return {
        identical: Object.keys(differences).length === 0,
        differences,
        totalSections: sections.length,
        differentSections: Object.keys(differences).length
      };

    } catch (error) {
      console.error('Failed to compare project configs:', error);
      throw error;
    }
  }

  /**
   * Get configuration health status for a project
   */
  async getConfigHealthStatus(projectName: string) {
    try {
      const [configResponse, backupResponse, projectDetails] = await Promise.all([
        configService.getProjectConfig(projectName),
        configService.getProjectBackupStatus(projectName),
        projectService.getProjectDetails(projectName)
      ]);

      const issues: string[] = [];
      let healthScore = 100;

      // Check config accessibility
      if (!configResponse.success) {
        issues.push('Configuration not accessible');
        healthScore -= 30;
      }

      // Check backup status
      const backupStatus = backupResponse.data?.data;
      if (!backupStatus?.backup_exists) {
        issues.push('No backup file found');
        healthScore -= 20;
      } else if (!backupStatus.has_complete_settings) {
        issues.push('Incomplete backup file');
        healthScore -= 15;
      }

      // Check project details
      if (!projectDetails.success) {
        issues.push('Project details not accessible');
        healthScore -= 15;
      }

      // Validate configuration structure if available
      if (configResponse.success) {
        const settings = configResponse.data.data as ProjectSettings;

        // Check for required API keys
        if (!settings.openai?.api_key) {
          issues.push('Missing OpenAI API key');
          healthScore -= 10;
        }

        // Check for reasonable configuration values
        if (settings.llm?.temperature > 1.5) {
          issues.push('LLM temperature may be too high');
          healthScore -= 5;
        }

        if (settings.text?.max_chunk_size > 2000) {
          issues.push('Chunk size may be too large');
          healthScore -= 5;
        }
      }

      return {
        healthy: issues.length === 0,
        healthScore: Math.max(0, healthScore),
        issues,
        configAccessible: configResponse.success,
        backupAvailable: backupStatus?.backup_exists || false,
        projectDetailsAvailable: projectDetails.success
      };

    } catch (error) {
      console.error('Failed to get config health status:', error);
      return {
        healthy: false,
        healthScore: 0,
        issues: ['Failed to perform health check'],
        configAccessible: false,
        backupAvailable: false,
        projectDetailsAvailable: false
      };
    }
  }
}

export const configManager = new ConfigManager();
export default configManager;
