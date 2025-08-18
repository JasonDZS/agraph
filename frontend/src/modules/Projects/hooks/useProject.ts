import { useState, useEffect, useCallback } from 'react';
import { projectService } from '@/services/projectService';
import { useProjectStore, useAppStore, notifications } from '@/store';
import type {
  EnhancedProject,
  ProjectOperationResult,
  ProjectFormData,
  ProjectDeleteConfirmation,
  ProjectStatistics,
} from '../types/project';

export const useProject = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const {
    projects,
    currentProject,
    getProjectByName,
    setProjects,
    setCurrentProject,
    addProject,
    updateProject,
    removeProject,
    setProjectsLoading,
    setProjectsError,
  } = useProjectStore();

  const {
    setCurrentProject: setGlobalCurrentProject,
    setLoading: setGlobalLoading,
  } = useAppStore();

  // Load projects from API
  const loadProjects = useCallback(
    async (skipCache: boolean = false): Promise<EnhancedProject[]> => {
      try {
        setProjectsLoading(true);
        setError(null);

        const [projectsResponse, currentResponse] = await Promise.all([
          projectService.listProjects(true, !skipCache), // 获取统计信息
          projectService.getCurrentProject(!skipCache),
        ]);

        if (!projectsResponse.success && projectsResponse.error) {
          throw new Error(projectsResponse.error.message);
        }

        if (!currentResponse.success && currentResponse.error) {
          throw new Error(currentResponse.error.message);
        }

        const projectNames = projectsResponse.data?.projects || [];
        const currentProjectName = (currentResponse.data as any)
          ?.current_project;
        const projectStatistics =
          (projectsResponse.data as any)?.project_statistics || {};

        // API response and statistics logging for debugging
        // console.log('Projects API Response:', projectsResponse);
        // console.log('Project Statistics:', projectStatistics);

        // Create enhanced projects with statistics from API
        const detailedProjects: EnhancedProject[] = projectNames.map(name => {
          const stats = projectStatistics[name];
          // console.log(`Stats for project ${name}:`, stats);

          const project: EnhancedProject = {
            name,
            description: '', // 可以从项目配置文件或其他API获取
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString(),
            document_count: stats?.document_count || 0,
            entity_count: stats?.entity_count || 0,
            relation_count: stats?.relation_count || 0,
            is_current: name === currentProjectName,
            status: name === currentProjectName ? 'active' : 'inactive',
            statistics: {
              document_count: stats?.document_count || 0,
              entity_count: stats?.entity_count || 0,
              relation_count: stats?.relation_count || 0,
              has_vector_db: stats?.has_vector_db || false,
              size_mb: stats?.size_mb || 0,
              error: stats?.error,
            },
          };

          // 如果有错误，标记项目状态为错误
          if (stats?.error) {
            project.status = 'error';
          }

          return project;
        });

        setProjects(detailedProjects);

        // Set current project in project store
        const currentProjectData = detailedProjects.find(p => p.is_current);
        setCurrentProject(currentProjectData || null);

        // Sync with global state
        setGlobalCurrentProject(currentProjectName || null);

        return detailedProjects;
      } catch (err) {
        const message =
          err instanceof Error ? err.message : 'Failed to load projects';
        setProjectsError(message);
        setError(message);
        notifications.error('Load Projects Failed', message);
        return [];
      } finally {
        setProjectsLoading(false);
      }
    },
    []
  );

  // Create a new project
  const createProject = useCallback(
    async (projectData: ProjectFormData): Promise<ProjectOperationResult> => {
      try {
        setLoading(true);
        setError(null);

        // Validate project name
        if (!projectService.validateProjectName(projectData.name)) {
          return {
            success: false,
            message:
              'Invalid project name. Please use only alphanumeric characters and avoid reserved names.',
            error: 'Validation failed',
          };
        }

        // Check if project already exists
        const exists = await projectService.checkProjectExists(
          projectData.name
        );
        if (exists) {
          return {
            success: false,
            message: `Project "${projectData.name}" already exists`,
            error: 'Project already exists',
          };
        }

        const response = await projectService.createProject({
          name: projectData.name,
          description: projectData.description,
        });

        if (!response.success && response.error) {
          return {
            success: false,
            message: response.error.message,
            error: response.error.message,
          };
        }

        const newProject: EnhancedProject = {
          name: projectData.name,
          description: projectData.description || '',
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
          document_count: 0,
          entity_count: 0,
          relation_count: 0,
          is_current: false,
          status: 'inactive',
          statistics: {
            document_count: 0,
            entity_count: 0,
            relation_count: 0,
            has_vector_db: false,
            cache_size: 0,
            size_mb: 0,
          },
        };

        addProject(newProject);
        notifications.success(
          'Project Created',
          `Project "${projectData.name}" created successfully`
        );

        // Refresh projects list without cache to get updated data
        await loadProjects(true);

        return {
          success: true,
          message: `Project "${projectData.name}" created successfully`,
          project: newProject,
        };
      } catch (err) {
        const message =
          err instanceof Error ? err.message : 'Failed to create project';
        setError(message);
        return {
          success: false,
          message,
          error: message,
        };
      } finally {
        setLoading(false);
      }
    },
    []
  );

  // Switch to a project
  const switchProject = useCallback(
    async (projectName: string | null): Promise<ProjectOperationResult> => {
      try {
        setGlobalLoading(true);
        setError(null);

        const response = await projectService.switchProject({
          project_name: projectName || undefined,
        });

        if (!response.success && response.error) {
          return {
            success: false,
            message: response.error.message,
            error: response.error.message,
          };
        }

        // Update project states
        projects.forEach(project => {
          updateProject(project.name, {
            is_current: project.name === projectName,
            status: project.name === projectName ? 'active' : 'inactive',
          });
        });

        const newCurrentProject = getProjectByName(projectName || '');
        setCurrentProject(newCurrentProject || null);
        setGlobalCurrentProject(projectName);

        const message = projectName
          ? `Switched to project "${projectName}"`
          : 'Switched to default workspace';

        notifications.success('Project Switched', message);

        // Refresh projects list without cache to get updated data
        await loadProjects(true);

        return {
          success: true,
          message,
          project: newCurrentProject,
        };
      } catch (err) {
        const message =
          err instanceof Error ? err.message : 'Failed to switch project';
        setError(message);
        notifications.error('Switch Failed', message);
        return {
          success: false,
          message,
          error: message,
        };
      } finally {
        setGlobalLoading(false);
      }
    },
    []
  );

  // Delete a project
  const deleteProject = useCallback(
    async (
      confirmation: ProjectDeleteConfirmation
    ): Promise<ProjectOperationResult> => {
      try {
        setLoading(true);
        setError(null);

        if (!confirmation.confirm || !confirmation.understoodConsequences) {
          return {
            success: false,
            message: 'Project deletion requires explicit confirmation',
            error: 'Confirmation required',
          };
        }

        const response = await projectService.deleteProject({
          project_name: confirmation.projectName,
          confirm: true,
        });

        if (!response.success && response.error) {
          return {
            success: false,
            message: response.error.message,
            error: response.error.message,
          };
        }

        removeProject(confirmation.projectName);

        // If we deleted the current project, clear current project
        if (currentProject?.name === confirmation.projectName) {
          setCurrentProject(null);
          setGlobalCurrentProject(null);
        }

        notifications.success(
          'Project Deleted',
          `Project "${confirmation.projectName}" deleted successfully`
        );

        // Refresh projects list without cache to get updated data
        await loadProjects(true);

        return {
          success: true,
          message: `Project "${confirmation.projectName}" deleted successfully`,
        };
      } catch (err) {
        const message =
          err instanceof Error ? err.message : 'Failed to delete project';
        setError(message);
        notifications.error('Delete Failed', message);
        return {
          success: false,
          message,
          error: message,
        };
      } finally {
        setLoading(false);
      }
    },
    []
  );

  // Refresh project data
  const refreshProject = useCallback(
    async (projectName: string): Promise<void> => {
      try {
        const response = await projectService.getProjectDetails(projectName);
        if (response.data?.data) {
          const projectData = response.data.data;
          updateProject(projectName, {
            ...(projectData as any),
            statistics: (projectData as any)?.statistics as ProjectStatistics,
            updated_at: new Date().toISOString(),
          });
        }
      } catch (err) {
        console.warn(`Failed to refresh project ${projectName}:`, err);
      }
    },
    []
  );

  // Initialize projects on mount
  useEffect(() => {
    loadProjects();
  }, []);

  return {
    // State
    projects,
    currentProject,
    loading,
    error,

    // Operations
    loadProjects,
    createProject,
    switchProject,
    deleteProject,
    refreshProject,

    // Utilities
    validateProjectName: projectService.validateProjectName,
    checkProjectExists: projectService.checkProjectExists,
  };
};
