import { useProjectStore } from '@/store/projectStore';
import { projectService } from '@/services/projectService';

/**
 * Utility to sync frontend project state with backend
 */
export class ProjectSync {
  private static _instance: ProjectSync;

  public static getInstance(): ProjectSync {
    if (!ProjectSync._instance) {
      ProjectSync._instance = new ProjectSync();
    }
    return ProjectSync._instance;
  }

  /**
   * Initialize project state by fetching current project from backend
   */
  async initializeProjectState(): Promise<boolean> {
    try {
      const response = await projectService.getCurrentProject();

      if (response.success && response.data) {
        const currentProjectName = (response.data as any)?.current_project;

        if (currentProjectName) {
          // Get or create project data
          let currentProject = useProjectStore
            .getState()
            .getProjectByName(currentProjectName);

          if (!currentProject) {
            // Create minimal project data if not exists
            currentProject = {
              name: currentProjectName,
              description: currentProjectName,
              created_at: new Date().toISOString(),
              updated_at: new Date().toISOString(),
              document_count: 0,
              entity_count: 0,
              relation_count: 0,
              is_current: true,
              status: 'active',
              statistics: {
                document_count: 0,
                entity_count: 0,
                relation_count: 0,
                has_vector_db: false,
                size_mb: 0,
              },
            };

            useProjectStore.getState().addProject(currentProject);
          }

          // Set as current project
          useProjectStore.getState().setCurrentProject(currentProject);

          console.log(
            `[ProjectSync] Initialized current project: ${currentProjectName}`
          );
          return true;
        }
      }

      console.warn('[ProjectSync] No current project found on backend');
      return false;
    } catch (error) {
      console.error('[ProjectSync] Failed to initialize project state:', error);
      return false;
    }
  }

  /**
   * Ensure project state is synchronized before making API calls
   */
  async ensureProjectSync(): Promise<string | null> {
    const currentProject = useProjectStore.getState().currentProject;

    if (!currentProject) {
      const initialized = await this.initializeProjectState();
      if (initialized) {
        return useProjectStore.getState().currentProject?.name || null;
      }
    }

    return currentProject?.name || null;
  }
}

export const projectSync = ProjectSync.getInstance();
