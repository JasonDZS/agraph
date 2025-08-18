import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';
import type { EnhancedProject } from '../modules/Projects/types/project';

export interface ProjectState {
  // Project List
  projects: EnhancedProject[];
  projectsLoading: boolean;
  projectsError: string | null;

  // Current Project
  currentProject: EnhancedProject | null;
  currentProjectLoading: boolean;

  // Project Operations
  isCreating: boolean;
  isDeleting: boolean;
  isSwitching: boolean;

  // Filters and Search
  searchQuery: string;
  sortBy: 'name' | 'created_at' | 'updated_at' | 'document_count';
  sortOrder: 'asc' | 'desc';

  // UI State
  showCreateModal: boolean;
  showDeleteConfirmModal: boolean;
  projectToDelete: string | null;
}

export interface ProjectActions {
  // Project List Actions
  setProjects: (projects: EnhancedProject[]) => void;
  addProject: (project: EnhancedProject) => void;
  updateProject: (
    projectName: string,
    updates: Partial<EnhancedProject>
  ) => void;
  removeProject: (projectName: string) => void;
  setProjectsLoading: (loading: boolean) => void;
  setProjectsError: (error: string | null) => void;

  // Current Project Actions
  setCurrentProject: (project: EnhancedProject | null) => void;
  setCurrentProjectLoading: (loading: boolean) => void;

  // Project Operations
  setIsCreating: (creating: boolean) => void;
  setIsDeleting: (deleting: boolean) => void;
  setIsSwitching: (switching: boolean) => void;

  // Filter and Search Actions
  setSearchQuery: (query: string) => void;
  setSortBy: (sortBy: ProjectState['sortBy']) => void;
  setSortOrder: (order: ProjectState['sortOrder']) => void;

  // UI Actions
  setShowCreateModal: (show: boolean) => void;
  setShowDeleteConfirmModal: (show: boolean) => void;
  setProjectToDelete: (projectName: string | null) => void;

  // Computed Values
  getFilteredProjects: () => EnhancedProject[];
  getProjectByName: (name: string) => EnhancedProject | undefined;
  getProjectStats: () => {
    totalProjects: number;
    totalDocuments: number;
    totalEntities: number;
    totalRelations: number;
  };

  // Reset Actions
  reset: () => void;
  resetOperationStates: () => void;
}

export type ProjectStore = ProjectState & ProjectActions;

const initialState: ProjectState = {
  projects: [],
  projectsLoading: false,
  projectsError: null,
  currentProject: null,
  currentProjectLoading: false,
  isCreating: false,
  isDeleting: false,
  isSwitching: false,
  searchQuery: '',
  sortBy: 'updated_at',
  sortOrder: 'desc',
  showCreateModal: false,
  showDeleteConfirmModal: false,
  projectToDelete: null,
};

export const useProjectStore = create<ProjectStore>()(
  devtools(
    persist(
      (set, get) => ({
        ...initialState,

        // Project List Actions
        setProjects: projects => {
          set({ projects, projectsError: null });
        },

        addProject: project => {
          set(state => ({
            projects: [...state.projects, project],
          }));
        },

        updateProject: (projectName, updates) => {
          set(state => ({
            projects: state.projects.map(project =>
              project.name === projectName
                ? { ...project, ...updates }
                : project
            ),
            currentProject:
              state.currentProject?.name === projectName
                ? { ...state.currentProject, ...updates }
                : state.currentProject,
          }));
        },

        removeProject: projectName => {
          set(state => ({
            projects: state.projects.filter(
              project => project.name !== projectName
            ),
            currentProject:
              state.currentProject?.name === projectName
                ? null
                : state.currentProject,
          }));
        },

        setProjectsLoading: projectsLoading => {
          set({ projectsLoading });
        },

        setProjectsError: projectsError => {
          set({ projectsError, projectsLoading: false });
        },

        // Current Project Actions
        setCurrentProject: currentProject => {
          set({ currentProject });
        },

        setCurrentProjectLoading: currentProjectLoading => {
          set({ currentProjectLoading });
        },

        // Project Operations
        setIsCreating: isCreating => {
          set({ isCreating });
        },

        setIsDeleting: isDeleting => {
          set({ isDeleting });
        },

        setIsSwitching: isSwitching => {
          set({ isSwitching });
        },

        // Filter and Search Actions
        setSearchQuery: searchQuery => {
          set({ searchQuery });
        },

        setSortBy: sortBy => {
          set({ sortBy });
        },

        setSortOrder: sortOrder => {
          set({ sortOrder });
        },

        // UI Actions
        setShowCreateModal: showCreateModal => {
          set({ showCreateModal });
        },

        setShowDeleteConfirmModal: showDeleteConfirmModal => {
          set({ showDeleteConfirmModal });
        },

        setProjectToDelete: projectToDelete => {
          set({ projectToDelete });
        },

        // Computed Values
        getFilteredProjects: () => {
          const { projects, searchQuery, sortBy, sortOrder } = get();

          let filtered = projects;

          // Apply search filter
          if (searchQuery.trim()) {
            const query = searchQuery.toLowerCase();
            filtered = filtered.filter(
              project =>
                project.name.toLowerCase().includes(query) ||
                project.description?.toLowerCase().includes(query)
            );
          }

          // Apply sorting
          filtered.sort((a, b) => {
            let aValue: any, bValue: any;

            switch (sortBy) {
              case 'name':
                aValue = a.name.toLowerCase();
                bValue = b.name.toLowerCase();
                break;
              case 'created_at':
                aValue = new Date(a.created_at || 0).getTime();
                bValue = new Date(b.created_at || 0).getTime();
                break;
              case 'updated_at':
                aValue = new Date(a.updated_at || 0).getTime();
                bValue = new Date(b.updated_at || 0).getTime();
                break;
              case 'document_count':
                aValue = a.document_count || 0;
                bValue = b.document_count || 0;
                break;
              default:
                return 0;
            }

            if (aValue < bValue) return sortOrder === 'asc' ? -1 : 1;
            if (aValue > bValue) return sortOrder === 'asc' ? 1 : -1;
            return 0;
          });

          return filtered;
        },

        getProjectByName: name => {
          const { projects } = get();
          return projects.find(project => project.name === name);
        },

        getProjectStats: () => {
          const { projects } = get();
          return projects.reduce(
            (stats, project) => ({
              totalProjects: stats.totalProjects + 1,
              totalDocuments:
                stats.totalDocuments + (project.document_count || 0),
              totalEntities: stats.totalEntities + (project.entity_count || 0),
              totalRelations:
                stats.totalRelations + (project.relation_count || 0),
            }),
            {
              totalProjects: 0,
              totalDocuments: 0,
              totalEntities: 0,
              totalRelations: 0,
            }
          );
        },

        // Reset Actions
        reset: () => {
          set(initialState);
        },

        resetOperationStates: () => {
          set({
            isCreating: false,
            isDeleting: false,
            isSwitching: false,
            projectsLoading: false,
            currentProjectLoading: false,
            projectsError: null,
          });
        },
      }),
      {
        name: 'agraph-project-store',
        partialize: state => ({
          currentProject: state.currentProject,
          searchQuery: state.searchQuery,
          sortBy: state.sortBy,
          sortOrder: state.sortOrder,
        }),
      }
    ),
    {
      name: 'ProjectStore',
    }
  )
);

// Project state change listeners
let projectListeners: ((projects: EnhancedProject[]) => void)[] = [];
let currentProjectListeners: ((project: EnhancedProject | null) => void)[] = [];

export const subscribeToProjects = (
  callback: (projects: EnhancedProject[]) => void
) => {
  projectListeners.push(callback);
  return () => {
    projectListeners = projectListeners.filter(
      listener => listener !== callback
    );
  };
};

export const subscribeToCurrentProject = (
  callback: (project: EnhancedProject | null) => void
) => {
  currentProjectListeners.push(callback);
  return () => {
    currentProjectListeners = currentProjectListeners.filter(
      listener => listener !== callback
    );
  };
};

// Subscribe to store changes to notify listeners
useProjectStore.subscribe((state, prevState) => {
  if (state.projects !== prevState.projects) {
    projectListeners.forEach(listener => listener(state.projects));
  }

  if (state.currentProject !== prevState.currentProject) {
    currentProjectListeners.forEach(listener => listener(state.currentProject));
  }
});
