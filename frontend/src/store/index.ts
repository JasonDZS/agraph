// Import all stores
import { useAppStore } from './appStore';
import { useProjectStore } from './projectStore';
import { useDocumentStore } from './documentStore';
import { useKnowledgeGraphStore } from './knowledgeGraphStore';

// Store exports
export { useAppStore } from './appStore';
export type { AppStore, AppState, AppActions, Notification } from './appStore';

export { useProjectStore } from './projectStore';
export type {
  ProjectStore,
  ProjectState,
  ProjectActions,
} from './projectStore';

export { useDocumentStore } from './documentStore';
export type {
  DocumentStore,
  DocumentState,
  DocumentActions,
  UploadProgress,
} from './documentStore';

export { useKnowledgeGraphStore } from './knowledgeGraphStore';
export type {
  KnowledgeGraphStore,
  KnowledgeGraphState,
  KnowledgeGraphActions,
  GraphLayout,
  GraphVisualState,
} from './knowledgeGraphStore';

// Store persistence utilities
export interface StoreHydrationStatus {
  appStore: boolean;
  projectStore: boolean;
  documentStore: boolean;
  knowledgeGraphStore: boolean;
}

class StoreManager {
  private hydrationStatus: StoreHydrationStatus = {
    appStore: false,
    projectStore: false,
    documentStore: false,
    knowledgeGraphStore: false,
  };

  private hydrationPromises: Promise<void>[] = [];
  private hydrationListeners: ((status: StoreHydrationStatus) => void)[] = [];

  constructor() {
    this.initializeStores();
  }

  private async initializeStores() {
    // Initialize stores and track hydration status
    try {
      await Promise.all([
        this.initializeAppStore(),
        this.initializeProjectStore(),
        this.initializeDocumentStore(),
        this.initializeKnowledgeGraphStore(),
      ]);
    } catch (error) {
      console.error('Error initializing stores:', error);
    }
  }

  private async initializeAppStore() {
    const { useAppStore } = await import('./appStore');
    return new Promise<void>(resolve => {
      const unsubscribe = useAppStore.persist.onFinishHydration(() => {
        this.hydrationStatus.appStore = true;
        this.notifyHydrationListeners();
        unsubscribe();
        resolve();
      });
    });
  }

  private async initializeProjectStore() {
    const { useProjectStore } = await import('./projectStore');
    return new Promise<void>(resolve => {
      const unsubscribe = useProjectStore.persist.onFinishHydration(() => {
        this.hydrationStatus.projectStore = true;
        this.notifyHydrationListeners();
        unsubscribe();
        resolve();
      });
    });
  }

  private async initializeDocumentStore() {
    const { useDocumentStore } = await import('./documentStore');
    return new Promise<void>(resolve => {
      const unsubscribe = useDocumentStore.persist.onFinishHydration(() => {
        this.hydrationStatus.documentStore = true;
        this.notifyHydrationListeners();
        unsubscribe();
        resolve();
      });
    });
  }

  private async initializeKnowledgeGraphStore() {
    const { useKnowledgeGraphStore } = await import('./knowledgeGraphStore');
    return new Promise<void>(resolve => {
      const unsubscribe = useKnowledgeGraphStore.persist.onFinishHydration(
        () => {
          this.hydrationStatus.knowledgeGraphStore = true;
          this.notifyHydrationListeners();
          unsubscribe();
          resolve();
        }
      );
    });
  }

  private notifyHydrationListeners() {
    this.hydrationListeners.forEach(listener =>
      listener({ ...this.hydrationStatus })
    );
  }

  public onHydrationComplete(
    callback: (status: StoreHydrationStatus) => void
  ): () => void {
    this.hydrationListeners.push(callback);

    // Immediately call with current status
    callback({ ...this.hydrationStatus });

    // Return unsubscribe function
    return () => {
      this.hydrationListeners = this.hydrationListeners.filter(
        listener => listener !== callback
      );
    };
  }

  public async waitForHydration(): Promise<StoreHydrationStatus> {
    await Promise.all(this.hydrationPromises);
    return { ...this.hydrationStatus };
  }

  public isHydrated(): boolean {
    return Object.values(this.hydrationStatus).every(Boolean);
  }

  public getHydrationStatus(): StoreHydrationStatus {
    return { ...this.hydrationStatus };
  }

  public clearAllStores(): void {
    try {
      useAppStore.getState().reset();
      useProjectStore.getState().reset();
      useDocumentStore.getState().reset();
      useKnowledgeGraphStore.getState().reset();

      // Clear localStorage
      localStorage.removeItem('agraph-app-store');
      localStorage.removeItem('agraph-project-store');
      localStorage.removeItem('agraph-document-store');
      localStorage.removeItem('agraph-knowledge-graph-store');

      console.log('All stores cleared successfully');
    } catch (error) {
      console.error('Error clearing stores:', error);
    }
  }

  public exportStoreData(): Record<string, any> {
    try {
      return {
        appStore: useAppStore.getState(),
        projectStore: useProjectStore.getState(),
        documentStore: useDocumentStore.getState(),
        knowledgeGraphStore: useKnowledgeGraphStore.getState(),
        exportedAt: new Date().toISOString(),
      };
    } catch (error) {
      console.error('Error exporting store data:', error);
      return {};
    }
  }

  public importStoreData(data: Record<string, any>): void {
    try {
      if (data.appStore) {
        // Note: Selective import to avoid overwriting runtime state
        const { theme, sidebarCollapsed, currentProject } = data.appStore;
        useAppStore.setState({ theme, sidebarCollapsed, currentProject });
      }

      if (data.projectStore) {
        const { searchQuery, sortBy, sortOrder } = data.projectStore;
        useProjectStore.setState({ searchQuery, sortBy, sortOrder });
      }

      if (data.documentStore) {
        const { pageSize, searchQuery, tagFilter, sortBy, sortOrder } =
          data.documentStore;
        useDocumentStore.setState({
          pageSize,
          searchQuery,
          tagFilter,
          sortBy,
          sortOrder,
        });
      }

      if (data.knowledgeGraphStore) {
        const {
          currentLayout,
          showNodeLabels,
          showEdgeLabels,
          nodeSize,
          edgeWidth,
          entityTypeFilter,
          relationTypeFilter,
          confidenceThreshold,
        } = data.knowledgeGraphStore;
        useKnowledgeGraphStore.setState({
          currentLayout,
          showNodeLabels,
          showEdgeLabels,
          nodeSize,
          edgeWidth,
          entityTypeFilter,
          relationTypeFilter,
          confidenceThreshold,
        });
      }

      console.log('Store data imported successfully');
    } catch (error) {
      console.error('Error importing store data:', error);
    }
  }
}

// Create singleton instance
export const storeManager = new StoreManager();

// Development tools integration
declare global {
  interface Window {
    __AGRAPH_STORES__: {
      app: ReturnType<typeof useAppStore.getState>;
      project: ReturnType<typeof useProjectStore.getState>;
      document: ReturnType<typeof useDocumentStore.getState>;
      knowledgeGraph: ReturnType<typeof useKnowledgeGraphStore.getState>;
      manager: StoreManager;
    };
  }
}

// Expose stores to development tools in development mode
if (process.env.NODE_ENV === 'development') {
  window.__AGRAPH_STORES__ = {
    get app() {
      return useAppStore.getState();
    },
    get project() {
      return useProjectStore.getState();
    },
    get document() {
      return useDocumentStore.getState();
    },
    get knowledgeGraph() {
      return useKnowledgeGraphStore.getState();
    },
    manager: storeManager,
  };

  console.log('🏪 AGraph stores are available in development tools:');
  console.log('- window.__AGRAPH_STORES__.app');
  console.log('- window.__AGRAPH_STORES__.project');
  console.log('- window.__AGRAPH_STORES__.document');
  console.log('- window.__AGRAPH_STORES__.knowledgeGraph');
  console.log('- window.__AGRAPH_STORES__.manager');
}

// Notification system integration
export class NotificationSystem {
  private static instance: NotificationSystem;

  public static getInstance(): NotificationSystem {
    if (!NotificationSystem.instance) {
      NotificationSystem.instance = new NotificationSystem();
    }
    return NotificationSystem.instance;
  }

  public success(title: string, message: string, duration = 5000): void {
    useAppStore.getState().addNotification({
      type: 'success',
      title,
      message,
      autoClose: true,
      duration,
    });
  }

  public error(title: string, message: string, duration = 8000): void {
    useAppStore.getState().addNotification({
      type: 'error',
      title,
      message,
      autoClose: true,
      duration,
    });
  }

  public warning(title: string, message: string, duration = 6000): void {
    useAppStore.getState().addNotification({
      type: 'warning',
      title,
      message,
      autoClose: true,
      duration,
    });
  }

  public info(title: string, message: string, duration = 5000): void {
    useAppStore.getState().addNotification({
      type: 'info',
      title,
      message,
      autoClose: true,
      duration,
    });
  }

  public persistent(
    type: 'success' | 'error' | 'warning' | 'info',
    title: string,
    message: string
  ): void {
    useAppStore.getState().addNotification({
      type,
      title,
      message,
      autoClose: false,
    });
  }
}

// Export notification instance
export const notifications = NotificationSystem.getInstance();

// Store synchronization utilities
export const synchronizeStores = {
  // Sync current project across stores
  syncCurrentProject: (projectName: string | null) => {
    useAppStore.getState().setCurrentProject(projectName);
    if (projectName) {
      const project = useProjectStore.getState().getProjectByName(projectName);
      useProjectStore.getState().setCurrentProject(project || null);
    } else {
      useProjectStore.getState().setCurrentProject(null);
    }
  },

  // Clear all temporary state when switching projects
  clearTemporaryState: () => {
    useDocumentStore.getState().clearSelection();
    useDocumentStore.getState().setCurrentPage(1);
    useKnowledgeGraphStore.getState().clearSelection();
    useKnowledgeGraphStore.getState().resetVisualState();
  },

  // Reset all operation states
  resetOperationStates: () => {
    useAppStore.getState().setLoading(false);
    useProjectStore.getState().resetOperationStates();
    useDocumentStore.getState().resetOperationStates();
    useKnowledgeGraphStore.getState().resetOperationStates();
  },
};

export default storeManager;
