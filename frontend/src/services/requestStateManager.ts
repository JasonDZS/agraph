// Request state manager for handling API request states and caching

export interface RequestState {
  loading: boolean;
  error: string | null;
  data: any;
  lastUpdated: number;
}

export interface RequestOptions {
  cacheKey?: string;
  cacheTtl?: number;
  onStart?: () => void;
  onSuccess?: (data: any) => void;
  onError?: (error: any) => void;
  onComplete?: () => void;
}

class RequestStateManager {
  private states = new Map<string, RequestState>();
  private cache = new Map<
    string,
    { data: any; timestamp: number; ttl: number }
  >();
  private listeners = new Map<string, Set<(state: RequestState) => void>>();

  getState(key: string): RequestState {
    return (
      this.states.get(key) || {
        loading: false,
        error: null,
        data: null,
        lastUpdated: 0,
      }
    );
  }

  setState(key: string, update: Partial<RequestState>): void {
    const current = this.getState(key);
    const newState = {
      ...current,
      ...update,
      lastUpdated: Date.now(),
    };

    this.states.set(key, newState);
    this.notifyListeners(key, newState);
  }

  subscribe(key: string, listener: (state: RequestState) => void): () => void {
    if (!this.listeners.has(key)) {
      this.listeners.set(key, new Set());
    }

    this.listeners.get(key)!.add(listener);

    // Return unsubscribe function
    return () => {
      const keyListeners = this.listeners.get(key);
      if (keyListeners) {
        keyListeners.delete(listener);
        if (keyListeners.size === 0) {
          this.listeners.delete(key);
        }
      }
    };
  }

  private notifyListeners(key: string, state: RequestState): void {
    const keyListeners = this.listeners.get(key);
    if (keyListeners) {
      keyListeners.forEach(listener => listener(state));
    }
  }

  async execute<T>(
    key: string,
    request: () => Promise<T>,
    options: RequestOptions = {}
  ): Promise<T> {
    const {
      cacheKey,
      cacheTtl = 5 * 60 * 1000,
      onStart,
      onSuccess,
      onError,
      onComplete,
    } = options;

    // Check cache first
    if (cacheKey) {
      const cached = this.getFromCache(cacheKey);
      if (cached) {
        this.setState(key, {
          loading: false,
          error: null,
          data: cached,
        });
        return cached;
      }
    }

    try {
      this.setState(key, {
        loading: true,
        error: null,
      });

      onStart?.();

      const result = await request();

      this.setState(key, {
        loading: false,
        error: null,
        data: result,
      });

      // Cache the result
      if (cacheKey) {
        this.setCache(cacheKey, result, cacheTtl);
      }

      onSuccess?.(result);
      return result;
    } catch (error: any) {
      const errorMessage = error.message || 'An error occurred';

      this.setState(key, {
        loading: false,
        error: errorMessage,
        data: null,
      });

      onError?.(error);
      throw error;
    } finally {
      onComplete?.();
    }
  }

  private getFromCache(key: string): any | null {
    const cached = this.cache.get(key);
    if (!cached) return null;

    const now = Date.now();
    if (now - cached.timestamp > cached.ttl) {
      this.cache.delete(key);
      return null;
    }

    return cached.data;
  }

  private setCache(key: string, data: any, ttl: number): void {
    this.cache.set(key, {
      data,
      timestamp: Date.now(),
      ttl,
    });
  }

  clearCache(pattern?: string): void {
    if (pattern) {
      const regex = new RegExp(pattern);
      for (const key of this.cache.keys()) {
        if (regex.test(key)) {
          this.cache.delete(key);
        }
      }
    } else {
      this.cache.clear();
    }
  }

  clearState(pattern?: string): void {
    if (pattern) {
      const regex = new RegExp(pattern);
      for (const key of this.states.keys()) {
        if (regex.test(key)) {
          this.states.delete(key);
          this.listeners.delete(key);
        }
      }
    } else {
      this.states.clear();
      this.listeners.clear();
    }
  }

  reset(): void {
    this.clearCache();
    this.clearState();
  }

  getDebugInfo(): {
    stateCount: number;
    cacheCount: number;
    listenerCount: number;
    states: Record<string, RequestState>;
  } {
    const states: Record<string, RequestState> = {};
    this.states.forEach((state, key) => {
      states[key] = state;
    });

    return {
      stateCount: this.states.size,
      cacheCount: this.cache.size,
      listenerCount: Array.from(this.listeners.values()).reduce(
        (sum, set) => sum + set.size,
        0
      ),
      states,
    };
  }
}

// Create singleton instance
export const requestStateManager = new RequestStateManager();

// Utility hook-like function for managing request state
export function createRequestHandler<T>(
  key: string,
  requestFn: () => Promise<T>,
  options: RequestOptions = {}
) {
  return {
    execute: () => requestStateManager.execute(key, requestFn, options),
    getState: () => requestStateManager.getState(key),
    subscribe: (listener: (state: RequestState) => void) =>
      requestStateManager.subscribe(key, listener),
    clearCache: () => requestStateManager.clearCache(key),
    clearState: () => requestStateManager.clearState(key),
  };
}

// Pre-configured request handlers for common operations
export const requestHandlers = {
  // Project operations
  loadProjects: () =>
    createRequestHandler(
      'projects.list',
      () =>
        import('./projectService').then(m => m.projectService.listProjects()),
      { cacheKey: 'projects.list', cacheTtl: 2 * 60 * 1000 }
    ),

  // Document operations
  loadDocuments: (page = 1, pageSize = 10) =>
    createRequestHandler(
      `documents.list.${page}.${pageSize}`,
      () =>
        import('./documentService').then(m =>
          m.documentService.listDocuments({ page, page_size: pageSize })
        ),
      {
        cacheKey: `documents.list.${page}.${pageSize}`,
        cacheTtl: 1 * 60 * 1000,
      }
    ),

  // Knowledge graph operations
  loadGraph: () =>
    createRequestHandler(
      'graph.load',
      () =>
        import('./knowledgeGraphService').then(m =>
          m.knowledgeGraphService.getGraph()
        ),
      { cacheKey: 'graph.load', cacheTtl: 5 * 60 * 1000 }
    ),

  // Configuration operations
  loadConfig: (projectName: string) =>
    createRequestHandler(
      'config.load',
      () =>
        import('./configService').then(m =>
          m.configService.getProjectConfig(projectName)
        ),
      { cacheKey: `config.load.${projectName}`, cacheTtl: 10 * 60 * 1000 }
    ),
};
