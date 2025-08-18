import { useEffect, useState, useCallback } from 'react';
import { useAppStore, type Notification } from '../store/appStore';

export interface UseNotificationsOptions {
  autoClose?: boolean;
  maxNotifications?: number;
  position?:
    | 'top-right'
    | 'top-left'
    | 'bottom-right'
    | 'bottom-left'
    | 'top-center'
    | 'bottom-center';
}

export interface NotificationActions {
  add: (notification: Omit<Notification, 'id' | 'timestamp'>) => void;
  remove: (id: string) => void;
  clear: () => void;
  success: (title: string, message: string, duration?: number) => void;
  error: (title: string, message: string, duration?: number) => void;
  warning: (title: string, message: string, duration?: number) => void;
  info: (title: string, message: string, duration?: number) => void;
}

export interface UseNotificationsReturn {
  notifications: Notification[];
  actions: NotificationActions;
  hasNotifications: boolean;
  unreadCount: number;
}

export const useNotifications = (
  options: UseNotificationsOptions = {}
): UseNotificationsReturn => {
  const { maxNotifications = 5 } = options;

  const {
    notifications: allNotifications,
    addNotification,
    removeNotification,
    clearNotifications,
  } = useAppStore(state => ({
    notifications: state.notifications,
    addNotification: state.addNotification,
    removeNotification: state.removeNotification,
    clearNotifications: state.clearNotifications,
  }));

  // Limit displayed notifications
  const notifications = allNotifications.slice(-maxNotifications);

  const actions: NotificationActions = {
    add: useCallback(
      (notification: Omit<Notification, 'id' | 'timestamp'>) => {
        addNotification(notification);
      },
      [addNotification]
    ),

    remove: useCallback(
      (id: string) => {
        removeNotification(id);
      },
      [removeNotification]
    ),

    clear: useCallback(() => {
      clearNotifications();
    }, [clearNotifications]),

    success: useCallback(
      (title: string, message: string, duration = 5000) => {
        addNotification({
          type: 'success',
          title,
          message,
          autoClose: true,
          duration,
        });
      },
      [addNotification]
    ),

    error: useCallback(
      (title: string, message: string, duration = 8000) => {
        addNotification({
          type: 'error',
          title,
          message,
          autoClose: true,
          duration,
        });
      },
      [addNotification]
    ),

    warning: useCallback(
      (title: string, message: string, duration = 6000) => {
        addNotification({
          type: 'warning',
          title,
          message,
          autoClose: true,
          duration,
        });
      },
      [addNotification]
    ),

    info: useCallback(
      (title: string, message: string, duration = 5000) => {
        addNotification({
          type: 'info',
          title,
          message,
          autoClose: true,
          duration,
        });
      },
      [addNotification]
    ),
  };

  return {
    notifications,
    actions,
    hasNotifications: notifications.length > 0,
    unreadCount: notifications.length,
  };
};

// Hook for listening to specific store changes
export const useStoreListener = <T>(
  storeSelector: () => T,
  callback: (value: T, prevValue: T) => void,
  deps: any[] = []
) => {
  const [prevValue, setPrevValue] = useState<T>(storeSelector);

  useEffect(() => {
    const currentValue = storeSelector();
    if (currentValue !== prevValue) {
      callback(currentValue, prevValue);
      setPrevValue(currentValue);
    }
  }, deps);
};

// Hook for project changes
export const useProjectListener = (
  callback: (projectName: string | null, prevProjectName: string | null) => void
) => {
  const currentProject = useAppStore(state => state.currentProject);
  const [prevProject, setPrevProject] = useState<string | null>(currentProject);

  useEffect(() => {
    if (currentProject !== prevProject) {
      callback(currentProject, prevProject);
      setPrevProject(currentProject);
    }
  }, [currentProject, prevProject, callback]);
};

// Hook for monitoring store hydration
export const useStoreHydration = () => {
  const [isHydrated, setIsHydrated] = useState(false);
  const [hydrationStatus, setHydrationStatus] = useState({
    appStore: false,
    projectStore: false,
    documentStore: false,
    knowledgeGraphStore: false,
  });

  useEffect(() => {
    const { storeManager } = require('../store');

    const unsubscribe = storeManager.onHydrationComplete(
      (status: typeof hydrationStatus) => {
        setHydrationStatus(status);
        setIsHydrated(Object.values(status).every(Boolean));
      }
    );

    return unsubscribe;
  }, []);

  return {
    isHydrated,
    hydrationStatus,
  };
};

// Hook for theme changes
export const useThemeListener = (
  callback: (theme: 'light' | 'dark', prevTheme: 'light' | 'dark') => void
) => {
  const theme = useAppStore(state => state.theme);
  const [prevTheme, setPrevTheme] = useState<'light' | 'dark'>(theme);

  useEffect(() => {
    if (theme !== prevTheme) {
      callback(theme, prevTheme);
      setPrevTheme(theme);

      // Apply theme to document
      document.documentElement.setAttribute('data-theme', theme);
      document.documentElement.classList.toggle('dark', theme === 'dark');
    }
  }, [theme, prevTheme, callback]);
};

// Hook for error handling
export const useErrorHandler = () => {
  const { setGlobalError, clearGlobalError } = useAppStore(state => ({
    setGlobalError: state.setGlobalError,
    clearGlobalError: state.clearGlobalError,
  }));

  const { error: showErrorNotification } = useNotifications().actions;

  const handleError = useCallback(
    (
      error: Error | string,
      options: {
        showNotification?: boolean;
        setGlobal?: boolean;
        title?: string;
      } = {}
    ) => {
      const {
        showNotification = true,
        setGlobal = false,
        title = 'Error',
      } = options;

      const message = typeof error === 'string' ? error : error.message;

      if (showNotification) {
        showErrorNotification(title, message);
      }

      if (setGlobal) {
        setGlobalError(message);
      }

      console.error('Error handled:', error);
    },
    [setGlobalError, showErrorNotification]
  );

  return {
    handleError,
    clearGlobalError,
  };
};

// Hook for loading states
export const useLoadingState = () => {
  const { loading, setLoading } = useAppStore(state => ({
    loading: state.loading,
    setLoading: state.setLoading,
  }));

  const withLoading = useCallback(
    async <T>(
      operation: () => Promise<T>,
      options: {
        showError?: boolean;
        errorTitle?: string;
      } = {}
    ): Promise<T | null> => {
      const { showError = true, errorTitle = 'Operation Failed' } = options;
      const { handleError } = useErrorHandler();

      setLoading(true);
      try {
        const result = await operation();
        return result;
      } catch (error) {
        if (showError) {
          handleError(error as Error, { title: errorTitle });
        }
        return null;
      } finally {
        setLoading(false);
      }
    },
    [setLoading]
  );

  return {
    loading,
    setLoading,
    withLoading,
  };
};

export default useNotifications;
