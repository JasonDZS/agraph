import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';
import type { AppConfig, SystemStats } from '../types/api';

export interface AppState {
  // UI State
  theme: 'light' | 'dark';
  sidebarCollapsed: boolean;
  loading: boolean;

  // Current Active Items
  currentProject: string | null;

  // System Configuration
  config: AppConfig | null;
  systemStats: SystemStats | null;

  // Notification System
  notifications: Notification[];

  // Error Handling
  globalError: string | null;
}

export interface Notification {
  id: string;
  type: 'success' | 'error' | 'warning' | 'info';
  title: string;
  message: string;
  timestamp: string;
  autoClose?: boolean;
  duration?: number;
}

export interface AppActions {
  // UI Actions
  toggleTheme: () => void;
  setTheme: (theme: 'light' | 'dark') => void;
  toggleSidebar: () => void;
  setSidebarCollapsed: (collapsed: boolean) => void;
  setLoading: (loading: boolean) => void;

  // Project Actions
  setCurrentProject: (projectName: string | null) => void;

  // Configuration Actions
  setConfig: (config: AppConfig) => void;
  updateConfig: (updates: Partial<AppConfig>) => void;
  setSystemStats: (stats: SystemStats) => void;

  // Notification Actions
  addNotification: (
    notification: Omit<Notification, 'id' | 'timestamp'>
  ) => void;
  showNotification: (notification: {
    type: 'success' | 'error' | 'warning' | 'info';
    message: string;
    description?: string;
    duration?: number;
  }) => void;
  removeNotification: (id: string) => void;
  clearNotifications: () => void;

  // Error Actions
  setGlobalError: (error: string | null) => void;
  clearGlobalError: () => void;

  // Reset Actions
  reset: () => void;
}

export type AppStore = AppState & AppActions;

const initialState: AppState = {
  theme: 'light',
  sidebarCollapsed: false,
  loading: false,
  currentProject: null,
  config: null,
  systemStats: null,
  notifications: [],
  globalError: null,
};

export const useAppStore = create<AppStore>()(
  devtools(
    persist(
      (set, get) => ({
        ...initialState,

        // UI Actions
        toggleTheme: () => {
          set(state => ({
            theme: state.theme === 'light' ? 'dark' : 'light',
          }));
        },

        setTheme: theme => {
          set({ theme });
        },

        toggleSidebar: () => {
          set(state => ({
            sidebarCollapsed: !state.sidebarCollapsed,
          }));
        },

        setSidebarCollapsed: collapsed => {
          set({ sidebarCollapsed: collapsed });
        },

        setLoading: loading => {
          set({ loading });
        },

        // Project Actions
        setCurrentProject: projectName => {
          set({ currentProject: projectName });
        },

        // Configuration Actions
        setConfig: config => {
          set({ config });
        },

        updateConfig: updates => {
          set(state => ({
            config: state.config ? { ...state.config, ...updates } : null,
          }));
        },

        setSystemStats: systemStats => {
          set({ systemStats });
        },

        // Notification Actions
        addNotification: notification => {
          const newNotification: Notification = {
            ...notification,
            id: `notification-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
            timestamp: new Date().toISOString(),
          };

          set(state => ({
            notifications: [...state.notifications, newNotification],
          }));

          // Auto-remove notification if specified
          if (notification.autoClose !== false) {
            const duration = notification.duration || 5000;
            setTimeout(() => {
              get().removeNotification(newNotification.id);
            }, duration);
          }
        },

        showNotification: notification => {
          get().addNotification({
            type: notification.type,
            title: notification.message,
            message: notification.description || '',
            duration: notification.duration,
            autoClose: true,
          });
        },

        removeNotification: id => {
          set(state => ({
            notifications: state.notifications.filter(n => n.id !== id),
          }));
        },

        clearNotifications: () => {
          set({ notifications: [] });
        },

        // Error Actions
        setGlobalError: globalError => {
          set({ globalError });
        },

        clearGlobalError: () => {
          set({ globalError: null });
        },

        // Reset Actions
        reset: () => {
          set(initialState);
        },
      }),
      {
        name: 'agraph-app-store',
        partialize: state => ({
          theme: state.theme,
          sidebarCollapsed: state.sidebarCollapsed,
          currentProject: state.currentProject,
        }),
      }
    ),
    {
      name: 'AppStore',
    }
  )
);

// Notification state change listeners
let notificationListeners: ((notifications: Notification[]) => void)[] = [];

export const subscribeToNotifications = (
  callback: (notifications: Notification[]) => void
) => {
  notificationListeners.push(callback);
  return () => {
    notificationListeners = notificationListeners.filter(
      listener => listener !== callback
    );
  };
};

// Subscribe to store changes to notify listeners
useAppStore.subscribe(state => {
  notificationListeners.forEach(listener => {
    listener(state.notifications);
  });
});
