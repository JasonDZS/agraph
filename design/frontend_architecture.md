# AGraph 前端架构设计方案

## 概述

本文档详细描述了基于 agraph API 的前端架构设计，涵盖项目管理、文档管理、知识图谱构建、可视化展示、智能搜索和对话等核心功能。

## 技术栈

### 核心技术栈

```typescript
// 前端框架
- React 18 + TypeScript
- Vite (构建工具)
- React Router (路由管理)

// 状态管理
- Zustand (轻量级状态管理)

// UI组件库
- Ant Design 5.x (企业级UI组件)

// HTTP客户端
- Axios (API请求)

// 可视化技术栈
- ECharts (知识图谱可视化、统计图表)
- D3.js (数据可视化)
- React-Flow (流程图)
```

## 项目结构

```
src/
├── components/           # 共享组件
│   ├── Layout/          # 布局组件
│   │   ├── MainLayout.tsx
│   │   ├── Sidebar.tsx
│   │   ├── Header.tsx
│   │   └── Footer.tsx
│   ├── Charts/          # 图表组件
│   │   ├── StatisticCard.tsx
│   │   ├── TrendChart.tsx
│   │   └── PieChart.tsx
│   ├── Upload/          # 上传组件
│   │   ├── FileUploader.tsx
│   │   ├── DropZone.tsx
│   │   └── ProgressIndicator.tsx
│   └── Common/          # 通用组件
│       ├── LoadingSpinner.tsx
│       ├── ErrorBoundary.tsx
│       ├── ConfirmModal.tsx
│       └── NotificationCenter.tsx
├── modules/             # 功能模块
│   ├── Projects/        # 项目管理
│   │   ├── components/
│   │   ├── hooks/
│   │   ├── services/
│   │   └── types/
│   ├── Documents/       # 文档管理
│   │   ├── components/
│   │   ├── hooks/
│   │   ├── services/
│   │   └── types/
│   ├── KnowledgeGraph/  # 知识图谱
│   │   ├── components/
│   │   ├── hooks/
│   │   ├── utils/
│   │   └── types/
│   ├── Search/          # 搜索功能
│   ├── Chat/           # 对话功能
│   ├── Visualization/   # 可视化配置
│   └── Settings/       # 系统设置
├── services/           # API服务层
│   ├── api.ts          # API客户端配置
│   ├── projectService.ts
│   ├── documentService.ts
│   ├── knowledgeGraphService.ts
│   ├── chatService.ts
│   └── searchService.ts
├── stores/            # 状态管理
│   ├── appStore.ts    # 全局状态
│   ├── projectStore.ts
│   ├── documentStore.ts
│   ├── knowledgeGraphStore.ts
│   └── chatStore.ts
├── types/             # 类型定义
│   ├── api.ts         # API响应类型
│   ├── entities.ts    # 业务实体类型
│   └── common.ts      # 通用类型
├── utils/             # 工具函数
│   ├── request.ts     # 请求工具
│   ├── format.ts      # 格式化工具
│   ├── validation.ts  # 验证工具
│   └── constants.ts   # 常量定义
├── hooks/             # 自定义hooks
│   ├── useApi.ts      # API调用hook
│   ├── useDebounce.ts # 防抖hook
│   └── useLocalStorage.ts
├── styles/            # 样式文件
│   ├── globals.css
│   ├── variables.css
│   └── themes/
└── assets/            # 静态资源
    ├── images/
    ├── icons/
    └── fonts/
```

## 路由设计

### 主要页面路由

```typescript
interface RouteConfig {
  path: string;
  component: React.ComponentType;
  title: string;
  icon: string;
  requiresAuth?: boolean;
}

const routes: RouteConfig[] = [
  // 工作台总览
  {
    path: '/',
    component: Dashboard,
    title: '工作台',
    icon: 'DashboardOutlined'
  },

  // 项目管理
  {
    path: '/projects',
    component: ProjectList,
    title: '项目管理',
    icon: 'FolderOutlined'
  },
  {
    path: '/projects/:id',
    component: ProjectDetail,
    title: '项目详情'
  },
  {
    path: '/projects/:id/settings',
    component: ProjectSettings,
    title: '项目设置'
  },

  // 文档管理
  {
    path: '/documents',
    component: DocumentList,
    title: '文档管理',
    icon: 'FileTextOutlined'
  },
  {
    path: '/documents/upload',
    component: DocumentUpload,
    title: '上传文档'
  },
  {
    path: '/documents/:id',
    component: DocumentViewer,
    title: '文档详情'
  },

  // 知识图谱
  {
    path: '/knowledge-graph',
    component: KnowledgeGraphOverview,
    title: '知识图谱',
    icon: 'NodeIndexOutlined'
  },
  {
    path: '/knowledge-graph/build',
    component: GraphBuilder,
    title: '构建图谱'
  },
  {
    path: '/knowledge-graph/visualize',
    component: GraphVisualizer,
    title: '图谱可视化'
  },
  {
    path: '/knowledge-graph/entities',
    component: EntityExplorer,
    title: '实体管理'
  },
  {
    path: '/knowledge-graph/relations',
    component: RelationExplorer,
    title: '关系管理'
  },

  // 搜索与查询
  {
    path: '/search',
    component: SearchInterface,
    title: '智能搜索',
    icon: 'SearchOutlined'
  },
  {
    path: '/search/advanced',
    component: AdvancedSearch,
    title: '高级搜索'
  },

  // 对话系统
  {
    path: '/chat',
    component: ChatInterface,
    title: '知识问答',
    icon: 'MessageOutlined'
  },
  {
    path: '/chat/:conversationId',
    component: ChatConversation,
    title: '对话详情'
  },

  // 可视化配置
  {
    path: '/visualization',
    component: VisualizationConfig,
    title: '可视化配置',
    icon: 'BarChartOutlined'
  },
  {
    path: '/visualization/dashboard',
    component: CustomDashboard,
    title: '自定义仪表板'
  },

  // 系统设置
  {
    path: '/settings',
    component: Settings,
    title: '系统设置',
    icon: 'SettingOutlined'
  },
  {
    path: '/settings/config',
    component: ConfigManagement,
    title: '配置管理'
  },
  {
    path: '/settings/cache',
    component: CacheManagement,
    title: '缓存管理'
  },
];
```

### 路由守卫

```typescript
// components/RouteGuard.tsx
interface RouteGuardProps {
  children: React.ReactNode;
  requiresProject?: boolean;
  requiresKnowledgeGraph?: boolean;
}

const RouteGuard: React.FC<RouteGuardProps> = ({
  children,
  requiresProject = false,
  requiresKnowledgeGraph = false,
}) => {
  const { currentProject } = useAppStore();
  const { graphBuilt } = useKnowledgeGraphStore();
  const navigate = useNavigate();

  useEffect(() => {
    if (requiresProject && !currentProject) {
      navigate('/projects', {
        state: { message: '请先选择或创建一个项目' }
      });
      return;
    }

    if (requiresKnowledgeGraph && !graphBuilt) {
      navigate('/knowledge-graph/build', {
        state: { message: '请先构建知识图谱' }
      });
      return;
    }
  }, [currentProject, graphBuilt, requiresProject, requiresKnowledgeGraph]);

  return <>{children}</>;
};
```

## 状态管理设计

### 全局应用状态

```typescript
// stores/appStore.ts
interface AppState {
  // 当前项目
  currentProject: string | null;
  setCurrentProject: (projectId: string | null) => void;

  // 应用配置
  config: AppConfig;
  updateConfig: (config: Partial<AppConfig>) => void;

  // 主题设置
  theme: 'light' | 'dark';
  setTheme: (theme: 'light' | 'dark') => void;

  // 通知系统
  notifications: Notification[];
  addNotification: (notification: Omit<Notification, 'id'>) => void;
  removeNotification: (id: string) => void;
  clearNotifications: () => void;

  // 加载状态管理
  loading: Record<string, boolean>;
  setLoading: (key: string, loading: boolean) => void;

  // 错误状态管理
  errors: Record<string, string | null>;
  setError: (key: string, error: string | null) => void;
  clearErrors: () => void;
}

export const useAppStore = create<AppState>()(
  devtools(
    persist(
      (set, get) => ({
        currentProject: null,
        config: defaultConfig,
        theme: 'light',
        notifications: [],
        loading: {},
        errors: {},

        setCurrentProject: (projectId) => {
          set({ currentProject: projectId }, false, 'setCurrentProject');
          // 触发项目切换时的清理工作
          useKnowledgeGraphStore.getState().resetState();
          useChatStore.getState().clearCurrentConversation();
        },

        updateConfig: (newConfig) =>
          set(
            (state) => ({ config: { ...state.config, ...newConfig } }),
            false,
            'updateConfig'
          ),

        setTheme: (theme) => {
          set({ theme }, false, 'setTheme');
          // 应用主题到DOM
          document.documentElement.setAttribute('data-theme', theme);
        },

        addNotification: (notification) => {
          const id = generateId();
          set(
            (state) => ({
              notifications: [
                ...state.notifications,
                { ...notification, id, timestamp: new Date() }
              ]
            }),
            false,
            'addNotification'
          );

          // 自动移除通知
          if (notification.duration !== 0) {
            setTimeout(() => {
              get().removeNotification(id);
            }, notification.duration || 4500);
          }
        },

        removeNotification: (id) =>
          set(
            (state) => ({
              notifications: state.notifications.filter(n => n.id !== id)
            }),
            false,
            'removeNotification'
          ),

        clearNotifications: () =>
          set({ notifications: [] }, false, 'clearNotifications'),

        setLoading: (key, loading) =>
          set(
            (state) => ({
              loading: { ...state.loading, [key]: loading }
            }),
            false,
            'setLoading'
          ),

        setError: (key, error) =>
          set(
            (state) => ({
              errors: { ...state.errors, [key]: error }
            }),
            false,
            'setError'
          ),

        clearErrors: () =>
          set({ errors: {} }, false, 'clearErrors'),
      }),
      {
        name: 'agraph-app-storage',
        partialize: (state) => ({
          currentProject: state.currentProject,
          config: state.config,
          theme: state.theme,
        }),
      }
    )
  )
);
```

## API服务层设计

### API客户端配置

```typescript
// services/api.ts
import axios, { AxiosRequestConfig, AxiosResponse } from 'axios';
import { useAppStore } from '../stores/appStore';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
const API_TIMEOUT = 30000;

// 创建axios实例
export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: API_TIMEOUT,
  headers: {
    'Content-Type': 'application/json',
  },
});

// 请求拦截器
apiClient.interceptors.request.use(
  (config) => {
    // 添加当前项目参数
    const currentProject = useAppStore.getState().currentProject;
    if (currentProject && config.method !== 'get') {
      config.params = { ...config.params, project_name: currentProject };
    }

    // 添加加载状态
    const loadingKey = getLoadingKey(config);
    if (loadingKey) {
      useAppStore.getState().setLoading(loadingKey, true);
    }

    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// 响应拦截器
apiClient.interceptors.response.use(
  (response: AxiosResponse) => {
    // 移除加载状态
    const loadingKey = getLoadingKey(response.config);
    if (loadingKey) {
      useAppStore.getState().setLoading(loadingKey, false);
    }

    return response;
  },
  (error) => {
    // 移除加载状态
    const loadingKey = getLoadingKey(error.config);
    if (loadingKey) {
      useAppStore.getState().setLoading(loadingKey, false);
    }

    // 统一错误处理
    handleApiError(error);
    return Promise.reject(error);
  }
);

// 生成加载状态key
function getLoadingKey(config: AxiosRequestConfig): string | null {
  if (!config.url) return null;
  return `${config.method}_${config.url}`.replace(/[\/\:]/g, '_');
}

// 统一错误处理
function handleApiError(error: any) {
  const { addNotification } = useAppStore.getState();

  let message = '请求失败';

  if (error.response) {
    // 服务器响应错误
    const { status, data } = error.response;
    message = data?.message || `请求失败 (${status})`;

    if (status === 401) {
      message = '认证失败，请重新登录';
    } else if (status === 403) {
      message = '没有权限执行此操作';
    } else if (status === 404) {
      message = '请求的资源不存在';
    } else if (status >= 500) {
      message = '服务器内部错误';
    }
  } else if (error.request) {
    // 网络错误
    message = '网络连接失败，请检查网络状态';
  }

  addNotification({
    type: 'error',
    title: '请求错误',
    message,
  });
}
```

### 项目服务

```typescript
// services/projectService.ts
import { apiClient } from './api';
import type {
  ProjectListResponse,
  ProjectResponse,
  ProjectCreateRequest,
  ProjectDeleteRequest,
  ProjectSwitchRequest,
} from '../types/api';

export const projectService = {
  // 获取项目列表
  list: async (includeStats = false): Promise<ProjectListResponse> => {
    const response = await apiClient.get('/projects/', {
      params: { include_stats: includeStats },
    });
    return response.data;
  },

  // 创建项目
  create: async (data: ProjectCreateRequest): Promise<ProjectResponse> => {
    const response = await apiClient.post('/projects/create', data);
    return response.data;
  },

  // 获取项目详情
  get: async (projectName: string): Promise<ProjectResponse> => {
    const response = await apiClient.get(`/projects/${projectName}`);
    return response.data;
  },

  // 获取当前项目
  getCurrent: async (): Promise<ProjectResponse> => {
    const response = await apiClient.get('/projects/current');
    return response.data;
  },

  // 切换项目
  switch: async (data: ProjectSwitchRequest): Promise<ProjectResponse> => {
    const response = await apiClient.post('/projects/switch', data);
    return response.data;
  },

  // 删除项目
  delete: async (data: ProjectDeleteRequest): Promise<ProjectResponse> => {
    const response = await apiClient.post('/projects/delete', data);
    return response.data;
  },
};
```

### 知识图谱服务

```typescript
// services/knowledgeGraphService.ts
import { apiClient } from './api';
import type {
  KnowledgeGraphBuildRequest,
  KnowledgeGraphBuildResponse,
  KnowledgeGraphUpdateRequest,
  KnowledgeGraphUpdateResponse,
  KnowledgeGraphStatusResponse,
} from '../types/api';

export const knowledgeGraphService = {
  // 构建知识图谱
  build: async (data: KnowledgeGraphBuildRequest): Promise<KnowledgeGraphBuildResponse> => {
    const response = await apiClient.post('/knowledge-graph/build', data);
    return response.data;
  },

  // 更新知识图谱
  update: async (data: KnowledgeGraphUpdateRequest): Promise<KnowledgeGraphUpdateResponse> => {
    const response = await apiClient.post('/knowledge-graph/update', data);
    return response.data;
  },

  // 获取知识图谱状态
  getStatus: async (): Promise<KnowledgeGraphStatusResponse> => {
    const response = await apiClient.get('/knowledge-graph/status');
    return response.data;
  },
};
```

## 组件设计模式

### 复合组件模式

```typescript
// components/DocumentUploader/index.tsx
interface DocumentUploaderProps {
  onUploadComplete?: (documents: Document[]) => void;
  acceptedTypes?: string[];
  maxFileSize?: number;
  multiple?: boolean;
}

const DocumentUploader: React.FC<DocumentUploaderProps> & {
  DropZone: typeof DropZone;
  FileList: typeof FileList;
  Progress: typeof Progress;
} = ({ onUploadComplete, acceptedTypes, maxFileSize, multiple = true }) => {
  const { uploadDocuments, uploading, progress, files } = useDocumentUpload();

  return (
    <div className="document-uploader">
      <DocumentUploader.DropZone
        acceptedTypes={acceptedTypes}
        maxFileSize={maxFileSize}
        multiple={multiple}
        onFilesSelected={uploadDocuments}
      />
      <DocumentUploader.FileList files={files} />
      <DocumentUploader.Progress
        visible={uploading}
        progress={progress}
      />
    </div>
  );
};

// 添加子组件
DocumentUploader.DropZone = DropZone;
DocumentUploader.FileList = FileList;
DocumentUploader.Progress = Progress;

export default DocumentUploader;
```

### Render Props模式

```typescript
// components/DataProvider/index.tsx
interface DataProviderProps<T> {
  fetcher: () => Promise<T>;
  children: (data: {
    data: T | null;
    loading: boolean;
    error: Error | null;
    refetch: () => void;
  }) => React.ReactNode;
}

function DataProvider<T>({ fetcher, children }: DataProviderProps<T>) {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  const refetch = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const result = await fetcher();
      setData(result);
    } catch (err) {
      setError(err as Error);
    } finally {
      setLoading(false);
    }
  }, [fetcher]);

  useEffect(() => {
    refetch();
  }, [refetch]);

  return (
    <>
      {children({ data, loading, error, refetch })}
    </>
  );
}

// 使用示例
<DataProvider fetcher={() => projectService.list()}>
  {({ data: projects, loading, error, refetch }) => (
    <div>
      {loading && <Spin />}
      {error && <Alert message={error.message} type="error" />}
      {projects && <ProjectList projects={projects} onRefresh={refetch} />}
    </div>
  )}
</DataProvider>
```

## 性能优化策略

### 代码分割

```typescript
// utils/lazyLoad.ts
import { lazy, ComponentType } from 'react';
import LoadingSpinner from '../components/Common/LoadingSpinner';

export function lazyLoad<T extends ComponentType<any>>(
  importFunc: () => Promise<{ default: T }>,
  fallback: React.ComponentType = LoadingSpinner
) {
  const LazyComponent = lazy(importFunc);

  return (props: React.ComponentProps<T>) => (
    <Suspense fallback={<fallback />}>
      <LazyComponent {...props} />
    </Suspense>
  );
}

// 路由中使用
const ProjectList = lazyLoad(() => import('../modules/Projects'));
const KnowledgeGraphVisualizer = lazyLoad(() => import('../modules/KnowledgeGraph/components/GraphVisualizer'));
```

### 虚拟滚动

```typescript
// components/VirtualList/index.tsx
interface VirtualListProps<T> {
  items: T[];
  itemHeight: number;
  containerHeight: number;
  renderItem: (item: T, index: number) => React.ReactNode;
  overscan?: number;
}

function VirtualList<T>({
  items,
  itemHeight,
  containerHeight,
  renderItem,
  overscan = 5,
}: VirtualListProps<T>) {
  const [scrollTop, setScrollTop] = useState(0);

  const startIndex = Math.max(0, Math.floor(scrollTop / itemHeight) - overscan);
  const endIndex = Math.min(
    items.length,
    Math.ceil((scrollTop + containerHeight) / itemHeight) + overscan
  );

  const visibleItems = items.slice(startIndex, endIndex);

  return (
    <div
      style={{ height: containerHeight, overflow: 'auto' }}
      onScroll={(e) => setScrollTop(e.currentTarget.scrollTop)}
    >
      <div style={{ height: items.length * itemHeight, position: 'relative' }}>
        {visibleItems.map((item, index) => (
          <div
            key={startIndex + index}
            style={{
              position: 'absolute',
              top: (startIndex + index) * itemHeight,
              height: itemHeight,
              width: '100%',
            }}
          >
            {renderItem(item, startIndex + index)}
          </div>
        ))}
      </div>
    </div>
  );
}
```

### 记忆化优化

```typescript
// hooks/useMemoizedData.ts
export function useMemoizedData<T, P extends any[]>(
  fetcher: (...args: P) => Promise<T>,
  deps: P,
  options: {
    enabled?: boolean;
    staleTime?: number;
    cacheTime?: number;
  } = {}
) {
  const { enabled = true, staleTime = 5 * 60 * 1000, cacheTime = 10 * 60 * 1000 } = options;

  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const lastFetchTime = useRef<number>(0);

  const cacheKey = useMemo(() => JSON.stringify(deps), deps);

  const fetchData = useCallback(async () => {
    if (!enabled) return;

    const now = Date.now();
    if (now - lastFetchTime.current < staleTime && data !== null) {
      return; // 数据仍然新鲜
    }

    setLoading(true);
    setError(null);

    try {
      const result = await fetcher(...deps);
      setData(result);
      lastFetchTime.current = now;
    } catch (err) {
      setError(err as Error);
    } finally {
      setLoading(false);
    }
  }, [fetcher, enabled, staleTime, cacheKey]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  // 缓存清理
  useEffect(() => {
    const timer = setTimeout(() => {
      if (Date.now() - lastFetchTime.current > cacheTime) {
        setData(null);
      }
    }, cacheTime);

    return () => clearTimeout(timer);
  }, [cacheTime, data]);

  return { data, loading, error, refetch: fetchData };
}
```

## 测试策略

### 单元测试

```typescript
// __tests__/components/ProjectCard.test.tsx
import { render, screen, fireEvent } from '@testing-library/react';
import { ProjectCard } from '../modules/Projects/components/ProjectCard';
import type { Project } from '../types/entities';

const mockProject: Project = {
  id: 'test-project',
  name: 'Test Project',
  description: 'A test project',
  createdAt: '2024-01-01T00:00:00Z',
  documentCount: 5,
  hasKnowledgeGraph: true,
  statistics: {
    sizeInMB: 10,
    documentCount: 5,
    entityCount: 100,
    relationCount: 50,
  },
};

describe('ProjectCard', () => {
  it('renders project information correctly', () => {
    render(<ProjectCard project={mockProject} />);

    expect(screen.getByText('Test Project')).toBeInTheDocument();
    expect(screen.getByText('A test project')).toBeInTheDocument();
    expect(screen.getByText('5')).toBeInTheDocument(); // document count
    expect(screen.getByText('已构建')).toBeInTheDocument(); // knowledge graph status
  });

  it('calls onSelect when clicked', () => {
    const onSelect = jest.fn();
    render(<ProjectCard project={mockProject} onSelect={onSelect} />);

    fireEvent.click(screen.getByRole('button'));
    expect(onSelect).toHaveBeenCalledWith(mockProject);
  });
});
```

### 集成测试

```typescript
// __tests__/integration/ProjectManagement.test.tsx
import { renderWithProviders } from '../utils/test-utils';
import { ProjectManagement } from '../modules/Projects';
import { projectService } from '../services/projectService';

jest.mock('../services/projectService');

describe('Project Management Integration', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('creates a new project successfully', async () => {
    const mockCreate = jest.spyOn(projectService, 'create');
    mockCreate.mockResolvedValue({
      status: 'success',
      data: mockProject,
    });

    render(<ProjectManagement />);

    // 点击创建项目按钮
    fireEvent.click(screen.getByText('创建项目'));

    // 填写表单
    fireEvent.change(screen.getByLabelText('项目名称'), {
      target: { value: 'New Project' }
    });

    // 提交表单
    fireEvent.click(screen.getByText('确定'));

    // 验证API调用
    await waitFor(() => {
      expect(mockCreate).toHaveBeenCalledWith({
        name: 'New Project',
      });
    });

    // 验证成功通知
    expect(screen.getByText('项目创建成功')).toBeInTheDocument();
  });
});
```

## 部署配置

### Docker配置

```dockerfile
# Dockerfile
FROM node:18-alpine as builder

WORKDIR /app

# 复制依赖文件
COPY package*.json ./
COPY yarn.lock ./

# 安装依赖
RUN yarn install --frozen-lockfile

# 复制源代码
COPY . .

# 构建应用
RUN yarn build

# 生产镜像
FROM nginx:alpine

# 复制构建产物
COPY --from=builder /app/dist /usr/share/nginx/html

# 复制nginx配置
COPY nginx.conf /etc/nginx/nginx.conf

# 暴露端口
EXPOSE 80

# 启动nginx
CMD ["nginx", "-g", "daemon off;"]
```

### Nginx配置

```nginx
# nginx.conf
events {
    worker_connections 1024;
}

http {
    include       /etc/nginx/mime.types;
    default_type  application/octet-stream;

    # Gzip压缩
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types
        text/plain
        text/css
        text/xml
        text/javascript
        application/javascript
        application/xml+rss
        application/json;

    server {
        listen 80;
        server_name localhost;

        # 静态文件
        location / {
            root /usr/share/nginx/html;
            index index.html index.htm;
            try_files $uri $uri/ /index.html;

            # 缓存设置
            location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg)$ {
                expires 1y;
                add_header Cache-Control "public, immutable";
            }
        }

        # API代理
        location /api/ {
            proxy_pass http://agraph-api:8000/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;

            # WebSocket支持
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
        }

        # 健康检查
        location /health {
            access_log off;
            return 200 "healthy\n";
            add_header Content-Type text/plain;
        }
    }
}
```

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  agraph-frontend:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "3000:80"
    environment:
      - NODE_ENV=production
    depends_on:
      - agraph-api
    networks:
      - agraph-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost/health"]
      interval: 30s
      timeout: 10s
      retries: 3

networks:
  agraph-network:
    external: true
```

## 安全考虑

### 内容安全策略

```typescript
// public/index.html
<meta http-equiv="Content-Security-Policy" content="
  default-src 'self';
  script-src 'self' 'unsafe-inline';
  style-src 'self' 'unsafe-inline';
  img-src 'self' data: https:;
  connect-src 'self' ws: wss:;
  font-src 'self';
  frame-src 'none';
">
```

### 输入验证

```typescript
// utils/validation.ts
import { z } from 'zod';

export const projectSchema = z.object({
  name: z.string()
    .min(1, '项目名称不能为空')
    .max(50, '项目名称不能超过50个字符')
    .regex(/^[a-zA-Z0-9_-]+$/, '项目名称只能包含字母、数字、下划线和连字符'),
  description: z.string()
    .max(500, '项目描述不能超过500个字符')
    .optional(),
});

export const documentSchema = z.object({
  files: z.array(z.instanceof(File))
    .min(1, '请至少选择一个文件')
    .max(10, '一次最多上传10个文件'),
  tags: z.array(z.string())
    .max(10, '标签数量不能超过10个')
    .optional(),
  metadata: z.record(z.any())
    .optional(),
});

// 使用示例
export function validateProject(data: unknown) {
  return projectSchema.parse(data);
}
```

### XSS防护

```typescript
// utils/sanitize.ts
import DOMPurify from 'dompurify';

export function sanitizeHtml(html: string): string {
  return DOMPurify.sanitize(html, {
    ALLOWED_TAGS: ['b', 'i', 'em', 'strong', 'p', 'br'],
    ALLOWED_ATTR: [],
  });
}

export function escapeText(text: string): string {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}
```

## 总结

本设计方案提供了一个完整、现代化、可扩展的前端架构，具有以下特点：

### 核心优势

1. **模块化架构** - 清晰的功能模块划分，便于开发和维护
2. **类型安全** - 完整的TypeScript类型定义，提高代码质量
3. **性能优化** - 代码分割、虚拟滚动、记忆化等优化策略
4. **用户体验** - 响应式设计、加载状态、错误处理、通知系统
5. **可扩展性** - 插件化架构、主题系统、国际化支持
6. **安全性** - 输入验证、XSS防护、CSP等安全措施

### 技术特色

- 使用最新的React 18和TypeScript
- Zustand轻量级状态管理
- Ant Design企业级UI组件
- ECharts高性能知识图谱可视化
- 完善的测试策略
- 容器化部署方案

这个架构设计能够充分发挥agraph API的功能，为用户提供优秀的知识图谱管理和查询体验。
