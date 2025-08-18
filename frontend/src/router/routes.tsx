import { createBrowserRouter } from 'react-router-dom';
import { Suspense } from 'react';
import { Spin } from 'antd';
import MainLayout from '@/layouts/MainLayout';
import { createRouteElement } from '@/utils/router';
import * as Pages from '@/pages';
import type { RouteConfig } from '@/types/router';

const LoadingSpinner = () => (
  <div
    style={{
      display: 'flex',
      justifyContent: 'center',
      alignItems: 'center',
      height: '200px',
    }}
  >
    <Spin size="large" />
  </div>
);

const routeConfigs: RouteConfig[] = [
  {
    path: '/',
    element: Pages.Dashboard,
    title: 'Dashboard',
    icon: 'DashboardOutlined',
  },
  {
    path: '/projects',
    element: Pages.Projects,
    title: 'Projects',
    icon: 'FolderOutlined',
    protected: true,
  },
  {
    path: '/documents',
    element: Pages.Documents,
    title: 'Documents',
    icon: 'FileTextOutlined',
    protected: true,
  },
  {
    path: '/knowledge-graph',
    element: Pages.KnowledgeGraph,
    title: 'Knowledge Graph',
    icon: 'NodeIndexOutlined',
    protected: true,
  },
  {
    path: '/chat',
    element: Pages.Chat,
    title: 'Chat',
    icon: 'MessageOutlined',
    protected: true,
  },
  {
    path: '/search',
    element: Pages.Search,
    title: 'Search',
    icon: 'SearchOutlined',
    protected: true,
  },
];

export const router = createBrowserRouter([
  {
    path: '/',
    element: <MainLayout />,
    children: routeConfigs.map(config => ({
      path: config.path === '/' ? '' : config.path,
      element: (
        <Suspense fallback={<LoadingSpinner />}>
          {createRouteElement(config)}
        </Suspense>
      ),
    })),
  },
]);

export { routeConfigs };
