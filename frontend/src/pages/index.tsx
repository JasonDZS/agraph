import { lazy } from 'react';

// Lazy load page components for code splitting
export const Dashboard = lazy(() => import('./Dashboard'));
export const Projects = lazy(
  () => import('../modules/Projects/components/ProjectList')
);
export const Documents = lazy(
  () => import('../modules/Documents/components/DocumentManagement')
);
export const KnowledgeGraph = lazy(
  () => import('../modules/KnowledgeGraph/components/KnowledgeGraphMain')
);
export const Chat = lazy(
  () => import('../modules/Chat/components/ChatInterface')
);
export const Search = lazy(
  () => import('../modules/Search/components/SearchInterface')
);
