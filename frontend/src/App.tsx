import { useEffect } from 'react';
import { RouterProvider } from 'react-router-dom';
import { App as AntdApp } from 'antd';
import ErrorBoundary from '@/components/Common/ErrorBoundary';
import { router } from '@/router/routes';
import { projectSync } from '@/utils/projectSync';
import 'antd/dist/reset.css';

function App() {
  useEffect(() => {
    // Initialize project state on app startup
    projectSync.initializeProjectState().catch(error => {
      console.warn('Failed to initialize project state:', error);
    });
  }, []);

  return (
    <ErrorBoundary>
      <AntdApp>
        <RouterProvider router={router} />
      </AntdApp>
    </ErrorBoundary>
  );
}

export default App;
