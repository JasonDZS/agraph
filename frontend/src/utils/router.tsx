import { Navigate } from 'react-router-dom';
import type { RouteConfig } from '@/types/router';

// Route guard component
export const ProtectedRoute = ({ children }: { children: React.ReactNode }) => {
  // TODO: Add authentication logic here
  const isAuthenticated = true; // Placeholder

  return isAuthenticated ? <>{children}</> : <Navigate to="/login" replace />;
};

// Route configuration helper
export const createRouteElement = (config: RouteConfig) => {
  const Component = config.element;

  if (config.protected) {
    return (
      <ProtectedRoute>
        <Component />
      </ProtectedRoute>
    );
  }

  return <Component />;
};
