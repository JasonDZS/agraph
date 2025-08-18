import React from 'react';
import { Button, Card, Space, Typography, Divider } from 'antd';
import {
  useAppStore,
  useProjectStore,
  useDocumentStore,
  useKnowledgeGraphStore,
  notifications,
} from '../../store';
import { useNotifications } from '../../hooks/useNotifications';

const { Title, Text } = Typography;

/**
 * Example component demonstrating the usage of the state management system.
 * This component shows how to:
 * 1. Access store state
 * 2. Use store actions
 * 3. Work with notifications
 * 4. Use custom hooks
 */
export const StoreExample: React.FC = () => {
  const {
    theme,
    currentProject,
    loading,
    toggleTheme,
    setCurrentProject,
    setLoading,
  } = useAppStore();

  const { getProjectStats, addProject, setProjectsLoading } = useProjectStore();

  const { documents, getUploadStats, setIsUploading } = useDocumentStore();

  const { getGraphStats, setIsBuilding } = useKnowledgeGraphStore();

  const { actions: notificationActions } = useNotifications();

  const projectStats = getProjectStats();
  const uploadStats = getUploadStats();
  const graphStats = getGraphStats();

  const handleAddProject = () => {
    const newProject = {
      name: `Test Project ${Date.now()}`,
      description: 'A test project created from the store example',
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      document_count: 0,
      entity_count: 0,
      relation_count: 0,
    };

    addProject(newProject);
    notificationActions.success(
      'Project Created',
      `Added project: ${newProject.name}`
    );
  };

  const handleSimulateLoading = () => {
    setLoading(true);
    setProjectsLoading(true);
    setIsUploading(true);
    setIsBuilding(true);

    setTimeout(() => {
      setLoading(false);
      setProjectsLoading(false);
      setIsUploading(false);
      setIsBuilding(false);
      notificationActions.info('Loading Complete', 'All operations finished');
    }, 2000);
  };

  const handleNotificationTest = () => {
    notificationActions.success('Success!', 'This is a success notification');

    setTimeout(() => {
      notificationActions.warning('Warning', 'This is a warning notification');
    }, 1000);

    setTimeout(() => {
      notificationActions.error('Error', 'This is an error notification');
    }, 2000);

    setTimeout(() => {
      notificationActions.info('Info', 'This is an info notification');
    }, 3000);
  };

  const handleGlobalNotification = () => {
    notifications.success(
      'Global Notification',
      'This notification was triggered using the global notification system'
    );
  };

  return (
    <div style={{ padding: 24 }}>
      <Title level={3}>State Management System Demo</Title>

      <Space direction="vertical" size="large" style={{ width: '100%' }}>
        {/* App Store Demo */}
        <Card title="App Store">
          <Space>
            <Text>
              Current Theme: <strong>{theme}</strong>
            </Text>
            <Button onClick={toggleTheme}>Toggle Theme</Button>
            <Button
              onClick={() =>
                setCurrentProject(currentProject ? null : 'demo-project')
              }
            >
              {currentProject ? 'Clear Project' : 'Set Demo Project'}
            </Button>
            <Text>
              Loading: <strong>{loading ? 'Yes' : 'No'}</strong>
            </Text>
          </Space>
        </Card>

        {/* Project Store Demo */}
        <Card title="Project Store">
          <Space direction="vertical">
            <Text>
              Total Projects: <strong>{projectStats.totalProjects}</strong>
            </Text>
            <Text>
              Total Documents: <strong>{projectStats.totalDocuments}</strong>
            </Text>
            <Text>
              Total Entities: <strong>{projectStats.totalEntities}</strong>
            </Text>
            <Text>
              Total Relations: <strong>{projectStats.totalRelations}</strong>
            </Text>
            <Button onClick={handleAddProject}>Add Test Project</Button>
          </Space>
        </Card>

        {/* Document Store Demo */}
        <Card title="Document Store">
          <Space direction="vertical">
            <Text>
              Documents Count: <strong>{documents.length}</strong>
            </Text>
            <Text>Upload Stats:</Text>
            <Text>- Total: {uploadStats.total}</Text>
            <Text>- Completed: {uploadStats.completed}</Text>
            <Text>- Failed: {uploadStats.failed}</Text>
            <Text>- In Progress: {uploadStats.inProgress}</Text>
          </Space>
        </Card>

        {/* Knowledge Graph Store Demo */}
        <Card title="Knowledge Graph Store">
          <Space direction="vertical">
            <Text>
              Entities: <strong>{graphStats.totalEntities}</strong>
            </Text>
            <Text>
              Relations: <strong>{graphStats.totalRelations}</strong>
            </Text>
            <Text>
              Text Chunks: <strong>{graphStats.totalTextChunks}</strong>
            </Text>
            <Text>
              Clusters: <strong>{graphStats.totalClusters}</strong>
            </Text>
          </Space>
        </Card>

        <Divider />

        {/* Action Buttons */}
        <Card title="Actions">
          <Space>
            <Button onClick={handleSimulateLoading} loading={loading}>
              Simulate Loading
            </Button>
            <Button onClick={handleNotificationTest}>Test Notifications</Button>
            <Button onClick={handleGlobalNotification}>
              Global Notification
            </Button>
          </Space>
        </Card>
      </Space>
    </div>
  );
};

export default StoreExample;
