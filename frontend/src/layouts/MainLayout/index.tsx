import { useState, useEffect } from 'react';
import { Layout, theme, ConfigProvider } from 'antd';
import { Outlet } from 'react-router-dom';
import Header from '../Header';
import Sidebar from '../Sidebar';
import { useProject } from '@/modules/Projects/hooks/useProject';

const { Content } = Layout;

const MainLayout = () => {
  const [collapsed, setCollapsed] = useState(false);
  const [isDarkMode, setIsDarkMode] = useState(false);
  const { token } = theme.useToken();
  const { loadProjects } = useProject();

  // Initialize projects on layout mount
  useEffect(() => {
    loadProjects();
  }, [loadProjects]);

  const toggleCollapsed = () => {
    setCollapsed(!collapsed);
  };

  const toggleTheme = () => {
    setIsDarkMode(!isDarkMode);
  };

  return (
    <ConfigProvider
      theme={{
        algorithm: isDarkMode ? theme.darkAlgorithm : theme.defaultAlgorithm,
      }}
    >
      <Layout style={{ minHeight: '100vh' }}>
        <Sidebar collapsed={collapsed} />

        <Layout style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
          <Header
            collapsed={collapsed}
            onToggleCollapsed={toggleCollapsed}
            isDarkMode={isDarkMode}
            onToggleTheme={toggleTheme}
          />

          <Content
            style={{
              margin: '16px',
              padding: '24px',
              background: token.colorBgContainer,
              borderRadius: token.borderRadius,
              minHeight: 'calc(100vh - 112px)', // 64px header + 32px margins + 16px padding
              overflow: 'auto',
              flex: 1, // 确保内容区域占据可用空间
              width: '100%', // 确保宽度为100%
            }}
          >
            <Outlet />
          </Content>
        </Layout>
      </Layout>
    </ConfigProvider>
  );
};

export default MainLayout;
