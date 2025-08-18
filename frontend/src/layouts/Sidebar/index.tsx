import { Layout, Menu, theme } from 'antd';
import { useNavigate, useLocation } from 'react-router-dom';
import {
  DashboardOutlined,
  FolderOutlined,
  FileTextOutlined,
  NodeIndexOutlined,
  MessageOutlined,
  SearchOutlined,
} from '@ant-design/icons';
import type { MenuProps } from 'antd';

const { Sider } = Layout;

interface SidebarProps {
  collapsed: boolean;
}

const Sidebar = ({ collapsed }: SidebarProps) => {
  const { token } = theme.useToken();
  const navigate = useNavigate();
  const location = useLocation();

  const menuItems: MenuProps['items'] = [
    {
      key: '/',
      icon: <DashboardOutlined />,
      label: 'Dashboard',
    },
    {
      key: '/projects',
      icon: <FolderOutlined />,
      label: 'Projects',
    },
    {
      key: '/documents',
      icon: <FileTextOutlined />,
      label: 'Documents',
    },
    {
      key: '/knowledge-graph',
      icon: <NodeIndexOutlined />,
      label: 'Knowledge Graph',
    },
    {
      key: '/chat',
      icon: <MessageOutlined />,
      label: 'Chat',
    },
    {
      key: '/search',
      icon: <SearchOutlined />,
      label: 'Search',
    },
  ];

  const handleMenuClick: MenuProps['onClick'] = ({ key }) => {
    navigate(key);
  };

  return (
    <Sider
      trigger={null}
      collapsible
      collapsed={collapsed}
      width={240}
      style={{
        background: token.colorBgContainer,
        borderRight: `1px solid ${token.colorBorder}`,
      }}
    >
      <div
        style={{
          height: '64px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          borderBottom: `1px solid ${token.colorBorder}`,
          fontSize: collapsed ? '16px' : '20px',
          fontWeight: 'bold',
          color: token.colorPrimary,
        }}
      >
        {collapsed ? 'AG' : 'AGraph'}
      </div>

      <Menu
        mode="inline"
        selectedKeys={[location.pathname]}
        items={menuItems}
        onClick={handleMenuClick}
        style={{
          height: 'calc(100vh - 64px)',
          borderRight: 0,
          background: 'transparent',
        }}
      />
    </Sider>
  );
};

export default Sidebar;
