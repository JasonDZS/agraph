import { Layout, Button, Dropdown, Avatar, Space, theme } from 'antd';
import {
  MenuOutlined,
  UserOutlined,
  SettingOutlined,
  LogoutOutlined,
  SunOutlined,
  MoonOutlined,
} from '@ant-design/icons';
import type { MenuProps } from 'antd';

const { Header: AntHeader } = Layout;

interface HeaderProps {
  collapsed: boolean;
  onToggleCollapsed: () => void;
  isDarkMode: boolean;
  onToggleTheme: () => void;
}

const Header = ({
  onToggleCollapsed,
  isDarkMode,
  onToggleTheme,
}: HeaderProps) => {
  const { token } = theme.useToken();

  const userMenuItems: MenuProps['items'] = [
    {
      key: 'profile',
      label: 'Profile',
      icon: <UserOutlined />,
    },
    {
      key: 'settings',
      label: 'Settings',
      icon: <SettingOutlined />,
    },
    {
      type: 'divider',
    },
    {
      key: 'logout',
      label: 'Logout',
      icon: <LogoutOutlined />,
      danger: true,
    },
  ];

  return (
    <AntHeader
      style={{
        padding: '0 16px',
        background: token.colorBgContainer,
        borderBottom: `1px solid ${token.colorBorder}`,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
      }}
    >
      <Space>
        <Button
          type="text"
          icon={<MenuOutlined />}
          onClick={onToggleCollapsed}
          style={{ fontSize: '16px', width: 40, height: 40 }}
        />
        <div
          style={{
            fontSize: '18px',
            fontWeight: 'bold',
            color: token.colorTextHeading,
          }}
        >
          AGraph
        </div>
      </Space>

      <Space>
        <Button
          type="text"
          icon={isDarkMode ? <SunOutlined /> : <MoonOutlined />}
          onClick={onToggleTheme}
          style={{ fontSize: '16px' }}
        />

        <Dropdown menu={{ items: userMenuItems }} placement="bottomRight">
          <Button type="text" style={{ height: 40 }}>
            <Space>
              <Avatar size="small" icon={<UserOutlined />} />
              <span>User</span>
            </Space>
          </Button>
        </Dropdown>
      </Space>
    </AntHeader>
  );
};

export default Header;
