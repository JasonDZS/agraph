import React, { useState, useEffect } from 'react';
import { notification, Button, Space, Drawer, Badge, List, Typography, Tag, Empty } from 'antd';
import {
  BellOutlined,
  CloseOutlined,
  DeleteOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  InfoCircleOutlined,
  WarningOutlined,
  CloseCircleOutlined
} from '@ant-design/icons';

const { Text, Title } = Typography;

export type NotificationType = 'success' | 'info' | 'warning' | 'error';

export interface NotificationItem {
  id: string;
  type: NotificationType;
  title: string;
  message?: string;
  timestamp: number;
  read?: boolean;
  persistent?: boolean;
  actions?: Array<{
    label: string;
    onClick: () => void;
    type?: 'primary' | 'default' | 'dashed' | 'link' | 'text';
  }>;
  data?: any;
}

interface NotificationCenterProps {
  maxItems?: number;
  showBadge?: boolean;
  placement?: 'top' | 'right' | 'bottom' | 'left';
  width?: number;
  onNotificationClick?: (notification: NotificationItem) => void;
  onNotificationDismiss?: (notification: NotificationItem) => void;
}

interface NotificationManagerState {
  notifications: NotificationItem[];
  unreadCount: number;
}

class NotificationManager {
  private static instance: NotificationManager;
  private listeners: Array<(state: NotificationManagerState) => void> = [];
  private notifications: NotificationItem[] = [];

  static getInstance(): NotificationManager {
    if (!NotificationManager.instance) {
      NotificationManager.instance = new NotificationManager();
    }
    return NotificationManager.instance;
  }

  subscribe(listener: (state: NotificationManagerState) => void) {
    this.listeners.push(listener);
    return () => {
      this.listeners = this.listeners.filter(l => l !== listener);
    };
  }

  private notify() {
    const unreadCount = this.notifications.filter(n => !n.read).length;
    const state: NotificationManagerState = {
      notifications: [...this.notifications],
      unreadCount,
    };
    this.listeners.forEach(listener => listener(state));
  }

  add(notification: Omit<NotificationItem, 'id' | 'timestamp'>) {
    const item: NotificationItem = {
      ...notification,
      id: `notification_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      timestamp: Date.now(),
      read: false,
    };

    this.notifications.unshift(item);

    // Keep only the latest 100 notifications
    if (this.notifications.length > 100) {
      this.notifications = this.notifications.slice(0, 100);
    }

    this.notify();

    // Show system notification if not persistent
    if (!item.persistent) {
      const config = {
        message: item.title,
        description: item.message,
        duration: item.type === 'error' ? 0 : 4.5,
        placement: 'topRight' as const,
      };

      switch (item.type) {
        case 'success':
          notification.success(config);
          break;
        case 'warning':
          notification.warning(config);
          break;
        case 'error':
          notification.error(config);
          break;
        default:
          notification.info(config);
          break;
      }
    }

    return item.id;
  }

  remove(id: string) {
    this.notifications = this.notifications.filter(n => n.id !== id);
    this.notify();
  }

  markAsRead(id: string) {
    const notification = this.notifications.find(n => n.id === id);
    if (notification) {
      notification.read = true;
      this.notify();
    }
  }

  markAllAsRead() {
    this.notifications.forEach(n => n.read = true);
    this.notify();
  }

  clear() {
    this.notifications = [];
    this.notify();
  }

  getNotifications() {
    return [...this.notifications];
  }

  getUnreadCount() {
    return this.notifications.filter(n => !n.read).length;
  }
}

const notificationManager = NotificationManager.getInstance();

const NotificationCenter: React.FC<NotificationCenterProps> = ({
  maxItems = 50,
  showBadge = true,
  placement = 'right',
  width = 400,
  onNotificationClick,
  onNotificationDismiss,
}) => {
  const [open, setOpen] = useState(false);
  const [state, setState] = useState<NotificationManagerState>({
    notifications: [],
    unreadCount: 0,
  });

  useEffect(() => {
    const unsubscribe = notificationManager.subscribe(setState);
    return unsubscribe;
  }, []);

  const handleNotificationClick = (item: NotificationItem) => {
    if (!item.read) {
      notificationManager.markAsRead(item.id);
    }
    if (onNotificationClick) {
      onNotificationClick(item);
    }
  };

  const handleNotificationDismiss = (item: NotificationItem, event?: React.MouseEvent) => {
    event?.stopPropagation();
    notificationManager.remove(item.id);
    if (onNotificationDismiss) {
      onNotificationDismiss(item);
    }
  };

  const handleClearAll = () => {
    notificationManager.clear();
  };

  const handleMarkAllRead = () => {
    notificationManager.markAllAsRead();
  };

  const getNotificationIcon = (type: NotificationType) => {
    const iconStyle = { fontSize: '16px' };
    switch (type) {
      case 'success':
        return <CheckCircleOutlined style={{ ...iconStyle, color: '#52c41a' }} />;
      case 'warning':
        return <WarningOutlined style={{ ...iconStyle, color: '#faad14' }} />;
      case 'error':
        return <CloseCircleOutlined style={{ ...iconStyle, color: '#ff4d4f' }} />;
      default:
        return <InfoCircleOutlined style={{ ...iconStyle, color: '#1890ff' }} />;
    }
  };

  const getNotificationColor = (type: NotificationType) => {
    switch (type) {
      case 'success':
        return '#52c41a';
      case 'warning':
        return '#faad14';
      case 'error':
        return '#ff4d4f';
      default:
        return '#1890ff';
    }
  };

  const formatTimestamp = (timestamp: number) => {
    const now = Date.now();
    const diff = now - timestamp;
    const minutes = Math.floor(diff / 60000);
    const hours = Math.floor(diff / 3600000);
    const days = Math.floor(diff / 86400000);

    if (minutes < 1) return 'Just now';
    if (minutes < 60) return `${minutes}m ago`;
    if (hours < 24) return `${hours}h ago`;
    if (days < 7) return `${days}d ago`;
    return new Date(timestamp).toLocaleDateString();
  };

  const displayedNotifications = state.notifications.slice(0, maxItems);

  return (
    <>
      <Badge count={showBadge ? state.unreadCount : 0} size="small">
        <Button
          type="text"
          icon={<BellOutlined />}
          onClick={() => setOpen(true)}
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        />
      </Badge>

      <Drawer
        title={
          <Space style={{ width: '100%', justifyContent: 'space-between' }}>
            <Title level={4} style={{ margin: 0 }}>
              Notifications
            </Title>
            <Badge count={state.unreadCount} size="small" />
          </Space>
        }
        placement={placement}
        open={open}
        onClose={() => setOpen(false)}
        width={width}
        extra={
          <Space>
            {state.unreadCount > 0 && (
              <Button size="small" onClick={handleMarkAllRead}>
                Mark all read
              </Button>
            )}
            {state.notifications.length > 0 && (
              <Button size="small" onClick={handleClearAll}>
                Clear all
              </Button>
            )}
          </Space>
        }
      >
        {displayedNotifications.length === 0 ? (
          <Empty
            description="No notifications"
            image={Empty.PRESENTED_IMAGE_SIMPLE}
            style={{ marginTop: '50px' }}
          />
        ) : (
          <List
            dataSource={displayedNotifications}
            renderItem={(item) => (
              <List.Item
                key={item.id}
                style={{
                  padding: '12px 0',
                  cursor: 'pointer',
                  backgroundColor: item.read ? 'transparent' : '#f6f8fa',
                  borderRadius: '6px',
                  marginBottom: '8px',
                  paddingLeft: '12px',
                  paddingRight: '12px',
                  border: item.read ? 'none' : `1px solid ${getNotificationColor(item.type)}20`,
                }}
                onClick={() => handleNotificationClick(item)}
                actions={[
                  <Button
                    key="dismiss"
                    type="text"
                    size="small"
                    icon={<CloseOutlined />}
                    onClick={(e) => handleNotificationDismiss(item, e)}
                    style={{ color: '#666' }}
                  />,
                ]}
              >
                <List.Item.Meta
                  avatar={getNotificationIcon(item.type)}
                  title={
                    <Space style={{ width: '100%', justifyContent: 'space-between' }}>
                      <Text
                        strong={!item.read}
                        style={{ fontSize: '14px' }}
                      >
                        {item.title}
                      </Text>
                      <Tag
                        color={getNotificationColor(item.type)}
                        style={{ fontSize: '10px', margin: 0 }}
                      >
                        {item.type.toUpperCase()}
                      </Tag>
                    </Space>
                  }
                  description={
                    <Space direction="vertical" size="small" style={{ width: '100%' }}>
                      {item.message && (
                        <Text type="secondary" style={{ fontSize: '12px' }}>
                          {item.message}
                        </Text>
                      )}
                      <Space style={{ width: '100%', justifyContent: 'space-between' }}>
                        <Text type="secondary" style={{ fontSize: '11px' }}>
                          {formatTimestamp(item.timestamp)}
                        </Text>
                        {item.actions && (
                          <Space size="small">
                            {item.actions.map((action, index) => (
                              <Button
                                key={index}
                                type={action.type || 'link'}
                                size="small"
                                onClick={(e) => {
                                  e.stopPropagation();
                                  action.onClick();
                                }}
                                style={{ fontSize: '11px', padding: '0 4px', height: '20px' }}
                              >
                                {action.label}
                              </Button>
                            ))}
                          </Space>
                        )}
                      </Space>
                    </Space>
                  }
                />
              </List.Item>
            )}
          />
        )}
      </Drawer>
    </>
  );
};

export default NotificationCenter;
export { notificationManager };

// Helper functions for easy usage
export const notify = {
  success: (title: string, message?: string, options?: Partial<NotificationItem>) =>
    notificationManager.add({ type: 'success', title, message, ...options }),

  info: (title: string, message?: string, options?: Partial<NotificationItem>) =>
    notificationManager.add({ type: 'info', title, message, ...options }),

  warning: (title: string, message?: string, options?: Partial<NotificationItem>) =>
    notificationManager.add({ type: 'warning', title, message, ...options }),

  error: (title: string, message?: string, options?: Partial<NotificationItem>) =>
    notificationManager.add({ type: 'error', title, message, ...options }),
};
