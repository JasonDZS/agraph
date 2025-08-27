import React, { useState } from 'react';
import {
  Card,
  Button,
  Space,
  Divider,
  Typography,
  List,
  Input,
  Row,
  Col,
  Tabs,
  Alert
} from 'antd';
import {
  LoadingSpinner,
  ErrorBoundary,
  ConfirmModal,
  NotificationCenter,
  VirtualList,
  confirm,
  notify
} from './index';

const { Title, Paragraph, Text } = Typography;

// Demo data for VirtualList
const generateDemoData = (count: number) => {
  return Array.from({ length: count }, (_, i) => ({
    id: `item-${i}`,
    title: `Item ${i + 1}`,
    content: `This is the content for item ${i + 1}. It contains some demo text to show how the virtual list handles different content.`,
    height: 60 + Math.random() * 40, // Random height between 60-100px
  }));
};

// Component to demonstrate ErrorBoundary
const ProblematicComponent: React.FC<{ shouldError: boolean }> = ({ shouldError }) => {
  if (shouldError) {
    throw new Error('This is a demo error to test ErrorBoundary functionality!');
  }
  return <div>This component is working fine!</div>;
};

const ComponentsDemo: React.FC = () => {
  const [loadingStates, setLoadingStates] = useState({
    spinner1: false,
    spinner2: false,
    wrapper: false,
  });

  const [errorState, setErrorState] = useState(false);
  const [confirmState, setConfirmState] = useState({
    basic: false,
    typed: false,
    danger: false,
  });

  const [virtualListData] = useState(() => generateDemoData(10000));
  const [scrollToIndex, setScrollToIndex] = useState<number>();

  const simulateLoading = (key: keyof typeof loadingStates, duration = 2000) => {
    setLoadingStates(prev => ({ ...prev, [key]: true }));
    setTimeout(() => {
      setLoadingStates(prev => ({ ...prev, [key]: false }));
    }, duration);
  };

  const handleNotificationDemo = (type: 'success' | 'info' | 'warning' | 'error') => {
    const messages = {
      success: { title: 'Success!', message: 'Operation completed successfully' },
      info: { title: 'Information', message: 'Here is some useful information' },
      warning: { title: 'Warning', message: 'Please check your input' },
      error: { title: 'Error', message: 'Something went wrong' },
    };

    notify[type](messages[type].title, messages[type].message);
  };

  const handleAdvancedNotification = () => {
    notify.info('Update Available', 'A new version is ready to install', {
      persistent: true,
      actions: [
        {
          label: 'Install Now',
          onClick: () => {
            notify.success('Installing...', 'Update is being installed');
          },
          type: 'primary'
        },
        {
          label: 'Later',
          onClick: () => {
            notify.info('Reminder Set', 'We\'ll remind you later');
          }
        }
      ]
    });
  };

  const handleConfirmDemo = async (type: 'basic' | 'typed' | 'danger') => {
    let result = false;

    switch (type) {
      case 'basic':
        result = await confirm({
          title: 'Basic Confirmation',
          content: 'Are you sure you want to proceed?',
          type: 'info',
        });
        break;

      case 'typed':
        result = await confirm({
          title: 'Typed Confirmation',
          content: 'This action requires typing confirmation.',
          type: 'warning',
          requiresConfirmation: true,
          confirmationText: 'CONFIRM',
        });
        break;

      case 'danger':
        result = await confirm({
          title: 'Dangerous Action',
          content: 'This will permanently delete all data. This action cannot be undone.',
          type: 'danger',
          danger: true,
          requiresConfirmation: true,
          confirmationText: 'DELETE',
        });
        break;
    }

    if (result) {
      notify.success('Confirmed', `${type} confirmation was successful`);
    } else {
      notify.info('Cancelled', `${type} confirmation was cancelled`);
    }
  };

  const tabItems = [
    {
      key: 'loading',
      label: 'Loading Spinner',
      children: (
        <Space direction="vertical" style={{ width: '100%' }} size="large">
          <Card title="Basic Loading Spinners" size="small">
            <Space wrap>
              <div>
                <Button
                  onClick={() => simulateLoading('spinner1')}
                  loading={loadingStates.spinner1}
                >
                  Load Small
                </Button>
                <LoadingSpinner
                  size="small"
                  spinning={loadingStates.spinner1}
                  style={{ marginLeft: 16 }}
                />
              </div>

              <div>
                <Button
                  onClick={() => simulateLoading('spinner2')}
                  loading={loadingStates.spinner2}
                >
                  Load Large
                </Button>
                <LoadingSpinner
                  size="large"
                  tip="Loading..."
                  spinning={loadingStates.spinner2}
                  style={{ marginLeft: 16 }}
                />
              </div>
            </Space>
          </Card>

          <Card title="Wrapper Mode" size="small">
            <Space direction="vertical" style={{ width: '100%' }}>
              <Button
                onClick={() => simulateLoading('wrapper')}
                type="primary"
              >
                Load Content
              </Button>

              <LoadingSpinner spinning={loadingStates.wrapper} tip="Loading content...">
                <Card>
                  <Paragraph>
                    This content will be covered by the loading spinner when loading.
                    You can wrap any component or content area with LoadingSpinner.
                  </Paragraph>
                  <Paragraph>
                    The spinner automatically centers itself and provides a semi-transparent overlay.
                  </Paragraph>
                </Card>
              </LoadingSpinner>
            </Space>
          </Card>
        </Space>
      ),
    },
    {
      key: 'errors',
      label: 'Error Boundary',
      children: (
        <Space direction="vertical" style={{ width: '100%' }} size="large">
          <Card title="Page-level Error Boundary" size="small">
            <Space direction="vertical" style={{ width: '100%' }}>
              <Text>This demonstrates page-level error handling with multiple recovery options:</Text>
              <Button
                danger
                onClick={() => setErrorState(!errorState)}
              >
                {errorState ? 'Fix Component' : 'Break Component'}
              </Button>

              <ErrorBoundary
                level="page"
                onError={(error) => console.log('Page error caught:', error)}
              >
                <Card>
                  <ProblematicComponent shouldError={errorState} />
                </Card>
              </ErrorBoundary>
            </Space>
          </Card>

          <Card title="Component-level Error Boundary" size="small">
            <Text>This demonstrates component-level error handling with compact display:</Text>
            <ErrorBoundary level="component" showReportButton>
              <Card style={{ marginTop: 16 }}>
                <ProblematicComponent shouldError={errorState} />
              </Card>
            </ErrorBoundary>
          </Card>
        </Space>
      ),
    },
    {
      key: 'notifications',
      label: 'Notifications',
      children: (
        <Space direction="vertical" style={{ width: '100%' }} size="large">
          <Alert
            message="Notification Center"
            description="Click the bell icon in the top-right corner to view all notifications. The badge shows unread count."
            type="info"
            showIcon
          />

          <Card title="Basic Notifications" size="small">
            <Space wrap>
              <Button
                type="primary"
                onClick={() => handleNotificationDemo('success')}
              >
                Success
              </Button>
              <Button
                onClick={() => handleNotificationDemo('info')}
              >
                Info
              </Button>
              <Button
                onClick={() => handleNotificationDemo('warning')}
              >
                Warning
              </Button>
              <Button
                danger
                onClick={() => handleNotificationDemo('error')}
              >
                Error
              </Button>
            </Space>
          </Card>

          <Card title="Advanced Notification" size="small">
            <Space direction="vertical">
              <Text>This creates a persistent notification with custom actions:</Text>
              <Button type="dashed" onClick={handleAdvancedNotification}>
                Show Update Notification
              </Button>
            </Space>
          </Card>
        </Space>
      ),
    },
    {
      key: 'confirmations',
      label: 'Confirmations',
      children: (
        <Space direction="vertical" style={{ width: '100%' }} size="large">
          <Card title="Confirmation Dialogs" size="small">
            <Space wrap>
              <Button onClick={() => handleConfirmDemo('basic')}>
                Basic Confirm
              </Button>
              <Button onClick={() => handleConfirmDemo('typed')}>
                Typed Confirm
              </Button>
              <Button danger onClick={() => handleConfirmDemo('danger')}>
                Danger Confirm
              </Button>
            </Space>
          </Card>

          <Card title="Modal-based Confirmations" size="small">
            <Space wrap>
              <Button onClick={() => setConfirmState(prev => ({ ...prev, basic: true }))}>
                Show Basic Modal
              </Button>
              <Button onClick={() => setConfirmState(prev => ({ ...prev, typed: true }))}>
                Show Typed Modal
              </Button>
              <Button danger onClick={() => setConfirmState(prev => ({ ...prev, danger: true }))}>
                Show Danger Modal
              </Button>
            </Space>

            <ConfirmModal
              open={confirmState.basic}
              title="Basic Confirmation"
              content="This is a basic confirmation modal. Do you want to proceed?"
              onConfirm={() => {
                notify.success('Confirmed', 'Action was confirmed');
                setConfirmState(prev => ({ ...prev, basic: false }));
              }}
              onCancel={() => setConfirmState(prev => ({ ...prev, basic: false }))}
            />

            <ConfirmModal
              open={confirmState.typed}
              title="Typed Confirmation"
              content="Please type CONFIRM to proceed with this action."
              type="warning"
              requiresConfirmation={true}
              confirmationText="CONFIRM"
              onConfirm={() => {
                notify.success('Confirmed', 'Typed confirmation was successful');
                setConfirmState(prev => ({ ...prev, typed: false }));
              }}
              onCancel={() => setConfirmState(prev => ({ ...prev, typed: false }))}
            />

            <ConfirmModal
              open={confirmState.danger}
              title="Dangerous Action"
              content="This action will permanently delete all data and cannot be undone."
              type="danger"
              danger={true}
              requiresConfirmation={true}
              confirmationText="DELETE"
              onConfirm={() => {
                notify.success('Deleted', 'Data has been permanently deleted');
                setConfirmState(prev => ({ ...prev, danger: false }));
              }}
              onCancel={() => setConfirmState(prev => ({ ...prev, danger: false }))}
            />
          </Card>
        </Space>
      ),
    },
    {
      key: 'virtual-list',
      label: 'Virtual List',
      children: (
        <Space direction="vertical" style={{ width: '100%' }} size="large">
          <Card title="Virtual List Demo" size="small">
            <Space direction="vertical" style={{ width: '100%' }}>
              <Text>
                This virtual list contains 10,000 items but only renders visible ones for optimal performance.
              </Text>

              <Space>
                <Text>Scroll to item:</Text>
                <Input
                  type="number"
                  placeholder="Item index (0-9999)"
                  style={{ width: 200 }}
                  onPressEnter={(e) => {
                    const value = parseInt((e.target as HTMLInputElement).value);
                    if (!isNaN(value) && value >= 0 && value < virtualListData.length) {
                      setScrollToIndex(value);
                    }
                  }}
                />
                <Button onClick={() => setScrollToIndex(Math.floor(Math.random() * virtualListData.length))}>
                  Random Item
                </Button>
              </Space>

              <VirtualList
                items={virtualListData}
                itemHeight={80}
                height={400}
                scrollToIndex={scrollToIndex}
                scrollToAlignment="center"
                renderItem={(item, index) => (
                  <Card
                    size="small"
                    style={{
                      margin: '4px 0',
                      backgroundColor: scrollToIndex === index ? '#e6f7ff' : undefined,
                      border: scrollToIndex === index ? '1px solid #1890ff' : '1px solid #d9d9d9'
                    }}
                  >
                    <Card.Meta
                      title={`${item.title} (Index: ${index})`}
                      description={item.content}
                    />
                  </Card>
                )}
                onScroll={(scrollTop, scrollHeight, clientHeight) => {
                  const scrollPercent = Math.round((scrollTop / (scrollHeight - clientHeight)) * 100);
                  // You could update a progress bar here
                }}
                getItemKey={(item) => item.id}
              />
            </Space>
          </Card>
        </Space>
      ),
    },
  ];

  return (
    <div style={{ padding: '24px', maxWidth: '1200px', margin: '0 auto' }}>
      <Row gutter={[16, 16]} align="top">
        <Col xs={24} lg={18}>
          <Space direction="vertical" style={{ width: '100%' }} size="large">
            <div>
              <Title level={2}>Common Components Library Demo</Title>
              <Paragraph>
                This page demonstrates all the common components available in the AGraph frontend.
                Each component is designed to be reusable, performant, and follows modern React patterns.
              </Paragraph>
            </div>

            <Tabs
              items={tabItems}
              defaultActiveKey="loading"
              size="large"
              style={{ width: '100%' }}
            />
          </Space>
        </Col>

        <Col xs={24} lg={6}>
          <Card title="Notification Center" size="small">
            <Space direction="vertical" style={{ width: '100%' }}>
              <Text type="secondary">
                The notification center appears as a bell icon. Try sending some notifications and then click it to see them.
              </Text>
              <NotificationCenter showBadge={true} />
            </Space>
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default ComponentsDemo;
