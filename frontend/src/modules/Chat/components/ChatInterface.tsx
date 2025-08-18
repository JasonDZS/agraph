import React, { useState, useRef, useEffect } from 'react';
import {
  Layout,
  Input,
  Button,
  Space,
  Drawer,
  List,
  Card,
  Typography,
  Tooltip,
  Dropdown,
  MenuProps,
  message,
  Modal,
  Form,
  Slider,
  Select,
  Switch,
  Empty,
  Spin,
} from 'antd';
import {
  SendOutlined,
  PlusOutlined,
  HistoryOutlined,
  SettingOutlined,
  DeleteOutlined,
  InfoCircleOutlined,
  ClearOutlined,
  MessageOutlined,
} from '@ant-design/icons';
import { ChatInterfaceProps, SendMessageOptions } from '../types';
import { useStreamChat } from '../hooks/useStreamChat';
import MessageBubble from './MessageBubble';
import ContextPanel from './ContextPanel';

const { Header, Content, Sider } = Layout;
const { TextArea } = Input;
const { Title, Text } = Typography;
const { confirm } = Modal;

const ChatInterface: React.FC<ChatInterfaceProps> = ({
  height = 600,
  showContext = true,
  defaultSettings = {},
}) => {
  const [inputValue, setInputValue] = useState('');
  const [showHistory, setShowHistory] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [settings, setSettings] = useState<SendMessageOptions>({
    entity_top_k: 5,
    relation_top_k: 5,
    text_chunk_top_k: 3,
    response_type: 'conversational',
    ...defaultSettings,
  });

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<any>(null);

  const {
    conversations,
    currentConversation,
    messages,
    isStreaming,
    error,
    currentContext,
    showContext: contextVisible,
    hasMessages,
    hasConversations,
    canRegenerate,
    sendStreamMessage,
    startNewConversation,
    switchToConversation,
    removeConversation,
    clearAllConversations,
    regenerateResponse,
    toggleContext,
    closeContext,
  } = useStreamChat({
    onError: errorMsg => {
      message.error(errorMsg);
    },
  });

  // Auto scroll to bottom when new messages arrive
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages]);

  // Focus input when not streaming
  useEffect(() => {
    if (!isStreaming && inputRef.current) {
      inputRef.current.focus();
    }
  }, [isStreaming]);

  const handleSendMessage = async () => {
    const content = inputValue.trim();
    if (!content || isStreaming) return;

    setInputValue('');
    await sendStreamMessage(content, settings);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const handleNewConversation = async () => {
    await startNewConversation();
    setShowHistory(false);
  };

  const handleDeleteConversation = (conversationId: string, title: string) => {
    confirm({
      title: '删除对话',
      content: `确定要删除对话 "${title}" 吗？此操作不可撤销。`,
      okText: '删除',
      okType: 'danger',
      cancelText: '取消',
      onOk: () => removeConversation(conversationId),
    });
  };

  const handleClearHistory = () => {
    confirm({
      title: '清空所有对话',
      content: '确定要清空所有对话历史吗？此操作不可撤销。',
      okText: '清空',
      okType: 'danger',
      cancelText: '取消',
      onOk: clearAllConversations,
    });
  };

  const handleRegenerateResponse = () => {
    regenerateResponse();
  };

  const conversationMenuItems: MenuProps['items'] = [
    {
      key: 'settings',
      label: '聊天设置',
      icon: <SettingOutlined />,
      onClick: () => setShowSettings(true),
    },
    {
      key: 'history',
      label: '对话历史',
      icon: <HistoryOutlined />,
      onClick: () => setShowHistory(true),
    },
    {
      key: 'context',
      label: contextVisible ? '隐藏上下文' : '显示上下文',
      icon: <InfoCircleOutlined />,
      onClick: toggleContext,
      disabled: !currentContext,
    },
    {
      type: 'divider',
    },
    {
      key: 'clear',
      label: '清空历史',
      icon: <ClearOutlined />,
      onClick: handleClearHistory,
      disabled: !hasConversations,
    },
  ];

  return (
    <div style={{ height, display: 'flex', flexDirection: 'column' }}>
      {/* Header */}
      <Header
        style={{
          background: '#fff',
          borderBottom: '1px solid #f0f0f0',
          padding: '0 16px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          height: 64,
        }}
      >
        <div>
          <Title level={4} style={{ margin: 0 }}>
            <MessageOutlined style={{ marginRight: 8 }} />
            智能对话
          </Title>
          {currentConversation && (
            <Text type="secondary" style={{ fontSize: 12 }}>
              {currentConversation.title}
            </Text>
          )}
        </div>

        <Space>
          <Button
            type="primary"
            icon={<PlusOutlined />}
            onClick={handleNewConversation}
            size="small"
          >
            新对话
          </Button>

          <Dropdown
            menu={{ items: conversationMenuItems }}
            placement="bottomRight"
            trigger={['click']}
          >
            <Button icon={<SettingOutlined />} size="small" />
          </Dropdown>
        </Space>
      </Header>

      {/* Content */}
      <Layout style={{ flex: 1, background: '#fff' }}>
        {/* Messages */}
        <Content
          style={{
            padding: '16px',
            overflow: 'auto',
            background: '#fafafa',
          }}
        >
          {!hasMessages ? (
            <div
              style={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                height: '100%',
                textAlign: 'center',
              }}
            >
              <MessageOutlined
                style={{
                  fontSize: 64,
                  color: '#d9d9d9',
                  marginBottom: 16,
                }}
              />
              <Title level={3} type="secondary">
                开始新的对话
              </Title>
              <Text type="secondary">
                向我提问任何关于知识图谱的问题，我会根据已有的知识为您提供准确的答案。
              </Text>
            </div>
          ) : (
            <>
              {messages.map((message, index) => (
                <MessageBubble
                  key={`${message.role}-${index}-${message.timestamp}`}
                  message={message}
                  index={index}
                  isStreaming={
                    isStreaming &&
                    index === messages.length - 1 &&
                    message.role === 'assistant'
                  }
                  context={
                    index === messages.length - 1 &&
                    message.role === 'assistant'
                      ? currentContext
                      : undefined
                  }
                  onRegenerate={
                    canRegenerate && index === messages.length - 1
                      ? handleRegenerateResponse
                      : undefined
                  }
                />
              ))}
              <div ref={messagesEndRef} />
            </>
          )}

          {error && (
            <Card
              size="small"
              style={{
                marginTop: 16,
                borderColor: '#ff4d4f',
                backgroundColor: '#fff2f0',
              }}
            >
              <Text type="danger">{error}</Text>
            </Card>
          )}
        </Content>

        {/* Context Sidebar */}
        {showContext && (
          <Sider
            width={400}
            style={{
              background: '#fff',
              borderLeft: '1px solid #f0f0f0',
            }}
          >
            <div style={{ padding: 16, height: '100%', overflow: 'auto' }}>
              <ContextPanel
                context={currentContext}
                visible={contextVisible}
                onClose={closeContext}
              />
            </div>
          </Sider>
        )}
      </Layout>

      {/* Input Area */}
      <div
        style={{
          borderTop: '1px solid #f0f0f0',
          padding: '16px',
          background: '#fff',
        }}
      >
        <Space.Compact style={{ width: '100%' }}>
          <TextArea
            ref={inputRef}
            value={inputValue}
            onChange={e => setInputValue(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder={isStreaming ? '正在生成回答...' : '请输入您的问题...'}
            disabled={isStreaming}
            autoSize={{ minRows: 1, maxRows: 4 }}
            style={{ resize: 'none' }}
          />
          <Button
            type="primary"
            icon={<SendOutlined />}
            onClick={handleSendMessage}
            disabled={!inputValue.trim() || isStreaming}
            loading={isStreaming}
            style={{ height: 'auto' }}
          >
            发送
          </Button>
        </Space.Compact>
      </div>

      {/* History Drawer */}
      <Drawer
        title="对话历史"
        placement="left"
        onClose={() => setShowHistory(false)}
        open={showHistory}
        width={350}
      >
        {hasConversations ? (
          <List
            dataSource={conversations}
            renderItem={conversation => (
              <List.Item
                actions={[
                  <Tooltip title="删除对话">
                    <Button
                      type="text"
                      danger
                      size="small"
                      icon={<DeleteOutlined />}
                      onClick={() =>
                        handleDeleteConversation(
                          conversation.id,
                          conversation.title
                        )
                      }
                    />
                  </Tooltip>,
                ]}
              >
                <List.Item.Meta
                  title={
                    <Button
                      type="link"
                      style={{
                        padding: 0,
                        height: 'auto',
                        fontWeight:
                          conversation.id === currentConversation?.id
                            ? 'bold'
                            : 'normal',
                      }}
                      onClick={() => {
                        switchToConversation(conversation.id);
                        setShowHistory(false);
                      }}
                    >
                      {conversation.title}
                    </Button>
                  }
                  description={
                    <Text type="secondary" style={{ fontSize: 12 }}>
                      {new Date(conversation.updated_at).toLocaleString(
                        'zh-CN'
                      )}
                    </Text>
                  }
                />
              </List.Item>
            )}
          />
        ) : (
          <Empty
            description="暂无对话历史"
            image={Empty.PRESENTED_IMAGE_SIMPLE}
          />
        )}
      </Drawer>

      {/* Settings Modal */}
      <Modal
        title="聊天设置"
        open={showSettings}
        onCancel={() => setShowSettings(false)}
        footer={null}
        width={500}
      >
        <Form
          layout="vertical"
          initialValues={settings}
          onFinish={values => {
            setSettings(values);
            setShowSettings(false);
            message.success('设置已保存');
          }}
        >
          <Form.Item
            label="实体检索数量"
            name="entity_top_k"
            tooltip="从知识图谱中检索的实体数量"
          >
            <Slider min={1} max={20} marks={{ 1: '1', 10: '10', 20: '20' }} />
          </Form.Item>

          <Form.Item
            label="关系检索数量"
            name="relation_top_k"
            tooltip="从知识图谱中检索的关系数量"
          >
            <Slider min={1} max={20} marks={{ 1: '1', 10: '10', 20: '20' }} />
          </Form.Item>

          <Form.Item
            label="文本片段数量"
            name="text_chunk_top_k"
            tooltip="从文档中检索的相关文本片段数量"
          >
            <Slider min={1} max={10} marks={{ 1: '1', 5: '5', 10: '10' }} />
          </Form.Item>

          <Form.Item
            label="回答类型"
            name="response_type"
            tooltip="选择AI回答的风格"
          >
            <Select>
              <Select.Option value="conversational">对话式</Select.Option>
              <Select.Option value="analytical">分析式</Select.Option>
              <Select.Option value="detailed">详细式</Select.Option>
              <Select.Option value="concise">简洁式</Select.Option>
            </Select>
          </Form.Item>

          <Form.Item>
            <Space>
              <Button type="primary" htmlType="submit">
                保存设置
              </Button>
              <Button onClick={() => setShowSettings(false)}>取消</Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>
    </div>
  );
};

export default ChatInterface;
