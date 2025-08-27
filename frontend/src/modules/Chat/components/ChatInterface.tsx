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
  LeftOutlined,
  RightOutlined,
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
  showContext: showContextProp = true,
  defaultSettings = {},
}) => {
  const [inputValue, setInputValue] = useState('');
  const [showHistory, setShowHistory] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [contextCollapsed, setContextCollapsed] = useState(false);
  const contextWidth = 600; // Fixed width
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
    showContext,
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

  // Handle responsive behavior
  useEffect(() => {
    const handleResize = () => {
      const containerWidth = window.innerWidth;
      // On small screens, auto-collapse the context panel
      if (
        containerWidth < 1024 &&
        !contextCollapsed &&
        showContextProp &&
        showContext
      ) {
        setContextCollapsed(true);
      }
    };

    window.addEventListener('resize', handleResize);
    return () => {
      window.removeEventListener('resize', handleResize);
    };
  }, [contextCollapsed, showContextProp, showContext]);

  const toggleContextPanel = () => {
    setContextCollapsed(!contextCollapsed);
  };

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

  const handleShowContextFromMessage = (context: any) => {
    // 设置当前上下文并确保侧边栏显示
    if (!showContext) {
      toggleContext();
    }
    // 展开侧边栏如果它是折叠的
    if (contextCollapsed) {
      setContextCollapsed(false);
    }
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
      type: 'divider',
    },
    {
      key: 'context',
      label: showContext ? '隐藏参考信息' : '显示参考信息',
      icon: <InfoCircleOutlined />,
      onClick: toggleContext,
      disabled: !showContextProp || !currentContext,
    },
    {
      key: 'context-collapse',
      label: contextCollapsed ? '展开侧边栏' : '折叠侧边栏',
      icon: contextCollapsed ? <RightOutlined /> : <LeftOutlined />,
      onClick: toggleContextPanel,
      disabled: !showContextProp || !showContext || !currentContext,
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
    <div
      style={{
        height: '100%', // Use full height of parent container
        maxHeight: '100%', // Prevent expansion beyond parent
        display: 'flex',
        flexDirection: 'column',
        position: 'relative',
        overflow: 'hidden', // Prevent root container from scrolling
        overflowX: 'hidden', // Explicitly prevent horizontal scrolling
        overflowY: 'hidden', // Explicitly prevent vertical scrolling
      }}
    >
      {/* Header */}
      <div
        style={{
          background: '#fff',
          borderBottom: '1px solid #f0f0f0',
          padding: '0 16px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          height: 64,
          minHeight: 64,
          maxHeight: 64,
          flexShrink: 0, // Prevent header from shrinking
          flexGrow: 0, // Prevent header from growing
          zIndex: 100, // Local z-index for stacking context
          overflow: 'hidden', // Prevent any scrolling
          overflowX: 'hidden', // Explicitly prevent horizontal scrolling
          overflowY: 'hidden', // Explicitly prevent vertical scrolling
          position: 'relative', // Ensure proper positioning
          boxSizing: 'border-box', // Include padding in height calculation
          marginRight:
            showContextProp && showContext && currentContext
              ? contextCollapsed
                ? '48px'
                : `${contextWidth}px`
              : '0px',
          transition: 'margin-right 0.3s ease',
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
      </div>

      {/* Content */}
      <div
        style={{
          flex: 1,
          display: 'flex',
          flexDirection: 'column',
          position: 'relative',
          background: '#fff',
          overflow: 'hidden', // Prevent parent container from scrolling
          overflowX: 'hidden', // Explicitly prevent horizontal scrolling
          overflowY: 'hidden', // Explicitly prevent vertical scrolling
          minHeight: 0, // Allow flex shrinking
          maxHeight: '100%', // Prevent expansion beyond parent
          height: '100%', // Take full height of parent
        }}
      >
        {/* Messages Area Container - Fixed size */}
        <div
          style={{
            flex: '1 1 0px', // Take remaining space, allow shrinking, base size 0
            display: 'flex',
            flexDirection: 'column',
            background: '#fafafa',
            marginRight:
              showContextProp && showContext && currentContext
                ? contextCollapsed
                  ? '48px'
                  : `${contextWidth}px`
                : '0px',
            transition: 'margin-right 0.3s ease',
            minHeight: 0, // Allow flex item to shrink
            maxHeight: '100%', // Prevent expansion beyond container
            height: 'auto', // Let flex control the height
            position: 'relative', // Establish positioning context
            overflow: 'hidden', // Container itself doesn't scroll
            overflowX: 'hidden', // Explicitly prevent horizontal scrolling
            overflowY: 'hidden', // Explicitly prevent vertical scrolling
          }}
        >
          {/* Scrollable Messages Content */}
          <div
            style={{
              flex: '1 1 0px', // Take remaining space, allow shrinking, base size 0
              padding: '16px',
              paddingBottom: '16px',
              overflow: 'hidden auto', // Only allow vertical scrolling of content
              overflowX: 'hidden', // Explicitly prevent horizontal scrolling
              scrollBehavior: 'smooth', // Smooth scrolling for better UX
              minHeight: 0, // Allow flex item to shrink
              maxHeight: '100%', // Prevent expansion beyond container
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
                    onShowContext={handleShowContextFromMessage}
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
          </div>
        </div>

        {/* Context Sidebar Container - Fixed positioned relative to entire chat */}
        {showContextProp && showContext && currentContext && (
          <div
            style={{
              position: 'fixed',
              top: 'calc(64px + 64px)', // MainLayout Header + Chat Header
              right: '0px',
              bottom: '0px', // To viewport bottom
              width: contextCollapsed ? '48px' : `${contextWidth}px`,
              minWidth: contextCollapsed ? '48px' : `${contextWidth}px`,
              maxWidth: contextCollapsed ? '48px' : `${contextWidth}px`,
              transition: 'width 0.3s ease',
              backgroundColor: '#fff',
              borderLeft: '1px solid #f0f0f0',
              zIndex: 1000,
              overflow: 'hidden', // Container itself doesn't scroll
              overflowX: 'hidden', // Explicitly prevent horizontal scrolling
              overflowY: 'hidden', // Explicitly prevent vertical scrolling
              display: 'flex',
              flexDirection: 'column',
            }}
          >
            {/* Collapse/Expand Button */}
            <div
              style={{
                position: 'absolute',
                top: '16px', // Position within Context Sidebar
                left: contextCollapsed ? '10px' : contextWidth - 40,
                zIndex: 20,
                transition: 'left 0.3s ease',
              }}
            >
              <Button
                type="text"
                size="small"
                icon={contextCollapsed ? <RightOutlined /> : <LeftOutlined />}
                onClick={toggleContextPanel}
                style={{
                  backgroundColor: 'rgba(255, 255, 255, 0.9)',
                  border: '1px solid #d9d9d9',
                  boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
                  borderRadius: '4px',
                  width: '28px',
                  height: '28px',
                }}
                title={contextCollapsed ? '展开参考信息' : '折叠参考信息'}
              />
            </div>

            {/* Collapsed Preview - Fixed content */}
            {contextCollapsed && currentContext && (
              <div
                style={{
                  padding: '60px 8px 16px 8px', // Normal padding since we're inside Messages Area
                  height: '100%',
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  gap: '12px',
                  overflow: 'hidden', // No scrolling in collapsed preview
                  overflowX: 'hidden',
                  overflowY: 'hidden',
                }}
              >
                <div
                  style={{
                    writing: 'vertical-rl',
                    textOrientation: 'mixed',
                    fontSize: '12px',
                    color: '#666',
                    fontWeight: 'bold',
                  }}
                >
                  参考信息
                </div>

                <div
                  style={{
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '8px',
                    alignItems: 'center',
                  }}
                >
                  <div
                    style={{
                      width: '24px',
                      height: '24px',
                      backgroundColor:
                        currentContext.entities?.length > 0
                          ? '#1890ff'
                          : '#f0f0f0',
                      borderRadius: '4px',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      fontSize: '10px',
                      color: '#fff',
                      fontWeight: 'bold',
                    }}
                    title={`${currentContext.entities?.length || 0} 个实体`}
                  >
                    {currentContext.entities?.length || 0}
                  </div>

                  <div
                    style={{
                      width: '24px',
                      height: '24px',
                      backgroundColor:
                        currentContext.relations?.length > 0
                          ? '#52c41a'
                          : '#f0f0f0',
                      borderRadius: '4px',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      fontSize: '10px',
                      color: '#fff',
                      fontWeight: 'bold',
                    }}
                    title={`${currentContext.relations?.length || 0} 个关系`}
                  >
                    {currentContext.relations?.length || 0}
                  </div>

                  <div
                    style={{
                      width: '24px',
                      height: '24px',
                      backgroundColor:
                        currentContext.text_chunks?.length > 0
                          ? '#722ed1'
                          : '#f0f0f0',
                      borderRadius: '4px',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      fontSize: '10px',
                      color: '#fff',
                      fontWeight: 'bold',
                    }}
                    title={`${currentContext.text_chunks?.length || 0} 个文本片段`}
                  >
                    {currentContext.text_chunks?.length || 0}
                  </div>
                </div>
              </div>
            )}

            {/* Panel Content Container - Fixed size */}
            {!contextCollapsed && (
              <div
                style={{
                  flex: 1, // Take remaining space
                  display: 'flex',
                  flexDirection: 'column',
                  paddingTop: '60px', // Give space for collapse button only
                  overflow: 'hidden', // Container itself doesn't scroll
                  maxHeight: '100%', // Prevent expansion
                }}
              >
                {/* Scrollable Panel Content */}
                <div
                  style={{
                    flex: 1,
                    padding: '16px',
                    paddingTop: '0px', // Already has padding from parent
                    overflow: 'hidden auto', // Only allow vertical scrolling of content
                    overflowX: 'hidden', // Explicitly prevent horizontal scrolling
                    height: '100%', // Take full height of container
                  }}
                >
                  <ContextPanel
                    context={currentContext}
                    visible={showContext}
                    onClose={closeContext}
                  />
                </div>
              </div>
            )}
          </div>
        )}

        {/* Input Area - Fixed at bottom with dynamic width */}
        <div
          style={{
            borderTop: '1px solid #f0f0f0',
            padding: '16px',
            background: '#fff',
            flexShrink: 0, // Prevent input area from shrinking
            flexGrow: 0, // Prevent input area from growing
            marginRight:
              showContextProp && showContext && currentContext
                ? contextCollapsed
                  ? '48px'
                  : `${contextWidth}px`
                : '0px',
            transition: 'margin-right 0.3s ease',
            overflow: 'hidden', // Prevent any scrolling in input area
            overflowX: 'hidden', // Explicitly prevent horizontal scrolling
            overflowY: 'hidden', // Explicitly prevent vertical scrolling
            minHeight: '80px', // Minimum height
            maxHeight: '160px', // Maximum height for input area
            position: 'relative', // Ensure proper positioning
            boxSizing: 'border-box', // Include padding in height calculation
          }}
        >
          <Space.Compact style={{ width: '100%' }}>
            <TextArea
              ref={inputRef}
              value={inputValue}
              onChange={e => setInputValue(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder={
                isStreaming ? '正在生成回答...' : '请输入您的问题...'
              }
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
