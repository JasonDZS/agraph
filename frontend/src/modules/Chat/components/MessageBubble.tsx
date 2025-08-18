import React, { useState } from 'react';
import {
  Avatar,
  Button,
  Card,
  Space,
  Tooltip,
  Typography,
  Modal,
  Badge,
} from 'antd';
import {
  UserOutlined,
  RobotOutlined,
  CopyOutlined,
  ReloadOutlined,
  InfoCircleOutlined,
  CheckOutlined,
} from '@ant-design/icons';
import { MessageBubbleProps } from '../types';
import ContextPanel from './ContextPanel';

const { Text, Paragraph } = Typography;

const MessageBubble: React.FC<MessageBubbleProps> = ({
  message,
  index,
  isStreaming = false,
  context,
  onRegenerate,
  onCopy,
}) => {
  const [copied, setCopied] = useState(false);
  const [showContext, setShowContext] = useState(false);
  const isUser = message.role === 'user';
  const isAssistant = message.role === 'assistant';

  const handleCopy = async () => {
    if (onCopy) {
      onCopy(message.content);
    } else {
      try {
        await navigator.clipboard.writeText(message.content);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
      } catch (error) {
        console.error('Failed to copy text:', error);
      }
    }
  };

  const handleRegenerate = () => {
    if (onRegenerate) {
      onRegenerate();
    }
  };

  const handleShowContext = () => {
    setShowContext(true);
  };

  const getMessageTime = () => {
    try {
      const date = new Date(message.timestamp);
      return date.toLocaleTimeString('zh-CN', {
        hour: '2-digit',
        minute: '2-digit',
      });
    } catch {
      return '';
    }
  };

  const hasContext =
    context &&
    ((context.entities && context.entities.length > 0) ||
      (context.relations && context.relations.length > 0) ||
      (context.text_chunks && context.text_chunks.length > 0));

  return (
    <div
      style={{
        display: 'flex',
        justifyContent: isUser ? 'flex-end' : 'flex-start',
        marginBottom: 16,
        alignItems: 'flex-start',
      }}
    >
      {!isUser && (
        <Avatar
          icon={<RobotOutlined />}
          style={{
            backgroundColor: '#1890ff',
            marginRight: 12,
            marginTop: 4,
            flexShrink: 0,
          }}
        />
      )}

      <div
        style={{
          maxWidth: '70%',
          minWidth: '200px',
        }}
      >
        <Card
          size="small"
          style={{
            backgroundColor: isUser ? '#1890ff' : '#f0f0f0',
            border: 'none',
            borderRadius: 12,
            overflow: 'hidden',
          }}
          bodyStyle={{
            padding: '12px 16px',
          }}
        >
          <div>
            <Paragraph
              style={{
                margin: 0,
                color: isUser ? 'white' : 'inherit',
                wordBreak: 'break-word',
                whiteSpace: 'pre-wrap',
              }}
              copyable={false}
            >
              {message.content}
              {isStreaming && isAssistant && (
                <span
                  style={{
                    display: 'inline-block',
                    width: '8px',
                    height: '16px',
                    backgroundColor: 'currentColor',
                    marginLeft: '4px',
                    animation: 'blink 1s infinite',
                  }}
                />
              )}
            </Paragraph>

            {/* Message actions */}
            <div
              style={{
                marginTop: message.content ? 8 : 0,
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
              }}
            >
              <Text
                style={{
                  fontSize: '12px',
                  color: isUser
                    ? 'rgba(255, 255, 255, 0.7)'
                    : 'rgba(0, 0, 0, 0.45)',
                }}
              >
                {getMessageTime()}
              </Text>

              <Space size="small">
                {hasContext && isAssistant && (
                  <Tooltip title="查看上下文信息">
                    <Button
                      type="text"
                      size="small"
                      icon={
                        <Badge
                          count={
                            (context?.entities?.length || 0) +
                            (context?.relations?.length || 0) +
                            (context?.text_chunks?.length || 0)
                          }
                          size="small"
                        >
                          <InfoCircleOutlined />
                        </Badge>
                      }
                      style={{
                        color: isUser
                          ? 'rgba(255, 255, 255, 0.7)'
                          : 'rgba(0, 0, 0, 0.45)',
                        border: 'none',
                        padding: '0 4px',
                      }}
                      onClick={handleShowContext}
                    />
                  </Tooltip>
                )}

                <Tooltip title={copied ? '已复制' : '复制消息'}>
                  <Button
                    type="text"
                    size="small"
                    icon={copied ? <CheckOutlined /> : <CopyOutlined />}
                    style={{
                      color: isUser
                        ? 'rgba(255, 255, 255, 0.7)'
                        : 'rgba(0, 0, 0, 0.45)',
                      border: 'none',
                      padding: '0 4px',
                    }}
                    onClick={handleCopy}
                  />
                </Tooltip>

                {isAssistant && onRegenerate && !isStreaming && (
                  <Tooltip title="重新生成">
                    <Button
                      type="text"
                      size="small"
                      icon={<ReloadOutlined />}
                      style={{
                        color: 'rgba(0, 0, 0, 0.45)',
                        border: 'none',
                        padding: '0 4px',
                      }}
                      onClick={handleRegenerate}
                    />
                  </Tooltip>
                )}
              </Space>
            </div>
          </div>
        </Card>
      </div>

      {isUser && (
        <Avatar
          icon={<UserOutlined />}
          style={{
            backgroundColor: '#52c41a',
            marginLeft: 12,
            marginTop: 4,
            flexShrink: 0,
          }}
        />
      )}

      {/* Context Modal */}
      <Modal
        title="消息上下文信息"
        open={showContext}
        onCancel={() => setShowContext(false)}
        footer={null}
        width={800}
        style={{ top: 20 }}
      >
        <ContextPanel
          context={context}
          visible={showContext}
          onClose={() => setShowContext(false)}
        />
      </Modal>

      <style jsx>{`
        @keyframes blink {
          0%,
          50% {
            opacity: 1;
          }
          51%,
          100% {
            opacity: 0;
          }
        }
      `}</style>
    </div>
  );
};

export default MessageBubble;
