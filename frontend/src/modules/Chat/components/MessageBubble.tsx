import React, { useState } from 'react';
import {
  Avatar,
  Button,
  Card,
  Space,
  Tooltip,
  Typography,
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
import MarkdownRenderer from '@/components/Common/MarkdownRenderer';

const { Text, Paragraph } = Typography;

const MessageBubble: React.FC<MessageBubbleProps> = ({
  message,
  index,
  isStreaming = false,
  context,
  onRegenerate,
  onCopy,
  onShowContext,
}) => {
  const [copied, setCopied] = useState(false);
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
    if (onShowContext && context) {
      onShowContext(context);
    }
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
            <MarkdownRenderer
              content={message.content + (isStreaming && isAssistant ? ' ▌' : '')}
              style={{
                margin: 0,
                color: isUser ? 'white' : 'inherit',
                wordBreak: 'break-word',
                fontSize: '14px',
                lineHeight: '1.6',
              }}
              className="message-content"
            />

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


      <style jsx global>{`
        .message-content {
          font-family: inherit;
        }

        /* 用户消息中的所有文本为白色 */
        .message-content p,
        .message-content h1,
        .message-content h2,
        .message-content h3,
        .message-content h4,
        .message-content h5,
        .message-content h6,
        .message-content li,
        .message-content span {
          color: inherit !important;
        }

        /* 用户消息中的代码块样式调整 */
        .message-content pre {
          background-color: ${isUser ? 'rgba(255, 255, 255, 0.15)' : '#f6f8fa'} !important;
          border: ${isUser ? '1px solid rgba(255, 255, 255, 0.2)' : '1px solid #d0d7de'} !important;
        }

        .message-content code {
          background-color: ${isUser ? 'rgba(255, 255, 255, 0.15)' : '#f6f8fa'} !important;
          color: ${isUser ? 'rgba(255, 255, 255, 0.9)' : 'inherit'} !important;
        }

        /* 用户消息中的链接样式 */
        .message-content a {
          color: ${isUser ? 'rgba(255, 255, 255, 0.9)' : '#0969da'} !important;
        }

        /* 用户消息中的引用块样式 */
        .message-content blockquote {
          border-left-color: ${isUser ? 'rgba(255, 255, 255, 0.3)' : '#d0d7de'} !important;
          color: ${isUser ? 'rgba(255, 255, 255, 0.8)' : '#656d76'} !important;
        }

        /* 用户消息中的表格样式 */
        .message-content table,
        .message-content th,
        .message-content td {
          border-color: ${isUser ? 'rgba(255, 255, 255, 0.3)' : '#d0d7de'} !important;
        }

        .message-content th {
          background-color: ${isUser ? 'rgba(255, 255, 255, 0.1)' : '#f6f8fa'} !important;
        }

        /* 分割线样式 */
        .message-content hr {
          border-top-color: ${isUser ? 'rgba(255, 255, 255, 0.3)' : '#d0d7de'} !important;
        }

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
