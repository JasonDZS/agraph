import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeHighlight from 'rehype-highlight';
import rehypeRaw from 'rehype-raw';
import 'highlight.js/styles/github.css';

interface MarkdownRendererProps {
  content: string;
  className?: string;
  style?: React.CSSProperties;
}

const MarkdownRenderer: React.FC<MarkdownRendererProps> = ({
  content,
  className,
  style,
}) => {
  return (
    <div className={className} style={style}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeHighlight, rehypeRaw]}
        components={{
          // 自定义代码块样式
          code: ({ node, inline, className, children, ...props }) => {
            const match = /language-(\w+)/.exec(className || '');
            return !inline && match ? (
              <pre style={{
                backgroundColor: '#f6f8fa',
                padding: '12px',
                borderRadius: '6px',
                overflow: 'auto',
                fontSize: '14px',
                lineHeight: '1.45'
              }}>
                <code className={className} {...props}>
                  {children}
                </code>
              </pre>
            ) : (
              <code
                className={className}
                style={{
                  backgroundColor: '#f6f8fa',
                  padding: '2px 4px',
                  borderRadius: '3px',
                  fontSize: '85%'
                }}
                {...props}
              >
                {children}
              </code>
            );
          },
          // 自定义表格样式
          table: ({ children, ...props }) => (
            <div style={{ overflow: 'auto', marginBottom: '16px' }}>
              <table
                style={{
                  width: '100%',
                  borderCollapse: 'collapse',
                  border: '1px solid #d0d7de'
                }}
                {...props}
              >
                {children}
              </table>
            </div>
          ),
          th: ({ children, ...props }) => (
            <th
              style={{
                padding: '6px 13px',
                border: '1px solid #d0d7de',
                backgroundColor: '#f6f8fa',
                fontWeight: 'bold'
              }}
              {...props}
            >
              {children}
            </th>
          ),
          td: ({ children, ...props }) => (
            <td
              style={{
                padding: '6px 13px',
                border: '1px solid #d0d7de'
              }}
              {...props}
            >
              {children}
            </td>
          ),
          // 自定义链接样式
          a: ({ children, ...props }) => (
            <a
              style={{
                color: '#0969da',
                textDecoration: 'none'
              }}
              target="_blank"
              rel="noopener noreferrer"
              {...props}
            >
              {children}
            </a>
          ),
          // 自定义引用块样式
          blockquote: ({ children, ...props }) => (
            <blockquote
              style={{
                borderLeft: '4px solid #d0d7de',
                paddingLeft: '16px',
                marginLeft: '0',
                color: '#656d76'
              }}
              {...props}
            >
              {children}
            </blockquote>
          ),
          // 自定义分割线样式
          hr: (props) => (
            <hr
              style={{
                border: 'none',
                borderTop: '1px solid #d0d7de',
                margin: '24px 0'
              }}
              {...props}
            />
          ),
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
};

export default MarkdownRenderer;
